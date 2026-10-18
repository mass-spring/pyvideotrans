"""生成フェーズ：手本を用意し、Job をエンジンごとにまとめて実行し、manifest.csv に記録する。

- エンジンは 1 つずつ読み込み、終わったら GPU メモリを手放す（T4 16GB で複数エンジンを順に動かすため）。
- 1 件の失敗では止まらない（失敗の理由を manifest.csv に書いて次へ）。
- やり直し（再実行）に強い：すでに音声ができている Job は飛ばす（--force で作り直し）。
- 早回し（C2）は、元の Job の音声ができた後に CPU で作る。
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio import read_wav, resample, time_stretch, trim_to, write_wav
from .conditions import Job, RefInfo, build_jobs, safe_name
from .config import ExperimentConfig, engine_options_for, resolve_path
from .engines import create_engine, parse_engine_spec
from .engines.base import EngineError, free_gpu
from .speakers import Clip, SpeakerPlan, load_speaker, plan_speaker
from .texts import TargetText, read_targets

MANIFEST = "manifest.csv"
MANIFEST_COLUMNS = [
    "job_id", "speaker", "engine", "engine_label", "condition", "lang", "text_id", "kind", "repeat", "seed",
    "ref_kind", "ref_file", "ref_source", "ref_sec", "ref_sr", "ref_text_used", "emotion_used",
    "text", "out_wav", "status", "mode", "note", "error", "gen_seconds", "audio_seconds", "sr",
]
STATUS_OK, STATUS_FAILED, STATUS_SKIPPED, STATUS_PENDING = "ok", "failed", "skipped", "pending"


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 手本の用意
# ---------------------------------------------------------------------------
@dataclass
class PreparedSpeaker:
    plan: SpeakerPlan
    line_refs: list[RefInfo]
    profile_refs: list[RefInfo]


def prepare_line_ref(clip: Clip, cfg: ExperimentConfig, out_dir: Path, speaker: str, write: bool = True) -> RefInfo:
    """C0 の手本：評価用の実声を 16 kHz・モノラルにし、長ければ max_s 秒に切る（切ったら書き起こしは使えない）。"""
    stem = Path(clip.path).stem
    dst = out_dir / "refs" / safe_name(speaker) / f"line_ref_16k_{safe_name(stem)}.wav"
    text, note, sec = clip.text, "", clip.duration
    if clip.duration > cfg.line_ref_max_s + 0.5:
        text = ""
        sec = cfg.line_ref_max_s
        note = f"{clip.duration:.1f}秒の実声を先頭 {cfg.line_ref_max_s:g} 秒に切ったので書き起こしは使いません"
    elif clip.duration < cfg.line_ref_min_s - 0.5:
        note = f"手本が短い（{clip.duration:.1f}秒）"
    if write:
        y, sr = read_wav(clip.path)
        if clip.duration > cfg.line_ref_max_s + 0.5:
            y = trim_to(y, sr, cfg.line_ref_max_s)
        y16 = resample(y, sr, 16000)
        write_wav(dst, y16, 16000)
        sec = len(y16) / 16000
    return RefInfo("line", str(dst), clip.path, text, clip.lang, float(sec), 16000, stem, note)


def prepare_profile_ref(clip: Clip, out_dir: Path, speaker: str, write: bool = True) -> RefInfo:
    """C1 の手本：元のサンプルレートのまま、モノラルの wav にそろえる。"""
    stem = Path(clip.path).stem
    dst = out_dir / "refs" / safe_name(speaker) / f"profile_{safe_name(stem)}.wav"
    note = "" if clip.text else "書き起こしがありません"
    if write:
        y, sr = read_wav(clip.path)
        write_wav(dst, y, sr)
    return RefInfo("profile", str(dst), clip.path, clip.text, clip.lang, float(clip.duration), int(clip.sr), stem, note)


def prepare_speakers(cfg: ExperimentConfig, out_dir: Path, write: bool = True) -> list[PreparedSpeaker]:
    out: list[PreparedSpeaker] = []
    for spec in cfg.speakers:
        data = load_speaker(spec, cfg)
        plan = plan_speaker(data, cfg)
        for n in plan.notes:
            say(f"  注意（{plan.name}）：{n}")
        lines = [prepare_line_ref(c, cfg, out_dir, plan.name, write) for c in plan.line_refs]
        profs = [prepare_profile_ref(c, out_dir, plan.name, write) for c in plan.profile_refs]
        out.append(PreparedSpeaker(plan, lines, profs))
    return out


def load_texts(cfg: ExperimentConfig) -> list[TargetText]:
    return read_targets(resolve_path(cfg, cfg.targets), cfg.target_langs, cfg.max_texts_per_lang)


def build_all_jobs(cfg: ExperimentConfig, prepared: list[PreparedSpeaker], texts: list[TargetText], dry_run: bool) -> list[Job]:
    jobs: list[Job] = []
    for ps in prepared:
        jobs += build_jobs(cfg, ps.plan.name, ps.line_refs, ps.profile_refs, texts, dry_run)
    return jobs


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def job_row(j: Job) -> dict:
    r = j.ref
    return {
        "job_id": j.job_id, "speaker": j.speaker, "engine": j.engine, "engine_label": j.engine_label,
        "condition": j.condition, "lang": j.lang, "text_id": j.text_id, "kind": j.kind, "repeat": j.repeat,
        "seed": j.seed, "ref_kind": r.kind if r else "", "ref_file": r.path if r else "",
        "ref_source": r.source if r else "", "ref_sec": f"{r.sec:.2f}" if r else "", "ref_sr": r.sr if r else "",
        "ref_text_used": "yes" if j.ref_text_arg else "no", "emotion_used": j.emotion_arg or "",
        "text": j.text, "out_wav": j.out_rel, "status": STATUS_PENDING, "mode": "", "note": "", "error": "",
        "gen_seconds": "", "audio_seconds": "", "sr": "",
    }


def read_manifest(out_dir: Path) -> dict[str, dict]:
    p = out_dir / MANIFEST
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        return {r["job_id"]: r for r in csv.DictReader(f)}


def write_manifest(out_dir: Path, rows: dict[str, dict]) -> Path:
    p = out_dir / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    tmp.replace(p)
    return p


def _merge_info(out_dir: Path, key: str, value: dict) -> None:
    p = out_dir / "engines_info.json"
    data = {}
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[key] = value
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def summarize_plan(jobs: list[Job]) -> dict:
    gen = [j for j in jobs if not j.is_derived and not j.skip_reason]
    der = [j for j in jobs if j.is_derived and not j.skip_reason]
    by_engine: dict[str, int] = {}
    for j in gen:
        by_engine[j.engine] = by_engine.get(j.engine, 0) + 1
    by_cond: dict[str, int] = {}
    for j in jobs:
        if not j.skip_reason:
            by_cond[j.condition] = by_cond.get(j.condition, 0) + 1
    return {
        "total": len(jobs), "generate": len(gen), "derived": len(der),
        "skipped": sum(1 for j in jobs if j.skip_reason), "by_engine": by_engine, "by_condition": by_cond,
    }


def run_generate(
    cfg: ExperimentConfig,
    out_dir: str | Path,
    *,
    dry_run: bool = False,
    only_engines: list[str] | None = None,
    force: bool = False,
    log: Callable[[str], None] = say,
) -> dict[str, dict]:
    """手本を用意して、Job を実行する。manifest（job_id → 行）を返す。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log(f"実験 '{cfg.name}'：出力先 {out.resolve()}" + ("  [ドライラン：ダミー音声]" if dry_run else ""))
    prepared = prepare_speakers(cfg, out)
    texts = load_texts(cfg)
    jobs = build_all_jobs(cfg, prepared, texts, dry_run)
    plan = summarize_plan(jobs)
    log(f"Job：合計 {plan['total']}（生成 {plan['generate']} / 早回しの後処理 {plan['derived']} / 対応外で飛ばす {plan['skipped']}）")

    wanted = {parse_engine_spec(e) for e in only_engines} if only_engines else None

    def selected(spec: str) -> bool:
        if wanted is None:
            return True
        n, v = parse_engine_spec(spec)
        return (n, v) in wanted or (n, "") in wanted

    old_manifest = read_manifest(out)
    manifest: dict[str, dict] = {}
    for j in jobs:
        old = old_manifest.get(j.job_id)
        runs_here = (not j.skip_reason) and selected(j.engine)
        if old is not None and not force:
            if old.get("status") == STATUS_OK and (out / old["out_wav"]).is_file():
                manifest[j.job_id] = old  # できているものは保つ
                continue
            if not runs_here:
                manifest[j.job_id] = old  # 今回は触らないエンジン：前回の記録（失敗の理由など）を残す
                continue
        row = job_row(j)
        if j.skip_reason:
            row.update(status=STATUS_SKIPPED, note=j.skip_reason)
        manifest[j.job_id] = row
    write_manifest(out, manifest)

    by_spec: dict[str, list[Job]] = {}
    for j in jobs:
        if not j.skip_reason and not j.is_derived and selected(j.engine):
            by_spec.setdefault(j.engine, []).append(j)

    for spec, js in by_spec.items():
        todo = [j for j in js if manifest[j.job_id]["status"] != STATUS_OK or force]
        if not todo:
            log(f"[{spec}] すべて作成済みです（飛ばします）")
        else:
            _run_engine(cfg, spec, todo, out, manifest, dry_run, log)
        _run_derived(cfg, [j for j in jobs if j.is_derived and not j.skip_reason and j.engine == spec], out, manifest, log, force)
        write_manifest(out, manifest)
    ok = sum(1 for r in manifest.values() if r["status"] == STATUS_OK)
    ng = sum(1 for r in manifest.values() if r["status"] == STATUS_FAILED)
    log(f"生成の結果：成功 {ok} / 失敗 {ng} / 飛ばした {sum(1 for r in manifest.values() if r['status'] == STATUS_SKIPPED)}  → {out / MANIFEST}")
    return manifest


def _run_engine(cfg, spec, todo, out, manifest, dry_run, log) -> None:
    log(f"[{spec}] {len(todo)} 件を生成します…（モデルを読み込み中）")
    engine = create_engine(spec, engine_options_for(cfg, spec), dry_run)
    t0 = time.time()
    try:
        engine.load()
    except EngineError as e:
        for j in todo:
            manifest[j.job_id].update(status=STATUS_FAILED, error=f"エンジンを読み込めません: {e}")
        log(f"[{spec}] 読み込みに失敗：{e}")
        return
    except Exception as e:  # noqa: BLE001 - 想定外でも他のエンジンは続ける
        for j in todo:
            manifest[j.job_id].update(status=STATUS_FAILED, error=f"エンジンの読み込み中のエラー: {type(e).__name__}: {str(e)[:300]}")
        log(f"[{spec}] 読み込み中のエラー：{type(e).__name__}: {str(e)[:200]}")
        return
    log(f"[{spec}] 読み込み完了（{time.time() - t0:.0f} 秒）")
    _merge_info(out, spec, engine.info())
    done = 0
    try:
        # 同じ手本をまとめて処理すると、手本の前処理のキャッシュが効くエンジンで速い
        for j in sorted(todo, key=lambda x: (x.ref.path if x.ref else "", x.condition, x.text_id, x.repeat)):
            row = manifest[j.job_id]
            t1 = time.time()
            try:
                res = engine.synthesize(
                    j.text, j.lang, j.ref.path, j.ref_text_arg, j.emotion_arg,
                    ref_lang=(j.ref.lang or None), seed=j.seed,
                )
                if res.wav.size == 0:
                    raise EngineError("音声が空でした")
                dst = out / j.out_rel
                write_wav(dst, res.wav, res.sr)
                row.update(
                    status=STATUS_OK, mode=res.mode, note=res.note, error="", sr=res.sr,
                    gen_seconds=f"{time.time() - t1:.2f}", audio_seconds=f"{len(res.wav) / res.sr:.2f}",
                )
            except Exception as e:  # noqa: BLE001 - 1 件の失敗で止めない
                msg = str(e) if isinstance(e, EngineError) else f"{type(e).__name__}: {str(e)[:300]}"
                row.update(status=STATUS_FAILED, error=msg, gen_seconds=f"{time.time() - t1:.2f}")
            done += 1
            if done % 10 == 0 or done == len(todo):
                log(f"[{spec}] {done}/{len(todo)} 件（失敗 {sum(1 for x in todo if manifest[x.job_id]['status'] == STATUS_FAILED)}）")
                write_manifest(out, manifest)
    finally:
        engine.close()
        free_gpu()
        write_manifest(out, manifest)


def _run_derived(cfg, derived: list[Job], out: Path, manifest, log, force: bool) -> None:
    """早回し（C2）：親の音声ができていれば作る。"""
    if not derived:
        return
    n_ok = n_ng = 0
    methods: set[str] = set()
    for j in derived:
        row = manifest[j.job_id]
        if row["status"] == STATUS_OK and (out / row["out_wav"]).is_file() and not force:
            continue
        parent = manifest.get(j.parent)
        if not parent or parent["status"] != STATUS_OK or not (out / parent["out_wav"]).is_file():
            row.update(status=STATUS_FAILED, error="元の音声（profile_ref）ができていないので作れません")
            n_ng += 1
            continue
        try:
            y, sr = read_wav(out / parent["out_wav"])
            y2, method = time_stretch(y, sr, j.factor, cfg.stretch_method)
            write_wav(out / j.out_rel, y2, sr)
            methods.add(method)
            row.update(status=STATUS_OK, mode=f"stretch_{method}", sr=sr, error="",
                       note=f"{j.factor:g}倍速（{method}）", audio_seconds=f"{len(y2) / sr:.2f}", gen_seconds="0")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            row.update(status=STATUS_FAILED, error=f"{type(e).__name__}: {str(e)[:300]}")
            n_ng += 1
    if n_ok or n_ng:
        log(f"早回しの後処理：成功 {n_ok} / 失敗 {n_ng}（方式: {', '.join(sorted(methods)) or '-'}）")
