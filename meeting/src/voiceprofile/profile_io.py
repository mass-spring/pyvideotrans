"""プロフィールフォルダへの書き出し (pyVideoTrans の videotrans/util/voice_profile.py が読む形)。

<out>/<名前>/
    profile.json          {"name", "refs": [{"wav","text","lang","sec","snr_db", ...追加項目}], "embedding", "stats", ...}
    refs/ref_ja_01.wav    手本 (録音と同じサンプリング周波数のまま。ノイズ除去・音量調整はしない)
    eval/eval_ja_01.wav   評価用に取り置いた発話 (手本に使っていない)
    eval/eval.json        評価用クリップの一覧
    emb.npy               声のセンタロイド (話者埋め込みの平均)
    enroll_report.xlsx    報告書 (report.py)

profile.json の追加項目 (score, text_status, src など) は pyVideoTrans 側では無視される。
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__
from .clips import assemble
from .pipeline import BuildError, BuildResult, _nan_none

PROFILE_JSON = "profile.json"
SPEAKER_MAP_JSON = "speaker_map.json"
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_folder_name(name: str) -> str:
    """フォルダ名に使えない文字を _ にする。名前そのもの (日本語可) は残す。"""
    s = _BAD_CHARS.sub("_", name.strip()).rstrip(". ")
    if not s or s in (".", ".."):
        raise BuildError(f"名前 '{name}' はフォルダ名にできません。別の名前にしてください")
    return s


def profile_dir_for(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / safe_folder_name(name)


def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def check_can_write(pdir: Path, append: bool, force: bool) -> None:
    """既存のプロフィールがあるとき、うっかり上書きしないための確認 (計算の前に呼ぶ)。"""
    pj = pdir / PROFILE_JSON
    if pj.exists() and not (append or force):
        raise BuildError(
            f"プロフィールがすでにあります: {pdir}\n"
            "  - 別の言語を足す / 同じ言語の手本だけ作り直す:  --append\n"
            "  - 全部作り直す (refs と eval を消して書き直す):   --force"
        )
    if append and pj.exists() and not isinstance(read_json(pj), dict):
        raise BuildError(f"{pj} を JSON として読めません。壊れている場合は --force で作り直してください")


def _round(v, nd=1):
    v = _nan_none(v)
    return None if v is None else round(v, nd)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"JSON にできない型: {type(o)}")


def _fmt(obj, level: int = 0, width: int = 100) -> str:
    """見やすい JSON: 短い配列・辞書は 1 行、長いものは字下げして複数行。"""
    obj = json.loads(json.dumps(obj, ensure_ascii=False, default=_json_default))
    one = json.dumps(obj, ensure_ascii=False)
    pad = "  " * level
    if isinstance(obj, dict) and obj:
        if len(one) + len(pad) <= width and all(not isinstance(v, (dict, list)) or len(json.dumps(v, ensure_ascii=False)) < 60 for v in obj.values()) and level > 1:
            return one
        items = [f'{pad}  {json.dumps(k, ensure_ascii=False)}: {_fmt(v, level + 1, width)}' for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list) and obj:
        if len(one) + len(pad) <= width:
            return one
        items = [f"{pad}  {_fmt(v, level + 1, width)}" for v in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return one


def dump_json(path: Path, obj) -> None:
    """UTF-8 (BOM なし) で書く。pyVideoTrans の読み込みは BOM があっても読める。"""
    path.write_text(_fmt(obj) + "\n", encoding="utf-8")


def write_profile(res: BuildResult) -> dict:
    """ファイルを書き出し、書いたものの情報 (パスなど) を返す。"""
    opts = res.options
    pdir = profile_dir_for(opts.out_dir, opts.name)
    old = read_json(pdir / PROFILE_JSON) if (pdir / PROFILE_JSON).exists() else None
    appending = bool(opts.append and isinstance(old, dict) and not opts.force)
    new_langs = sorted({c.lang for c in res.refs} | {e.lang for e in res.evals})
    pdir.mkdir(parents=True, exist_ok=True)
    refs_dir, eval_dir = pdir / "refs", pdir / "eval"

    # ---- 古いファイルの片付け -------------------------------------------------------------------
    if appending:
        for lang in new_langs:
            for d, prefix in ((refs_dir, "ref"), (eval_dir, "eval")):
                for p in d.glob(f"{prefix}_{lang}_*.wav") if d.is_dir() else []:
                    p.unlink()
    else:
        for d in (refs_dir, eval_dir):
            if d.is_dir():
                shutil.rmtree(d)
        for p in pdir.glob("emb*.npy"):
            p.unlink()
    refs_dir.mkdir(exist_ok=True)

    # ---- 手本 ---------------------------------------------------------------------------------------
    ref_entries = []
    for c in res.refs:
        f = res.files[c.file]
        wave = assemble(f.y, f.sr, c.pieces)
        sf.write(str(refs_dir / c.ref_name), wave, f.sr, subtype=f.out_subtype)
        ref_entries.append({
            "wav": f"refs/{c.ref_name}",
            "text": c.text,
            "lang": c.lang,
            "sec": round(len(wave) / f.sr, 2),
            "snr_db": _round(c.snr_db, 1),
            # ---- 以下は追加項目 (pyVideoTrans は使わない) ----
            "score": _round(c.score, 1),
            "text_status": c.status,
            "sentence_ids": c.slot_ids,
            "loudness_dbfs": _round(c.speech_dbfs, 1),
            "peak_dbfs": _round(c.peak_dbfs, 1),
            "clip_ratio": round(float(c.clip_ratio), 6),
            "consistency": {k: _round(v, 3) for k, v in {**c.consistency, **c.consistency_true}.items()},
            "sample_rate": f.sr,
            "src": {"file": f.path.name, "pieces": [[round(a, 3), round(b, 3)] for a, b in c.pieces]},
        })

    # ---- 評価用 -------------------------------------------------------------------------------------
    eval_entries = []
    if res.evals:
        eval_dir.mkdir(exist_ok=True)
    for ev in res.evals:
        f = res.files[ev.file]
        wave = assemble(f.y, f.sr, ev.pieces)
        sf.write(str(eval_dir / ev.name), wave, f.sr, subtype=f.out_subtype)
        eval_entries.append({
            "wav": f"eval/{ev.name}", "text": ev.text, "lang": ev.lang, "sec": round(len(wave) / f.sr, 2),
            "snr_db": _round(ev.snr_db, 1), "text_status": ev.status, "sentence_ids": ev.slot_ids,
            "consistency": {k: _round(v, 3) for k, v in ev.consistency.items()},
            "src": {"file": f.path.name, "pieces": [[round(a, 3), round(b, 3)] for a, b in ev.pieces]},
        })
    if appending:
        old_eval = read_json(eval_dir / "eval.json")
        kept = [e for e in (old_eval or {}).get("clips", []) if e.get("lang") not in new_langs] if isinstance(old_eval, dict) else []
        eval_entries = kept + eval_entries
    if eval_entries:
        eval_dir.mkdir(exist_ok=True)
        dump_json(eval_dir / "eval.json", {"name": opts.name, "clips": eval_entries})

    # ---- 声のセンタロイド (話者埋め込みの平均) ---------------------------------------------------------
    emb_files: dict[str, dict[str, str]] = {}
    for lang, per in res.centroids.items():
        for name, vec in per.items():
            fn = f"emb_{lang}_{name}.npy"
            np.save(pdir / fn, vec.astype(np.float32))
            emb_files.setdefault(lang, {})[name] = fn
    primary_lang = res.refs[0].lang
    embedding = None
    if res.embedder_names and primary_lang in emb_files:
        first = next((n for n in res.embedder_names if n in emb_files[primary_lang]), None)
        if first:
            shutil.copyfile(pdir / emb_files[primary_lang][first], pdir / "emb.npy")
            embedding = {"model": first, "file": "emb.npy", "dim": int(res.centroids[primary_lang][first].shape[0]),
                         "lang": primary_lang}

    # ---- profile.json -----------------------------------------------------------------------------------
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    stats_by_lang = {lang: s for lang, s in res.stats.items()}
    consent_units = [u for u in res.units if u.role == "consent"]
    enrollment = {
        "tool": "voiceprofile", "version": __version__, "date": now,
        "inputs": [{"file": f.path.name, "sample_rate": f.sr, "sec": round(f.dur, 2)} for f in res.files],
        "script": opts.script_path or None, "langs": [x for x in opts.langs],
        "consent_segments": [{"file": res.files[u.file].path.name, "start": round(u.start, 2), "end": round(u.end, 2)}
                             for u in consent_units],
        "embedders": res.embedder_names, "vad": res.files[0].seg.vad_used, "asr": res.asr_name or None,
        "f0_method": res.f0_method or None,
        "params": {"n_refs": opts.n_refs, "min_sec": opts.min_sec, "max_sec": opts.max_sec, "target_sec": opts.target_sec,
                   "min_pause": opts.min_pause},
        "warnings": res.warnings,
    }
    profile: dict = {"name": opts.name}
    history: list = []
    if appending:
        profile = dict(old)
        old_refs = [r for r in profile.get("refs", []) if isinstance(r, dict) and r.get("lang") not in new_langs]
        ref_entries = old_refs + ref_entries
        sb = dict(profile.get("stats_by_lang") or {})
        sb.update(stats_by_lang)
        stats_by_lang = sb
        if isinstance(profile.get("enrollment"), dict):
            history = list(profile.get("history") or []) + [profile["enrollment"]]
        old_emb = profile.get("embedding") if isinstance(profile.get("embedding"), dict) else None
        if old_emb and old_emb.get("lang") not in new_langs:
            embedding = old_emb
        old_stats = profile.get("stats") if isinstance(profile.get("stats"), dict) else None
        keep_old_stats = old_stats is not None and (profile.get("stats_lang") not in new_langs)
    else:
        keep_old_stats = False
    profile["name"] = opts.name
    profile["refs"] = ref_entries
    if embedding:
        profile["embedding"] = embedding
    else:
        profile.pop("embedding", None)
    profile["embedding_files"] = {**(profile.get("embedding_files") or {} if appending else {}), **emb_files}
    if stats_by_lang:
        if not keep_old_stats:
            profile["stats"] = stats_by_lang.get(primary_lang) or next(iter(stats_by_lang.values()))
            profile["stats_lang"] = primary_lang if primary_lang in stats_by_lang else next(iter(stats_by_lang))
        profile["stats_by_lang"] = stats_by_lang
    if eval_entries:
        profile["eval"] = {"index": "eval/eval.json", "n": len(eval_entries)}
    else:
        profile.pop("eval", None)
    profile["enrollment"] = enrollment
    if history:
        profile["history"] = history
    profile["updated"] = now
    profile.setdefault("created", now)
    dump_json(pdir / PROFILE_JSON, profile)
    return {"dir": pdir, "profile_json": pdir / PROFILE_JSON, "refs": [pdir / e["wav"] for e in ref_entries],
            "eval": [pdir / e["wav"] for e in eval_entries], "appended": appending}


# ---- speaker_map.json -----------------------------------------------------------------------------------
def parse_set_args(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        label, sep, name = it.partition("=")
        label, name = label.strip(), name.strip()
        if not sep or not label or not name:
            raise ValueError(f"--set の書き方が違います: '{it}'  （例: --set spk0=Aさん）")
        out[label] = name
    return out


def write_speaker_map(profiles_dir: Path, mapping: dict[str, str], *, replace: bool = False,
                      allow_missing: bool = False) -> tuple[Path, dict[str, str], list[str]]:
    """profiles_dir/speaker_map.json を書く。戻り値: (パス, 書いた内容, 警告のリスト)。"""
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.is_dir():
        if not allow_missing:
            raise FileNotFoundError(f"プロフィールのフォルダがありません: {profiles_dir}（先に voiceprofile build を実行してください）")
        profiles_dir.mkdir(parents=True, exist_ok=True)
    problems = []
    for label, name in mapping.items():
        pj = profiles_dir / name / PROFILE_JSON
        if not pj.is_file():
            problems.append(f"{label} -> {name}: プロフィール {profiles_dir / name} がありません")
    if problems and not allow_missing:
        have = sorted(p.parent.name for p in profiles_dir.glob(f"*/{PROFILE_JSON}"))
        raise ValueError("\n".join(problems) + f"\n  あるプロフィール: {', '.join(have) if have else '(なし)'}"
                         "\n  名前が合っているか確認してください（それでも書くなら --allow-missing）")
    path = profiles_dir / SPEAKER_MAP_JSON
    current: dict[str, str] = {}
    if path.is_file() and not replace:
        data = read_json(path)
        if isinstance(data, dict):
            current = {str(k): str(v) for k, v in data.items() if v}
    current.update(mapping)
    dump_json(path, current)
    return path, current, problems
