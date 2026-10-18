"""録音 → 発話区間 → 台本との対応 → 手本の候補 → 品質と点数 → 選ぶ → 評価用の取り置き、までの流れ。

ファイルへの書き出しは profile_io.py、Excel は report.py。ここは計算だけ (音声ファイルは読むが、何も書かない)。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from voicecheck.audio import TARGET_SR, load_audio, resample
from voicecheck.embedders import Embedder, load_embedders
from voicecheck.features import extract_features

from . import __version__
from .align import (
    STATUS_ASR_ONLY,
    STATUS_NONE,
    STATUS_UNASSIGNED,
    AlignResult,
    align,
)
from .clips import (
    Atom,
    Candidate,
    EvalClip,
    Unit,
    assemble,
    clip_pieces,
    eligible_runs,
    enumerate_candidates,
    pieces_duration,
    score_candidate,
    select_refs,
)
from .quality import FrameInfo, build_frame_info, clipping_ratio, measure_quality, noise_drift_db, peak_dbfs
from .script import Script, Slot, ScriptError
from .segment import NoSpeechError, Segment, Segmentation, detect_segments
from .text_utils import join_texts, norm_lang

MIN_OUTPUT_SR = 44100
MIN_TOTAL_SPEECH_SEC = 60.0
MIN_EVAL_UNIT_SEC = 1.0
EMBED_MIN_SEC = 0.5
STATS_MAX_SEC = 120.0          # 声の特徴を測る発話の最大の長さ (これを超えた分は測らない)


class BuildError(ValueError):
    """録音や設定に問題があり、プロフィールを作れない。メッセージは日本語。"""


@dataclass
class BuildOptions:
    name: str
    inputs: list[Path]
    out_dir: Path
    script: Script | None = None
    script_path: str = ""
    langs: list[str] = field(default_factory=lambda: ["ja"])
    skip_first: int | None = None      # None = 自動 (台本があれば同意文も対応づけの対象にする)
    consent: bool = True               # False = 録音に同意文は入っていない
    asr: str = "none"                  # none / whisper
    asr_model: str = "small"
    n_refs: int = 4
    n_refs_other: int = 2
    min_sec: float = 8.0
    max_sec: float = 20.0
    target_sec: float = 12.0
    min_pause: float = 0.35
    embedder: str = "ecapa"
    f0: str = "harvest"
    vad: str = "auto"
    append: bool = False
    force: bool = False
    stats: bool = True
    excel: bool = True


@dataclass
class FileData:
    idx: int
    path: Path
    y: np.ndarray
    sr: int
    y16: np.ndarray
    dur: float
    out_subtype: str
    seg: Segmentation
    info: FrameInfo
    atom_ids: list[int] = field(default_factory=list)

    def segs(self) -> list[Segment]:
        return self.seg.segments


@dataclass
class BuildResult:
    options: BuildOptions
    files: list[FileData]
    atoms: list[Atom]
    units: list[Unit]
    candidates: list[Candidate]
    refs: list[Candidate]
    evals: list[EvalClip]
    unit_consistency: dict[int, dict[str, float]]
    centroids: dict[str, dict[str, np.ndarray]]      # {lang: {embedder: centroid}}
    stats: dict[str, dict]                             # {lang: 声の統計}
    align_notes: list[str]
    warnings: list[str]
    embedder_names: list[str]
    embedder_failed: dict[str, str]
    asr_name: str
    f0_method: str = ""


def _log_default(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _out_subtype(path: Path) -> str:
    try:
        st = sf.info(str(path)).subtype
    except Exception:  # noqa: BLE001 - m4a など libsndfile が読めない形式
        return "PCM_16"
    return "PCM_24" if st in ("PCM_24", "PCM_32", "FLOAT", "DOUBLE") else "PCM_16"


def _uniq(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def build_profile(
    opts: BuildOptions,
    *,
    embedders: list[Embedder] | None = None,
    asr_engine=None,
    log: Callable[[str], None] = _log_default,
) -> BuildResult:
    """計算だけを行い BuildResult を返す。embedders / asr_engine を渡すと、それを使う (テスト用)。"""
    warnings: list[str] = []
    langs = [norm_lang(x) for x in opts.langs if x.strip()] or ["ja"]
    if opts.script is not None:
        for lg in langs:
            opts.script.sentences(lg)  # 台本にない言語なら ScriptError
    if not opts.inputs:
        raise BuildError("録音ファイル (--in) を指定してください")
    if opts.min_sec <= 0 or opts.max_sec < opts.min_sec:
        raise BuildError("--min-sec と --max-sec の指定が正しくありません")

    # ---- 1. 読み込みと発話区間 ----------------------------------------------------------
    files: list[FileData] = []
    atoms: list[Atom] = []
    for k, path in enumerate(opts.inputs):
        log(f"[{k + 1}/{len(opts.inputs)}] 録音を読み込み中: {path.name}")
        y, sr = load_audio(path)
        if sr < MIN_OUTPUT_SR:
            warnings.append(f"{path.name}: サンプリング周波数が {sr} Hz で 44100 Hz 未満です。手本の高音域の情報が足りず、"
                            "音質が下がります（48 kHz で録り直すのがおすすめ）。出力の手本もこの周波数のままです")
        y16 = resample(y, sr, TARGET_SR)
        try:
            seg = detect_segments(y16, min_pause=opts.min_pause, vad=opts.vad)
        except NoSpeechError as e:
            raise BuildError(f"{path.name}: {e}") from e
        warnings += [f"{path.name}: {n}" for n in seg.notes]
        for r in seg.rejected:
            warnings.append(f"{path.name}: {r.seg.start:.1f}〜{r.seg.end:.1f} 秒を除外しました ({r.reason})")
        fd = FileData(k, path, y, sr, y16, len(y) / float(sr), _out_subtype(path), seg, build_frame_info(seg))
        drift = noise_drift_db(fd.info)
        if drift > 8.0:
            warnings.append(f"{path.name}: 雑音の大きさが録音の途中で約 {drift:.0f} dB 変わっています（扇風機・エアコン・自動音量調整など）。"
                            "発話区間の判定や SNR がずれることがあります。雑音が一定の状態で録り直すのがおすすめです")
        for pos, s in enumerate(seg.segments):
            a = Atom(len(atoms), k, pos, s)
            atoms.append(a)
            fd.atom_ids.append(a.idx)
        files.append(fd)
        log(f"    発話区間 {len(seg.segments)} 個 / 雑音の床 {seg.noise_db:.0f} dBFS / 話し声 {seg.speech_db:.0f} dBFS "
            f"/ 判定: {seg.vad_used}")
    if opts.script is None and opts.skip_first is None:
        warnings.append("台本なしで録音の最初に同意文が入っている場合は、--skip-first N（同意文の区間数。台本どおりなら 3）で除いてください"
                        "（付けないと同意文も手本の候補になります）")
    total_speech = sum(a.seg.dur for a in atoms)
    if total_speech < MIN_TOTAL_SPEECH_SEC:
        warnings.append(f"話している時間が合計 {total_speech:.0f} 秒と短いです（目安: 60 秒以上、台本どおりなら約 90 秒）。"
                        "手本の選択肢が少なくなります")

    # ---- 2. ASR (任意) ------------------------------------------------------------------
    asr_texts: list[str] | None = None
    asr_name = ""
    if asr_engine is None and opts.asr not in ("", "none"):
        from .asr import AsrUnavailable, make_asr

        log(f"ASR ({opts.asr}, モデル {opts.asr_model}) を準備中... 初回はモデルのダウンロードがあります")
        try:
            asr_engine = make_asr(opts.asr, opts.asr_model)
        except AsrUnavailable as e:
            raise BuildError(str(e)) from e
    slots: list[Slot] = []
    if opts.script is not None:
        use_consent = opts.consent and opts.skip_first is None
        slots = opts.script.slots(langs, with_consent=use_consent)
    if asr_engine is not None:
        asr_name = getattr(asr_engine, "name", "asr")
        slot_langs = {s.lang for s in slots} or set(langs)
        asr_lang = next(iter(slot_langs)) if len(slot_langs) == 1 else None
        log(f"各発話区間を書き起こし中 ({len(atoms)} 個)...")
        asr_texts = []
        for a in atoms:
            f = files[a.file]
            lo, hi = max(a.seg.start - 0.1, 0.0), min(a.seg.end + 0.1, f.dur)
            asr_texts.append(asr_engine.transcribe(f.y16[int(lo * TARGET_SR):int(hi * TARGET_SR)], asr_lang or ""))

    # ---- 3. 台本との対応づけ -> 単位 (Unit) ---------------------------------------------------
    units: list[Unit] = []
    forced = 0
    if opts.skip_first:
        first_file_atoms = [a for a in atoms if a.file == 0]
        forced = min(opts.skip_first, len(first_file_atoms))
        if opts.skip_first > len(first_file_atoms):
            warnings.append(f"--skip-first {opts.skip_first} は最初の録音の発話区間の数 ({len(first_file_atoms)}) より大きいです")
        for a in first_file_atoms[:forced]:
            units.append(Unit(len(units), a.file, [a.idx], a.seg.start, a.seg.end, a.seg.dur, [], "", langs[0],
                              "consent", STATUS_NONE, notes=["--skip-first で除外（同意文）"]))
    rest = atoms[forced:]
    align_notes: list[str] = []
    if rest:
        gaps = [(rest[i + 1].seg.start - rest[i].seg.end) if i + 1 < len(rest) and rest[i + 1].file == rest[i].file else 5.0
                for i in range(len(rest))]
        res: AlignResult = align([a.seg.dur for a in rest], [a.file for a in rest], slots,
                                 [asr_texts[a.idx] for a in rest] if asr_texts is not None else None, gaps)
        align_notes = list(res.notes)
        for b in res.blocks:
            members = [rest[i] for i in b.atoms]
            sl = [slots[j] for j in b.slots]
            slot_langs = {s.lang for s in sl}
            lang = next(iter(slot_langs)) if len(slot_langs) == 1 else langs[0]
            if not sl:
                role = "unassigned" if slots else "speech"
            elif {s.role for s in sl} == {"speech"} and len(slot_langs) == 1:
                role = "speech"
            elif {s.role for s in sl} == {"consent"}:
                role = "consent"
            else:
                role = "mixed"
            text = join_texts([s.text for s in sl], lang)
            status = b.status
            if not slots:
                status = STATUS_NONE
            asr_text = join_texts([asr_texts[a.idx] for a in members], lang) if asr_texts is not None else ""
            if not slots and asr_texts is not None:
                text, status = asr_text, STATUS_ASR_ONLY
            if role == "unassigned":
                status = STATUS_UNASSIGNED
            mora = sum(s.mora for s in sl if s.mora) if sl and all(s.mora for s in sl) else None
            units.append(Unit(
                len(units), members[0].file, [a.idx for a in members], members[0].seg.start, members[-1].seg.end,
                sum(a.seg.dur for a in members), [s.id for s in sl], text, lang, role, status, b.sim, asr_text, mora,
                notes=(["台本の複数の文が 1 つの区間にまとまっています（間をあけずに読んだ可能性）"] if len(sl) > 1 else [])
                + (["1 つの文が途中の間で複数の区間に分かれています"] if len(members) > 1 else []),
            ))
        if res.missing_slots:
            warnings.append("読まれなかったように見える文: " + ", ".join(slots[j].id for j in res.missing_slots))
    warnings += align_notes
    speech_units = [u for u in units if u.role == "speech"]
    if not speech_units:
        raise BuildError("手本にできる発話が見つかりませんでした（すべて同意文または台本に対応しない区間と判定されました）。"
                         "--skip-first や --no-consent を確かめてください")

    file_atoms = {f.idx: f.segs() for f in files}
    file_durs = {f.idx: f.dur for f in files}
    units_by_idx = {u.idx: u for u in units}

    # ---- 4. 候補 -> 品質 ---------------------------------------------------------------------
    cands = enumerate_candidates(units, atoms, file_atoms, file_durs, opts.min_sec, opts.max_sec)
    if not cands:
        relaxed = 5.0
        cands = enumerate_candidates(units, atoms, file_atoms, file_durs, relaxed, opts.max_sec)
        if cands:
            warnings.append(f"{opts.min_sec:g} 秒以上のまとまりが作れなかったため、{relaxed:g} 秒以上まで広げて候補を作りました"
                            "（短い手本は声の再現が不安定になります。もう少し長く続けて読んだ録音がおすすめ）")
    if not cands:
        raise BuildError(f"手本にできる長さ（{opts.min_sec:g}〜{opts.max_sec:g} 秒）の発話のまとまりが見つかりませんでした。"
                         "1 文が長すぎる/短すぎる、または文の間の無音が足りない可能性があります（--min-pause を小さくすると区間が細かくなります）")
    log(f"手本の候補: {len(cands)} 個。品質を測っています...")
    for c in cands:
        f = files[c.file]
        member_atoms = [atoms[i].seg for uid in c.units for i in units_by_idx[uid].atoms]
        q = measure_quality(f.info, member_atoms, c.pieces)
        c.snr_db, c.speech_dbfs, c.noise_dbfs = q.snr_db, q.speech_dbfs, q.noise_dbfs
        wave = assemble(f.y, f.sr, c.pieces)
        c.clip_ratio, c.peak_dbfs = clipping_ratio(wave), peak_dbfs(wave)
        c.notes += q.notes
        if c.clip_ratio > 0.0005:
            c.notes.append(f"音が割れている可能性（割れ率 {c.clip_ratio:.2%}）")
        if np.isfinite(c.snr_db) and c.snr_db < 20.0:
            c.notes.append(f"雑音が多め（SNR {c.snr_db:.0f} dB < 20 dB）")
        if c.status in ("check", "asr_weak"):
            c.notes.append("書き起こしの対応が怪しいので、聞いて確認してください")

    # ---- 5. 埋め込み (声の一貫性) -----------------------------------------------------------------
    embedder_failed: dict[str, str] = {}
    if embedders is None:
        embedders, embedder_failed = load_embedders(opts.embedder)
        for n, why in embedder_failed.items():
            warnings.append(f"埋め込み {n} を使えません（一貫性の列は空欄になります）: {why}")
    unit_emb: dict[str, dict[int, np.ndarray]] = {e.name: {} for e in embedders}
    if embedders:
        log(f"声の埋め込みを計算中 ({', '.join(e.name for e in embedders)}): 発話 {len(units)} 個")
    for u in units:
        if u.speech_sec < EMBED_MIN_SEC:
            continue
        f = files[u.file]
        i0, i1 = atoms[u.atoms[0]].pos, atoms[u.atoms[-1]].pos
        w = assemble(f.y16, TARGET_SR, clip_pieces(file_atoms[f.idx], i0, i1, f.dur))
        for e in embedders:
            try:
                unit_emb[e.name][u.idx] = e.embed(w)
            except Exception as ex:  # noqa: BLE001 - 1 区間の失敗で止めない
                u.notes.append(f"{e.name}: 埋め込みを計算できません ({str(ex)[:80]})")
    centroids: dict[str, dict[str, np.ndarray]] = {}
    for lang in _uniq([u.lang for u in speech_units]):
        centroids[lang] = {}
        for e in embedders:
            vs = [(unit_emb[e.name][u.idx], u.speech_sec) for u in speech_units if u.lang == lang and u.idx in unit_emb[e.name]]
            if vs:
                m = np.sum([v * w for v, w in vs], axis=0)
                n = np.linalg.norm(m)
                if n > 0:
                    centroids[lang][e.name] = m / n
    unit_consistency: dict[int, dict[str, float]] = {}
    for u in units:
        lang = u.lang
        for e in embedders:
            cen = centroids.get(lang, {}).get(e.name)
            if cen is not None and u.idx in unit_emb[e.name]:
                unit_consistency.setdefault(u.idx, {})[e.name] = float(np.dot(unit_emb[e.name][u.idx], cen))
    for c in cands:
        for e in embedders:
            cen = centroids.get(c.lang, {}).get(e.name)
            vs = [(unit_emb[e.name][i], units_by_idx[i].speech_sec) for i in c.units if i in unit_emb[e.name]]
            if cen is None or not vs:
                continue
            m = np.sum([v * w for v, w in vs], axis=0)
            n = np.linalg.norm(m)
            if n > 0:
                c.consistency[e.name] = float(np.dot(m / n, cen))
        low = [n for n, v in c.consistency.items() if v < (0.5 if n == "ecapa" else 0.75)]
        if low:
            c.notes.append("この録音の他の部分と声の特徴が違う可能性（" + ", ".join(low) + "）: 別の人の声・咳・声色の変化がないか聞いて確認")
        score_candidate(c, opts.target_sec)

    # ---- 6. 選ぶ ------------------------------------------------------------------------------------
    refs: list[Candidate] = []
    for li, lang in enumerate(_uniq([c.lang for c in cands])):
        sub = [c for c in cands if c.lang == lang]
        if li == 0:
            chosen = select_refs(sub, opts.n_refs, units_by_idx)
        else:
            chosen = select_refs(sub, opts.n_refs_other, units_by_idx, min_heldout_sec=0.0, min_refs=1)
        chosen.sort(key=lambda c: (-c.score, c.idx))
        for k, c in enumerate(chosen, start=1):
            c.selected, c.ref_name = True, f"ref_{lang}_{k:02d}.wav"
        refs += chosen
        used = {u for c in chosen for u in c.units}
        for c in sub:
            if c.selected:
                continue
            if not used.isdisjoint(c.units):
                c.not_chosen_reason = "上位の手本と同じ文を含む（重なり）"
            elif c.clip_ratio > 0.005 and any(x.clip_ratio <= 0.005 for x in sub):
                c.not_chosen_reason = "音割れが多い"
            else:
                c.not_chosen_reason = "点数が低い"
        if len(chosen) < (3 if li == 0 else 1):
            warnings.append(f"言語 {lang}: 手本が {len(chosen)} 本しか取れませんでした（目安: 3〜4 本）。もう少し読み足してください")
    if not refs:
        raise BuildError("手本を選べませんでした")
    for c in refs:  # 採用した手本は、クリップそのものの埋め込みでも一貫性を測る
        w = assemble(files[c.file].y16, TARGET_SR, c.pieces)
        for e in embedders:
            cen = centroids.get(c.lang, {}).get(e.name)
            if cen is None:
                continue
            try:
                c.consistency_true[e.name] = float(np.dot(e.embed(w), cen))
            except Exception as ex:  # noqa: BLE001
                c.notes.append(f"{e.name}: {str(ex)[:80]}")
    for c in refs:
        warnings += [f"手本 {c.ref_name}: {n}" for n in c.notes]

    # ---- 7. 評価用の取り置き (手本に使わなかった発話) -------------------------------------------------
    used_units = {u for c in refs for u in c.units}
    evals: list[EvalClip] = []
    for run in eligible_runs(units):
        for u in run:
            if u.idx in used_units or u.speech_sec < MIN_EVAL_UNIT_SEC:
                continue
            f = files[u.file]
            i0, i1 = atoms[u.atoms[0]].pos, atoms[u.atoms[-1]].pos
            pieces = clip_pieces(file_atoms[f.idx], i0, i1, f.dur)
            q = measure_quality(f.info, [atoms[i].seg for i in u.atoms], pieces)
            wave = assemble(f.y, f.sr, pieces)
            evals.append(EvalClip(len(evals), f.idx, u.lang, u.idx, pieces, pieces_duration(pieces), u.text, u.status,
                                  u.slot_ids, q.snr_db, q.speech_dbfs, clipping_ratio(wave),
                                  unit_consistency.get(u.idx, {})))
    per_lang_counter: dict[str, int] = {}
    for ev in evals:
        per_lang_counter[ev.lang] = per_lang_counter.get(ev.lang, 0) + 1
        ev.name = f"eval_{ev.lang}_{per_lang_counter[ev.lang]:02d}.wav"
    first_lang = refs[0].lang
    if sum(ev.dur for ev in evals if ev.lang == first_lang) < 10.0:
        warnings.append("評価用に取り置いた発話が合計 10 秒未満です。手本に使わなかった録音が少ないので、似ている度合いの検証が不安定になります"
                        "（もう少し長く録音するか、--n-refs 3 にしてください）")

    # ---- 8. 声の統計 (voicecheck の特徴量) ----------------------------------------------------------
    stats: dict[str, dict] = {}
    f0_method = ""
    if opts.stats:
        log("声の特徴（高さ・抑揚・傾き・CPPS・話速）を測っています...")
        for lang in _uniq([u.lang for u in speech_units]):
            parts, n_mora, sp_sec, mora_ok = [], 0, 0.0, True
            for run in eligible_runs(units):
                if run[0].lang != lang:
                    continue
                f = files[run[0].file]
                i0, i1 = atoms[run[0].atoms[0]].pos, atoms[run[-1].atoms[-1]].pos
                parts.append(assemble(f.y16, TARGET_SR, clip_pieces(file_atoms[f.idx], i0, i1, f.dur, gap_keep=0.4)))
                for u in run:
                    sp_sec += u.speech_sec
                    if u.n_mora is None or u.status in ("check", "asr_weak", "estimated"):
                        mora_ok = False
                    else:
                        n_mora += u.n_mora
            wave = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            wave = wave[: int(STATS_MAX_SEC * TARGET_SR)]  # 長い録音でも測る時間が増えすぎないように
            if len(wave) < TARGET_SR:
                continue
            feats, fnotes = extract_features(wave, TARGET_SR, opts.f0)
            f0_method = str(feats.get("f0_method") or f0_method)
            stats[lang] = _stats_dict(feats, n_mora / sp_sec if (mora_ok and sp_sec > 0 and n_mora) else None,
                                      len(wave) / TARGET_SR)
            if fnotes:
                warnings += [f"声の特徴 ({lang}): {n}" for n in fnotes]

    return BuildResult(opts, files, atoms, units, cands, refs, evals, unit_consistency, centroids, stats,
                       align_notes, _uniq(warnings), [e.name for e in embedders], embedder_failed, asr_name, f0_method)


def _nan_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _stats_dict(feats: dict, mora_per_sec: float | None, speech_sec: float) -> dict:
    """voicecheck の特徴量を、設計書 (design.md 5 章) のキー名にして返す。"""
    return {
        "f0_median_hz": _nan_none(feats.get("f0_median_hz")),
        "f0_iqr_semitone": _nan_none(feats.get("f0_iqr_st")),
        "rate_mora_per_sec": _nan_none(mora_per_sec),      # 台本が日本語で対応が確かなときだけ (発話中のモーラ/秒)
        "ltas_tilt_db": _nan_none(feats.get("tilt_db_oct")),  # 単位は dB/オクターブ
        "cpps_db": _nan_none(feats.get("cpps_db")),
        "voiced_ratio": _nan_none(feats.get("voiced_ratio")),
        "loudness_dbfs": _nan_none(feats.get("loudness_dbfs")),
        "centroid_hz": _nan_none(feats.get("centroid_hz")),
        "rate_syl_per_voiced_s": _nan_none(feats.get("rate_syl_per_voiced_s")),
        "f0_method": str(feats.get("f0_method") or ""),
        "analyzed_speech_sec": round(float(speech_sec), 2),
    }


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}秒"
