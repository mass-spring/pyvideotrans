"""正解と認識結果を突き合わせて、全部の指標を計算する。"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .diar import DerResult, Seg, SpeakerAccResult, compute_der, speaker_accuracy
from .editdist import ErrorCounts
from .ioformats import InputError, read_hypothesis
from .metrics import CpCerResult, TextResult, cp_cer, text_errors
from .model import Utt, rec_key, sort_by_time
from .textnorm import NormOptions

DER_COLLARS = (0.0, 0.25)


@dataclass
class EvalOptions:
    norm: NormOptions = field(default_factory=NormOptions)
    collar: float = 0.0   # KPI 判定に使う DER の collar


@dataclass
class RecResult:
    rec: str
    ref: list[Utt]
    hyp: list[Utt]
    text: TextResult | None = None
    cp: CpCerResult | None = None
    der: dict[float, DerResult] = field(default_factory=dict)
    spk: SpeakerAccResult | None = None
    ref_speech_s: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class EngineResult:
    name: str
    recs: list[RecResult]
    kind: str = ""                                  # 入力形式の説明
    source: str = ""                                # 入力のパス
    notes: list[str] = field(default_factory=list)
    elapsed_s: float | None = None                  # 処理にかかった時間 (秒)
    audio_s: float | None = None                    # 音声の長さ (秒)

    # ---- まとめ (全録音を合算) ----
    def _sum(self, attr: str, sub: str | None = None) -> ErrorCounts | None:
        total = ErrorCounts()
        got = False
        for r in self.recs:
            t = getattr(r, attr)
            if t is None:
                continue
            c = getattr(t, sub) if sub else t
            if c is None:
                continue
            total += c
            got = True
        return total if got else None

    @property
    def counts(self) -> ErrorCounts | None:
        return self._sum("text", "counts")

    @property
    def raw_counts(self) -> ErrorCounts | None:
        return self._sum("text", "raw_counts")

    @property
    def cp_counts(self) -> ErrorCounts | None:
        return self._sum("cp", "counts")

    def der_total(self, collar: float) -> DerResult | None:
        total = DerResult(collar=collar)
        got = False
        for r in self.recs:
            d = r.der.get(collar)
            if d is not None:
                total.add(d)
                got = True
        return total if got else None

    @property
    def spk_counts(self) -> tuple[int, int] | None:
        n = sum(r.spk.n for r in self.recs if r.spk)
        k = sum(r.spk.n_correct for r in self.recs if r.spk)
        return (k, n) if any(r.spk for r in self.recs) and n else None

    @property
    def spk_time(self) -> float | None:
        tot = cor = 0.0
        for r in self.recs:
            if r.spk:
                tot += sum(u.duration_s for u in r.spk.utts)
                cor += sum(u.duration_s for u in r.spk.utts if u.correct)
        return cor / tot if tot > 0 else None

    @property
    def minutes_per_hour(self) -> float | None:
        """1時間の音声あたりの処理時間 (分)。"""
        if self.elapsed_s is None or not self.audio_s:
            return None
        return self.elapsed_s / self.audio_s * 60.0


def audio_duration(path: str | Path) -> float | None:
    """音声ファイルの長さ (秒)。soundfile → ffprobe の順に試し、だめなら None。"""
    p = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(p))
        if info.samplerate and info.frames:
            return info.frames / info.samplerate
    except Exception:  # noqa: BLE001
        pass
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(p)],
                                 capture_output=True, text=True, timeout=60)
            return float(out.stdout.strip())
        except Exception:  # noqa: BLE001
            return None
    return None


def _speech_time(utts: list[Utt]) -> float:
    from .diar import merge_same_speaker

    segs = [(u.start, u.end, u.speaker) for u in utts if u.has_time and u.speaker]
    return sum(e - s for s, e, _ in merge_same_speaker(segs))


def evaluate_recording(rec: str, ref: list[Utt], hyp: list[Utt], diar_hyp: list[Utt] | None, opts: EvalOptions,
                       uem: tuple[float, float] | None = None, expect_text: bool = False, expect_speakers: bool = False) -> RecResult:
    """expect_text / expect_speakers: 他の録音の認識結果には文字 / 話者があるとき True。
    この録音の認識結果が空 (認識結果が無かった) でも、『全部脱落・全部見逃し』として数える。"""
    res = RecResult(rec=rec, ref=sort_by_time(ref), hyp=sort_by_time(hyp))
    ref_segs: list[Seg] = [(u.start, u.end, u.speaker) for u in res.ref if u.has_time and u.speaker]  # type: ignore[misc]
    res.ref_speech_s = _speech_time(res.ref)
    if expect_text or any(u.text.strip() for u in hyp):
        res.text = text_errors(res.ref, res.hyp, opts.norm)
        res.cp = cp_cer(res.ref, res.hyp, opts.norm, force=expect_speakers)
        if res.cp is None and ref_segs:
            res.notes.append("認識結果に話者がないため cpCER は計算しません")
    else:
        res.notes.append("認識結果に文字がないため CER は計算しません (話者区間だけの入力)")

    src = diar_hyp if diar_hyp is not None else hyp
    hyp_segs: list[Seg] = [(u.start, u.end, u.speaker) for u in sort_by_time(src) if u.has_time and u.speaker]  # type: ignore[misc]
    if (hyp_segs or expect_speakers) and ref_segs:
        for c in sorted({*DER_COLLARS, opts.collar}):
            res.der[c] = compute_der(ref_segs, hyp_segs, collar=c, uem=uem)
        # index が res.ref (時刻順) の位置と一致するよう、時刻や話者のない行は空の話者にして渡す (中で読み飛ばされる)
        res.spk = speaker_accuracy(
            [(u.start or 0.0, u.end or 0.0, u.speaker if u.has_time else "") for u in res.ref],
            hyp_segs,
        )
    elif not hyp_segs:
        res.notes.append("認識結果に話者または時刻がないため、DER・話者正解率は計算しません")
    return res


def load_meta(path: Path) -> dict:
    """run が書いた <名前>.meta.json (処理時間など) があれば読む。"""
    cands = [path.with_suffix(".meta.json"), Path(str(path) + ".meta.json")] if path.is_file() else [path / "meta.json"]
    for c in cands:
        if c.is_file():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def evaluate_engine(
    name: str,
    ref_by_rec: dict[str, list[Utt]],
    rec_names: dict[str, str],
    hyp_specs: list[tuple[str, str]],
    diar_spec: str | None,
    opts: EvalOptions,
    lang: str = "ja",
    audio_s: dict[str, float] | None = None,
    elapsed_s: float | None = None,
) -> EngineResult:
    """hyp_specs: [(録音の指定 (空なら自動), パス)]。"""
    eng = EngineResult(name=name, recs=[])
    audio_s = audio_s or {}
    hyp_by_rec: dict[str, list[Utt]] = {}
    diar_by_rec: dict[str, list[Utt]] = {}
    kinds: list[str] = []
    metas: list[dict] = []
    only = next(iter(ref_by_rec)) if len(ref_by_rec) == 1 else None

    def assign(utts: list[Utt], rec_hint: str, path: str, bucket: dict[str, list[Utt]]) -> None:
        warned = False
        for u in utts:
            if rec_hint:
                key = rec_key(rec_hint)
            elif u.rec and (only is None or rec_key(u.rec) in ref_by_rec):
                key = rec_key(u.rec)
            elif only is not None:
                key = only
                if u.rec and not warned:   # RTTM や run の出力は音声ファイル名が入っている。録音が1本ならそのまま使う
                    eng.notes.append(f"{Path(path).name}: 録音名 '{u.rec}' は正解の録音 '{rec_names[only]}' と違いますが、録音が1本なのでそのまま使います")
                    warned = True
            else:
                raise InputError(
                    f"認識結果 {path} のどの録音かが分かりません。録音が複数あるので、"
                    "--hyp 名前@録音名=パス の形で録音を指定するか、ファイルに file 列を付けてください"
                )
            bucket.setdefault(key, []).append(u)

    for rec_hint, path in hyp_specs:
        utts, notes, kind = read_hypothesis(path, lang)
        kinds.append(kind)
        eng.notes += [f"{Path(path).name}: {n}" for n in notes]
        assign(utts, rec_hint, path, hyp_by_rec)
        m = load_meta(Path(path))
        if m:
            metas.append(m)
    if diar_spec:
        utts, notes, _ = read_hypothesis(diar_spec, lang)
        eng.notes += [f"{Path(diar_spec).name}: {n}" for n in notes]
        assign(utts, "", diar_spec, diar_by_rec)
    eng.kind = " + ".join(dict.fromkeys(kinds))
    eng.source = "; ".join(p for _, p in hyp_specs)

    expect_text = any(u.text.strip() for lst in hyp_by_rec.values() for u in lst)
    expect_speakers = any(u.has_time and u.speaker for lst in (diar_by_rec or hyp_by_rec).values() for u in lst)
    unknown = [k for k in hyp_by_rec if k not in ref_by_rec]
    for k in unknown:
        eng.notes.append(f"認識結果の録音 '{k}' は正解にありません。無視します")
    for key, ref in ref_by_rec.items():
        hyp = hyp_by_rec.get(key)
        if hyp is None:
            eng.notes.append(f"録音 '{rec_names[key]}' の認識結果がありません。空の認識結果として数えます (CER 100%)")
            hyp = []
        diar = diar_by_rec.get(key) if diar_spec else None
        r = evaluate_recording(rec_names[key], ref, hyp, diar, opts, expect_text=expect_text, expect_speakers=expect_speakers)
        eng.recs.append(r)

    if elapsed_s is None:
        vals = [m.get("elapsed_sec") for m in metas if isinstance(m.get("elapsed_sec"), (int, float))]
        elapsed_s = float(sum(vals)) if vals else None
    eng.elapsed_s = elapsed_s
    meta_audio = [m.get("audio_sec") for m in metas if isinstance(m.get("audio_sec"), (int, float))]
    if audio_s:
        eng.audio_s = sum(audio_s.get(k, 0.0) for k in ref_by_rec) or None
    if eng.audio_s is None and meta_audio:
        eng.audio_s = float(sum(meta_audio))
    return eng
