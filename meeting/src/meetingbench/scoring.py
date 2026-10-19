"""測定結果を asrcheck で採点する（CER・cpCER・DER・話者正解率）。

各条件のフォルダ（pipeline.stage_assign が作る）を、2通りで採点する。
  字幕単位 : 字幕1行に1人の話者（speaker.json）だけを使う。アプリが実際に出す結果と同じ見方（話者正解率の KPI はこちら）
  生出力   : 話者分離モデルが出した話者ごとの区間（raw_diar.csv）で DER と話者正解率を測る（話者分離モデルそのものの力）
さらに、区間の切り方だけが原因で下がる分を知るために、各字幕に『正解の話者』を付けた条件（上限）も採点する。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from asrcheck.cli import load_reference
from asrcheck.evaluate import EngineResult, EvalOptions, evaluate_engine
from asrcheck.kpi import DEFAULT_KPIS, Kpi
from asrcheck.model import Utt
from asrcheck.report import summary_rows
from asrcheck.textnorm import NormOptions

from . import diarize as dz
from .pipeline import Layout

TRUTH_CFG = "truth"   # 正解の話者を付けた（区間の切り方だけの影響を見る）上限の条件


@dataclass
class Cond:
    variant: str
    seg: str
    cfg: str
    n: str
    dir: Path
    sub: EngineResult
    raw: EngineResult | None
    row_sub: dict
    row_raw: dict | None
    times: dict = field(default_factory=dict)   # vad / asr / diar / total（秒）と rtf
    info: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.variant}:{self.seg}/{self.cfg}/{self.n}"


# ---------------------------------------------------------------------------
# 純粋な計算
# ---------------------------------------------------------------------------
def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def truth_labels(ref: list[Utt], items: list[dict]) -> list[str]:
    """各字幕（start_ms,end_ms）に、いちばん長く重なった正解の話者名を付ける（重なりがなければ最も近い発言の話者）。"""
    out = []
    for it in items:
        s, e = it["start_ms"] / 1000, it["end_ms"] / 1000
        acc: dict[str, float] = {}
        for u in ref:
            if u.has_time:
                acc[u.speaker] = acc.get(u.speaker, 0.0) + overlap(s, e, u.start, u.end)  # type: ignore[arg-type]
        best = max(acc, key=acc.get) if acc and max(acc.values()) > 0 else None
        if best is None:
            mid = (s + e) / 2
            best = min((u for u in ref if u.has_time), key=lambda u: abs((u.start + u.end) / 2 - mid)).speaker  # type: ignore[operator]
        out.append(best)
    return out


def segment_stats(ref: list[Utt], items: list[dict], min_overlap_s: float = 0.3) -> dict:
    """区間の切り方の診断。

    mixed_pct  : 字幕のうち、正解の話者が2人以上（それぞれ min_overlap_s 秒以上）含まれる割合 → 1人しか割り当てられないので必ず誤る
    split_pct  : 正解の発言のうち、2つ以上の字幕にまたがって（それぞれ min_overlap_s 秒以上）切れた割合
    """
    n = len(items)
    mixed = 0
    for it in items:
        s, e = it["start_ms"] / 1000, it["end_ms"] / 1000
        spk = {u.speaker for u in ref if u.has_time and overlap(s, e, u.start, u.end) >= min_overlap_s}  # type: ignore[arg-type]
        mixed += len(spk) >= 2
    split = 0
    for u in ref:
        if not u.has_time:
            continue
        k = sum(1 for it in items if overlap(it["start_ms"] / 1000, it["end_ms"] / 1000, u.start, u.end) >= min_overlap_s)  # type: ignore[arg-type]
        split += k >= 2
    durs = sorted((it["end_ms"] - it["start_ms"]) / 1000 for it in items)
    return {
        "segments": n,
        "median_s": durs[len(durs) // 2] if durs else None,
        "max_s": durs[-1] if durs else None,
        "mixed_pct": mixed / n if n else None,
        "split_pct": split / max(1, sum(1 for u in ref if u.has_time)),
    }


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------
def default_opts() -> EvalOptions:
    return EvalOptions(norm=NormOptions(lang="ja"), collar=0.0)


def write_truth_conds(lay: Layout, v: str, ref: list[Utt], say=print) -> list[Path]:
    """文字起こしがある区間の種類ごとに、正解の話者を付けた評価用フォルダを作る（上限の条件）。"""
    outs = []
    for asr_file in sorted(lay.runs(v).glob("asr_*.json")):
        seg = asr_file.stem[len("asr_"):]
        items = [it for it in json.loads(asr_file.read_text(encoding="utf-8"))["items"] if it["text"].strip()]
        cd = lay.runs(v) / "cond" / f"{seg}__{TRUTH_CFG}__none"
        cd.mkdir(parents=True, exist_ok=True)
        dz.write_srt(cd / f"meeting{v}.srt", items)
        (cd / "speaker.json").write_text(json.dumps(truth_labels(ref, items)), encoding="utf-8")
        outs.append(cd)
    return outs


def _times(lay: Layout, v: str, seg: str, cfg: str, n: str, audio_s: float) -> dict:
    t: dict = {}
    sf = lay.runs(v) / f"seg_{seg}.json"
    if sf.is_file():
        t["vad"] = json.loads(sf.read_text(encoding="utf-8"))["meta"].get("elapsed_sec") if seg != "oracle" else 0.0
    af = lay.runs(v) / f"asr_{seg}.json"
    if af.is_file():
        t["asr"] = json.loads(af.read_text(encoding="utf-8"))["meta"].get("elapsed_sec")
    if cfg != TRUTH_CFG:
        mf = lay.runs(v) / "diar" / f"{cfg}_{n}" / "meta.json"
        if mf.is_file():
            t["diar"] = json.loads(mf.read_text(encoding="utf-8")).get("elapsed_sec")
        parts = [t.get("vad"), t.get("asr"), t.get("diar")]
        if all(p is not None for p in parts):
            t["total"] = sum(parts)
            t["rtf"] = t["total"] / audio_s
    return t


def score_variant(lay: Layout, v: str, kpis: dict[str, Kpi] | None = None, opts: EvalOptions | None = None) -> list[Cond]:
    kpis = kpis or DEFAULT_KPIS
    opts = opts or default_opts()
    ref_by, names, _notes, _enc, durs = load_reference(str(lay.ref(v)), [str(lay.wav(v))])
    ref_utts = next(iter(ref_by.values()))
    write_truth_conds(lay, v, ref_utts)
    audio_s = next(iter(durs.values()))
    conds: list[Cond] = []
    for cd in sorted((lay.runs(v) / "cond").glob("*__*__*")):
        seg, cfg, n = cd.name.split("__")
        t = _times(lay, v, seg, cfg, n, audio_s)
        name = f"{v}:{seg}/{cfg}/{n}"
        total = t.get("total")
        sub = evaluate_engine(name, ref_by, names, [("", str(cd))], None, opts, lang="ja", audio_s=durs, elapsed_s=total)
        raw = None
        rawcsv = cd / "raw_diar.csv"
        if rawcsv.is_file():
            raw = evaluate_engine(name + "·生", ref_by, names, [("", str(cd))], str(rawcsv), opts, lang="ja", audio_s=durs, elapsed_s=total)
        row_sub = summary_rows([sub], kpis, opts)[0]
        row_raw = summary_rows([raw], kpis, opts)[0] if raw else None
        info = json.loads((cd / "meta.json").read_text(encoding="utf-8")) if (cd / "meta.json").is_file() else {}
        conds.append(Cond(v, seg, cfg, n, cd, sub, raw, row_sub, row_raw, t, info))
    return conds


def segment_report(lay: Layout, v: str) -> list[dict]:
    ref_by, _n, _no, _e, _d = load_reference(str(lay.ref(v)), [str(lay.wav(v))])
    ref = next(iter(ref_by.values()))
    out = []
    for af in sorted(lay.runs(v).glob("asr_*.json")):
        seg = af.stem[len("asr_"):]
        d = json.loads(af.read_text(encoding="utf-8"))
        items = [it for it in d["items"] if it["text"].strip()]
        st = segment_stats(ref, items)
        st.update({"variant": v, "seg": seg, "empty_dropped": len(d["items"]) - len(items), "asr_sec": d["meta"].get("elapsed_sec"),
                   "asr_rtf": d["meta"].get("rtf")})
        sf = lay.runs(v) / f"seg_{seg}.json"
        st["vad_sec"] = json.loads(sf.read_text(encoding="utf-8"))["meta"].get("elapsed_sec") if seg != "oracle" and sf.is_file() else None
        out.append(st)
    return out
