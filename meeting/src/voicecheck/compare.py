"""compare: 手本 (登録音声) と評価音声の比較。行ごとの指標と、話者ごとの集計を作る。"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .analysis import Analyzer, ClipResult
from .pairs import PairRow, emotion_match
from .similarity import centroid, cosine

NAN = float("nan")


@dataclass
class RowResult:
    no: int
    row: PairRow
    sims: dict[str, float] = field(default_factory=dict)       # 手本の重心との SIM {埋め込み名: 値}
    sims_src: dict[str, float] = field(default_factory=dict)   # 元発話との SIM
    diffs: dict[str, float] = field(default_factory=dict)      # 手本との特徴量の差
    diffs_src: dict[str, float] = field(default_factory=dict)  # 元発話との特徴量の差
    emotion_match: bool | None = None
    notes: list[str] = field(default_factory=list)
    test_duration_s: float = NAN
    n_refs_used: int = 0


# ---- 特徴量の差 -----------------------------------------------------------------
def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float, np.floating)) and math.isfinite(v) for v in vals)


def _sub(a: float, b: float) -> float:
    return float(a - b) if _finite(a, b) else NAN


def _ratio(a: float, b: float) -> float:
    return float(a / b) if _finite(a, b) and b > 0 else NAN


def _semitones(a: float, b: float) -> float:
    return float(12.0 * math.log2(a / b)) if _finite(a, b) and a > 0 and b > 0 else NAN


def aggregate_features(feats: list[dict]) -> dict:
    """手本が複数のとき、各特徴量の中央値で1つにまとめる。"""
    out: dict = {}
    for k in feats[0]:
        vals = [f.get(k, NAN) for f in feats]
        nums = [float(v) for v in vals if isinstance(v, (int, float, np.floating)) and math.isfinite(v)]
        out[k] = float(np.median(nums)) if nums else NAN
    return out


def feature_diffs(test: dict, ref: dict) -> dict[str, float]:
    """評価音声 (test) と基準 (ref) の差。比は test / ref、差は test - ref。"""
    g = lambda d, k: d.get(k, NAN)  # noqa: E731
    return {
        "delta_f0_st": _semitones(g(test, "f0_median_hz"), g(ref, "f0_median_hz")),
        "f0_iqr_ratio": _ratio(g(test, "f0_iqr_st"), g(ref, "f0_iqr_st")),
        "tilt_diff": _sub(g(test, "tilt_db_oct"), g(ref, "tilt_db_oct")),
        "rate_ratio": _ratio(g(test, "rate_syl_per_voiced_s"), g(ref, "rate_syl_per_voiced_s")),
        "loud_diff": _sub(g(test, "loudness_dbfs"), g(ref, "loudness_dbfs")),
        "centroid_diff": _sub(g(test, "centroid_hz"), g(ref, "centroid_hz")),
        "cpps_diff": _sub(g(test, "cpps_db"), g(ref, "cpps_db")),
    }


# ---- 行ごと ----------------------------------------------------------------------
def _tag(prefix: str, clip: ClipResult) -> list[str]:
    if not clip.ok:
        return [f"{prefix}: {clip.error}"]
    return [f"{prefix}: {n}" for n in clip.notes]


def run_compare(rows: list[PairRow], analyzer: Analyzer, names: list[str]) -> list[RowResult]:
    """pairs.csv の全行を処理する。失敗した行は備考に理由を書いて続行する。"""
    paths = [p for r in rows for p in [*r.refs, r.test, r.source] if p]
    analyzer.prefetch(paths)

    results: list[RowResult] = []
    for no, row in enumerate(rows, start=1):
        res = RowResult(no=no, row=row, notes=list(row.problems))
        res.emotion_match = emotion_match(row.emotion_src, row.emotion_test)

        test = analyzer.get(row.test) if row.test else None
        if test is not None:
            res.notes += _tag("評価音声", test)
            res.test_duration_s = test.duration_s

        refs: list[ClipResult] = []
        for k, rp in enumerate(row.refs, start=1):
            clip = analyzer.get(rp)
            label = f"手本{k}({os.path.basename(rp)})" if len(row.refs) > 1 else "手本"
            res.notes += _tag(label, clip)
            if clip.ok:
                refs.append(clip)
        res.n_refs_used = len(refs)

        src = analyzer.get(row.source) if row.source else None
        if src is not None:
            res.notes += _tag("元音声", src)

        if test is not None and test.ok:
            for name in names:
                tv = test.embeddings.get(name)
                rvs = [c.embeddings[name] for c in refs if name in c.embeddings]
                if tv is not None and rvs:
                    res.sims[name] = cosine(tv, centroid(rvs))
                if src is not None and src.ok and name in src.embeddings and tv is not None:
                    res.sims_src[name] = cosine(tv, src.embeddings[name])
            if test.features and refs and all(c.features for c in refs):
                res.diffs = feature_diffs(test.features, aggregate_features([c.features for c in refs]))
            if src is not None and src.ok and test.features and src.features:
                res.diffs_src = feature_diffs(test.features, src.features)
            if refs and not res.sims and names:
                res.notes.append("SIM を計算できませんでした (上の備考を確認)")
        results.append(res)
    return results


# ---- 話者ごと・ラベルごとの集計 ---------------------------------------------------------
def _vals(xs: list[float]) -> list[float]:
    return [x for x in xs if _finite(x)]


def _mean(xs: list[float]) -> float:
    v = _vals(xs)
    return float(np.mean(v)) if v else NAN


def _std(xs: list[float]) -> float:
    v = _vals(xs)
    return float(np.std(v, ddof=1)) if len(v) >= 2 else NAN


def _min(xs: list[float]) -> float:
    v = _vals(xs)
    return float(np.min(v)) if v else NAN


ALL_SPEAKERS = "（全員）"


def _aggregate(speaker: str, label: str, rs: list[RowResult], names: list[str], thresholds: dict[str, float]) -> dict:
    out: dict = {"speaker": speaker, "label": label, "n_rows": len(rs)}
    out["n_valid"] = sum(1 for r in rs if r.sims)
    for name in names:
        sims = [r.sims.get(name, NAN) for r in rs]
        out[f"sim_mean:{name}"] = _mean(sims)
        out[f"sim_min:{name}"] = _min(sims)
        out[f"sim_std:{name}"] = _std(sims)
        v = _vals(sims)
        t = thresholds.get(name)
        out[f"pass_rate:{name}"] = (sum(1 for x in v if x >= t) / len(v)) if (t is not None and v) else NAN
    d = lambda k: [r.diffs.get(k, NAN) for r in rs]  # noqa: E731
    out["delta_f0_st_mean"] = _mean(d("delta_f0_st"))
    out["delta_f0_st_absmean"] = _mean([abs(x) if _finite(x) else NAN for x in d("delta_f0_st")])
    out["f0_iqr_ratio_mean"] = _mean(d("f0_iqr_ratio"))
    out["tilt_diff_mean"] = _mean(d("tilt_diff"))
    out["rate_ratio_mean"] = _mean(d("rate_ratio"))
    out["loud_diff_mean"] = _mean(d("loud_diff"))
    em = [r.emotion_match for r in rs if r.emotion_match is not None]
    out["emotion_n"] = len(em)
    out["emotion_match_rate"] = (sum(em) / len(em)) if em else NAN
    return out


def summarize(results: list[RowResult], names: list[str], thresholds: dict[str, float]) -> list[dict]:
    """ラベルごとに、話者別の行 + 全員の行を作る。"""
    summary: list[dict] = []
    labels = list(dict.fromkeys(r.row.label for r in results))
    for label in labels:
        subset = [r for r in results if r.row.label == label]
        speakers = list(dict.fromkeys(r.row.speaker for r in subset))
        for sp in speakers:
            summary.append(_aggregate(sp, label, [r for r in subset if r.row.speaker == sp], names, thresholds))
        if len(speakers) > 1:
            summary.append(_aggregate(ALL_SPEAKERS, label, subset, names, thresholds))
    return summary
