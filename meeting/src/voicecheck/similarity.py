"""類似度 (コサイン)・重心・EER (等誤り率) などの計算。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度 (-1〜1)。1 に近いほど似ている。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """埋め込みの平均 (長さ1に正規化してから平均し、もう一度正規化)。"""
    units = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    units = [v / np.linalg.norm(v) for v in units if np.linalg.norm(v) > 0]
    m = np.mean(units, axis=0)
    return m / np.linalg.norm(m)


@dataclass
class EerResult:
    eer: float        # 等誤り率 (0〜1)
    threshold: float  # そのときのしきい値 (SIM がこの値以上なら「同じ人」)
    far: float        # しきい値での誤受理率: 別人なのに通ってしまう割合
    frr: float        # しきい値での誤拒否率: 本人なのに落ちる割合


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> EerResult:
    """本人どうしのスコア (genuine) と他人どうしのスコア (impostor) から EER としきい値を求める。

    しきい値 t のとき「SIM >= t なら同じ人」とみなし、
    FAR = 他人のうち t 以上の割合、FRR = 本人のうち t 未満の割合。FAR と FRR が等しくなる点が EER。
    """
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    i = np.sort(np.asarray(impostor, dtype=np.float64))
    if len(g) == 0 or len(i) == 0:
        raise ValueError("同じ人どうし・別の人どうしの両方のペアが必要です")
    thr = np.unique(np.concatenate([g, i]))
    frr = np.searchsorted(g, thr, side="left") / len(g)
    far = 1.0 - np.searchsorted(i, thr, side="left") / len(i)
    gap = np.abs(far - frr)
    best = np.flatnonzero(gap <= gap.min() + 1e-12)
    k = int(best[len(best) // 2])  # 同点が続くときは真ん中
    eer = float((far[k] + frr[k]) / 2.0)
    t = float(thr[k])
    if far[k] == 0 and frr[k] == 0:  # 完全に分かれている → 隙間の真ん中を使う
        t = float((i.max() + g.min()) / 2.0)
    return EerResult(eer=eer, threshold=t, far=float(far[k]), frr=float(frr[k]))


def threshold_at_far(impostor: np.ndarray, far: float = 0.01) -> float:
    """別人どうしのスコアの上位 far (例 1%) の境目。これ以上を「同じ人」にすると別人はほぼ通らない。"""
    return float(np.quantile(np.asarray(impostor, dtype=np.float64), 1.0 - far))


def rate_at_or_above(scores: np.ndarray, t: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    return float((s >= t).mean()) if len(s) else float("nan")


def pair_scores(vectors: np.ndarray, speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """全ペアのコサイン類似度を (同じ話者どうし, 別の話者どうし) に分けて返す。

    vectors は 1 行 1 ファイルで、行ごとに長さ1に正規化されていること。
    """
    X = np.asarray(vectors, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    sp = np.asarray(speakers)
    same = sp[:, None] == sp[None, :]
    iu = np.triu_indices(len(sp), k=1)
    s, m = S[iu], same[iu]
    return s[m], s[~m]


def histogram_table(
    groups: dict[str, np.ndarray], step: float = 0.05
) -> tuple[list[tuple[float, float]], dict[str, np.ndarray]]:
    """スコアを step 幅の区間ごとに数える。全グループで同じ区間をそろえる。"""
    allv = np.concatenate([np.asarray(v, dtype=np.float64) for v in groups.values() if len(v)] or [np.zeros(1)])
    lo = np.floor(allv.min() / step) * step
    hi = np.ceil(allv.max() / step) * step
    if hi <= lo:
        hi = lo + step
    edges = np.round(np.arange(lo, hi + step / 2, step), 10) + 0.0  # + 0.0 は -0.0 を 0.0 にするため
    bins = [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    counts = {k: np.histogram(np.asarray(v, dtype=np.float64), bins=edges)[0] for k, v in groups.items()}
    return bins, counts
