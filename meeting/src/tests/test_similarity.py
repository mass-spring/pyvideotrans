"""類似度・EER・ヒストグラムの計算。"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from voicecheck.similarity import (centroid, compute_eer, cosine, histogram_table, pair_scores, threshold_at_far)


def test_cosine_and_centroid():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, b) == pytest.approx(0.0)
    assert cosine(a, -a) == pytest.approx(-1.0)
    assert np.isnan(cosine(a, np.zeros(3)))
    c = centroid([a, b])
    assert np.linalg.norm(c) == pytest.approx(1.0)
    assert cosine(c, a) == pytest.approx(np.sqrt(0.5))
    # 長さが違っても向きだけで平均する
    assert cosine(centroid([a * 10, b]), a) == pytest.approx(np.sqrt(0.5))


def test_eer_two_gaussians():
    """N(0.7, 0.1) と N(0.2, 0.1): しきい値 0.45、EER = Φ(-2.5) ≒ 0.62%。"""
    rng = np.random.default_rng(0)
    g = rng.normal(0.7, 0.1, 50000)
    i = rng.normal(0.2, 0.1, 50000)
    r = compute_eer(g, i)
    assert r.threshold == pytest.approx(0.45, abs=0.02)
    assert r.eer == pytest.approx(norm.cdf(-2.5), abs=0.002)
    assert r.far == pytest.approx(r.frr, abs=0.003)


def test_eer_perfect_separation_uses_midpoint_of_gap():
    r = compute_eer(np.array([0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3]))
    assert r.eer == 0.0
    assert r.threshold == pytest.approx(0.5)  # 0.3 と 0.7 の真ん中


def test_eer_identical_distributions_is_about_half():
    rng = np.random.default_rng(1)
    r = compute_eer(rng.normal(0.5, 0.1, 20000), rng.normal(0.5, 0.1, 20000))
    assert r.eer == pytest.approx(0.5, abs=0.02)


def test_eer_small_hand_example():
    # 本人 [0.4, 0.6, 0.8, 0.9]、他人 [0.1, 0.2, 0.5, 0.7]
    # しきい値 0.6: FRR=1/4(0.4が落ちる), FAR=1/4(0.7が通る) → EER 25%
    r = compute_eer(np.array([0.4, 0.6, 0.8, 0.9]), np.array([0.1, 0.2, 0.5, 0.7]))
    assert r.eer == pytest.approx(0.25)
    assert 0.5 < r.threshold <= 0.7


def test_eer_needs_both_groups():
    with pytest.raises(ValueError):
        compute_eer(np.array([]), np.array([0.1]))


def test_threshold_at_far():
    imp = np.linspace(0, 1, 1001)
    assert threshold_at_far(imp, 0.01) == pytest.approx(0.99, abs=0.005)


def test_pair_scores_splits_same_and_different():
    X = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
    same, diff = pair_scores(X, ["a", "a", "b", "b"])
    assert len(same) == 2 and len(diff) == 4        # a-a, b-b / 4通りの a-b
    assert same.min() > 0.99 and diff.max() < 0.2


def test_histogram_counts_sum_to_pairs():
    rng = np.random.default_rng(2)
    same, diff = rng.normal(0.7, 0.1, 500), rng.normal(0.2, 0.1, 800)
    bins, counts = histogram_table({"same": same, "diff": diff}, step=0.05)
    assert counts["same"].sum() == 500 and counts["diff"].sum() == 800
    assert all(b > a for a, b in bins)
    assert bins[0][0] <= min(same.min(), diff.min()) and bins[-1][1] >= max(same.max(), diff.max())


def test_strict_threshold_never_below_eer_threshold_when_separated(tmp_path, monkeypatch):
    """完全に分かれた少人数データでも、厳しめしきい値はおすすめ (EER) しきい値以上になる"""
    import numpy as np
    from voicecheck.similarity import compute_eer, threshold_at_far
    same = np.array([0.70, 0.75, 0.80, 0.85])
    diff = np.array([0.10, 0.15, 0.20, 0.38])
    eer = compute_eer(same, diff)
    assert threshold_at_far(diff, 0.01) < eer.threshold  # 素の値は逆転する (これが不具合の原因)
    assert max(threshold_at_far(diff, 0.01), eer.threshold) >= eer.threshold
