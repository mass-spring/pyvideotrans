"""asrcheck の指標 (編集距離・CER・WER・cpCER・DER・話者正解率) を、手計算の小さな例と、総当たり(全探索)の計算で検証する。"""
from __future__ import annotations

import itertools
import math
import random

import pytest

from asrcheck import editdist
from asrcheck.diar import compute_der, speaker_accuracy
from asrcheck.metrics import cp_cer, text_errors
from asrcheck.model import Utt
from asrcheck.textnorm import NormOptions

JA = NormOptions()
EN = NormOptions(lang="en")


def U(start, end, spk, text):
    return Utt(float(start), float(end), spk, text)


def lev(a, b):
    """素朴な (遅い) 編集距離。検証用。"""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# 編集距離
# ---------------------------------------------------------------------------
def test_edit_distance_hand_examples():
    assert editdist.distance("kitten", "sitting") == 3
    assert editdist.distance("", "abc") == 3 and editdist.distance("abc", "") == 3 and editdist.distance("", "") == 0
    c = editdist.count_ops(editdist.align("kitten", "sitting"))
    assert (c.n_ref, c.errors, c.sub, c.dele, c.ins) == (6, 3, 2, 0, 1)
    assert editdist.align("abc", "axc") == [("=", 0, 0), ("S", 1, 1), ("=", 2, 2)]
    assert editdist.align("ab", "") == [("D", 0, -1), ("D", 1, -1)]
    assert editdist.align("", "ab") == [("I", -1, 0), ("I", -1, 1)]


def test_edit_distance_matches_reference_dp_on_random_strings():
    rnd = random.Random(1)
    for _ in range(200):
        a = "".join(rnd.choice("abcd") for _ in range(rnd.randint(0, 30)))
        b = "".join(rnd.choice("abcd") for _ in range(rnd.randint(0, 30)))
        assert editdist.distance(a, b) == lev(a, b)
        ops = editdist.align(a, b)
        c = editdist.count_ops(ops)
        assert c.errors == lev(a, b) and c.n_ref == len(a)
        # 操作列を実際に適用すると b になる
        out = []
        for op, i, j in ops:
            if op in "=S":
                out.append(b[j])
            elif op == "I":
                out.append(b[j])
        assert "".join(out) == b


def test_hirschberg_path_equals_full_dp(monkeypatch):
    monkeypatch.setattr(editdist, "_SMALL_CELLS", 50)   # 小さい入力でも分割統治の経路を通す
    rnd = random.Random(7)
    for _ in range(60):
        a = "".join(rnd.choice("あいうえ") for _ in range(rnd.randint(20, 80)))
        b = list(a)
        for _ in range(rnd.randint(0, 15)):
            k = rnd.randrange(len(b) + 1)
            r = rnd.random()
            if r < 0.4 and k < len(b):
                b[k] = "x"
            elif r < 0.7 and k < len(b):
                del b[k]
            else:
                b.insert(k, "y")
        b = "".join(b)
        c = editdist.count_ops(editdist.align(a, b))
        assert c.errors == lev(a, b) and c.n_ref == len(a)


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------
def test_cer_hand_computed():
    # 正解 7 文字 「今日は晴れです」 / 認識 「今日は雨です」: 晴れ→雨 = 置換1 + 脱落1 → 2/7
    r = text_errors([U(0, 3, "A", "今日は晴れです")], [U(0, 3, "A", "今日は雨です")], JA)
    c = r.counts
    assert (c.n_ref, c.sub, c.dele, c.ins, c.errors) == (7, 1, 1, 0, 2)
    assert c.rate == pytest.approx(2 / 7)


def test_cer_can_exceed_100_percent_with_insertions():
    r = text_errors([U(0, 1, "A", "あ")], [U(0, 1, "A", "あいう")], JA)
    assert r.counts.rate == pytest.approx(2.0)


def test_cer_ignores_punctuation_and_notation_variants():
    ref = [U(0, 3, "A", "今日は、2025年の会議です。"), U(3, 6, "B", "コーヒーを２杯ください！")]
    hyp = [U(0, 3, "x", "今日は 二千二十五年の会議です"), U(3, 6, "y", "ｺｰﾋｰを二杯ください")]
    r = text_errors(ref, hyp, JA)
    assert r.counts.errors == 0 and r.counts.n_ref == 24      # 句読点・全角半角・数字の書き方の違いは誤りにしない
    assert r.raw_counts.errors > 0                             # 正規化なしでは誤りに見える


def test_cer_number_normalization_effect():
    ref = [U(0, 3, "A", "売上は三千万円でした")]
    hyp = [U(0, 3, "A", "売上は3000万円でした")]
    assert text_errors(ref, hyp, JA).counts.errors == 0
    assert text_errors(ref, hyp, NormOptions(numbers="keep")).counts.errors > 0
    # 正規化なしの CER は別に持つ
    r = text_errors(ref, hyp, JA)
    assert r.raw_counts.errors > 0


def test_cer_orders_utterances_by_time_and_ignores_speaker():
    ref = [U(5, 8, "B", "二番目"), U(0, 3, "A", "一番目")]
    hyp = [U(0, 3, "x", "一番目"), U(5, 8, "x", "二番目")]
    assert text_errors(ref, hyp, JA).counts.errors == 0


def test_cer_empty_hypothesis_is_all_deletions():
    r = text_errors([U(0, 3, "A", "こんにちは")], [], JA)
    assert (r.counts.n_ref, r.counts.dele, r.counts.rate) == (5, 5, 1.0)


def test_wer_hand_computed_en():
    # the cat sat / the bat sat down → 置換1 + 挿入1 = 2/3
    r = text_errors([U(0, 1, "A", "The cat sat.")], [U(0, 1, "A", "the bat sat down")], EN)
    c = r.counts
    assert (c.n_ref, c.sub, c.ins, c.dele) == (3, 1, 1, 0)
    assert c.rate == pytest.approx(2 / 3)
    assert r.unit == "単語"


def test_wer_fr_apostrophes_and_case():
    ref = [U(0, 1, "A", "L’homme est là.")]
    hyp = [U(0, 1, "A", "l'homme est la")]
    assert text_errors(ref, hyp, NormOptions(lang="fr")).counts.errors == 1               # là ≠ la (アクセントは区別)
    assert text_errors(ref, hyp, NormOptions(lang="fr", strip_accents=True)).counts.errors == 0


def test_per_utterance_diffs_sum_to_total():
    rnd = random.Random(3)
    chars = "あいうえおかきくけこ"
    ref = [U(i * 3, i * 3 + 2, "AB"[i % 2], "".join(rnd.choice(chars) for _ in range(rnd.randint(3, 12)))) for i in range(12)]
    hyp = []
    for u in ref:
        t = list(u.text)
        for _ in range(rnd.randint(0, 3)):
            k = rnd.randrange(len(t) + 1)
            r = rnd.random()
            if r < 0.4 and k < len(t):
                t[k] = "x"
            elif r < 0.7 and k < len(t):
                del t[k]
            else:
                t.insert(k, "y")
        hyp.append(U(u.start, u.end, u.speaker, "".join(t)))
    res = text_errors(ref, hyp, NormOptions(numbers="keep"))
    tot = res.counts
    assert sum(d.counts.sub for d in res.diffs) == tot.sub
    assert sum(d.counts.dele for d in res.diffs) == tot.dele
    assert sum(d.counts.ins for d in res.diffs) == tot.ins
    assert sum(d.counts.n_ref for d in res.diffs) == tot.n_ref
    # 色付けの断片をつなぐと、正解側は正解の文字列、認識側は認識結果の文字列に戻る
    assert "".join(p.text for d in res.diffs for p in d.ref_pieces) == "".join(u.text for u in ref)
    assert "".join(d.hyp_text for d in res.diffs) == "".join(u.text for u in hyp)


def test_diff_pieces_mark_kinds():
    res = text_errors([U(0, 1, "A", "あいうえお")], [U(0, 1, "A", "あXうえおか")], JA)
    d = res.diffs[0]
    assert [(p.kind, p.text) for p in d.ref_pieces] == [("ok", "あ"), ("sub", "い"), ("ok", "うえお")]
    assert [(p.kind, p.text) for p in d.hyp_pieces] == [("ok", "あ"), ("sub", "x"), ("ok", "うえお"), ("ins", "か")]


# ---------------------------------------------------------------------------
# cpCER
# ---------------------------------------------------------------------------
def test_cpcer_perfect_with_renamed_speakers():
    ref = [U(0, 3, "A", "こんにちは"), U(3, 6, "B", "さようなら"), U(6, 9, "A", "また明日")]
    hyp = [U(0, 3, "spk1", "こんにちは"), U(3, 6, "spk0", "さようなら"), U(6, 9, "spk1", "また明日")]
    r = cp_cer(ref, hyp, JA)
    assert r.counts.errors == 0 and r.mapping == {"A": "spk1", "B": "spk0"}


def test_cpcer_speaker_confusion_counts_errors_but_cer_does_not():
    ref = [U(0, 3, "A", "ああああ"), U(3, 6, "B", "いいいい")]
    hyp = [U(0, 3, "x", "ああああ"), U(3, 6, "x", "いいいい")]   # 全部1人にまとめた
    assert text_errors(ref, hyp, JA).counts.errors == 0
    r = cp_cer(ref, hyp, JA)
    # x は ref の片方にだけ対応: 対応した側は 4 文字ぶん挿入、対応のない側は 4 文字ぶん脱落 → 誤り 8 / 8 文字
    assert (r.counts.n_ref, r.counts.ins, r.counts.dele, r.counts.errors) == (8, 4, 4, 8)


def test_cpcer_extra_and_missing_speakers():
    ref = [U(0, 3, "A", "あいうえお")]
    hyp = [U(0, 3, "x", "あいうえお"), U(3, 6, "y", "かきく")]
    r = cp_cer(ref, hyp, JA)
    assert r.counts.ins == 3 and r.counts.errors == 3 and r.unmatched_hyp == ["y"]
    ref2 = [U(0, 3, "A", "あいうえお"), U(3, 6, "B", "かきく")]
    hyp2 = [U(0, 3, "x", "あいうえお")]
    r2 = cp_cer(ref2, hyp2, JA)
    assert r2.counts.dele == 3 and r2.counts.errors == 3 and r2.per_speaker["B"].dele == 3


def test_cpcer_none_without_hyp_speakers():
    assert cp_cer([U(0, 1, "A", "あ")], [U(0, 1, "", "あ")], JA) is None


def test_cpcer_equals_bruteforce_permutations():
    rnd = random.Random(5)
    for _ in range(25):
        nr, nh = rnd.randint(1, 3), rnd.randint(1, 4)
        rs = {f"R{i}": "".join(rnd.choice("あいう") for _ in range(rnd.randint(1, 8))) for i in range(nr)}
        hs = {f"H{i}": "".join(rnd.choice("あいう") for _ in range(rnd.randint(1, 8))) for i in range(nh)}
        ref = [U(i * 10, i * 10 + 5, k, v) for i, (k, v) in enumerate(rs.items())]
        hyp = [U(i * 10, i * 10 + 5, k, v) for i, (k, v) in enumerate(hs.items())]
        got = cp_cer(ref, hyp, NormOptions(numbers="keep")).counts.errors
        # 全順列: 正解の話者と認識側の話者 (足りない側は空の話者で補う) の全ての1対1
        rl, hl = list(rs.values()), list(hs.values())
        n = max(len(rl), len(hl))
        rl += [""] * (n - len(rl))
        hl += [""] * (n - len(hl))
        best = min(sum(lev(a, b) for a, b in zip(rl, perm)) for perm in itertools.permutations(hl))
        assert got == best


# ---------------------------------------------------------------------------
# DER
# ---------------------------------------------------------------------------
def test_der_hand_computed_confusion():
    r = compute_der([(0, 10, "A"), (10, 20, "B")], [(0, 8, "X"), (8, 20, "Y")])
    assert r.total == 20 and r.confusion == pytest.approx(2) and r.miss == 0 and r.false_alarm == 0
    assert r.der == pytest.approx(0.1) and r.mapping == {"A": "X", "B": "Y"}


def test_der_overlap_is_counted_as_miss():
    # A:0-10, B:5-15 (5-10 が重なり) / 認識: X:0-10, Y:10-15 → 5-10 で正解2人・認識1人 → 見逃し5秒、全体20秒
    r = compute_der([(0, 10, "A"), (5, 15, "B")], [(0, 10, "X"), (10, 15, "Y")])
    assert (r.total, r.miss, r.false_alarm, r.confusion) == (20, 5, 0, 0)
    assert r.der == pytest.approx(0.25) and r.ref_overlap_time == pytest.approx(5)


def test_der_miss_and_false_alarm():
    r = compute_der([(0, 10, "A")], [(2, 12, "X")])
    assert (r.miss, r.false_alarm, r.confusion, r.total) == (2, 2, 0, 10)
    assert r.der == pytest.approx(0.4)


def test_der_collar_removes_boundary_errors():
    ref, hyp = [(0, 10, "A")], [(0.2, 10.1, "X")]
    r0 = compute_der(ref, hyp, collar=0.0)
    assert r0.miss == pytest.approx(0.2) and r0.false_alarm == pytest.approx(0.1) and r0.der == pytest.approx(0.03)
    r1 = compute_der(ref, hyp, collar=0.25)
    assert r1.der == 0 and r1.total == pytest.approx(9.5)   # 前後 0.25 秒ずつ採点から外れる
    assert compute_der(ref, [(0.4, 10.0, "X")], collar=0.25).miss == pytest.approx(0.15)   # 0.25 を超えた分だけ誤り


def test_der_collar_merges_touching_same_speaker_segments():
    # 同じ話者の隣り合う区間の境目 (5.0) は境界とみなさない
    r = compute_der([(0, 5, "A"), (5, 10, "A")], [(0, 4.9, "X"), (5.1, 10, "X")], collar=0.25)
    assert r.miss == pytest.approx(0.2) and r.total == pytest.approx(9.5)


def test_der_no_hypothesis_is_all_miss():
    r = compute_der([(0, 10, "A")], [])
    assert r.miss == 10 and r.der == pytest.approx(1.0)


def test_der_empty_reference_is_nan():
    assert math.isnan(compute_der([], [(0, 5, "X")]).der)


def test_der_speaker_names_do_not_matter():
    ref = [(0, 4, "A"), (4, 9, "B"), (9, 12, "A")]
    h1 = [(0, 4, "p"), (4, 9, "q"), (9, 12, "p")]
    h2 = [(0, 4, "q"), (4, 9, "p"), (9, 12, "q")]
    assert compute_der(ref, h1).der == compute_der(ref, h2).der == 0


def _brute_der(ref, hyp, step=0.5):
    """時間を step 秒の格子に刻み、話者の対応を全順列で探して md-eval の式で DER を出す (検証用)。"""
    T = max(e for _, e, _ in ref + hyp)
    n = int(round(T / step))

    def frames(segs):
        d = {}
        for s, e, spk in segs:
            d.setdefault(spk, set()).update(range(int(round(s / step)), int(round(e / step))))
        return d

    rf, hf = frames(ref), frames(hyp)
    rn, hn = sorted(rf), sorted(hf)
    pad = hn + [None] * max(0, len(rn) - len(hn))
    best = None
    for perm in itertools.permutations(pad, len(rn)):
        ov = sum(len(rf[r] & hf[h]) for r, h in zip(rn, perm) if h is not None)
        if best is None or ov > best[0]:
            best = (ov, perm)
    mapping = dict(zip(rn, best[1]))
    miss = fa = conf = tot = 0.0
    for t in range(n):
        ra = [r for r in rn if t in rf[r]]
        ha = [h for h in hn if t in hf[h]]
        nref, nsys = len(ra), len(ha)
        ncorr = sum(1 for r in ra if mapping[r] is not None and mapping[r] in ha)
        tot += nref
        miss += max(0, nref - nsys)
        fa += max(0, nsys - nref)
        conf += min(nref, nsys) - ncorr
    return tot * step, miss * step, fa * step, conf * step


def test_der_equals_bruteforce_on_random_segments():
    rnd = random.Random(11)
    for _ in range(40):
        def segs(names):
            out = []
            for spk in names:
                t = 0.0
                for _ in range(rnd.randint(1, 4)):
                    t += rnd.choice([0, 0.5, 1.0, 2.0])
                    d = rnd.choice([0.5, 1.0, 2.5, 4.0])
                    out.append((t, t + d, spk))
                    t += d
            return out

        ref = segs(["A", "B", "C"][: rnd.randint(1, 3)])
        hyp = segs(["x", "y", "z", "w"][: rnd.randint(1, 4)])
        tot, miss, fa, conf = _brute_der(ref, hyp)
        r = compute_der(ref, hyp)
        assert (r.total, r.miss, r.false_alarm, r.confusion) == pytest.approx((tot, miss, fa, conf)), (ref, hyp)


# ---------------------------------------------------------------------------
# 話者正解率 (発言単位)
# ---------------------------------------------------------------------------
def test_speaker_accuracy_hand_computed():
    ref = [(0, 5, "A"), (5, 10, "B"), (10, 12, "A")]
    hyp = [(0, 5, "x"), (5, 10, "x"), (10, 12, "y")]
    r = speaker_accuracy(ref, hyp)
    # 対応づけは「正解になる発言が最大」: y→A, x→B で 2/3
    assert r.n == 3 and r.n_correct == 2 and r.accuracy == pytest.approx(2 / 3)
    assert r.mapping == {"y": "A", "x": "B"}
    assert [u.correct for u in r.utts] == [False, True, True]


def test_speaker_accuracy_majority_overlap_decides():
    ref = [(0, 10, "A")]
    hyp = [(0, 3, "x"), (3, 10, "y")]      # y が 7 秒で多数
    r = speaker_accuracy(ref, hyp)
    assert r.utts[0].dominant == "y" and r.accuracy == 1.0   # 1発言1話者なら対応づけで必ず正解にできる


def test_speaker_accuracy_no_overlap_counts_as_wrong():
    r = speaker_accuracy([(0, 5, "A"), (20, 25, "B")], [(0, 5, "x")])
    assert r.n == 2 and r.n_correct == 1 and r.utts[1].dominant is None and not r.utts[1].correct
    assert int(r.no_overlap_counts.sum()) == 1


def test_speaker_accuracy_time_weighted():
    ref = [(0, 1, "A"), (1, 10, "B")]
    hyp = [(0, 1, "p"), (1, 10, "p")]      # 全部1人 → 長い B の発言だけ対応
    r = speaker_accuracy(ref, hyp)
    assert r.accuracy == pytest.approx(0.5) and r.accuracy_time == pytest.approx(9 / 10)


def test_speaker_accuracy_ignores_utterances_without_speaker_or_time():
    r = speaker_accuracy([(0, 5, "A"), (5, 5, "B"), (6, 8, "")], [(0, 5, "x")])
    assert r.n == 1 and r.accuracy == 1.0
