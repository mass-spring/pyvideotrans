"""録音の発話区間と台本の文の対応づけ (align.py) のテスト。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voiceprofile.align import (
    STATUS_ASR,
    STATUS_ASR_WEAK,
    STATUS_CHECK,
    STATUS_COUNT_OK,
    STATUS_ESTIMATED,
    STATUS_GAP_OK,
    STATUS_NONE,
    STATUS_UNASSIGNED,
    align,
    merge_by_gaps,
    worst_status,
)
from voiceprofile.script import load_script

SCRIPT_JSON = Path(__file__).resolve().parents[2] / "enroll" / "script_ja.json"


@pytest.fixture(scope="module")
def slots():
    return load_script(SCRIPT_JSON).slots(["ja"])        # 同意文 3 + 日本語 18 = 21


def _base(slots, seed, jitter=0.08):
    rng = np.random.default_rng(seed)
    return [s.est_sec * (1 + rng.uniform(-jitter, jitter)) for s in slots], rng


def _blocks(res, slots):
    return [(tuple(b.atoms), tuple(slots[j].id for j in b.slots)) for b in res.blocks]


def _identity_expected(slots):
    return [((i,), (s.id,)) for i, s in enumerate(slots)]


def test_equal_counts_map_in_reading_order_without_asr(slots):
    durs, _ = _base(slots, 0)
    res = align(durs, [0] * len(durs), slots)
    assert res.mode == "order"
    assert _blocks(res, slots) == _identity_expected(slots)
    assert {b.status for b in res.blocks} == {STATUS_COUNT_OK}
    assert res.missing_slots == [] and res.notes == []


def test_equal_counts_but_very_different_length_is_flagged_for_checking(slots):
    durs, _ = _base(slots, 0)
    durs[5] *= 6                                           # 1 つだけ極端に長い
    res = align(durs, [0] * len(durs), slots)
    assert res.blocks[5].status == STATUS_CHECK
    assert res.blocks[4].status == STATUS_COUNT_OK
    assert any("確認" in n for n in res.notes)


def test_no_slots_means_no_script(slots):
    res = align([2.0, 3.0], [0, 0], [])
    assert [b.status for b in res.blocks] == [STATUS_NONE, STATUS_NONE]
    assert all(b.slots == [] for b in res.blocks)
    with pytest.raises(ValueError):
        align([], [], slots)


@pytest.mark.parametrize("seed", range(5))
def test_two_sentences_read_without_a_pause_are_mapped_to_one_segment(slots, seed):
    durs, rng = _base(slots, seed)
    k = int(rng.integers(3, 18))
    merged = durs[:k] + [durs[k] + durs[k + 1]] + durs[k + 2:]
    res = align(merged, [0] * len(merged), slots, None, [1.1] * len(merged))
    exp = _identity_expected(slots)
    exp = exp[:k] + [((k,), (slots[k].id, slots[k + 1].id))] + [((i - 1,), ids) for (i,), ids in exp[k + 2:]]
    assert _blocks(res, slots) == exp
    assert res.mode == "duration" and any("推定" in n for n in res.notes)
    assert res.blocks[k].status == STATUS_ESTIMATED


@pytest.mark.parametrize("seed", range(5))
def test_a_sentence_split_by_a_mid_sentence_pause_is_one_unit(slots, seed):
    durs, rng = _base(slots, seed)
    k = int(rng.integers(3, 20))
    f = rng.uniform(0.4, 0.6)
    split = durs[:k] + [durs[k] * f * 0.9, durs[k] * (1 - f) * 0.9] + durs[k + 1:]
    gaps = [1.1] * len(split)
    gaps[k] = 0.5                                          # 文の途中の短い間
    res = align(split, [0] * len(split), slots, None, gaps)
    exp = _identity_expected(slots)
    exp = exp[:k] + [((k, k + 1), (slots[k].id,))] + [((i + 1,), ids) for (i,), ids in exp[k + 1:]]
    assert _blocks(res, slots) == exp


@pytest.mark.parametrize("seed", range(5))
def test_a_false_start_after_a_long_pause_is_left_unassigned(slots, seed):
    durs, rng = _base(slots, seed)
    k = int(rng.integers(3, 20))
    with_extra = durs[:k] + [float(rng.uniform(0.8, 1.6))] + durs[k:]
    gaps = [float(rng.uniform(0.9, 1.5)) for _ in with_extra]
    res = align(with_extra, [0] * len(with_extra), slots, None, gaps)
    assert res.blocks[k].slots == [] and res.blocks[k].status == STATUS_UNASSIGNED
    assert any("対応しなかった" in n for n in res.notes)
    # それ以外は順番どおり
    rest = [b for b in res.blocks if b.slots]
    assert [tuple(slots[j].id for j in b.slots) for b in rest] == [(s.id,) for s in slots]


def test_a_skipped_sentence_is_found_often_but_never_marked_trustworthy(slots):
    """長さだけでは、飛ばした文を言い当てられないことがある。当たらなくても『数が一致』の確かな状態にはならないこと。"""
    hits = 0
    for seed in range(20):
        durs, rng = _base(slots, seed)
        k = int(rng.integers(4, 20))
        res = align(durs[:k] + durs[k + 1:], [0] * (len(durs) - 1), slots, None, [1.1] * (len(durs) - 1))
        assert all(b.status != STATUS_COUNT_OK for b in res.blocks)
        if k in res.missing_slots and len(res.missing_slots) == 1:
            hits += 1
    assert hits >= 8                                        # 長さだけの推定。約半分は当たる (外れる時もある)


def test_segments_from_different_files_are_never_merged_into_one_sentence(slots):
    durs, _ = _base(slots, 1)
    split = durs[:5] + [durs[5] * 0.5, durs[5] * 0.5] + durs[6:]
    files = [0] * 6 + [1] * (len(split) - 6)               # 6 番目と 7 番目のあいだでファイルが変わる
    res = align(split, files, slots, None, [0.5] * len(split))
    assert all(len({files[a] for a in b.atoms}) == 1 for b in res.blocks)


# ---- ASR の結果 (文字列) を使う対応づけ -------------------------------------------------------------------
def _asr(texts, noise_every=0):
    out = []
    for i, t in enumerate(texts):
        if noise_every and i % noise_every == 0:
            t = t.replace("ー", "").replace("、", "")[: max(len(t) - 3, 4)]   # 語尾が欠ける・長音が落ちる程度の誤認識
        out.append(t)
    return out


def test_asr_alignment_is_exact_with_a_skipped_sentence(slots):
    durs, _ = _base(slots, 3)
    k = 9
    keep = [i for i in range(len(slots)) if i != k]
    res = align([durs[i] for i in keep], [0] * len(keep), slots, _asr([slots[i].text for i in keep], noise_every=3))
    assert res.mode == "asr"
    assert res.missing_slots == [k]
    assert _blocks(res, slots) == [((a,), (slots[i].id,)) for a, i in enumerate(keep)]
    assert all(b.status in (STATUS_ASR, STATUS_ASR_WEAK) for b in res.blocks)
    assert sum(b.status == STATUS_ASR for b in res.blocks) >= len(keep) - 2


def test_asr_alignment_handles_skip_merge_and_false_start_together(slots):
    durs, rng = _base(slots, 4)
    seq_durs, seq_text, expected = [], [], []
    i = 0
    atom = 0
    while i < len(slots):
        if i == 6:                                             # 飛ばす
            i += 1
            continue
        if i == 10:                                            # 2 文が続けて読まれて 1 区間
            seq_durs.append(durs[i] + durs[i + 1])
            seq_text.append(slots[i].text + slots[i + 1].text)
            expected.append(((atom,), (slots[i].id, slots[i + 1].id)))
            atom += 1
            i += 2
            continue
        if i == 14:                                            # 言い直し: 直前に短い失敗の区間
            seq_durs.append(1.2)
            seq_text.append("えーと、あの")
            expected.append(((atom,), ()))
            atom += 1
        seq_durs.append(durs[i])
        seq_text.append(slots[i].text)
        expected.append(((atom,), (slots[i].id,)))
        atom += 1
        i += 1
    res = align(seq_durs, [0] * len(seq_durs), slots, seq_text, [1.1] * len(seq_durs))
    assert _blocks(res, slots) == expected
    assert res.missing_slots == [6]
    assert res.blocks[[b.atoms for b in res.blocks].index([next(a for (a,), ids in expected if ids == ())])].status == STATUS_UNASSIGNED


def test_asr_text_that_differs_from_the_script_is_not_trusted(slots):
    durs, _ = _base(slots, 5)
    texts = [s.text for s in slots]
    texts[7] = "全然ちがう内容の文章を話しています"
    res = align(durs, [0] * len(durs), slots, texts, [1.1] * len(durs))
    by_atom = {tuple(b.atoms): b for b in res.blocks}
    b7 = by_atom[(7,)]
    # 文と対応づけるなら「怪しい」状態、そうでなければ『どの文にも対応しない区間』+『読まれなかった文』になる
    assert b7.status in (STATUS_ASR_WEAK, STATUS_CHECK, STATUS_UNASSIGNED)
    if b7.status == STATUS_UNASSIGNED:
        assert 7 in res.missing_slots
    assert by_atom[(8,)].status == STATUS_ASR and by_atom[(8,)].slots == [8]    # 隣は影響されない


def test_worst_status_picks_the_least_trustworthy():
    assert worst_status([STATUS_ASR, STATUS_COUNT_OK]) == STATUS_COUNT_OK
    assert worst_status([STATUS_COUNT_OK, STATUS_ESTIMATED, STATUS_ASR]) == STATUS_ESTIMATED
    assert worst_status([STATUS_ASR, STATUS_CHECK]) == STATUS_CHECK
    assert worst_status([]) == STATUS_NONE


# ---- 読点で息をついて分かれた区間を、間の長さで文にまとめる -------------------------------------------------
def _commas(slots, seed, comma_gap=(0.36, 0.6), sentence_gap=(0.85, 1.5), p_split=0.5):
    """各文を 1〜3 個の区間に分け (読点で息をつく)、区間の長さ・間・正解のまとまりを返す。"""
    rng = np.random.default_rng(seed)
    durs, gaps, exp = [], [], []
    for j, s in enumerate(slots):
        total = s.est_sec * (1 + rng.uniform(-0.08, 0.08))
        k = 1 if rng.random() > p_split else int(rng.integers(2, 4))
        cuts = np.sort(rng.uniform(0.25, 0.75, size=k - 1)) if k > 1 else np.array([])
        edges = np.concatenate([[0.0], cuts, [1.0]])
        first = len(durs)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            durs.append(total * (b - a))
            gaps.append(float(rng.uniform(*comma_gap)))
        gaps[-1] = float(rng.uniform(*sentence_gap))
        exp.append((tuple(range(first, len(durs))), (s.id,)))
    return durs, gaps, exp


@pytest.mark.parametrize("seed", range(8))
def test_commas_are_merged_by_gap_length_when_the_gap_structure_is_clear(slots, seed):
    durs, gaps, exp = _commas(slots, seed)
    assert len(durs) > len(slots)                          # 区間の方が文より多い
    res = align(durs, [0] * len(durs), slots, None, gaps)
    assert res.mode == "gap"
    assert _blocks(res, slots) == exp
    assert {b.status for b in res.blocks} == {STATUS_GAP_OK}
    assert any("読点" in n for n in res.notes)


def test_gap_merge_is_not_used_when_comma_and_sentence_gaps_overlap(slots):
    durs, gaps, exp = _commas(slots, 3, comma_gap=(0.4, 1.3), sentence_gap=(0.8, 1.4))      # 読点の間が文の間と同じくらい長い
    res = align(durs, [0] * len(durs), slots, None, gaps)
    assert res.mode == "duration"                          # 長さで推定する方に切り替わる
    assert all(b.status != STATUS_GAP_OK for b in res.blocks)


def test_merge_by_gaps_unit():
    files = [0] * 6
    gaps = [0.4, 1.2, 0.45, 1.1, 0.0, 0.0]
    assert merge_by_gaps(files, gaps, 3) == [[0, 1], [2, 3], [4, 5]]     # 長い間 1.2 と 1.1 が切れ目 (次は 0.45 で、はっきり離れている)
    assert merge_by_gaps(files, gaps, 6) is None            # 区間 (6) と文 (6) が同じ数なら使わない
    assert merge_by_gaps(files, gaps, 4) is None            # 4 つにするには 0.45 も切れ目にするが、次の 0.4 と 0.05 しか違わず決められない
    assert merge_by_gaps([0, 0, 1, 1], [0.4, 9.0, 0.4, 0.0], 2) == [[0, 1], [2, 3]]    # ファイルの変わり目は最初から切れ目


@pytest.mark.parametrize("seed", range(8))
def test_gap_merge_does_not_force_a_wrong_grouping_after_a_skipped_sentence(slots, seed):
    """文を 1 つ読み飛ばし、そのうえ読点の区間が増えて区間数が文数より多くなっても、間の長さで無理にまとめて『確か』と言わない。"""
    durs, gaps, exp = _commas(slots, seed, p_split=0.7)
    rng = np.random.default_rng(100 + seed)
    k = int(rng.integers(5, 20))
    drop = set(exp[k][0])
    keep = [i for i in range(len(durs)) if i not in drop]
    d2, g2 = [durs[i] for i in keep], [gaps[i] for i in keep]
    if len(d2) <= len(slots):
        pytest.skip("この乱数では区間数が文数以下になった")
    res = align(d2, [0] * len(d2), slots, None, g2)
    assert all(b.status != STATUS_GAP_OK for b in res.blocks)
    assert res.mode == "duration"
