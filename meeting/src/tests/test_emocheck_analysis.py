"""emocheck の集計: 感情の正解率・混同表・信頼区間、CER (手計算と突き合わせ)"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from asrcheck.textnorm import NormOptions

from emocheck import analysis
from emocheck.dataset import Clip
from emocheck.runners import ClipResult, ModelRun

OPTS = NormOptions(lang="ja")


def make_run(preds: dict[str, tuple[str, str]], key="sensevoice", label="SenseVoiceSmall", emotion=True, sec=0.5, dur=2.0):
    """preds: file → (認識した文字, 感情タグ)"""
    run = ModelRun(key, label, emotion)
    for f, (text, tag) in preds.items():
        run.results[f] = ClipResult(text, tag, sec)
        run.durations[f] = dur
    return run


def clip(name, emo, text="", spk="", style=""):
    return Clip(name, Path(name), emo, text, spk, style)


def test_wilson_known_values():
    lo, hi = analysis.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = analysis.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = analysis.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)
    assert all(math.isnan(x) for x in analysis.wilson(0, 0))


def test_emotion_stats_matrix_and_rates():
    clips = [
        clip("a1", "happy", spk="F1"), clip("a2", "happy", spk="F1"), clip("a3", "happy", spk="M1"),
        clip("b1", "angry", spk="F1"), clip("b2", "angry", spk="M1"),
        clip("c1", "sad", spk="M1"), clip("c2", "sad", spk="M1"),
        clip("d1", "fearful", spk="F1"),
    ]
    run = make_run({
        "a1": ("", "HAPPY"), "a2": ("", "NEUTRAL"), "a3": ("", "HAPPY"),
        "b1": ("", "ANGRY"), "b2": ("", "EMO_UNKNOWN"),
        "c1": ("", ""), "c2": ("", "SAD"),
        "d1": ("", "SURPRISED"),
    })
    st = analysis.emotion_stats(clips, run)
    assert (st.n, st.correct) == (8, 4)          # a1 a3 b1 c2
    assert st.accuracy == pytest.approx(0.5)
    assert st.true_labels == ["happy", "angry", "sad", "fearful"]
    assert st.matrix["happy"]["happy"] == 2 and st.matrix["happy"]["neutral"] == 1
    assert st.matrix["angry"]["unknown"] == 1 and st.matrix["sad"]["notag"] == 1 and st.matrix["fearful"]["surprised"] == 1
    # 感情ごとの正解率と平均再現率
    rec = {r.label: r for r in st.per_emotion}
    assert rec["happy"].recall == pytest.approx(2 / 3) and rec["angry"].recall == pytest.approx(0.5)
    assert rec["sad"].recall == pytest.approx(0.5) and rec["fearful"].recall == 0
    assert st.macro_recall == pytest.approx((2 / 3 + 0.5 + 0.5 + 0) / 4)
    # 適合率: 喜びと予測した 2 件は両方正解。恐れと予測した件は 0 件 (nan)
    assert rec["happy"].pred_n == 2 and rec["happy"].precision == pytest.approx(1.0)
    assert rec["sad"].precision == pytest.approx(1.0) and math.isnan(rec["fearful"].precision)
    assert "驚き" in rec["fearful"].top_wrong and "平常" in rec["happy"].top_wrong
    # 平常・不明・タグなしの割合 (pyVideoTrans 流に読み替えると 平常 = NEUTRAL + EMO_UNKNOWN + タグなし)
    assert st.neutral_rate == pytest.approx(1 / 8) and st.unknown_rate == pytest.approx(1 / 8) and st.notag_rate == pytest.approx(1 / 8)
    assert st.folded_neutral_rate == pytest.approx(3 / 8)
    assert st.emotion_detected_rate == pytest.approx(5 / 8)
    assert st.has_neutral_truth is False
    # 3感情の見方: 恐れの音声は対象外。喜び 3 + 怒り 2 + 悲しみ 2 = 7 件、正解は a1 a3 b1 c2 = 4
    assert (st.core3_n, st.core3_correct) == (7, 4)
    assert st.core3["happy"] == {"happy": 2, "angry": 0, "sad": 0, "other": 1}
    assert st.core3["angry"]["other"] == 1 and st.core3["sad"]["other"] == 1
    assert st.core3_accuracy == pytest.approx(4 / 7)
    # 話者別
    by = {r.label: r for r in st.by_speaker}
    assert by["F1"].n == 4 and by["F1"].correct == 2 and by["M1"].n == 4 and by["M1"].correct == 2
    assert st.speaker_emotion[("M1", "happy")] == (1, 1)
    assert st.by_style == []        # style 列なし


def test_emotion_stats_skips_unprocessed_and_failed():
    clips = [clip("a", "happy"), clip("b", "happy"), clip("c", "happy")]
    run = make_run({"a": ("", "HAPPY"), "b": ("", "SAD")})
    run.results["c"] = ClipResult("", "", 0.1, error="boom")
    st = analysis.emotion_stats(clips, run)
    assert st.n == 2 and st.correct == 1


def test_emotion_stats_with_neutral_truth():
    clips = [clip("a", "neutral"), clip("b", "happy")]
    st = analysis.emotion_stats(clips, make_run({"a": ("", "NEUTRAL"), "b": ("", "HAPPY")}))
    assert st.has_neutral_truth and st.accuracy == 1.0


def test_clip_counts_matches_hand_calculation():
    # 正解 5 文字 ('今日は晴れ' の句読点は無視)、1 置換 + 1 脱落
    c = analysis.clip_counts("今日は、晴れ。", "今日わ晴", OPTS)
    assert (c.n_ref, c.sub, c.dele, c.ins) == (5, 1, 1, 0)
    c = analysis.clip_counts("あいう", "あいうえお", OPTS)
    assert (c.n_ref, c.ins) == (3, 2)
    # 表記ゆれ (全角数字・記号・スペース) は誤りにしない
    assert analysis.clip_counts("2025年です。", "２０２５年 です", OPTS).errors == 0
    assert analysis.clip_cer("", "なにか", OPTS) is None


def test_cer_table_groups_use_total_errors_over_total_chars():
    clips = [
        clip("a", "happy", "あいうえお", "F1", "regular"), clip("b", "happy", "かきくけこ", "M1", "free"),
        clip("c", "sad", "さしすせそ", "F1", "regular"), clip("d", "sad", "", "F1", "regular"),   # 正解の文章なし → 対象外
    ]
    run = make_run({"a": ("あいうえお", ""), "b": ("かきく", ""), "c": ("さしすせそそ", ""), "d": ("ゴミ", "")}, emotion=False)
    t = analysis.cer_table(clips, run, OPTS, "全件")
    assert t.n_clips == 3
    assert t.overall.counts.n_ref == 15 and t.overall.counts.errors == 3     # 脱落2 + 挿入1
    assert t.overall.cer == pytest.approx(3 / 15)
    by = {r.label: r for r in t.by_emotion}
    assert by["happy"].cer == pytest.approx(2 / 10) and by["sad"].cer == pytest.approx(1 / 5)
    assert {r.label: r.cer for r in t.by_speaker} == {"F1": pytest.approx(1 / 10), "M1": pytest.approx(2 / 5)}
    assert {r.label for r in t.by_style} == {"regular", "free"}
    assert t.raw_overall is not None


def test_cer_tables_adds_common_scope_when_models_differ():
    clips = [clip("a", "happy", "あいう"), clip("b", "sad", "えおか"), clip("c", "sad", "きくけ")]
    full = make_run({"a": ("あいう", ""), "b": ("えおか", ""), "c": ("きくけ", "")}, "sensevoice", "SV")
    part = make_run({"a": ("あいう", ""), "c": ("きくこ", "")}, "whisper", "WH", emotion=False)
    tables = analysis.cer_tables(clips, [full, part], OPTS)
    assert [(t.model, t.n_clips) for t in tables] == [("SV", 3), ("WH", 2), ("SV", 2), ("WH", 2)]
    assert "共通" in tables[2].scope
    # 同じ範囲なら共通の表は足さない
    same = analysis.cer_tables(clips, [full, make_run({"a": ("あ", ""), "b": ("", ""), "c": ("", "")}, "w", "W", False)], OPTS)
    assert len(same) == 2


def test_speed_excludes_failed():
    run = make_run({"a": ("", ""), "b": ("", "")}, sec=1.0, dur=4.0)
    run.results["c"] = ClipResult("", "", 99.0, error="x")
    run.durations["c"] = 100.0
    sp = analysis.speed(run)
    assert sp["n"] == 2 and sp["rtf"] == pytest.approx(0.25) and sp["sec_per_clip"] == pytest.approx(1.0)


def test_cer_tables_skip_runs_that_do_not_report_cer():
    clips = [clip("a", "happy", "あいう")]
    sv = make_run({"a": ("あいう", "HAPPY")}, "sensevoice", "SV")
    ban = make_run({"a": ("あいう", "SAD")}, "sensevoice_nounk", "SV禁止")
    ban.report_cer = False
    tables = analysis.cer_tables(clips, [sv, ban], OPTS)
    assert [t.model for t in tables] == ["SV"]
    # 感情の集計は両方できる
    assert analysis.emotion_stats(clips, sv).accuracy == 1.0 and analysis.emotion_stats(clips, ban).accuracy == 0.0


def test_precision_exposes_a_model_that_answers_angry_for_everything():
    clips = [clip(f"a{i}", "angry") for i in range(4)] + [clip(f"s{i}", "sad") for i in range(4)] + [clip(f"h{i}", "happy") for i in range(2)]
    run = make_run({c.file: ("", "ANGRY") for c in clips})
    rows = {r.label: r for r in analysis.emotion_stats(clips, run).per_emotion}
    assert rows["angry"].recall == 1.0 and rows["angry"].precision == pytest.approx(0.4)      # 再現率は 100% でも、適合率は 4/10
    assert rows["sad"].recall == 0 and rows["happy"].recall == 0
