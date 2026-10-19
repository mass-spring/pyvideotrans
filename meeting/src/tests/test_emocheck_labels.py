"""emocheck の感情ラベル: 名前の読み替え・SenseVoice タグの取り出し・pyVideoTrans 側との対応の一致"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from emocheck import labels

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("text,expected", [
    ("anger", "angry"), ("Angry", "angry"), ("怒り", "angry"), ("happy", "happy"), ("喜び", "happy"), ("sad", "sad"),
    ("悲しみ", "sad"), ("fear", "fearful"), ("FEARFUL", "fearful"), ("disgust", "disgusted"), ("嫌悪", "disgusted"),
    ("surprise", "surprised"), ("驚き", "surprised"), ("neutral", "neutral"), ("平常", "neutral"), (" HAPPY ", "happy"),
])
def test_parse_label(text, expected):
    assert labels.parse_label(text) == expected


def test_parse_label_unknown_has_helpful_message():
    with pytest.raises(ValueError) as e:
        labels.parse_label("ぷんぷん")
    assert "ぷんぷん" in str(e.value) and "喜び" in str(e.value)


def test_split_sensevoice_text_keeps_unknown_tag():
    assert labels.split_sensevoice_text("<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは") == ("こんにちは", "HAPPY")
    # asrcheck.runners.split_sensevoice_tags は EMO_UNKNOWN を捨てるが、ここでは残す
    assert labels.split_sensevoice_text("<|ja|><|EMO_UNKNOWN|><|Speech|><|withitn|>あ い") == ("あ い", "EMO_UNKNOWN")
    assert labels.split_sensevoice_text("<|ja|><|Speech|>だけ") == ("だけ", "")
    assert labels.split_sensevoice_text("") == ("", "")
    # 小文字で来ても大文字にそろえる。最初に見つかった感情タグを使う
    assert labels.split_sensevoice_text("<|sad|><|HAPPY|>x")[1] == "SAD"


def test_predicted_label():
    assert labels.predicted_label("ANGRY") == "angry"
    assert labels.predicted_label("NEUTRAL") == "neutral"
    assert labels.predicted_label("EMO_UNKNOWN") == labels.UNKNOWN
    assert labels.predicted_label("") == labels.NOTAG
    assert labels.predicted_label("SOMETHING_NEW") == labels.UNKNOWN


def test_fold_and_core3():
    assert labels.fold_to_pvt(labels.UNKNOWN) == "neutral" and labels.fold_to_pvt(labels.NOTAG) == "neutral"
    assert labels.fold_to_pvt("sad") == "sad"
    assert labels.core3_view("happy") == "happy"
    assert labels.core3_view("fearful") == labels.OTHER and labels.core3_view(labels.UNKNOWN) == labels.OTHER


def test_every_label_has_japanese_name():
    for k in labels.PRED_LABELS:
        assert k in labels.JA
    assert set(labels.ORDER) == set(labels.EMOTIONS)


def test_matches_pyvideotrans_emotion_module():
    """videotrans/util/emotion.py (標準ライブラリだけの小さいファイル) を直接読み込んで、対応表が一致していることを確かめる。"""
    path = REPO / "videotrans" / "util" / "emotion.py"
    if not path.is_file():
        pytest.skip("videotrans/util/emotion.py が無い")
    spec = importlib.util.spec_from_file_location("pvt_emotion_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(labels.EMOTIONS) == set(mod.LABELS)
    assert labels.SV_TAG_TO_LABEL == mod.SENSEVOICE_TAG_MAP
    assert labels.SV_EMOTION_TAGS == set(mod.SENSEVOICE_EMOTION_TAGS)
    # pyVideoTrans は 不明・タグなし・未知のタグを平常に読み替える。fold_to_pvt(predicted_label(...)) が同じ結果になる
    for tag in [*mod.SENSEVOICE_EMOTION_TAGS, "", None, "WHATEVER"]:
        assert labels.fold_to_pvt(labels.predicted_label(tag or "")) == mod.map_sensevoice_tag(tag)[0]


def test_emotion2vec_names_are_read_like_sensevoice_tags():
    # emotion2vec+ の分類名 (小文字) も、SenseVoice のタグ (大文字) と同じ読み方で 予測の感情になる
    for name in ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"):
        assert labels.predicted_label(name) == name and labels.predicted_label(name.upper()) == name
    assert labels.predicted_label("OTHER") == labels.EMO_OTHER and labels.predicted_label("other") == labels.EMO_OTHER
    assert labels.predicted_label("UNKNOWN") == labels.UNKNOWN and labels.predicted_label("<unk>") == labels.UNKNOWN
    # SenseVoice の読み方は変わらない (OTHER という SenseVoice のタグは無い)
    assert "OTHER" not in labels.SV_TAG_TO_LABEL and labels.SV_TAG_TO_LABEL["HAPPY"] == "happy"


def test_other_is_a_separate_prediction_column_and_folds_to_neutral():
    assert labels.EMO_OTHER in labels.ALL_PRED_LABELS and labels.EMO_OTHER not in labels.PRED_LABELS      # 既存の SenseVoice の表の列は変わらない
    assert labels.EMO_OTHER != labels.OTHER                                                                # 3感情の見方の『その他』とは別
    assert labels.fold_to_pvt(labels.EMO_OTHER) == "neutral"
    assert labels.core3_view(labels.EMO_OTHER) == labels.OTHER
    for k in labels.ALL_PRED_LABELS:
        assert k in labels.JA


def test_canonical_e2v():
    assert labels.canonical_e2v("生气/angry") == "angry" and labels.canonical_e2v("开心/happy") == "happy"
    assert labels.canonical_e2v("中立/neutral") == "neutral" and labels.canonical_e2v("其他/other") == "other"
    assert labels.canonical_e2v("<unk>") == "unknown" and labels.canonical_e2v("happy") == "happy"
    with pytest.raises(ValueError, match="読めません"):
        labels.canonical_e2v("unuse1")


def test_four_and_six_classes_match_plan():
    assert labels.FOUR == ("neutral", "happy", "angry", "sad")        # PLAN §5.3: 平常・喜び・怒り・悲しみ
    assert set(labels.SIX) == set(labels.EMOTIONS) - {"neutral"}
