"""朗読台本 (meeting/enroll/script_ja.json / script_ja.md) と、文字処理の部品のテスト。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voiceprofile.script import ScriptError, load_script, render_consent, total_est_sec
from voiceprofile.text_utils import (
    arabic_to_kanji,
    count_mora,
    estimate_seconds,
    norm_lang,
    split_mora,
    text_similarity,
)

ENROLL = Path(__file__).resolve().parents[2] / "enroll"
SCRIPT_JSON = ENROLL / "script_ja.json"
SCRIPT_MD = ENROLL / "script_ja.md"

# 日本語の基本のモーラ (ぢ・づ は じ・ず と同じ音なので数えない。ゐ・ゑ・ゔ は使わない)
BASIC_MORAS = """あ い う え お か き く け こ さ し す せ そ た ち つ て と な に ぬ ね の は ひ ふ へ ほ ま み む め も や ゆ よ ら り る れ ろ わ を ん
が ぎ ぐ げ ご ざ じ ず ぜ ぞ だ で ど ば び ぶ べ ぼ ぱ ぴ ぷ ぺ ぽ
きゃ きゅ きょ ぎゃ ぎゅ ぎょ しゃ しゅ しょ じゃ じゅ じょ ちゃ ちゅ ちょ にゃ にゅ にょ ひゃ ひゅ ひょ びゃ びゅ びょ ぴゃ ぴゅ ぴょ みゃ みゅ みょ りゃ りゅ りょ
っ ー""".split()
# 外来語で使う拡張のモーラ (全部は必須でないが、できるだけ入れる)
EXTENDED_MORAS = "ふぁ ふぃ ふぇ ふぉ てぃ でぃ うぃ うぇ しぇ じぇ ちぇ".split()


@pytest.fixture(scope="module")
def script():
    return load_script(SCRIPT_JSON)


def test_count_mora_basics():
    assert count_mora("きゃっかん") == 4          # きゃ・っ・か・ん
    assert count_mora("コンピュータ") == 5         # コ・ン・ピュ・ー・タ
    assert split_mora("ファイル") == ["ふぁ", "い", "る"]
    assert count_mora("みなさん、おはようございます。") == 13
    assert count_mora("") == 0


def test_estimate_seconds_by_language():
    assert estimate_seconds("", "ja", yomi="あいうえお" * 15) == pytest.approx(10.0)         # 75 モーラ / 7.5
    assert estimate_seconds("Good morning everyone", "en") == pytest.approx(len("Goodmorningeveryone") / 13.5)
    assert estimate_seconds("大家早上好", "zh") == pytest.approx(5 / 4.3)
    assert norm_lang("ja-JP") == "ja" and norm_lang("FR_fr") == "fr" and norm_lang(None) == ""


def test_text_similarity_tolerates_digits_kana_and_case():
    assert text_similarity("3月12日の衝突試験", "三月十二日の衝突試験", "ja") == pytest.approx(1.0)
    assert text_similarity("ニュートン", "にゅーとん", "ja") == pytest.approx(1.0)
    assert text_similarity("Commençons par les résultats", "commencons par les resultats", "fr") == pytest.approx(1.0)
    assert text_similarity("これは違う文章です", "全く別の内容を話しています", "ja") < 0.4
    assert text_similarity("", "あ", "ja") == 0.0
    assert arabic_to_kanji(345) == "三百四十五" and arabic_to_kanji(2026) == "二千二十六"


def test_script_json_structure(script):
    ja = script.sentences("ja")
    assert 15 <= len(ja) <= 20
    assert [s.id for s in ja] == [f"ja{i:02d}" for i in range(1, len(ja) + 1)]
    ids = [s.id for s in script.consent] + [s.id for lang in script.sets for s in script.sets[lang]]
    assert len(ids) == len(set(ids))
    assert set(script.sets) == {"ja", "en", "fr", "zh"}
    assert all(s.yomi for s in ja)                 # 日本語は全部に読み仮名 (モーラ数の計算に使う)
    assert all(s.role == "consent" for s in script.consent) and len(script.consent) == 3


def test_script_covers_all_basic_moras_and_total_length_about_90_seconds(script):
    seen = set()
    total = 0
    for s in script.sentences("ja"):
        ms = split_mora(s.yomi)
        total += len(ms)
        seen.update(ms)
    missing = [m for m in BASIC_MORAS if m not in seen]
    assert missing == [], f"台本に出てこないモーラ: {missing}"
    ext_missing = [m for m in EXTENDED_MORAS if m not in seen]
    assert len(ext_missing) <= 1, f"外来語の拡張モーラが足りない: {ext_missing}"
    assert 450 <= total <= 650                      # 1 秒 7〜8 モーラで約 75〜85 秒の発話
    est = total_est_sec(script.sentences("ja"))
    assert 80 <= est <= 105, est                     # 文の間の無音を含めて「約 90 秒」


def test_script_has_numbers_dates_loanwords_and_technical_terms(script):
    text = "".join(s.text for s in script.sentences("ja"))
    for must in ("三月十二日", "午後三時四十五分", "八百五十万円", "六百ヘルツ", "二千ニュートン"):   # 数字・日付・時刻・金額・単位
        assert must in text
    for must in ("コンピュータ", "セーフティ", "フィードバック", "シミュレーション", "ボルト", "チャンネル"):   # 外来語
        assert must in text
    for must in ("加速度", "剛性", "荷重", "波形", "衝突試験"):                                     # 技術用語
        assert must in text


def test_foreign_language_sets_are_about_15_seconds(script):
    for lang in ("en", "fr", "zh"):
        sents = script.sentences(lang)
        sec = sum(s.est_sec for s in sents)
        assert 10 <= sec <= 20, (lang, sec)


def test_consent_template_has_placeholders_and_renders(script):
    template = script.render_consent()
    assert "○○" in template and "{" not in template
    filled = script.render_consent("山田太郎", "株式会社テスト", "2027年3月末")
    assert "山田太郎" in filled and "株式会社テスト" in filled and "2027年3月末" in filled and "○○" not in filled
    raw = " ".join(s.text for s in script.consent)
    for key in ("{name}", "{organization}", "{retention}"):
        assert key in raw
    assert "撤回" in filled and "合成音声" in filled
    assert render_consent("私は{name}です。", name="A") == "私はAです。"


def test_script_md_contains_every_sentence_of_the_json(script):
    """script_ja.md (人が読む用) と script_ja.json (ツールが読む用) が食い違っていないこと。"""
    md = SCRIPT_MD.read_text(encoding="utf-8")
    for lang, sents in script.sets.items():
        for s in sents:
            assert s.text in md, f"script_ja.md に {s.id} の文がありません: {s.text}"
            if s.yomi:
                assert s.yomi in md, f"script_ja.md に {s.id} の読みがありません"
    for s in script.consent:
        for fragment in [f for f in _split_placeholders(s.text) if len(f) >= 3]:
            assert fragment in md, f"script_ja.md に同意文の一部がありません: {fragment}"
    for word in ("48", "WAV", "ノイズ", "10〜20", "同意"):   # 録音の注意
        assert word in md


def _split_placeholders(text: str) -> list[str]:
    import re

    return [p for p in re.split(r"\{[a-z]+\}", text) if p]


def test_script_json_loads_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_script(tmp_path / "none.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScriptError):
        load_script(bad)
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps({"sets": {"ja": {"sentences": [{"id": "a", "text": "あ"}, {"id": "a", "text": "い"}]}}}), encoding="utf-8")
    with pytest.raises(ScriptError, match="重複"):
        load_script(dup)
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"sets": {"ja": {"sentences": [{"id": "a", "text": "あいう"}]}}}), encoding="utf-8")
    sc = load_script(ok)
    with pytest.raises(ScriptError, match="fr"):
        sc.sentences("fr")
    assert sc.sentences("ja")[0].est_sec > 0         # yomi がなくても長さを見積もる
