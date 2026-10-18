"""asrcheck.textnorm: 日本語の正規化 (NFKC・記号・長音・数字・フィラー) と英仏の単語分割"""
from __future__ import annotations

import pytest

from asrcheck.textnorm import NormOptions, normalize_ja, normalize_numbers, normalize_text, normalize_words

DEFAULT = NormOptions()


def ja(text: str, **kw) -> str:
    return normalize_ja(text, NormOptions(**kw))


def test_nfkc_width_and_case():
    assert ja("ＡＢＣ１２３") == "abc123"          # 全角英数 → 半角、小文字
    assert ja("ｶﾀｶﾅ") == "カタカナ"                 # 半角カナ → 全角カナ
    assert ja("㌔") == "キロ"                       # 互換文字


def test_punctuation_and_spaces_removed():
    assert ja("こんにちは、世界！ 「元気」ですか？\n はい。") == "こんにちは世界元気ですかはい"
    assert ja("a, b; c") == "abc"
    assert ja("★☆♪→") == ""                        # 記号 (S) も除く


def test_percent_becomes_word():
    assert ja("50%") == ja("50パーセント") == "50パーセント"
    assert ja("５０％") == "50パーセント"


@pytest.mark.parametrize("src", ["コーヒー", "コ－ヒ－", "ｺｰﾋｰ", "コ―ヒ―", "コ−ヒ−", "コ〜ヒ〜", "コーーヒーー"])
def test_long_vowel_marks_are_unified(src):
    assert ja(src) == "コーヒー"


def test_dash_after_non_kana_is_dropped():
    assert ja("2025-10-18") == "20251018"
    assert ja("A-B") == "ab"


def test_long_vowel_drop_option():
    assert ja("コンピューター", long_vowel="drop") == ja("コンピュータ", long_vowel="drop") == "コンピュタ"
    assert ja("コンピューター") != ja("コンピュータ")   # 既定は残す


def test_kana_fold():
    assert ja("カタカナ", kana_fold=True) == "かたかな"
    assert ja("コーヒー", kana_fold=True) == "こーひー"
    assert ja("カタカナ") == "カタカナ"


def test_fillers_option():
    assert ja("えーと、そうですね", fillers="drop") == "そうですね"
    assert ja("あのー、えっと、お願いします", fillers="drop") == "お願いします"
    assert ja("えーと、そうですね") == "えーとそうですね"      # 既定は残す
    assert ja("ええ、そうです", fillers="drop") == "ええそうです"  # 「ええ」(はい) は消さない


@pytest.mark.parametrize(
    "src,expected",
    [
        ("二千二十五年", "2025年"),
        ("2025年", "2025年"),
        ("二〇二五年", "2025年"),            # 単位なし → 桁をそのまま
        ("〇九〇", "090"),                    # 先頭の 0 を保つ
        ("三万五千円", "35000円"),
        ("一万二千三百四十五", "12345"),
        ("十月", "10月"),
        ("二十三日", "23日"),
        ("百", "100"),
        ("千円", "1000円"),
        ("万円", "10000円"),
        ("3,000万円", "30000000円"),
        ("3,000円", "3000円"),
        ("1.5万円", "15000円"),
        ("1万5千", "15000"),
        ("一億二千万", "120000000"),
        ("1.5", "1点5"),
        ("一点五", "1点5"),
    ],
)
def test_numbers_arabic(src, expected):
    assert normalize_numbers(src, "arabic") == expected


@pytest.mark.parametrize("a,b", [("二千二十五年", "2025年"), ("三千円", "3,000円"), ("１０月", "十月"), ("三万五千円", "3万5千円")])
def test_number_spellings_match_after_normalization(a, b):
    assert ja(a) == ja(b)


def test_numbers_kanji_mode():
    assert ja("2025年", numbers="kanji") == ja("二〇二五年", numbers="kanji") == "二〇二五年"
    assert ja("二千二十五年", numbers="kanji") == "二〇二五年"


def test_numbers_ignore_and_keep():
    assert ja("2025年3月に三千円", numbers="ignore") == "年月に円"
    assert ja("二千二十五年", numbers="keep") == "二千二十五年"     # 何もしない (NFKC のみ)
    assert ja("２０２５年", numbers="keep") == "2025年"


def test_lexical_kanji_numbers_are_consistent_on_both_sides():
    # 「一緒」「千代田」も同じ規則で変換されるが、両側に同じ規則なので一致は変わらない
    assert ja("一緒に千代田へ") == ja("一緒に千代田へ")


def test_raw_options_leave_numbers_and_long_vowels():
    raw = NormOptions(numbers="arabic", long_vowel="drop", fillers="drop", kana_fold=True).raw()
    assert (raw.numbers, raw.long_vowel, raw.fillers, raw.kana_fold) == ("keep", "keep", "keep", False)


def test_invalid_options_raise():
    for kw in ({"lang": "de"}, {"numbers": "x"}, {"fillers": "x"}, {"long_vowel": "x"}):
        with pytest.raises(ValueError):
            NormOptions(**kw)


def test_normalize_words_en_fr():
    assert normalize_words("Hello, World! It’s 3.5 o'clock.", NormOptions(lang="en")) == ["hello", "world", "it's", "3", "5", "o'clock"]
    assert normalize_words("L’homme peut-être là", NormOptions(lang="fr")) == ["l'homme", "peut", "être", "là"]
    assert normalize_words("Café", NormOptions(lang="fr", strip_accents=True)) == ["cafe"]
    assert normalize_words("'quoted' word", NormOptions(lang="en")) == ["quoted", "word"]   # 語頭・語末の ' は落とす


def test_normalize_text_dispatch():
    assert normalize_text("Hello, World", NormOptions(lang="en")) == "hello world"
    assert normalize_text("今日は、晴れ。", NormOptions()) == "今日は晴れ"


def test_describe_mentions_choices():
    d = NormOptions(numbers="ignore", fillers="drop").describe()
    assert "数字=無視" in d and "フィラー=消す" in d
