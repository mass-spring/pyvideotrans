"""朗読台本まわりの文字処理: モーラの数え方、読む長さの見積もり、文字列の似かた。

どれも「台本の文と録音の区間を結びつける」ための目安であり、言語学的に厳密なものではない。
"""
from __future__ import annotations

import difflib
import re
import unicodedata

# ひらがなに直したあと、直前の仮名とくっついて 1 モーラになる小書きの仮名
_COMBINING_SMALL = set("ゃゅょゎぁぃぅぇぉ")
_PUNCT_RE = re.compile(r"[\s、。，．,.!！?？・「」『』（）()\[\]【】〜～…\-‐‒–—_\"'’“”]")

# 読む速さの目安 (台本の「読む長さ」の見積もりだけに使う。録音の区間との対応づけで相対的に使う)
JA_MORA_PER_SEC = 7.5
LATIN_LETTERS_PER_SEC = 13.5
ZH_CHARS_PER_SEC = 4.3
PAUSE_BETWEEN_SENTENCES_SEC = 0.8


def norm_lang(lang: str | None) -> str:
    """'ja-JP' / 'ja_JP' / 'JA' → 'ja'"""
    return re.split(r"[-_]", str(lang or "").strip().lower())[0]


def to_hiragana(text: str) -> str:
    """カタカナをひらがなにする (長音符 ー はそのまま)。"""
    out = []
    for ch in text:
        o = ord(ch)
        out.append(chr(o - 0x60) if 0x30A1 <= o <= 0x30F6 else ch)
    return "".join(out)


def split_mora(kana: str) -> list[str]:
    """かな文字列 (ひらがな・カタカナ・ー) をモーラに分ける。拗音 (きゃ)・ふぁ などは 1 モーラ、っ・ん・ー は 1 モーラ。"""
    s = to_hiragana(_PUNCT_RE.sub("", kana))
    moras: list[str] = []
    for ch in s:
        if ch in _COMBINING_SMALL and moras:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


def count_mora(kana: str) -> int:
    return len(split_mora(kana))


def latin_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ord(ch) < 0x2E80)


def han_chars(text: str) -> int:
    return sum(1 for ch in text if "一" <= ch <= "鿿")


def estimate_seconds(text: str, lang: str, yomi: str = "") -> float:
    """その文を普通の速さで読んだときの長さ (秒) の目安。"""
    lang = norm_lang(lang)
    if lang == "ja":
        if yomi:
            return count_mora(yomi) / JA_MORA_PER_SEC
        kana = sum(1 for ch in text if "぀" <= ch <= "ヿ")
        kanji = han_chars(text)
        other = sum(1 for ch in text if ch.isalnum() and not ("぀" <= ch <= "ヿ") and not ("一" <= ch <= "鿿"))
        return (kana + 2.2 * kanji + other) / JA_MORA_PER_SEC
    if lang == "zh":
        return max(han_chars(text), 1) / ZH_CHARS_PER_SEC
    return max(latin_letters(text), 1) / LATIN_LETTERS_PER_SEC


def join_texts(texts: list[str], lang: str) -> str:
    """文を連結する。日本語・中国語は詰め、ほかはスペース区切り。"""
    parts = [t.strip() for t in texts if t and t.strip()]
    return ("" if norm_lang(lang) in ("ja", "zh") else " ").join(parts)


# ---- 文字列の似かた (ASR の結果と台本を比べる) --------------------------------------
_DIGITS = "〇一二三四五六七八九"


def arabic_to_kanji(n: int) -> str:
    """12 → 十二、345 → 三百四十五 (一万未満は位取り、それ以上は 1 桁ずつ)。比較用の近似。"""
    if n < 0 or n >= 10000:
        return "".join(_DIGITS[int(c)] for c in str(n))
    if n == 0:
        return "零"
    out = ""
    for unit, name in ((1000, "千"), (100, "百"), (10, "十")):
        d, n = divmod(n, unit)
        if d:
            out += ("" if d == 1 else _DIGITS[d]) + name
    if n:
        out += _DIGITS[n]
    return out


def normalize_for_match(text: str, lang: str) -> str:
    """ASR の結果と台本を比べるための正規化。記号・空白・大文字小文字・かな種・数字の書き方の違いをならす。"""
    lang = norm_lang(lang)
    s = unicodedata.normalize("NFKC", text or "").lower()
    if lang in ("ja", "zh"):
        s = re.sub(r"\d+", lambda m: arabic_to_kanji(int(m.group())) if len(m.group()) < 5 else m.group(), s)
        s = to_hiragana(s)
    else:
        s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return "".join(ch for ch in s if ch.isalnum())


def text_similarity(a: str, b: str, lang: str) -> float:
    """0〜1。1 に近いほど同じ文章。ASR の誤認識を許す緩い比較。"""
    na, nb = normalize_for_match(a, lang), normalize_for_match(b, lang)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def containment(part: str, whole: str, lang: str) -> float:
    """part の文字のうち、whole の中に (順番どおりに) 見つかる割合。part が whole の一部なら 1 に近い。"""
    npart, nwhole = normalize_for_match(part, lang), normalize_for_match(whole, lang)
    if not npart:
        return 1.0
    sm = difflib.SequenceMatcher(None, npart, nwhole, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(npart)
