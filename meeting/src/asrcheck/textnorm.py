"""文字の整え方 (正規化)。CER / WER を測る前に、正解と認識結果の両方に同じ処理をかける。

目的は「表記ゆれ」(句読点・全角半角・数字の書き方など) を誤りとして数えないこと。
処理は正解側と認識結果側に **同じ関数を同じ設定で** 適用するので、どちらかに有利になることはない。

日本語 (lang="ja") の処理の順番:
 1. NFKC 正規化          全角英数→半角、半角カナ→全角カナ、「㌔」→「キロ」など互換文字の統一
 2. 小文字化              英字は大文字小文字を区別しない
 3. 「%」→「パーセント」   数字と単位の書き方 (50% / 50パーセント) をそろえる
 4. 数字の書き方         numbers: arabic (既定) / kanji / ignore / keep  → 下の説明
 5. 記号・空白の除去     句読点・括弧・記号・空白・改行をすべて取り除く (Unicode の P/S/Z/C 分類)
 6. 長音記号の統一       「ー」「−」「―」「～」など伸ばす棒を「ー」1つにそろえる (かなの直後のみ。連続は1つに)
 7. (任意) フィラー除去   fillers="drop" のとき「えーと」「あのー」などを消す
 8. (任意) 長音を消す     long_vowel="drop" のとき「ー」を全部消す (コンピューター/コンピュータ の違いを許す)
 9. (任意) かな統一       kana_fold=True のとき カタカナ→ひらがな

数字 (numbers) の選択肢:
 - arabic (既定): 漢数字を算用数字にそろえる。「二千二十五年」→「2025年」、「三万五千円」→「35000円」、
                   「3,000」→「3000」、「1.5万」→「15000」、「二〇二五」→「2025」(単位が無い並びは桁をそのまま並べる)
 - kanji       : 上と同じ計算をした後、各桁を漢数字 (〇一二…九) に直す。照合用の内部表現で、読みやすさのためではない
 - ignore      : 数字の並び (漢数字・算用数字・十百千万億兆) を両側から全部取り除く。数字の誤りは数えない
 - keep        : 数字には何もしない (NFKC だけ)
 漢数字を含む普通の語 (「一緒」「千代田」「万が一」) も同じ規則で変換されるが、正解側と認識結果側に同じ規則をかけるので
 一致/不一致の判定は変わらない (文字数だけが少し変わる)。

英語・フランス語 (lang="en"/"fr") は単語単位 (WER) で測るため normalize_words を使う:
 NFKC → 小文字 → 「’」を「'」に統一 → 記号を空白にする (単語内の「'」は残す。ハイフンは空白扱い) → 空白で分割。
 アクセント記号は残す (strip_accents=True で消せる)。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

NUMBER_MODES = ("arabic", "kanji", "ignore", "keep")
FILLER_MODES = ("keep", "drop")
LONG_VOWEL_MODES = ("keep", "drop")
LANGS = ("ja", "en", "fr")


@dataclass(frozen=True)
class NormOptions:
    lang: str = "ja"
    numbers: str = "arabic"
    long_vowel: str = "keep"
    fillers: str = "keep"
    kana_fold: bool = False
    strip_accents: bool = False  # en/fr のみ

    def __post_init__(self) -> None:
        if self.lang not in LANGS:
            raise ValueError(f"lang は {', '.join(LANGS)} のどれかにしてください: {self.lang}")
        if self.numbers not in NUMBER_MODES:
            raise ValueError(f"numbers は {', '.join(NUMBER_MODES)} のどれかにしてください: {self.numbers}")
        if self.fillers not in FILLER_MODES:
            raise ValueError(f"fillers は {', '.join(FILLER_MODES)} のどれかにしてください: {self.fillers}")
        if self.long_vowel not in LONG_VOWEL_MODES:
            raise ValueError(f"long_vowel は {', '.join(LONG_VOWEL_MODES)} のどれかにしてください: {self.long_vowel}")

    def describe(self) -> str:
        if self.lang != "ja":
            return f"言語={self.lang} (単語単位。NFKC・小文字化・記号除去" + (("・アクセント除去") if self.strip_accents else "") + ")"
        parts = [
            "NFKC・小文字化・記号と空白の除去・長音記号の統一",
            {"arabic": "数字=算用数字にそろえる", "kanji": "数字=漢数字にそろえる", "ignore": "数字=無視(取り除く)", "keep": "数字=そのまま"}[self.numbers],
            "長音「ー」=" + ("消す" if self.long_vowel == "drop" else "残す"),
            "フィラー=" + ("消す" if self.fillers == "drop" else "残す"),
            "カタカナ→ひらがな=" + ("する" if self.kana_fold else "しない"),
        ]
        return " / ".join(parts)

    def raw(self) -> "NormOptions":
        """比較用の『ほぼ正規化なし』: NFKC・小文字・記号と空白だけ。数字・長音・フィラーは手を付けない。"""
        return replace(self, numbers="keep", long_vowel="keep", fillers="keep", kana_fold=False)


# ---------------------------------------------------------------------------
# 数字
# ---------------------------------------------------------------------------
_KDIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10**4, "億": 10**8, "兆": 10**12}
_NUM_CHARS = "0-9〇零一二三四五六七八九十百千万億兆"
_NUM_RUN = re.compile(f"[{_NUM_CHARS}]+")
_TO_KANJI = str.maketrans("0123456789", "〇一二三四五六七八九")
_DECIMAL_UNIT = re.compile(r"(\d+)\.(\d+)([万億兆])")
_THOUSAND_COMMA = re.compile(r"(?<=\d),(?=\d)")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")


def _run_to_arabic(run: str) -> str:
    """数字の並び (算用数字と漢数字の混在可) を算用数字の文字列にする。"""
    if not any(c in _SMALL or c in _BIG for c in run):
        # 単位が無い: 桁をそのまま並べる (二〇二五 → 2025, 〇九〇 → 090)
        return "".join(str(_KDIGIT[c]) if c in _KDIGIT else c for c in run)
    total = 0
    section = 0
    current: int | None = None
    i = 0
    while i < len(run):
        c = run[i]
        if c.isdigit() and c.isascii():
            j = i
            while j < len(run) and run[j].isascii() and run[j].isdigit():
                j += 1
            n = int(run[i:j])
            current = n if current is None else current * (10 ** (j - i)) + n
            i = j
            continue
        if c in _KDIGIT:
            current = _KDIGIT[c] if current is None else current * 10 + _KDIGIT[c]
        elif c in _SMALL:
            section += (current if current is not None else 1) * _SMALL[c]
            current = None
        else:  # 万 億 兆
            section += current if current is not None else 0
            if section == 0:
                section = 1
            total += section * _BIG[c]
            section = 0
            current = None
        i += 1
    total += section + (current or 0)
    return str(total)


def normalize_numbers(text: str, mode: str) -> str:
    if mode == "keep":
        return text
    if mode == "ignore":
        return _NUM_RUN.sub("", text)
    text = _THOUSAND_COMMA.sub("", text)
    text = _DECIMAL_UNIT.sub(lambda m: str(int(round(float(f"{m.group(1)}.{m.group(2)}") * _BIG[m.group(3)]))), text)
    text = _DECIMAL_POINT.sub("点", text)
    text = _NUM_RUN.sub(lambda m: _run_to_arabic(m.group(0)), text)
    if mode == "kanji":
        text = text.translate(_TO_KANJI)
    return text


# ---------------------------------------------------------------------------
# 日本語
# ---------------------------------------------------------------------------
_DASH_LIKE = set("-‐‑‒–—―−─━〜~゠ー")
_FILLER_RE = re.compile("えーっと|えーと|ええと|えっと|えー|あー|うーん|んー|あのー|そのー|まー|ふーん|はー|ほー")


def _is_kana(c: str) -> bool:
    return "ぁ" <= c <= "ゟ" or "ァ" <= c <= "ヺ" or c == "ー"


def _strip_symbols_and_unify_dashes(text: str) -> str:
    out: list[str] = []
    for c in text:
        if c in _DASH_LIKE:
            if out and _is_kana(out[-1]):
                if out[-1] != "ー":
                    out.append("ー")
            continue
        cat = unicodedata.category(c)
        if cat[0] in "PSZC":
            continue
        out.append(c)
    return "".join(out)


def normalize_ja(text: str, opts: NormOptions | None = None) -> str:
    opts = opts or NormOptions()
    t = unicodedata.normalize("NFKC", text or "").casefold()
    t = t.replace("%", "パーセント")
    t = normalize_numbers(t, opts.numbers)
    t = _strip_symbols_and_unify_dashes(t)
    if opts.fillers == "drop":
        t = _FILLER_RE.sub("", t)
    if opts.long_vowel == "drop":
        t = t.replace("ー", "")
    if opts.kana_fold:
        t = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in t)
    return t


# ---------------------------------------------------------------------------
# 英語・フランス語 (単語単位)
# ---------------------------------------------------------------------------
_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})


def normalize_words(text: str, opts: NormOptions | None = None) -> list[str]:
    opts = opts or NormOptions(lang="en")
    t = unicodedata.normalize("NFKC", text or "").casefold().translate(_APOS)
    if opts.strip_accents:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    chars: list[str] = []
    for i, c in enumerate(t):
        if c.isalnum() or unicodedata.category(c).startswith("M"):
            chars.append(c)
        elif c == "'" and 0 < i < len(t) - 1 and t[i - 1].isalnum() and t[i + 1].isalnum():
            chars.append(c)
        else:
            chars.append(" ")
    return "".join(chars).split()


def normalize_text(text: str, opts: NormOptions) -> str:
    """CER 用の文字列。ja は normalize_ja、en/fr は単語を空白でつないだ文字列。"""
    if opts.lang == "ja":
        return normalize_ja(text, opts)
    return " ".join(normalize_words(text, opts))
