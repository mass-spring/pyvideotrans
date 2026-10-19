"""感情ラベルの名前と、SenseVoice の感情タグの読み替え。

内部で使う感情の名前 (キー) は、pyVideoTrans 側 (videotrans/util/emotion.py の LABELS) と同じ:
  neutral happy angry sad fearful disgusted surprised
これに『予測だけに出てくる』2つを足す:
  unknown  SenseVoice が <|EMO_UNKNOWN|> を出した (『自信がない』『判定しない』に近い)
  notag    出力に感情タグがひとつも無かった
  emo_other  emotion2vec+ が other (9分類のうち『その他』) を出した。SenseVoice は出さない

pyVideoTrans の map_sensevoice_tag との違い:
  pyVideoTrans は EMO_UNKNOWN・タグなし・未知のタグを『平常 (neutral、信頼度0)』に読み替える。
  この道具は『不明』『タグなし』を別の列として残す (平常と判定されたのか、判定できなかったのかを区別して数えるため)。
  『不明を平常に読み替えた場合』の割合も、まとめに併記する (fold_to_pvt)。
  対応表そのものは tests/test_emocheck_labels.py で videotrans/util/emotion.py と突き合わせている。
"""
from __future__ import annotations

import re

EMOTIONS = ("neutral", "happy", "angry", "sad", "fearful", "disgusted", "surprised")
UNKNOWN = "unknown"
NOTAG = "notag"
EMO_OTHER = "emo_other"     # emotion2vec+ の 'other'。labels.OTHER (3感情の見方の『その他』) とは別物
PRED_LABELS = EMOTIONS + (UNKNOWN, NOTAG)               # SenseVoice の予測 (これまでの表の列)
ALL_PRED_LABELS = EMOTIONS + (EMO_OTHER, UNKNOWN, NOTAG)  # emotion2vec+ の other も含めた予測

JA = {
    "neutral": "平常", "happy": "喜び", "angry": "怒り", "sad": "悲しみ",
    "fearful": "恐れ", "disgusted": "嫌悪", "surprised": "驚き",
    UNKNOWN: "不明（EMO_UNKNOWN）", NOTAG: "タグなし", EMO_OTHER: "その他（other）",
}

# 表に並べる順番 (PLAN の 平常・喜び・怒り・悲しみ を先頭に)
ORDER = ("neutral", "happy", "angry", "sad", "fearful", "disgusted", "surprised")

# PLAN §5.3 の 4 分類のうち、JVNV で測れる 3 感情
CORE3 = ("happy", "angry", "sad")
OTHER = "other"   # 3 分類の見方での『それ以外』
# PLAN §5.3 の 4 分類 (平常・喜び・怒り・悲しみ)。平常の音声 (FLEURS など) があれば測れる
FOUR = ("neutral", "happy", "angry", "sad")
# JVNV の 6 感情 (平常なし)
SIX = ("happy", "angry", "sad", "fearful", "disgusted", "surprised")

# SenseVoice のタグ名 (<|HAPPY|> の中身) → 感情。videotrans/util/emotion.py の SENSEVOICE_TAG_MAP と同じ対応
SV_TAG_TO_LABEL = {
    "NEUTRAL": "neutral", "HAPPY": "happy", "ANGRY": "angry", "SAD": "sad",
    "FEARFUL": "fearful", "DISGUSTED": "disgusted", "SURPRISED": "surprised",
}
SV_UNKNOWN_TAG = "EMO_UNKNOWN"
SV_EMOTION_TAGS = frozenset(SV_TAG_TO_LABEL) | {SV_UNKNOWN_TAG}
SV_TAG_RE = re.compile(r"<\|([^|]*)\|>")

# emotion2vec+ の 9 分類 (tokens.txt: 生气/angry 厌恶/disgusted 恐惧/fearful 开心/happy 中立/neutral 其他/other 难过/sad 吃惊/surprised <unk>)
E2V_CLASSES = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")
# タグ名 (大文字) → 予測。SenseVoice の SV_TAG_TO_LABEL に無いもの。'OTHER' は emotion2vec+ だけが出す
E2V_EXTRA_TAGS = {"OTHER": EMO_OTHER}

_ALIASES: dict[str, str] = {}


def _alias(label: str, *names: str) -> None:
    for n in names:
        _ALIASES[n.strip().casefold()] = label


_alias("neutral", "neutral", "normal", "calm", "平常", "平静", "平穏", "中立", "中性", "ニュートラル", "普通")
_alias("happy", "happy", "happiness", "joy", "joyful", "喜び", "喜", "嬉しい", "うれしい", "楽しい", "ハッピー")
_alias("angry", "angry", "anger", "怒り", "怒", "怒る", "アングリー")
_alias("sad", "sad", "sadness", "悲しみ", "悲", "悲しい", "かなしい")
_alias("fearful", "fearful", "fear", "afraid", "恐れ", "恐怖", "怖い", "恐")
_alias("disgusted", "disgusted", "disgust", "嫌悪", "嫌", "嫌い")
_alias("surprised", "surprised", "surprise", "驚き", "驚", "驚く", "びっくり")


def parse_label(value: str) -> str:
    """正解の感情の書き方 (英語・日本語・SenseVoice のタグ名) を内部の名前にする。読めなければ ValueError。"""
    key = (value or "").strip().casefold()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"感情の名前が読めません: '{value}'。使える書き方: "
        + "、".join(f"{JA[k]}({k})" for k in ORDER)
    )


def split_sensevoice_text(raw: str) -> tuple[str, str]:
    """'<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは' → ('こんにちは', 'HAPPY')。感情タグが無ければタグは ''。

    タグは大文字にそろえて返す。EMO_UNKNOWN もそのまま返す (asrcheck.runners.split_sensevoice_tags は捨てる)。
    pyVideoTrans の _extract_emotion_tag と同じく、最初に見つかった感情タグを使う。
    """
    tag = ""
    for name in SV_TAG_RE.findall(raw or ""):
        if name.upper() in SV_EMOTION_TAGS:
            tag = name.upper()
            break
    return SV_TAG_RE.sub("", raw or "").strip(), tag


def predicted_label(tag: str) -> str:
    """タグ名 → 予測の感情。'' はタグなし、EMO_UNKNOWN と知らないタグは不明。

    SenseVoice のタグ (HAPPY など) と、emotion2vec+ の分類名 (happy など。大文字小文字は問わない) の両方を読む。
    emotion2vec+ の 'other' は EMO_OTHER、'unknown' (<unk>) は不明。
    """
    t = (tag or "").strip().upper()
    if not t:
        return NOTAG
    return SV_TAG_TO_LABEL.get(t) or E2V_EXTRA_TAGS.get(t) or UNKNOWN


def fold_to_pvt(label: str) -> str:
    """pyVideoTrans (map_sensevoice_tag) と同じ読み替え: 不明・タグなしは平常にする。emotion2vec+ の other も同じ扱い (要検討)。"""
    return "neutral" if label in (UNKNOWN, NOTAG, EMO_OTHER) else label


def canonical_e2v(name: str) -> str:
    """emotion2vec+ の分類名 ('生气/angry' '<unk>' など) → 'angry' 'unknown' のような英語の小文字名。読めなければ ValueError。"""
    n = (name or "").strip().casefold()
    if n in ("<unk>", "unk", "unknown", "未知"):
        return "unknown"
    n = n.rsplit("/", 1)[-1].strip()
    if n not in E2V_CLASSES:
        raise ValueError(f"emotion2vec+ の分類名が読めません: '{name}'（期待: {', '.join(E2V_CLASSES)}）")
    return n


def core3_view(label: str) -> str:
    """3 感情 (喜び・怒り・悲しみ) の見方: それ以外はすべて『その他』にまとめる。"""
    return label if label in CORE3 else OTHER
