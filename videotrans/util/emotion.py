# -*- coding: utf-8 -*-
"""
Per-line emotion data (EMOTION module, part 1).

字幕(SRT)は本文と時刻しか持たないため、行ごとの感情は別ファイル(サイドカー)に持つ:
    {cache_folder}/emotion.json      最終の原文字幕1行につき1件。target_dir にもコピーする(speaker.json と同じ)
    {cache_folder}/emotion_raw.json  SenseVoice が出した感情タグの控え(ASR のセグメント単位。タグ除去前に保存)

行との対応は「開始時刻(ms)の近さ」で取る(行番号だけに頼らない。字幕を編集すると行番号はずれる)。
このモジュールは標準ライブラリだけを使う(ASR の別プロセスからも import されるため軽く保つ)。
"""
import bisect
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EMOTION_FILE = 'emotion.json'
EMOTION_RAW_FILE = 'emotion_raw.json'

# 感情ラベル(これ以外は neutral に丸める)
LABELS = ('neutral', 'happy', 'angry', 'sad', 'fearful', 'disgusted', 'surprised')

# 開始時刻の対応付けで許す差(ms)
MATCH_TOLERANCE_MS = 300
# 翻訳の指示文に載せる最低の信頼度
HINT_MIN_CONFIDENCE = 0.5

# SenseVoice の感情タグ(<|HAPPY|> の中身) -> ラベル。EMO_UNKNOWN と未知のタグは neutral(信頼度0)
SENSEVOICE_TAG_MAP = {
    'NEUTRAL': 'neutral',
    'HAPPY': 'happy',
    'ANGRY': 'angry',
    'SAD': 'sad',
    'FEARFUL': 'fearful',
    'DISGUSTED': 'disgusted',
    'SURPRISED': 'surprised',
}
SENSEVOICE_UNKNOWN_TAG = 'EMO_UNKNOWN'
# 取り込み側が感情タグとして扱う名前(言語タグ <|ja|>、イベントタグ <|Speech|> などと区別する)
SENSEVOICE_EMOTION_TAGS = frozenset(SENSEVOICE_TAG_MAP) | {SENSEVOICE_UNKNOWN_TAG}

# 暫定値: SenseVoice は確率を出さないため固定値を置く。音の強さ(音響)から求める強さは後で置き換える
SENSEVOICE_INTENSITY = 0.5
SENSEVOICE_CONFIDENCE = 0.6

SOURCE_SENSEVOICE = 'sensevoice'
SOURCE_NONE = 'none'  # 対応する検出結果が無かった行(neutral / 信頼度0)


@dataclass
class EmotionEntry:
    start_time: int = 0
    end_time: int = 0
    label: str = 'neutral'
    intensity: float = 0.0
    confidence: float = 0.0
    source: str = SOURCE_NONE

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'EmotionEntry':
        label = str(d.get('label', 'neutral')).strip().lower()
        confidence = _to_float(d.get('confidence'), 0.0)
        if label not in LABELS:
            label, confidence = 'neutral', 0.0
        return cls(
            start_time=int(_to_float(d.get('start_time'), 0)),
            end_time=int(_to_float(d.get('end_time'), 0)),
            label=label,
            intensity=_clip01(_to_float(d.get('intensity'), 0.0)),
            confidence=_clip01(confidence),
            source=str(d.get('source') or SOURCE_NONE),
        )


def _to_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _as_entry(e) -> EmotionEntry:
    return e if isinstance(e, EmotionEntry) else EmotionEntry.from_dict(e)


# ---------------------------------------------------------------- SenseVoice

def map_sensevoice_tag(tag: Optional[str]) -> Tuple[str, float, float]:
    """SenseVoice の感情タグ名('HAPPY' など。<| |> は除く) -> (label, intensity, confidence)

    タグ無し・EMO_UNKNOWN・未知のタグは ('neutral', 0, 0)。NEUTRAL は ('neutral', 0, 0.6)。
    強さと信頼度は暫定の固定値(SENSEVOICE_INTENSITY / SENSEVOICE_CONFIDENCE)。
    """
    label = SENSEVOICE_TAG_MAP.get((tag or '').strip().upper())
    if label is None:
        return 'neutral', 0.0, 0.0
    if label == 'neutral':
        return 'neutral', 0.0, SENSEVOICE_CONFIDENCE
    return label, SENSEVOICE_INTENSITY, SENSEVOICE_CONFIDENCE


def save_raw_capture(cache_folder: str, items: Sequence[Dict]) -> Optional[str]:
    """ASR(別プロセス)が控えた [{start_time, end_time, tag}] を emotion_raw.json に保存する"""
    if not cache_folder or not items:
        return None
    p = Path(cache_folder) / EMOTION_RAW_FILE
    p.write_text(json.dumps(list(items), ensure_ascii=False), encoding='utf-8')
    return p.as_posix()


def load_raw_capture(cache_folder: str) -> List[Dict]:
    p = Path(cache_folder) / EMOTION_RAW_FILE
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    return [d for d in data if isinstance(d, dict) and 'start_time' in d] if isinstance(data, list) else []


# ------------------------------------------------------------ load / save

def load_emotion(path: str) -> List[EmotionEntry]:
    """emotion.json を読む。無い・壊れている場合は空リスト。開始時刻順に並べて返す"""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    entries = [EmotionEntry.from_dict(d) for d in data if isinstance(d, dict)]
    entries.sort(key=lambda e: e.start_time)
    return entries


def save_emotion(entries: Iterable, cache_folder: str, target_dir: Optional[str] = None) -> str:
    """{cache_folder}/emotion.json に保存し、target_dir があればコピーする(speaker.json と同じ扱い)"""
    p = Path(cache_folder) / EMOTION_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [_as_entry(e).to_dict() for e in entries]
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    if target_dir and Path(target_dir).is_dir():
        shutil.copy2(p, Path(target_dir) / EMOTION_FILE)
    return p.as_posix()


# ----------------------------------------------------------- match by time

class EmotionIndex:
    """開始時刻(ms)で最も近いエントリを返す。差が tolerance_ms を超えたら None。同じ差なら早い方"""

    def __init__(self, entries: Iterable, tolerance_ms: int = MATCH_TOLERANCE_MS):
        self.entries: List[EmotionEntry] = sorted((_as_entry(e) for e in (entries or [])), key=lambda e: e.start_time)
        self._starts = [e.start_time for e in self.entries]
        self.tolerance_ms = tolerance_ms

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:  # ログに全件を出さない(translator の {self=} 出力用)
        return f'EmotionIndex(n={len(self.entries)}, tolerance_ms={self.tolerance_ms})'

    def find(self, start_time) -> Optional[EmotionEntry]:
        if not self.entries:
            return None
        t = _to_float(start_time, 0)
        i = bisect.bisect_left(self._starts, t)
        best, best_diff = None, None
        for j in (i - 1, i):
            if 0 <= j < len(self.entries):
                diff = abs(self._starts[j] - t)
                if best_diff is None or diff < best_diff:
                    best, best_diff = self.entries[j], diff
        return best if best_diff is not None and best_diff <= self.tolerance_ms else None


def covers_lines(entries: Iterable, lines: Sequence, tolerance_ms: int = MATCH_TOLERANCE_MS) -> bool:
    """lines(start_time を持つ字幕行)の全てに entries の対応が付くか"""
    idx = EmotionIndex(entries, tolerance_ms)
    return bool(lines) and all(idx.find(ln['start_time']) is not None for ln in lines)


def is_aligned(entries: Iterable, lines: Sequence, tolerance_ms: int = MATCH_TOLERANCE_MS) -> bool:
    """entries が lines と1対1に対応しているか(件数が同じで、順に開始時刻が tolerance_ms 以内)。

    covers_lines より厳しい: 字幕の統合・削除で余った古いエントリがあると False になる。
    """
    es = sorted((_as_entry(e) for e in entries), key=lambda e: e.start_time)
    ls = sorted(lines, key=lambda ln: ln['start_time'])
    return bool(ls) and len(es) == len(ls) and all(abs(e.start_time - ln['start_time']) <= tolerance_ms for e, ln in zip(es, ls))


def build_entries_from_sensevoice(lines: Sequence, raw_items: Sequence[Dict],
                                  tolerance_ms: int = MATCH_TOLERANCE_MS) -> List[EmotionEntry]:
    """最終字幕 lines の各行について、開始時刻が近い SenseVoice の控え(raw_items)からエントリを作る。

    控えが見つからない行は neutral / 強さ0 / 信頼度0 / source='none'。
    戻り値は lines と同じ順・同じ件数(時刻は字幕行のもの)。
    """
    raw_idx = {}
    starts = []
    for d in raw_items or []:
        s = int(_to_float(d.get('start_time'), 0))
        raw_idx[s] = d
        starts.append(s)
    starts.sort()

    out: List[EmotionEntry] = []
    for ln in lines:
        start, end = int(ln['start_time']), int(ln['end_time'])
        raw = None
        if starts:
            i = bisect.bisect_left(starts, start)
            cands = [starts[j] for j in (i - 1, i) if 0 <= j < len(starts)]
            near = min(cands, key=lambda s: abs(s - start))
            if abs(near - start) <= tolerance_ms:
                raw = raw_idx[near]
        if raw is None:
            out.append(EmotionEntry(start, end, 'neutral', 0.0, 0.0, SOURCE_NONE))
            continue
        label, intensity, confidence = map_sensevoice_tag(raw.get('tag'))
        out.append(EmotionEntry(start, end, label, intensity, confidence, SOURCE_SENSEVOICE))
    return out


# ------------------------------------------------------------ prompt hints

def select_hint_rows(keyed_starts: Iterable[Tuple[int, int]], entries: Iterable,
                     min_confidence: float = HINT_MIN_CONFIDENCE,
                     tolerance_ms: int = MATCH_TOLERANCE_MS) -> List[Tuple[int, str, float]]:
    """翻訳1バッチ分の (キー, 開始時刻) から、指示文に載せる行 (キー, ラベル, 強さ) を選ぶ。

    キーは SRT モードなら SRT の番号、テキストモードならバッチ内の位置(1始まり)。
    neutral の行と、信頼度が min_confidence 未満の行は載せない。
    """
    idx = entries if isinstance(entries, EmotionIndex) else EmotionIndex(entries, tolerance_ms)
    rows = []
    for key, start in keyed_starts:
        e = idx.find(start)
        if e is None or e.label == 'neutral' or e.confidence < min_confidence:
            continue
        rows.append((key, e.label, e.intensity))
    return rows


def format_hint_table(rows: Sequence[Tuple[int, str, float]]) -> str:
    """選んだ行を Markdown の表にする(行が無ければ空文字)"""
    if not rows:
        return ''
    lines = ['| Line | Emotion | Intensity |', '| --- | --- | --- |']
    lines += [f'| {key} | {label} | {intensity:.1f} |' for key, label, intensity in rows]
    return '\n'.join(lines)
