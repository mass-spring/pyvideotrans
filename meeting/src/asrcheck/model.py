"""データの型。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utt:
    """1つの発言 (または1つの字幕行)。"""

    start: float | None
    end: float | None
    speaker: str
    text: str
    rec: str = ""        # 録音の名前 (複数の録音をまとめて評価するときの区別)
    row: int = 0         # 元ファイルでの行番号 (メッセージ用)

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    @property
    def has_time(self) -> bool:
        return self.start is not None and self.end is not None and self.end > self.start


def sort_by_time(utts: list[Utt]) -> list[Utt]:
    """開始時刻順 (同時刻は終了時刻順、それも同じなら元の順)。時刻が1つでも無ければ元の順のまま。"""
    if any(u.start is None for u in utts):
        return list(utts)
    return sorted(utts, key=lambda u: (u.start, u.end if u.end is not None else u.start))


def rec_key(name: str) -> str:
    """録音の名前の照合用キー (拡張子と大文字小文字を無視)。"""
    return Path(name).stem.casefold() if name else ""
