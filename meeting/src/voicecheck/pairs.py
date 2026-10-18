"""pairs.csv の読み込み。

列: speaker, ref, test (必須) / source, label, emotion_src, emotion_test (任意)
ref は ';' 区切りで複数指定できる (登録用の手本クリップ)。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED = ["speaker", "ref", "test"]
OPTIONAL = ["source", "label", "emotion_src", "emotion_test"]

# Excel で日本語の見出しを付けても読めるようにする別名
ALIASES = {
    "話者": "speaker",
    "手本": "ref",
    "参照": "ref",
    "登録": "ref",
    "評価": "test",
    "テスト": "test",
    "評価音声": "test",
    "元音声": "source",
    "元発話": "source",
    "ラベル": "label",
    "条件": "label",
    "感情_元": "emotion_src",
    "感情（元）": "emotion_src",
    "感情_評価": "emotion_test",
    "感情（評価）": "emotion_test",
}


class PairsError(ValueError):
    """pairs.csv の形式が違うときのエラー (メッセージは利用者向けの日本語)。"""


@dataclass
class PairRow:
    line: int                   # CSV の行番号 (見出しが1行目)
    speaker: str
    refs: list[str]
    test: str
    source: str = ""
    label: str = ""
    emotion_src: str = ""
    emotion_test: str = ""
    ref_text: str = ""          # CSV に書かれたままの ref 欄
    problems: list[str] = field(default_factory=list)


def split_refs(text: str) -> list[str]:
    """'a.wav; b.wav' → ['a.wav', 'b.wav']。全角の；も区切りとみなす。"""
    return [p.strip().strip('"').strip() for p in text.replace("；", ";").split(";") if p.strip()]


def _decode(raw: bytes) -> str:
    """UTF-8 (BOM 付き可) を試し、だめなら Windows の Excel 標準 (cp932) として読む。"""
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_pairs(path: str | Path) -> list[PairRow]:
    p = Path(path)
    if not p.is_file():
        raise PairsError(f"pairs.csv が見つかりません: {p}")
    text = _decode(p.read_bytes())
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise PairsError(f"pairs.csv が空です: {p}") from None
    cols = [ALIASES.get(h.strip(), h.strip()).lower() for h in header]
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise PairsError(
            f"pairs.csv に必要な列がありません: {', '.join(missing)}\n"
            f"  見つかった列: {', '.join(h.strip() for h in header)}\n"
            f"  必須の列: {', '.join(REQUIRED)} (任意: {', '.join(OPTIONAL)})"
        )
    idx = {c: cols.index(c) for c in REQUIRED + OPTIONAL if c in cols}

    def cell(rec: list[str], name: str) -> str:
        i = idx.get(name)
        return rec[i].strip() if i is not None and i < len(rec) else ""

    base = p.parent
    rows: list[PairRow] = []
    for n, rec in enumerate(reader, start=2):
        if not any(c.strip() for c in rec):
            continue  # 空行
        row = PairRow(
            line=n,
            speaker=cell(rec, "speaker"),
            refs=[_resolve(r, base) for r in split_refs(cell(rec, "ref"))],
            test=_resolve(cell(rec, "test"), base),
            source=_resolve(cell(rec, "source"), base),
            label=cell(rec, "label"),
            emotion_src=cell(rec, "emotion_src"),
            emotion_test=cell(rec, "emotion_test"),
            ref_text=cell(rec, "ref"),
        )
        if not row.speaker:
            row.problems.append("speaker が空です")
        if not row.refs:
            row.problems.append("ref (手本) が空です")
        if not row.test:
            row.problems.append("test (評価音声) が空です")
        rows.append(row)
    return rows


def _resolve(path_text: str, base: Path) -> str:
    """相対パスは pairs.csv のあるフォルダ基準 (なければ実行場所基準) で解決する。空文字はそのまま。"""
    if not path_text:
        return ""
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return str(p)
    cand = base / p
    if cand.exists() or not p.exists():
        return str(cand)
    return str(p.resolve())


def emotion_match(a: str, b: str) -> bool | None:
    """両方書かれていれば一致(True)/不一致(False)、どちらか空なら None。"""
    a, b = a.strip().casefold(), b.strip().casefold()
    if not a or not b:
        return None
    return a == b
