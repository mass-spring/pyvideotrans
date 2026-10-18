"""編集距離 (レーベンシュタイン距離) と整列。純 Python + numpy。

- 距離だけが欲しいとき: distance(a, b)  … メモリは O(長さ)
- どこが置換・脱落・挿入かも欲しいとき: align(a, b)  … Hirschberg 法で長い文字列 (1時間の会議 ≒ 2万文字) でも省メモリ

記号の意味 (正解 ref を hyp に直すのに必要な操作):
  "="  一致    "S"  置換 (ref の1文字が別の文字になった)
  "D"  脱落 (ref にあるが hyp に無い)    "I"  挿入 (hyp にだけある)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Op = tuple[str, int, int]  # (記号, ref の位置 or -1, hyp の位置 or -1)

_SMALL_CELLS = 1_000_000  # これ以下の大きさなら表を全部作って逆にたどる


def to_ids(a: Sequence, b: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """文字や単語の並びを整数の配列にする (2つの並びで共通の番号を使う)。"""
    table: dict = {}
    ia = np.fromiter((table.setdefault(x, len(table)) for x in a), dtype=np.int64, count=len(a))
    ib = np.fromiter((table.setdefault(x, len(table)) for x in b), dtype=np.int64, count=len(b))
    return ia, ib


def _last_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a と b の各接頭辞との編集距離 (表の最後の行)。長さ len(b)+1。"""
    m = len(b)
    prev = np.arange(m + 1, dtype=np.int32)
    j = np.arange(m + 1, dtype=np.int32)
    for x in a:
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = prev[0] + 1
        cur[1:] = np.minimum(prev[:-1] + (b != x), prev[1:] + 1)
        # 挿入 (同じ行で左から右へ +1) を累積最小値で一度に反映する
        prev = np.minimum.accumulate(cur - j) + j
    return prev


def distance(a: Sequence, b: Sequence) -> int:
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    ia, ib = to_ids(a, b)
    return int(_last_row(ia, ib)[-1])


def _full_align(a: np.ndarray, b: np.ndarray, a0: int, b0: int) -> list[Op]:
    n, m = len(a), len(b)
    d = np.empty((n + 1, m + 1), dtype=np.int32)
    d[0] = np.arange(m + 1)
    j = np.arange(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        cur[1:] = np.minimum(d[i - 1, :-1] + (b != a[i - 1]), d[i - 1, 1:] + 1)
        d[i] = np.minimum.accumulate(cur - j) + j
    ops: list[Op] = []
    i, k = n, m
    while i > 0 or k > 0:
        if i > 0 and k > 0 and d[i, k] == d[i - 1, k - 1] + (a[i - 1] != b[k - 1]):
            ops.append(("=" if a[i - 1] == b[k - 1] else "S", a0 + i - 1, b0 + k - 1))
            i -= 1
            k -= 1
        elif i > 0 and d[i, k] == d[i - 1, k] + 1:
            ops.append(("D", a0 + i - 1, -1))
            i -= 1
        else:
            ops.append(("I", -1, b0 + k - 1))
            k -= 1
    ops.reverse()
    return ops


def _hirschberg(a: np.ndarray, b: np.ndarray, a0: int, b0: int, out: list[Op]) -> None:
    n, m = len(a), len(b)
    if n == 0:
        out.extend(("I", -1, b0 + k) for k in range(m))
        return
    if m == 0:
        out.extend(("D", a0 + i, -1) for i in range(n))
        return
    if n == 1 or (n + 1) * (m + 1) <= _SMALL_CELLS:
        out.extend(_full_align(a, b, a0, b0))
        return
    mid = n // 2
    fwd = _last_row(a[:mid], b)
    bwd = _last_row(a[mid:][::-1], b[::-1])[::-1]
    k = int(np.argmin(fwd + bwd))
    _hirschberg(a[:mid], b[:k], a0, b0, out)
    _hirschberg(a[mid:], b[k:], a0 + mid, b0 + k, out)


def align(a: Sequence, b: Sequence) -> list[Op]:
    """a (正解) を b (認識結果) に直す操作列。"""
    ia, ib = to_ids(a, b)
    out: list[Op] = []
    _hirschberg(ia, ib, 0, 0, out)
    return out


@dataclass
class ErrorCounts:
    n_ref: int = 0
    hit: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        """誤り率 = (置換+脱落+挿入) ÷ 正解の長さ。正解が空なら nan。"""
        return self.errors / self.n_ref if self.n_ref else float("nan")

    def __iadd__(self, other: "ErrorCounts") -> "ErrorCounts":
        self.n_ref += other.n_ref
        self.hit += other.hit
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        return self


def count_ops(ops: Sequence[Op]) -> ErrorCounts:
    c = ErrorCounts()
    for op, _, _ in ops:
        if op == "=":
            c.hit += 1
        elif op == "S":
            c.sub += 1
        elif op == "D":
            c.dele += 1
        else:
            c.ins += 1
    c.n_ref = c.hit + c.sub + c.dele
    return c
