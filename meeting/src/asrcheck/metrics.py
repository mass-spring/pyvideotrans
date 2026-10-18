"""文字・単語の誤り率: CER / WER / cpCER と、発言ごとの差分。

CER (文字誤り率) = (置換 + 脱落 + 挿入) ÷ 正解の文字数
  - 正解と認識結果を、それぞれ「開始時刻順に全発言をつないだ1本の文字列」にして、全体で整列 (編集距離) する。
    話者は区別しない (誰が言ったかは cpCER / 話者正解率 / DER で見る)。
  - つなぐ前に textnorm の規則で表記ゆれを整える。
WER (単語誤り率) は英語・フランス語用。単語を1単位として同じ計算をする。
cpCER (concatenated minimum-permutation CER)
  = 話者ごとに発言をつないだ文字列を作り、正解の話者と認識側の話者の組み合わせを「誤りが最小になる」1対1に選んで、
    全話者の誤りの合計 ÷ 正解の文字数。「誰が何を言ったか」を同時に評価する。
    話者の数が合わないときは、余った認識側の話者の文字は全部「挿入」、対応のない正解側の話者の文字は全部「脱落」。
    (全順列を試す定義と同じ結果を、Hungarian 法で速く求める)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import editdist
from .editdist import ErrorCounts, Op
from .model import Utt, sort_by_time
from .textnorm import NormOptions, normalize_text, normalize_words


@dataclass
class DiffPiece:
    kind: str      # "ok" 一致 / "sub" 置換 / "del" 脱落 (正解側のみ) / "ins" 挿入 (認識側のみ)
    text: str


@dataclass
class UttDiff:
    """正解の1発言に割り当てた整列結果。"""

    index: int                       # 正解の発言の通し番号 (0始まり、時刻順)
    counts: ErrorCounts = field(default_factory=ErrorCounts)
    ref_pieces: list[DiffPiece] = field(default_factory=list)
    hyp_pieces: list[DiffPiece] = field(default_factory=list)

    @property
    def hyp_text(self) -> str:
        return "".join(p.text for p in self.hyp_pieces)


@dataclass
class TextResult:
    counts: ErrorCounts                 # 正規化あり
    raw_counts: ErrorCounts | None      # 正規化をほとんどしない場合 (CER (正規化なし))
    unit: str                           # "文字" or "単語"
    diffs: list[UttDiff]
    ref_sorted: list[Utt]               # diffs[i] に対応する正解の発言 (時刻順に並べ替えたもの)
    ref_total_chars: int = 0
    hyp_total_chars: int = 0


def _tokens(text: str, opts: NormOptions) -> list[str]:
    """照合に使う単位の並び: ja は 1文字ずつ、en/fr は単語。"""
    if opts.lang == "ja":
        return list(normalize_text(text, opts))
    return normalize_words(text, opts)


def _join(tokens: list[str], opts: NormOptions) -> str:
    return "".join(tokens) if opts.lang == "ja" else " ".join(tokens)


def _pieces(items: list[tuple[str, str]], opts: NormOptions) -> list[DiffPiece]:
    out: list[DiffPiece] = []
    sep = "" if opts.lang == "ja" else " "
    for kind, tok in items:
        if out and out[-1].kind == kind:
            out[-1].text += (sep + tok) if sep else tok
        else:
            out.append(DiffPiece(kind, tok))
    return out


def text_errors(ref: list[Utt], hyp: list[Utt], opts: NormOptions, with_raw: bool = True) -> TextResult:
    """全体の誤り率と、正解の発言ごとの差分を求める。"""
    ref_sorted = sort_by_time(ref)
    ref_tokens: list[str] = []
    owner: list[int] = []   # 正解のトークンごとの発言番号
    for k, u in enumerate(ref_sorted):
        t = _tokens(u.text, opts)
        ref_tokens += t
        owner += [k] * len(t)
    hyp_tokens: list[str] = []
    for u in sort_by_time(hyp):
        hyp_tokens += _tokens(u.text, opts)

    ops = editdist.align(ref_tokens, hyp_tokens)
    counts = editdist.count_ops(ops)
    diffs = [UttDiff(index=k) for k in range(len(ref_sorted))]
    raw_counts = None
    if with_raw:
        ro = opts.raw()
        if ro == opts:
            raw_counts = counts
        else:
            rt = [t for u in ref_sorted for t in _tokens(u.text, ro)]
            ht = [t for u in sort_by_time(hyp) for t in _tokens(u.text, ro)]
            raw_counts = editdist.count_ops(editdist.align(rt, ht)) if (rt or ht) else ErrorCounts()

    if diffs:
        per_ref: list[list[tuple[str, str]]] = [[] for _ in diffs]
        per_hyp: list[list[tuple[str, str]]] = [[] for _ in diffs]
        per_ops: list[list[Op]] = [[] for _ in diffs]
        cur = 0   # 直前の正解トークンの持ち主 (挿入の割り当て先)
        pending_ins: list[Op] = []
        first_seen = False
        for op in ops:
            kind, ri, hi = op
            if kind == "I":
                if not first_seen:
                    pending_ins.append(op)
                    continue
                k = cur
                per_ops[k].append(op)
                per_hyp[k].append(("ins", hyp_tokens[hi]))
                continue
            k = owner[ri]
            if not first_seen:
                first_seen = True
                for p in pending_ins:   # 先頭の挿入は最初の発言に割り当てる
                    per_ops[k].append(p)
                    per_hyp[k].append(("ins", hyp_tokens[p[2]]))
                pending_ins = []
            cur = k
            per_ops[k].append(op)
            if kind == "=":
                per_ref[k].append(("ok", ref_tokens[ri]))
                per_hyp[k].append(("ok", hyp_tokens[hi]))
            elif kind == "S":
                per_ref[k].append(("sub", ref_tokens[ri]))
                per_hyp[k].append(("sub", hyp_tokens[hi]))
            else:
                per_ref[k].append(("del", ref_tokens[ri]))
        if pending_ins:   # 正解が空 (トークンが1つも無い) とき
            per_ops[0] += pending_ins
            per_hyp[0] += [("ins", hyp_tokens[p[2]]) for p in pending_ins]
        for k, d in enumerate(diffs):
            d.counts = editdist.count_ops(per_ops[k])
            d.ref_pieces = _pieces(per_ref[k], opts)
            d.hyp_pieces = _pieces(per_hyp[k], opts)
    return TextResult(
        counts=counts, raw_counts=raw_counts, unit="文字" if opts.lang == "ja" else "単語",
        diffs=diffs, ref_sorted=ref_sorted, ref_total_chars=len(ref_tokens), hyp_total_chars=len(hyp_tokens),
    )


# ---------------------------------------------------------------------------
# cpCER
# ---------------------------------------------------------------------------
@dataclass
class CpCerResult:
    counts: ErrorCounts
    mapping: dict[str, str]                       # 正解の話者 → 認識側の話者
    per_speaker: dict[str, ErrorCounts]           # 正解の話者ごとの誤り (対応先との比較)
    unmatched_hyp: list[str] = field(default_factory=list)
    unmatched_hyp_chars: int = 0


def cp_cer(ref: list[Utt], hyp: list[Utt], opts: NormOptions, force: bool = False) -> CpCerResult | None:
    """話者付きの認識結果が無ければ None (force=True なら、空の認識結果として全部脱落で数える)。"""
    if not force and not any(u.speaker for u in hyp):
        return None

    def per_speaker(utts: list[Utt]) -> dict[str, list[str]]:
        d: dict[str, list[str]] = {}
        for u in sort_by_time(utts):
            if not u.speaker:
                continue
            d.setdefault(u.speaker, []).extend(_tokens(u.text, opts))
        return d

    rs, hs = per_speaker(ref), per_speaker(hyp)
    rn, hn = sorted(rs), sorted(hs)
    n = max(len(rn), len(hn))
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a = rs[rn[i]] if i < len(rn) else []
            b = hs[hn[j]] if j < len(hn) else []
            cost[i, j] = editdist.distance(a, b)
    rr, cc = linear_sum_assignment(cost)
    total = ErrorCounts()
    mapping: dict[str, str] = {}
    per: dict[str, ErrorCounts] = {}
    unmatched: list[str] = []
    unmatched_chars = 0
    for i, j in zip(rr, cc):
        a = rs[rn[i]] if i < len(rn) else []
        b = hs[hn[j]] if j < len(hn) else []
        if i < len(rn) and j < len(hn):
            c = editdist.count_ops(editdist.align(a, b))
            mapping[rn[i]] = hn[j]
            per[rn[i]] = c
        elif i < len(rn):   # 対応する認識側の話者がいない → 全部脱落
            c = ErrorCounts(n_ref=len(a), dele=len(a))
            per[rn[i]] = c
        else:               # 余った認識側の話者 → 全部挿入
            c = ErrorCounts(ins=len(b))
            unmatched.append(hn[j])
            unmatched_chars += len(b)
        total += c
    return CpCerResult(total, mapping, per, unmatched, unmatched_chars)
