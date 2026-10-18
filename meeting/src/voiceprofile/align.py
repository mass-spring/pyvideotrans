"""録音の発話区間 (atom) と、台本の文 (slot) を順番どおりに対応づける。

- 区間の数と文の数が同じで、ASR を使わない時: 1 対 1 で順番どおり (状態 count_ok。長さが極端に違えば check)。
- 数が違う時 / ASR の結果がある時: 動的計画法 (DP) で「読み飛ばした文」「途中で止まって分かれた区間」
  「間をあけずに続けて読んで 1 つになった区間」「言い直し・物音の余分な区間」を許して最も自然な対応を探す。
  費用は ASR があれば文字列の似かた、なければ読む長さの比 (文の長さの見積もりと比べる)。

対応の確かさは状態コード (STATUS_*) で返し、手本の選び方 (点数) と Excel に使う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .script import Slot
from .text_utils import join_texts, text_similarity

STATUS_COUNT_OK = "count_ok"      # 区間数と文数が一致し、順番どおりに対応づけた
STATUS_ESTIMATED = "estimated"    # 長さから推定した (数が合わないため)
STATUS_ASR = "asr"                # ASR の結果が台本とよく一致した
STATUS_ASR_WEAK = "asr_weak"      # ASR の結果が台本と少ししか一致しない
STATUS_CHECK = "check"            # 長さが極端に違うなど、対応が怪しい
STATUS_NONE = "none"              # 台本なし (書き起こしは空)
STATUS_ASR_ONLY = "asr_only"      # 台本なしで、ASR の結果をそのまま書き起こしにした (誤認識があり得る)
STATUS_UNASSIGNED = "unassigned"  # どの文にも対応しなかった区間 (言い直し・物音など)

STATUS_JA = {
    STATUS_COUNT_OK: "順番どおり（数が一致）",
    STATUS_ESTIMATED: "推定（長さから）",
    STATUS_ASR: "ASRで確認",
    STATUS_ASR_WEAK: "ASRと一致度が低い",
    STATUS_CHECK: "要確認",
    STATUS_NONE: "台本なし（要入力）",
    STATUS_ASR_ONLY: "ASRのみ（要確認）",
    STATUS_UNASSIGNED: "どの文にも対応しない",
}
# 手本の点数に使う「書き起こしの信頼度」(0〜1)
STATUS_TRUST = {
    STATUS_ASR: 1.0, STATUS_COUNT_OK: 0.9, STATUS_ESTIMATED: 0.5, STATUS_NONE: 0.5, STATUS_ASR_ONLY: 0.5,
    STATUS_ASR_WEAK: 0.35, STATUS_CHECK: 0.0, STATUS_UNASSIGNED: 0.0,
}
STATUS_RANK = [STATUS_CHECK, STATUS_UNASSIGNED, STATUS_ASR_WEAK, STATUS_ESTIMATED, STATUS_ASR_ONLY, STATUS_NONE,
               STATUS_COUNT_OK, STATUS_ASR]

# DP の費用 (長さの比の対数 |ln r| と同じ尺度。1 文 1 区間でぴったり = 0)
PEN_BLOCK = {(1, 1): 0.0, (2, 1): 0.45, (3, 1): 1.0, (1, 2): 0.55, (1, 3): 1.2}
COST_SKIP_ATOM = 1.4
COST_SKIP_SLOT = 1.2
DURATION_SHARPNESS = 8.0  # 長さの比の対数の 2 乗にかける重み (比 1.13 のずれで費用約 0.12、比 1.7 で約 2.2)
GAP_OK_SEC = 0.8          # 1 つの文の途中の間として自然な長さ。これより長い間で分かれた区間を 1 文にまとめるときは費用を足す
GAP_WEIGHT = 2.0          # 上の超過 1 秒あたりの費用 (言い直し・咳の区間を文の一部にしないため)
ASR_WEIGHT = 2.5          # ASR の似かた (1 - sim) にかける重み
ASR_DURATION_WEIGHT = 0.25
ORDER_MAX_LN_RATIO = math.log(2.2)   # 数が一致した時、長さの比がこれを超える対応は「要確認」
DP_MAX_LN_RATIO = math.log(1.7)
ASR_OK_SIM = 0.7
ASR_WEAK_SIM = 0.45


def worst_status(statuses: list[str]) -> str:
    """複数の状態のうち、いちばん信頼できないもの。"""
    if not statuses:
        return STATUS_NONE
    return min(statuses, key=lambda s: STATUS_RANK.index(s) if s in STATUS_RANK else 0)


@dataclass
class Block:
    atoms: list[int]
    slots: list[int]
    status: str
    sim: float | None = None
    ratio: float | None = None     # 実際の長さ / 見積もりの長さ


@dataclass
class AlignResult:
    blocks: list[Block]
    missing_slots: list[int]
    mode: str                      # order / duration / asr
    scale: float                   # この話者の「見積もりに対する実際の長さ」の倍率
    notes: list[str] = field(default_factory=list)


def _estimate_scale(durs: list[float], weights: list[float]) -> float:
    tot_w = sum(weights)
    return min(max(sum(durs) / tot_w, 0.4), 3.0) if tot_w > 0 and durs else 1.0


def _dp(durs, files, gaps, slots, asr_texts, scale):
    n, m = len(durs), len(slots)
    w = [s.est_sec for s in slots]
    cd = np.concatenate([[0.0], np.cumsum(durs)])
    cw = np.concatenate([[0.0], np.cumsum(w)])

    def block_cost(i, k, j, l):
        d, ww = cd[i + k] - cd[i], cw[j + l] - cw[j]
        ln_r = math.log(max(d, 1e-3) / max(scale * ww, 1e-3))
        if asr_texts is None:
            return DURATION_SHARPNESS * ln_r * ln_r, None, ln_r  # 2 乗: 小さなずれは軽く、大きなずれは重く
        lang = slots[j].lang
        sim = text_similarity(join_texts([asr_texts[x] for x in range(i, i + k)], lang),
                              join_texts([slots[x].text for x in range(j, j + l)], lang), lang)
        return ASR_WEIGHT * (1.0 - sim) + ASR_DURATION_WEIGHT * abs(ln_r), sim, ln_r

    inf = float("inf")
    dp = np.full((n + 1, m + 1), inf)
    dp[0, 0] = 0.0
    back: dict[tuple[int, int], tuple] = {}

    def relax(i, j, cost, move):
        if cost < dp[i, j] - 1e-12:
            dp[i, j] = cost
            back[(i, j)] = move

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i, j]
            if cur == inf:
                continue
            if i < n:
                relax(i + 1, j, cur + COST_SKIP_ATOM, ("skip_atom", i, j))
            if j < m:
                relax(i, j + 1, cur + COST_SKIP_SLOT, ("skip_slot", i, j))
            for (k, l), pen in PEN_BLOCK.items():
                if i + k > n or j + l > m:
                    continue
                if k > 1 and len(set(files[i:i + k])) > 1:
                    continue  # 別の録音ファイルにまたがる区間は 1 文にしない
                c, sim, ln_r = block_cost(i, k, j, l)
                long_gap = sum(max(0.0, gaps[x] - GAP_OK_SEC) for x in range(i, i + k - 1)) * GAP_WEIGHT
                relax(i + k, j + l, cur + pen + c + long_gap, ("match", i, j, k, l, sim, ln_r))

    path = []
    i, j = n, m
    while (i, j) != (0, 0):
        mv = back[(i, j)]
        path.append(mv)
        i, j = mv[1], mv[2]
    path.reverse()
    return path, float(dp[n, m])


def align(
    atom_durs: list[float],
    atom_files: list[int],
    slots: list[Slot],
    asr_texts: list[str] | None = None,
    atom_gaps: list[float] | None = None,
) -> AlignResult:
    """atom_durs[i]: 区間 i の話している長さ (秒)。atom_files[i]: どの録音ファイルか。asr_texts[i]: 区間 i の ASR 結果 (任意)。
    atom_gaps[i]: 区間 i と i+1 の間の無音の長さ (秒。省略時は 0.5 秒とみなす)。"""
    n, m = len(atom_durs), len(slots)
    gaps = list(atom_gaps) if atom_gaps is not None else [0.5] * n
    if n == 0:
        raise ValueError("対応づける区間がありません")
    notes: list[str] = []
    if m == 0:
        return AlignResult([Block([i], [], STATUS_NONE) for i in range(n)], [], "none", 1.0)

    weights = [s.est_sec for s in slots]
    scale = _estimate_scale(atom_durs, weights)

    # ---- 1 対 1、順番どおり ----
    if asr_texts is None and n == m:
        blocks = []
        for i in range(n):
            ln_r = math.log(max(atom_durs[i], 1e-3) / max(scale * weights[i], 1e-3))
            blocks.append(Block([i], [i], STATUS_COUNT_OK if abs(ln_r) <= ORDER_MAX_LN_RATIO else STATUS_CHECK,
                                None, math.exp(ln_r)))
        bad = sum(b.status == STATUS_CHECK for b in blocks)
        if bad:
            notes.append(f"区間数と文数は一致しましたが、{bad} 件で長さが文の長さと大きく違います。対応がずれていないか確認してください")
        return AlignResult(blocks, [], "order", scale, notes)

    # ---- DP (長さ、または ASR の文字列の似かた) ----
    path, total = _dp(atom_durs, atom_files, gaps, slots, asr_texts, scale)
    matched_d = sum(sum(atom_durs[mv[1]:mv[1] + mv[3]]) for mv in path if mv[0] == "match")
    matched_w = sum(sum(weights[mv[2]:mv[2] + mv[4]]) for mv in path if mv[0] == "match")
    if matched_w > 0:  # 対応づけた分だけで倍率を取り直してもう一度
        scale2 = min(max(matched_d / matched_w, 0.4), 3.0)
        if abs(math.log(scale2 / scale)) > 0.05:
            scale = scale2
            path, total = _dp(atom_durs, atom_files, gaps, slots, asr_texts, scale)

    blocks: list[Block] = []
    missing: list[int] = []
    for mv in path:
        if mv[0] == "skip_atom":
            blocks.append(Block([mv[1]], [], STATUS_UNASSIGNED))
        elif mv[0] == "skip_slot":
            missing.append(mv[2])
        else:
            _, i, j, k, l, sim, ln_r = mv
            if asr_texts is not None:
                status = STATUS_ASR if sim >= ASR_OK_SIM else STATUS_ASR_WEAK if sim >= ASR_WEAK_SIM else STATUS_CHECK
            else:
                status = STATUS_ESTIMATED if abs(ln_r) <= DP_MAX_LN_RATIO else STATUS_CHECK
            blocks.append(Block(list(range(i, i + k)), list(range(j, j + l)), status, sim, math.exp(ln_r)))
    if asr_texts is None:
        notes.append(f"区間数 ({n}) と文数 ({m}) が違うため、長さから対応づけを推定しました。書き起こしを必ず確認してください"
                     "（--asr whisper を付けると文字の一致で確かめられます）")
    if missing:
        notes.append("読まれなかったように見える文: " + ", ".join(slots[j].id for j in missing))
    extra = [b for b in blocks if not b.slots]
    if extra:
        notes.append(f"どの文にも対応しなかった区間が {len(extra)} 個あります（言い直し・咳・物音の可能性）。手本には使いません")
    return AlignResult(blocks, missing, "asr" if asr_texts is not None else "duration", scale, notes)
