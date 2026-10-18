"""手本クリップの候補づくり・点数づけ・選び方。

- 単位 (Unit): 台本の 1 文に対応する発話区間のまとまり (台本なしなら発話区間 1 つ)。クリップはこの単位を丸ごと並べて作る
  → 単語の途中では絶対に切らない (切り口は必ず発話区間と発話区間の間の無音の中)。
- クリップ: 続いている単位を並べたもの。長さが min_sec〜max_sec に入る組み合わせを全部候補にする。
  前後に pad 秒の余白 (録音の雑音の床) を付け、長い無音 (gap_keep 秒超) は gap_keep 秒に詰める。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import STATUS_TRUST, worst_status
from .segment import Segment
from .text_utils import join_texts

PAD_SEC = 0.12          # クリップの前後に付ける余白
GAP_KEEP_SEC = 0.8      # 単位どうしの間がこれより長い時は、この長さに詰める (前後 0.4 秒ずつ残す)
FADE_SEC = 0.008        # 切り口のクリックを防ぐ短いフェード

# 点数の重み (合計 1)。埋め込みが使えない時は consistency の分を残りに配る
WEIGHTS = {"snr": 0.30, "consistency": 0.25, "duration": 0.20, "clipping": 0.10, "text": 0.10, "edge": 0.05}
SNR_LO, SNR_HI = 15.0, 40.0           # この間を 0〜1 に写す (dB)
CLIP_BAD_RATIO = 0.001                # 割れ率がこれ以上なら割れ点 0
CLIP_EXCLUDE_RATIO = 0.005            # これを超える候補は、ほかに候補があれば選ばない
DURATION_SPREAD = 8.0                 # 目標の長さからこれだけ離れると長さ点 0
# 埋め込みごとの「一貫性 (センタロイドとのコサイン類似度)」を 0〜1 点に写す範囲 (目安。要較正)
CONSISTENCY_RANGE = {"ecapa": (0.50, 0.90), "resemblyzer": (0.80, 0.97)}
CONSISTENCY_RANGE_DEFAULT = (0.50, 0.95)


@dataclass
class Atom:
    idx: int
    file: int
    pos: int          # そのファイルの中で何番目の区間か
    seg: Segment


@dataclass
class Unit:
    idx: int
    file: int
    atoms: list[int]            # Atom.idx のリスト (続いている)
    start: float                # 先頭の区間の始まり (秒)
    end: float                  # 最後の区間の終わり (秒)
    speech_sec: float           # 区間の長さの合計
    slot_ids: list[str]
    text: str
    lang: str
    role: str                   # speech / consent / unassigned / mixed
    status: str
    sim: float | None = None
    asr_text: str = ""
    n_mora: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    idx: int
    file: int
    lang: str
    units: list[int]                    # Unit.idx
    pieces: list[tuple[float, float]]
    dur: float
    text: str
    status: str
    slot_ids: list[str]
    edge: float
    snr_db: float = float("nan")
    speech_dbfs: float = float("nan")
    noise_dbfs: float = float("nan")
    clip_ratio: float = 0.0
    peak_dbfs: float = float("nan")
    consistency: dict[str, float] = field(default_factory=dict)        # 点数に使う値: 文ごとの埋め込みの長さ加重平均 と センタロイドの近さ
    consistency_true: dict[str, float] = field(default_factory=dict)   # 採用した手本だけ: クリップそのものの埋め込みで測った値
    score: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    selected: bool = False
    ref_name: str = ""
    not_chosen_reason: str = ""


@dataclass
class EvalClip:
    """手本に使わなかった発話 (評価用に取り置く)。1 単位 = 1 クリップ。"""

    idx: int
    file: int
    lang: str
    unit: int
    pieces: list[tuple[float, float]]
    dur: float
    text: str
    status: str
    slot_ids: list[str]
    snr_db: float = float("nan")
    speech_dbfs: float = float("nan")
    clip_ratio: float = 0.0
    consistency: dict[str, float] = field(default_factory=dict)
    name: str = ""


# ---- 切り出し --------------------------------------------------------------------------
def clip_pieces(file_atoms: list[Segment], i0: int, i1: int, file_dur: float,
                pad: float = PAD_SEC, gap_keep: float = GAP_KEEP_SEC) -> list[tuple[float, float]]:
    """file_atoms[i0..i1] (その録音ファイルの続いた区間) を含む切り出し範囲 (秒) のリスト。

    前後は pad 秒の余白 (となりの区間があれば、その間の真ん中まで)。区間どうしの間が gap_keep 秒を超える時は
    間の真ん中を捨てて gap_keep 秒に詰める (そのぶん範囲が複数に分かれる)。
    """
    first, last = file_atoms[i0], file_atoms[i1]
    lead = first.start - pad
    if i0 > 0:
        lead = max(lead, (file_atoms[i0 - 1].end + first.start) / 2.0)
    lead = max(lead, 0.0)
    tail = last.end + pad
    if i1 + 1 < len(file_atoms):
        tail = min(tail, (last.end + file_atoms[i1 + 1].start) / 2.0)
    tail = min(tail, file_dur)
    pieces: list[tuple[float, float]] = []
    cur = lead
    half = gap_keep / 2.0
    for k in range(i0, i1):
        g0, g1 = file_atoms[k].end, file_atoms[k + 1].start
        if g1 - g0 > gap_keep:
            pieces.append((cur, g0 + half))
            cur = g1 - half
    pieces.append((cur, tail))
    return pieces


def pieces_duration(pieces: list[tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in pieces))


def assemble(y: np.ndarray, sr: int, pieces: list[tuple[float, float]], fade_sec: float = FADE_SEC) -> np.ndarray:
    """原音の波形 y から pieces の範囲を切り出して順に並べる。切り口に短いフェードを付けるだけで、音量・音質は変えない。"""
    parts = []
    nf = int(round(fade_sec * sr))
    for a, b in pieces:
        ia, ib = max(int(round(a * sr)), 0), min(int(round(b * sr)), len(y))
        seg = np.array(y[ia:ib], dtype=y.dtype, copy=True)
        k = min(nf, len(seg) // 2)
        if k > 1:
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
            seg[:k] *= ramp
            seg[-k:] *= ramp[::-1]
        parts.append(seg)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=y.dtype)


def edge_score(file_atoms: list[Segment], i0: int, i1: int) -> float:
    """クリップの前後の無音が長い (= 文の切れ目で切っている) ほど 1 に近い。"""
    def gap_score(g: float) -> float:
        return float(min(max((g - 0.3) / 0.5, 0.0), 1.0))
    before = file_atoms[i0].start - file_atoms[i0 - 1].end if i0 > 0 else 1.0
    after = file_atoms[i1 + 1].start - file_atoms[i1].end if i1 + 1 < len(file_atoms) else 1.0
    return (gap_score(before) + gap_score(after)) / 2.0


# ---- 候補の列挙 ---------------------------------------------------------------------------
def eligible_runs(units: list[Unit]) -> list[list[Unit]]:
    """手本にできる単位 (role=speech) の、時間的に続いた並び。同意文・対応しない区間・ファイル/言語の変わり目で切れる。"""
    runs: list[list[Unit]] = []
    cur: list[Unit] = []
    for u in units:
        if u.role != "speech":
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur and (cur[-1].file != u.file or cur[-1].lang != u.lang):
            runs.append(cur)
            cur = []
        cur.append(u)
    if cur:
        runs.append(cur)
    return runs


def enumerate_candidates(
    units: list[Unit],
    atoms: list[Atom],
    file_atoms: dict[int, list[Segment]],
    file_durs: dict[int, float],
    min_sec: float,
    max_sec: float,
) -> list[Candidate]:
    cands: list[Candidate] = []
    for run in eligible_runs(units):
        for a in range(len(run)):
            for b in range(a, len(run)):
                f = run[a].file
                i0, i1 = atoms[run[a].atoms[0]].pos, atoms[run[b].atoms[-1]].pos
                pieces = clip_pieces(file_atoms[f], i0, i1, file_durs[f])
                dur = pieces_duration(pieces)
                if dur > max_sec:
                    break
                if dur < min_sec:
                    continue
                members = run[a:b + 1]
                cands.append(Candidate(
                    idx=len(cands), file=f, lang=run[a].lang, units=[u.idx for u in members], pieces=pieces, dur=dur,
                    text=join_texts([u.text for u in members], run[a].lang),
                    status=worst_status([u.status for u in members]),
                    slot_ids=[s for u in members for s in u.slot_ids],
                    edge=edge_score(file_atoms[f], i0, i1),
                ))
    return cands


# ---- 点数 ----------------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def consistency_to_unit(name: str, cos: float) -> float:
    lo, hi = CONSISTENCY_RANGE.get(name, CONSISTENCY_RANGE_DEFAULT)
    return _clamp01((cos - lo) / (hi - lo))


def score_candidate(c: Candidate, target_sec: float) -> None:
    """c.score (0〜100) と c.parts (各項目の 0〜100) を埋める。"""
    parts: dict[str, float] = {
        "snr": _clamp01((c.snr_db - SNR_LO) / (SNR_HI - SNR_LO)) if np.isfinite(c.snr_db) else 0.0,
        "duration": _clamp01(1.0 - abs(c.dur - target_sec) / DURATION_SPREAD),
        "clipping": _clamp01(1.0 - c.clip_ratio / CLIP_BAD_RATIO),
        "text": STATUS_TRUST.get(c.status, 0.5),
        "edge": _clamp01(c.edge),
    }
    weights = dict(WEIGHTS)
    if c.consistency:
        parts["consistency"] = float(np.mean([consistency_to_unit(n, v) for n, v in c.consistency.items()]))
    else:
        weights.pop("consistency")
    total_w = sum(weights.values())
    c.score = float(100.0 * sum(weights[k] * parts[k] for k in weights) / total_w)
    c.parts = {k: 100.0 * v for k, v in parts.items()}


# ---- 選び方 --------------------------------------------------------------------------------
def select_refs(cands: list[Candidate], n_refs: int, units_by_idx: dict[int, Unit],
                min_heldout_sec: float = 15.0, min_refs: int = 3) -> list[Candidate]:
    """点数の高い順に、すでに選んだものと同じ単位を共有しないものを選ぶ。

    n_refs 本選ぶと取り置き (評価用) の話し声が min_heldout_sec 秒に満たない時は、min_refs 本まで減らす。
    割れ率が CLIP_EXCLUDE_RATIO を超える候補は、ほかに候補があるかぎり選ばない。
    """
    ranked = sorted(cands, key=lambda c: (-c.score, c.idx))
    ok = [c for c in ranked if c.clip_ratio <= CLIP_EXCLUDE_RATIO]
    pool = ok if ok else ranked

    def greedy(limit: int) -> list[Candidate]:
        chosen: list[Candidate] = []
        used: set[int] = set()
        for c in pool:
            if len(chosen) >= limit:
                break
            if used.isdisjoint(c.units):
                chosen.append(c)
                used.update(c.units)
        return chosen

    all_speech = {u.idx for c in cands for u in (units_by_idx[i] for i in c.units)}
    total_sec = sum(units_by_idx[i].speech_sec for i in all_speech)
    for limit in range(n_refs, max(min_refs, 1) - 1, -1):
        chosen = greedy(limit)
        used_sec = sum(units_by_idx[i].speech_sec for c in chosen for i in c.units)
        if len(chosen) < limit:
            continue  # この本数は取れない → もっと少なく
        if total_sec - used_sec >= min_heldout_sec or limit == max(min_refs, 1):
            return chosen
    return greedy(n_refs)
