"""話者に関する指標: DER (話者分離誤り率) と 話者正解率 (発言単位)。純 Python + numpy (+ scipy の Hungarian 法)。

DER の定義 (NIST md-eval の SPEAKER 誤りと同じ考え方):
    DER = (見逃し + 誤検出 + 話者混同) ÷ 正解側の話している時間の合計
  時間を「正解側と認識側で話している人の集合が変わらない短い区間」に刻み、各区間 (長さ d) で
    N_ref  = 正解側で同時に話している人数        N_sys = 認識側で同時に話している人数
    N_corr = 正解側の話者のうち、対応づけた認識側の話者も話している人数
  として
    見逃し (Miss)       = max(0, N_ref - N_sys) × d
    誤検出 (False alarm) = max(0, N_sys - N_ref) × d
    話者混同 (Confusion) = (min(N_ref, N_sys) - N_corr) × d
  を足し上げる。重なり発話は N_ref ≥ 2 の区間としてそのまま数える (除外しない)。
  正解側の話者名と認識側の話者名の対応は、一致時間の合計が最大になる 1対1 の組み合わせ (Hungarian 法) を使う。

collar (許容幅): 正解側の話者区間の始まりと終わりの前後 collar 秒を採点から外す (NIST の -c と同じ。前後それぞれ collar 秒)。
  pyannote.metrics の collar は「前後あわせた幅」なので、pyannote の collar=0.5 は ここでの collar=0.25 に相当する。
  同じ話者の区間どうしが重なる/接している場合は先に1つにつないでから境界を決める。

話者正解率 (発言単位): 正解の発言ごとに、その時間帯でいちばん長く重なっている認識側の話者 (多数決) を決め、
  認識側の話者名→正解側の話者名の対応を「正解になる発言が最大になる」1対1の組み合わせで決めて (同数なら発言の合計時間が長い方)、
  一致した発言の割合を出す。
  重なりが全く無い発言 (認識側がその時間に誰も話していない) は不正解として数える。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

Seg = tuple[float, float, str]  # (開始秒, 終了秒, 話者名)


def clean_segments(segs: list[Seg]) -> list[Seg]:
    return [(float(s), float(e), str(spk)) for s, e, spk in segs if spk not in ("", None) and e > s]


def merge_same_speaker(segs: list[Seg]) -> list[Seg]:
    by: dict[str, list[tuple[float, float]]] = {}
    for s, e, spk in segs:
        by.setdefault(spk, []).append((s, e))
    out: list[Seg] = []
    for spk, lst in by.items():
        lst.sort()
        cs, ce = lst[0]
        for s, e in lst[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                out.append((cs, ce, spk))
                cs, ce = s, e
        out.append((cs, ce, spk))
    out.sort()
    return out


def _regions_minus_zones(lo: float, hi: float, zones: list[tuple[float, float]]) -> list[tuple[float, float]]:
    zones = sorted((max(lo, a), min(hi, b)) for a, b in zones if b > lo and a < hi)
    out: list[tuple[float, float]] = []
    cur = lo
    for a, b in zones:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out


@dataclass
class DerResult:
    total: float = 0.0          # 正解側の話している時間の合計 (秒、人数×時間)
    miss: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    collar: float = 0.0
    mapping: dict[str, str] = field(default_factory=dict)   # 正解の話者 → 認識側の話者
    ref_speakers: list[str] = field(default_factory=list)
    hyp_speakers: list[str] = field(default_factory=list)
    overlap: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # 採点した範囲での 一致時間 [正解話者, 認識話者]
    ref_overlap_time: float = 0.0   # 正解側で2人以上が同時に話している時間 (採点範囲)

    @property
    def errors(self) -> float:
        return self.miss + self.false_alarm + self.confusion

    @property
    def der(self) -> float:
        return self.errors / self.total if self.total > 0 else float("nan")

    def rate(self, part: str) -> float:
        v = {"miss": self.miss, "false_alarm": self.false_alarm, "confusion": self.confusion}[part]
        return v / self.total if self.total > 0 else float("nan")

    def add(self, other: "DerResult") -> None:
        self.total += other.total
        self.miss += other.miss
        self.false_alarm += other.false_alarm
        self.confusion += other.confusion
        self.ref_overlap_time += other.ref_overlap_time


def compute_der(ref: list[Seg], hyp: list[Seg], collar: float = 0.0, uem: tuple[float, float] | None = None) -> DerResult:
    ref = clean_segments(ref)
    hyp = clean_segments(hyp)
    res = DerResult(collar=collar)
    res.ref_speakers = sorted({s for _, _, s in ref})
    res.hyp_speakers = sorted({s for _, _, s in hyp})
    res.overlap = np.zeros((len(res.ref_speakers), len(res.hyp_speakers)))
    if not ref:
        res.false_alarm = sum(e - s for s, e, _ in merge_same_speaker(hyp))
        return res
    if uem is None:
        allt = [t for s, e, _ in ref + hyp for t in (s, e)]
        uem = (min(allt), max(allt))
    lo, hi = uem

    zones: list[tuple[float, float]] = []
    if collar > 0:
        for s, e, _ in merge_same_speaker(ref):
            zones += [(s - collar, s + collar), (e - collar, e + collar)]
    regions = _regions_minus_zones(lo, hi, zones)
    if not regions:
        return res

    pts = {lo, hi}
    for s, e, _ in ref + hyp:
        pts.update((s, e))
    for a, b in regions:
        pts.update((a, b))
    breaks = np.array(sorted(t for t in pts if lo <= t <= hi))
    n = len(breaks) - 1
    if n <= 0:
        return res
    dur = np.diff(breaks)

    def activity(segs: list[Seg], names: list[str]) -> np.ndarray:
        act = np.zeros((len(names), n), dtype=bool)
        idx = {nm: i for i, nm in enumerate(names)}
        for s, e, spk in segs:
            i0 = int(np.searchsorted(breaks, max(s, lo)))
            i1 = int(np.searchsorted(breaks, min(e, hi)))
            act[idx[spk], i0:i1] = True
        return act

    ra = activity(ref, res.ref_speakers)
    ha = activity(hyp, res.hyp_speakers)
    scored = np.zeros(n, dtype=bool)
    for a, b in regions:
        scored[int(np.searchsorted(breaks, a)):int(np.searchsorted(breaks, b))] = True
    w = dur * scored

    if len(res.hyp_speakers):
        res.overlap = (ra * w) @ ha.T.astype(float)
        rr, cc = linear_sum_assignment(-res.overlap)
        res.mapping = {res.ref_speakers[r]: res.hyp_speakers[c] for r, c in zip(rr, cc) if res.overlap[r, c] > 0}
    nref = ra.sum(axis=0)
    nsys = ha.sum(axis=0)
    ncorr = np.zeros(n)
    for r_name, h_name in res.mapping.items():
        ncorr += ra[res.ref_speakers.index(r_name)] & ha[res.hyp_speakers.index(h_name)]
    res.total = float((nref * w).sum())
    res.miss = float((np.maximum(0, nref - nsys) * w).sum())
    res.false_alarm = float((np.maximum(0, nsys - nref) * w).sum())
    res.confusion = float(((np.minimum(nref, nsys) - ncorr) * w).sum())
    res.ref_overlap_time = float(((nref >= 2) * w).sum())
    return res


# ---------------------------------------------------------------------------
# 話者正解率 (発言単位)
# ---------------------------------------------------------------------------
@dataclass
class UttSpeaker:
    index: int
    ref_speaker: str
    dominant: str | None         # 認識側でいちばん長く重なった話者 (名前は認識側のまま)
    predicted: str | None        # 対応づけ後の名前 (正解側の話者名) / 対応なしは None
    overlap_s: float
    duration_s: float
    correct: bool


@dataclass
class SpeakerAccResult:
    utts: list[UttSpeaker] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)       # 認識側の話者 → 正解側の話者
    ref_speakers: list[str] = field(default_factory=list)
    hyp_speakers: list[str] = field(default_factory=list)
    count_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))  # [正解話者, 認識話者]
    no_overlap_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))   # 正解話者ごとの「対応なし」件数

    @property
    def n(self) -> int:
        return len(self.utts)

    @property
    def n_correct(self) -> int:
        return sum(u.correct for u in self.utts)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")

    @property
    def accuracy_time(self) -> float:
        tot = sum(u.duration_s for u in self.utts)
        return sum(u.duration_s for u in self.utts if u.correct) / tot if tot > 0 else float("nan")


def speaker_accuracy(ref_utts: list[tuple[float, float, str]], hyp: list[Seg]) -> SpeakerAccResult:
    """ref_utts: [(開始, 終了, 話者)] 正解の発言。hyp: 認識側の話者区間。"""
    hyp = clean_segments(hyp)
    res = SpeakerAccResult()
    res.ref_speakers = sorted({spk for s, e, spk in ref_utts if spk and e > s})
    res.hyp_speakers = sorted({spk for _, _, spk in hyp})
    hs = np.array([s for s, _, _ in hyp])
    he = np.array([e for _, e, _ in hyp])
    hn = [spk for _, _, spk in hyp]
    rows: list[tuple[int, str, str | None, float, float]] = []
    for i, (s, e, spk) in enumerate(ref_utts):
        if not spk or e <= s:
            continue
        per: dict[str, float] = {}
        if hyp:
            ov = np.minimum(he, e) - np.maximum(hs, s)
            for k in np.nonzero(ov > 1e-9)[0]:
                per[hn[k]] = per.get(hn[k], 0.0) + float(ov[k])
        dom, best = (None, 0.0)
        for name in res.hyp_speakers:  # 名前順で走査 → 同点なら名前の小さい方
            if per.get(name, 0.0) > best + 1e-12:
                dom, best = name, per[name]
        rows.append((i, spk, dom, best, e - s))
    nr, nh = len(res.ref_speakers), len(res.hyp_speakers)
    cm = np.zeros((nr, nh), dtype=int)
    dm = np.zeros((nr, nh))   # 同数のときの決め手: 発言の長さの合計
    no_ov = np.zeros(nr, dtype=int)
    for _, spk, dom, _, d in rows:
        r = res.ref_speakers.index(spk)
        if dom is None:
            no_ov[r] += 1
        else:
            cm[r, res.hyp_speakers.index(dom)] += 1
            dm[r, res.hyp_speakers.index(dom)] += d
    res.count_matrix, res.no_overlap_counts = cm, no_ov
    if nr and nh:
        rr, cc = linear_sum_assignment(-(cm + 1e-6 * dm))
        res.mapping = {res.hyp_speakers[c]: res.ref_speakers[r] for r, c in zip(rr, cc) if cm[r, c] > 0}
    for i, spk, dom, ovs, d in rows:
        pred = res.mapping.get(dom) if dom is not None else None
        res.utts.append(UttSpeaker(i, spk, dom, pred, ovs, d, pred == spk))
    return res
