"""事前に登録した声との照合で、字幕の区間に話者名を付ける（教師あり識別）。

PLAN §4.1 の ②「事前登録した声と照合して名前を付ける」を試すための部品。
pyVideoTrans の内蔵 built は『誰が話しているか』を教師なしクラスタリングで決めるので精度が出ない（research.md 22章）。
ここでは参加者ごとに事前に登録した声（約3分、D14）を使い、字幕の区間ごとに『いちばん近い登録者』を選ぶ。

- この中身は numpy だけで動く（モデルは使わない）。埋め込みを作る部分は enroll_embed.py、採点と Excel は enroll_eval.py。
- 登録に使う発話は『会議で使っていない発話』だけ（manifest.json の turns_detail と照合する）。
- 点数は『区間の埋め込み』と『各登録者の平均埋め込み』のコサイン類似度（rule="mean"）、
  または『登録した各発話の埋め込みとの類似度の最大』（rule="max"）。
- しきい値を下回ったら『不明』にできる（しきい値は登録音声だけから決める。会議の正解は見ない）。
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

EMOTIONS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
CALM = ("happy", "sad")                       # 落ち着いた朗読に近い感情（JVNV に『平常』はない）
ENROLL_MODES: dict[str, tuple[str, ...]] = {"mixed": EMOTIONS, "calm": CALM}
MODE_JA = {"mixed": "感情を混ぜる（6種）", "calm": "落ち着いた感情だけ（喜び・悲しみ）"}
RULES = ("mean", "max")
RULE_JA = {"mean": "平均との類似度", "max": "登録発話との類似度の最大"}
UNKNOWN = "unknown"
POOL_SEED = 20261019
POOL_SEC_PER_EMOTION = 150.0                  # 話者×感情ごとの登録候補の量（秒）。ここから 30/60/180 秒ぶんを選ぶ
KEY_ENR = "enr:"                              # 埋め込みの保存キー: enr:<元の wav 名> / seg:<録音>:<区間の種類>:<開始ms>:<終了ms>


def seg_key(variant: str, seg: str, start_ms: int, end_ms: int) -> str:
    return f"seg:{variant}:{seg}:{int(start_ms)}:{int(end_ms)}"


def enr_key(wav_name: str) -> str:
    return KEY_ENR + wav_name


def parse_clip_name(name: str) -> tuple[str, str, str]:
    """'F1_anger_regular_02.wav' → ('F1', 'anger', 'regular')"""
    parts = Path(name).stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"JVNV の wav 名の形が違います: {name}")
    return parts[0], parts[1], parts[2]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


# ---------------------------------------------------------------------------
# 1. 登録に使う発話の選び方
# ---------------------------------------------------------------------------
def used_in_meeting(manifest: dict) -> set[tuple[str, str]]:
    """疑似会議（A と B）で使った発話 {(話者, 元の wav 名)}。登録には使わない。"""
    out: set[tuple[str, str]] = set()
    for v in manifest.get("variants", {}).values():
        for t in v.get("turns_detail", []):
            out.add((t["speaker"], t["source_wav"]))
    if not out:
        raise ValueError("manifest.json に turns_detail がありません。疑似会議を作り直してください（meetingbench build）")
    return out


def build_pool(clips: Iterable, used: set[tuple[str, str]], per_emotion_sec: float = POOL_SEC_PER_EMOTION,
               seed: int = POOL_SEED) -> list:
    """登録の候補にする発話を決める。clips は jvnv.Clip（speaker, emotion, path, duration を持つもの）。

    会議で使っていない発話だけを、話者×感情ごとに乱数で並べ替え、元の長さの合計が per_emotion_sec 以上になるまで取る。
    （全部の発話の埋め込みを作ると CPU の時間がかかるので、候補を絞る。どの埋め込み器でも同じ候補を使う）
    """
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for c in sorted(clips, key=lambda c: (c.speaker, c.emotion, c.path.name)):
        if (c.speaker, c.path.name) in used:
            continue
        groups[(c.speaker, c.emotion)].append(c)
    out = []
    for key in sorted(groups):
        lst = groups[key]
        rng.shuffle(lst)
        acc = 0.0
        for c in lst:
            if acc >= per_emotion_sec:
                break
            out.append(c)
            acc += float(c.duration)
    return out


def pick_enrollment(avail: dict[str, list[tuple[str, float]]], emotions: Sequence[str], target_sec: float,
                    seed: int | str) -> tuple[list[str], float]:
    """1人ぶんの登録発話を選ぶ。avail: {感情: [(キー, 秒)]}。

    感情の順番と、各感情の中の順番を seed で決め、感情を順番に回しながら 1 本ずつ足し、合計が target_sec 以上になったら止める
    （最後の 1 本で少し超える）。候補が足りなければ、ある分だけ（戻り値の秒数が target_sec 未満になる）。
    """
    rng = random.Random(f"pick-{seed}")
    emos = [e for e in emotions if avail.get(e)]
    rng.shuffle(emos)
    queues = {e: sorted(avail[e]) for e in emos}
    for e in emos:
        rng.shuffle(queues[e])
    keys: list[str] = []
    total = 0.0
    i = 0
    while total < target_sec and any(queues.values()):
        e = emos[i % len(emos)]
        i += 1
        if queues[e]:
            k, sec = queues[e].pop()
            keys.append(k)
            total += float(sec)
    return keys, total


# ---------------------------------------------------------------------------
# 2. 埋め込みの保存（モデルを回し直さずに、条件を変えて何度も試すため）
# ---------------------------------------------------------------------------
class EmbeddingStore:
    """キー → 埋め込み（と音声の秒数）。1 つの埋め込み器につき npz 1 ファイル。"""

    def __init__(self) -> None:
        self.vec: dict[str, np.ndarray] = {}
        self.sec: dict[str, float] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.vec

    def __len__(self) -> int:
        return len(self.vec)

    def put(self, key: str, v: np.ndarray, sec: float) -> None:
        self.vec[key] = np.asarray(v, dtype=np.float32).ravel()
        self.sec[key] = float(sec)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.vec)
        mat = np.stack([self.vec[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:                      # np.savez は拡張子を勝手に付けるので、ファイルオブジェクトで書く
            np.savez(f, keys=np.array(keys), mat=mat, sec=np.array([self.sec[k] for k in keys], dtype=np.float64))
        os.replace(tmp, path)                           # 途中で止まっても壊れたファイルを残さない

    @classmethod
    def load(cls, path: Path | str) -> "EmbeddingStore":
        st = cls()
        path = Path(path)
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                for k, v, s in zip(z["keys"].tolist(), z["mat"], z["sec"]):
                    st.vec[str(k)] = v
                    st.sec[str(k)] = float(s)
        return st

    def enrollment_avail(self) -> dict[str, dict[str, list[tuple[str, float]]]]:
        """登録候補 {話者: {感情: [(キー, 秒)]}}（保存されている enr: だけ）。"""
        out: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
        for k in self.vec:
            if k.startswith(KEY_ENR):
                spk, emo, _ = parse_clip_name(k[len(KEY_ENR):])
                out[spk][emo].append((k, self.sec[k]))
        return {s: dict(d) for s, d in sorted(out.items())}

    def matrix(self, keys: Sequence[str]) -> np.ndarray:
        return np.stack([self.vec[k] for k in keys]).astype(np.float64)


@dataclass
class Enrollment:
    """登録した声。speakers[i] の発話の埋め込みが clips[i]（行ごとに長さ1）。"""

    speakers: list[str]
    clips: list[np.ndarray]
    secs: list[float]
    keys: list[list[str]]

    @property
    def centroids(self) -> np.ndarray:
        return np.stack([unit(unit(c).mean(axis=0)) for c in self.clips])


def make_enrollment(store: EmbeddingStore, mode: str, target_sec: float, seed: int, speakers: Sequence[str] | None = None) -> Enrollment:
    avail = store.enrollment_avail()
    spks = list(speakers) if speakers else list(avail)
    clips, secs, keys = [], [], []
    for s in spks:
        if s not in avail:
            raise ValueError(f"話者 {s} の登録候補がありません（enroll-embed を実行してください）")
        ks, sec = pick_enrollment(avail[s], ENROLL_MODES[mode], target_sec, f"{seed}-{s}")
        clips.append(unit(store.matrix(ks)))
        secs.append(sec)
        keys.append(ks)
    return Enrollment(spks, clips, secs, keys)


# ---------------------------------------------------------------------------
# 3. 照合（いちばん近い登録者を選ぶ）
# ---------------------------------------------------------------------------
def similarity(seg_embs: np.ndarray, enr: Enrollment, rule: str = "mean") -> np.ndarray:
    """(区間数, 話者数) のコサイン類似度。rule="mean": 平均埋め込みとの類似度 / "max": 登録発話ごとの類似度の最大。"""
    if rule not in RULES:
        raise ValueError(f"rule は {', '.join(RULES)} のどれか: {rule}")
    x = unit(seg_embs)
    if rule == "mean":
        return x @ enr.centroids.T
    return np.stack([(x @ c.T).max(axis=1) for c in enr.clips], axis=1)


def decide(sim: np.ndarray, speakers: Sequence[str], threshold: float | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """戻り値: (話者名のリスト, 最高点, 2位との差)。threshold を下回る区間は 'unknown'（None なら使わない）。"""
    order = np.argsort(-sim, axis=1)
    best = sim[np.arange(len(sim)), order[:, 0]]
    second = sim[np.arange(len(sim)), order[:, 1]] if sim.shape[1] > 1 else np.zeros(len(sim))
    labels = [speakers[j] for j in order[:, 0]]
    if threshold is not None:
        labels = [UNKNOWN if b < threshold else lab for lab, b in zip(labels, best)]
    return labels, best, best - second


def loo_genuine_scores(enr: Enrollment, rule: str = "mean") -> np.ndarray:
    """登録した発話 1 本ずつを『その本を除いた登録』と照合したときの、本人への点数（しきい値を決める材料。会議の正解は使わない）。"""
    out = []
    for c in enr.clips:
        if len(c) < 2:
            continue
        for i in range(len(c)):
            rest = np.delete(c, i, axis=0)
            out.append(float(c[i] @ unit(rest.mean(axis=0))) if rule == "mean" else float((rest @ c[i]).max()))
    return np.array(out)


def threshold_from_enrollment(enr: Enrollment, pct: float = 5.0, rule: str = "mean") -> float:
    """本人への点数（loo_genuine_scores）の下から pct パーセントの値。これを下回る区間を『不明』にする。"""
    s = loo_genuine_scores(enr, rule)
    if len(s) == 0:
        raise ValueError("しきい値を決めるには、1人あたり 2 本以上の登録発話が必要です")
    return float(np.percentile(s, pct))


def reject_curve(best: np.ndarray, correct: np.ndarray, thresholds: Iterable[float]) -> list[dict]:
    """しきい値ごとの『不明にした割合』『受け入れた中の正解率』『不明にした中で本当は誤りだった割合』。"""
    best, correct = np.asarray(best), np.asarray(correct, dtype=bool)
    rows = []
    for t in thresholds:
        acc = best >= t
        rej = ~acc
        rows.append({
            "threshold": float(t),
            "rejected": int(rej.sum()),
            "rejected_pct": float(rej.mean()) if len(best) else 0.0,
            "accuracy_accepted": float(correct[acc].mean()) if acc.any() else None,
            "wrong_among_rejected": float((~correct[rej]).mean()) if rej.any() else None,
            "errors_caught_pct": float((~correct[rej]).sum() / max(1, (~correct).sum())),
        })
    return rows


# ---------------------------------------------------------------------------
# 4. 平滑化（短い区間・連続する区間の扱い）
# ---------------------------------------------------------------------------
def pool_continuous(seg_embs: np.ndarray, starts_ms: Sequence[int], ends_ms: Sequence[int], max_gap_ms: int = 150,
                    max_group: int = 3) -> list[list[int]]:
    """隙間が max_gap_ms 以下でつながっている区間（VAD が最長秒数で切った同じ発言の続き）を最大 max_group 個まで 1 グループにする。
    戻り値: グループごとの区間番号のリスト（入力の並び順）。"""
    order = sorted(range(len(starts_ms)), key=lambda i: (starts_ms[i], ends_ms[i]))
    groups: list[list[int]] = []
    for i in order:
        if groups and len(groups[-1]) < max_group and starts_ms[i] - ends_ms[groups[-1][-1]] <= max_gap_ms:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def smooth_by_groups(sim: np.ndarray, durations_s: Sequence[float], groups: list[list[int]]) -> np.ndarray:
    """同じグループの区間は、長さで重みをつけた類似度の平均を使う（同じ発言の続きなので、同じ話者のはず）。"""
    out = sim.copy()
    d = np.asarray(durations_s, dtype=np.float64)
    for g in groups:
        if len(g) > 1:
            w = d[g] / d[g].sum()
            out[g] = (sim[g] * w[:, None]).sum(axis=0)
    return out


def smooth_short(labels: Sequence[str], sim: np.ndarray, starts_ms: Sequence[int], ends_ms: Sequence[int], speakers: Sequence[str],
                 min_sec: float = 1.0, max_gap_ms: int = 300) -> list[str]:
    """短い区間（min_sec 未満）は、隙間が max_gap_ms 以下でつながる前後の区間のうち、自分の点数が高い方の話者を引き継ぐ。
    つながる区間がなければそのまま。"""
    out = list(labels)
    n = len(labels)
    idx = {s: j for j, s in enumerate(speakers)}
    for i in range(n):
        if (ends_ms[i] - starts_ms[i]) / 1000.0 >= min_sec:
            continue
        cands = []
        if i > 0 and starts_ms[i] - ends_ms[i - 1] <= max_gap_ms:
            cands.append(i - 1)
        if i + 1 < n and starts_ms[i + 1] - ends_ms[i] <= max_gap_ms:
            cands.append(i + 1)
        if cands:
            best = max(cands, key=lambda j: sim[i, idx[labels[j]]] if labels[j] in idx else -1.0)
            out[i] = labels[best]
    return out
