"""公開データ JVNV から「疑似会議」を作る。

4人が順番に話す会議の録音を、1発話ずつの朗読音声をつないで作る（乱数の種を固定するので、何度作っても同じ）。
  A（重ならない）: 発言のあいだに 0.2〜1.0 秒の間
  B（重なる）    : 発言のうち約 12.5%（10〜15% の範囲）が、直前の発言に 0.3〜1.0 秒かぶさって始まる
                   （A と同じ発言を同じ順番で使う。違いは間の取り方だけ）
正解の書き起こし（start,end,speaker,text）は asrcheck がそのまま読める形で出す。

注意（結果を読むときに必ず思い出すこと）: スタジオ録音の読み上げ文で、雑音も反響もなく、感情が大げさです。
実際の会議より「やさしい」条件なので、ここで出る数字は実会議の精度の上限に近い参考値です。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .jvnv import Clip

SR_HI = 48000
SR_LO = 16000
VARIANTS = {
    "A": "重ならない（間 0.2〜1.0 秒）",
    "B": "約12.5%の発言が直前の発言に0.3〜1.0秒かぶさる",
}


@dataclass
class BuildParams:
    seed: int = 20261019
    minutes_min: float = 6.0       # 1人あたりの話す時間（分）の範囲。人ごとにこの範囲から1つ選ぶ
    minutes_max: float = 8.0
    gap_min: float = 0.2
    gap_max: float = 1.0
    overlap_ratio: float = 0.125   # B で重ねる発言の割合（直前の発言と話者が違うものから選ぶ）
    overlap_min: float = 0.3
    overlap_max: float = 1.0
    lead_in: float = 1.0           # 録音の頭と尻の無音（秒）
    tail: float = 1.0
    trim_db: float = 35.0          # 発話の前後の無音を切る基準（その発話の最大音量から何 dB 下まで残すか）
    pad_s: float = 0.03            # 切ったあとに前後へ残す余白
    target_rms_db: float = -24.0   # 発話の音量をそろえる目標（発話区間の RMS、dBFS）


@dataclass
class Placed:
    no: int
    clip: Clip
    start: float = 0.0
    end: float = 0.0
    i0: int = 0          # 元音声（48kHz）の中で使う範囲（無音を切ったあと）
    i1: int = 0

    @property
    def dur(self) -> float:
        return (self.i1 - self.i0) / SR_HI


# ---------------------------------------------------------------------------
# 1. どの発話をどの順番で使うか
# ---------------------------------------------------------------------------
def plan_turns(clips: list[Clip], params: BuildParams, speakers: list[str] | None = None,
               duration_of=None) -> tuple[list[Clip], dict]:
    """話者ごとの持ち時間（分の範囲から乱数で決める）に達するまで、順番に発話を選ぶ。
    duration_of(clip) → 秒: 持ち時間の数え方（既定は元ファイルの長さ。build では無音を切ったあとの長さを渡す）。

    - 同じ文（キー）は会議の中で1回しか使わない（4人が同じ文を読んでいるので、本文の重複を避ける）
    - 感情は話者ごとに均等に選ぶ（その話者に残っている感情からランダム）
    - 直前と違う話者を優先する（残りが1人になったときだけ同じ人が続く）
    """
    rng = random.Random(params.seed)
    spk = sorted({c.speaker for c in clips}) if speakers is None else list(speakers)
    budget = {s: rng.uniform(params.minutes_min, params.minutes_max) * 60.0 for s in spk}
    pools: dict[str, dict[str, list[Clip]]] = {s: {} for s in spk}
    for c in sorted(clips, key=lambda c: (c.speaker, c.key)):
        if c.speaker in pools:
            pools[c.speaker].setdefault(c.emotion, []).append(c)
    for s in spk:
        for emo in pools[s]:
            rng.shuffle(pools[s][emo])

    used_keys: set[str] = set()
    used_s = {s: 0.0 for s in spk}
    active = list(spk)
    order: list[Clip] = []
    prev = None
    short: list[str] = []
    while active:
        cands = [s for s in active if s != prev] or list(active)
        weights = [max(budget[s] - used_s[s], 1.0) for s in cands]
        s = rng.choices(cands, weights)[0]
        pick = None
        emos = sorted(e for e, lst in pools[s].items() if lst)
        while emos and pick is None:
            e = rng.choice(emos)
            lst = pools[s][e]
            while lst and lst[-1].key in used_keys:
                lst.pop()
            if lst:
                pick = lst.pop()
            else:
                emos.remove(e)
        if pick is None:   # この話者の発話を使い切った
            active.remove(s)
            short.append(s)
            continue
        used_keys.add(pick.key)
        used_s[s] += (duration_of or (lambda c: c.duration))(pick)
        order.append(pick)
        prev = s
        if used_s[s] >= budget[s]:
            active.remove(s)
    info = {"budget_min": {s: round(budget[s] / 60, 2) for s in spk}, "ran_out": short}
    return order, info


# ---------------------------------------------------------------------------
# 2. 無音を切る・音量をそろえる
# ---------------------------------------------------------------------------
def trim_range(y: np.ndarray, sr: int, trim_db: float, pad_s: float) -> tuple[int, int]:
    """発話の前後の無音を切った範囲 [i0, i1)。10ms ごとの音量が、最大より trim_db 以上小さい部分を無音とみなす。"""
    hop = max(1, int(sr * 0.01))
    n = len(y) // hop
    if n < 2:
        return 0, len(y)
    frames = y[: n * hop].reshape(n, hop).astype(np.float64)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)
    active = np.nonzero(db > db.max() - trim_db)[0]
    if len(active) == 0:
        return 0, len(y)
    pad = int(pad_s * sr)
    return max(0, int(active[0]) * hop - pad), min(len(y), (int(active[-1]) + 1) * hop + pad)


def read_mono(path: Path, sr: int = SR_HI) -> np.ndarray:
    import soundfile as sf

    y, fs = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
    if fs != sr:
        from scipy.signal import resample_poly

        g = math.gcd(int(fs), sr)
        y = resample_poly(y, sr // g, int(fs) // g).astype(np.float32)
    return y


def rms_db(y: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) + 1e-10)


class TrimCache:
    """発話ごとの『使う範囲（無音を切ったあと）』を1回だけ計算して覚えておく。"""

    def __init__(self, params: BuildParams):
        self.params = params
        self._d: dict[Path, tuple[int, int]] = {}

    def range_of(self, clip: Clip) -> tuple[int, int]:
        if clip.path not in self._d:
            y = read_mono(clip.path)
            self._d[clip.path] = trim_range(y, SR_HI, self.params.trim_db, self.params.pad_s)
        return self._d[clip.path]

    def duration_of(self, clip: Clip) -> float:
        i0, i1 = self.range_of(clip)
        return (i1 - i0) / SR_HI


def prepare_turns(order: list[Clip], cache: TrimCache) -> list[Placed]:
    placed = []
    for i, c in enumerate(order, start=1):
        i0, i1 = cache.range_of(c)
        placed.append(Placed(no=i, clip=c, i0=i0, i1=i1))
    return placed


# ---------------------------------------------------------------------------
# 3. 時間の割り当て（A: 重ならない / B: 重なる）
# ---------------------------------------------------------------------------
def schedule(placed: list[Placed], params: BuildParams, variant: str) -> list[int]:
    """各発言の start/end を決める。戻り値: 直前にかぶせた発言の番号（no）のリスト（A は空）。"""
    if variant not in VARIANTS:
        raise ValueError(f"variant は {', '.join(VARIANTS)} のどれか: {variant}")
    rng = random.Random(params.seed + (101 if variant == "A" else 202))
    n = len(placed)
    chosen: set[int] = set()
    if variant == "B" and n > 1:
        eligible = [i for i in range(1, n)
                    if placed[i].clip.speaker != placed[i - 1].clip.speaker
                    and min(placed[i].dur, placed[i - 1].dur) > params.overlap_max + 0.5]
        k = min(len(eligible), round(params.overlap_ratio * (n - 1)))
        chosen = set(rng.sample(eligible, k))
    prev_end = 0.0
    for i, p in enumerate(placed):
        if i == 0:
            start = params.lead_in
        elif i in chosen:
            start = prev_end - rng.uniform(params.overlap_min, params.overlap_max)
        else:
            start = prev_end + rng.uniform(params.gap_min, params.gap_max)
        p.start = round(start, 3)
        p.end = round(start + p.dur, 3)
        prev_end = p.end
    return [placed[i].no for i in sorted(chosen)]


# ---------------------------------------------------------------------------
# 4. 音声を合成する
# ---------------------------------------------------------------------------
def render(placed: list[Placed], params: BuildParams) -> tuple[np.ndarray, float]:
    """48kHz モノラルで合成する。戻り値: (音声, 最後にかけた全体の音量補正 dB)。"""
    total = int(math.ceil((max(p.end for p in placed) + params.tail) * SR_HI))
    mix = np.zeros(total, dtype=np.float32)
    fade = int(0.01 * SR_HI)
    for p in placed:
        y = read_mono(p.clip.path)[p.i0:p.i1].copy()
        gain = 10 ** ((params.target_rms_db - rms_db(y)) / 20)
        y *= np.float32(gain)
        np.clip(y, -1.0, 1.0, out=y)
        if len(y) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            y[:fade] *= ramp
            y[-fade:] *= ramp[::-1]
        a = int(round(p.start * SR_HI))
        mix[a:a + len(y)] += y[: max(0, total - a)]
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    scale_db = 0.0
    if peak > 0.97:
        mix *= np.float32(0.97 / peak)
        scale_db = 20 * math.log10(0.97 / peak)
    return mix, scale_db


def to_16k(y48: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(y48, 1, SR_HI // SR_LO).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. 書き出し
# ---------------------------------------------------------------------------
REF_COLUMNS = ["no", "start", "end", "speaker", "text", "emotion", "style", "source_wav"]


def write_reference(path: Path, placed: list[Placed]) -> None:
    rows = sorted(placed, key=lambda p: (p.start, p.end))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(REF_COLUMNS)
        for p in rows:
            w.writerow([p.no, f"{p.start:.3f}", f"{p.end:.3f}", p.clip.speaker, p.clip.text, p.clip.emotion, p.clip.style,
                        p.clip.path.name])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def variant_stats(placed: list[Placed], overlaps: list[int], audio_s: float) -> dict:
    by_spk: dict[str, float] = {}
    emo: dict[str, int] = {}
    for p in placed:
        by_spk[p.clip.speaker] = by_spk.get(p.clip.speaker, 0.0) + p.dur
        emo[p.clip.emotion] = emo.get(p.clip.emotion, 0) + 1
    return {
        "audio_sec": round(audio_s, 2),
        "turns": len(placed),
        "overlapped_turns": len(overlaps),
        "overlap_ratio": round(len(overlaps) / max(1, len(placed) - 1), 4),
        "speech_min_per_speaker": {s: round(v / 60, 2) for s, v in sorted(by_spk.items())},
        "turns_per_speaker": {s: sum(1 for p in placed if p.clip.speaker == s) for s in sorted(by_spk)},
        "emotion_turns": dict(sorted(emo.items())),
        "ref_chars": sum(len(p.clip.text) for p in placed),
    }


def build_pseudo_meeting(clips: list[Clip], out_dir: Path | str, params: BuildParams | None = None,
                         variants: tuple[str, ...] = ("A", "B"), keep_48k: bool = False,
                         speakers: list[str] | None = None, progress=print) -> dict:
    """疑似会議を作って out_dir に書き出す。戻り値は manifest（同じ内容を manifest.json にも書く）。"""
    import soundfile as sf

    params = params or BuildParams()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = TrimCache(params)
    order, plan_info = plan_turns(clips, params, speakers, duration_of=cache.duration_of)
    progress(f"発言 {len(order)} 件を選びました。")
    placed = prepare_turns(order, cache)
    manifest: dict = {
        "tool": "meetingbench.build",
        "params": asdict(params),
        "plan": plan_info,
        "source": "JVNV corpus (non-verbal removed) — CC BY-SA 4.0",
        "variants": {},
    }
    for v in variants:
        overlaps = schedule(placed, params, v)
        progress(f"[{v}] {VARIANTS[v]}: 合成しています ...")
        y48, scale_db = render(placed, params)
        name = f"meeting{v}"
        files = {}
        sf.write(str(out / f"{name}.wav"), to_16k(y48), SR_LO, subtype="PCM_16")
        files["wav16k"] = f"{name}.wav"
        if keep_48k:
            sf.write(str(out / f"{name}_48k.wav"), y48, SR_HI, subtype="PCM_16")
            files["wav48k"] = f"{name}_48k.wav"
        write_reference(out / f"ref_{name}.csv", placed)
        files["reference"] = f"ref_{name}.csv"
        st = variant_stats(placed, overlaps, len(y48) / SR_HI)
        st.update({"description": VARIANTS[v], "files": files, "master_gain_db": round(scale_db, 2),
                   "overlap_turn_nos": overlaps, "sha256_16k": sha256_of(out / f"{name}.wav")})
        # 発言の元ファイル（どの JVNV の wav を使ったか）も残す
        st["turns_detail"] = [{"no": p.no, "speaker": p.clip.speaker, "key": p.clip.key, "source_wav": p.clip.path.name,
                               "start": p.start, "end": p.end} for p in placed]
        manifest["variants"][v] = st
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest
