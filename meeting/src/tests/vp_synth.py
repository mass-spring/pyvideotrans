"""voiceprofile のテスト用: 「発話区間 + 無音 + 雑音」でできた合成の録音と、正解の位置。

発話区間 = 短い母音っぽい音 (倍音の足し合わせ) が間隔をあけて並んだもの。音と音の間には 20〜80 ms の
「へこみ」があるが、どれも 0.35 秒より短いので 1 つの発話区間として扱われるはず。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 48000


@dataclass
class SynthRec:
    y: np.ndarray
    sr: int
    segments: list[tuple[float, float]]            # 正解: 発話区間 (開始秒, 終了秒)
    noise_amps: list[float]                        # 区間ごとの雑音 (標準偏差)
    voiced_power: list[float] = field(default_factory=list)  # 区間ごとの音の出ている部分の平均パワー

    def true_snr_db(self, k: int) -> float:
        return float(10 * np.log10(self.voiced_power[k] / self.noise_amps[k] ** 2))

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr


def _burst(rng: np.random.Generator, dur: float, f0: float, sr: int, amp: float) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 * 2.0 ** (rng.uniform(-3, 3) / 12.0)
    y = np.zeros(n)
    k = 1
    while k * f < 4000:
        y += (1.0 / k) * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 2 * np.pi))
        k += 1
    y = y / np.max(np.abs(y))
    att, dec = int(0.015 * sr), int(0.025 * sr)
    env = np.ones(n)
    env[:att] = np.linspace(0, 1, att)
    env[-dec:] = np.linspace(1, 0, dec)
    return (amp * y * env).astype(np.float32)


def make_segment(rng: np.random.Generator, dur: float, f0: float, sr: int, amp: float) -> tuple[np.ndarray, float]:
    """長さ約 dur 秒の発話区間 (音の出ている部分の平均パワーも返す)。"""
    parts: list[np.ndarray] = []
    voiced: list[np.ndarray] = []
    total = 0.0
    while total < dur - 0.2:
        b = _burst(rng, rng.uniform(0.12, 0.22), f0, sr, amp)
        gap = np.zeros(int(rng.uniform(0.02, 0.08) * sr), dtype=np.float32)
        parts += [b, gap]
        voiced.append(b)
        total += (len(b) + len(gap)) / sr
    parts.pop()  # 最後の隙間は捨てる (発話区間は音で終わる)
    y = np.concatenate(parts)
    v = np.concatenate(voiced)
    return y, float(np.mean(v.astype(np.float64) ** 2))


def make_recording(
    seg_durs: list[float],
    pauses: list[float] | float = 1.0,
    *,
    sr: int = SR,
    f0: float = 120.0,
    noise_amps: list[float] | float = 0.002,
    amps: list[float] | float = 0.2,
    lead: float = 1.2,
    tail: float = 1.2,
    seed: int = 0,
    clip_segments: tuple[int, ...] = (),
) -> SynthRec:
    """seg_durs[i] 秒の発話区間を pauses 秒の無音をはさんで並べ、雑音 (白色) を重ねる。

    noise_amps: 全体で 1 つ、または区間ごと (その区間と直前の無音に適用。最後の無音は最後の区間と同じ)。
    clip_segments: この番号の区間は大きく増幅して振幅を ±1 に張り付ける (音割れ)。
    """
    rng = np.random.default_rng(seed)
    n = len(seg_durs)
    pauses = [pauses] * (n - 1) if isinstance(pauses, (int, float)) else list(pauses)
    amps = [amps] * n if isinstance(amps, (int, float)) else list(amps)
    noise_amps = [noise_amps] * n if isinstance(noise_amps, (int, float)) else list(noise_amps)
    pieces: list[np.ndarray] = []
    segments: list[tuple[float, float]] = []
    levels: list[np.ndarray] = []
    voiced_power: list[float] = []
    pos = 0.0

    def silence(sec: float, k_noise: int):
        nonlocal pos
        pieces.append(np.zeros(int(sec * sr), dtype=np.float32))
        levels.append(np.full(len(pieces[-1]), noise_amps[k_noise], dtype=np.float32))
        pos += sec

    silence(lead, 0)
    for k, d in enumerate(seg_durs):
        seg, vp = make_segment(rng, d, f0, sr, amps[k])
        if k in clip_segments:
            seg = np.clip(seg * 8.0, -1.0, 1.0).astype(np.float32)
            vp = float(np.mean(np.clip(seg, -1, 1).astype(np.float64) ** 2))
        pieces.append(seg)
        levels.append(np.full(len(seg), noise_amps[k], dtype=np.float32))
        segments.append((pos, pos + len(seg) / sr))
        voiced_power.append(vp)
        pos += len(seg) / sr
        if k < n - 1:
            silence(pauses[k], k + 1)
    silence(tail, n - 1)
    y = np.concatenate(pieces)
    lev = np.concatenate(levels)
    y = y + (rng.standard_normal(len(y)) * lev).astype(np.float32)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return SynthRec(y, sr, segments, list(noise_amps), voiced_power)


def durations_for_slots(est_secs: list[float], scale: float = 1.0, jitter: float = 0.08, seed: int = 1) -> list[float]:
    """台本の読む長さの見積もりに比例した発話区間の長さ (少しばらつかせる)。"""
    rng = np.random.default_rng(seed)
    return [max(0.8, e * scale * (1.0 + rng.uniform(-jitter, jitter))) for e in est_secs]
