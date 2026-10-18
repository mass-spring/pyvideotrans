"""テスト用の合成音声 (既知の F0・傾き・音節数を持つ信号) とダミー埋め込み。"""
from __future__ import annotations

import numpy as np

from voicecheck.embedders import Embedder

SR = 16000


def harmonic_tone(f0: float, dur: float = 1.5, sr: int = SR, rolloff_db_oct: float = -6.0,
                  amp: float = 0.3, seed: int | None = None) -> np.ndarray:
    """倍音の足し合わせ。倍音 k の振幅は k の 2 倍ごとに rolloff_db_oct dB ずつ変わる。"""
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(seed)
    y = np.zeros_like(t)
    k = 1
    while k * f0 < sr / 2 * 0.95:
        phase = rng.uniform(0, 2 * np.pi) if seed is not None else 0.0
        y += 10 ** (rolloff_db_oct * np.log2(k) / 20) * np.sin(2 * np.pi * k * f0 * t + phase)
        k += 1
    return (amp * y / np.max(np.abs(y))).astype(np.float32)


def pulse_train(f0: float, dur: float = 1.5, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """声帯の振動に似たパルス列 (1 周期に 1 回の鋭いパルス) を 1 次ローパスでなめらかにしたもの。"""
    n = int(dur * sr)
    y = np.zeros(n)
    period = sr / f0
    for i in range(int(n / period)):
        y[int(round(i * period))] = 1.0
    out = np.zeros(n)
    a = 0.9  # 1 次ローパス (声道の丸め)
    acc = 0.0
    for i in range(n):
        acc = a * acc + (1 - a) * y[i]
        out[i] = acc
    return (amp * out / np.max(np.abs(out))).astype(np.float32)


def gliding_tone(f0: float, depth_st: float, dur: float = 4.0, mod_hz: float = 0.5, sr: int = SR) -> np.ndarray:
    """F0 が中心 f0 の周りを ±depth_st 半音でゆっくり揺れる倍音の音 (IQR の検証用)。"""
    t = np.arange(int(dur * sr)) / sr
    f = f0 * 2.0 ** (depth_st * np.sin(2 * np.pi * mod_hz * t) / 12.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros_like(t)
    for k in range(1, 12):
        y += (1.0 / k) * np.sin(k * phase)
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


def syllable_bursts(n: int, period: float, dur: float, f0: float = 150.0, sr: int = SR) -> np.ndarray:
    """音節っぽい音量の山を n 個、period 秒おきに並べた信号 (各山は dur 秒)。"""
    y = np.zeros(int((n * period + 0.3) * sr), dtype=np.float32)
    tone = harmonic_tone(f0, dur=dur, sr=sr)
    env = np.hanning(len(tone)).astype(np.float32)
    for i in range(n):
        a = int((0.15 + i * period) * sr)
        y[a:a + len(tone)] += env * tone
    return y


def white_noise(dur: float = 3.0, sr: int = SR, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(dur * sr))).astype(np.float32)


def pink_noise(dur: float = 3.0, sr: int = SR, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    w = np.random.default_rng(seed).standard_normal(int(dur * sr))
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(len(w), 1 / sr)
    f[0] = 1.0
    p = np.fft.irfft(spec / np.sqrt(f), n=len(w))
    return (amp * p / np.std(p)).astype(np.float32)


def voice_like(speaker: int, noise_seed: int, dur: float = 2.0, sr: int = SR) -> np.ndarray:
    """『話者』= 基本周波数 + 倍音の強さのパターン(音色)。同じ speaker なら似た音になる。"""
    rng = np.random.default_rng(1000 + speaker)
    f0 = 90 + 25 * speaker
    gains = rng.uniform(0.2, 1.0, size=40)
    t = np.arange(int(dur * sr)) / sr
    nrng = np.random.default_rng(noise_seed)
    y = np.zeros_like(t)
    for k in range(1, 40):
        if k * f0 >= sr / 2 * 0.95:
            break
        y += gains[k] / k * np.sin(2 * np.pi * k * f0 * t + nrng.uniform(0, 2 * np.pi))
    y = y / np.max(np.abs(y)) * 0.3 + 0.002 * nrng.standard_normal(len(t))
    return y.astype(np.float32)


class FakeEmbedder(Embedder):
    """モデル無しで動くダミー埋め込み (平均 log スペクトルを 64 帯域にしたもの)。"""

    name = "fake"

    def _load(self) -> None:
        pass

    def _embed_chunk(self, y: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(y[: SR * 5], n=4096))[:2048]
        bands = np.log1p(100 * spec.reshape(64, -1).mean(axis=1))
        return bands - bands.mean()
