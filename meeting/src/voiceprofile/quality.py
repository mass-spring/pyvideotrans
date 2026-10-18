"""手本クリップの品質: SNR (話し声 対 雑音の床)、音割れ (クリッピング)、音量。

SNR は 10 ms ごとの音量から求める:
  話し声 = 発話区間の中で、音が出ているフレームの平均パワー
  雑音   = クリップ内の「話していない所」のフレームのパワーの中央値 (息づかい・咳の外れ値に強い)
  SNR    = 10 log10((話し声 - 雑音) / 雑音)
クリップ内に話していない所が短すぎる時は、録音全体の雑音の床で代用する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .segment import FRAME_SEC, Segment, Segmentation

GUARD_SEC = 0.06          # 発話区間の前後これだけは「雑音」として数えない (音の余韻・息の立ち上がり)
MIN_NOISE_FRAMES = 15     # クリップ内の雑音フレームがこれ未満なら録音全体の値で代用 (0.15 秒)
SNR_MIN_DB, SNR_MAX_DB = -20.0, 60.0
SILENT_NOISE_DBFS = -85.0  # 雑音がこれより小さい = 完全な無音に近い (ノイズ除去・ノイズゲートの疑い)
CLIP_LEVEL = 0.999


@dataclass
class FrameInfo:
    """1 つの録音ファイルの 10 ms ごとの情報。"""

    power: np.ndarray            # フレームのパワー (線形)
    frame_db: np.ndarray
    times: np.ndarray            # フレーム中心の時刻 (秒)
    near_speech: np.ndarray      # どれかの発話区間とその前後 GUARD_SEC にあるフレーム
    off_db: float
    file_noise_power: float      # 録音全体の雑音パワー (話していないフレームの中央値)


@dataclass
class Quality:
    snr_db: float
    speech_dbfs: float
    noise_dbfs: float
    n_noise_frames: int
    notes: list[str] = field(default_factory=list)


def build_frame_info(seg: Segmentation) -> FrameInfo:
    db = seg.frame_db
    times = np.arange(len(db)) * FRAME_SEC
    atom = np.zeros(len(db), dtype=bool)
    for s in seg.segments:
        atom |= (times >= s.start) & (times < s.end)
    guard = max(1, int(round(GUARD_SEC / FRAME_SEC)))
    near = ndimage.binary_dilation(atom, structure=np.ones(2 * guard + 1, dtype=bool))
    power = 10.0 ** (db / 10.0)
    quiet = power[~near]
    if len(quiet) >= MIN_NOISE_FRAMES:
        noise = float(np.median(quiet))
    else:  # ほとんど間がない録音: 下位 5% のパワーで代用
        noise = float(np.percentile(power, 5))
    return FrameInfo(power, db, times, near, seg.off_db, max(noise, 1e-12))


def _in_ranges(times: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(times), dtype=bool)
    for a, b in ranges:
        m |= (times >= a) & (times < b)
    return m


def measure_quality(info: FrameInfo, atoms: list[Segment], pieces: list[tuple[float, float]]) -> Quality:
    """atoms: クリップに入っている発話区間。pieces: クリップとして切り出す時間範囲 (秒)。"""
    notes: list[str] = []
    speech = _in_ranges(info.times, [(a.start, a.end) for a in atoms]) & (info.frame_db >= info.off_db)
    if not speech.any():
        return Quality(float("nan"), float("nan"), float("nan"), 0, ["話し声のフレームが見つかりません"])
    ps = float(np.mean(info.power[speech]))
    noise = _in_ranges(info.times, pieces) & ~info.near_speech
    n_noise = int(noise.sum())
    if n_noise >= MIN_NOISE_FRAMES:
        pn = float(np.median(info.power[noise]))
    else:
        pn = info.file_noise_power
        notes.append("クリップ内に無音の部分が短いので、録音全体の雑音の値で SNR を出しました")
    pn = max(pn, 1e-12)
    snr = 10.0 * np.log10(max(ps - pn, 1e-3 * ps) / pn)
    snr = float(min(max(snr, SNR_MIN_DB), SNR_MAX_DB))
    noise_dbfs = float(10.0 * np.log10(pn))
    if noise_dbfs < SILENT_NOISE_DBFS:
        notes.append(f"無音部分が完全に無音に近い ({noise_dbfs:.0f} dBFS): ノイズ除去・ノイズゲートがかかっている可能性。"
                     "かけていると声の特徴が変わるので、外して録り直すのがおすすめです")
    return Quality(snr, float(10.0 * np.log10(ps)), noise_dbfs, n_noise, notes)


def clipping_ratio(y: np.ndarray) -> float:
    """音が割れている (振幅が上限に張り付いた) サンプルの割合。連続 3 サンプル以上の張り付きだけ数える。"""
    if len(y) == 0:
        return 0.0
    hit = np.abs(y) >= CLIP_LEVEL
    if not hit.any():
        return 0.0
    three = np.ones(3, dtype=np.int8)
    run_center = np.convolve(hit.astype(np.int8), three, mode="same") >= 3
    covered = np.convolve(run_center.astype(np.int8), three, mode="same") > 0
    return float(np.count_nonzero(covered & hit) / len(y))


def peak_dbfs(y: np.ndarray) -> float:
    p = float(np.max(np.abs(y))) if len(y) else 0.0
    return float(20.0 * np.log10(max(p, 1e-10)))
