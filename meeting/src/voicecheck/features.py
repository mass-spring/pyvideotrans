"""声の特徴量 (声の高さ・音量・スペクトルの傾き・CPPS・話速の目安)。

入力は「モノラルの波形」。内部で 16 kHz にそろえてから測る。
どの値も「同じ条件で測った値どうしを比べる」ための目安であり、絶対的な正解ではない。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from .audio import TARGET_SR, resample

# ---- 設定値 -----------------------------------------------------------------
HOP = 160            # 10 ms (16 kHz)
FRAME = 400          # 25 ms
N_FFT = 2048         # 長時間平均スペクトル用 (周波数の刻み 7.8 Hz)
F0_MIN, F0_MAX = 70.0, 500.0   # 声の高さを探す範囲 (Hz)
SILENCE_REL_DB = 30.0          # 「発話フレーム」: 上位5%の音量から 30 dB 以内
SILENCE_FLOOR_DB = -70.0       # これより小さい音は無音とみなす (dBFS)
TILT_RANGE = (100.0, 5000.0)   # スペクトルの傾きを回帰する範囲
SHORT_SEC = 1.0                # これ未満は「短い音声」として注意 (警告は analysis.py が出す)
MIN_VOICED_FRAMES = 10         # 声の高さを出すのに必要な有声フレーム数 (0.1 秒)

# 内部で使う特徴量の名前 (Excel の見出しは report.py で日本語にする)
FEATURE_KEYS = [
    "duration_s",
    "f0_median_hz",
    "f0_iqr_st",
    "voiced_ratio",
    "loudness_dbfs",
    "tilt_db_oct",
    "centroid_hz",
    "cpps_db",
    "rate_syl_per_voiced_s",
]


@dataclass
class Frames:
    """フレームごとの音量と「発話フレームか」の判定。"""

    db: np.ndarray
    active: np.ndarray


def _frame_rms(y: np.ndarray) -> np.ndarray:
    """25 ms 窓・10 ms 間隔の RMS (線形)。フレーム数は 1 + len//HOP。"""
    pad = FRAME // 2
    yp = np.pad(y, (pad, pad), mode="constant")
    n = 1 + len(y) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    idx = np.minimum(idx, len(yp) - 1)
    seg = yp[idx]
    return np.sqrt(np.mean(seg.astype(np.float64) ** 2, axis=1))


def analyze_frames(y: np.ndarray) -> Frames:
    """フレームごとの dBFS と、発話フレームの判定を作る。"""
    rms = _frame_rms(y)
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    ref = float(np.percentile(db, 95))
    if ref < SILENCE_FLOOR_DB:
        return Frames(db, np.zeros_like(db, dtype=bool))
    thr = max(ref - SILENCE_REL_DB, SILENCE_FLOOR_DB)
    return Frames(db, db > thr)


# ---- 声の高さ ----------------------------------------------------------------
def extract_f0(y: np.ndarray, sr: int = TARGET_SR, method: str = "harvest") -> tuple[np.ndarray, str]:
    """F0 (Hz) を 10 ms ごとに返す。無声は 0。pyworld harvest が使えなければ librosa.pyin。"""
    if method == "harvest":
        try:
            with warnings.catch_warnings():  # pyworld が出す pkg_resources の警告を隠す
                warnings.simplefilter("ignore")
                import pyworld as pw

            f0, _t = pw.harvest(
                y.astype(np.float64), sr, f0_floor=F0_MIN, f0_ceil=F0_MAX, frame_period=1000.0 * HOP / sr
            )
            return np.asarray(f0, dtype=np.float64), "harvest"
        except Exception:  # noqa: BLE001 - 入っていない / 失敗 → pyin
            pass
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, _p = librosa.pyin(
            y.astype(np.float32), fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=1024, hop_length=HOP
        )
    f0 = np.where(voiced & np.isfinite(f0), f0, 0.0)
    return f0.astype(np.float64), "pyin"


# ---- 長時間平均スペクトル (LTAS) ----------------------------------------------
def _ltas(y: np.ndarray, active: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """発話フレームだけの平均パワー・平均振幅スペクトルと周波数軸を返す。"""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann", center=True))  # (bin, frame)
    n = min(S.shape[1], len(active))
    mask = active[:n]
    if mask.sum() < 3:
        return None
    S = S[:, :n][:, mask]
    freqs = np.arange(S.shape[0]) * sr / N_FFT
    return (S**2).mean(axis=1), S.mean(axis=1), freqs


def spectral_tilt(power: np.ndarray, freqs: np.ndarray) -> float:
    """100 Hz〜5 kHz の傾き (dB/オクターブ)。1/3 オクターブ帯ごとの平均パワーを直線で近似する。"""
    lo, hi = TILT_RANGE
    n_bands = int(np.floor(3 * np.log2(hi / lo)))
    edges = lo * 2.0 ** (np.arange(n_bands + 1) / 3.0)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        sel = (freqs >= a) & (freqs < b)
        if not sel.any():
            continue
        p = float(power[sel].mean())
        if p <= 0:
            continue
        xs.append(np.log2(np.sqrt(a * b)))
        ys.append(10.0 * np.log10(p))
    if len(xs) < 4:
        return float("nan")
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def spectral_centroid(mag: np.ndarray, freqs: np.ndarray, fmin: float = 100.0) -> float:
    """平均振幅スペクトルの重心 (Hz)。fmin〜ナイキスト。"""
    sel = freqs >= fmin
    w = mag[sel]
    if w.sum() <= 0:
        return float("nan")
    return float((freqs[sel] * w).sum() / w.sum())


# ---- CPPS (声質の明瞭さ) --------------------------------------------------------
def compute_cpps(y: np.ndarray, sr: int = TARGET_SR) -> float:
    """Praat (parselmouth) の CPPS (dB)。使えない・失敗したら nan。"""
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:  # noqa: BLE001
        return float("nan")
    if len(y) < sr * 0.3:
        return float("nan")
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        cepstrogram = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        # 傾きを引く・時間平滑 0.02 s・ケフレンシー平滑 0.0005 s・探索 60-400 Hz
        val = call(cepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 400, 0.05, "Parabolic", 0.001, 0, "Straight", "Robust")
        return float(val) if np.isfinite(val) else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


# ---- 話速の目安 ---------------------------------------------------------------
def count_syllable_nuclei(y: np.ndarray, f0: np.ndarray, frames: Frames, sr: int = TARGET_SR) -> int:
    """音節の核 (母音のあたりの音量の山) の数。有声で発話中の山だけ数える。"""
    sos = butter(4, [250.0, 3500.0], btype="band", fs=sr, output="sos")
    yb = sosfiltfilt(sos, y.astype(np.float64))
    env = 20.0 * np.log10(np.maximum(_frame_rms(yb), 1e-10))
    env = uniform_filter1d(env, size=5, mode="nearest")  # 50 ms でなめらかに
    n = min(len(env), len(f0), len(frames.active))
    if n < 3:
        return 0
    env = env[:n]
    peaks, _ = find_peaks(env, prominence=3.0, distance=10)  # 谷との差 3 dB 以上、100 ms 以上あける
    keep = (f0[:n][peaks] > 0) & frames.active[:n][peaks]
    return int(keep.sum())


# ---- まとめ -------------------------------------------------------------------
def extract_features(y: np.ndarray, sr: int, f0_method: str = "harvest") -> tuple[dict, list[str]]:
    """1つの音声 (モノラル波形) から特徴量の辞書と備考のリストを返す。"""
    notes: list[str] = []
    feats: dict[str, Any] = {k: float("nan") for k in FEATURE_KEYS}
    feats["duration_s"] = len(y) / float(sr)
    feats["n_syllables"] = float("nan")
    feats["f0_method"] = ""  # 使った F0 抽出法 (harvest / pyin)

    y = resample(y, sr, TARGET_SR)
    y = y - float(np.mean(y))  # 直流成分を除く
    frames = analyze_frames(y)
    if not frames.active.any():
        notes.append("無音、または音量が小さすぎて測れません")
        return feats, notes

    # 音量: 発話フレームの RMS (dBFS)
    rms = _frame_rms(y)
    feats["loudness_dbfs"] = float(10.0 * np.log10(np.mean(rms[frames.active] ** 2) + 1e-20))

    # 声の高さ
    f0, used = extract_f0(y, TARGET_SR, f0_method)
    feats["f0_method"] = used
    n = min(len(f0), len(frames.active))
    voiced = (f0[:n] > 0) & frames.active[:n]
    feats["voiced_ratio"] = float(voiced.sum() / max(n, 1))
    if voiced.sum() >= MIN_VOICED_FRAMES:
        f0v = f0[:n][voiced]
        med = float(np.median(f0v))
        st = 12.0 * np.log2(f0v / med)
        feats["f0_median_hz"] = med
        feats["f0_iqr_st"] = float(np.percentile(st, 75) - np.percentile(st, 25))
        voiced_sec = voiced.sum() * HOP / TARGET_SR
        n_syl = count_syllable_nuclei(y, f0, frames, TARGET_SR)
        feats["n_syllables"] = float(n_syl)
        feats["rate_syl_per_voiced_s"] = n_syl / voiced_sec if voiced_sec >= 0.3 else float("nan")
    else:
        notes.append("声のある区間 (有声音) が短く、声の高さ・話速は出せません")

    # 長時間平均スペクトル → 傾き・重心
    lt = _ltas(y, frames.active, TARGET_SR)
    if lt is not None:
        power, mag, freqs = lt
        feats["tilt_db_oct"] = spectral_tilt(power, freqs)
        feats["centroid_hz"] = spectral_centroid(mag, freqs)

    # CPPS: 前後の無音を除いた波形で測る
    act_idx = np.flatnonzero(frames.active)
    a, b = act_idx[0] * HOP, min((act_idx[-1] + 1) * HOP, len(y))
    feats["cpps_db"] = compute_cpps(y[a:b], TARGET_SR)
    return feats, notes


def backend_status() -> dict[str, bool]:
    """pyworld / parselmouth が使えるか (使えない時は代用手段で動く)。"""
    status = {}
    for mod in ("pyworld", "parselmouth"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                __import__(mod)
            status[mod] = True
        except Exception:  # noqa: BLE001
            status[mod] = False
    return status
