"""音声ファイルの読み込みとリサンプル (モノラル化・16kHz 化)。"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000  # 特徴量・埋め込みの内部サンプルレート

# フォルダ内を探すときに音声とみなす拡張子
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """音声を読み、(モノラル float32, 元のサンプルレート) を返す。

    ステレオは左右の平均。soundfile で読めない形式 (m4a など) は librosa に任せる。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {p}")
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)  # (サンプル, チャンネル)
    except Exception as e_sf:  # noqa: BLE001 - 読めない理由は何でも librosa に再挑戦
        try:
            import librosa

            data, sr = librosa.load(str(p), sr=None, mono=False)
            data = np.atleast_2d(data).T
        except Exception as e_lb:  # noqa: BLE001
            raise ValueError(f"音声を読めません ({p.name}): {e_sf} / {e_lb}") from e_lb
    if data.shape[0] == 0:
        raise ValueError(f"音声が空です: {p.name}")
    y = np.nan_to_num(data.mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y, int(sr)


def resample(y: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """多相フィルタでリサンプル。同じレートならそのまま返す。"""
    if sr == target_sr:
        return y.astype(np.float32, copy=False)
    g = gcd(int(sr), int(target_sr))
    out = resample_poly(y, target_sr // g, sr // g)
    return out.astype(np.float32)


def list_audio_files(folder: str | Path) -> list[Path]:
    """フォルダ直下の音声ファイルを名前順で返す (サブフォルダは見ない)。"""
    return sorted(
        p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
