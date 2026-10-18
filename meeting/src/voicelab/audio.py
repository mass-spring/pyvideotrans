"""音声ファイルの読み書き・リサンプル・速度変更（numpy / soundfile / scipy だけで動く部分が中心）。

librosa と rubberband は「あれば使う」。無くても、読み書き・リサンプルは動く。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """音声を読み、(モノラル float32, サンプルレート) を返す。ステレオは左右の平均。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {p}")
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - m4a / mp3 など soundfile で読めない形式は librosa に任せる
        import librosa

        y, sr = librosa.load(str(p), sr=None, mono=False)
        data = np.atleast_2d(y).T
    if data.shape[0] == 0:
        raise ValueError(f"音声が空です: {p.name}")
    y = np.nan_to_num(data.mean(axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y, int(sr)


def duration_of(path: str | Path) -> tuple[float, int]:
    """(長さ[秒], サンプルレート)。中身を全部読まずに調べる。"""
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate)
    except Exception:  # noqa: BLE001
        y, sr = read_wav(path)
        return len(y) / sr, sr


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> Path:
    """16bit PCM の wav に書く。割れ（1.0 超え）は全体を縮めて防ぐ。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.999
    sf.write(str(p), y, int(sr), subtype="PCM_16")
    return p


def resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """多相フィルタでリサンプル。同じレートならそのまま返す。"""
    if int(sr) == int(target_sr):
        return np.asarray(y, dtype=np.float32)
    from scipy.signal import resample_poly

    g = gcd(int(sr), int(target_sr))
    return resample_poly(y, int(target_sr) // g, int(sr) // g).astype(np.float32)


def trim_to(y: np.ndarray, sr: int, max_s: float) -> np.ndarray:
    """先頭から max_s 秒だけ残す。"""
    n = int(max_s * sr)
    return y[:n] if len(y) > n else y


# ---------------------------------------------------------------------------
# 速度変更（声の高さは変えない）
# ---------------------------------------------------------------------------
def rubberband_available() -> bool:
    return shutil.which("rubberband") is not None


def _stretch_rubberband(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """rubberband コマンド（-T：テンポを factor 倍）。"""
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), y, sr, subtype="FLOAT")
        subprocess.run(
            ["rubberband", "-q", "-T", f"{factor:.6f}", str(src), str(dst)],
            check=True, capture_output=True, text=True, timeout=300,
        )
        out, _ = sf.read(str(dst), dtype="float32", always_2d=False)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def _stretch_librosa(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=float(factor)).astype(np.float32)


def time_stretch(y: np.ndarray, sr: int, factor: float, method: str = "auto") -> tuple[np.ndarray, str]:
    """factor 倍の速さにする（1.3 なら長さは 1/1.3）。(波形, 使った方式) を返す。

    method: "auto"（rubberband があれば使い、無ければ librosa）/ "rubberband" / "librosa"
    """
    if factor <= 0:
        raise ValueError(f"速度の倍率は正の数にしてください: {factor}")
    if abs(factor - 1.0) < 1e-9:
        return np.asarray(y, dtype=np.float32), "none"
    if method in ("auto", "rubberband") and rubberband_available():
        try:
            return _stretch_rubberband(y, sr, factor), "rubberband"
        except Exception as e:  # noqa: BLE001
            if method == "rubberband":
                raise RuntimeError(f"rubberband での速度変更に失敗しました: {e}") from e
    elif method == "rubberband":
        raise RuntimeError("rubberband コマンドが見つかりません（Colab: apt-get install rubberband-cli）")
    return _stretch_librosa(y, sr, factor), "librosa"


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """声の高さだけを semitones 半音ずらす（--dry-run のダミー用）。librosa が無ければ、速さごと変わる簡易版。"""
    if abs(semitones) < 1e-6:
        return np.asarray(y, dtype=np.float32)
    try:
        import librosa

        return librosa.effects.pitch_shift(np.asarray(y, dtype=np.float32), sr=sr, n_steps=float(semitones)).astype(np.float32)
    except ImportError:
        ratio = 2.0 ** (semitones / 12.0)
        return resample(y, int(round(sr * ratio)), sr)
