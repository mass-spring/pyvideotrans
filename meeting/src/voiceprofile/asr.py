"""任意の ASR (faster-whisper) で各発話区間を書き起こし、台本との対応づけを確かめる。

faster-whisper は既定では入れていない (重いため)。使うときだけ `uv sync --group asr`。
音声は手元のパソコンの中だけで処理する。ネットにつなぐのは、初回に ASR のモデルファイルをダウンロードするときだけ。
"""
from __future__ import annotations

import numpy as np


class AsrUnavailable(RuntimeError):
    """ASR が使えない (未インストール・モデルを取得できない)。理由は日本語。"""


class WhisperAsr:
    name = "whisper"

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise AsrUnavailable(
                "faster-whisper が入っていません。`uv sync --group asr` を実行してください（または --asr none で ASR なしに）"
            ) from e
        try:
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:  # noqa: BLE001 - ネットワーク・ディスクなど
            raise AsrUnavailable(
                f"ASR のモデル '{self.model_size}' を読み込めません ({type(e).__name__}: {str(e)[:200]})。初回はネット接続が必要です"
            ) from e

    def transcribe(self, y16: np.ndarray, lang: str) -> str:
        """16 kHz モノラル波形を文字にする。"""
        self.load()
        segments, _info = self._model.transcribe(
            np.asarray(y16, dtype=np.float32), language=lang or None, beam_size=1, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
        )
        return "".join(s.text for s in segments).strip()


def make_asr(kind: str, model_size: str = "small"):
    if kind in ("", "none"):
        return None
    if kind == "whisper":
        asr = WhisperAsr(model_size)
        asr.load()
        return asr
    raise ValueError(f"--asr は none / whisper のどちらかです: {kind}")
