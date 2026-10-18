from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from synth import SR


@pytest.fixture
def write_wav(tmp_path):
    """write_wav("a.wav", y, sr=16000) → 書き出した Path"""

    def _write(name: str, y: np.ndarray, sr: int = SR):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, y, sr)
        return p

    return _write
