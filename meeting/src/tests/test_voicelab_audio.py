"""voicelab：音声の読み書き・リサンプル・速度変更。"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

import voicelab.audio as va
from synth import SR, harmonic_tone, voice_like


@pytest.mark.parametrize("factor", [1.3, 2.0])
def test_time_stretch_changes_duration_by_the_factor(factor):
    y = voice_like(0, 1, dur=3.0)
    out, method = va.time_stretch(y, SR, factor, "librosa")
    assert method == "librosa" and len(out) / len(y) == pytest.approx(1 / factor, rel=0.03)


def test_time_stretch_keeps_pitch():
    y = harmonic_tone(150.0, dur=2.0)
    out, _ = va.time_stretch(y, SR, 1.3, "librosa")

    def peak(x):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        return np.fft.rfftfreq(len(x), 1 / SR)[1 + int(np.argmax(spec[1:400]))]

    assert peak(out) == pytest.approx(peak(y), rel=0.04)


def test_auto_falls_back_to_librosa_without_rubberband(monkeypatch):
    monkeypatch.setattr(va.shutil, "which", lambda name: None)
    y = voice_like(0, 1, dur=1.5)
    out, method = va.time_stretch(y, SR, 1.3, "auto")
    assert method == "librosa" and len(out) < len(y)
    with pytest.raises(RuntimeError):
        va.time_stretch(y, SR, 1.3, "rubberband")


@pytest.mark.skipif(not va.rubberband_available(), reason="rubberband コマンドが無い")
def test_rubberband_when_available():
    y = voice_like(0, 1, dur=2.0)
    out, method = va.time_stretch(y, SR, 2.0, "auto")
    assert method == "rubberband" and len(out) / len(y) == pytest.approx(0.5, rel=0.05)


def test_stretch_factor_one_is_noop_and_zero_is_error():
    y = voice_like(0, 1, dur=1.0)
    out, method = va.time_stretch(y, SR, 1.0)
    assert method == "none" and np.array_equal(out, y)
    with pytest.raises(ValueError):
        va.time_stretch(y, SR, 0)


def test_resample_and_write_wav_roundtrip(tmp_path):
    y = voice_like(0, 1, dur=1.0)
    y24 = va.resample(y, SR, 24000)
    assert len(y24) == pytest.approx(24000, abs=2)
    p = va.write_wav(tmp_path / "a" / "x.wav", y24 * 5.0, 24000)  # 1.0 を超える入力は割れないように縮める
    back, sr = va.read_wav(p)
    assert sr == 24000 and np.max(np.abs(back)) <= 1.0
    assert va.duration_of(p) == (pytest.approx(1.0, abs=0.01), 24000)


def test_read_wav_stereo_is_averaged_and_missing_is_error(tmp_path):
    st = np.stack([voice_like(0, 1, dur=1.0), voice_like(0, 1, dur=1.0)], axis=1)
    sf.write(tmp_path / "s.wav", st, SR)
    y, sr = va.read_wav(tmp_path / "s.wav")
    assert y.ndim == 1 and sr == SR
    with pytest.raises(FileNotFoundError):
        va.read_wav(tmp_path / "none.wav")


def test_pitch_shift_moves_the_pitch():
    y = harmonic_tone(150.0, dur=1.5)
    up = va.pitch_shift(y, SR, 12.0)

    def peak(x):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        return np.fft.rfftfreq(len(x), 1 / SR)[1 + int(np.argmax(spec[1:600]))]

    assert peak(up) == pytest.approx(2 * peak(y), rel=0.06)
