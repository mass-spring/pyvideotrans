"""特徴量: 既知の F0・音量・傾き・音節数を持つ合成信号で検証する。"""
from __future__ import annotations

import numpy as np
import pytest

from synth import (SR, gliding_tone, harmonic_tone, pink_noise, pulse_train, syllable_bursts, white_noise)
from voicecheck.analysis import Analyzer
from voicecheck.audio import load_audio, resample
from voicecheck.features import FEATURE_KEYS, extract_features


def feats(y, sr=SR, **kw):
    f, notes = extract_features(y, sr, **kw)
    return f


@pytest.mark.parametrize("f0", [120.0, 220.0])
@pytest.mark.parametrize("maker", [harmonic_tone, pulse_train], ids=["倍音", "パルス列"])
def test_f0_median_within_3_percent(maker, f0):
    f = feats(maker(f0))
    assert f["f0_median_hz"] == pytest.approx(f0, rel=0.03)
    assert f["f0_method"] == "harvest"
    assert f["voiced_ratio"] > 0.9


def test_f0_pyin_fallback_also_works():
    """pyworld が無い環境の代用 (librosa.pyin)。"""
    f = feats(harmonic_tone(150.0, dur=1.0), f0_method="pyin")
    assert f["f0_method"] == "pyin"
    assert f["f0_median_hz"] == pytest.approx(150.0, rel=0.03)


def test_f0_falls_back_to_pyin_when_pyworld_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyworld":
            raise ImportError("pyworld がありません (テスト)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    f = feats(harmonic_tone(150.0, dur=1.0))
    assert f["f0_method"] == "pyin"
    assert f["f0_median_hz"] == pytest.approx(150.0, rel=0.03)


@pytest.mark.parametrize("depth", [1.0, 2.0, 3.0])
def test_f0_iqr_semitones(depth):
    """F0 を ±depth 半音で正弦波状に揺らすと IQR は約 1.414*depth 半音。"""
    f = feats(gliding_tone(150.0, depth))
    assert f["f0_iqr_st"] == pytest.approx(1.414 * depth, abs=0.3)
    assert f["f0_median_hz"] == pytest.approx(150.0, rel=0.03)


def test_loudness_is_speech_level_not_diluted_by_silence():
    t = np.arange(SR * 2) / SR
    sine = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    expected = 20 * np.log10(0.1 / np.sqrt(2))  # 約 -23.0 dBFS
    assert feats(sine)["loudness_dbfs"] == pytest.approx(expected, abs=0.3)
    padded = np.concatenate([sine, np.zeros(SR * 2, dtype=np.float32)])  # 無音を足しても変わらない
    assert feats(padded)["loudness_dbfs"] == pytest.approx(expected, abs=0.5)
    half = feats(sine * 0.5)["loudness_dbfs"]  # 振幅半分 = -6 dB
    assert half == pytest.approx(expected - 6.02, abs=0.3)


def test_voiced_ratio_with_silence():
    y = np.concatenate([harmonic_tone(150.0, dur=1.0), np.zeros(SR, dtype=np.float32)])
    assert feats(y)["voiced_ratio"] == pytest.approx(0.5, abs=0.06)


def test_spectral_tilt_sign_and_values():
    white = feats(white_noise())["tilt_db_oct"]
    pink = feats(pink_noise())["tilt_db_oct"]
    assert abs(white) < 0.7                      # 白色雑音は平ら (0 dB/oct)
    assert pink == pytest.approx(-3.0, abs=0.7)  # ピンクノイズは -3 dB/oct
    assert pink < white                          # 高音が弱い方がマイナス
    bright = np.diff(white_noise(), prepend=0).astype(np.float32)  # 微分 = 高音を強調
    assert feats(bright)["tilt_db_oct"] > white + 3  # 高音が強いとプラスに寄る


def test_tilt_and_centroid_follow_harmonic_rolloff():
    steep = feats(harmonic_tone(150.0, rolloff_db_oct=-12.0))
    flat = feats(harmonic_tone(150.0, rolloff_db_oct=0.0))
    assert steep["tilt_db_oct"] < flat["tilt_db_oct"] - 5
    assert steep["centroid_hz"] < flat["centroid_hz"]


def test_cpps_clean_vs_noisy():
    clean = feats(harmonic_tone(150.0, dur=2.0))["cpps_db"]
    noisy = feats(harmonic_tone(150.0, dur=2.0) + white_noise(2.0, amp=0.15))["cpps_db"]
    assert np.isfinite(clean) and np.isfinite(noisy)
    assert clean > noisy


def test_syllable_nuclei_count_and_rate_ratio():
    f = feats(syllable_bursts(10, period=0.25, dur=0.16))
    assert f["n_syllables"] == pytest.approx(10, abs=1)
    fast = feats(syllable_bursts(16, period=0.15, dur=0.10))["rate_syl_per_voiced_s"]
    slow = feats(syllable_bursts(8, period=0.30, dur=0.20))["rate_syl_per_voiced_s"]
    assert 1.4 < fast / slow < 2.3   # 本当は 2 倍
    steady = feats(harmonic_tone(150.0))  # 山が無い (一定の音) → 0 に近い
    assert steady["n_syllables"] <= 1


def test_silent_and_tiny_inputs_do_not_crash():
    f = feats(np.zeros(SR, dtype=np.float32))
    assert set(FEATURE_KEYS) <= set(f)
    assert np.isnan(f["f0_median_hz"]) and np.isnan(f["loudness_dbfs"])
    f2 = feats(np.zeros(10, dtype=np.float32))
    assert np.isnan(f2["f0_median_hz"])


# ---- 読み込み・リサンプル・Analyzer ---------------------------------------------------
def test_load_audio_mono_stereo_and_sample_rates(write_wav):
    y = harmonic_tone(150.0, dur=1.0, sr=44100)
    mono = write_wav("mono44.wav", y, 44100)
    stereo = write_wav("stereo22.wav", np.stack([harmonic_tone(150.0, 1.0, 22050), 0.5 * harmonic_tone(150.0, 1.0, 22050)], 1), 22050)
    ym, srm = load_audio(mono)
    ys, srs = load_audio(stereo)
    assert srm == 44100 and ym.ndim == 1
    assert srs == 22050 and ys.ndim == 1 and len(ys) == 22050
    assert len(resample(ym, srm)) == pytest.approx(16000, abs=2)
    for p in (mono, stereo):
        c = Analyzer(verbose=False).get(str(p))
        assert c.ok
        assert c.features["f0_median_hz"] == pytest.approx(150.0, rel=0.03)
        assert c.duration_s == pytest.approx(1.0, abs=0.01)


def test_analyzer_short_clip_missing_file_and_low_samplerate(write_wav, tmp_path):
    an = Analyzer(verbose=False)
    short = an.get(str(write_wav("short.wav", harmonic_tone(150.0, dur=0.5))))
    assert short.ok and any("短い" in n for n in short.notes)
    missing = an.get(str(tmp_path / "nothing.wav"))
    assert not missing.ok and "見つかりません" in missing.error
    low = an.get(str(write_wav("tel.wav", harmonic_tone(150.0, dur=1.5, sr=8000), 8000)))
    assert low.ok and any("サンプルレート" in n for n in low.notes)


def test_analyzer_broken_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio")
    c = Analyzer(verbose=False).get(str(bad))
    assert not c.ok and c.error


def test_clipping_note(write_wav):
    y = np.clip(harmonic_tone(150.0, dur=1.5) * 8, -1, 1).astype(np.float32)
    c = Analyzer(verbose=False).get(str(write_wav("clip.wav", y)))
    assert any("クリッピング" in n for n in c.notes)
