"""発話区間の検出・SNR・音割れ・切り出し範囲 のテスト (合成音声)。"""
from __future__ import annotations

import numpy as np
import pytest

from vp_synth import SR, make_recording
from voicecheck.audio import TARGET_SR, resample
from voiceprofile.clips import assemble, clip_pieces, pieces_duration
from voiceprofile.quality import build_frame_info, clipping_ratio, measure_quality, noise_drift_db, peak_dbfs
from voiceprofile.segment import NoSpeechError, Segment, detect_segments, voiced_fraction


def _detect(rec, **kw):
    return detect_segments(resample(rec.y, rec.sr, TARGET_SR), **kw)


def test_detects_every_segment_within_150ms():
    rec = make_recording([2.0, 3.5, 1.2, 4.0, 2.7], pauses=[1.0, 0.9, 1.4, 1.1], noise_amps=0.002)
    seg = _detect(rec)
    assert len(seg.segments) == 5
    for found, (t0, t1) in zip(seg.segments, rec.segments, strict=True):
        assert found.start == pytest.approx(t0, abs=0.15)
        assert found.end == pytest.approx(t1, abs=0.15)
    assert seg.vad_used in ("voicing+webrtcvad", "voicing")
    assert seg.on_db > seg.noise_db and seg.off_db < seg.on_db


def test_short_dips_inside_a_segment_do_not_split_it():
    # 合成の発話区間は音のあいだに 20〜80 ms のへこみがある。それで分かれたら 1 区間にならない
    rec = make_recording([6.0], noise_amps=0.001)
    assert len(_detect(rec).segments) == 1


def test_pause_threshold_min_pause_controls_splitting():
    rec = make_recording([2.0, 2.0], pauses=0.5, noise_amps=0.001)
    assert len(_detect(rec, min_pause=0.35).segments) == 2   # 0.5 秒の間は分ける
    assert len(_detect(rec, min_pause=0.8).segments) == 1    # 0.8 秒未満の間は分けない
    rec2 = make_recording([2.0, 2.0], pauses=0.2, noise_amps=0.001)
    assert len(_detect(rec2, min_pause=0.35).segments) == 1  # 0.2 秒の間は分けない


def test_click_is_rejected_and_speech_kept():
    rec = make_recording([3.0, 3.0], pauses=2.0, noise_amps=0.001)
    y = rec.y.copy()
    i = int((rec.segments[0][1] + 1.0) * SR)             # 2 つの発話のあいだに 0.06 秒の短い音 (クリック)
    n = int(0.06 * SR)
    y[i:i + n] += (0.3 * np.hanning(n) * np.sin(2 * np.pi * 1500 * np.arange(n) / SR)).astype(np.float32)
    seg = detect_segments(resample(y, SR, TARGET_SR))
    assert len(seg.segments) == 2
    assert any("短すぎる" in r.reason for r in seg.rejected)


@pytest.mark.parametrize("kind", ["white", "cough", "thud"])
def test_loud_non_voice_sounds_are_rejected_by_the_voicing_check(kind):
    """咳・物音のように大きいが周期的でない音は、声の区間にしない (webrtcvad は大きな雑音を声と判定しがちなので、周期性で見る)。"""
    from scipy.signal import butter, sosfilt

    rec = make_recording([3.0, 3.0], pauses=2.5, noise_amps=0.001)
    rng = np.random.default_rng(1)
    n = int(0.5 * SR)
    if kind == "white":
        burst = rng.standard_normal(n) * np.hanning(n)
    elif kind == "cough":
        burst = sosfilt(butter(4, [300, 3000], btype="band", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.12 * SR))
    else:                                                    # 机を叩く音のような、低い音域の雑音が減衰していくもの
        burst = sosfilt(butter(4, 400, fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.1 * SR))
    burst = (0.15 * burst / np.max(np.abs(burst))).astype(np.float32)
    y = rec.y.copy()
    i = int((rec.segments[0][1] + 0.8) * SR)
    y[i:i + n] += burst
    y16 = resample(y, SR, TARGET_SR)
    assert len(detect_segments(y16, vad="energy").segments) == 3          # 音量だけだと声と区別できない
    seg = detect_segments(y16)
    assert len(seg.segments) == 2
    assert any("周期的" in r.reason for r in seg.rejected)


def test_voiced_fraction_separates_speech_from_noise():
    rec = make_recording([3.0], noise_amps=0.001)
    y16 = resample(rec.y, rec.sr, TARGET_SR)
    t0, t1 = rec.segments[0]
    assert voiced_fraction(y16, Segment(t0, t1)) > 0.6
    noise = (0.1 * np.random.default_rng(0).standard_normal(TARGET_SR * 2)).astype(np.float32)
    assert voiced_fraction(noise, Segment(0.0, 2.0)) < 0.05
    assert voiced_fraction(noise, Segment(0.0, 0.01)) is None            # 短すぎて測れない


def test_energy_only_and_webrtc_options():
    rec = make_recording([3.0, 3.0], pauses=1.0, noise_amps=0.002)
    y16 = resample(rec.y, rec.sr, TARGET_SR)
    assert detect_segments(y16, vad="energy").vad_used == "energy"
    assert detect_segments(y16, vad="webrtc").vad_used == "voicing+webrtcvad"
    with pytest.raises(ValueError):
        detect_segments(y16, vad="bogus")


def test_noise_only_or_silence_raises_no_speech():
    rng = np.random.default_rng(0)
    noise = (0.002 * rng.standard_normal(TARGET_SR * 5)).astype(np.float32)
    with pytest.raises(NoSpeechError):
        detect_segments(noise)
    with pytest.raises(NoSpeechError):
        detect_segments(np.zeros(TARGET_SR * 3, dtype=np.float32))


def test_quiet_recording_is_reported_as_too_quiet():
    rec = make_recording([3.0, 3.0], pauses=1.0, noise_amps=1e-6, amps=1e-4)
    with pytest.raises(NoSpeechError, match="小さ"):
        _detect(rec)


@pytest.mark.parametrize("noise", [0.0005, 0.002, 0.01, 0.03])
def test_snr_estimate_is_close_to_true_snr(noise):
    rec = make_recording([4.0, 4.0, 4.0], pauses=1.2, noise_amps=noise, seed=3)
    seg = _detect(rec)
    info = build_frame_info(seg)
    for k, s in enumerate(seg.segments):
        pieces = clip_pieces(seg.segments, k, k, rec.duration)
        q = measure_quality(info, [s], pieces)
        assert q.snr_db == pytest.approx(rec.true_snr_db(k), abs=3.0), (noise, k, q)


def test_snr_ordering_follows_noise_level():
    snrs = []
    for amp in (0.0005, 0.002, 0.008, 0.03):          # 録音ごとに雑音を 12 dB ずつ大きくする
        rec = make_recording([3.5] * 3, pauses=1.3, noise_amps=amp, seed=5)
        seg = _detect(rec)
        info = build_frame_info(seg)
        snrs.append(float(np.mean([measure_quality(info, [s], clip_pieces(seg.segments, k, k, rec.duration)).snr_db
                                   for k, s in enumerate(seg.segments)])))
    assert snrs == sorted(snrs, reverse=True)           # 雑音が大きいほど SNR は小さい
    assert snrs[0] - snrs[-1] > 25                       # 4 段階で 36 dB の差がある


def test_noise_drift_is_reported_when_the_noise_floor_changes_midway():
    rec = make_recording([3.0] * 9, pauses=1.2, noise_amps=[0.0008] * 5 + [0.004] * 4, seed=2)   # 途中で雑音が約 14 dB 上がる
    info = build_frame_info(_detect(rec))
    assert noise_drift_db(info) > 8.0
    steady = make_recording([3.0] * 9, pauses=1.2, noise_amps=0.001, seed=2)
    assert noise_drift_db(build_frame_info(_detect(steady))) < 3.0


def test_digital_silence_between_speech_is_flagged_as_possible_noise_gate():
    rec = make_recording([3.0, 3.0, 3.0], pauses=1.2, noise_amps=1e-9)    # 雑音がほぼ 0 (ノイズゲートをかけたような録音)
    seg = _detect(rec)
    info = build_frame_info(seg)
    q = measure_quality(info, seg.segments[:1], clip_pieces(seg.segments, 0, 0, rec.duration))
    assert any("ノイズゲート" in n for n in q.notes)
    assert q.snr_db <= 60.0


def test_clipping_ratio_counts_only_flat_tops():
    t = np.arange(SR) / SR
    y = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    assert clipping_ratio(y) == 0.0
    y_peak = y.copy()
    y_peak[1000] = 1.0                                    # 1 サンプルだけ最大値 (正規化した録音でよくある)。割れではない
    assert clipping_ratio(y_peak) == 0.0
    y_clip = np.clip(3.0 * np.sin(2 * np.pi * 200 * t), -1.0, 1.0).astype(np.float32)
    assert clipping_ratio(y_clip) > 0.2
    assert peak_dbfs(y_clip) == pytest.approx(0.0, abs=0.01)
    assert clipping_ratio(np.zeros(0, dtype=np.float32)) == 0.0


def test_clip_pieces_pads_stay_inside_pauses_and_long_gaps_are_shortened():
    segs = [Segment(1.0, 3.0), Segment(3.2, 5.0), Segment(9.0, 11.0), Segment(11.6, 12.0)]
    # 前の区間と 0.2 秒しか離れていない: 前の余白は間の真ん中 (3.1) までに収まり、後ろは 0.12 秒付ける
    p = clip_pieces(segs, 1, 1, 20.0)
    assert p == [(pytest.approx(3.1), pytest.approx(5.12))]
    # 間が十分に長ければ前後とも 0.12 秒
    assert clip_pieces(segs, 2, 2, 20.0)[0] == (pytest.approx(8.88), pytest.approx(11.12))
    # 5.0 → 9.0 の 4 秒の間は 0.8 秒に詰める (前後 0.4 秒ずつ残す)
    p = clip_pieces(segs, 1, 2, 20.0)
    assert len(p) == 2
    assert p[0][1] == pytest.approx(5.4) and p[1][0] == pytest.approx(8.6)
    assert pieces_duration(p) == pytest.approx((5.4 - 3.1) + (11.12 - 8.6))
    # 先頭では 0 秒より前に出ない、最後では録音の長さを超えない
    assert clip_pieces(segs, 0, 0, 20.0)[0][0] == pytest.approx(0.88)
    assert clip_pieces([Segment(0.05, 1.0)], 0, 0, 1.05)[0] == (0.0, pytest.approx(1.05))
    # 切り口 (範囲の端) はどの発話区間の中にも入らない
    for i0, i1 in [(0, 0), (1, 1), (1, 2), (2, 3), (0, 3)]:
        for a, b in clip_pieces(segs, i0, i1, 20.0):
            for sg in segs:
                assert not (sg.start < a < sg.end) and not (sg.start < b < sg.end)


def test_assemble_keeps_audio_untouched_except_short_fades():
    y = np.full(SR * 3, 0.25, dtype=np.float32)
    out = assemble(y, SR, [(0.5, 1.0), (2.0, 2.5)])
    n = int(0.5 * SR)
    assert len(out) == 2 * n
    for part in (out[:n], out[n:]):
        assert np.all(part[int(0.02 * SR):-int(0.02 * SR)] == 0.25)    # フェード以外は 1 サンプルも変えない
        assert part[0] == 0.0 and part[-1] == 0.0                      # 切り口はフェードで 0 になる (クリックを防ぐ)
    assert np.all(y == 0.25) and y.dtype == np.float32                 # 元の配列は壊さない
