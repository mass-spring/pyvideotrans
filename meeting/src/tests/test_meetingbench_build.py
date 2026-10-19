from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import soundfile as sf
from asrcheck.ioformats import read_utts_table
from mb_helpers import SPEAKERS, make_fake_jvnv
from meetingbench import build as B
from meetingbench.jvnv import JvnvError, load_index, parse_key
from synth import harmonic_tone


def small_params(**kw):
    base = dict(seed=7, minutes_min=0.12, minutes_max=0.16, lead_in=1.0, tail=1.0)
    base.update(kw)
    return B.BuildParams(**base)


@pytest.fixture
def jvnv(tmp_path):
    return make_fake_jvnv(tmp_path / "jvnv")


def test_trim_range_cuts_silence():
    tone = harmonic_tone(200.0, dur=1.0, sr=48000)
    y = np.concatenate([np.zeros(24000, np.float32), tone, np.zeros(24000, np.float32)])
    i0, i1 = B.trim_range(y, 48000, trim_db=35.0, pad_s=0.03)
    assert abs(i0 / 48000 - (0.5 - 0.03)) < 0.03
    assert abs(i1 / 48000 - (1.5 + 0.03)) < 0.03
    assert B.trim_range(np.zeros(4800, np.float32), 48000, 35.0, 0.03) in {(0, 4800)}   # 無音だけ → そのまま


def test_parse_key_and_index(jvnv):
    assert parse_key("anger_regular_01") == ("anger", "regular")
    with pytest.raises(JvnvError):
        parse_key("bad")
    clips = load_index(jvnv)
    assert len(clips) == 4 * 3 * 6
    assert {c.speaker for c in clips} == set(SPEAKERS)
    assert all(c.text.endswith("です。") and c.duration > 3 for c in clips)
    assert load_index(jvnv, speakers=["F1"])[0].speaker == "F1"


def test_load_index_errors(tmp_path):
    with pytest.raises(JvnvError):
        load_index(tmp_path)   # transcription_no_nv.csv が無い
    (tmp_path / "transcription_no_nv.csv").write_text("x_y_01|あ\n", encoding="utf-8")
    with pytest.raises(JvnvError):
        load_index(tmp_path)   # wav が無い


def test_plan_turns_unique_keys_budget_and_determinism(jvnv):
    clips = load_index(jvnv)
    p = small_params()
    cache = B.TrimCache(p)
    order, info = B.plan_turns(clips, p, duration_of=cache.duration_of)
    keys = [c.key for c in order]
    assert len(keys) == len(set(keys))                       # 同じ文は1回だけ
    for s in SPEAKERS:
        got = sum(cache.duration_of(c) for c in order if c.speaker == s)
        if s not in info["ran_out"]:
            assert got >= info["budget_min"][s] * 60 - 1e-6   # 持ち時間に達するまで選ぶ
    assert len({c.speaker for c in order}) == 4
    assert len({c.emotion for c in order}) >= 2              # 感情が混ざる
    order2, _ = B.plan_turns(clips, p, duration_of=B.TrimCache(p).duration_of)
    assert [c.key + c.speaker for c in order] == [c.key + c.speaker for c in order2]
    order3, _ = B.plan_turns(clips, small_params(seed=8), duration_of=B.TrimCache(small_params(seed=8)).duration_of)
    assert [c.key + c.speaker for c in order] != [c.key + c.speaker for c in order3]


def _placed(jvnv, p):
    clips = load_index(jvnv)
    cache = B.TrimCache(p)
    order, _ = B.plan_turns(clips, p, duration_of=cache.duration_of)
    return B.prepare_turns(order, cache)


def test_schedule_a_gaps_and_b_overlaps(jvnv):
    p = small_params()
    placed = _placed(jvnv, p)
    n = len(placed)
    assert n >= 8
    assert B.schedule(placed, p, "A") == []
    for a, b in zip(placed, placed[1:]):
        assert p.gap_min - 1e-6 <= b.start - a.end <= p.gap_max + 1e-6
    assert placed[0].start == p.lead_in
    overl = B.schedule(placed, p, "B")
    assert len(overl) == round(p.overlap_ratio * (n - 1))
    for i, (a, b) in enumerate(zip(placed, placed[1:]), start=1):
        gap = b.start - a.end
        if placed[i].no in overl:
            assert -p.overlap_max - 1e-6 <= gap <= -p.overlap_min + 1e-6    # 0.3〜1.0 秒かぶさる
            assert a.clip.speaker != b.clip.speaker                          # 自分自身には重ねない
        else:
            assert p.gap_min - 1e-6 <= gap <= p.gap_max + 1e-6
    with pytest.raises(ValueError):
        B.schedule(placed, p, "C")


def test_build_end_to_end(jvnv, tmp_path):
    clips = load_index(jvnv)
    out = tmp_path / "pm"
    m = B.build_pseudo_meeting(clips, out, small_params(), progress=lambda *_: None)
    assert set(m["variants"]) == {"A", "B"}
    for v in "AB":
        wav = out / f"meeting{v}.wav"
        y, sr = sf.read(wav)
        assert sr == 16000 and y.ndim == 1 and np.max(np.abs(y)) <= 0.98
        utts, notes, _ = read_utts_table(out / f"ref_meeting{v}.csv", role="reference")   # asrcheck がそのまま読める
        assert not notes and len(utts) == m["variants"][v]["turns"]
        assert {u.speaker for u in utts} == set(SPEAKERS)
        # 正解の区間には音がある・最初の発言の前（lead_in）は静か
        first = min(utts, key=lambda u: u.start)
        rms = lambda a, b: float(np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2)))   # noqa: E731
        assert rms(first.start + 0.1, first.start + 0.6) > 10 * rms(0.0, 0.5) + 1e-4
        assert len(y) / sr >= max(u.end for u in utts) + 0.9
    assert not (out / "meetingA_48k.wav").exists()                # 既定では 48k を残さない
    assert m["variants"]["A"]["overlapped_turns"] == 0
    assert m["variants"]["B"]["overlapped_turns"] >= 1
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["source"].startswith("JVNV")
    rows = list(csv.DictReader(open(out / "ref_meetingB.csv", encoding="utf-8-sig")))
    assert rows[0]["source_wav"].endswith(".wav") and rows[0]["emotion"] in ("anger", "happy", "sad")


def test_build_is_reproducible_and_keeps_48k(jvnv, tmp_path):
    clips = load_index(jvnv)
    m1 = B.build_pseudo_meeting(clips, tmp_path / "a", small_params(), variants=("B",), keep_48k=True, progress=lambda *_: None)
    m2 = B.build_pseudo_meeting(clips, tmp_path / "b", small_params(), variants=("B",), progress=lambda *_: None)
    assert m1["variants"]["B"]["sha256_16k"] == m2["variants"]["B"]["sha256_16k"]      # 同じ種なら同じ録音
    assert (tmp_path / "a" / "meetingB_48k.wav").exists() and sf.info(tmp_path / "a" / "meetingB_48k.wav").samplerate == 48000
