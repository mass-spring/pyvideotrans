"""
Tests for speaker_type = 'enrolled' (videotrans/process/_enrolled_speakers.py): name each subtitle line after the closest
enrolled speaker (cosine similarity to each person's mean embedding).

No model is downloaded or loaded: a fake embedder turns a tone into a "voice" (a person = a pure tone, the embedding = the
energy in a few frequency bands), so the assignment logic can be checked on tiny synthetic audio.
"""
import json
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from videotrans.process import _enrolled_speakers as es
from videotrans.util import voice_profile as vp

SR = 16000
BANDS = [0, 500, 1000, 2000, 4000, 8001]
FREQ = {"Aさん": 300.0, "Bさん": 700.0, "Cさん": 1500.0}


# ---------------------------------------------------------------------------
# helpers: synthetic voices, wav files, profile folders, fake embedder
# ---------------------------------------------------------------------------
def tone(freq, sec, amp=0.3, sr=SR):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(sec, sr=SR):
    return np.zeros(int(sec * sr), dtype=np.float32)


def write_wav(path, y, sr=SR, channels=1, width=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float64)
    if width == 2:
        raw = (np.clip(y, -1, 1) * 32767).astype('<i2')
    elif width == 4:
        raw = (np.clip(y, -1, 1) * 2147483647).astype('<i4')
    elif width == 3:
        v = (np.clip(y, -1, 1) * 8388607).astype('<i4')
        raw = np.stack([(v >> 0) & 255, (v >> 8) & 255, (v >> 16) & 255], axis=1).astype(np.uint8)
    else:
        raise ValueError(width)
    if channels > 1:
        raw = np.repeat(raw.reshape(len(y), -1), channels, axis=1)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(raw.tobytes())
    return path


class FakeEmbedder:
    """Band energies of the clip, normalised: a pure tone gives one-hot-like vectors, so different tones are orthogonal."""

    key = "fake"

    def __init__(self, model_id="fake:v1"):
        self.model_id = model_id
        self.calls = 0

    def embed(self, wav):
        self.calls += 1
        wav = np.asarray(wav, dtype=np.float64)
        if len(wav) < 160:
            return None
        spec = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** 2
        freqs = np.fft.rfftfreq(len(wav), 1 / SR)
        v = np.array([spec[(freqs >= a) & (freqs < b)].sum() for a, b in zip(BANDS[:-1], BANDS[1:])])
        n = np.linalg.norm(v)
        return None if n == 0 else v / n


def make_profile(root, name, freq, n_refs=2, n_eval=1, sec=3.0):
    """<root>/<name>/profile.json + refs/*.wav (+ eval/*.wav), each clip a tone of `freq`."""
    folder = Path(root) / name
    refs = []
    for i in range(n_refs):
        write_wav(folder / "refs" / f"r{i}.wav", tone(freq, sec + i))
        refs.append({"wav": f"refs/r{i}.wav", "text": "テキスト", "lang": "ja", "sec": sec + i})
    for i in range(n_eval):
        write_wav(folder / "eval" / f"e{i}.wav", tone(freq, 2.0))
    (folder / "profile.json").write_text(json.dumps({"name": name, "refs": refs}, ensure_ascii=False), encoding="utf-8")
    return folder


@pytest.fixture
def profiles(tmp_path):
    root = tmp_path / "profiles"
    for name, f in FREQ.items():
        make_profile(root, name, f)
    return root


def meeting(spec, gap=0.5):
    """spec: [(freq or None, sec)] -> (audio, [[start_ms, end_ms], ...]); None = a 'voice' made of white noise."""
    rng = np.random.default_rng(0)
    parts, subs, t = [silence(gap)], [], gap
    for f, sec in spec:
        y = tone(f, sec) if f is not None else (0.1 * rng.standard_normal(int(sec * SR))).astype(np.float32)
        parts += [y, silence(gap)]
        subs.append([int(round(t * 1000)), int(round((t + sec) * 1000))])
        t += sec + gap
    return np.concatenate(parts), subs


def centroids_of(profiles_dir, embedder=None):
    embedder = embedder or FakeEmbedder()
    return es.enrolled_centroids(profiles_dir, embedder, embedder.key)


# ---------------------------------------------------------------------------
# settings / option parsing
# ---------------------------------------------------------------------------
class TestSettings:
    def test_defaults_declared_and_speaker_type_unchanged(self):
        from videotrans.configure._app_settings import AppSettings

        d = AppSettings._get_defaults(AppSettings.__new__(AppSettings))
        assert d["speaker_type"] == "built"  # default behaviour is unchanged
        assert d["enrolled_speaker_model"] == "titanet"
        assert d["enrolled_min_sim"] == ""

    @pytest.mark.parametrize("raw", [None, "", "  ", "none", "OFF"])
    def test_min_sim_off(self, raw):
        assert es.parse_min_sim(raw) is None

    @pytest.mark.parametrize("raw,expected", [("0.35", 0.35), (0.5, 0.5), ("0", 0.0), (" -0.2 ", -0.2)])
    def test_min_sim_number(self, raw, expected):
        assert es.parse_min_sim(raw) == expected

    @pytest.mark.parametrize("raw", ["abc", "1.5", "nan", "-2"])
    def test_min_sim_invalid_is_a_japanese_error(self, raw):
        with pytest.raises(ValueError, match="enrolled_min_sim"):
            es.parse_min_sim(raw)

    def test_model_path_unknown_model(self):
        with pytest.raises(ValueError, match="titanet"):
            es.model_path("whisper")

    def test_model_path_missing_file_says_where_to_get_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(es, "ROOT_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="models/onnx"):
            es.model_path("titanet")

    def test_model_path_finds_shipped_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr(es, "ROOT_DIR", str(tmp_path))
        (tmp_path / "models" / "onnx").mkdir(parents=True)
        for key, fname in es.MODEL_FILES.items():
            (tmp_path / "models" / "onnx" / fname).write_bytes(b"x")
            assert es.model_path(key.upper()) == (key, tmp_path / "models" / "onnx" / fname)
        # the same files built_speakers loads
        assert es.MODEL_FILES["titanet"] == "nemo_en_titanet_small.onnx"
        assert es.MODEL_FILES["eres2net"] == "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"

    def test_enrolled_kwargs(self, profiles, tmp_path, monkeypatch):
        monkeypatch.setattr(es, "ROOT_DIR", str(tmp_path))
        (tmp_path / "models" / "onnx").mkdir(parents=True)
        (tmp_path / "models" / "onnx" / es.MODEL_FILES["titanet"]).write_bytes(b"x")
        s = {"voice_profile_dir": str(profiles), "enrolled_speaker_model": "titanet", "enrolled_min_sim": "0.4"}
        settings = types.SimpleNamespace(get=lambda k, d=None: s.get(k, d))
        assert es.enrolled_kwargs(settings) == {"profile_dir": str(profiles), "model": "titanet", "min_sim": 0.4}
        s["enrolled_min_sim"] = ""
        assert es.enrolled_kwargs(settings)["min_sim"] is None

    def test_enrolled_kwargs_without_profiles_is_an_error(self, tmp_path):
        settings = types.SimpleNamespace(get=lambda k, d=None: {"voice_profile_dir": ""}.get(k, d))
        with pytest.raises(ValueError, match="登録"):
            es.enrolled_kwargs(settings)


class TestProfileDirProblem:
    def test_ok(self, profiles):
        assert es.profile_dir_problem(profiles) is None
        assert es.profile_dir_problem(str(profiles) + "  ") is None

    @pytest.mark.parametrize("empty", ["", "   ", None])
    def test_empty_setting(self, empty):
        msg = es.profile_dir_problem(empty)
        assert "voice_profile_dir" in msg and "指定されていません" in msg

    def test_not_a_folder(self, tmp_path):
        assert "見つかりません" in es.profile_dir_problem(tmp_path / "nothing")

    def test_folder_without_profiles(self, tmp_path):
        (tmp_path / "Aさん").mkdir()  # no profile.json
        assert "1人も見つかりません" in es.profile_dir_problem(tmp_path)

    def test_profile_whose_wavs_are_missing_does_not_count(self, tmp_path):
        folder = make_profile(tmp_path, "Aさん", 300.0, n_refs=1, n_eval=0)
        (folder / "refs" / "r0.wav").unlink()
        assert "1人も見つかりません" in es.profile_dir_problem(tmp_path)


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------
class TestAudio:
    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_stdlib_reader_all_pcm_widths(self, tmp_path, width):
        y = tone(440, 0.2)
        p = write_wav(tmp_path / "a.wav", y, width=width)
        got, sr = es._read_wav_stdlib(p)
        assert sr == SR and got.shape == (len(y), 1)
        assert np.max(np.abs(got[:, 0] - y)) < 1e-3

    def test_read_audio_16k_resamples_and_downmixes(self, tmp_path):
        y = tone(300, 1.0, sr=48000)
        p = write_wav(tmp_path / "s.wav", y, sr=48000, channels=2)
        out = es.read_audio_16k(p)
        assert out.dtype == np.float32 and out.ndim == 1
        assert abs(len(out) - SR) <= 2
        # still a 300 Hz tone: the energy is in the first band
        v = FakeEmbedder().embed(out)
        assert np.argmax(v) == 0 and v[0] > 0.99

    def test_trim_edges_drops_silence_but_not_the_voice(self):
        y = np.concatenate([silence(1.0), tone(300, 1.0), silence(1.0)])
        t = es.trim_edges(y)
        assert 0.9 * SR < len(t) < 1.2 * SR

    def test_trim_edges_keeps_silence_and_tiny_input_unchanged(self):
        assert len(es.trim_edges(silence(1.0))) == SR
        assert len(es.trim_edges(tone(300, 0.01))) == int(0.01 * SR)

    def test_pad_min(self):
        assert len(es.pad_min(tone(300, 0.1))) == int(es.PAD_SEC * SR)
        assert len(es.pad_min(tone(300, 1.0))) == SR
        assert len(es.pad_min(np.zeros(0, dtype=np.float32))) == int(es.PAD_SEC * SR)


# ---------------------------------------------------------------------------
# profile loading and the mean embedding (+ cache)
# ---------------------------------------------------------------------------
class TestEnrolledCentroids:
    def test_names_are_the_profile_folder_names(self, profiles):
        names, cents, counts = centroids_of(profiles)
        assert names == list(vp.VoiceProfiles.load(profiles).profiles) == ["Aさん", "Bさん", "Cさん"]
        assert cents.shape == (3, len(BANDS) - 1)
        assert np.allclose(np.linalg.norm(cents, axis=1), 1.0)
        assert counts == {"Aさん": 3, "Bさん": 3, "Cさん": 3}  # 2 refs + 1 eval clip each

    def test_each_centroid_matches_its_own_tone(self, profiles):
        names, cents, _ = centroids_of(profiles)
        e = FakeEmbedder()
        for i, n in enumerate(names):
            sims = cents @ e.embed(tone(FREQ[n], 1.0))
            assert int(np.argmax(sims)) == i and sims[i] > 0.99

    def test_eval_clips_are_used_when_present(self, tmp_path):
        make_profile(tmp_path, "Aさん", 300.0, n_refs=1, n_eval=0)
        make_profile(tmp_path, "Bさん", 700.0, n_refs=1, n_eval=2)
        _, _, counts = centroids_of(tmp_path)
        assert counts == {"Aさん": 1, "Bさん": 3}

    def test_profile_without_usable_clip_is_skipped(self, tmp_path):
        make_profile(tmp_path, "Aさん", 300.0, n_refs=1, n_eval=0)
        make_profile(tmp_path, "Bさん", 700.0, n_refs=1, n_eval=0)
        # an all-silent reference cannot be embedded by the fake embedder
        write_wav(tmp_path / "Bさん" / "refs" / "r0.wav", silence(2.0))
        names, _, _ = centroids_of(tmp_path)
        assert names == ["Aさん"]

    def test_nobody_usable_is_an_error(self, tmp_path):
        make_profile(tmp_path, "Aさん", 300.0, n_refs=1, n_eval=0)
        write_wav(tmp_path / "Aさん" / "refs" / "r0.wav", silence(2.0))
        with pytest.raises(ValueError, match="特徴を計算できません"):
            centroids_of(tmp_path)


class TestEmbeddingCache:
    def test_second_run_uses_the_cache(self, profiles):
        e1 = FakeEmbedder()
        c1 = centroids_of(profiles, e1)
        assert e1.calls == 9
        assert (profiles / "Aさん" / "enrolled_emb_fake.json").is_file()
        e2 = FakeEmbedder()
        c2 = centroids_of(profiles, e2)
        assert e2.calls == 0
        assert np.allclose(c1[1], c2[1], atol=1e-6)

    def test_changed_wav_is_embedded_again_and_moves_the_mean(self, profiles):
        centroids_of(profiles)
        before = centroids_of(profiles)[1][0].copy()
        write_wav(profiles / "Aさん" / "refs" / "r0.wav", tone(700.0, 3.0))  # same name, other content
        e = FakeEmbedder()
        after = centroids_of(profiles, e)[1][0]
        assert e.calls == 1  # only the replaced clip
        assert not np.allclose(before, after, atol=1e-3)

    def test_added_and_removed_clips(self, profiles):
        centroids_of(profiles)
        write_wav(profiles / "Aさん" / "eval" / "e9.wav", tone(300.0, 2.0))
        e = FakeEmbedder()
        _, _, counts = centroids_of(profiles, e)
        assert e.calls == 1 and counts["Aさん"] == 4
        (profiles / "Aさん" / "eval" / "e9.wav").unlink()
        e = FakeEmbedder()
        _, _, counts = centroids_of(profiles, e)
        assert e.calls == 0 and counts["Aさん"] == 3
        stored = json.loads((profiles / "Aさん" / "enrolled_emb_fake.json").read_text(encoding="utf-8"))
        assert "eval/e9.wav" not in stored["clips"]

    def test_other_model_means_everything_is_recomputed(self, profiles):
        centroids_of(profiles, FakeEmbedder("fake:v1"))
        e = FakeEmbedder("fake:v2")  # e.g. the onnx file was replaced (other size)
        centroids_of(profiles, e)
        assert e.calls == 9

    def test_broken_cache_file_is_ignored(self, profiles):
        centroids_of(profiles)
        (profiles / "Aさん" / "enrolled_emb_fake.json").write_text("{broken", encoding="utf-8")
        e = FakeEmbedder()
        centroids_of(profiles, e)
        assert e.calls == 3

    def test_cache_that_cannot_be_written_does_not_break_anything(self, profiles):
        (profiles / "Aさん" / "enrolled_emb_fake.json").mkdir()  # a directory is in the way
        e = FakeEmbedder()
        names, _, counts = centroids_of(profiles, e)
        assert names == ["Aさん", "Bさん", "Cさん"] and counts["Aさん"] == 3

    def test_cache_stays_valid_when_the_folder_is_moved(self, profiles, tmp_path):
        centroids_of(profiles)
        moved = tmp_path / "moved"
        profiles.rename(moved)
        e = FakeEmbedder()
        centroids_of(moved, e)
        assert e.calls == 0  # keys are relative to the profile folder, so a copy of the folder keeps its cache


# ---------------------------------------------------------------------------
# assignment per subtitle line
# ---------------------------------------------------------------------------
class TestLabelSegments:
    def label(self, profiles, spec, **kw):
        audio, subs = meeting(spec)
        names, cents, _ = centroids_of(profiles)
        labels, diag = es.label_segments(audio, subs, names, cents, FakeEmbedder(), **kw)
        return labels, diag, subs

    def test_closest_enrolled_speaker_wins(self, profiles):
        spec = [(300.0, 2.0), (700.0, 2.0), (1500.0, 2.0), (300.0, 2.5), (700.0, 3.0)]
        labels, diag, subs = self.label(profiles, spec)
        assert labels == ["Aさん", "Bさん", "Cさん", "Aさん", "Bさん"]
        assert len(diag["segments"]) == len(subs)
        r = diag["segments"][0]
        assert r["speaker"] == "Aさん" and r["best"] > 0.99 and r["second"] < 0.2
        assert r["margin"] == pytest.approx(r["best"] - r["second"], abs=1e-3)
        assert r["second_speaker"] in ("Bさん", "Cさん")
        assert not r["flags"] and not r["inherited"]
        assert diag["summary"]["review"] == 0 and diag["speakers"] == ["Aさん", "Bさん", "Cさん"]

    def test_ambiguous_line_is_flagged_low_margin(self, profiles):
        audio_a, _ = meeting([(300.0, 2.0)])
        mixed = tone(300.0, 2.0) + tone(700.0, 2.0) * 0.95
        audio = np.concatenate([silence(0.5), mixed, silence(0.5)])
        names, cents, _ = centroids_of(profiles)
        labels, diag = es.label_segments(audio, [[500, 2500]], names, cents, FakeEmbedder())
        r = diag["segments"][0]
        assert r["margin"] < es.LOW_MARGIN and "low_margin" in r["flags"]
        assert set([r["speaker"], r["second_speaker"]]) == {"Aさん", "Bさん"}

    def test_speaker_change_inside_a_line_is_flagged_but_not_split(self, profiles):
        audio = np.concatenate([silence(0.5), tone(300.0, 2.5), tone(700.0, 2.5), silence(0.5)])
        names, cents, _ = centroids_of(profiles)
        labels, diag = es.label_segments(audio, [[500, 5500]], names, cents, FakeEmbedder())
        assert len(labels) == 1  # the subtitle is not split
        r = diag["segments"][0]
        assert "mid_change" in r["flags"]
        assert [p["speaker"] for p in r["pieces"]] == ["Aさん", "Bさん"]
        assert abs(r["pieces"][0]["end_ms"] - 3000) <= 600  # the change is near 3.0 s (0.5 + 2.5)
        assert diag["summary"]["mid_change"] == 1

    def test_window_check_can_be_turned_off(self, profiles):
        audio = np.concatenate([silence(0.5), tone(300.0, 2.5), tone(700.0, 2.5), silence(0.5)])
        names, cents, _ = centroids_of(profiles)
        _, diag = es.label_segments(audio, [[500, 5500]], names, cents, FakeEmbedder(), window_check=False)
        assert "mid_change" not in diag["segments"][0]["flags"]

    def test_pure_single_speaker_line_is_not_flagged_mid_change(self, profiles):
        _, diag, _ = self.label(profiles, [(300.0, 6.0)])
        assert "mid_change" not in diag["segments"][0]["flags"]

    def test_short_line_inherits_the_previous_speaker(self, profiles):
        spec = [(300.0, 2.0), (700.0, 0.3), (1500.0, 2.0)]  # the 0.3 s line is a 700 Hz voice, but too short to trust
        labels, diag, _ = self.label(profiles, spec)
        assert labels == ["Aさん", "Aさん", "Cさん"]
        r = diag["segments"][1]
        assert r["inherited"] and r["flags"] == ["inherited"] and r["best"] is None
        assert diag["summary"]["inherited"] == 1

    def test_first_line_short_takes_the_first_embeddable_line(self, profiles):
        labels, diag, _ = self.label(profiles, [(300.0, 0.2), (300.0, 0.3), (700.0, 2.0), (300.0, 2.0)])
        assert labels == ["Bさん", "Bさん", "Bさん", "Aさん"]
        assert [r["inherited"] for r in diag["segments"]] == [True, True, False, False]

    def test_min_seg_sec_is_configurable(self, profiles):
        labels, _, _ = self.label(profiles, [(300.0, 2.0), (700.0, 0.3)], min_seg_sec=0.2)
        assert labels == ["Aさん", "Bさん"]

    def test_unusable_times_inherit(self, profiles):
        audio, subs = meeting([(300.0, 2.0), (700.0, 2.0)])
        subs = [subs[0], [3000, 3000], [4000, 3500], subs[1], [900000, 905000]]  # empty, reversed, after the end of the audio
        names, cents, _ = centroids_of(profiles)
        labels, diag = es.label_segments(audio, subs, names, cents, FakeEmbedder())
        assert labels == ["Aさん", "Aさん", "Aさん", "Bさん", "Bさん"]
        assert [r["inherited"] for r in diag["segments"]] == [False, True, True, False, True]

    def test_nothing_embeddable_is_an_error(self, profiles):
        names, cents, _ = centroids_of(profiles)
        with pytest.raises(ValueError, match="埋め込める"):
            es.label_segments(silence(3.0), [[0, 300], [500, 2500]], names, cents, FakeEmbedder())

    def test_min_sim_marks_unknown(self, profiles):
        spec = [(300.0, 2.0), (None, 2.0), (700.0, 2.0)]  # the middle "voice" is noise: close to nobody
        labels, diag, _ = self.label(profiles, spec, min_sim=0.9)
        assert labels == ["Aさん", "unknown", "Bさん"]
        assert "unknown" in diag["segments"][1]["flags"] and diag["summary"]["unknown"] == 1
        assert diag["min_sim"] == 0.9

    def test_min_sim_off_by_default(self, profiles):
        labels, _, _ = self.label(profiles, [(300.0, 2.0), (None, 2.0)])
        assert "unknown" not in labels and labels[0] == "Aさん"

    def test_short_line_after_unknown_inherits_unknown(self, profiles):
        labels, _, _ = self.label(profiles, [(None, 2.0), (300.0, 0.3)], min_sim=0.9)
        assert labels == ["unknown", "unknown"]

    def test_single_enrolled_speaker(self, tmp_path):
        make_profile(tmp_path, "Aさん", 300.0)
        audio, subs = meeting([(300.0, 2.0), (700.0, 2.0)])
        names, cents, _ = centroids_of(tmp_path)
        labels, diag = es.label_segments(audio, subs, names, cents, FakeEmbedder())
        assert labels == ["Aさん", "Aさん"]
        assert diag["segments"][0]["second"] is None and diag["segments"][0]["margin"] is None


class TestPiecesFromWindows:
    def test_two_speakers(self):
        spans = es.window_spans(0, 6000)
        sim = np.array([[1.0, 0.0] if (a + b) / 2 < 3000 else [0.0, 1.0] for a, b in spans])
        pieces = es.pieces_from_windows(spans, sim, 0, 6000)
        assert [k for _, _, k in pieces] == [0, 1] and pieces[0][0] == 0 and pieces[-1][1] == 6000
        assert abs(pieces[0][1] - 3000) <= 1000

    def test_short_blip_is_absorbed(self):
        spans = es.window_spans(0, 6000)
        sim = np.array([[0.0, 1.0] if 2800 <= (a + b) / 2 <= 3200 else [1.0, 0.0] for a, b in spans])
        assert len(es.pieces_from_windows(spans, sim, 0, 6000)) == 1

    def test_window_spans(self):
        assert es.window_spans(1000, 2200) == [(1000, 2200)]
        s = es.window_spans(0, 4000)
        assert s[0] == (0, 1500) and s[-1] == (2500, 4000) and all(b - a == 1500 for a, b in s)


# ---------------------------------------------------------------------------
# the entry point (process callback): files, formats, return protocol
# ---------------------------------------------------------------------------
class TestEnrolledSpeakers:
    def run(self, tmp_path, profiles, spec, **kw):
        audio, subs = meeting(spec)
        wav = write_wav(tmp_path / "meeting.wav", audio)
        sub_file = tmp_path / "diariz-1.json"
        sub_file.write_text(json.dumps(subs), encoding="utf-8")
        out = tmp_path / "cache"
        out.mkdir(exist_ok=True)
        speak = out / "speaker.json"
        rs = es.enrolled_speakers(input_file=str(wav), subtitles_file=str(sub_file), speak_file=str(speak), profile_dir=str(profiles),
                                  embedder=FakeEmbedder(), **kw)
        return rs, speak, subs

    def test_writes_speaker_json_in_the_format_built_speakers_writes(self, tmp_path, profiles):
        (ok, err), speak, subs = self.run(tmp_path, profiles, [(300.0, 2.0), (700.0, 2.0), (1500.0, 2.0), (300.0, 2.0)])
        assert (ok, err) == (True, None)  # same return protocol as built_speakers
        text = speak.read_text(encoding="utf-8")
        labels = json.loads(text)
        # built_speakers: Path(speak_file).write_text(json.dumps(output), encoding='utf-8') with output = one str per subtitle line
        assert text == json.dumps(labels)
        assert isinstance(labels, list) and len(labels) == len(subs) and all(isinstance(x, str) for x in labels)
        assert labels == ["Aさん", "Bさん", "Cさん", "Aさん"]

    def test_extra_kwargs_from_the_process_manager_are_accepted(self, tmp_path, profiles):
        rs, _, _ = self.run(tmp_path, profiles, [(300.0, 2.0)], num_speakers=4, is_cuda=False, device_index=0, device_name="auto",
                            language="ja")
        assert rs == (True, None)

    def test_names_flow_into_clone_reference_selection(self, tmp_path, profiles):
        """The labels are profile folder names, so voice_profile.choose_clone_ref finds the profile with no speaker_map.json."""
        _, speak, _ = self.run(tmp_path, profiles, [(300.0, 2.0), (700.0, 2.0)])
        labels = vp.read_speaker_labels(speak)
        vprofiles = vp.VoiceProfiles.load(profiles)
        assert not vprofiles.speaker_map
        for i, name in enumerate(["Aさん", "Bさん"]):
            assert vprofiles.profile_name(labels[i]) == name
            ref = vp.choose_clone_ref(vprofiles, labels, i, "ja")
            assert ref is not None and ref.profile == name and Path(ref.wav).is_file()

    def test_unknown_falls_back_to_the_line_reference(self, tmp_path, profiles):
        _, speak, _ = self.run(tmp_path, profiles, [(None, 2.0)], min_sim="0.9")
        labels = vp.read_speaker_labels(speak)
        assert labels == ["unknown"]
        assert vp.choose_clone_ref(vp.VoiceProfiles.load(profiles), labels, 0, "ja") is None

    def test_diagnostics_files_next_to_speaker_json(self, tmp_path, profiles):
        _, speak, subs = self.run(tmp_path, profiles, [(300.0, 2.0), (300.0, 0.3), (700.0, 2.0)])
        diag = json.loads((speak.parent / es.DIAG_JSON).read_text(encoding="utf-8"))
        assert len(diag["segments"]) == len(subs) and diag["model"] == "fake" and diag["summary"]["inherited"] == 1
        assert diag["clips_per_speaker"] == {"Aさん": 3, "Bさん": 3, "Cさん": 3}
        raw = (speak.parent / es.DIAG_CSV).read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")  # BOM: opens correctly in Excel
        import csv
        rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
        assert rows[0][:5] == ["行", "開始(秒)", "終了(秒)", "話者", "最高点"]
        assert len(rows) == 1 + len(subs)
        assert rows[2][3] == "Aさん" and rows[2][8] == "●" and "引き継いだ" in rows[2][9]
        assert rows[1][8] == "" and rows[3][8] == ""

    def test_copy_review_files(self, tmp_path, profiles):
        _, speak, _ = self.run(tmp_path, profiles, [(300.0, 2.0)])
        target = tmp_path / "out"
        target.mkdir()
        es.copy_review_files(speak.parent, target)
        assert (target / es.DIAG_JSON).is_file() and (target / es.DIAG_CSV).is_file()
        es.copy_review_files(tmp_path / "nothing-here", target)  # missing files / folders never raise

    def test_empty_subtitles_fail_like_built(self, tmp_path, profiles):
        rs, speak, _ = self.run(tmp_path, profiles, [])
        assert rs[0] is False and not speak.exists()

    def test_no_profiles_returns_false_with_a_japanese_reason(self, tmp_path):
        empty = tmp_path / "empty-profiles"
        empty.mkdir()
        rs, speak, _ = self.run(tmp_path, empty, [(300.0, 2.0)])
        assert rs[0] is False and "登録" in rs[1] and not speak.exists()

    def test_invalid_min_sim_returns_false(self, tmp_path, profiles):
        rs, _, _ = self.run(tmp_path, profiles, [(300.0, 2.0)], min_sim="abc")
        assert rs[0] is False and "enrolled_min_sim" in rs[1]

    def test_missing_audio_returns_false(self, tmp_path, profiles):
        sub_file = tmp_path / "s.json"
        sub_file.write_text("[[0, 1000]]", encoding="utf-8")
        rs = es.enrolled_speakers(input_file=str(tmp_path / "nope.wav"), subtitles_file=str(sub_file), speak_file=str(tmp_path / "speaker.json"),
                                  profile_dir=str(profiles), embedder=FakeEmbedder())
        assert rs[0] is False

    def test_picklable_for_the_process_pool(self):
        import pickle

        assert pickle.loads(pickle.dumps(es.enrolled_speakers)) is es.enrolled_speakers

    def test_is_exported_where_the_dispatch_looks_for_it(self):
        import inspect

        params = inspect.signature(es.enrolled_speakers).parameters
        for name in ("input_file", "subtitles_file", "speak_file", "profile_dir", "model", "min_sim"):
            assert name in params
        assert params["model"].default == "titanet" and params["min_sim"].default is None
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# ---------------------------------------------------------------------------
# dispatch in BaseTask._diariz_common (everything heavy mocked)
# ---------------------------------------------------------------------------
class TestDiarizCommonDispatch:
    @pytest.fixture
    def host(self, tmp_path, monkeypatch):
        from videotrans.task import _base

        cache, target = tmp_path / "cache", tmp_path / "target"
        cache.mkdir()
        target.mkdir()
        seen = {"signals": [], "calls": []}

        t = _base.BaseTask.__new__(_base.BaseTask)
        t.cfg = types.SimpleNamespace(cache_folder=str(cache), target_dir=str(target), source_wav=str(tmp_path / "a.wav"), is_cuda=True,
                                      detect_language="ja")
        t.source_srt_list = [{"start_time": 0, "end_time": 1000}, {"start_time": 1000, "end_time": 2000}]
        t.max_speakers = 4
        t.precent = 1
        t.signal = lambda **kw: seen["signals"].append(kw)
        t._process_callback = lambda d: None

        def new_process(callback=None, title="", is_cuda=False, kwargs=None):
            seen["calls"].append({"callback": callback, "is_cuda": is_cuda, "kwargs": dict(kwargs)})
            Path(kwargs["speak_file"]).write_text('["Aさん", "Bさん"]', encoding="utf-8")
            (Path(kwargs["speak_file"]).parent / es.DIAG_JSON).write_text("{}", encoding="utf-8")
            (Path(kwargs["speak_file"]).parent / es.DIAG_CSV).write_text("x", encoding="utf-8")
            return True

        t._new_process = new_process
        monkeypatch.setattr(_base, "is_connect_hf", lambda: True)
        return types.SimpleNamespace(task=t, seen=seen, base=_base, monkeypatch=monkeypatch, cache=cache, target=target)

    def set_settings(self, host, **values):
        data = {"speaker_type": "enrolled", "hf_token": "", **values}
        host.monkeypatch.setattr(host.base, "settings", types.SimpleNamespace(get=lambda k, d=None: data.get(k, d)))

    def test_enrolled_branch_calls_enrolled_speakers_on_cpu(self, host, profiles, tmp_path):
        self.set_settings(host, voice_profile_dir=str(profiles), enrolled_speaker_model="titanet", enrolled_min_sim="0.4")
        host.monkeypatch.setattr(es, "model_path", lambda m: ("titanet", tmp_path / "x.onnx"))
        host.task._diariz_common()
        (call,) = host.seen["calls"]
        assert call["callback"] is es.enrolled_speakers
        assert call["is_cuda"] is False  # sherpa-onnx runs on the CPU process pool even when cuda is on
        kw = call["kwargs"]
        assert (kw["profile_dir"], kw["model"], kw["min_sim"]) == (str(profiles), "titanet", 0.4)
        assert kw["speak_file"] == str(host.cache / "speaker.json") and kw["input_file"] == str(tmp_path / "a.wav")
        assert json.loads(Path(kw["subtitles_file"]).read_text(encoding="utf-8")) == [[0, 1000], [1000, 2000]]
        # speaker.json and the review files end up in the output folder
        assert json.loads((host.target / "speaker.json").read_text(encoding="utf-8")) == ["Aさん", "Bさん"]
        assert (host.target / es.DIAG_JSON).is_file() and (host.target / es.DIAG_CSV).is_file()

    def test_no_profiles_skips_diarization_with_a_message(self, host):
        self.set_settings(host, voice_profile_dir="")
        host.task._diariz_common()
        assert host.seen["calls"] == []
        assert any("voice_profile_dir" in str(s.get("text")) for s in host.seen["signals"])
        assert not (host.target / "speaker.json").exists()

    def test_other_speaker_types_do_not_touch_the_enrolled_module(self, host):
        """built stays the default path: it still asks for the built-in models (download is mocked) and the old callback."""
        self.set_settings(host, speaker_type="built")
        host.monkeypatch.setattr("videotrans.util.help_down.down_file_from_hf", lambda *a, **k: None)
        host.task._diariz_common()
        (call,) = host.seen["calls"]
        from videotrans.process.prepare_audio import built_speakers

        assert call["callback"] is built_speakers
        assert "profile_dir" not in call["kwargs"]
        assert not (host.target / es.DIAG_JSON).exists()
