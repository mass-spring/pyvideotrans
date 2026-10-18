"""asrcheck run の部品 (区間分け・用語リスト・出力 CSV) と、各エンジン API の呼び方 (本物のモデルは使わず、偽のモジュールで確認)"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asrcheck import cli, runners
from asrcheck.ioformats import InputError, read_utts_table

SR = runners.SR


def tone(sec, f=220.0, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


# ---------------------------------------------------------------------------
def test_energy_segments_finds_speech_between_silences():
    rng = np.random.default_rng(0)
    noise = lambda s: (0.002 * rng.standard_normal(int(s * SR))).astype(np.float32)  # noqa: E731
    y = np.concatenate([noise(1.0), tone(2.0), noise(1.5), tone(3.0, 300), noise(1.0)])
    segs = runners.energy_segments(y)
    assert len(segs) == 2
    (s1, e1), (s2, e2) = segs
    assert s1 == pytest.approx(1.0, abs=0.25) and e1 == pytest.approx(3.0, abs=0.25)
    assert s2 == pytest.approx(4.5, abs=0.25) and e2 == pytest.approx(7.5, abs=0.25)


def test_energy_segments_silence_and_short_blips():
    assert runners.energy_segments(silence(5)) == []
    assert runners.energy_segments(np.zeros(10, dtype=np.float32)) == []
    y = np.concatenate([silence(1), tone(0.1), silence(1)])
    assert runners.energy_segments(y, min_len=0.5) == []     # 0.1 秒だけの音は捨てる


def test_energy_segments_fills_short_pauses_and_splits_long_speech():
    y = np.concatenate([silence(0.5), tone(1.0), silence(0.3), tone(1.0), silence(0.5)])
    assert len(runners.energy_segments(y, min_silence=0.5)) == 1          # 0.3 秒の間は同じ区間
    burst = np.concatenate([tone(5.0), silence(0.2)])          # 0.2 秒の息継ぎをはさんだ 70 秒ほどの連続した発話
    long = np.concatenate([silence(0.5), np.tile(burst, 13), silence(0.5)])
    segs = runners.energy_segments(long, max_len=30.0)
    assert len(segs) >= 3 and all(e - s <= 30.5 for s, e in segs)


def test_turns_to_segments_merge_split_and_drop():
    turns = [(0, 2, "A"), (2.5, 4, "A"), (4.2, 6, "B"), (6.1, 7, "A"), (10, 10.1, "B")]
    segs = runners.turns_to_segments(turns, merge_gap=0.8, max_len=30, min_len=0.2)
    assert segs == [(0, 4, "A"), (4.2, 6, "B"), (6.1, 7, "A")]          # 同じ話者が続くときだけ連結。0.1 秒の区間は捨てる
    segs = runners.turns_to_segments([(0, 70, "A")], max_len=30)
    assert len(segs) == 3 and segs[0][0] == 0 and segs[-1][1] == pytest.approx(70) and all(e - s <= 30 for s, e, _ in segs)


def test_relabel_speakers_in_order_of_appearance():
    out = runners.relabel_speakers([(5, 6, "SPEAKER_01"), (0, 1, "SPEAKER_07"), (2, 3, "SPEAKER_01")])
    assert out == [(0, 1, "spk0"), (2, 3, "spk1"), (5, 6, "spk1")]


def test_read_hotwords(tmp_path):
    p = tmp_path / "w.txt"
    p.write_bytes("# 用語集\n田中\n\nPyVideoTrans\n田中\n  衝突安全  \n".encode("utf-8-sig"))
    assert runners.read_hotwords(p) == ["田中", "PyVideoTrans", "衝突安全"]
    assert runners.read_hotwords(None) == []
    with pytest.raises(InputError):
        runners.read_hotwords(tmp_path / "none.txt")


def test_split_sensevoice_tags():
    assert runners.split_sensevoice_tags("<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは。") == ("こんにちは。", "HAPPY")
    assert runners.split_sensevoice_tags("<|ja|><|NEUTRAL|><|Speech|>はい") == ("はい", "NEUTRAL")
    assert runners.split_sensevoice_tags("ただの文") == ("ただの文", "")


def test_load_audio_16k_resamples_and_mixes_down(tmp_path):
    t = np.arange(8000) / 8000                                  # 8 kHz で 1 秒
    y = np.stack([0.3 * np.sin(2 * np.pi * 200 * t), 0.3 * np.sin(2 * np.pi * 400 * t)], axis=1).astype(np.float32)
    sf.write(tmp_path / "s.wav", y, 8000)
    w = runners.load_audio_16k(tmp_path / "s.wav")
    assert w.dtype == np.float32 and w.ndim == 1 and len(w) == pytest.approx(SR, abs=2)
    with pytest.raises(InputError):
        runners.load_audio_16k(tmp_path / "none.wav")


# ---------------------------------------------------------------------------
class FakeEngine(runners.Engine):
    name = "fake"
    device = "cpu"

    def __init__(self, native=False, **kw):
        self.native_segments = native
        self.calls = []

    def transcribe_batch(self, waves):
        self.calls.append([len(w) / SR for w in waves])
        return [(f"text{len(self.calls)}-{i}", "HAPPY" if i == 0 else "") for i, _ in enumerate(waves)]

    def transcribe_long(self, wave):
        return [(0.0, 1.0, "long1"), (2.0, 3.0, "long2")]


def test_transcribe_audio_with_turns_uses_speaker_per_turn():
    wave = np.zeros(10 * SR, dtype=np.float32)
    rows = runners.transcribe_audio(FakeEngine(), wave, [(0, 3, "spk0"), (4, 8, "spk1")])
    assert [(r["start"], r["end"], r["speaker"]) for r in rows] == [(0, 3, "spk0"), (4, 8, "spk1")]
    assert rows[0]["emotion"] == "HAPPY"


def test_transcribe_audio_native_segments_and_energy_vad():
    wave = np.zeros(10 * SR, dtype=np.float32)
    rows = runners.transcribe_audio(FakeEngine(native=True), wave, None)
    assert [r["text"] for r in rows] == ["long1", "long2"] and all(r["speaker"] == "" for r in rows)
    y = np.concatenate([silence(0.5), tone(2.0), silence(1.0), tone(2.0), silence(0.5)])
    rows = runners.transcribe_audio(FakeEngine(native=False), y, None)
    assert len(rows) == 2


def test_cmd_run_writes_csv_meta_and_diar(tmp_path, monkeypatch):
    sf.write(tmp_path / "meeting.wav", np.concatenate([silence(0.5), tone(2.0), silence(1.0), tone(2.0), silence(0.5)]), SR)
    monkeypatch.setitem(runners.ENGINE_CLASSES, "whisper", lambda model, device, lang, hot: FakeEngine(native=False))
    monkeypatch.setattr(runners, "pyannote_turns", lambda *a, **k: [(0.5, 2.5, "SPEAKER_03"), (3.5, 5.5, "SPEAKER_00")])
    out = tmp_path / "hyp.csv"
    rc = cli.main(["run", "--engine", "whisper", "--diarizer", "pyannote", "--audio", str(tmp_path / "meeting.wav"),
                   "--num-speakers", "2", "--out", str(out)])
    assert rc == 0
    utts, _, enc = read_utts_table(out, role="hypothesis")
    assert enc == "utf-8-sig" and [u.speaker for u in utts] == ["spk0", "spk1"] and utts[0].start == pytest.approx(0.5)
    meta = json.loads((tmp_path / "hyp.meta.json").read_text(encoding="utf-8"))
    assert meta["engine"] == "whisper" and meta["diarizer"] == "pyannote" and meta["num_speakers"] == 2 and meta["audio_sec"] == pytest.approx(6.0, abs=0.1)
    diar, _, _ = read_utts_table(tmp_path / "hyp.diar.csv", role="hypothesis")
    assert [(u.speaker, u.text) for u in diar] == [("spk0", ""), ("spk1", "")]


def test_cmd_run_multiple_audio_adds_file_column(tmp_path, monkeypatch):
    for n in ("a", "b"):
        sf.write(tmp_path / f"{n}.wav", np.concatenate([silence(0.5), tone(2.0), silence(0.5)]), SR)
    monkeypatch.setitem(runners.ENGINE_CLASSES, "whisper", lambda model, device, lang, hot: FakeEngine(native=False))
    out = tmp_path / "h.csv"
    assert cli.main(["run", "--engine", "whisper", "--audio", str(tmp_path / "a.wav"), "--audio", str(tmp_path / "b.wav"), "--out", str(out)]) == 0
    utts, _, _ = read_utts_table(out, role="hypothesis")
    assert [u.rec for u in utts] == ["a", "b"]


def test_run_errors(tmp_path, capsys):
    assert cli.main(["run", "--engine", "whisper", "--audio", str(tmp_path / "none.wav")]) == 2
    assert "見つかりません" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 各エンジンの API の呼び方 (公式ページの例と同じ形で呼んでいるか)。偽のモジュールを差し込む。
# ---------------------------------------------------------------------------
def test_whisper_engine_call_shape(monkeypatch):
    calls = {}

    class Seg:
        def __init__(self, s, e, t):
            self.start, self.end, self.text = s, e, t

    class FakeModel:
        def __init__(self, size, device, compute_type):
            calls["init"] = (size, device, compute_type)

        def transcribe(self, audio, **kw):
            calls["kw"] = kw
            return iter([Seg(0.0, 1.5, " こんにちは "), Seg(2.0, 3.0, "")]), object()

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    eng = runners.WhisperEngine("small", "cpu", "ja", ["田中", "衝突安全"])
    assert calls["init"] == ("small", "cpu", "int8")                        # README: CPU は int8
    segs = eng.transcribe_long(np.zeros(SR, dtype=np.float32))
    assert segs == [(0.0, 1.5, "こんにちは")]                              # 空の文は捨てる
    assert calls["kw"]["language"] == "ja" and calls["kw"]["vad_filter"] is True
    assert calls["kw"]["hotwords"] == "田中、衝突安全" and calls["kw"]["beam_size"] == 5
    assert eng.transcribe_one(np.zeros(SR, dtype=np.float32)) == "こんにちは"      # ジェネレータを回して連結


def test_qwen3asr_engine_call_shape(monkeypatch):
    calls = {}

    class FakeQ:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls["init"] = (name, kw)
            return cls()

        def transcribe(self, audio, context="", language=None, return_time_stamps=False):
            calls["transcribe"] = (audio, context, language)
            return [types.SimpleNamespace(text=f" t{i} ", language="Japanese") for i, _ in enumerate(audio)]

    fake_torch = types.SimpleNamespace(bfloat16="bf16", float32="f32")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", types.SimpleNamespace(Qwen3ASRModel=FakeQ))
    eng = runners.Qwen3AsrEngine("Qwen/Qwen3-ASR-1.7B", "cpu", "ja", ["田中", "衝突安全"])
    assert calls["init"][0] == "Qwen/Qwen3-ASR-1.7B" and calls["init"][1]["device_map"] == "cpu" and calls["init"][1]["dtype"] == "f32"
    out = eng.transcribe_batch([np.zeros(SR, dtype=np.float32)] * 3)
    assert [t for t, _ in out] == ["t0", "t1", "t2"]
    audio, ctx, lang = calls["transcribe"]
    assert lang == "Japanese" and ctx == "田中、衝突安全" and isinstance(audio[0], tuple) and audio[0][1] == SR     # (np.ndarray, sr)


def test_cohere_engine_call_shape(monkeypatch):
    calls = {}

    class FakeInputs(dict):
        def to(self, *a, **kw):
            calls["to"] = (a, kw)
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, name):
            calls["proc"] = name
            return cls()

        def __call__(self, audio, sampling_rate, return_tensors, language):
            calls["proc_call"] = (sampling_rate, return_tensors, language)
            return FakeInputs(input_features="x")

        def decode(self, outputs, skip_special_tokens=True, **kw):
            calls["decode"] = (skip_special_tokens, kw)
            return [" 結果 "]

    class FakeModel:
        device, dtype = "cpu", "f32"

        @classmethod
        def from_pretrained(cls, name, device_map=None):
            calls["model"] = (name, device_map)
            return cls()

        def to(self, d):
            return self

        def generate(self, **kw):
            calls["gen"] = kw
            return "ids"

    fake_tf = types.SimpleNamespace(AutoProcessor=FakeProcessor, CohereAsrForConditionalGeneration=FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    eng = runners.CohereEngine("CohereLabs/cohere-transcribe-03-2026", "cpu", "ja", ["田中"])
    assert eng.transcribe_one(np.zeros(SR, dtype=np.float32)) == "結果"
    assert calls["proc_call"] == (16000, "pt", "ja") and calls["gen"]["max_new_tokens"] >= 256
    assert calls["decode"][0] is True


def test_parakeet_engine_call_shape(monkeypatch):
    seen = {}

    class FakeASR:
        @staticmethod
        def from_pretrained(model_name):
            seen["name"] = model_name
            return FakeASR()

        def transcribe(self, paths):
            seen["paths"] = list(paths)
            assert all(Path(p).is_file() and p.endswith(".wav") for p in paths)   # 16kHz wav のファイルを渡す
            return [types.SimpleNamespace(text=f" 文{i} ") for i, _ in enumerate(paths)]

    nemo_asr = types.SimpleNamespace(models=types.SimpleNamespace(ASRModel=FakeASR))
    monkeypatch.setitem(sys.modules, "nemo", types.ModuleType("nemo"))
    monkeypatch.setitem(sys.modules, "nemo.collections", types.ModuleType("nemo.collections"))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo_asr)
    monkeypatch.setattr(sys.modules["nemo.collections"], "asr", nemo_asr, raising=False)
    monkeypatch.setattr(sys.modules["nemo"], "collections", sys.modules["nemo.collections"], raising=False)
    eng = runners.ParakeetEngine("nvidia/parakeet-tdt_ctc-0.6b-ja", "cpu", "ja", [])
    assert [t for t, _ in eng.transcribe_batch([np.zeros(SR, dtype=np.float32)] * 2)] == ["文0", "文1"]
    assert seen["name"] == "nvidia/parakeet-tdt_ctc-0.6b-ja" and len(seen["paths"]) == 2
    with pytest.raises(InputError):
        runners.ParakeetEngine("m", "cpu", "en", [])


def test_sensevoice_engine_call_shape(monkeypatch):
    seen = {}

    class FakeAuto:
        def __init__(self, model, device, disable_update):
            seen["init"] = (model, device)

        def generate(self, input, cache, language, use_itn, batch_size_s):
            seen["gen"] = (input, cache, language, use_itn, batch_size_s)
            return [{"text": "<|ja|><|ANGRY|><|Speech|><|withitn|>ふざけるな"}]

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAuto))
    eng = runners.SenseVoiceEngine("iic/SenseVoiceSmall", "cpu", "ja", [])
    assert eng.transcribe_batch([np.zeros(SR, dtype=np.float32)]) == [("ふざけるな", "ANGRY")]
    assert seen["init"] == ("iic/SenseVoiceSmall", "cpu") and seen["gen"][2:] == ("ja", True, 60)


def test_pyannote_call_shape_uses_exclusive_output(monkeypatch):
    seen = {}

    class Track:
        def __init__(self, s, e):
            self.start, self.end = s, e

    class Ann:
        def __init__(self, items):
            self.items = items

        def itertracks(self, yield_label=False):
            assert yield_label
            return iter([(Track(s, e), None, spk) for s, e, spk in self.items])

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, name, token=None):
            seen["init"] = (name, token)
            return cls()

        def to(self, dev):
            seen["dev"] = dev

        def __call__(self, audio, **kw):
            seen["call"] = (audio, kw)
            return types.SimpleNamespace(
                speaker_diarization=Ann([(0, 9, "WRONG")]), exclusive_speaker_diarization=Ann([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]))

    class FakeTensor:
        def unsqueeze(self, d):
            return ("tensor", d)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda w: FakeTensor(), cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", types.SimpleNamespace(Pipeline=FakePipeline))
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    turns = runners.pyannote_turns(np.zeros(SR, dtype=np.float32), 2, None, None, "cpu")
    assert turns == [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]           # exclusive の方を使う
    assert seen["init"] == ("pyannote/speaker-diarization-community-1", "hf_dummy")
    audio, kw = seen["call"]
    assert audio["sample_rate"] == 16000 and audio["waveform"] == ("tensor", 0) and kw == {"num_speakers": 2}
    runners.pyannote_turns(np.zeros(SR, dtype=np.float32), None, 2, 5, "cpu")
    assert seen["call"][1] == {"min_speakers": 2, "max_speakers": 5}


def test_missing_package_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)    # import すると ImportError になる
    with pytest.raises(InputError) as e:
        runners.WhisperEngine("small", "cpu", "ja", [])
    assert "faster-whisper" in str(e.value) and "uv run --with" in str(e.value)
