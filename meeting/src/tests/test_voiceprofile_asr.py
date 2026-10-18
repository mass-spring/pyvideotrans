"""ASR (faster-whisper) の薄いラッパーのテスト。faster-whisper 本体は入れていなくても動くよう、偽のモジュールで確かめる。

本物の faster-whisper での通し確認は、開発時に手作業で行った (英語の公開音声。README の「限界」参照)。
"""
from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest

from voiceprofile.asr import AsrUnavailable, WhisperAsr, make_asr


def test_make_asr_none_and_unknown():
    assert make_asr("none") is None and make_asr("") is None
    with pytest.raises(ValueError):
        make_asr("bogus")


def test_missing_faster_whisper_gives_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "faster_whisper":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AsrUnavailable, match="uv sync --group asr"):
        WhisperAsr().load()


def test_transcribe_passes_language_and_joins_segments(monkeypatch):
    calls = {}

    class Seg:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def __init__(self, size, device="auto", compute_type="int8"):
            calls["init"] = (size, device, compute_type)

        def transcribe(self, audio, **kw):
            calls["audio"], calls["kw"] = audio, kw
            return iter([Seg(" こんにちは、"), Seg("会議を始めます。 ")]), types.SimpleNamespace(language="ja")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    asr = make_asr("whisper", "tiny")
    text = asr.transcribe(np.zeros(16000, dtype=np.float64), "ja")
    assert text == "こんにちは、会議を始めます。"
    assert calls["init"][0] == "tiny" and calls["init"][2] == "int8"
    assert calls["kw"]["language"] == "ja" and calls["kw"]["vad_filter"] is False and calls["kw"]["beam_size"] == 1
    assert calls["audio"].dtype == np.float32
    asr.transcribe(np.zeros(100, dtype=np.float32), "")
    assert calls["kw"]["language"] is None                     # 言語が決まらないときは自動判定


def test_model_load_failure_is_reported_as_unavailable(monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise OSError("network down")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = Broken
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    with pytest.raises(AsrUnavailable, match="ネット接続"):
        make_asr("whisper")
