"""emocheck のモデル実行部分: SenseVoice / Whisper の呼び方 (偽のモジュールで確認)、途中保存と再開、失敗の扱い"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from asrcheck.ioformats import InputError

from emocheck import runners
from emocheck.dataset import Clip
from synth import SR


class FakeRec:
    key, label, has_emotion = "fake", "偽モデル", True

    def __init__(self, fail_on=()):
        self.calls: list[str] = []
        self.fail_on = set(fail_on)

    def recognize(self, path):
        self.calls.append(Path(path).name)
        if Path(path).name in self.fail_on:
            raise RuntimeError("壊れた音声")
        return runners.Recognition(f"文字{Path(path).stem}", "HAPPY")

    housekept = 0

    def housekeeping(self):
        type(self).housekept += 1

    def info(self):
        return [("モデル", "fake")]

    def signature(self):
        return {"model": "fake", "v": 1}


def make_clips(tmp_path, n=3):
    clips = []
    for i in range(n):
        p = tmp_path / f"c{i}.wav"
        sf.write(p, np.zeros(SR * 2, dtype=np.float32), SR)      # 2 秒
        clips.append(Clip(f"c{i}.wav", p, "happy", "x"))
    return clips


def test_run_model_collects_results_and_durations(tmp_path):
    clips = make_clips(tmp_path)
    msgs: list[str] = []
    run = runners.run_model(FakeRec(), clips, None, load_sec=1.5, progress=msgs.append, every=2)
    assert run.results["c1.wav"].text == "文字c1" and run.results["c1.wav"].tag == "HAPPY"
    assert run.durations["c0.wav"] == pytest.approx(2.0)
    assert run.load_sec == 1.5 and run.n_cached == 0 and run.n_errors == 0
    assert run.has_emotion and run.label == "偽モデル"
    assert any("3/3" in m for m in msgs)


def test_run_model_calls_housekeeping_every_50_clips(tmp_path):
    FakeRec.housekept = 0
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(SR, dtype=np.float32), SR)
    clips = [Clip(f"{i}.wav", p, "happy") for i in range(120)]
    runners.run_model(FakeRec(), clips, None, progress=lambda m: None, every=1000)
    assert FakeRec.housekept == 2


def test_run_model_resumes_from_cache_and_ignores_other_settings(tmp_path):
    clips = make_clips(tmp_path)
    cache = tmp_path / "cache" / "fake.jsonl"
    first = FakeRec()
    runners.run_model(first, clips[:2], cache, progress=lambda m: None)
    assert first.calls == ["c0.wav", "c1.wav"]
    # 続きから: 前に処理した 2 件は再利用し、残りだけ処理する
    second = FakeRec()
    run = runners.run_model(second, clips, cache, progress=lambda m: None)
    assert second.calls == ["c2.wav"] and run.n_cached == 2
    assert run.results["c0.wav"].text == "文字c0"
    # 設定 (signature) が変わったら使い回さない
    third = FakeRec()
    third.signature = lambda: {"model": "fake", "v": 2}
    runners.run_model(third, clips, cache, progress=lambda m: None)
    assert len(third.calls) == 3


def test_run_model_survives_truncated_cache_line(tmp_path):
    clips = make_clips(tmp_path, 2)
    cache = tmp_path / "fake.jsonl"
    runners.run_model(FakeRec(), clips[:1], cache, progress=lambda m: None)
    with open(cache, "a", encoding="utf-8") as f:
        f.write('{"file": "c1.wav", "sig": {"model": "fa')       # 途中で止まって切れた行
    rec = FakeRec()
    run = runners.run_model(rec, clips, cache, progress=lambda m: None)
    assert rec.calls == ["c1.wav"] and run.n_cached == 1
    # 切れた行の後ろに書き足した結果が、次の回でも読める (壊れた行にくっついて失われない)
    again = FakeRec()
    run2 = runners.run_model(again, clips, cache, progress=lambda m: None)
    assert again.calls == [] and run2.n_cached == 2


def test_run_model_records_failure_and_retries_next_time(tmp_path):
    clips = make_clips(tmp_path)
    cache = tmp_path / "fake.jsonl"
    run = runners.run_model(FakeRec(fail_on={"c1.wav"}), clips, cache, progress=lambda m: None)
    assert run.n_errors == 1 and "壊れた音声" in run.results["c1.wav"].error
    assert run.results["c0.wav"].error == "" and run.results["c2.wav"].text == "文字c2"
    again = FakeRec()
    run2 = runners.run_model(again, clips, cache, progress=lambda m: None)
    assert again.calls == ["c1.wav"] and run2.n_errors == 0       # 失敗した件だけ、もう一度やる


def test_sensevoice_recognizer_calls_funasr_like_pyvideotrans(tmp_path, monkeypatch):
    calls: dict = {}

    class FakeAutoModel:
        def __init__(self, **kw):
            calls["init"] = kw

        def generate(self, **kw):
            calls["gen"] = kw
            return [{"key": "x", "text": "<|ja|><|SAD|><|Speech|><|withitn|>悲しい ね"}]

    fake_torch = types.SimpleNamespace(set_num_threads=lambda n: calls.setdefault("threads", n))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))
    rec = runners.SenseVoiceRecognizer("iic/SenseVoiceSmall", "auto", "cpu", 3)
    assert "init" not in calls                         # 作っただけでは読み込まない
    out = rec.recognize(tmp_path / "a.wav")
    assert out == runners.Recognition("悲しい ね", "SAD")
    assert calls["init"]["model"] == "iic/SenseVoiceSmall" and calls["init"]["device"] == "cpu" and calls["init"]["disable_update"] is True
    g = calls["gen"]
    assert g["input"].endswith("a.wav") and g["language"] == "auto" and g["use_itn"] is True and g["cache"] == {}
    assert "vad_model" not in calls["init"]            # VAD なし (pyVideoTrans と同じ)
    assert calls["threads"] == 3 and calls["init"]["ncpu"] == 3      # funasr の ncpu (既定4) がスレッド数を上書きしないように
    info = dict(rec.info())
    assert "FunASR" in info["ライセンス"] and "商用" in info["ライセンス"] and "出典" in info["ライセンス"]
    assert rec.signature()["language"] == "auto" and rec.has_emotion


def test_sensevoice_ban_unk_variant_passes_ban_emo_unk(tmp_path, monkeypatch):
    calls: list[dict] = []

    class FakeAutoModel:
        def __init__(self, **kw):
            pass

        def generate(self, **kw):
            calls.append(kw)
            return [{"text": "<|ja|><|ANGRY|><|Speech|><|withitn|>ふざけるな"}]

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))
    plain = runners.SenseVoiceRecognizer("iic/SenseVoiceSmall", "auto", "cpu", 1)
    banned = runners.SenseVoiceRecognizer("iic/SenseVoiceSmall", "auto", "cpu", 1, ban_unk=True)
    plain.recognize(tmp_path / "a.wav")
    banned.recognize(tmp_path / "a.wav")
    assert "ban_emo_unk" not in calls[0] and calls[1]["ban_emo_unk"] is True
    assert (plain.key, banned.key) == ("sensevoice", "sensevoice_nounk")
    assert plain.report_cer and not banned.report_cer          # 同じ文章なので CER は片方だけ載せる
    assert plain.signature() != banned.signature()              # 途中結果の使い回しで混ざらない
    assert "ban_emo_unk=True" in dict(banned.info())["設定"] and "禁止" in banned.label
    run = runners.run_model(banned, make_clips(tmp_path, 1), None, progress=lambda m: None)
    assert run.report_cer is False and run.key == "sensevoice_nounk" and run.results["c0.wav"].tag == "ANGRY"


def test_sensevoice_missing_funasr_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "funasr", None)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda n: None))
    rec = runners.SenseVoiceRecognizer()
    with pytest.raises(InputError, match="uv run --with funasr"):
        rec.prepare()
    # 読み込みの失敗は、1本ずつの失敗にせず、最初に止める
    with pytest.raises(InputError, match="funasr"):
        runners.run_model(rec, [Clip("x.wav", Path("x.wav"), "happy")], None, progress=lambda m: None)


def test_whisper_recognizer_uses_asrcheck_engine(tmp_path, monkeypatch):
    made: dict = {}

    class FakeModel:
        def __init__(self, model_id, device, compute_type):
            made.update(model_id=model_id, device=device, compute_type=compute_type)

        def transcribe(self, wave, **kw):
            made["kw"] = kw
            return iter([types.SimpleNamespace(text="こんにちは"), types.SimpleNamespace(text="。")]), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(SR, dtype=np.float32), SR)
    rec = runners.WhisperRecognizer("large-v3-turbo", "ja", "cpu", 2)
    assert "model_id" not in made                      # 作っただけでは読み込まない
    assert rec.recognize(p) == runners.Recognition("こんにちは。")
    assert made["model_id"] == "large-v3-turbo" and made["device"] == "cpu" and made["compute_type"] == "int8"
    assert made["kw"]["language"] == "ja" and made["kw"]["beam_size"] == 5 and "vad_filter" not in made["kw"]
    assert not rec.has_emotion and "MIT" in dict(rec.info())["ライセンス"]


def test_audio_seconds(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(SR * 3, dtype=np.float32), SR)
    assert runners.audio_seconds(p) == pytest.approx(3.0)
    assert runners.audio_seconds(tmp_path / "none.wav") == 0.0


def test_cache_lines_are_valid_json(tmp_path):
    clips = make_clips(tmp_path, 1)
    cache = tmp_path / "c.jsonl"
    runners.run_model(FakeRec(), clips, cache, progress=lambda m: None)
    d = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert d["file"] == "c0.wav" and d["tag"] == "HAPPY" and d["sig"] == {"model": "fake", "v": 1}


def test_run_model_does_not_load_model_when_everything_is_cached(tmp_path):
    clips = make_clips(tmp_path, 2)
    cache = tmp_path / "fake.jsonl"
    runners.run_model(FakeRec(), clips, cache, progress=lambda m: None)

    class Loader(FakeRec):
        prepared = 0

        def prepare(self):
            type(self).prepared += 1

    run = runners.run_model(Loader(), clips, cache, progress=lambda m: None)
    assert Loader.prepared == 0 and run.n_cached == 2
    runners.run_model(Loader(), make_clips(tmp_path, 3), cache, progress=lambda m: None)
    assert Loader.prepared == 1                          # 新しい音声があるときだけ読み込む


def test_threads_are_recorded_with_each_result_and_survive_the_cache(tmp_path):
    clips = make_clips(tmp_path, 2)
    cache = tmp_path / "t.jsonl"
    rec = FakeRec()
    rec.threads = 3
    run = runners.run_model(rec, clips, cache, progress=lambda m: None)
    assert {r.threads for r in run.results.values()} == {3}
    again = runners.run_model(FakeRec(), clips, cache, progress=lambda m: None)         # 再利用でも残る
    assert {r.threads for r in again.results.values()} == {3}
    # スレッド数を書いていない古い途中結果も読める (0 = 記録なし)
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"file": "c0.wav", "sig": {"model": "fake", "v": 1}, "text": "x", "tag": "", "sec": 1.0}) + "\n", encoding="utf-8")
    assert runners._load_cache(old, {"model": "fake", "v": 1})["c0.wav"].threads == 0
