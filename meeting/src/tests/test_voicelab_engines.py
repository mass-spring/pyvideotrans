"""voicelab：各 TTS アダプタ。本物のモデルは使わず、公式 API と同じ形の「偽の部品」を差し込んで、
呼び方（引数の名前・方式の切り替え・サンプルレート・言語名）が公式 README どおりかを確かめる。"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from voicelab.engines import (
    FORBIDDEN,
    available_engines,
    create_engine,
    engine_caps,
    engine_class,
    parse_engine_spec,
    validate_engine_spec,
)
from voicelab.engines.base import EngineError, emotion_intensity, emotion_text, to_mono_float32


_FAKE_PREFIXES = ("voxcpm", "qwen_tts", "chatterbox", "pykakasi", "cosyvoice")


@pytest.fixture
def isolate_modules():
    """テストで差し込む偽モジュール（voxcpm・qwen_tts・chatterbox・pykakasi・cosyvoice）と sys.path だけを、終わったら元に戻す。
    torch などの本物の C 拡張モジュールは触らない（消すと次の import で落ちる）。"""

    def is_fake(name: str) -> bool:
        return name.split(".")[0] in _FAKE_PREFIXES

    saved = {k: v for k, v in sys.modules.items() if is_fake(k)}
    path = list(sys.path)
    for k in saved:
        del sys.modules[k]
    yield
    for k in [k for k in sys.modules if is_fake(k)]:
        del sys.modules[k]
    sys.modules.update(saved)
    sys.path[:] = path


# ---- 登録・性質・ライセンス ---------------------------------------------------------------------------------
def test_registry_has_exactly_the_four_commercial_engines():
    assert available_engines() == ["voxcpm2", "qwen3tts", "cosyvoice3", "chatterbox"]
    for bad in ("f5tts", "omnivoice", "higgs", "vevo", "indextts"):
        with pytest.raises(EngineError):
            engine_class(bad)
    assert set(FORBIDDEN) >= {"f5tts", "omnivoice", "higgs", "vevo", "indextts"}


def test_importing_adapters_does_not_import_heavy_libraries():
    code = (
        "import sys, voicelab.cli, voicelab.scoring, voicelab.engines.voxcpm2, voicelab.engines.qwen3tts, "
        "voicelab.engines.cosyvoice3, voicelab.engines.chatterbox, voicelab.engines.fake\n"
        "bad = [m for m in ('torch','torchaudio','voxcpm','qwen_tts','chatterbox','cosyvoice','transformers','librosa') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_spec_parsing_and_capabilities():
    assert parse_engine_spec("CosyVoice3:Zero_Shot") == ("cosyvoice3", "zero_shot")
    assert parse_engine_spec("voxcpm2") == ("voxcpm2", "")
    validate_engine_spec("cosyvoice3:instruct2")
    with pytest.raises(EngineError):
        validate_engine_spec("cosyvoice3:nope")
    assert engine_caps("voxcpm2").emotion_modes == ("instruct", "refonly", "similarity")
    assert engine_caps("qwen3tts").emotion_modes == ()
    assert engine_caps("cosyvoice3:instruct2").emotion_only and engine_caps("cosyvoice3:instruct2").emotion_modes == ("instruct",)
    assert not engine_caps("cosyvoice3").emotion_only
    assert {"ja", "en", "fr", "zh"} <= set(engine_caps("voxcpm2").langs)
    for e in ("qwen3tts", "cosyvoice3", "chatterbox"):
        assert {"ja", "en", "fr", "zh"} <= set(engine_caps(e).langs)


def test_emotion_helpers():
    assert emotion_text(None) == "" and emotion_text(" angry ") == "angry"
    assert emotion_text({"label": "sad", "instruction": "sad, slower"}) == "sad, slower"
    assert emotion_intensity({"intensity": 3}) == 1.0 and emotion_intensity(None, 0.4) == 0.4
    assert to_mono_float32(np.ones((1, 10))).shape == (10,)
    assert to_mono_float32(np.ones((2, 10))).shape == (10,)


def test_dry_run_engine_copies_capabilities_and_returns_audio(tmp_path):
    import soundfile as sf

    from synth import voice_like

    ref = tmp_path / "r.wav"
    sf.write(ref, voice_like(0, 1, dur=3.0), 16000)
    eng = create_engine("cosyvoice3:instruct2", None, dry_run=True)
    eng.load()
    assert eng.emotion_only and eng.spec == "cosyvoice3:instruct2" and eng.supports_lang("ja") and not eng.supports_lang("xx")
    res = eng.synthesize("テストの文です。", "ja", str(ref), None, "angry", seed=1)
    assert res.sr == 16000 and res.wav.ndim == 1 and res.wav.size > 8000 and res.mode == "dry_run"
    # 感情の指示を入れると、ダミーの声の高さが変わる（条件の差が出ることの確認用）
    res2 = eng.synthesize("テストの文です。", "ja", str(ref), None, None, seed=1)
    assert not np.allclose(res.wav[:8000], res2.wav[:8000])


# ---- VoxCPM2 -------------------------------------------------------------------------------------------------
def _fake_voxcpm(with_seed: bool):
    calls: list[dict] = []
    loads: list[dict] = []

    if with_seed:
        def _generate(self, text, prompt_wav_path=None, prompt_text=None, reference_wav_path=None, cfg_value=2.0,
                      inference_timesteps=10, seed=None):  # noqa: D401
            pass
    else:
        def _generate(self, text, prompt_wav_path=None, prompt_text=None, reference_wav_path=None, cfg_value=2.0,
                      inference_timesteps=10):
            pass

    class Model:
        tts_model = types.SimpleNamespace(sample_rate=48000)

        def generate(self, **kw):
            calls.append(kw)
            return np.zeros(4800, dtype=np.float32)

    Model._generate = _generate

    class VoxCPM:
        @classmethod
        def from_pretrained(cls, model_id, **kw):
            loads.append({"id": model_id, **kw})
            return Model()

    return types.SimpleNamespace(VoxCPM=VoxCPM), calls, loads


def test_voxcpm2_modes_follow_decision_d13(isolate_modules):
    mod, calls, loads = _fake_voxcpm(with_seed=True)
    sys.modules["voxcpm"] = mod
    eng = create_engine("voxcpm2", {"optimize": False})
    # 書き起こしあり・指示なし → Ultimate Cloning（prompt + reference）
    r = eng.synthesize("Hello.", "en", "/r.wav", "手本の書き起こし", seed=7)
    assert loads[0]["id"] == "openbmb/VoxCPM2" and loads[0]["load_denoiser"] is False and loads[0]["optimize"] is False
    assert r.sr == 48000 and r.mode == "ultimate_clone" and r.wav.dtype == np.float32
    assert calls[0]["prompt_wav_path"] == "/r.wav" and calls[0]["prompt_text"] == "手本の書き起こし"
    assert calls[0]["reference_wav_path"] == "/r.wav" and calls[0]["text"] == "Hello." and calls[0]["seed"] == 7
    # 書き起こしなし → 手本だけ（Controllable Cloning）、指示なし
    r = eng.synthesize("Hello.", "en", "/r.wav", None)
    assert r.mode == "controllable_clone" and "prompt_text" not in calls[1] and calls[1]["text"] == "Hello."
    # 感情の指示 → 文頭に (指示)、書き起こしは使わない
    r = eng.synthesize("Hello.", "en", "/r.wav", "手本の書き起こし", "angry, tense")
    assert r.mode == "controllable_clone+instruction" and calls[2]["text"] == "(angry, tense)Hello."
    assert "prompt_text" not in calls[2] and "prompt_wav_path" not in calls[2] and "書き起こし" in r.note


def test_voxcpm2_seed_is_replaced_when_the_installed_version_has_no_seed_argument(isolate_modules):
    """PyPI の voxcpm 2.0.3 の generate には seed 引数が無い（GitHub main には有る）。無ければ渡さない。"""
    mod, calls, _ = _fake_voxcpm(with_seed=False)
    sys.modules["voxcpm"] = mod
    eng = create_engine("voxcpm2")
    eng.synthesize("Hello.", "en", "/r.wav", None, seed=5)
    assert "seed" not in calls[0]


def test_voxcpm2_missing_package_is_a_friendly_error(isolate_modules):
    sys.modules.pop("voxcpm", None)
    sys.modules["voxcpm"] = None  # import を失敗させる
    with pytest.raises(EngineError) as e:
        create_engine("voxcpm2").load()
    assert "voxcpm==2.0.3" in str(e.value)


# ---- Qwen3-TTS -----------------------------------------------------------------------------------------------
def test_qwen3tts_uses_language_names_and_prompt_cache(isolate_modules):
    pytest.importorskip("torch")
    seen: dict = {"prompts": [], "gen": [], "load": None}

    class M:
        @classmethod
        def from_pretrained(cls, name, **kw):
            seen["load"] = (name, kw)
            return cls()

        def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
            seen["prompts"].append((ref_audio, ref_text, x_vector_only_mode))
            return ["P"]

        def generate_voice_clone(self, text, language=None, voice_clone_prompt=None, **kw):
            seen["gen"].append((text, language, voice_clone_prompt))
            return [np.zeros(2400, dtype=np.float32)], 24000

    sys.modules["qwen_tts"] = types.SimpleNamespace(Qwen3TTSModel=M)
    eng = create_engine("qwen3tts")
    r = eng.synthesize("Bonjour.", "fr", "/r.wav", "手本")
    assert seen["load"][0] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base" and seen["load"][1]["attn_implementation"] in ("sdpa", "flash_attention_2")
    assert seen["gen"][0] == ("Bonjour.", "French", ["P"]) and r.sr == 24000 and r.mode == "icl_clone"
    assert seen["prompts"][0] == ("/r.wav", "手本", False)
    eng.synthesize("Hello.", "en", "/r.wav", "手本")
    assert len(seen["prompts"]) == 1  # 同じ手本は 1 回だけ前処理
    r = eng.synthesize("你好。", "zh", "/r2.wav", None, "angry")  # 書き起こしなし → x-vector だけ。感情は無視
    assert seen["prompts"][1] == ("/r2.wav", None, True) and seen["gen"][-1][1] == "Chinese"
    assert r.mode == "x_vector_only" and "無視" in r.note and "x_vector" in r.note
    assert [seen["gen"][i][1] for i in (0, 1)] == ["French", "English"]
    with pytest.raises(EngineError):
        eng.synthesize("x", "ar", "/r.wav", None)


# ---- Chatterbox ----------------------------------------------------------------------------------------------
def test_chatterbox_arguments(isolate_modules):
    torch = pytest.importorskip("torch")
    calls: list[dict] = []
    loads: list[dict] = []

    class M:
        sr = 24000

        @classmethod
        def from_pretrained(cls, device):  # 0.1.7 と同じ：t3_model 引数は無い
            loads.append({"device": device})
            return cls()

        def generate(self, text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5):
            calls.append(dict(text=text, language_id=language_id, audio_prompt_path=audio_prompt_path,
                              exaggeration=exaggeration, cfg_weight=cfg_weight))
            return torch.zeros(1, 2400)

    pkg = types.ModuleType("chatterbox")
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = M
    sys.modules["chatterbox"], sys.modules["chatterbox.mtl_tts"] = pkg, mtl
    eng = create_engine("chatterbox", {"t3_model": "v3"})
    r = eng.synthesize("Bonjour.", "fr", "/r.wav", "使わない", ref_lang="fr")
    assert calls[0] == dict(text="Bonjour.", language_id="fr", audio_prompt_path="/r.wav", exaggeration=0.5, cfg_weight=0.5)
    assert r.sr == 24000 and r.wav.shape == (2400,) and "t3_model" in r.note  # 古い版では t3_model を無視して注意を残す
    eng.synthesize("Bonjour.", "fr", "/r.wav", None, ref_lang="ja")  # 言語をまたぐ → cfg_weight=0（README）
    assert calls[1]["cfg_weight"] == 0.0
    r = eng.synthesize("Bonjour.", "fr", "/r.wav", None, "angry", ref_lang="fr")  # 感情 → exaggeration を上げる
    assert calls[2]["exaggeration"] == 0.8 and calls[2]["cfg_weight"] == 0.3 and r.mode == "clone+exaggeration"
    with pytest.raises(EngineError):
        eng.synthesize("x", "xx", "/r.wav", None)


# ---- CosyVoice3 ----------------------------------------------------------------------------------------------
def _fake_cosy_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "CosyVoice"
    pkg = repo / "cosyvoice" / "cli"
    pkg.mkdir(parents=True)
    (repo / "cosyvoice" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "cosyvoice.py").write_text(
        "import torch\n"
        "CALLS = []\n"
        "class _M:\n"
        "    sample_rate = 24000\n"
        "    def __init__(self, model_dir, fp16=False):\n"
        "        self.model_dir, self.fp16 = model_dir, fp16\n"
        "    def _out(self):\n"
        "        yield {'tts_speech': torch.zeros(1, 1200)}\n"
        "        yield {'tts_speech': torch.zeros(1, 1200)}\n"
        "    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):\n"
        "        CALLS.append(('zero_shot', tts_text, prompt_text, prompt_wav, stream)); return self._out()\n"
        "    def inference_cross_lingual(self, tts_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):\n"
        "        CALLS.append(('cross_lingual', tts_text, prompt_wav, stream)); return self._out()\n"
        "    def inference_instruct2(self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):\n"
        "        CALLS.append(('instruct2', tts_text, instruct_text, prompt_wav, stream)); return self._out()\n"
        "def AutoModel(**kw):\n"
        "    return _M(**kw)\n",
        encoding="utf-8")
    (repo / "third_party" / "Matcha-TTS").mkdir(parents=True)
    (repo / "pretrained_models" / "Fun-CosyVoice3-0.5B").mkdir(parents=True)  # あれば snapshot_download しない
    return repo


def test_cosyvoice3_three_modes(tmp_path, isolate_modules):
    pytest.importorskip("torch")
    repo = _fake_cosy_repo(tmp_path)
    opts = {"repo_dir": str(repo), "ja_katakana": False}
    P, E = "You are a helpful assistant.", "<|endofprompt|>"

    zs = create_engine("cosyvoice3:zero_shot", opts)
    r = zs.synthesize("Hello.", "en", "/r.wav", "手本の文", ref_lang="ja")
    import cosyvoice.cli.cosyvoice as cv

    assert cv.CALLS[-1] == ("zero_shot", "Hello.", f"{P}{E}手本の文", "/r.wav", False)
    assert r.sr == 24000 and r.wav.shape == (2400,) and r.mode == "zero_shot"  # 2 つに分かれて返る音声を連結
    r = zs.synthesize("Hello.", "en", "/r.wav", None)  # 書き起こしなし → cross_lingual に切替
    assert cv.CALLS[-1] == ("cross_lingual", f"{P}{E}Hello.", "/r.wav", False) and r.mode == "cross_lingual" and "切り替え" in r.note

    cl = create_engine("cosyvoice3:cross_lingual", opts)
    cl.synthesize("Bonjour.", "fr", "/r.wav", "無視される")
    assert cv.CALLS[-1] == ("cross_lingual", f"{P}{E}Bonjour.", "/r.wav", False)

    inst = create_engine("cosyvoice3:instruct2", opts)
    inst.synthesize("Nein!", "de", "/r.wav", "x", "angry, tense, slightly faster")
    assert cv.CALLS[-1] == ("instruct2", "Nein!", f"{P} 请非常生气地说一句话。{E}", "/r.wav", False)
    with pytest.raises(EngineError):
        inst.synthesize("x", "en", "/r.wav", None, None)  # 感情の指示なしは使えない
    with pytest.raises(EngineError):
        inst.synthesize("x", "en", "/r.wav", None, "mysterious")  # 公式に無い指示は作らない


def test_cosyvoice3_without_repo_is_a_friendly_error(tmp_path):
    with pytest.raises(EngineError) as e:
        create_engine("cosyvoice3", {"repo_dir": str(tmp_path / "none")}).load()
    assert "git clone --recursive" in str(e.value)


def test_cosyvoice_instruction_mapping():
    from voicelab.engines.cosyvoice3 import map_instruction

    assert "生气" in map_instruction("angry, tense, slightly faster")
    assert "开心" in map_instruction("cheerful, bright, slightly faster")
    assert "伤心" in map_instruction("sad, subdued, slower")
    assert "快" in map_instruction("slightly faster")
    with pytest.raises(EngineError):
        map_instruction("sarcastic")


def test_katakana_conversion_with_and_without_pykakasi(isolate_modules):
    from voicelab.engines.cosyvoice3 import to_katakana

    sys.modules["pykakasi"] = None  # 入っていない状態
    out, note = to_katakana("私は行きます。")
    assert out == "私は行きます。" and "pykakasi" in note

    class K:
        def convert(self, text):
            return [{"orig": "私", "kana": "ワタシ"}, {"orig": "は", "kana": "ハ"}, {"orig": "行き", "kana": "イキ"},
                    {"orig": "ます", "kana": "マス"}, {"orig": "。", "kana": "。"}]

    sys.modules["pykakasi"] = types.SimpleNamespace(kakasi=lambda: K())
    out, note = to_katakana("私は行きます。")
    assert out == "ワタシ ワ イキ マス。" and note == ""
