"""Qwen3-TTS-12Hz-1.7B-Base（声の複製用、Apache-2.0）のアダプタ。

公式 API の確認（2026-10-18、いずれも自分で取得して読んだ）:
- README  https://github.com/QwenLM/Qwen3-TTS/blob/main/README.md （"Voice Clone" の節）
- ソース  https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
          （PyPI の wheel 0.1.1 の中身でも create_voice_clone_prompt / generate_voice_clone を確認）
- PyPI    https://pypi.org/project/qwen-tts/ （最新 0.1.1 = 2026-02-06。transformers==4.57.3・accelerate==1.12.0 を固定で要求）
- モデル  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base （license: apache-2.0）

確認できた使い方:
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                          device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    items = model.create_voice_clone_prompt(ref_audio=<wav パス>, ref_text=<書き起こし>, x_vector_only_mode=False)
    wavs, sr = model.generate_voice_clone(text=..., language="Japanese", voice_clone_prompt=items)   # wavs[0] が波形
- language は英語名（Chinese / English / Japanese / Korean / German / French / Russian / Portuguese / Spanish / Italian）か "Auto"。
  対応外の名前は ValueError（get_supported_languages() で確認できる）。
- ref_text が無いときは x_vector_only_mode=True（話者埋め込みだけ。公式：「複製の質は下がりうる」）。
- Base モデルに感情・スタイルの指示はない（research.md 11.1）→ emotion は無視する。
- flash-attn は Ampere 以降（T4 は Turing なので不可）。modeling_qwen3_tts.py に `_supports_sdpa = True` があるので、
  T4 では attn_implementation="sdpa"、dtype は bfloat16 非対応の GPU なら float16 にする。
未確認（GPU が無く動かしていない）: 実際の合成、T4 でのメモリ・速度、float16 での音質。
"""
from __future__ import annotations

from typing import Any

from .base import Engine, EngineError, SynthResult, pkg_version, seed_everything, to_mono_float32

LANG_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German",
    "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}


class Qwen3TTSEngine(Engine):
    name = "qwen3tts"
    LANGS = frozenset(LANG_NAMES)
    emotion_modes = ()  # Base には指示機能が無い

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise EngineError(f"qwen-tts が入っていません（{e}）。Colab: pip install qwen-tts==0.1.1") from e
        dtype_opt = str(self.opt("dtype", "auto"))
        if dtype_opt == "auto":
            cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
            dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_opt]
        attn = str(self.opt("attn_implementation", "auto"))
        if attn == "auto":
            try:
                import flash_attn  # noqa: F401

                attn = "flash_attention_2"
            except ImportError:
                attn = "sdpa"
        model_id = self.opt("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        device_map = self.opt("device_map", "cuda:0" if torch.cuda.is_available() else "cpu")
        last: Exception | None = None
        for a in dict.fromkeys([attn, "sdpa"]):
            try:
                self._m_model = Qwen3TTSModel.from_pretrained(
                    model_id, device_map=device_map, dtype=dtype, attn_implementation=a
                )
                self._attn = a
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if last is not None:
            raise EngineError(f"Qwen3-TTS を読み込めません（{type(last).__name__}: {str(last)[:300]}）") from last
        self._prompts: dict[tuple[str, str], Any] = {}
        self._loaded = True

    def _prompt(self, ref_wav: str, ref_text: str | None):
        key = (ref_wav, ref_text or "")
        if key not in self._prompts:
            self._prompts[key] = self._m_model.create_voice_clone_prompt(
                ref_audio=ref_wav, ref_text=ref_text or None, x_vector_only_mode=not bool(ref_text)
            )
        return self._prompts[key]

    def synthesize(
        self,
        text: str,
        lang: str,
        ref_wav: str,
        ref_text: str | None,
        emotion: Any = None,
        *,
        ref_lang: str | None = None,
        seed: int | None = None,
    ) -> SynthResult:
        if not self._loaded:
            self.load()
        if lang not in LANG_NAMES:
            raise EngineError(f"Qwen3-TTS は {lang} に対応していません（対応: {', '.join(LANG_NAMES)}）")
        notes = []
        if emotion:
            notes.append("Qwen3-TTS Base は感情の指示に対応していないので無視しました")
        if not ref_text:
            notes.append("書き起こしが無いので話者埋め込みだけで複製しました（x_vector_only_mode）")
        seed_everything(seed)
        try:
            items = self._prompt(ref_wav, ref_text)
            wavs, sr = self._m_model.generate_voice_clone(
                text=text, language=LANG_NAMES[lang], voice_clone_prompt=items
            )
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"Qwen3-TTS の合成に失敗（{type(e).__name__}: {str(e)[:300]}）") from e
        mode = "icl_clone" if ref_text else "x_vector_only"
        return SynthResult(to_mono_float32(wavs[0]), int(sr), mode, " / ".join(notes))

    def close(self) -> None:
        self._prompts = {}
        super().close()

    def info(self) -> dict[str, str]:
        return {
            "engine": self.spec,
            "qwen-tts": pkg_version("qwen-tts"),
            "transformers": pkg_version("transformers"),
            "model": str(self.opt("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")),
            "attn": getattr(self, "_attn", ""),
        }
