"""Chatterbox Multilingual（Resemble AI、MIT）のアダプタ。

公式 API の確認（2026-10-18、いずれも自分で取得して読んだ）:
- README  https://github.com/resemble-ai/chatterbox/blob/master/README.md
- ソース  PyPI chatterbox-tts 0.1.7（2026-03-26）の wheel の chatterbox/mtl_tts.py を直接読んだ
          https://pypi.org/project/chatterbox-tts/0.1.7/
- モデル  https://huggingface.co/ResembleAI/chatterbox （license: mit）

確認できた使い方（0.1.7）:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")        # 0.1.7 の引数は device だけ
    wav = model.generate(text, language_id="fr", audio_prompt_path=<手本wav>,
                         exaggeration=0.5, cfg_weight=0.5)                   # → torch (1, N)。サンプルレートは model.sr（24 kHz）
- language_id は ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh の 23 言語。
- 書き起こしは使わない（手本の音声だけ）。
- README の注意: 「手本の言語と language_id が違うと、手本のなまりが移る。cfg_weight=0 にすると和らぐ」→ 言語をまたぐ時は cfg_weight=0 にする。
- 感情の種類は指定できない。強さだけ exaggeration（README：表現豊かにするなら exaggeration 0.7 以上・cfg_weight 0.3 前後）。
- README（master）には `from_pretrained(device=..., t3_model="v3")`（Multilingual V3）の記述があるが、PyPI の 0.1.7 の
  from_pretrained にはその引数が無い（V2 = t3_mtl23ls_v2.safetensors のみ）。→ t3_model は「引数があるときだけ」渡す。
- 出力には Perth の透かしが入る。
- 注意: chatterbox-tts 0.1.7 は torch==2.6.0・transformers==5.2.0 を固定で要求する（qwen-tts の transformers==4.57.3 と衝突）。
  → ノートブックではエンジンごとに別の仮想環境（venv）に入れる。
未確認（GPU が無く動かしていない）: 実際の合成、T4 でのメモリ・速度、t3_model="v3" の挙動（GitHub master のみ）。
"""
from __future__ import annotations

import inspect
from typing import Any

from .base import MODE_INSTRUCT, MODE_SIMILARITY, Engine, EngineError, SynthResult, emotion_text, pkg_version, seed_everything, to_mono_float32

_LANGS = frozenset("ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split())


class ChatterboxEngine(Engine):
    name = "chatterbox"
    LANGS = _LANGS
    emotion_modes = (MODE_INSTRUCT, MODE_SIMILARITY)  # 強さ（exaggeration）だけ

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as e:
            raise EngineError(f"chatterbox-tts が入っていません（{e}）。Colab: pip install chatterbox-tts==0.1.7") from e
        device = self.opt("device", "cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {"device": device}
        t3 = self.opt("t3_model", None)
        self._note_load = ""
        if t3:
            if "t3_model" in inspect.signature(ChatterboxMultilingualTTS.from_pretrained).parameters:
                kwargs["t3_model"] = t3
            else:
                self._note_load = f"この版の from_pretrained には t3_model 引数が無いので無視しました（{t3}）"
        try:
            self._m_model = ChatterboxMultilingualTTS.from_pretrained(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"Chatterbox を読み込めません（{type(e).__name__}: {str(e)[:300]}）") from e
        self._loaded = True

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
        if lang not in _LANGS:
            raise EngineError(f"Chatterbox は {lang} に対応していません")
        exaggeration = float(self.opt("exaggeration", 0.5))
        cfg = float(self.opt("cfg_weight", 0.5))
        notes = [self._note_load] if getattr(self, "_note_load", "") else []
        mode = "clone"
        if emotion_text(emotion):
            exaggeration = float(self.opt("emotion_exaggeration", 0.8))
            cfg = float(self.opt("emotion_cfg_weight", 0.3))
            mode = "clone+exaggeration"
        if ref_lang and ref_lang != lang and self.opt("cfg_weight_cross_lang", 0.0) is not None:
            cfg = float(self.opt("cfg_weight_cross_lang", 0.0))  # README：言語をまたぐ時は 0
            notes.append("言語をまたぐので cfg_weight を下げました")
        seed_everything(seed)
        try:
            wav = self._m_model.generate(
                text, language_id=lang, audio_prompt_path=ref_wav, exaggeration=exaggeration, cfg_weight=cfg
            )
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"Chatterbox の合成に失敗（{type(e).__name__}: {str(e)[:300]}）") from e
        return SynthResult(to_mono_float32(wav), int(self._m_model.sr), mode, " / ".join(notes))

    def info(self) -> dict[str, str]:
        return {"engine": self.spec, "chatterbox-tts": pkg_version("chatterbox-tts"), "transformers": pkg_version("transformers")}
