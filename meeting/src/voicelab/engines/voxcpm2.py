"""VoxCPM2（openbmb/VoxCPM2、Apache-2.0）のアダプタ。

公式 API の確認（2026-10-18、いずれも自分で取得して読んだ）:
- README  https://github.com/OpenBMB/VoxCPM/blob/main/README.md
- ソース  https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/core.py
- PyPI    https://pypi.org/project/voxcpm/  （最新 2.0.3 = 2026-05-11。requires_python>=3.10。README は Python 3.10〜3.12、PyTorch>=2.5、CUDA>=12）
- モデル  https://huggingface.co/openbmb/VoxCPM2 （license: apache-2.0）

確認できた使い方:
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=..., device=...)
    wav = model.generate(text=..., cfg_value=2.0, inference_timesteps=10,
                         reference_wav_path=..., prompt_wav_path=..., prompt_text=...)   # → 1 次元 float32 numpy
    sf.write(path, wav, model.tts_model.sample_rate)          # VoxCPM2 は 48 kHz（README）
- 声の複製は 3 通り:
    (a) 手本だけ      : reference_wav_path のみ（Controllable Cloning。文の頭に "(指示)" を付けて感情・速さを指示できる）
    (b) 手本＋書き起こし : prompt_wav_path + prompt_text（+ 同じ wav を reference_wav_path にも）＝ Ultimate Cloning。
                         続きを話す方式なので、指示（括弧書き）は効かない（research.md 14章：公式アプリは指示を捨てる）。
    (c) 手本なし      : 文の頭に "(声の説明)" ＝ Voice Design（この実験では使わない）
- 言語タグは不要（30 言語を文字から判断。ja / en / fr / zh を含む）。
- 注意（自分で確認した差）: PyPI の 2.0.3 の generate() には `seed` 引数が無い（GitHub の main には有る）。
  → 引数を調べて、あれば渡し、無ければ torch.manual_seed で代用する。
未確認（GPU が無く、このセッションでは動かしていない）: 実際の合成、T4 でのメモリ・速度、optimize=True の挙動。
"""
from __future__ import annotations

import inspect
from typing import Any

from .base import (
    MODE_INSTRUCT,
    MODE_REFONLY,
    MODE_SIMILARITY,
    Engine,
    EngineError,
    SynthResult,
    emotion_text,
    pkg_version,
    seed_everything,
    to_mono_float32,
)

# README の「30 languages」を 2 文字コードにしたもの
_LANGS = frozenset(
    "ar my zh da nl en fi fr de el he hi id it ja km ko lo ms no pl pt ru es sw sv tl th tr vi".split()
)


class VoxCPM2Engine(Engine):
    name = "voxcpm2"
    LANGS = _LANGS
    emotion_modes = (MODE_INSTRUCT, MODE_REFONLY, MODE_SIMILARITY)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from voxcpm import VoxCPM
        except ImportError as e:
            raise EngineError(
                f"voxcpm が入っていません（{e}）。Colab: pip install voxcpm==2.0.3"
            ) from e
        model_id = self.opt("model_id", "openbmb/VoxCPM2")
        try:
            self._m_model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=False,                       # ノイズ除去は使わない（類似度が下がる報告があるため）
                optimize=bool(self.opt("optimize", False)),  # torch.compile。既定の True は初回が遅く不安定になりうるので切る
                device=self.opt("device", None),
            )
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"VoxCPM2 を読み込めません（{type(e).__name__}: {str(e)[:300]}）") from e
        try:
            self._seed_ok = "seed" in inspect.signature(self._m_model._generate).parameters
        except Exception:  # noqa: BLE001
            self._seed_ok = False
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
        model = self._m_model
        kwargs: dict[str, Any] = {
            "cfg_value": float(self.opt("cfg_value", 2.0)),
            "inference_timesteps": int(self.opt("inference_timesteps", 10)),
        }
        if seed is not None:
            if self._seed_ok:
                kwargs["seed"] = int(seed)
            else:
                seed_everything(int(seed))

        instr = emotion_text(emotion).replace("(", " ").replace(")", " ").strip()
        forced = str(self.opt("mode", "auto"))  # auto / ultimate / controllable
        note = ""
        if instr:
            # 感情の指示を使う行は、書き起こしを使わない複製にする（D13：emotion 優先。公式アプリと同じ）
            mode = "controllable_clone+instruction"
            call = {"text": f"({instr}){text}", "reference_wav_path": ref_wav}
            if ref_text:
                note = "感情の指示を使うため書き起こしは使っていません（D13）"
        elif ref_text and forced != "controllable":
            mode = "ultimate_clone"
            call = {"text": text, "prompt_wav_path": ref_wav, "prompt_text": ref_text, "reference_wav_path": ref_wav}
        else:
            mode = "controllable_clone"
            call = {"text": text, "reference_wav_path": ref_wav}
            if forced == "ultimate" and not ref_text:
                note = "書き起こしが無いので手本だけの複製に切り替えました"
        try:
            wav = model.generate(**call, **kwargs)
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"VoxCPM2 の合成に失敗（{type(e).__name__}: {str(e)[:300]}）") from e
        sr = int(model.tts_model.sample_rate)
        return SynthResult(to_mono_float32(wav), sr, mode, note)

    def info(self) -> dict[str, str]:
        return {"engine": self.spec, "voxcpm": pkg_version("voxcpm"), "model": str(self.opt("model_id", "openbmb/VoxCPM2"))}
