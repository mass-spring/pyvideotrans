"""TTS エンジンの登録と生成。

エンジンの指定は `名前` または `名前:variant`（例 cosyvoice3:zero_shot）。
重いライブラリは、各アダプタの load() を呼ぶまで import されない
（このファイルとアダプタのファイルを import するだけなら何も重くならない）。

使わないもの（ライセンス）: F5-TTS・OmniVoice・Higgs Audio・Vevo は重みが非商用、
IndexTTS は大企業は別契約（CLAUDE.md のデータ・ライセンス規則）。登録しないだけでなく、指定されたら止める。
"""
from __future__ import annotations

import importlib
from typing import Any

from .base import Caps, Engine, EngineError, SynthResult  # noqa: F401  (SynthResult は再公開)

# 名前 → (モジュール, クラス名)
_REGISTRY: dict[str, tuple[str, str]] = {
    "voxcpm2": ("voicelab.engines.voxcpm2", "VoxCPM2Engine"),
    "qwen3tts": ("voicelab.engines.qwen3tts", "Qwen3TTSEngine"),
    "cosyvoice3": ("voicelab.engines.cosyvoice3", "CosyVoice3Engine"),
    "chatterbox": ("voicelab.engines.chatterbox", "ChatterboxEngine"),
}

# 商用利用できない（または別契約が要る）ので使わないもの
FORBIDDEN = {
    "f5tts": "F5-TTS の重みは非商用（CC-BY-NC）",
    "f5-tts": "F5-TTS の重みは非商用（CC-BY-NC）",
    "omnivoice": "OmniVoice の重みは非商用（CC-BY-NC）",
    "higgs": "Higgs Audio は研究・非商用",
    "vevo": "Vevo は非商用（CC-BY-NC）",
    "indextts": "IndexTTS は大企業は別契約が必要",
    "index-tts": "IndexTTS は大企業は別契約が必要",
}


def parse_engine_spec(spec: str) -> tuple[str, str]:
    """'cosyvoice3:zero_shot' → ('cosyvoice3', 'zero_shot')。'voxcpm2' → ('voxcpm2', '')。"""
    name, _, variant = str(spec).strip().partition(":")
    return name.strip().lower(), variant.strip().lower()


def available_engines() -> list[str]:
    return list(_REGISTRY)


def engine_class(name: str) -> type[Engine]:
    name = name.strip().lower()
    for bad, why in FORBIDDEN.items():
        if bad in name.replace("_", "-"):
            raise EngineError(f"{name} は使えません（{why}）。CLAUDE.md の規則：商用利用できないモデルは使わない")
    if name not in _REGISTRY:
        raise EngineError(f"知らないエンジン '{name}' です。選べるもの: {', '.join(_REGISTRY)}")
    mod, cls = _REGISTRY[name]
    return getattr(importlib.import_module(mod), cls)


def validate_engine_spec(spec: str) -> None:
    """指定が正しいか確認する（モデルは読み込まない）。間違いなら EngineError。"""
    name, variant = parse_engine_spec(spec)
    cls = engine_class(name)
    if variant not in cls.VARIANTS:
        raise EngineError(
            f"{name}:{variant} は知りません。選べる variant: {', '.join(v or '(なし)' for v in cls.VARIANTS)}"
        )


def engine_caps(spec: str) -> Caps:
    name, variant = parse_engine_spec(spec)
    return engine_class(name).caps(variant)


def create_engine(spec: str, options: dict[str, Any] | None = None, dry_run: bool = False) -> Engine:
    """エンジンを作る（モデルの読み込みは load() で行う）。dry_run なら中身はダミー。"""
    name, variant = parse_engine_spec(spec)
    cls = engine_class(name)
    if dry_run:
        from .fake import FakeEngine

        return FakeEngine(cls, variant, options)
    return cls(variant, options)
