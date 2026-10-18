"""--dry-run 用のダミーエンジン。モデルを一切使わず、手本の音声を加工して「合成音声のふり」をする。

目的は、実験の流れ全体（条件の展開 → 生成 → pairs.csv → voicecheck → Excel）を、GPU もモデルも無い PC で
通しでテストすること。**ここで出る SIM の値に意味はない**（実際の TTS の結果ではない）。

動き:
- 手本を繰り返し並べて、文の長さに比例した長さ（約 0.07 秒/文字、1〜12 秒）にする
- エンジン名・言語・感情に応じて、声の高さを少しずらす（条件ごとに値が変わるのを確かめるため）
- seed が違えば、ごくわずかに違う音にする（繰り返しの確認用）
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from ..audio import pitch_shift, read_wav
from .base import Engine, SynthResult, emotion_text


class FakeEngine(Engine):
    needs_gpu = False

    def __init__(self, real_cls: type[Engine], variant: str = "", options: dict[str, Any] | None = None) -> None:
        # 本物のエンジンと同じ性質（対応言語・感情の条件）を持たせる。モデルは読み込まない。
        self._real = real_cls
        caps = real_cls.caps(variant)
        self.name = real_cls.name
        self.VARIANTS = (variant,)
        self.LANGS = caps.langs
        self.emotion_modes = caps.emotion_modes
        self.emotion_only = caps.emotion_only
        super().__init__(variant, options)

    def load(self) -> None:
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
        y, sr = read_wav(ref_wav)
        want = float(min(12.0, max(1.0, 0.07 * len(text) + 0.5)))
        n = int(want * sr)
        reps = int(np.ceil(n / max(len(y), 1)))
        y = np.tile(y, reps)[:n]
        h = zlib.crc32(f"{self.spec}|{lang}".encode()) % 5
        semitones = (h - 2) * 0.5
        instr = emotion_text(emotion)
        if instr:
            semitones += 1.0
        if seed is not None:
            semitones += ((seed % 7) - 3) * 0.02
        y = pitch_shift(y, sr, semitones)
        return SynthResult(y.astype(np.float32), sr, "dry_run", "ダミー音声（--dry-run）")

    def info(self) -> dict[str, str]:
        return {"engine": self.spec, "dry_run": "yes"}
