"""TTS エンジンの共通の形（アダプタの土台）。

どのエンジンも次の 1 つの関数だけを持つ:

    synthesize(text, lang, ref_wav, ref_text, emotion=None, *, ref_lang=None, seed=None) -> SynthResult

- text      : 読み上げる文
- lang      : 読み上げる言語（ja / en / fr / zh などの 2 文字コード）
- ref_wav   : 声の手本の wav ファイルのパス
- ref_text  : 手本の書き起こし（無ければ None。エンジンごとに「無い時の動き」が決まっている）
- emotion   : 感情の指示。文字列（例 "angry, tense, slightly faster"）か、
              {"label": "angry", "intensity": 0.6, "instruction": "..."} の辞書。使わないなら None
- ref_lang  : 手本の言語（言語をまたぐかどうかの判断に使うエンジンがある）
- seed      : 乱数の種（同じ種なら、同じ条件どうしの比較が「対」になる）

重いライブラリ（torch・各 TTS）は load() の中で import する。
このファイルは numpy だけで動く（import しても何も重くならない）。
"""
from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

# 感情の指示の使われ方（Engine.emotion_modes に入れる）
MODE_INSTRUCT = "instruct"      # 手本＋感情の指示
MODE_REFONLY = "refonly"        # 手本だけ（書き起こし無し・指示無し）：対照
MODE_SIMILARITY = "similarity"  # 手本＋書き起こし（指示無し）：似かた優先


class EngineError(RuntimeError):
    """エンジンが使えない・合成に失敗した（メッセージは利用者向けの日本語）。"""


@dataclass
class SynthResult:
    wav: np.ndarray          # float32・モノラル・1 次元
    sr: int                  # サンプルレート
    mode: str = ""           # 実際に使った方式（例 "ultimate_clone"）。条件の取り違えを後から確認できるように残す
    note: str = ""           # 注意（書き起こしが無くて方式を切り替えた、など）


def emotion_text(emotion: Any) -> str:
    """emotion（文字列 or 辞書 or None）から、指示の文字列を取り出す。"""
    if emotion is None:
        return ""
    if isinstance(emotion, str):
        return emotion.strip()
    if isinstance(emotion, dict):
        return str(emotion.get("instruction") or emotion.get("label") or "").strip()
    return str(emotion).strip()


def emotion_intensity(emotion: Any, default: float = 0.7) -> float:
    """感情の強さ（0〜1）。辞書に intensity があればそれ、無ければ default。"""
    if isinstance(emotion, dict) and emotion.get("intensity") is not None:
        try:
            return float(min(1.0, max(0.0, float(emotion["intensity"]))))
        except (TypeError, ValueError):
            pass
    return default


def seed_everything(seed: int | None) -> None:
    """可能な範囲で乱数の種をそろえる（torch が入っていれば torch も）。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def free_gpu() -> None:
    """GPU メモリを手放す（次のエンジンを読み込む前に呼ぶ）。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def pkg_version(dist_name: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(dist_name)
    except Exception:  # noqa: BLE001
        return ""


def to_mono_float32(x: Any) -> np.ndarray:
    """numpy / torch テンソル / リストのどれでも、1 次元 float32 の numpy にする。"""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 2:  # (チャンネル, サンプル) または (サンプル, チャンネル)
        a = a.mean(axis=0) if (a.shape[0] <= 8 and a.shape[0] < a.shape[1]) else a.mean(axis=1)
    return np.nan_to_num(a.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class Caps:
    """エンジンの性質（モデルを読み込まなくても分かる部分）。"""

    name: str
    variant: str
    emotion_modes: tuple[str, ...]
    emotion_only: bool
    langs: frozenset[str] | None
    needs_gpu: bool = True


class Engine:
    """TTS エンジンの土台。サブクラスは load / synthesize を実装する。"""

    name = ""                         # 例 "voxcpm2"
    VARIANTS: tuple[str, ...] = ("",)  # 選べる variant（"" は variant なし）
    LANGS: frozenset[str] | None = None   # 対応言語（None なら制限なし）
    emotion_modes: tuple[str, ...] = ()   # このエンジンで比べられる感情条件（MODE_*）
    emotion_only = False              # True なら感情の条件だけに使う（CosyVoice3 の instruct2 など）
    needs_gpu = True

    def __init__(self, variant: str = "", options: dict[str, Any] | None = None) -> None:
        if variant not in self.VARIANTS:
            raise EngineError(
                f"{self.name}:{variant} は知りません。選べる variant: "
                f"{', '.join(v or '(なし)' for v in self.VARIANTS)}"
            )
        self.variant = variant
        self.options = dict(options or {})
        self._loaded = False

    @property
    def spec(self) -> str:
        return f"{self.name}:{self.variant}" if self.variant else self.name

    @classmethod
    def caps(cls, variant: str = "") -> "Caps":
        """モデルを読み込まずに分かるエンジンの性質（計画・ドライラン用）。"""
        return Caps(cls.name, variant, cls.emotion_modes, cls.emotion_only, cls.LANGS, cls.needs_gpu)

    def supports_lang(self, lang: str) -> bool:
        return self.LANGS is None or lang in self.LANGS

    def opt(self, key: str, default: Any) -> Any:
        return self.options.get(key, default)

    # --- サブクラスが実装する
    def load(self) -> None:  # pragma: no cover - 実モデルが要る
        raise NotImplementedError

    def synthesize(  # pragma: no cover - 実モデルが要る
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
        raise NotImplementedError

    def close(self) -> None:
        """モデルを手放す。"""
        for k in [k for k in vars(self) if k.startswith("_m_")]:
            setattr(self, k, None)
        self._loaded = False
        free_gpu()

    def info(self) -> dict[str, str]:
        """版などの記録（結果の Excel に残す）。"""
        return {"engine": self.spec}
