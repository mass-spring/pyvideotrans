"""話者埋め込み (声の特徴を数百次元のベクトルにしたもの) の差し替え可能な実装。

- ecapa       : speechbrain/spkrec-ecapa-voxceleb (Apache-2.0)。初回だけモデルをダウンロードする。
- resemblyzer : Resemblyzer (Apache-2.0)。モデルは pip パッケージに同梱。

どちらも import は `load()` の中で行う (入っていなくても voicecheck 自体は動く)。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .audio import TARGET_SR

MIN_EMBED_SEC = 0.3     # これより短いと埋め込みを計算しない
CHUNK_SEC = 30.0        # 長い音声はこの長さで区切って平均する


class EmbedderUnavailable(RuntimeError):
    """埋め込みが使えない (未インストール・モデルが取得できない等)。理由は日本語で持つ。"""


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _chunks(y: np.ndarray, sr: int = TARGET_SR) -> list[np.ndarray]:
    """長い波形を CHUNK_SEC ごとに分ける。端の短すぎる切れ端は直前にくっつける。"""
    size = int(CHUNK_SEC * sr)
    if len(y) <= size * 1.5:
        return [y]
    parts = [y[i : i + size] for i in range(0, len(y), size)]
    if len(parts[-1]) < size // 2:
        parts[-2] = np.concatenate([parts[-2], parts.pop()])
    return parts


def trim_silence(y: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """前後の無音を落とす (全部無音なら元のまま)。"""
    import librosa

    t, _ = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=160)
    return t if len(t) >= 160 else y


class Embedder:
    """埋め込み器の共通部分。サブクラスは `_load` と `_embed_chunk` を作る。"""

    name = "base"

    def __init__(self) -> None:
        self._ready = False

    def load(self) -> None:
        """モデルを読み込む。使えなければ EmbedderUnavailable。"""
        if self._ready:
            return
        self._load()
        self._ready = True

    def embed(self, y16k: np.ndarray) -> np.ndarray:
        """16 kHz モノラル波形 → 長さ1に正規化した 1 次元ベクトル。"""
        self.load()
        if len(y16k) / TARGET_SR < MIN_EMBED_SEC:
            raise ValueError(f"短すぎて埋め込みを計算できません ({len(y16k) / TARGET_SR:.2f}秒 < {MIN_EMBED_SEC}秒)")
        y = trim_silence(y16k.astype(np.float32))
        vecs = [_unit(self._embed_chunk(c)) for c in _chunks(y)]
        return _unit(np.mean(vecs, axis=0))

    def _load(self) -> None:  # pragma: no cover - サブクラスで実装
        raise NotImplementedError

    def _embed_chunk(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class EcapaEmbedder(Embedder):
    name = "ecapa"
    SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            try:
                from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier  # 古い版
        except ImportError as e:
            raise EmbedderUnavailable(
                f"ecapa: speechbrain / torch が入っていません ({e})。`uv sync` をやり直してください"
            ) from e
        cache = Path(os.environ.get("VOICECHECK_CACHE", Path.home() / ".cache" / "voicecheck")) / "ecapa"
        kwargs = {}
        try:  # Windows はシンボリックリンクに管理者権限が要るので、コピー方式にする
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:  # noqa: BLE001
            pass
        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.SOURCE, savedir=str(cache), run_opts={"device": "cpu"}, **kwargs
            )
        except Exception as e:  # noqa: BLE001 - ネットワーク・ディスクなど
            raise EmbedderUnavailable(
                f"ecapa: モデル {self.SOURCE} を取得・読み込みできません ({type(e).__name__}: {str(e)[:200]})。"
                "初回はネット接続が必要です"
            ) from e

    def _embed_chunk(self, y: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            emb = self._model.encode_batch(torch.from_numpy(y).unsqueeze(0))
        return emb.squeeze().cpu().numpy()


class ResemblyzerEmbedder(Embedder):
    name = "resemblyzer"

    def _load(self) -> None:
        try:
            from resemblyzer import VoiceEncoder
        except ImportError as e:
            raise EmbedderUnavailable(
                f"resemblyzer: resemblyzer / torch が入っていません ({e})。`uv sync` をやり直してください"
            ) from e
        try:
            self._encoder = VoiceEncoder("cpu", verbose=False)
        except Exception as e:  # noqa: BLE001
            raise EmbedderUnavailable(f"resemblyzer: モデルを読み込めません ({type(e).__name__}: {str(e)[:200]})") from e

    def _embed_chunk(self, y: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(y, source_sr=TARGET_SR)  # 音量そろえ + 無音除去
        if len(wav) < TARGET_SR * 0.3:
            wav = y  # 無音除去で短くなりすぎたら元の波形を使う
        return self._encoder.embed_utterance(wav)


REGISTRY: dict[str, type[Embedder]] = {
    EcapaEmbedder.name: EcapaEmbedder,
    ResemblyzerEmbedder.name: ResemblyzerEmbedder,
}
CHOICES = [*REGISTRY, "all", "none"]


def resolve_names(spec: str) -> list[str]:
    """'ecapa' / 'resemblyzer' / 'all' / 'none' / 'ecapa,resemblyzer' → 名前のリスト。"""
    names: list[str] = []
    for part in [s.strip().lower() for s in spec.split(",") if s.strip()]:
        if part == "all":
            names += list(REGISTRY)
        elif part == "none":
            continue
        elif part in REGISTRY:
            names.append(part)
        else:
            raise ValueError(f"未知の埋め込み '{part}'。選べるもの: {', '.join(CHOICES)}")
    return list(dict.fromkeys(names))  # 重複を除く


def load_embedders(spec: str) -> tuple[list[Embedder], dict[str, str]]:
    """指定された埋め込み器を読み込む。(使えるもの, {名前: 使えなかった理由}) を返す。"""
    ok: list[Embedder] = []
    failed: dict[str, str] = {}
    for name in resolve_names(spec):
        emb = REGISTRY[name]()
        try:
            emb.load()
            ok.append(emb)
        except EmbedderUnavailable as e:
            failed[name] = str(e)
        except Exception as e:  # noqa: BLE001 - 想定外でも測定器全体は止めない
            failed[name] = f"{name}: 読み込み中にエラー ({type(e).__name__}: {str(e)[:200]})"
    return ok, failed
