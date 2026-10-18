"""1つの音声ファイルを解析する (特徴量 + 埋め込み)。同じファイルは1回だけ解析して使い回す。"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import TARGET_SR, load_audio, resample
from .embedders import Embedder
from .features import SHORT_SEC, extract_features


@dataclass
class ClipResult:
    path: str
    ok: bool = True                       # 読み込めたか
    error: str = ""                       # ok=False のときの理由
    duration_s: float = float("nan")
    sample_rate: int = 0
    notes: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # {埋め込み名: ベクトル}


class Analyzer:
    """ファイルごとの解析結果をキャッシュする。"""

    def __init__(
        self,
        embedders: list[Embedder] | None = None,
        with_features: bool = True,
        f0_method: str = "harvest",
        verbose: bool = True,
    ) -> None:
        self.embedders = embedders or []
        self.with_features = with_features
        self.f0_method = f0_method
        self.verbose = verbose
        self._cache: dict[str, ClipResult] = {}

    def prefetch(self, paths: list[str]) -> None:
        """重複を除いた順に解析して進み具合を表示する。"""
        uniq = list(dict.fromkeys(p for p in paths if p))
        todo = [p for p in uniq if os.path.abspath(p) not in self._cache]
        for i, p in enumerate(todo, start=1):
            if self.verbose:
                print(f"  [{i}/{len(todo)}] 解析中: {os.path.basename(p)}", file=sys.stderr, flush=True)
            self.get(p)

    def clips(self) -> list[ClipResult]:
        return list(self._cache.values())

    def get(self, path: str) -> ClipResult:
        key = os.path.abspath(path)
        if key not in self._cache:
            self._cache[key] = self._analyze(path)
        return self._cache[key]

    def _analyze(self, path: str) -> ClipResult:
        res = ClipResult(path=path)
        try:
            y, sr = load_audio(path)
        except Exception as e:  # noqa: BLE001 - 1ファイルの失敗で全体を止めない
            res.ok, res.error = False, str(e)
            return res

        res.sample_rate = sr
        res.duration_s = len(y) / float(sr)
        if res.duration_s < SHORT_SEC:
            res.notes.append(f"短い音声です ({res.duration_s:.2f}秒 < {SHORT_SEC:.0f}秒): 値が不安定です")
        if sr < TARGET_SR:
            res.notes.append(f"サンプルレートが低い ({sr} Hz): 高い音域の特徴は信頼できません")
        if float(np.mean(np.abs(y) >= 0.999)) > 0.001:
            res.notes.append("音が割れている (クリッピング) 可能性があります")

        y16 = resample(y, sr, TARGET_SR)
        if self.with_features:
            try:
                res.features, notes = extract_features(y16, TARGET_SR, self.f0_method)
                res.features["duration_s"] = res.duration_s
                res.notes += notes
            except Exception as e:  # noqa: BLE001
                res.notes.append(f"特徴量の計算に失敗: {type(e).__name__}: {str(e)[:120]}")
        for emb in self.embedders:
            try:
                res.embeddings[emb.name] = emb.embed(y16)
            except Exception as e:  # noqa: BLE001
                res.notes.append(f"{emb.name}: {str(e)[:150]}")
        return res
