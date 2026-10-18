"""calibrate: 登録音声から「同じ人どうし」「別の人どうし」の SIM 分布を測り、EER としきい値を出す。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis import Analyzer
from .audio import list_audio_files
from .embedders import Embedder
from .similarity import (
    EerResult,
    compute_eer,
    histogram_table,
    pair_scores,
    rate_at_or_above,
    threshold_at_far,
)

NAN = float("nan")
STRICT_FAR = 0.01  # 「厳しめ」のしきい値: 別人の通過率 1%


class CalibError(ValueError):
    """入力フォルダの形が違うときなど (メッセージは利用者向けの日本語)。"""


@dataclass
class SpeakerInfo:
    name: str
    n_files: int
    total_sec: float
    note: str = ""


@dataclass
class EmbedderCalib:
    name: str
    n_speakers: int = 0
    n_files: int = 0
    n_same: int = 0
    n_diff: int = 0
    same_mean: float = NAN
    same_std: float = NAN
    diff_mean: float = NAN
    diff_std: float = NAN
    eer: EerResult | None = None
    strict_threshold: float = NAN
    strict_pass_rate: float = NAN       # 厳しめしきい値のとき、本人どうしが通る割合
    note: str = ""


@dataclass
class CalibResult:
    per_embedder: list[EmbedderCalib] = field(default_factory=list)
    bins: list[tuple[float, float]] = field(default_factory=list)
    counts: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # {埋め込み名: {"same": .., "diff": ..}}
    speakers: list[SpeakerInfo] = field(default_factory=list)


def _std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) >= 2 else NAN


def find_speaker_dirs(enroll_dir: str | Path) -> list[Path]:
    """登録フォルダの中の話者フォルダ (サブフォルダ) を返す。形が違えば CalibError。"""
    root = Path(enroll_dir)
    if not root.is_dir():
        raise CalibError(f"フォルダが見つかりません: {root}")
    spk_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    if len(spk_dirs) < 2:
        raise CalibError(
            f"話者ごとのサブフォルダが2つ以上必要です: {root}\n"
            "  例:  enroll/Aさん/*.wav  enroll/Bさん/*.wav  (1人1フォルダ)"
        )
    return spk_dirs


def run_calibrate(enroll_dir: str | Path, embedders: list[Embedder], verbose: bool = True) -> CalibResult:
    spk_dirs = find_speaker_dirs(enroll_dir)
    files = {d.name: list_audio_files(d) for d in spk_dirs}
    if not embedders:
        raise CalibError("使える埋め込みがありません。--embedder と uv sync を確認してください")

    analyzer = Analyzer(embedders, with_features=False, verbose=verbose)
    analyzer.prefetch([str(p) for fs in files.values() for p in fs])

    result = CalibResult()
    vecs: dict[str, list[tuple[str, np.ndarray]]] = {e.name: [] for e in embedders}
    for spk, fs in files.items():
        total, n_ok, bad = 0.0, 0, []
        for p in fs:
            clip = analyzer.get(str(p))
            if not clip.ok:
                bad.append(f"{p.name}: {clip.error}")
                continue
            n_ok += 1
            total += clip.duration_s
            for name, v in clip.embeddings.items():
                vecs[name].append((spk, v))
        note = "; ".join(bad)
        if n_ok == 0:
            note = (note + " " if note else "") + "読める音声がありません"
        elif n_ok == 1:
            note = (note + " " if note else "") + "音声が1つだけです (同じ人どうしのペアが作れません)"
        result.speakers.append(SpeakerInfo(spk, n_ok, total, note))

    groups: dict[str, np.ndarray] = {}
    for emb in embedders:
        c = EmbedderCalib(name=emb.name)
        items = vecs[emb.name]
        c.n_files = len(items)
        c.n_speakers = len({s for s, _ in items})
        if c.n_speakers >= 2 and len(items) >= 3:
            same, diff = pair_scores(np.vstack([v for _, v in items]), [s for s, _ in items])
            c.n_same, c.n_diff = len(same), len(diff)
            groups[f"{emb.name}:same"], groups[f"{emb.name}:diff"] = same, diff
            if len(diff):
                c.diff_mean, c.diff_std = float(diff.mean()), _std(diff)
                c.strict_threshold = threshold_at_far(diff, STRICT_FAR)
            if len(same) and len(diff):
                c.eer = compute_eer(same, diff)
                # 「厳しめ」はおすすめ (EER) より低くしない。少人数で分布が完全に分かれると、別人の上位1%点が EER 点より下に来るため
                c.strict_threshold = max(c.strict_threshold, c.eer.threshold)
            if len(same):
                c.same_mean, c.same_std = float(same.mean()), _std(same)
                c.strict_pass_rate = rate_at_or_above(same, c.strict_threshold)
            if not (len(same) and len(diff)):
                c.note = "同じ人どうしのペアが作れず、EER を出せません (1人2つ以上の音声が必要)"
        else:
            c.note = "有効な音声が足りません (2人以上、合計3つ以上の音声が必要)"
        result.per_embedder.append(c)

    if groups:
        result.bins, hist = histogram_table(groups)
        for emb in embedders:
            if f"{emb.name}:same" in hist:
                result.counts[emb.name] = {"same": hist[f"{emb.name}:same"], "diff": hist[f"{emb.name}:diff"]}
    return result
