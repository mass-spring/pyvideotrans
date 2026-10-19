"""参考値: 正解の発言ごとに ECAPA 話者埋め込みを取り、人数を 4 と教えて凝集型クラスタリングしたときの発言単位の正解率。

pyVideoTrans の話者分離の数字が低いとき、『データが難しいのか、その話者分離が弱いのか』を切り分ける目安にする
（ECAPA-TDNN は voicecheck で較正済みの話者埋め込み。クラスタリングは平均連結＋コサイン距離の素朴なもの）。
正解の区間を使うので、区間の切り方の影響はない。pyVideoTrans の本体は使わない。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def cluster_accuracy(embs: np.ndarray, labels: list[str], k: int) -> tuple[float, np.ndarray, list[str]]:
    """embs (N, D) をコサイン距離の平均連結で k 個に分け、名前の対応を最適にしたときの正解率。戻り値: (正解率, 混同行列[正解話者, クラスタ], 話者名)。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment

    spk = sorted(set(labels))
    truth = np.array([spk.index(x) for x in labels])
    cl = fcluster(linkage(embs, "average", metric="cosine"), k, "maxclust") - 1
    m = np.zeros((len(spk), int(cl.max()) + 1), dtype=int)
    for t, c in zip(truth, cl):
        m[t, c] += 1
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(truth)), m, spk


def ecapa_check(wav: Path, ref_csv: Path, k: int = 4, embedder=None) -> dict:
    """embedder: .embed(16kHz波形) → ベクトル を持つもの（既定は voicecheck の ECAPA。モデルは初回だけ取得）。"""
    import soundfile as sf

    if embedder is None:
        from voicecheck.embedders import EcapaEmbedder

        embedder = EcapaEmbedder()
        embedder.load()
    y, sr = sf.read(str(wav), dtype="float32")
    rows = list(csv.DictReader(open(ref_csv, encoding="utf-8-sig", newline="")))
    X, labels = [], []
    for r in rows:
        seg = y[int(float(r["start"]) * sr):int(float(r["end"]) * sr)]
        X.append(embedder.embed(seg))
        labels.append(r["speaker"])
    acc, m, spk = cluster_accuracy(np.array(X), labels, k)
    return {"method": "ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) + 平均連結・コサイン距離、人数=%d" % k, "utterances": len(labels),
            "accuracy": acc, "speakers": spk, "confusion": m.tolist()}
