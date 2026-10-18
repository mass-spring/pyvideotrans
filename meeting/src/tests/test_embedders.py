"""本物の埋め込み (ECAPA / Resemblyzer)。モデルを読み込めない環境 (オフラインなど) では自動でスキップする。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synth import SR, voice_like
from voicecheck.audio import load_audio, resample
from voicecheck.embedders import REGISTRY, EmbedderUnavailable, load_embedders, resolve_names
from voicecheck.similarity import cosine

DATA = Path(__file__).resolve().parents[2] / "data" / "testdata" / "sv_testclean_wavs"


def get_embedder(name):
    ok, failed = load_embedders(name)
    if not ok:
        pytest.skip(f"{name} を使えないためスキップ: {failed.get(name, '')[:150]}")
    return ok[0]


def test_resolve_names():
    assert resolve_names("all") == ["ecapa", "resemblyzer"]
    assert resolve_names("none") == []
    assert resolve_names("ecapa,ecapa") == ["ecapa"]
    with pytest.raises(ValueError):
        resolve_names("unknown")


def test_unavailable_embedder_is_reported_not_raised(monkeypatch):
    """モデルを取得できないとき、load_embedders は理由を返し、例外にしない。"""

    class Broken(REGISTRY["ecapa"]):
        def _load(self):
            raise EmbedderUnavailable("ecapa: テスト用に失敗")

    monkeypatch.setitem(REGISTRY, "ecapa", Broken)
    ok, failed = load_embedders("ecapa")
    assert ok == [] and "テスト用" in failed["ecapa"]


@pytest.mark.parametrize("name", ["ecapa", "resemblyzer"])
def test_real_embedder_shape_norm_and_determinism(name):
    emb = get_embedder(name)
    y = voice_like(1, 1, dur=3.0)
    v1, v2 = emb.embed(y), emb.embed(y)
    assert v1.ndim == 1 and v1.shape[0] in (192, 256)
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)
    assert cosine(v1, v2) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError):
        emb.embed(y[: SR // 10])  # 短すぎる


@pytest.mark.skipif(not DATA.exists(), reason="公開テスト音声が無い (src/scripts/prepare_testdata.py で作る)")
@pytest.mark.parametrize("name", ["ecapa", "resemblyzer"])
def test_real_embedder_same_speaker_scores_higher(name):
    emb = get_embedder(name)
    spk = sorted(DATA.iterdir())[:3]

    def vec(p):
        y, sr = load_audio(p)
        return emb.embed(resample(y, sr))

    files = {s.name: sorted(s.glob("*.wav"))[:3] for s in spk}
    v = {k: [vec(p) for p in ps] for k, ps in files.items()}
    names = list(v)
    same = [cosine(v[n][0], v[n][1]) for n in names]
    diff = [cosine(v[a][0], v[b][0]) for i, a in enumerate(names) for b in names[i + 1:]]
    assert min(same) > max(diff)
