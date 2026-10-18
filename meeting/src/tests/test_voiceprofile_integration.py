"""公開音声 (LibriSpeech test-clean、data/testdata/sv_testclean_wavs) で voiceprofile を通しで動かす。

- 1 人の話者の発話を、間 (雑音の床) をはさんで 1 本の長い wav につなぎ、「録音」に見立てる。
- 話者埋め込みは本物の ECAPA (voicecheck) を使う。モデルがない・データがない時は skip。
- 英語の朗読なので台本は使わない (書き起こしは空欄)。サンプリング周波数は 16 kHz (警告が出るのが正しい)。

`uv run pytest src/tests/test_voiceprofile_integration.py -s` で、選ばれた手本と点数が表示される。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vp_helpers import load_fork_voice_profile
from voicecheck.audio import TARGET_SR, load_audio, resample
from voicecheck.embedders import load_embedders
from voiceprofile.pipeline import BuildOptions, build_profile
from voiceprofile.profile_io import write_profile
from voiceprofile.report import write_report

DATA = Path(__file__).resolve().parents[2] / "data" / "testdata" / "sv_testclean_wavs"
SPEAKER, OTHER = "1320", "1089"

pytestmark = pytest.mark.skipif(not (DATA / SPEAKER).is_dir() or not (DATA / OTHER).is_dir(), reason="公開テストデータがありません")


def make_recording(speaker_dir: Path, out: Path, seed: int = 0) -> tuple[Path, list[tuple[float, float]]]:
    """発話をつなぐ。間 (0.9〜1.5 秒) は、その話者の録音の中でいちばん静かな 0.1 秒を並べたもの (雑音の質をそろえる)。
    戻り値: (wav, 各発話の (開始秒, 終了秒))"""
    rng = np.random.default_rng(seed)
    utts = [sf.read(str(p), dtype="float32")[0] for p in sorted(speaker_dir.glob("*.wav"))]
    sr = 16000
    n = int(0.1 * sr)

    def quietest(y):
        e = np.convolve(y.astype(np.float64) ** 2, np.ones(n) / n, mode="valid")
        i = int(np.argmin(e))
        return y[i:i + n]

    tiles = np.concatenate([quietest(u) for u in utts])

    def pause(sec):
        out_ = np.zeros(int(sec * sr), dtype=np.float32)
        pos = 0
        while pos < len(out_):
            k = int(rng.integers(0, len(tiles) - n))
            seg = (tiles[k:k + n] * np.hanning(n) ** 0.5).astype(np.float32)
            m = min(n, len(out_) - pos)
            out_[pos:pos + m] += seg[:m]
            pos += n // 2
        return out_

    parts, spans, pos = [pause(1.2)], [], 1.2
    for u in utts:
        parts.append(u)
        spans.append((pos, pos + len(u) / sr))
        p = pause(float(rng.uniform(0.9, 1.5)))
        parts.append(p)
        pos += len(u) / sr + len(p) / sr
    wav = out / f"rec_{speaker_dir.name}.wav"
    sf.write(wav, np.concatenate(parts), sr, subtype="PCM_16")
    return wav, spans


@pytest.fixture(scope="module")
def ecapa():
    embedders, failed = load_embedders("ecapa")
    if not embedders:
        pytest.skip(f"ECAPA を読み込めません: {failed}")
    return embedders


@pytest.fixture(scope="module")
def built(tmp_path_factory, ecapa):
    tmp = tmp_path_factory.mktemp("libri")
    wav, spans = make_recording(DATA / SPEAKER, tmp)
    opts = BuildOptions(name="LibriSpeech1320", inputs=[wav], out_dir=tmp / "profiles", langs=["en"], skip_first=0, stats=False)
    res = build_profile(opts, embedders=ecapa, log=lambda m: None)
    written = write_profile(res)
    write_report(res, written["dir"] / "enroll_report.xlsx", "integration test", written)
    print("\n== 公開音声での通し実行: 話者", SPEAKER, f"録音 {sf.info(wav).duration:.0f} 秒 / 発話区間 {len(res.atoms)} 個 / 候補 {len(res.candidates)} 個")
    for c in res.refs:
        print(f"   {c.ref_name} {c.dur:5.1f}秒  SNR {c.snr_db:5.1f} dB  割れ {c.clip_ratio:.3%}  "
              f"一貫性(ecapa) {c.consistency_true.get('ecapa', float('nan')):.3f} (点数用 {c.consistency.get('ecapa', float('nan')):.3f})  点 {c.score:5.1f}")
    return dict(tmp=tmp, wav=wav, spans=spans, res=res, written=written, pdir=written["dir"])


def test_refs_from_public_speech_meet_the_format_and_bounds(built):
    res = built["res"]
    assert 3 <= len(res.refs) <= 4
    for c in res.refs:
        assert 8.0 <= c.dur <= 20.0 and c.lang == "en"
        assert c.snr_db > 10 and c.clip_ratio == 0.0
        assert c.consistency_true["ecapa"] > 0.7                  # 同じ話者の他の部分の平均と近い
    assert any("44100 Hz 未満" in w for w in res.warnings)       # LibriSpeech は 16 kHz なので警告が出るのが正しい
    vp = load_fork_voice_profile()
    refs = vp.load_profile(built["pdir"])
    assert len(refs) == len(res.refs) and all(r.lang == "en" for r in refs)
    assert vp.select_ref(refs, "en") is not None
    for r in refs:
        assert sf.info(r.wav).samplerate == 16000                  # 録音のサンプリング周波数のまま


def test_cut_points_are_in_pauses_in_real_speech(built):
    """手本の切り口は、どれも話していない所 (発話区間と発話区間のあいだの無音) にある。"""
    res = built["res"]
    f = res.files[0]
    db = f.seg.frame_db
    for c in res.refs:
        for a, b in c.pieces:
            for t in (a, b):
                k = min(int(round(t / 0.01)), len(db) - 1)
                assert db[max(k - 2, 0):k + 3].max() < f.seg.off_db, f"{c.ref_name}: {t:.2f} 秒の切り口が発話中 ({db[k]:.0f} dB)"
            for atom in (x.seg for x in res.atoms):
                assert not (atom.start < a < atom.end or atom.start < b < atom.end)
        # 手本の波形の最初と最後の 20 ms は、発話のしきい値 (off_db) より小さい = 発話の途中ではない
        w, sr = sf.read(built["pdir"] / "refs" / c.ref_name, dtype="float32")
        edge = int(0.02 * sr)
        for part in (w[:edge], w[-edge:]):
            level = 20 * np.log10(np.sqrt(np.mean(part.astype(np.float64) ** 2)) + 1e-12)
            assert level < f.seg.off_db


def test_refs_sound_like_the_same_speaker_as_the_held_out_clips(built, ecapa):
    """手本と評価用 (手本に使っていない同じ話者の発話) は似ていて、別の話者の発話とは似ていない。"""
    ecapa = ecapa[0]
    pdir = built["pdir"]
    emb = lambda p: ecapa.embed(resample(load_audio(p)[0], load_audio(p)[1], TARGET_SR))   # noqa: E731
    refs = [emb(p) for p in sorted((pdir / "refs").glob("*.wav"))]
    evals = [emb(pdir / e["wav"]) for e in json.loads((pdir / "eval" / "eval.json").read_text(encoding="utf-8"))["clips"]]
    assert evals
    others = [emb(p) for p in sorted((DATA / OTHER).glob("*.wav"))[:3]]
    same = np.mean([float(r @ e) for r in refs for e in evals])
    diff = np.mean([float(r @ o) for r in refs for o in others])
    print(f"   ecapa 類似度: 手本 vs 評価用(同じ話者) 平均 {same:.3f} / 手本 vs 別の話者 平均 {diff:.3f}")
    assert same > 0.6 and diff < 0.4 and same - diff > 0.3
    # 保存したセンタロイドは、手本どうしの平均に近い
    cen = np.load(pdir / "emb.npy")
    assert float(cen @ np.mean(refs, axis=0)) > 0.9


def test_evaluation_clips_are_disjoint_from_refs_in_time(built):
    res = built["res"]
    ref_ranges = [(a, b) for c in res.refs for a, b in c.pieces]
    assert res.evals
    for e in res.evals:
        for a, b in e.pieces:
            for ra, rb in ref_ranges:
                assert b <= ra + 1e-6 or a >= rb - 1e-6, (e.name, (a, b), (ra, rb))
