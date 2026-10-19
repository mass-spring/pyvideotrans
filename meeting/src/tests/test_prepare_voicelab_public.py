"""prepare_voicelab_public（公開データ JVNV から voicelab の話者フォルダを作る）のテスト。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_voicelab_public as pvp  # noqa: E402


def _fake_jvnv(root: Path, durs: dict[str, float]) -> None:
    (root / "F1").mkdir(parents=True)
    lines = []
    for key, sec in durs.items():
        sf.write(root / "F1" / f"F1_{key}.wav", np.zeros(int(sec * 8000), np.float32), 8000)
        lines.append(f"{key}|{key} の文。")
    (root / "transcription_no_nv.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_plan_jvnv_picks_calm_refs_near_12s_and_balanced_eval(tmp_path):
    durs = {
        "happy_regular_01": 12.1, "happy_regular_02": 7.0, "sad_regular_01": 14.9, "sad_regular_02": 11.0,
        "anger_regular_01": 12.0,                      # 怒りは 12 秒ちょうどでも手本にしない
        "anger_regular_02": 5.0, "anger_regular_03": 3.0,  # 4 秒未満は評価にも使わない
        "fear_regular_01": 6.0, "happy_free_01": 12.0,     # free は対象外
    }
    _fake_jvnv(tmp_path, durs)
    rows = pvp.plan_jvnv("F1", root=tmp_path, n_ref=2, eval_per_emotion=1)
    refs = [r["file"] for r in rows if r["role"] == "ref"]
    evals = [r["file"] for r in rows if r["role"] == "eval"]
    assert refs == ["F1_happy_regular_01.wav", "F1_sad_regular_02.wav"]
    assert "F1_anger_regular_01.wav" in evals and "F1_anger_regular_03.wav" not in evals
    assert not set(refs) & set(evals)
    assert all(r["lang"] == "ja" and r["text"].endswith("の文。") for r in rows)
    assert not any("free" in f for f in refs + evals)


def test_write_folder_makes_voicelab_transcripts(tmp_path):
    _fake_jvnv(tmp_path / "jv", {"happy_regular_01": 11.0, "sad_regular_01": 6.0})
    rows = pvp.plan_jvnv("F1", root=tmp_path / "jv", n_ref=1, eval_per_emotion=2)
    out = tmp_path / "out" / "jvnv_F1"
    pvp.write_folder(out, rows, tmp_path / "jv" / "F1")
    with open(out / "transcripts.csv", encoding="utf-8-sig") as f:
        got = list(csv.DictReader(f))
    assert [r["role"] for r in got] == ["ref", "eval"]
    assert all((out / r["file"]).is_file() for r in got)
