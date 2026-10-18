"""voicelab のテスト用の部品（テスト本体は無い）。他の test_voicelab_*.py が import して使う。

合成音声（synth.voice_like）で、voiceprofile 形式のプロフィールフォルダや wav フォルダを作る。
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from synth import SR, voice_like
from voicecheck.audio import resample

SHORT_TARGETS = (
    "id,lang,kind,emotion,text\n"
    "m1-ja,ja,meeting,,はい。\n"
    "m1-en,en,meeting,,Yes.\n"
    "e1-ja,ja,emotional,\"angry, tense, slightly faster\",だめです。\n"
    "e1-en,en,emotional,\"angry, tense, slightly faster\",No way.\n"
)


def write_clip(path: Path, speaker: int, seed: int, dur: float, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = voice_like(speaker, seed, dur=dur)
    if sr != SR:
        y = resample(y, SR, sr)
    sf.write(path, y, sr)
    return path


def make_profile(root: Path, name: str, speaker: int, *, ref_durs=(12.0, 13.0, 11.0), eval_durs=(3.0, 4.0, 3.5, 2.5, 3.0),
                 ref_sr: int = 24000, voiceprofile_format: bool = True) -> Path:
    """voiceprofile ツールと同じ形（profile.json の eval.index → eval/eval.json）のプロフィールを作る。"""
    d = root / name
    refs, evals = [], []
    for k, dur in enumerate(ref_durs):
        write_clip(d / "refs" / f"ref_ja_{k + 1:02d}.wav", speaker, 10 + k, dur, ref_sr)
        refs.append({"wav": f"refs/ref_ja_{k + 1:02d}.wav", "text": f"登録の書き起こし{k + 1}", "lang": "ja",
                     "sec": dur, "snr_db": 30.0 - k, "score": 90.0 - 5 * k})
    for k, dur in enumerate(eval_durs):
        write_clip(d / "eval" / f"eval_ja_{k + 1:02d}.wav", speaker, 100 + k, dur)
        evals.append({"wav": f"eval/eval_ja_{k + 1:02d}.wav", "text": f"評価{k + 1}" if k % 2 == 0 else "", "lang": "ja", "sec": dur})
    prof: dict = {"name": name, "refs": refs}
    if voiceprofile_format:
        (d / "eval").mkdir(exist_ok=True)
        (d / "eval" / "eval.json").write_text(json.dumps({"name": name, "clips": evals}, ensure_ascii=False), encoding="utf-8")
        prof["eval"] = {"index": "eval/eval.json", "n": len(evals)}
    else:
        prof["eval"] = evals
    (d / "profile.json").write_text(json.dumps(prof, ensure_ascii=False), encoding="utf-8")
    return d


def make_cfg_dict(speakers: list[dict], targets: str = "targets.csv", **over) -> dict:
    cfg = {
        "name": "t", "speakers": speakers, "targets": targets, "target_langs": ["ja", "en"],
        "engines": ["voxcpm2", "qwen3tts"], "repeats": 1, "embedder": "none",
    }
    cfg.update(over)
    return cfg


def write_cfg(tmp: Path, cfg: dict, targets_csv: str = SHORT_TARGETS, fname: str = "exp.json") -> Path:
    (tmp / "targets.csv").write_text(targets_csv, encoding="utf-8-sig")
    p = tmp / fname
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return p


def test_helpers_make_a_profile(tmp_path):
    d = make_profile(tmp_path, "A", 0)
    assert (d / "profile.json").is_file() and (d / "eval" / "eval.json").is_file()
