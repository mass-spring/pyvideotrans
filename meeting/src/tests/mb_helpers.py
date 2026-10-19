"""meetingbench のテスト用: 合成音声で作った小さな JVNV 風フォルダと、本体の _assign_speakers の読み込み。"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from synth import harmonic_tone

REPO = Path(__file__).resolve().parents[3]
SPEAKERS = {"F1": 220.0, "F2": 260.0, "M1": 120.0, "M2": 150.0}
EMOTIONS = ("anger", "happy", "sad")


def make_fake_jvnv(root: Path, per_emotion: int = 6, dur: float = 3.0, pad: float = 0.4) -> Path:
    """4人 × 3感情 × per_emotion 本の 48kHz モノラル wav（前後に pad 秒の無音つき）と transcription_no_nv.csv。"""
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for emo in EMOTIONS:
        for i in range(1, per_emotion + 1):
            lines.append(f"{emo}_regular_{i:02d}|{emo}の文{i}です。")
    (root / "transcription_no_nv.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for k, (spk, f0) in enumerate(SPEAKERS.items()):
        d = root / spk
        d.mkdir(exist_ok=True)
        for emo in EMOTIONS:
            for i in range(1, per_emotion + 1):
                tone = harmonic_tone(f0, dur=dur + 0.1 * i, sr=48000, amp=0.3)
                sil = np.zeros(int(pad * 48000), dtype=np.float32)
                sf.write(d / f"{spk}_{emo}_regular_{i:02d}.wav", np.concatenate([sil, tone, sil]), 48000, subtype="PCM_32")
    return root


def load_pvt_speakers_module(monkeypatch):
    """本体の videotrans/process/_audio_speakers.py を、設定読み込み（PySide6 など）を偽物に差し替えて読み込む。無ければ None。"""
    src = REPO / "videotrans" / "process" / "_audio_speakers.py"
    if not src.is_file():
        return None
    cfg = types.ModuleType("videotrans.configure.config")
    cfg.ROOT_DIR = str(REPO)
    cfg.logger = logging.getLogger("mb-test")
    cfg.app_cfg = types.SimpleNamespace(proxy="")
    for name, mod in (("videotrans", types.ModuleType("videotrans")), ("videotrans.configure", types.ModuleType("videotrans.configure")),
                      ("videotrans.configure.config", cfg)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("pvt_audio_speakers_under_test", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
