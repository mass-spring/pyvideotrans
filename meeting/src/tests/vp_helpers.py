"""voiceprofile のテスト用の共通部品: fork 本体の videotrans/util/voice_profile.py を読み込む。

voice_profile.py は `from videotrans.configure.config import logger` だけを本体から使う。
meeting の仮想環境には GUI ライブラリ (PySide6) が入っていないので、videotrans パッケージ全体は import できない。
そこで fork の根を sys.path に足したうえで、
  - PySide6 があれば普通に `videotrans.util.voice_profile` を import する
  - なければ、logger だけの偽の `videotrans.configure.config` を一時的に差し込んで、voice_profile.py 本物を
    ファイルから読み込む (関数の中身は本物。差し込んだ偽のモジュールは読み込み後に取り除く)
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

FORK_ROOT = Path(__file__).resolve().parents[3]          # .../pyvideotrans (meeting の 1 つ上)
VOICE_PROFILE_PY = FORK_ROOT / "videotrans" / "util" / "voice_profile.py"

_cached = None


def load_fork_voice_profile():
    """videotrans.util.voice_profile (本物) のモジュールを返す。"""
    global _cached
    if _cached is not None:
        return _cached
    if str(FORK_ROOT) not in sys.path:
        sys.path.insert(0, str(FORK_ROOT))
    if importlib.util.find_spec("PySide6") is not None:
        _cached = importlib.import_module("videotrans.util.voice_profile")
        return _cached
    fake = ["videotrans", "videotrans.configure", "videotrans.configure.config"]
    saved = {k: sys.modules.get(k) for k in fake}
    try:
        for name in fake:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["videotrans.configure.config"].logger = logging.getLogger("voice_profile_under_test")
        spec = importlib.util.spec_from_file_location("voice_profile_under_test", VOICE_PROFILE_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    _cached = mod
    return mod
