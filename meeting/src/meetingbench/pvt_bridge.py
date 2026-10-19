"""pyVideoTrans 本体の関数を、本体を一切変更せずに呼び出すための入口。

- pyVideoTrans のコードは リポジトリのルート（meeting/ の1つ上）にある。ここでは import するだけで、書き換えない。
- 本体の設定読み込み（videotrans.configure.config）は GUI の部品 PySide6 を import するが、画面は使わない。
  PySide6 が入っていない環境（この測定用の環境）では、tests/conftest.py と同じく空の部品に差し替えて import する。
- 設定ファイル（videotrans/cfg.json）は本体が読むだけ。モデルは ROOT_DIR/models/ の下にある前提
  （本体と同じ置き場所。モデルのダウンロードは model_fetch.py）。
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]   # meeting/src/meetingbench/pvt_bridge.py → リポジトリのルート


def ensure_pvt_importable() -> Path:
    """pyVideoTrans を import できるようにする（sys.path と、PySide6 の差し替え）。戻り値: リポジトリのルート。"""
    if not (REPO_ROOT / "videotrans" / "configure" / "config.py").is_file():
        raise FileNotFoundError(f"pyVideoTrans のコードが見つかりません: {REPO_ROOT}/videotrans（meeting/ の1つ上にある想定です）")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if "PySide6" not in sys.modules and importlib.util.find_spec("PySide6") is None:
        from unittest.mock import MagicMock

        stub = MagicMock()
        stub.QtCore = MagicMock()
        sys.modules["PySide6"] = stub
        sys.modules["PySide6.QtCore"] = stub.QtCore
    return REPO_ROOT


def speakers_module() -> ModuleType:
    """videotrans.process._audio_speakers（built_speakers / cam_speakers / _assign_speakers が入っている）。"""
    ensure_pvt_importable()
    return importlib.import_module("videotrans.process._audio_speakers")


def settings():
    ensure_pvt_importable()
    return importlib.import_module("videotrans.configure.config").settings


def vad_module() -> ModuleType:
    ensure_pvt_importable()
    return importlib.import_module("videotrans.process.vad")


def funasr_module() -> ModuleType:
    ensure_pvt_importable()
    return importlib.import_module("videotrans.process.stt_funasr")


def models_dir() -> Path:
    ensure_pvt_importable()
    return Path(importlib.import_module("videotrans.configure.config").ROOT_DIR) / "models"


def builtin_model_urls() -> list[str]:
    """内蔵の話者分離（built）が使う onnx の取得先（本体の定数。ModelScope 側）。"""
    ensure_pvt_importable()
    return list(importlib.import_module("videotrans.configure.constants").BUILTINT_URL_MS)
