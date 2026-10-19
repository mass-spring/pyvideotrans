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
import os
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


# 本体は config を import するとき、証明書とプロキシの環境変数を書き換える（_set_env / fix_ssl_cert_env）:
#   SSL_CERT_FILE・REQUESTS_CA_BUNDLE・CURL_CA_BUNDLE を certifi のものに、no_proxy を自分の一覧に。
# 会社やクラウドの環境にある独自の証明書（検査つきプロキシ）が使えなくなり、ModelScope などへの通信が失敗するので、
# import の前後でこれらだけ元に戻す（本体のファイルは変更しない。本体は import 後も GUI/CLI の通常動作ではこの値を使う）。
_PRESERVE_ENV = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "no_proxy", "NO_PROXY")


def _import(name: str) -> ModuleType:
    ensure_pvt_importable()
    saved = {k: os.environ.get(k) for k in _PRESERVE_ENV}
    try:
        return importlib.import_module(name)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def speakers_module() -> ModuleType:
    """videotrans.process._audio_speakers（built_speakers / cam_speakers / _assign_speakers が入っている）。"""
    return _import("videotrans.process._audio_speakers")


def settings():
    return _import("videotrans.configure.config").settings


def vad_module() -> ModuleType:
    return _import("videotrans.process.vad")


def funasr_module() -> ModuleType:
    return _import("videotrans.process.stt_funasr")


def models_dir() -> Path:
    return Path(_import("videotrans.configure.config").ROOT_DIR) / "models"


def builtin_model_urls() -> list[str]:
    """内蔵の話者分離（built）が使う onnx の取得先（本体の定数。ModelScope 側）。"""
    return list(_import("videotrans.configure.constants").BUILTINT_URL_MS)
