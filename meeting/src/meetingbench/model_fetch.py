"""モデルの取得（pyVideoTrans の help_down と同じファイル・同じ置き場所）。

本体の down_file_from_hf は TLS の検証を切って（verify=False）通信するため使わず、検証ありで取得する。
  - built: ROOT_DIR/models/onnx/ に seg_model.onnx など3つ（取得先 URL は本体の constants.BUILTINT_URL_MS）
  - ali_CAM: modelscope の iic/speech_campplus_speaker-diarization_common → ROOT_DIR/models/speech_campplus_speaker-diarization_common
  - SenseVoiceSmall: modelscope の iic/SenseVoiceSmall → ROOT_DIR/models/SenseVoiceSmall
  - fsmn-vad（長めの区間の比較用）: iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from . import pvt_bridge

MS_MODELS = {
    "cam": ("iic/speech_campplus_speaker-diarization_common", "speech_campplus_speaker-diarization_common"),
    "sensevoice": ("iic/SenseVoiceSmall", "SenseVoiceSmall"),
    "fsmn_vad": ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_builtin(progress=print) -> list[Path]:
    """built 用の onnx を models/onnx/ に取得（あれば何もしない）。TLS は検証する。"""
    dst = pvt_bridge.models_dir() / "onnx"
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for url in pvt_bridge.builtin_model_urls():
        p = dst / url.split("?")[0].rsplit("/", 1)[-1]
        out.append(p)
        if p.is_file() and p.stat().st_size > 0:
            continue
        progress(f"ダウンロード: {url}")
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:   # 既定で TLS を検証する
            shutil.copyfileobj(r, f)
        tmp.replace(p)
    return out


def fetch_modelscope(kind: str, progress=print) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    model_id, sub = MS_MODELS[kind]
    local = pvt_bridge.models_dir() / sub
    progress(f"モデルを確認: {model_id}")
    snapshot_download(model_id=model_id, local_dir=str(local))
    return local


def model_info() -> list[dict]:
    """実行情報シート用: 使うモデルのファイル名・サイズ・SHA-256（あるものだけ）。"""
    md = pvt_bridge.models_dir()
    rows = []
    for p in sorted((md / "onnx").glob("*.onnx")):
        rows.append({"file": f"onnx/{p.name}", "bytes": p.stat().st_size, "sha256": sha256(p)})
    return rows
