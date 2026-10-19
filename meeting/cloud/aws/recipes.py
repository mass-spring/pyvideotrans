"""エンジンごとの実行環境の作り方（ノートブック voice_baseline_colab.ipynb の手順 5-0〜5-4 と同じ）。

research.md 17章: chatterbox-tts は torch==2.6.0 / transformers==5.2.0 を、qwen-tts は transformers==4.57.3 を、
CosyVoice は torch==2.3.1 / numpy==1.26.4 を固定で要求し、同じ環境には入らない。
→ エンジンごとに別の仮想環境（uv venv）を作り、別々のプロセスで generate を実行する。

ここにあるのは「データ」だけ。インスタンスの上では remote_job.py がこれを読んで実行する。
入れるバージョンは、ノートブックが PyPI に存在することを確認して固定したものと一致させる
（test_cloud_aws_run.py が、ノートブックの文字列と食い違っていないかを確かめる）。
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .common import ToolError

REPO_URL = "https://github.com/mass-spring/pyvideotrans"
REPO_BRANCH = "meeting-translator"
COSY_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
UV_VERSION = "0.11.32"
WORKDIR = "/opt/voicelab"
APT_PACKAGES = ["rubberband-cli", "sox", "libsox-dev", "ffmpeg", "git", "curl"]   # 先頭の4つはノートブックと同じ

# 使わないエンジン（重みが非商用。README の voicelab 2章）
FORBIDDEN_ENGINES = {
    "f5tts": "F5-TTS の重みは非商用",
    "f5-tts": "F5-TTS の重みは非商用",
    "omnivoice": "OmniVoice の重みは非商用",
    "higgs": "Higgs Audio の重みは非商用",
    "vevo": "Vevo の重みは非商用",
    "indextts": "IndexTTS は大企業は別契約",
}


@dataclasses.dataclass
class EnvSpec:
    name: str
    python: str
    packages: list[str]
    pre: list[list[str]] = dataclasses.field(default_factory=list)           # 先に入れるもの（別のインデックス用）
    optional_after: list[list[str]] = dataclasses.field(default_factory=list)  # 入らなくてもよい追加
    check_import: str | None = None
    clone: dict[str, Any] | None = None             # {"url":..., "dest":"{cosy_repo}", "recursive":True, "marker":"cosyvoice"}
    requirements_file: str | None = None            # 例 "{cosy_repo}/requirements.txt"
    drop_prefixes: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EngineGroup:
    name: str                     # voicelab generate --engines に渡す名前（= 仮想環境の名前）
    env: EnvSpec
    variants: list[str]           # 設定ファイルの engines に入れる名前

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "env": self.env.to_dict(), "variants": list(self.variants)}


# ノートブック 5-0: 測定用（CPU 版の torch）。voicecheck(ecapa) と voicelab の「測定」「ダミー確認」が動く
SCORE_ENV = EnvSpec(
    name="score",
    python="3.12",
    packages=["numpy", "scipy", "soundfile", "librosa", "openpyxl", "pandas", "praat-parselmouth", "speechbrain==1.0.3"],
    pre=[["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.6.0", "torchaudio==2.6.0"]],
    optional_after=[["setuptools<81", "pyworld"]],   # 入らなくても動く（声の高さを librosa.pyin で測る＝遅い）
)

ENGINE_GROUPS: dict[str, EngineGroup] = {
    "voxcpm2": EngineGroup(
        "voxcpm2",
        EnvSpec("voxcpm2", "3.12", ["torch==2.6.0", "torchaudio==2.6.0", "voxcpm==2.0.3", "numpy", "soundfile", "scipy", "librosa"],
                check_import="voxcpm"),
        ["voxcpm2"]),
    "qwen3tts": EngineGroup(
        "qwen3tts",
        EnvSpec("qwen3tts", "3.12", ["torch==2.6.0", "torchaudio==2.6.0", "qwen-tts==0.1.1", "numpy", "soundfile", "scipy", "librosa"],
                check_import="qwen_tts"),
        ["qwen3tts"]),
    # chatterbox-tts 0.1.7 は torch==2.6.0 / transformers==5.2.0 / numpy<2 を固定で要求する
    # 透かし部品 resemble-perth が pkg_resources を使うので setuptools<81 も入れる（無いと読み込みで TypeError）
    "chatterbox": EngineGroup(
        "chatterbox",
        EnvSpec("chatterbox", "3.11", ["chatterbox-tts==0.1.7", "soundfile", "scipy", "setuptools<81"],
                check_import="chatterbox.mtl_tts"),
        ["chatterbox"]),
    # CosyVoice3: pip では入らないので、公式リポジトリをクローンして requirements.txt を入れる（公式は Python 3.10）。
    # 推論に要らない重い部品は除く。いちばん壊れやすい（Colab でも未確認）
    "cosyvoice3": EngineGroup(
        "cosyvoice3",
        EnvSpec("cosyvoice3", "3.10", ["soundfile", "scipy", "huggingface_hub", "pykakasi==2.3.0"], check_import="onnxruntime",
                clone={"url": COSY_REPO_URL, "dest": "{cosy_repo}", "recursive": True, "marker": "cosyvoice"},
                requirements_file="{cosy_repo}/requirements.txt",
                drop_prefixes=["--", "#", "deepspeed", "tensorrt", "gradio", "fastapi", "grpcio", "tensorboard", "uvicorn", "pyworld"]),
        ["cosyvoice3:zero_shot", "cosyvoice3:cross_lingual"]),
}
DEFAULT_ENGINES = ["voxcpm2", "qwen3tts", "chatterbox"]


def parse_engines(text: str) -> list[EngineGroup]:
    """'voxcpm2,qwen3tts,chatterbox' → EngineGroup のリスト（重複は除く）。"""
    names = [t.strip().lower() for t in (text or "").split(",") if t.strip()]
    if not names:
        raise ToolError("--engines にエンジン名をカンマ区切りで指定してください（例: voxcpm2,qwen3tts,chatterbox）")
    out: list[EngineGroup] = []
    for n in names:
        if n in FORBIDDEN_ENGINES:
            raise ToolError(f"{n} は使いません（{FORBIDDEN_ENGINES[n]}）。")
        if ":" in n:
            raise ToolError(f"{n}: エンジンは名前だけで指定してください（cosyvoice3 は zero_shot と cross_lingual の両方が入ります）")
        if n not in ENGINE_GROUPS:
            raise ToolError(f"知らないエンジン名: {n}。使えるもの: {', '.join(ENGINE_GROUPS)}")
        g = ENGINE_GROUPS[n]
        if g not in out:
            out.append(g)
    return out


def config_engines(groups: list[EngineGroup]) -> list[str]:
    return [v for g in groups for v in g.variants]
