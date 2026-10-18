"""voicelab：Colab ノートブック（meeting/notebooks/voice_baseline_colab.ipynb）の形と中身の検査。"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from voicelab.cli import build_parser
from voicelab.config import config_from_dict

NB = Path(__file__).resolve().parents[2] / "notebooks" / "voice_baseline_colab.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _code(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _md(nb) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]


def test_notebook_is_valid_nbformat_v4(nb):
    assert nb["nbformat"] == 4 and isinstance(nb["nbformat_minor"], int)
    assert {"cells", "metadata"} <= set(nb)
    assert nb["cells"], "セルがありません"
    ids = []
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        assert isinstance(c["source"], list) and all(isinstance(s, str) for s in c["source"])
        assert isinstance(c["metadata"], dict)
        if c["cell_type"] == "code":
            assert "outputs" in c and "execution_count" in c
        if nb["nbformat_minor"] >= 5:
            ids.append(c["id"])
    assert len(ids) == len(set(ids))
    try:
        import nbformat
    except ImportError:
        return
    nbformat.validate(nbformat.reads(NB.read_text(encoding="utf-8"), as_version=4))


def test_notebook_has_no_saved_outputs_or_secrets(nb):
    text = NB.read_text(encoding="utf-8")
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    assert not re.search(r"ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}", text)


def test_all_code_cells_are_valid_python(nb):
    for src in _code(nb):
        body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        ast.parse(body)


def test_data_rule_warning_and_consent_are_present(nb):
    md = "\n".join(_md(nb))
    assert "自分自身の声" in md and "同意を得た模擬会議" in md and "公開データ" in md
    assert "会社の会議音声" in md and "アップロードしない" in md
    code = _code(nb)
    consent = next(i for i, s in enumerate(code) if "I_AGREE" in s)
    assert "assert I_AGREE" in code[consent]
    # 同意の確認は、GPU 確認・Drive・インストールより前
    for needle in ("drive.mount", "git\", \"clone", "make_env("):
        first = next(i for i, s in enumerate(code) if needle in s)
        assert consent < first, needle


def test_notebook_covers_the_requested_steps(nb):
    code = "\n".join(_code(nb))
    assert "nvidia-smi" in code                                              # GPU 確認
    assert "drive.mount" in code and "MyDrive" in code                       # Drive に保存
    assert "https://github.com/mass-spring/pyvideotrans" in code and "meeting-translator" in code
    for pinned in ("voxcpm==2.0.3", "qwen-tts==0.1.1", "chatterbox-tts==0.1.7", "speechbrain==1.0.3", "uv==0.11.32"):
        assert pinned in code, pinned
    assert "prepare_testdata" in code                                        # 公開データでの最初の試行
    assert "files.upload" in code and "files.download" in code               # 自分の声のアップロード／Excel の取得
    assert "write_targets" in code and "voicelab_result.xlsx" in code
    assert "FunAudioLLM/CosyVoice" in code and "--recursive" in code        # CosyVoice は pip ではなく git clone
    # 各エンジンは別の仮想環境（ライブラリの版が衝突するため）で、使うものだけ入れる
    for env in ("voxcpm2", "qwen3tts", "chatterbox", "cosyvoice3", "score"):
        assert f'make_env(\n    "{env}"' in code or f'make_env("{env}"' in code, env
    assert "USE_COSYVOICE3 = False" in code                                  # 一番壊れやすいので既定はオフ


def test_notebook_does_not_use_non_commercial_models(nb):
    code = "\n".join(_code(nb)).lower()
    for bad in ("f5-tts", "f5tts", "omnivoice", "higgs", "vevo", "indextts", "index-tts"):
        assert bad not in code, bad


def test_notebook_uses_only_existing_voicelab_subcommands(nb):
    choices = set(build_parser()._subparsers._group_actions[0].choices)
    code = "\n".join(_code(nb))
    used = set(re.findall(r'vl\(\s*[^,()]+,\s*"([a-z]+)"', code)) | set(re.findall(r'args = \["([a-z]+)"', code))
    used |= set(re.findall(r'"voicelab",\s*"([a-z]+)"', code))
    assert used and used <= choices, used - choices


def test_notebook_settings_make_a_config_that_voicelab_accepts(nb, tmp_path):
    """ノートブックの「設定」セルと「設定ファイルを書く」セルを実行して、できた設定が voicelab に通ることを確かめる。"""
    cells = {s.splitlines()[0]: s for s in _code(nb)}
    settings = next(s for k, s in cells.items() if k.startswith("#@title 3."))
    cfg_cell = next(s for k, s in cells.items() if k.startswith("#@title 8-1."))
    settings = settings.replace('"/content/drive/MyDrive/voicelab"', repr(str(tmp_path / "drive")))
    settings = settings.replace('REPO = "/content/pyvideotrans"', f'REPO = {str(NB.parents[2])!r}')
    for mode in ("quick", "full"):
        ns: dict = {}
        exec(settings.replace('MODE = "quick"', f'MODE = "{mode}"').replace("USE_COSYVOICE3 = False", "USE_COSYVOICE3 = True"), ns)
        ns["TARGETS_CSV"] = str(tmp_path / "targets.csv")
        ns["SPEAKER_SPEC"] = {"name": "A", "profile": str(tmp_path / "profiles" / "A")}
        ns["CFG_PATH"] = str(tmp_path / f"exp_{mode}.json")
        ns["json"] = json
        ns["sh"] = lambda *a, **k: 0
        head = cfg_cell.split("def vl(")[0]   # 設定の書き出しまで（voicelab の実行は含めない）
        exec(head, ns)
        cfg = config_from_dict(json.loads(Path(ns["CFG_PATH"]).read_text(encoding="utf-8")), tmp_path)
        assert cfg.engines == ["voxcpm2", "qwen3tts", "cosyvoice3:zero_shot", "cosyvoice3:cross_lingual", "chatterbox"]
        assert cfg.emotion_engines == ["voxcpm2"] and cfg.target_langs == ["ja", "en", "fr", "zh"]
        assert (cfg.repeats, cfg.max_texts_per_lang) == ((1, 1) if mode == "quick" else (2, None))
        assert cfg.conditions == ["line_ref_16k", "profile_ref", "stretch", "emotion"]
