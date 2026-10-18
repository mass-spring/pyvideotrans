"""Fun-CosyVoice3-0.5B-2512（Apache-2.0）のアダプタ。variant = zero_shot / cross_lingual / instruct2。

公式 API の確認（2026-10-18、いずれも自分で取得して読んだ）:
- README   https://github.com/FunAudioLLM/CosyVoice/blob/main/README.md
- 使用例   https://github.com/FunAudioLLM/CosyVoice/blob/main/example.py （cosyvoice3_example）
- 実装     https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/cosyvoice.py （AutoModel / CosyVoice3）
- 指示一覧 https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/utils/common.py （instruct_list）
- モデル   https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 （license: apache-2.0）
- pip パッケージは無い。`git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git` して requirements.txt を入れる
  （README は conda の Python 3.10。requirements.txt は torch==2.3.1・transformers==4.51.3・numpy==1.26.4 などを固定）。

確認できた使い方（CosyVoice3）:
    sys.path.append('third_party/Matcha-TTS')
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')   # cv.sample_rate
    # zero_shot  : prompt_text の頭に 'You are a helpful assistant.<|endofprompt|>' を付ける（CosyVoice3 の決まり）
    cv.inference_zero_shot(tts_text, 'You are a helpful assistant.<|endofprompt|>' + 手本の書き起こし, 手本wavパス, stream=False)
    # cross_lingual: 書き起こし不要。tts_text の頭に同じ接頭辞
    cv.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>' + tts_text, 手本wavパス, stream=False)
    # instruct2  : 感情・速さ・方言の指示。指示文は instruct_list にあるもの
    cv.inference_instruct2(tts_text, 'You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>', 手本wavパス, stream=False)
    # どれも generator。各要素 out['tts_speech'] は (1, N) の torch テンソル。長い文は文ごとに分かれて返るので連結する
- 感情の指示は公式に「开心（喜）・伤心（悲）・生气（怒）」の 3 種と、速い・遅い・大きい・小さい のみ（強さの段階なし）。
  → emotion の文字列から該当する語を探して、公式の固定文に置き換える。見つからなければエラー（勝手な指示文は作らない）。
- 日本語は example.py の注記「for Japanese usage, you must translate it to katakana」に従い、pykakasi があればカタカナに直す
  （助詞 は→ワ・へ→エ・を→オ の簡易処理。変換の質は未検証）。無ければそのまま渡して注意を残す。
- 手本は 30 秒以内（cosyvoice/cli/frontend.py の `_extract_speech_token` に assert がある。この実験の手本は 2〜15 秒なので問題なし）。
  frontend.py は `import whisper`（openai-whisper）と `onnxruntime` を import するので、requirements.txt から除いてはいけない。
未確認（GPU が無く動かしていない）: Colab での requirements の解決（onnxruntime-gpu・deepspeed・tensorrt などは推論に不要なので
  ノートブックでは除いてある）、T4 でのメモリ・速度、日本語カタカナ化の効果、言語またぎでの zero_shot と cross_lingual の差。
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any

from .base import (
    MODE_INSTRUCT,
    Caps,
    Engine,
    EngineError,
    SynthResult,
    emotion_text,
    pkg_version,
    seed_everything,
)

PREFIX = "You are a helpful assistant."
END = "<|endofprompt|>"

# 公式の固定の指示（cosyvoice/utils/common.py の instruct_list）
INSTRUCTIONS = {
    "angry": "请非常生气地说一句话。",
    "sad": "请非常伤心地说一句话。",
    "happy": "请非常开心地说一句话。",
    "fast": "请用尽可能快地语速说一句话。",
    "slow": "请用尽可能慢地语速说一句话。",
    "loud": "Please say a sentence as loudly as possible.",
    "soft": "Please say a sentence in a very soft voice.",
}
# 指示文字列のどの語がどの固定文に当たるか（上から順に最初に当たったもの）
_KEYWORDS = [
    ("angry", ("angry", "anger", "furious", "irritat", "生气", "怒")),
    ("sad", ("sad", "sorrow", "disappoint", "regret", "伤心", "悲")),
    ("happy", ("happy", "cheerful", "joy", "delight", "glad", "开心", "喜")),
    ("loud", ("loud",)),
    ("soft", ("soft", "whisper", "quiet")),
    ("fast", ("faster", "fast", "quick")),
    ("slow", ("slower", "slow")),
]


def map_instruction(emotion: Any) -> str:
    """emotion の文字列を CosyVoice3 の公式の指示文に直す。当たる語が無ければ EngineError。"""
    raw = emotion_text(emotion)
    low = raw.lower()
    for key, words in _KEYWORDS:
        if any(w in low for w in words):
            return INSTRUCTIONS[key]
    raise EngineError(
        f"CosyVoice3 の instruct2 は公式の固定の指示（怒・悲・喜・速い・遅い・大きい・小さい）だけに対応しています。"
        f"感情の指示 '{raw}' から該当する語を見つけられませんでした"
    )


_PUNCT = "、。，．！？!?,.;；:：…"


def to_katakana(text: str) -> tuple[str, str]:
    """日本語をカタカナ（分かち書き）に直す。(変換後, 注意) を返す。pykakasi が無ければそのまま返す。"""
    try:
        import pykakasi
    except ImportError:
        return text, "pykakasi が無いので日本語をカタカナに直していません（pip install pykakasi）"
    try:
        items = pykakasi.kakasi().convert(text)
    except Exception as e:  # noqa: BLE001
        return text, f"カタカナ化に失敗したので元の文を使いました（{type(e).__name__}）"
    particles = {"は": "ワ", "へ": "エ", "を": "オ"}
    toks = [particles.get(it.get("orig", ""), it.get("kana") or it.get("orig", "")) for it in items]
    out = " ".join(t for t in toks if t)
    out = re.sub(rf"\s+([{re.escape(_PUNCT)}])", r"\1", out)
    return out, ""


class CosyVoice3Engine(Engine):
    name = "cosyvoice3"
    VARIANTS = ("", "zero_shot", "cross_lingual", "instruct2")
    # 公式の対応言語（研究メモ research.md 11.1 の「9言語」）
    LANGS = frozenset("zh en ja ko de es fr it ru".split())

    @classmethod
    def caps(cls, variant: str = "") -> Caps:
        v = variant or "zero_shot"
        if v == "instruct2":
            return Caps(cls.name, variant, (MODE_INSTRUCT,), True, cls.LANGS, True)
        return Caps(cls.name, variant, (), False, cls.LANGS, True)

    @property
    def mode(self) -> str:
        return self.variant or "zero_shot"

    def load(self) -> None:
        if self._loaded:
            return
        repo = str(self.opt("repo_dir", os.environ.get("COSYVOICE_REPO", "/content/CosyVoice")))
        if not os.path.isdir(os.path.join(repo, "cosyvoice")):
            raise EngineError(
                f"CosyVoice のコードが見つかりません: {repo}\n"
                "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git でダウンロードし、"
                "engine_options の repo_dir にそのパスを書いてください"
            )
        for p in (repo, os.path.join(repo, "third_party", "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        model_dir = str(self.opt("model_dir", os.path.join(repo, "pretrained_models", "Fun-CosyVoice3-0.5B")))
        try:
            if not os.path.isdir(model_dir):
                from huggingface_hub import snapshot_download

                snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=model_dir)
            from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415 (CosyVoice のリポジトリ側のコード)

            self._m_model = AutoModel(model_dir=model_dir, fp16=bool(self.opt("fp16", False)))
        except ImportError as e:
            raise EngineError(f"CosyVoice の部品が足りません（{e}）。requirements.txt を入れ直してください") from e
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"CosyVoice3 を読み込めません（{type(e).__name__}: {str(e)[:300]}）") from e
        self._loaded = True

    def synthesize(
        self,
        text: str,
        lang: str,
        ref_wav: str,
        ref_text: str | None,
        emotion: Any = None,
        *,
        ref_lang: str | None = None,
        seed: int | None = None,
    ) -> SynthResult:
        if not self._loaded:
            self.load()
        import torch

        notes: list[str] = []
        mode = self.mode
        text_in = text
        if lang == "ja" and self.opt("ja_katakana", True):
            text_in, n = to_katakana(text)
            if n:
                notes.append(n)
        if mode == "zero_shot" and not ref_text:
            mode = "cross_lingual"
            notes.append("書き起こしが無いので cross_lingual に切り替えました")
        if emotion and mode != "instruct2":
            notes.append("このvariantは感情の指示を使わないので無視しました")

        model = self._m_model
        seed_everything(seed)
        try:
            if mode == "zero_shot":
                rt = ref_text or ""
                if (ref_lang or "") == "ja" and self.opt("ja_katakana", True):
                    rt, n = to_katakana(rt)
                    if n:
                        notes.append(n)
                gen = model.inference_zero_shot(text_in, f"{PREFIX}{END}{rt}", ref_wav, stream=False)
            elif mode == "cross_lingual":
                gen = model.inference_cross_lingual(f"{PREFIX}{END}{text_in}", ref_wav, stream=False)
            else:  # instruct2
                if not emotion:
                    raise EngineError("cosyvoice3:instruct2 は感情の指示が必要です（感情の条件でだけ使います）")
                instr = map_instruction(emotion)
                gen = model.inference_instruct2(text_in, f"{PREFIX} {instr}{END}", ref_wav, stream=False)
            chunks = [out["tts_speech"] for out in gen]
        except EngineError:
            raise
        except Exception as e:  # noqa: BLE001
            raise EngineError(f"CosyVoice3 の合成に失敗（{type(e).__name__}: {str(e)[:300]}）") from e
        if not chunks:
            raise EngineError("CosyVoice3 が音声を返しませんでした")
        wav = torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()
        return SynthResult(wav, int(model.sample_rate), mode, " / ".join(notes))

    def info(self) -> dict[str, str]:
        return {"engine": self.spec, "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "pykakasi": pkg_version("pykakasi")}
