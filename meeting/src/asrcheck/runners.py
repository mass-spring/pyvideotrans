"""(任意) 音声から認識結果の CSV を作る: asrcheck run --engine ... --diarizer ...

評価 (asrcheck eval) には不要です。重い部品 (torch / transformers / faster-whisper / NeMo / funasr / pyannote) は
使うエンジンを選んだときに初めて import します (lazy import)。入っていなければ、入れ方を日本語で案内します。

公式ページで確認した API (確認日 2026-10-18)。(要検証) と書いたものは、公式の資料だけでは確かめきれていない部分:

- faster-whisper (MIT)  https://github.com/SYSTRAN/faster-whisper (README / faster_whisper/transcribe.py, v1.2.1)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio, language="ja", beam_size=5, vad_filter=True, hotwords="...")
    audio は パス / ファイルオブジェクト / numpy 配列 (16kHz の float32)。segments は『ジェネレータ』で、回して初めて処理が進む。
    hotwords は transcribe() の引数 (「モデルへのヒントとなる語句。prefix 指定時は無効」)。
- Cohere Transcribe (Apache-2.0)  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026  (モデルカードは同意が必要な gated)
    https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/cohere_asr.md  (公開されている transformers の説明)
    processor = AutoProcessor.from_pretrained(ID); model = CohereAsrForConditionalGeneration.from_pretrained(ID, device_map="auto")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="ja"); inputs.to(model.device, dtype=model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=256); text = processor.decode(outputs, skip_special_tokens=True)
    transformers>=5.4.0 (モデルカードの要約より (要検証))。言語は手動指定 (自動判定なし)。タイムスタンプ・話者分離・語句ヒントはなし。
    長い音声は processor が自動で分割し、audio_chunk_index を decode に渡して結合する (本ツールは30秒以下に切って渡すので使わない)。
    ※モデルカード本文は gated で直接読めず、transformers の公式ドキュメントと検索結果の要約で確認した (要検証)。
- Qwen3-ASR (Apache-2.0)  https://github.com/QwenLM/Qwen3-ASR (README と qwen_asr/inference/qwen3_asr.py, qwen-asr 0.0.6)
    model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=32, max_new_tokens=256)
    results = model.transcribe(audio=(np配列, 16000), context="用語…", language="Japanese")  → results[0].text
    context は『文脈・用語』の文字列 (リストなら音声ごと)。language は英語名 ("Japanese")。audio は パス / URL / (np.ndarray, sr)。
    context の推奨の書き方は公式に記載なし (要検証: 本ツールは用語を読点でつないで渡す)。
- NVIDIA parakeet-tdt_ctc-0.6b-ja (CC-BY-4.0)  https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-0.6b-ja")
    output = asr_model.transcribe(["speech.wav"]); output[0].text     入力は 16kHz モノラル wav。デコーダ既定は TDT (CTC は decoding_type='ctc')。
    語句ヒントの仕組みは公式ページにない。macOS での動作は公式に記載なし (要検証)。
- SenseVoiceSmall (コード MIT、重みは FunASR 独自ライセンス)  https://github.com/FunAudioLLM/SenseVoice
    model = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0")   # VAD を付けるなら vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}
    res = model.generate(input=パス, cache={}, language="ja", use_itn=True, batch_size_s=60)  → res[0]["text"] に <|ja|><|HAPPY|> などのタグが付く
    語句ヒントは SenseVoiceSmall 本体にはない (別配布の SenseVoiceSmall_hotword がある)。重みは初回に ModelScope から取得 (オフライン運用は要検証)。
- pyannote.audio 4.x (コード MIT)  / pyannote/speaker-diarization-community-1 (CC-BY-4.0, gated)
    https://github.com/pyannote/pyannote-audio (README と src/pyannote/audio/pipelines/speaker_diarization.py, v4.0.7)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HFトークン)
    output = pipeline({"waveform": Tensor(1,T), "sample_rate": 16000}, num_speakers=N)   # min_speakers / max_speakers も可
    output.speaker_diarization / output.exclusive_speaker_diarization  (どちらも pyannote.core.Annotation。
    exclusive は『同時に話す人を1人にした』版で、文字起こしの時刻との突き合わせ用)。
    ※モデルカード本文は gated で直接読めないため、num_speakers とプロパティ名はソースコード (docstring) で確認した。

公式の例に無く、動作を確かめていないもの (要検証):
- Cohere / Qwen3-ASR を CPU で動かす指定 (公式の例は device_map="auto" / "cuda:0")。Apple の MPS は選ばない。
- parakeet の Mac での動作、NeMo の map_location 等の細かい指定 (本ツールは既定のまま)。
- SenseVoice: AutoModel(disable_update=True) (README にはない引数)、重みの取得元 ModelScope をオフラインで使う方法。
- 実際に動かして確かめたのは faster-whisper (small / base, CPU int8) だけ。ほかは偽のモジュールで「公式の例と同じ形で呼んでいるか」だけをテストしている。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from .ioformats import InputError

SR = 16000
ENGINES = ("whisper", "cohere", "qwen3asr", "parakeet", "sensevoice")
DIARIZERS = ("none", "pyannote")
DEFAULT_MODELS = {
    "whisper": "large-v3",
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
    "qwen3asr": "Qwen/Qwen3-ASR-1.7B",
    "parakeet": "nvidia/parakeet-tdt_ctc-0.6b-ja",
    "sensevoice": "iic/SenseVoiceSmall",
}
DIARIZER_MODEL = "pyannote/speaker-diarization-community-1"
LANG_NAMES = {"ja": "Japanese", "en": "English", "fr": "French"}
INSTALL_HINTS = {
    "faster_whisper": "uv run --with faster-whisper python -m asrcheck run ...   (または pip install faster-whisper)",
    "transformers": "uv run --with 'transformers>=5.4' --with torch --with accelerate --with sentencepiece --with librosa python -m asrcheck run ...",
    "qwen_asr": "uv run --with qwen-asr python -m asrcheck run ...   (または pip install -U qwen-asr)",
    "nemo": "pip install 'nemo_toolkit[asr]'   (PyTorch を入れた後に。Linux/NVIDIA GPU 向けで、Mac での動作は公式に記載なし)",
    "funasr": "uv run --with funasr --with torch --with torchaudio python -m asrcheck run ...",
    "pyannote": "uv run --with pyannote.audio python -m asrcheck run ...   (Hugging Face で pyannote/speaker-diarization-community-1 に同意し、環境変数 HF_TOKEN にトークンを設定)",
}


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 用語リスト
# ---------------------------------------------------------------------------
def read_hotwords(path: str | Path | None) -> list[str]:
    """1行1語。空行と # で始まる行は無視。重複は除く。"""
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        raise InputError(f"用語リストが見つかりません: {p}")
    from .ioformats import read_text

    words: list[str] = []
    for ln in read_text(p)[0].splitlines():
        w = ln.strip()
        if w and not w.startswith("#"):
            words.append(w)
    return list(dict.fromkeys(words))


# ---------------------------------------------------------------------------
# 音声の読み込み (16kHz モノラル float32)
# ---------------------------------------------------------------------------
def resample(y: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return y.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(int(sr), int(target))
    return resample_poly(y, target // g, sr // g).astype(np.float32)


def load_audio_16k(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"音声ファイルが見つかりません: {p}")
    try:
        import soundfile as sf

        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        return resample(y.mean(axis=1), sr)
    except Exception:  # noqa: BLE001  (m4a / aac / mp4 など soundfile で開けない形式は ffmpeg に任せる)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise InputError(
                f"音声を開けません: {p}\n  wav / flac / ogg 以外 (m4a, mp4 など) には ffmpeg が必要です。ffmpeg を入れるか、wav に変換してください"
            ) from None
        r = subprocess.run([ffmpeg, "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
        if r.returncode != 0 or not r.stdout:
            raise InputError(f"ffmpeg で音声を開けませんでした: {p}\n  {r.stderr.decode(errors='replace')[:300]}") from None
        return np.frombuffer(r.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# 区間の作り方
# ---------------------------------------------------------------------------
def energy_segments(
    y: np.ndarray, sr: int = SR, frame_ms: int = 30, min_silence: float = 0.5, min_len: float = 0.3,
    max_len: float = 30.0, pad: float = 0.1, margin_db: float = 12.0,
) -> list[tuple[float, float]]:
    """音量だけで話している区間を見つける簡単な VAD (numpy のみ)。静かな部屋の録音向けの粗い切り出し。

    音量 (dB) が『雑音の底 (下位2%点) + margin_db』より大きい所を発話とみなし、min_silence 秒以下の無音は埋める。
    max_len 秒を超える区間は、その中でいちばん静かな所で切る。
    """
    n = int(sr * frame_ms / 1000)
    if len(y) < n:
        return []
    nf = len(y) // n
    frames = y[: nf * n].reshape(nf, n)
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    floor = float(np.percentile(db, 2))
    peak = float(np.percentile(db, 98))
    if peak - floor < 6.0:   # 音量の差がほとんど無い (無音、または一定のノイズ) → 発話なし
        return []
    thr = max(floor + margin_db, peak - 40.0)
    speech = db > thr
    fill = max(1, int(round(min_silence * 1000 / frame_ms)))
    idx = np.nonzero(speech)[0]
    if len(idx) == 0:
        return []
    runs: list[list[int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > fill:
            runs.append([start, prev])
            start = i
        prev = i
    runs.append([start, prev])
    max_frames = int(max_len * 1000 / frame_ms)
    out: list[tuple[float, float]] = []
    for a, b in runs:
        while b - a + 1 > max_frames:
            lo, hi = a + max_frames // 2, a + max_frames
            cut = lo + int(np.argmin(db[lo:hi]))
            out.append((a, cut))
            a = cut + 1
        out.append((a, b))
    dur = len(y) / sr
    segs = [(max(0.0, a * frame_ms / 1000 - pad), min(dur, (b + 1) * frame_ms / 1000 + pad)) for a, b in out]
    return [(s, e) for s, e in segs if e - s >= min_len]


def turns_to_segments(turns: list[tuple[float, float, str]], merge_gap: float = 0.8, max_len: float = 30.0,
                      min_len: float = 0.2) -> list[tuple[float, float, str]]:
    """話者の区間を、文字起こしにかける単位にそろえる: 同じ話者が続くとき (間に他の人が入らない) は gap 秒以内なら1つにつなぎ、
    max_len 秒を超えるものは等分する。min_len 秒未満は捨てる。"""
    turns = sorted(turns)
    merged: list[list[Any]] = []
    for s, e, spk in turns:
        if merged and merged[-1][2] == spk and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, spk])
    out: list[tuple[float, float, str]] = []
    for s, e, spk in merged:
        if e - s < min_len:
            continue
        k = int(np.ceil((e - s) / max_len))
        step = (e - s) / k
        out += [(s + i * step, s + (i + 1) * step, spk) for i in range(k)]
    return out


# ---------------------------------------------------------------------------
# エンジン
# ---------------------------------------------------------------------------
def _missing(pkg: str, key: str) -> InputError:
    return InputError(f"{pkg} が入っていません。次のように実行すると一時的に入れて動かせます:\n  {INSTALL_HINTS[key]}")


def pick_device(device: str) -> str:
    """auto → cuda があれば cuda、なければ cpu (mps は選ばない: エンジンごとの対応が確認できていないため)。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class Engine:
    name = ""
    supports_hotwords = False
    native_segments = False

    def __init__(self, model: str, device: str, lang: str, hotwords: list[str]) -> None:
        self.model_id, self.device, self.lang, self.hotwords = model, pick_device(device), lang, hotwords
        if hotwords and not self.supports_hotwords:
            say(f"注意: {self.name} は用語リスト (--hotwords) に対応していません。無視します")

    def transcribe_batch(self, waves: list[np.ndarray]) -> list[tuple[str, str]]:
        """(文字, 感情タグ) のリスト。感情タグは SenseVoice だけ。"""
        return [(self.transcribe_one(w), "") for w in waves]

    def transcribe_one(self, wave: np.ndarray) -> str:
        raise NotImplementedError

    def transcribe_long(self, wave: np.ndarray) -> list[tuple[float, float, str]]:
        raise NotImplementedError


class WhisperEngine(Engine):
    name = "whisper"
    supports_hotwords = True
    native_segments = True

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise _missing("faster-whisper", "faster_whisper") from None
        compute = "float16" if self.device == "cuda" else "int8"   # README: GPU は float16、CPU は int8
        self.model = WhisperModel(self.model_id, device="cuda" if self.device == "cuda" else "cpu", compute_type=compute)

    def _kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"language": self.lang, "beam_size": 5, "condition_on_previous_text": False}
        if self.hotwords:
            kw["hotwords"] = "、".join(self.hotwords) if self.lang == "ja" else ", ".join(self.hotwords)
        return kw

    def transcribe_one(self, wave: np.ndarray) -> str:
        segs, _ = self.model.transcribe(wave, **self._kw())
        return "".join(s.text for s in segs).strip()   # ジェネレータなので回して初めて処理が進む

    def transcribe_long(self, wave: np.ndarray) -> list[tuple[float, float, str]]:
        segs, _ = self.model.transcribe(wave, vad_filter=True, **self._kw())
        return [(float(s.start), float(s.end), s.text.strip()) for s in segs if s.text.strip()]


class CohereEngine(Engine):
    name = "cohere"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        try:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        except ImportError:
            raise _missing("transformers (5.4 以上)", "transformers") from None
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(self.model_id, device_map="auto" if self.device == "cuda" else None)
        if self.device != "cuda":
            self.model.to("cpu")

    def transcribe_one(self, wave: np.ndarray) -> str:
        inputs = self.processor(wave, sampling_rate=SR, return_tensors="pt", language=self.lang)
        chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=512)
        if chunk_index is not None:
            text = self.processor.decode(out, skip_special_tokens=True, audio_chunk_index=chunk_index, language=self.lang)
        else:
            text = self.processor.decode(out, skip_special_tokens=True)
        return (text[0] if isinstance(text, list) else text).strip()


class Qwen3AsrEngine(Engine):
    name = "qwen3asr"
    supports_hotwords = True

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise _missing("qwen-asr", "qwen_asr") from None
        cuda = self.device == "cuda"
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id, dtype=torch.bfloat16 if cuda else torch.float32, device_map="cuda:0" if cuda else "cpu",
            max_inference_batch_size=8, max_new_tokens=512,
        )

    def transcribe_batch(self, waves: list[np.ndarray]) -> list[tuple[str, str]]:
        ctx = "、".join(self.hotwords) if self.hotwords else ""
        out: list[tuple[str, str]] = []
        for i in range(0, len(waves), 8):
            res = self.model.transcribe(audio=[(w, SR) for w in waves[i:i + 8]], context=ctx, language=LANG_NAMES[self.lang])
            out += [(r.text.strip(), "") for r in res]
        return out


class ParakeetEngine(Engine):
    name = "parakeet"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        if self.lang != "ja":
            raise InputError("parakeet-tdt_ctc-0.6b-ja は日本語専用です (--lang ja)")
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise _missing("NeMo (nemo_toolkit[asr])", "nemo") from None
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)

    def transcribe_batch(self, waves: list[np.ndarray]) -> list[tuple[str, str]]:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, w in enumerate(waves):
                p = str(Path(td) / f"s{i:05d}.wav")
                sf.write(p, w, SR, subtype="PCM_16")   # 入力は 16kHz モノラル wav (公式)
                paths.append(p)
            res = self.model.transcribe(paths)
        texts = [(r.text if hasattr(r, "text") else str(r)).strip() for r in res]
        return [(t, "") for t in texts]


_SV_TAG = re.compile(r"<\|([^|]*)\|>")
_SV_EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}


def split_sensevoice_tags(raw: str) -> tuple[str, str]:
    """'<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは' → ('こんにちは', 'HAPPY')。"""
    tags = _SV_TAG.findall(raw)
    emo = next((t for t in tags if t in _SV_EMOTIONS), "")
    return _SV_TAG.sub("", raw).strip(), emo


class SenseVoiceEngine(Engine):
    name = "sensevoice"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        try:
            from funasr import AutoModel
        except ImportError:
            raise _missing("funasr", "funasr") from None
        self.model = AutoModel(model=self.model_id, device="cuda:0" if self.device == "cuda" else "cpu", disable_update=True)

    def transcribe_batch(self, waves: list[np.ndarray]) -> list[tuple[str, str]]:
        import soundfile as sf

        out: list[tuple[str, str]] = []
        with tempfile.TemporaryDirectory() as td:
            for i, w in enumerate(waves):
                p = str(Path(td) / f"s{i:05d}.wav")
                sf.write(p, w, SR, subtype="PCM_16")
                res = self.model.generate(input=p, cache={}, language=self.lang, use_itn=True, batch_size_s=60)
                out.append(split_sensevoice_tags(res[0]["text"] if res else ""))
        return out


ENGINE_CLASSES: dict[str, type[Engine]] = {
    "whisper": WhisperEngine, "cohere": CohereEngine, "qwen3asr": Qwen3AsrEngine, "parakeet": ParakeetEngine, "sensevoice": SenseVoiceEngine,
}


# ---------------------------------------------------------------------------
# 話者分離
# ---------------------------------------------------------------------------
def pyannote_turns(wave: np.ndarray, num_speakers: int | None, min_speakers: int | None, max_speakers: int | None,
                   device: str) -> list[tuple[float, float, str]]:
    """pyannote community-1 の exclusive (同時に1人だけ) 出力を (開始, 終了, 話者) にして返す。"""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise _missing("pyannote.audio", "pyannote") from None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(DIARIZER_MODEL, token=token)
    except Exception as e:  # noqa: BLE001
        raise InputError(
            f"pyannote のモデルを読み込めませんでした: {e}\n"
            f"  初回は Hugging Face で {DIARIZER_MODEL} の利用条件に同意し、環境変数 HF_TOKEN にトークンを設定してください。"
            "一度ダウンロードできれば、次からは HF_HUB_OFFLINE=1 でネット無しでも動きます"
        ) from None
    if pick_device(device) == "cuda":
        pipeline.to(torch.device("cuda"))
    kw: dict[str, int] = {}
    if num_speakers:
        kw["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kw["min_speakers"] = min_speakers
        if max_speakers:
            kw["max_speakers"] = max_speakers
    out = pipeline({"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": SR}, **kw)
    ann = out.exclusive_speaker_diarization
    return [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]


def relabel_speakers(turns: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """話者名を、最初に話した順に spk0, spk1, ... (pyVideoTrans の speaker.json と同じ書き方) に付け替える。"""
    names: dict[str, str] = {}
    out = []
    for s, e, spk in sorted(turns):
        out.append((s, e, names.setdefault(spk, f"spk{len(names)}")))
    return out


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def transcribe_audio(
    engine: Engine, wave: np.ndarray, turns: list[tuple[float, float, str]] | None, batch: int = 16,
) -> list[dict[str, Any]]:
    """戻り値: [{start, end, speaker, text, emotion}]"""
    rows: list[dict[str, Any]] = []
    if turns is None and engine.native_segments:
        for s, e, t in engine.transcribe_long(wave):
            rows.append({"start": s, "end": e, "speaker": "", "text": t, "emotion": ""})
        return rows
    if turns is None:
        segs = [(s, e, "") for s, e in energy_segments(wave)]
    else:
        segs = turns_to_segments(turns)
    for i in range(0, len(segs), batch):
        chunk = segs[i:i + batch]
        waves = [wave[int(s * SR):int(e * SR)] for s, e, _ in chunk]
        for (s, e, spk), (text, emo) in zip(chunk, engine.transcribe_batch(waves)):
            if text.strip():
                rows.append({"start": s, "end": e, "speaker": spk, "text": text.strip(), "emotion": emo})
    return rows


def write_hyp_csv(path: Path, rows: list[dict[str, Any]], with_file: bool, with_emotion: bool) -> None:
    cols = (["file"] if with_file else []) + ["start", "end", "speaker", "text"] + (["emotion"] if with_emotion else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM 付き UTF-8 (Excel でそのまま開ける)
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.2f}" if c in ("start", "end") else r.get(c, "") for c in cols])


def add_run_arguments(r: argparse.ArgumentParser) -> None:
    r.add_argument("--engine", required=True, choices=ENGINES, help="文字起こしエンジン")
    r.add_argument("--diarizer", default="none", choices=DIARIZERS, help="話者分離: pyannote (community-1) / none (話者なし)")
    r.add_argument("--audio", required=True, action="append", help="録音ファイル (繰り返し可。複数のときは出力に file 列が付く)")
    r.add_argument("--num-speakers", type=int, help="参加人数 (pyannote に渡す。分かっているときは必ず指定すると精度が上がる)")
    r.add_argument("--min-speakers", type=int, help="人数の下限 (--num-speakers が無いとき)")
    r.add_argument("--max-speakers", type=int, help="人数の上限 (--num-speakers が無いとき)")
    r.add_argument("--hotwords", help="用語・人名リスト (1行1語のテキスト)。対応エンジン: whisper, qwen3asr")
    r.add_argument("--lang", default="ja", choices=("ja", "en", "fr"), help="言語 (既定 ja)")
    r.add_argument("--model", help="モデル名/フォルダ (既定: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODELS.items()) + ")")
    r.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="実行する場所 (既定 auto: GPU があれば使う)")
    r.add_argument("--out", help="出力する CSV (既定: <音声名>.<エンジン>.csv)")


def cmd_run(args: argparse.Namespace) -> int:
    audio = [Path(a) for a in args.audio]
    for a in audio:
        if not a.is_file():
            raise InputError(f"音声ファイルが見つかりません: {a}")
    hot = read_hotwords(args.hotwords)
    model_id = args.model or DEFAULT_MODELS[args.engine]
    out = Path(args.out) if args.out else Path(f"{audio[0].stem}.{args.engine}.csv")
    t_all = time.perf_counter()
    say(f"モデルを準備しています: {args.engine} ({model_id}) ...")
    t0 = time.perf_counter()
    engine = ENGINE_CLASSES[args.engine](model_id, args.device, args.lang, hot)
    load_s = time.perf_counter() - t0
    rows_all: list[dict[str, Any]] = []
    audio_total = 0.0
    diar_rows: list[dict[str, Any]] = []
    for a in audio:
        say(f"音声を読み込んでいます: {a.name}")
        wave = load_audio_16k(a)
        audio_total += len(wave) / SR
        turns = None
        if args.diarizer == "pyannote":
            say("話者を分けています (pyannote community-1) ...")
            turns = relabel_speakers(pyannote_turns(wave, args.num_speakers, args.min_speakers, args.max_speakers, args.device))
            say(f"  話者区間 {len(turns)} 個 / 話者 {len({t[2] for t in turns})} 人")
            diar_rows += [{"file": a.stem, "start": s, "end": e, "speaker": spk, "text": "", "emotion": ""} for s, e, spk in turns]
        say("文字起こししています ...")
        rows = transcribe_audio(engine, wave, turns)
        for r in rows:
            r["file"] = a.stem
        rows_all += rows
        say(f"  発言 {len(rows)} 件")
    elapsed = time.perf_counter() - t_all
    write_hyp_csv(out, rows_all, with_file=len(audio) > 1, with_emotion=any(r["emotion"] for r in rows_all))
    if diar_rows:
        write_hyp_csv(out.with_suffix(".diar.csv"), diar_rows, with_file=len(audio) > 1, with_emotion=False)
    meta = {
        "engine": args.engine, "model": model_id, "diarizer": args.diarizer, "device": engine.device, "lang": args.lang,
        "num_speakers": args.num_speakers, "hotwords": len(hot), "audio": [str(a) for a in audio],
        "audio_sec": round(audio_total, 2), "elapsed_sec": round(elapsed, 2), "model_load_sec": round(load_s, 2),
        "rtf": round(elapsed / audio_total, 4) if audio_total else None,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    say(f"完了: {out.resolve()}  ({len(rows_all)} 件 / 処理 {elapsed:.1f} 秒 / 音声 {audio_total:.1f} 秒 / RTF {meta['rtf']})")
    if args.diarizer == "pyannote":
        say(f"  話者区間そのもの: {out.with_suffix('.diar.csv').resolve()}  (eval の --diar に使えます)")
    say(f"次は: asrcheck eval --ref 正解.csv --hyp {args.engine}={out} --out report.xlsx")
    return 0


