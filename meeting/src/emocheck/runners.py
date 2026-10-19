"""モデルを動かす部分 (SenseVoiceSmall / faster-whisper) と、結果の保存 (途中から再開できる)。

重い部品 (funasr / torch / faster-whisper) は、そのモデルを使うときに初めて import します (lazy import)。
入っていなければ、入れ方を日本語で案内します。テストでは、同じ形の『偽のモデル』(Recognizer) を使うので、モデルのダウンロードは要りません。

公式ページで確認した API (確認日 2026-10-19):
- SenseVoiceSmall (コード MIT、重みは FunASR 独自ライセンス)  https://github.com/FunAudioLLM/SenseVoice
    model = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    res = model.generate(input=wavのパス, cache={}, language="auto" か "ja", use_itn=True, batch_size_s=60)
    res[0]["text"] = '<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは' (言語・感情・音の種類・ITN のタグが先頭に付く)
  pyVideoTrans の SenseVoiceSmall は、あらかじめ切った音声を 1 本ずつ渡し (VAD なし)、言語は指定していない。
  この道具も同じく、1 本の wav を VAD なしで渡す。言語は --sv-language で選べる (既定 auto = pyVideoTrans と同じ)。
- faster-whisper (MIT)  https://github.com/SYSTRAN/faster-whisper  → asrcheck.runners.WhisperEngine をそのまま使う
    ('large-v3-turbo' は mobiuslabsgmbh/faster-whisper-large-v3-turbo。CPU は int8、beam_size=5、VAD なし)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from asrcheck.ioformats import InputError

from .dataset import Clip
from .labels import split_sensevoice_text

SV_DEFAULT_MODEL = "iic/SenseVoiceSmall"
WHISPER_DEFAULT_MODEL = "large-v3-turbo"
INSTALL_HINTS = {
    "funasr": ("uv run --with funasr --with torch --with torchaudio python -m emocheck run ...   (Linux の CPU だけの PC は torch を "
               "--index-url https://download.pytorch.org/whl/cpu から入れるとダウンロードが小さい)"),
    "faster_whisper": "uv run --with faster-whisper python -m emocheck run ...   (または pip install faster-whisper)",
}


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Recognition:
    text: str
    tag: str = ""          # SenseVoice の感情タグ名 ('HAPPY' など。タグなしは '')


class Recognizer(Protocol):
    """1本の音声から (文字, 感情タグ) を返すモデル。テストでは偽物を使う。"""

    key: str               # 短い名前 (キャッシュのファイル名にも使う): sensevoice / whisper
    label: str             # Excel に出す名前
    has_emotion: bool      # 感情タグを出すモデルか

    def recognize(self, path: Path) -> Recognition: ...

    def info(self) -> list[tuple[str, Any]]: ...      # 『実行情報』シートに載せる項目

    def signature(self) -> dict[str, Any]: ...        # 設定が変わったら結果を使い回さないための印


@dataclass
class ClipResult:
    text: str
    tag: str = ""
    sec: float = 0.0       # この1本の処理にかかった秒 (モデルの読み込みは含まない)
    error: str = ""


@dataclass
class ModelRun:
    key: str
    label: str
    has_emotion: bool
    results: dict[str, ClipResult] = field(default_factory=dict)    # file → 結果
    durations: dict[str, float] = field(default_factory=dict)       # file → 音声の長さ (秒)
    info: list[tuple[str, Any]] = field(default_factory=list)
    load_sec: float = 0.0
    n_cached: int = 0      # 前回までの結果を再利用した件数
    n_errors: int = 0


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "(未インストール)"


# ---------------------------------------------------------------------------
# SenseVoiceSmall
# ---------------------------------------------------------------------------
class SenseVoiceRecognizer:
    key = "sensevoice"
    has_emotion = True

    def __init__(self, model: str = SV_DEFAULT_MODEL, language: str = "auto", device: str = "cpu", threads: int = 2) -> None:
        try:
            import torch
            from funasr import AutoModel
        except ImportError:
            raise InputError(f"funasr / torch が入っていません。次のように実行すると一時的に入れて動かせます:\n  {INSTALL_HINTS['funasr']}") from None
        torch.set_num_threads(max(1, threads))
        self.model_id, self.language, self.device, self.threads = model, language, device, threads
        self.label = "SenseVoiceSmall"
        self.model = AutoModel(model=model, device=device, disable_update=True, disable_pbar=True, disable_log=True)

    def recognize(self, path: Path) -> Recognition:
        res = self.model.generate(input=str(path), cache={}, language=self.language, use_itn=True, batch_size_s=60, disable_pbar=True)
        text, tag = split_sensevoice_text(res[0]["text"] if res else "")
        return Recognition(text, tag)

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id, "language": self.language, "use_itn": True}

    def info(self) -> list[tuple[str, Any]]:
        return [
            ("モデル", f"SenseVoiceSmall ({self.model_id})"),
            ("ライブラリ", f"funasr {_version('funasr')} / torch {_version('torch')}"),
            ("設定", f"language={self.language} / use_itn=True / VAD なし / {self.device} / スレッド {self.threads} / 1本ずつ処理"),
            ("ライセンス", ("コードは MIT。重み (モデル本体) は FunASR のモデルライセンス: 商用利用可、ただし出典の表記が必要 "
                            "(https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)。PLAN の方針ではこの条件で使用可")),
        ]


# ---------------------------------------------------------------------------
# faster-whisper
# ---------------------------------------------------------------------------
class WhisperRecognizer:
    key = "whisper"
    has_emotion = False

    def __init__(self, model: str = WHISPER_DEFAULT_MODEL, lang: str = "ja", device: str = "cpu", threads: int = 2) -> None:
        from asrcheck.runners import WhisperEngine, load_audio_16k

        self._load = load_audio_16k
        self.model_id, self.lang, self.device, self.threads = model, lang, device, threads
        self.label = f"Whisper {model}"
        # スレッド数は環境変数 OMP_NUM_THREADS で効く (cli が設定する)。asrcheck と同じ設定 (beam_size=5・VAD なし・CPU は int8)
        self.engine = WhisperEngine(model, device, lang, [])

    def recognize(self, path: Path) -> Recognition:
        return Recognition(self.engine.transcribe_one(self._load(path)))

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id, "language": self.lang, "beam_size": 5}

    def info(self) -> list[tuple[str, Any]]:
        return [
            ("モデル", f"faster-whisper {self.model_id}"),
            ("ライブラリ", f"faster-whisper {_version('faster-whisper')} / ctranslate2 {_version('ctranslate2')}"),
            ("設定", f"language={self.lang} / beam_size=5 / VAD なし / condition_on_previous_text=False / {self.device} int8 / スレッド {self.threads} / 1本ずつ処理"),
            ("ライセンス", "Whisper のコード・重みとも MIT (faster-whisper も MIT)。商用利用可"),
        ]


# ---------------------------------------------------------------------------
# 実行と保存 (途中から再開できる)
# ---------------------------------------------------------------------------
def audio_seconds(path: Path) -> float:
    """音声の長さ (秒)。読めなければ 0。"""
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return 0.0


def _load_cache(path: Path, sig: dict[str, Any]) -> dict[str, ClipResult]:
    out: dict[str, ClipResult] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue   # 途中で止まって壊れた最後の行は捨てる
        if isinstance(d, dict) and d.get("sig") == sig and "file" in d and not d.get("error"):
            out[d["file"]] = ClipResult(d.get("text", ""), d.get("tag", ""), float(d.get("sec", 0.0)))
    return out


def run_model(
    rec: Recognizer, clips: list[Clip], cache_path: Path | None, load_sec: float = 0.0,
    progress: Callable[[str], None] = say, every: int = 20,
) -> ModelRun:
    """clips を 1 本ずつ rec に通す。結果は 1 本ごとに cache_path (JSON Lines) に書き足すので、止まっても続きから再開できる。"""
    run = ModelRun(rec.key, rec.label, rec.has_emotion, info=rec.info(), load_sec=load_sec)
    sig = rec.signature()
    cached = _load_cache(cache_path, sig) if cache_path else {}
    todo = [c for c in clips if c.file not in cached]
    for c in clips:
        run.durations[c.file] = audio_seconds(c.path)
        if c.file in cached:
            run.results[c.file] = cached[c.file]
    run.n_cached = len(clips) - len(todo)
    if run.n_cached:
        progress(f"  前回の結果 {run.n_cached} 件を再利用します ({cache_path})")
    fh = None
    if cache_path and todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(cache_path, "a", encoding="utf-8")
    t_start = time.perf_counter()
    try:
        for i, c in enumerate(todo, start=1):
            t0 = time.perf_counter()
            try:
                r = rec.recognize(c.path)
                res = ClipResult(r.text, r.tag, time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001  (1本の失敗で全体を止めない。失敗は表に残す)
                res = ClipResult("", "", time.perf_counter() - t0, error=f"{type(e).__name__}: {e}"[:200])
                run.n_errors += 1
            run.results[c.file] = res
            if fh:
                fh.write(json.dumps({"file": c.file, "sig": sig, "text": res.text, "tag": res.tag, "sec": round(res.sec, 3),
                                     "error": res.error}, ensure_ascii=False) + "\n")
                fh.flush()
            if i % every == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                progress(f"  {rec.label}: {i}/{len(todo)} 件 ({el:.0f} 秒経過 / 残り約 {el / i * (len(todo) - i):.0f} 秒)")
    finally:
        if fh:
            fh.close()
    return run
