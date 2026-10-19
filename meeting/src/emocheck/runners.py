"""モデルを動かす部分 (SenseVoiceSmall / faster-whisper) と、結果の保存 (途中から再開できる)。

重い部品 (funasr / torch / faster-whisper) は、そのモデルを使うときに初めて import します (lazy import)。
入っていなければ、入れ方を日本語で案内します。テストでは、同じ形の『偽のモデル』(Recognizer) を使うので、モデルのダウンロードは要りません。

公式ページで確認した API (確認日 2026-10-19):
- SenseVoiceSmall (コード MIT、重みは FunASR 独自ライセンス)  https://github.com/FunAudioLLM/SenseVoice
    model = AutoModel(model="iic/SenseVoiceSmall", device="cpu", ncpu=スレッド数, disable_update=True)
    res = model.generate(input=wavのパス, cache={}, language="auto" か "ja", use_itn=True, batch_size_s=60)
    res[0]["text"] = '<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは' (言語・感情・音の種類・ITN のタグが先頭に付く)
  ban_emo_unk=True を generate に足すと、EMO_UNKNOWN を出さずに、ほかの感情タグから選ばせる (funasr 1.4.16 の SenseVoiceSmall.inference
  のソースで確認。公式 README での説明は未確認 (要検証))
  pyVideoTrans の SenseVoiceSmall は、あらかじめ切った音声を 1 本ずつ渡し (VAD なし)、言語は指定していない。
  この道具も同じく、1 本の wav を VAD なしで渡す。言語は --sv-language で選べる (既定 auto = pyVideoTrans と同じ)。
- faster-whisper (MIT)  https://github.com/SYSTRAN/faster-whisper  → asrcheck.runners.WhisperEngine をそのまま使う
    ('large-v3-turbo' は mobiuslabsgmbh/faster-whisper-large-v3-turbo。CPU は int8、beam_size=5、VAD なし)
"""
from __future__ import annotations

import gc
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


def trim_memory() -> None:
    """使い終わったメモリを OS に返す (Linux のみ。ほかでは何もしない)。SenseVoice は処理を続けるとメモリが 1.5 → 3 GB に増えるため。"""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


@dataclass
class Recognition:
    text: str
    tag: str = ""          # SenseVoice の感情タグ名 ('HAPPY' など。タグなしは '')


class Recognizer(Protocol):
    """1本の音声から (文字, 感情タグ) を返すモデル。テストでは偽物を使う。"""

    key: str               # 短い名前 (キャッシュのファイル名にも使う): sensevoice / sensevoice_nounk / whisper
    label: str             # Excel に出す名前
    has_emotion: bool      # 感情タグを出すモデルか
    report_cer: bool       # CER の表に載せるか (同じ文章を出す別設定は False にして重複を避ける)

    def recognize(self, path: Path) -> Recognition: ...

    def info(self) -> list[tuple[str, Any]]: ...      # 『実行情報』シートに載せる項目

    def signature(self) -> dict[str, Any]: ...        # 設定が変わったら結果を使い回さないための印


@dataclass
class ClipResult:
    text: str
    tag: str = ""
    sec: float = 0.0       # この1本の処理にかかった秒 (モデルの読み込みは含まない)
    error: str = ""
    threads: int = 0       # 処理に使った CPU スレッド数 (0 = 記録なし)


@dataclass
class ModelRun:
    key: str
    label: str
    has_emotion: bool
    report_cer: bool = True
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
    has_emotion = True

    def __init__(self, model: str = SV_DEFAULT_MODEL, language: str = "auto", device: str = "cpu", threads: int = 2,
                 ban_unk: bool = False) -> None:
        self.model_id, self.language, self.device, self.threads = model, language, device, max(1, threads)
        self.model: Any = None      # prepare() で読み込む (処理する音声が無ければ読み込まない)
        self.ban_unk = ban_unk
        # ban_unk=True は、EMO_UNKNOWN を出させずに、ほかの感情タグから選ばせる設定 (文章は変わらない。感情タグだけが変わる)
        self.key = "sensevoice_nounk" if ban_unk else "sensevoice"
        self.label = "SenseVoiceSmall（EMO_UNKNOWN 禁止）" if ban_unk else "SenseVoiceSmall"
        self.report_cer = not ban_unk

    def prepare(self) -> None:
        """モデルを読み込む (2回目以降は何もしない)。funasr / torch が無ければ、入れ方を案内する。"""
        if self.model is not None:
            return
        try:
            import torch
            from funasr import AutoModel
        except ImportError:
            raise InputError(f"funasr / torch が入っていません。次のように実行すると一時的に入れて動かせます:\n  {INSTALL_HINTS['funasr']}") from None
        # ncpu を必ず渡す: funasr は ncpu (既定 4) で torch のスレッド数を上書きする。ほかの重い処理と同時に動かすと、
        # 4 スレッドの待ち合わせで 1 本 30 秒以上かかった (ncpu=1 だと約 1 秒。2026-10-19 に実測)
        self.model = AutoModel(model=self.model_id, device=self.device, ncpu=self.threads, disable_update=True,
                               disable_pbar=True, disable_log=True)
        torch.set_num_threads(self.threads)
        trim_memory()

    def housekeeping(self) -> None:
        trim_memory()

    def recognize(self, path: Path) -> Recognition:
        self.prepare()
        extra = {"ban_emo_unk": True} if self.ban_unk else {}
        res = self.model.generate(input=str(path), cache={}, language=self.language, use_itn=True, batch_size_s=60,
                                  disable_pbar=True, **extra)
        text, tag = split_sensevoice_text(res[0]["text"] if res else "")
        return Recognition(text, tag)

    def signature(self) -> dict[str, Any]:
        sig: dict[str, Any] = {"model": self.model_id, "language": self.language, "use_itn": True}
        if self.ban_unk:      # 既定の設定では書かない (ban_unk を足す前に保存した途中結果をそのまま使えるように)
            sig["ban_emo_unk"] = True
        return sig

    def info(self) -> list[tuple[str, Any]]:
        return [
            ("モデル", f"SenseVoiceSmall ({self.model_id})"),
            ("ライブラリ", f"funasr {_version('funasr')} / torch {_version('torch')}"),
            ("設定", f"language={self.language} / use_itn=True / VAD なし / {self.device} / 1本ずつ処理"
                    + (" / ban_emo_unk=True（EMO_UNKNOWN を禁止。pyVideoTrans の今の使い方とは違う設定）" if self.ban_unk else " / ban_emo_unk なし（pyVideoTrans の今の使い方と同じ）")),
            ("ライセンス", ("コードは MIT。重み (モデル本体) は FunASR のモデルライセンス: 商用利用可、ただし出典の表記が必要 "
                            "(https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)。PLAN の方針ではこの条件で使用可")),
        ]


# ---------------------------------------------------------------------------
# faster-whisper
# ---------------------------------------------------------------------------
class WhisperRecognizer:
    key = "whisper"
    has_emotion = False
    report_cer = True

    def __init__(self, model: str = WHISPER_DEFAULT_MODEL, lang: str = "ja", device: str = "cpu", threads: int = 2) -> None:
        self.model_id, self.lang, self.device, self.threads = model, lang, device, threads
        self.label = f"Whisper {model}"
        self.engine: Any = None     # prepare() で読み込む (処理する音声が無ければ読み込まない)
        self._load: Any = None

    def prepare(self) -> None:
        if self.engine is not None:
            return
        from asrcheck.runners import WhisperEngine, load_audio_16k

        self._load = load_audio_16k
        # スレッド数は環境変数 OMP_NUM_THREADS で効く (cli が設定する)。asrcheck と同じ設定 (beam_size=5・VAD なし・CPU は int8)
        self.engine = WhisperEngine(self.model_id, self.device, self.lang, [])

    def recognize(self, path: Path) -> Recognition:
        self.prepare()
        return Recognition(self.engine.transcribe_one(self._load(path)))

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id, "language": self.lang, "beam_size": 5}

    def info(self) -> list[tuple[str, Any]]:
        return [
            ("モデル", f"faster-whisper {self.model_id}"),
            ("ライブラリ", f"faster-whisper {_version('faster-whisper')} / ctranslate2 {_version('ctranslate2')}"),
            ("設定", f"language={self.lang} / beam_size=5 / VAD なし / condition_on_previous_text=False / {self.device} int8 / 1本ずつ処理"),
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
            out[d["file"]] = ClipResult(d.get("text", ""), d.get("tag", ""), float(d.get("sec", 0.0)), threads=int(d.get("threads", 0)))
    return out


def run_model(
    rec: Recognizer, clips: list[Clip], cache_path: Path | None, load_sec: float = 0.0,
    progress: Callable[[str], None] = say, every: int = 20,
) -> ModelRun:
    """clips を 1 本ずつ rec に通す。結果は 1 本ごとに cache_path (JSON Lines) に書き足すので、止まっても続きから再開できる。"""
    run = ModelRun(rec.key, rec.label, rec.has_emotion, getattr(rec, "report_cer", True), info=rec.info(), load_sec=load_sec)
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
    if todo and hasattr(rec, "prepare"):
        progress(f"  モデルを準備しています: {rec.label} ...")
        t_load = time.perf_counter()
        rec.prepare()      # 読み込みの失敗 (ライブラリが無いなど) は、1本ずつの失敗にせず、ここで止める
        run.load_sec = load_sec + time.perf_counter() - t_load
    fh = None
    if cache_path and todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(cache_path, "a", encoding="utf-8")
        if cache_path.stat().st_size and not cache_path.read_bytes().endswith(b"\n"):
            fh.write("\n")      # 前回、書きかけで止まっていたら、次の行が壊れた行にくっつかないように改行を足す
    t_start = time.perf_counter()
    try:
        for i, c in enumerate(todo, start=1):
            t0 = time.perf_counter()
            try:
                r = rec.recognize(c.path)
                res = ClipResult(r.text, r.tag, time.perf_counter() - t0, threads=int(getattr(rec, "threads", 0)))
            except Exception as e:  # noqa: BLE001  (1本の失敗で全体を止めない。失敗は表に残す)
                res = ClipResult("", "", time.perf_counter() - t0, error=f"{type(e).__name__}: {e}"[:200])
                run.n_errors += 1
            run.results[c.file] = res
            if fh:
                fh.write(json.dumps({"file": c.file, "sig": sig, "text": res.text, "tag": res.tag, "sec": round(res.sec, 3),
                                     "error": res.error, "threads": res.threads}, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 50 == 0 and hasattr(rec, "housekeeping"):
                rec.housekeeping()
            if i % every == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                progress(f"  {rec.label}: {i}/{len(todo)} 件 ({el:.0f} 秒経過 / 残り約 {el / i * (len(todo) - i):.0f} 秒)")
    finally:
        if fh:
            fh.close()
    return run
