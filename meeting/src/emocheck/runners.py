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
- emotion2vec+ (コード MIT、重みは FunASR のモデルライセンス)  https://github.com/ddlBoJack/emotion2vec / https://huggingface.co/emotion2vec/emotion2vec_plus_large
    model = AutoModel(model="iic/emotion2vec_plus_large", device="cpu", ncpu=スレッド数, disable_update=True)
    res = model.generate(input=wavのパス, granularity="utterance", extract_embedding=False)
    res[0] = {"key": ..., "labels": ["生气/angry", "厌恶/disgusted", ..., "<unk>"], "scores": [0.01, ...]}  (9 分類の softmax。合計 1)
  funasr 1.4.16 の emotion2vec/model.py のソースと、モデルの tokens.txt で確認。HF の id (emotion2vec/emotion2vec_plus_large) は hub="hf" で読める。
  1 本の音声を、そのまま (VAD・分割なしで) 1 回で処理する。
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from asrcheck.ioformats import InputError

from .dataset import Clip
from .labels import canonical_e2v, split_sensevoice_text

SV_DEFAULT_MODEL = "iic/SenseVoiceSmall"
E2V_LARGE_MODEL = "iic/emotion2vec_plus_large"
E2V_BASE_MODEL = "iic/emotion2vec_plus_base"
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
    tag: str = ""          # 感情タグ名 ('HAPPY' など。タグなしは ''。emotion2vec+ は 9 分類のうち最大のもの。'OTHER' 'UNKNOWN' もありうる)
    scores: dict[str, float] | None = None    # emotion2vec+ の 9 分類のスコア ({'angry': 0.01, ...})。SenseVoice は None


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
    scores: dict[str, float] | None = None    # emotion2vec+ の 9 分類のスコア (無いモデルは None)


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
    meta: dict[str, Any] = field(default_factory=dict)    # 最後にモデルを読み込んだ実行の記録 (読み込み秒・メモリ・日時。途中結果の隣の .meta.json)

    @property
    def has_scores(self) -> bool:
        """分類ごとのスコアを持つモデルか (emotion2vec+)。『4つから選ぶ』の見方ができる。"""
        return any(r.scores for r in self.results.values())


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
            ("ライセンス", ("コードは MIT。重み (モデル本体) は FunASR のモデルライセンス (https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE): "
                            "使用・複製・改変・共有でき、出典と作者・モデル名の表示が必要。ただし条文に『参考と学習の目的のみで提供』とも書いてあり、"
                            "商用利用の可否の法務確認は未了 (PLAN Q5。要確認)。PLAN の決定 D10 で、中国系のモデルは使ってよいとしている")),
        ]


# ---------------------------------------------------------------------------
# emotion2vec+ (FunASR)
# ---------------------------------------------------------------------------
class Emotion2VecRecognizer:
    """emotion2vec+ (large / base)。9 分類 (angry disgusted fearful happy neutral other sad surprised unknown) のスコアを返す。

    SenseVoice と違い、文字起こしは出さない (text は空)。タグは、スコアが最大の分類名 (大文字)。スコアは全部 scores に残すので、
    『4つ (平常・喜び・怒り・悲しみ) の中から最大のものを選ぶ』見方も後から計算できる。
    """
    has_emotion = True
    report_cer = False
    has_scores = True

    def __init__(self, model: str = E2V_LARGE_MODEL, device: str = "cpu", threads: int = 2, hub: str = "auto") -> None:
        self.model_id, self.device, self.threads = model, device, max(1, threads)
        # 'iic/...' は ModelScope、'emotion2vec/...' は Hugging Face のモデル名。hub='auto' なら名前で選ぶ
        self.hub = ("hf" if model.startswith("emotion2vec/") else "ms") if hub == "auto" else hub
        tail = model.rsplit("/", 1)[-1].lower()
        self.key = "".join(ch if ch.isalnum() else "_" for ch in tail)       # 例: emotion2vec_plus_large (途中結果のファイル名にもなる)
        self.label = "emotion2vec+ " + ("large" if "large" in tail else "base" if "base" in tail else tail)
        self.model: Any = None

    def prepare(self) -> None:
        """モデルを読み込む (2回目以降は何もしない)。funasr / torch が無ければ、入れ方を案内する。"""
        if self.model is not None:
            return
        try:
            import torch
            from funasr import AutoModel
        except ImportError:
            raise InputError(f"funasr / torch が入っていません。次のように実行すると一時的に入れて動かせます:\n  {INSTALL_HINTS['funasr']}") from None
        # SenseVoice と同じく ncpu を必ず渡す (渡さないと funasr が torch のスレッド数を 4 に上書きする)
        logging.disable(logging.INFO)       # 読み込み中に重みの名前が何百行も出るのを止める (警告とエラーは出す)
        try:
            self.model = AutoModel(model=self.model_id, hub=self.hub, device=self.device, ncpu=self.threads, disable_update=True,
                                   disable_pbar=True, disable_log=True)
        finally:
            logging.disable(logging.NOTSET)
        torch.set_num_threads(self.threads)
        trim_memory()

    def housekeeping(self) -> None:
        trim_memory()

    def recognize(self, path: Path) -> Recognition:
        self.prepare()
        # extract_embedding=False: 特徴ベクトルは要らない (返さない・保存しない)
        res = self.model.generate(input=str(path), granularity="utterance", extract_embedding=False, disable_pbar=True)
        if not res:
            raise RuntimeError("emotion2vec+ が結果を返しませんでした")
        labels, vals = res[0]["labels"], res[0]["scores"]
        if not labels or len(labels) != len(vals):
            raise RuntimeError(f"emotion2vec+ の出力が読めません（分類 {len(labels or [])} 個 / スコア {len(vals or [])} 個）")
        scores: dict[str, float] = {}
        for lab, v in zip(labels, vals, strict=True):
            scores[canonical_e2v(lab)] = round(float(v), 6)
        top = max(scores, key=lambda k: scores[k])
        return Recognition("", top.upper(), scores)

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id, "granularity": "utterance", "v": 1}

    def info(self) -> list[tuple[str, Any]]:
        return [
            ("モデル", f"{self.label} ({self.model_id}。取得元: {'ModelScope' if self.hub == 'ms' else 'Hugging Face' if self.hub == 'hf' else self.hub})"),
            ("ライブラリ", f"funasr {_version('funasr')} / torch {_version('torch')}"),
            ("設定", f"granularity=utterance / extract_embedding=False / VAD・分割なし / {self.device} / 1本ずつ処理 / 9 分類のスコアを保存"),
            ("ライセンス", ("コードは MIT。重みは FunASR のモデルライセンス（Hugging Face のカードでは license: other / model-license。SenseVoiceSmall と同じ系統）。"
                            "条文（https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE ）は、使用・複製・改変・共有を認め、出典と作者・モデル名の表示を求める一方、"
                            "『参考と学習の目的のみで提供』とも書いてある。商用利用の可否は法務確認が未了（PLAN Q5。要確認）")),
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
def _rss_mb() -> tuple[float | None, float | None]:
    """(今のメモリ使用量, これまでの最大) を MB で。Linux / macOS だけ。分からなければ None。"""
    peak = cur = None
    try:
        import resource

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = v / (1024 * 1024) if sys.platform == "darwin" else v / 1024      # Linux は KB、macOS は バイト
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/self/statm") as f:
            import os

            cur = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return cur, peak


def meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem + ".meta.json")


def _load_meta(cache_path: Path | None) -> dict[str, Any]:
    if cache_path is None or not meta_path(cache_path).is_file():
        return {}
    try:
        d = json.loads(meta_path(cache_path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


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
            sc = d.get("scores")
            out[d["file"]] = ClipResult(d.get("text", ""), d.get("tag", ""), float(d.get("sec", 0.0)), threads=int(d.get("threads", 0)),
                                        scores={str(k): float(v) for k, v in sc.items()} if isinstance(sc, dict) and sc else None)
    return out


def run_model(
    rec: Recognizer, clips: list[Clip], cache_path: Path | None, load_sec: float = 0.0,
    progress: Callable[[str], None] = say, every: int = 20,
) -> ModelRun:
    """clips を 1 本ずつ rec に通す。結果は 1 本ごとに cache_path (JSON Lines) に書き足すので、止まっても続きから再開できる。"""
    run = ModelRun(rec.key, rec.label, rec.has_emotion, getattr(rec, "report_cer", True), info=rec.info(), load_sec=load_sec,
                   meta=_load_meta(cache_path))
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
    loaded_here = False
    after_load: tuple[float | None, float | None] = (None, None)
    if todo and hasattr(rec, "prepare"):
        progress(f"  モデルを準備しています: {rec.label} ...")
        t_load = time.perf_counter()
        rec.prepare()      # 読み込みの失敗 (ライブラリが無いなど) は、1本ずつの失敗にせず、ここで止める
        run.load_sec = load_sec + time.perf_counter() - t_load
        after_load = _rss_mb()
        loaded_here = True
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
                res = ClipResult(r.text, r.tag, time.perf_counter() - t0, threads=int(getattr(rec, "threads", 0)), scores=r.scores)
            except Exception as e:  # noqa: BLE001  (1本の失敗で全体を止めない。失敗は表に残す)
                res = ClipResult("", "", time.perf_counter() - t0, error=f"{type(e).__name__}: {e}"[:200])
                run.n_errors += 1
            run.results[c.file] = res
            if fh:
                rec_line = {"file": c.file, "sig": sig, "text": res.text, "tag": res.tag, "sec": round(res.sec, 3),
                            "error": res.error, "threads": res.threads}
                if res.scores:        # スコアを持つモデルだけ書く (SenseVoice の途中結果の形は今までと同じ)
                    rec_line["scores"] = res.scores
                fh.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 50 == 0 and hasattr(rec, "housekeeping"):
                rec.housekeeping()
            if i % every == 0 or i == len(todo):
                el = time.perf_counter() - t_start
                progress(f"  {rec.label}: {i}/{len(todo)} 件 ({el:.0f} 秒経過 / 残り約 {el / i * (len(todo) - i):.0f} 秒)")
    finally:
        if fh:
            fh.close()
    if loaded_here and cache_path:       # モデルを読み込んだ実行だけ、読み込み秒とメモリを記録する (あとで Excel の『実行情報』に載せる)
        cur, peak = _rss_mb()
        meta = {"model": rec.label, "load_sec": round(run.load_sec, 1), "rss_after_load_mb": _r(after_load[0]), "peak_rss_mb": _r(peak),
                "rss_end_mb": _r(cur), "n_processed": len(todo), "threads": int(getattr(rec, "threads", 0)),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            meta_path(cache_path).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            run.meta = meta
        except OSError:
            pass
    return run


def _r(v: float | None) -> float | None:
    return None if v is None else round(v)
