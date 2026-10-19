"""話者埋め込みを作る部分（モデルを動かす。重い処理はここだけ）。

- 埋め込み器: ecapa（voicecheck の ECAPA をそのまま使う）、titanet / eres2net / campp（sherpa-onnx の onnx。pyVideoTrans が持っているモデルを再利用）
- 作るもの: ① 登録候補の発話（会議で使っていない JVNV の発話）の埋め込み ② 疑似会議の字幕の区間（正解の区間 / silero）の埋め込み
- 結果は npz に保存（runs/enroll/emb_<名前>.npz）。途中で止まっても続きから再開できる。
- 登録発話は、疑似会議を作るときと同じ処理（前後の無音を切る・音量をそろえる・16kHz）を通す。会議と登録で録音条件が同じ
  （＝実際より有利な）設定であることに注意。
"""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from . import pvt_bridge
from .build import SR_HI, SR_LO, BuildParams, read_mono, rms_db, to_16k, trim_range
from . import enroll_assign as ea
from .enroll_assign import (POOL_SEC_PER_EMOTION, POOL_SEED, EmbeddingStore, build_pool, enr_key, seg_key, unit,
                            used_in_meeting)
from .pipeline import Layout

MIN_SEC = 0.5          # これより短い波形は、0.5 秒になるまで繰り返してから埋め込む
SEG_KINDS_EMBED = ("oracle", "silero")


@dataclass(frozen=True)
class EmbedderInfo:
    name: str
    label: str
    kind: str            # "ecapa" か "sherpa"
    file: str | None     # sherpa の onnx ファイル名
    license: str
    origin: str


EMBEDDERS: dict[str, EmbedderInfo] = {
    "ecapa": EmbedderInfo("ecapa", "ECAPA-TDNN（speechbrain/spkrec-ecapa-voxceleb）", "ecapa", None,
                          "Apache-2.0（Hugging Face のモデルカードの license 欄で確認）。学習データ VoxCeleb1+2 の利用条件は (要検証)",
                          "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb"),
    "titanet": EmbedderInfo("titanet", "NeMo TitaNet small（英語用・sherpa-onnx）", "sherpa", "nemo_en_titanet_small.onnx",
                            "NGC のモデルカードに『NeMo ツールキットのライセンスに従う』とある（ツールキットは Apache-2.0 とされる (要検証)）。sherpa-onnx 用に変換された onnx の再配布条件も (要検証)。pyVideoTrans の built が language=ja で使う埋め込み",
                            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models"),
    "eres2net": EmbedderInfo("eres2net", "3D-Speaker ERes2Net large（中国語用・sherpa-onnx）", "sherpa",
                             "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx",
                             "3D-Speaker のコードは Apache-2.0（リポジトリで確認）。事前学習済み重みのライセンスはリポジトリに記載なし (要検証)。pyVideoTrans の built が language=zh で使う埋め込み",
                             "https://github.com/modelscope/3D-Speaker"),
    "campp": EmbedderInfo("campp", "3D-Speaker CAM++（中国語用・sherpa-onnx）", "sherpa",
                          "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
                          "3D-Speaker のコードは Apache-2.0（リポジトリで確認）。事前学習済み重みのライセンスはリポジトリに記載なし (要検証)",
                          "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models（onnx。元は https://github.com/modelscope/3D-Speaker）"),
}


def onnx_search_dirs() -> list[Path]:
    root = pvt_bridge.REPO_ROOT
    return [root / "models" / "onnx", root / "meeting" / "data" / "models"]


def find_onnx(file: str) -> Path | None:
    for d in onnx_search_dirs():
        if (d / file).is_file():
            return d / file
    return None


def trim_edges(y: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """前後の無音を落とす（voicecheck の埋め込み器と同じ設定。短くなりすぎたら元のまま）。"""
    import librosa

    t, _ = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=160)
    return t if len(t) >= 160 else y


def pad_min(y: np.ndarray, min_sec: float = MIN_SEC) -> np.ndarray:
    n = int(min_sec * SR_LO)
    if len(y) >= n:
        return y
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.tile(y, int(np.ceil(n / len(y))))[:n]


class SpeakerEmbedder:
    name = "base"

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def embed(self, y16k: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def versions(self) -> dict:
        return {}


class SherpaEmbedder(SpeakerEmbedder):
    """sherpa-onnx の SpeakerEmbeddingExtractor（pyVideoTrans の built と同じ部品・同じ設定。CPU 1 スレッド）。"""

    def __init__(self, name: str, model: Path, threads: int = 1):
        self.name, self.model, self.threads = name, Path(model), threads
        self._ex = None

    def load(self) -> None:
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.model), num_threads=self.threads, provider="cpu")
        if not cfg.validate():
            raise RuntimeError(f"sherpa-onnx の設定が正しくありません: {self.model}")
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, y16k: np.ndarray) -> np.ndarray:
        if self._ex is None:
            self.load()
        y = pad_min(trim_edges(np.asarray(y16k, dtype=np.float32)))
        s = self._ex.create_stream()
        s.accept_waveform(sample_rate=SR_LO, waveform=y)
        s.input_finished()
        return unit(np.asarray(self._ex.compute(s), dtype=np.float64))

    def versions(self) -> dict:
        return _versions(["sherpa-onnx", "onnxruntime"])


class EcapaEmbedder(SpeakerEmbedder):
    """voicecheck の ECAPA 埋め込み器（~/.cache/voicecheck/ecapa のモデルを再利用）。speechbrain と torch が要る（meeting の uv 環境）。"""

    name = "ecapa"

    def __init__(self, threads: int = 1):
        self.threads = threads
        self._e = None

    def load(self) -> None:
        import torch

        torch.set_num_threads(self.threads)
        cache = Path(os.environ.get("VOICECHECK_CACHE", Path.home() / ".cache" / "voicecheck")) / "ecapa"
        if (cache / "embedding_model.ckpt").is_file():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")      # 取得済みなら通信しない
        from voicecheck.embedders import EcapaEmbedder as _Ecapa

        self._e = _Ecapa()
        self._e.load()

    def embed(self, y16k: np.ndarray) -> np.ndarray:
        if self._e is None:
            self.load()
        return unit(np.asarray(self._e.embed(pad_min(np.asarray(y16k, dtype=np.float32))), dtype=np.float64))

    def versions(self) -> dict:
        return _versions(["speechbrain", "torch"])


def _versions(pkgs: list[str]) -> dict:
    out = {}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def make_embedder(name: str, threads: int = 1) -> SpeakerEmbedder:
    if name not in EMBEDDERS:
        raise ValueError(f"埋め込み器は {', '.join(EMBEDDERS)} のどれか: {name}")
    info = EMBEDDERS[name]
    if info.kind == "ecapa":
        return EcapaEmbedder(threads)
    path = find_onnx(info.file or "")
    if path is None:
        raise FileNotFoundError(f"{info.file} が見つかりません（探した場所: {', '.join(map(str, onnx_search_dirs()))}）")
    return SherpaEmbedder(name, path, threads)


# ---------------------------------------------------------------------------
def prepare_clip_16k(path: Path, params: BuildParams | None = None) -> np.ndarray:
    """登録発話を、疑似会議に入れるときと同じ処理にする（前後の無音を切る・RMS を -24 dBFS にそろえる・端を 10ms でなめらかに・16kHz）。"""
    p = params or BuildParams()
    y = read_mono(Path(path))
    i0, i1 = trim_range(y, SR_HI, p.trim_db, p.pad_s)
    y = y[i0:i1].copy()
    y *= np.float32(10 ** ((p.target_rms_db - rms_db(y)) / 20))
    np.clip(y, -1.0, 1.0, out=y)
    fade = int(0.01 * SR_HI)
    if len(y) > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return to_16k(y)


def seg_items(lay: Layout, v: str, seg: str) -> list[dict]:
    """字幕の区間（文字起こしが空でないもの。pipeline.stage_assign と同じ条件）。"""
    d = json.loads((lay.runs(v) / f"asr_{seg}.json").read_text(encoding="utf-8"))
    return [it for it in d["items"] if it["text"].strip()]


def store_path(lay: Layout, name: str) -> Path:
    return lay.root / "runs" / "enroll" / f"emb_{name}.npz"


def meta_path(lay: Layout, name: str) -> Path:
    return lay.root / "runs" / "enroll" / f"emb_{name}.meta.json"


def _acc(meta: dict, group: str, key: str, n: int, audio_sec: float, elapsed: float) -> None:
    g = meta.setdefault(group, {}).setdefault(key, {"n": 0, "audio_sec": 0.0, "elapsed_sec": 0.0})
    g["n"] += n
    g["audio_sec"] += audio_sec
    g["elapsed_sec"] += elapsed


def run_embedding(name: str, lay: Layout, jvnv_dir: Path | str, say=print, variants=("A", "B"), segs=SEG_KINDS_EMBED,
                  pool_sec: float = POOL_SEC_PER_EMOTION, threads: int = 1, save_every: int = 150, embedder: SpeakerEmbedder | None = None,
                  windows: bool = False) -> dict:
    """登録候補と字幕の区間の埋め込みを作って保存する（あるものは飛ばす）。戻り値: メタ情報。
    windows=True なら、silero の区間の中を窓（1.5秒、0.5秒ずつ）に分けた埋め込みも作る（区間の切り直しの見積もり用）。"""
    import soundfile as sf

    from .jvnv import load_index

    manifest = json.loads((lay.root / "manifest.json").read_text(encoding="utf-8"))
    used = used_in_meeting(manifest)
    pool = build_pool(load_index(jvnv_dir), used, pool_sec, POOL_SEED)
    emb = embedder or make_embedder(name, threads)
    emb.load()
    sp, mp = store_path(lay, name), meta_path(lay, name)
    store = EmbeddingStore.load(sp)
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    info = EMBEDDERS[name]
    meta.update({"embedder": name, "label": info.label, "license": info.license, "origin": info.origin, "threads": threads,
                 "model_file": str(getattr(emb, "model", "") or "~/.cache/voicecheck/ecapa"), "versions": emb.versions(),
                 "python": platform.python_version(), "pool_sec_per_emotion": pool_sec, "pool_seed": POOL_SEED,
                 "pool_clips": len(pool)})
    if len(store):
        meta["dim"] = int(len(next(iter(store.vec.values()))))
    if isinstance(emb, SherpaEmbedder) and emb.model.is_file():
        import hashlib

        h = hashlib.sha256()
        with open(emb.model, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        meta["model_sha256"] = h.hexdigest()
        meta["model_bytes"] = emb.model.stat().st_size

    # ① 登録候補
    todo = [c for c in pool if enr_key(c.path.name) not in store]
    say(f"[{name}] 登録候補 {len(pool)} 本のうち、未処理 {len(todo)} 本")
    n = 0
    t_emb = a_sec = 0.0
    for i, c in enumerate(todo, start=1):
        y = prepare_clip_16k(c.path)
        t0 = time.perf_counter()
        v = emb.embed(y)
        t_emb += time.perf_counter() - t0
        a_sec += len(y) / SR_LO
        store.put(enr_key(c.path.name), v, len(y) / SR_LO)
        n += 1
        meta["dim"] = int(len(v))
        if i % save_every == 0:
            store.save(sp)
            say(f"    登録候補 {i}/{len(todo)}（{t_emb:.0f} 秒）")
    if n:
        _acc(meta, "enroll", "all", n, a_sec, t_emb)
    store.save(sp)
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ② 字幕の区間
    for v in variants:
        y16, sr = sf.read(str(lay.wav(v)), dtype="float32", always_2d=False)
        if sr != SR_LO:
            raise ValueError(f"{lay.wav(v)} は {SR_LO}Hz ではありません（{sr}Hz）")
        if y16.ndim > 1:
            y16 = y16[:, 0]
        for seg in segs:
            items = seg_items(lay, v, seg)
            todo_s = [it for it in items if seg_key(v, seg, it["start_ms"], it["end_ms"]) not in store]
            say(f"[{name}] 録音{v} 区間={seg}: {len(items)} 区間のうち未処理 {len(todo_s)}")
            t_emb = a_sec = 0.0
            for i, it in enumerate(todo_s, start=1):
                y = y16[int(it["start_ms"] * SR_LO / 1000):int(it["end_ms"] * SR_LO / 1000)]
                t0 = time.perf_counter()
                vec = emb.embed(y)
                t_emb += time.perf_counter() - t0
                a_sec += len(y) / SR_LO
                store.put(seg_key(v, seg, it["start_ms"], it["end_ms"]), vec, len(y) / SR_LO)
                meta["dim"] = int(len(vec))
                if i % save_every == 0:
                    store.save(sp)
                    say(f"    {i}/{len(todo_s)}（{t_emb:.0f} 秒）")
            if todo_s:
                _acc(meta, "segments", f"{v}:{seg}", len(todo_s), a_sec, t_emb)
            store.save(sp)
            mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            if windows and seg == "silero":
                spans = sorted({sp for it in items for sp in ea.window_spans(it["start_ms"], it["end_ms"])})
                todo_w = [sp for sp in spans if ea.win_key(v, seg, *sp) not in store]
                say(f"[{name}] 録音{v} 窓（区間の中を {ea.WIN_MS}ms ずつ）: {len(spans)} 個のうち未処理 {len(todo_w)}")
                t_emb = a_sec = 0.0
                for i, (a, b) in enumerate(todo_w, start=1):
                    y = y16[int(a * SR_LO / 1000):int(b * SR_LO / 1000)]
                    t0 = time.perf_counter()
                    vec = emb.embed(y)
                    t_emb += time.perf_counter() - t0
                    a_sec += len(y) / SR_LO
                    store.put(ea.win_key(v, seg, a, b), vec, len(y) / SR_LO)
                    if i % (save_every * 4) == 0:
                        store.save(sp)
                        say(f"    窓 {i}/{len(todo_w)}（{t_emb:.0f} 秒）")
                if todo_w:
                    _acc(meta, "windows", f"{v}:{seg}", len(todo_w), a_sec, t_emb)
                store.save(sp)
                mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        del y16
    say(f"[{name}] 完了: 埋め込み {len(store)} 個 → {sp}")
    return meta
