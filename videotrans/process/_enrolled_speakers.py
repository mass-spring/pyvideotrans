# -*- coding: utf-8 -*-
"""
speaker_type = 'enrolled'（登録した声と照合）

事前に登録した各人の声（voice_profile_dir の <名前>/profile.json と手本の wav）の「平均の声の特徴」と、
字幕の1行ごとの声の特徴（話者埋め込み）を比べて、いちばん近い人の名前をその行の話者にする。
内蔵の built（教師なしクラスタリング）より日本語の疑似会議で大きく良かった（話者正解率 約50% → 約94%。meeting/docs/research.md 22・24章）。

- 話者ラベルは **登録フォルダの名前そのもの**（例：「Aさん」）。videotrans/util/voice_profile.py の
  VoiceProfiles.profile_name() は「speaker_map に無いラベルは、ラベルと同じ名前の登録フォルダを使う」ので、
  speaker_map.json を書かなくても、そのまま声の複製の手本選びに使われる。
- 埋め込みは sherpa-onnx（built と同じ部品）で計算する。モデルは models/onnx/ にある onnx を再利用する
  （既定 titanet = nemo_en_titanet_small.onnx、eres2net = 3dspeaker の onnx）。ダウンロードはしない。
- 出力 speak_file は built_speakers と同じ形式（字幕の行順に話者名を並べた JSON の配列。例：["Aさん", "Bさん", ...]）。
- 0.5 秒未満の行、音声が取れない行、埋め込めなかった行は、**直前の行の話者を引き継ぐ**
  （先頭の行は、最初に埋め込めた行の話者）。診断ファイルには「inherited」と記録する。
- min_sim を指定すると、最高点がそれ未満の行は 'unknown' になる（既定は使わない。しきい値は決めにくいため）。
- 診断ファイル speaker_diag.json / speaker_diag.csv を speak_file の隣に書く（最高点・2位の点・差・要確認の印）。
  1.5 秒の窓で行の中を判定し、途中で話者が替わったように見える行には印を付けるだけで、字幕は分割しない
  （分割すると文字起こしのやり直しが必要になるため）。
"""
import csv
import hashlib
import json
import math
import os
import shutil
import time
import traceback
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from videotrans.configure.config import ROOT_DIR, logger
from videotrans.util.voice_profile import ProfileRef, VoiceProfiles

SAMPLE_RATE = 16000
DEFAULT_MODEL = 'titanet'
# 設定値 -> models/onnx/ の中のファイル名（built_speakers が使うものと同じ）
MODEL_FILES = {
    'titanet': 'nemo_en_titanet_small.onnx',
    'eres2net': '3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx',
}
UNKNOWN = 'unknown'

MIN_SEG_SEC = 0.5  # これより短い行は埋め込まず、直前の行の話者を引き継ぐ
TRIM_DB = 40.0  # 行の前後の無音を落とす（最大の音から -40dB 以下）
PAD_SEC = 0.5  # 埋め込む波形がこれより短ければ、繰り返して 0.5 秒にする
WIN_MS, HOP_MS, MIN_PIECE_MS = 1500, 500, 1000  # 行の中の判定の窓（1.5 秒、0.5 秒ずつ）、これより短い部分は隣に吸収
LOW_MARGIN = 0.10  # 1位と2位の点の差がこれ未満の行を「要確認」にする
EMB_CACHE_VERSION = 1
DIAG_JSON = 'speaker_diag.json'
DIAG_CSV = 'speaker_diag.csv'

FLAG_JA = {
    'inherited': '短い／埋め込めないため、前の行の話者を引き継いだ',
    'low_margin': '1位と2位の差が小さい（別の人かもしれない）',
    'mid_change': '話者が途中で替わった可能性',
    'unknown': '最高点がしきい値（min_sim）未満',
}


# ---------------------------------------------------------------------------
# 設定値の読み取りと確認（メッセージは利用者向けの日本語）
# ---------------------------------------------------------------------------
def parse_min_sim(value) -> Optional[float]:
    """'' / None / 'none' -> None（使わない）。それ以外は -1〜1 の小数。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() in ('none', 'null', 'off'):
        return None
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f'enrolled_min_sim は 0.35 のような小数、または空欄にしてください: {value!r}') from None
    if not math.isfinite(v) or not -1.0 <= v <= 1.0:
        raise ValueError(f'enrolled_min_sim は -1〜1 の範囲で指定してください: {value!r}')
    return v


def model_path(model: str) -> Tuple[str, Path]:
    """設定値（'titanet' など）-> (正規化した名前, onnx のパス)。無ければ FileNotFoundError。"""
    key = str(model or DEFAULT_MODEL).strip().lower() or DEFAULT_MODEL
    if key not in MODEL_FILES:
        raise ValueError(f'enrolled_speaker_model は {" / ".join(MODEL_FILES)} のどれかにしてください: {model!r}')
    path = Path(ROOT_DIR) / 'models' / 'onnx' / MODEL_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(
            f'話者の特徴を計算するモデルが見つかりません: {path}\n'
            f'speaker_type を built にして話者分離を一度動かすと、models/onnx/ に取得されます'
            f'（または meeting/src/meetingbench の「python -m meetingbench models」。こちらは通信の証明書を検証します）。')
    return key, path


def profile_dir_problem(profile_dir) -> Optional[str]:
    """登録フォルダが使えないときの理由（日本語）。使えるなら None。"""
    profile_dir = str(profile_dir or '').strip()
    if not profile_dir:
        return ('声の登録フォルダが指定されていません。voice_profile_dir に登録フォルダ（profiles）を指定してください'
                '（cli.py なら --voice-profile-dir。登録の作り方は docs/meeting_enrolled_speakers_ja.md）。')
    if not Path(profile_dir).is_dir():
        return f'声の登録フォルダが見つかりません: {profile_dir}'
    if not VoiceProfiles.load(profile_dir).profiles:
        return (f'声の登録が1人も見つかりません: {profile_dir}\n'
                f'「<名前>/profile.json」と、そこに書かれた手本の wav がそろったフォルダが必要です'
                f'（meeting の voiceprofile build で作ります）。')
    return None


def enrolled_kwargs(settings) -> dict:
    """設定（settings）から enrolled_speakers に渡す引数を作る。使えない設定なら ValueError / FileNotFoundError（日本語）。"""
    profile_dir = str(settings.get('voice_profile_dir', '') or '').strip()
    problem = profile_dir_problem(profile_dir)
    if problem:
        raise ValueError(problem)
    model = str(settings.get('enrolled_speaker_model', DEFAULT_MODEL) or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    model_path(model)  # 先に確認する（無ければここで止める）
    return {'profile_dir': profile_dir, 'model': model, 'min_sim': parse_min_sim(settings.get('enrolled_min_sim', ''))}


# ---------------------------------------------------------------------------
# 音声の読み込みと前処理
# ---------------------------------------------------------------------------
def _read_wav_stdlib(path) -> Tuple[np.ndarray, int]:
    """soundfile が無いときの代わり（整数 PCM の WAV だけ）。(波形 float32 (n, ch), サンプリング周波数)"""
    with wave.open(str(path), 'rb') as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        y = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        y = ((b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
             .astype(np.float32) / 8388608.0)
    elif width == 4:
        y = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f'未対応の WAV 形式です: {path}')
    return y.reshape(-1, ch), sr


def _resample(y: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target or len(y) == 0:
        return y
    try:
        import librosa
        return np.asarray(librosa.resample(y, orig_sr=sr, target_sr=target), dtype=np.float32)
    except Exception:  # librosa が無い／動かない環境では scipy、それも無ければ線形補間（話者の照合には十分）
        pass
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), int(target))
        return np.asarray(resample_poly(y, int(target) // g, int(sr) // g), dtype=np.float32)
    except Exception:
        n = int(round(len(y) * target / sr))
        return np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)


def read_audio_16k(path: Union[str, Path]) -> np.ndarray:
    """音声ファイル -> 16kHz・モノラル・float32 の波形。"""
    try:
        import soundfile as sf
        y, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except ImportError:
        y, sr = _read_wav_stdlib(path)
    y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
    return np.ascontiguousarray(_resample(np.asarray(y, dtype=np.float32), int(sr)), dtype=np.float32)


def trim_edges(y: np.ndarray, top_db: float = TRIM_DB, frame: int = 512, hop: int = 160) -> np.ndarray:
    """前後の無音を落とす（最大のフレームより top_db 小さいフレームを無音とみなす）。落としすぎる場合は元のまま。"""
    if len(y) < frame + hop:
        return y
    c = np.concatenate([[0.0], np.cumsum(np.asarray(y, dtype=np.float64) ** 2)])
    starts = np.arange(0, len(y) - frame + 1, hop)
    power = (c[starts + frame] - c[starts]) / frame
    peak = power.max()
    if peak <= 0:
        return y
    loud = np.nonzero(10.0 * np.log10(np.maximum(power, 1e-20) / peak) > -top_db)[0]
    if len(loud) == 0:
        return y
    a, b = int(starts[loud[0]]), min(len(y), int(starts[loud[-1]]) + frame)
    return y[a:b] if b - a >= 160 else y


def pad_min(y: np.ndarray, min_sec: float = PAD_SEC) -> np.ndarray:
    n = int(min_sec * SAMPLE_RATE)
    if len(y) >= n:
        return y
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.tile(y, int(math.ceil(n / len(y))))[:n]


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


# ---------------------------------------------------------------------------
# 話者埋め込み器
# ---------------------------------------------------------------------------
class SherpaEmbedder:
    """sherpa-onnx の SpeakerEmbeddingExtractor（built_speakers と同じ部品）。embed() は 16kHz の波形を受け取る。"""

    def __init__(self, key: str, onnx_path: Union[str, Path], num_threads: int = 1):
        self.key, self.path, self.num_threads = key, Path(onnx_path), max(int(num_threads), 1)
        # 登録の埋め込みキャッシュの鍵：モデルを替えた（ファイル名・大きさが違う）ら作り直す。前処理を変えたら EMB_CACHE_VERSION を上げる。
        self.model_id = f'{key}:{self.path.name}:{self.path.stat().st_size}'
        self._ex = None

    def _load(self):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.path), num_threads=self.num_threads, provider='cpu')
        if not cfg.validate():
            raise RuntimeError(f'sherpa-onnx の設定が正しくありません（モデルが壊れている可能性）: {self.path}')
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, wav16k: np.ndarray) -> Optional[np.ndarray]:
        """長さ1の単位ベクトル。計算できなければ None。"""
        if self._ex is None:
            self._load()
        y = pad_min(trim_edges(np.asarray(wav16k, dtype=np.float32)))
        try:
            s = self._ex.create_stream()
            s.accept_waveform(sample_rate=SAMPLE_RATE, waveform=y)
            s.input_finished()
            if not self._ex.is_ready(s):
                return None
            v = np.asarray(self._ex.compute(s), dtype=np.float64)
        except Exception as e:
            logger.warning(f'[enrolled] embedding failed: {e}')
            return None
        n = float(np.linalg.norm(v))
        return v / n if np.all(np.isfinite(v)) and n > 0 else None


# ---------------------------------------------------------------------------
# 登録した声の平均（キャッシュ付き）
# ---------------------------------------------------------------------------
def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def profile_clip_files(folder: Path, refs: Sequence[ProfileRef]) -> List[Path]:
    """その人の登録に使う wav：profile.json の手本（voice_profile.load_profile が読めたもの）＋ eval/*.wav（あれば）。"""
    files, seen = [], set()
    for p in [Path(r.wav) for r in refs] + sorted((folder / 'eval').glob('*.wav')):
        rp = p.resolve()
        if rp not in seen and p.is_file():
            seen.add(rp)
            files.append(p)
    return files


def cache_path(folder: Path, model_key: str) -> Path:
    return folder / f'enrolled_emb_{model_key}.json'


def _load_cache(path: Path, model_id: str) -> Dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') == EMB_CACHE_VERSION and data.get('model_id') == model_id and isinstance(data.get('clips'), dict):
            return data['clips']
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def _save_cache(path: Path, model_id: str, clips: Dict[str, dict]) -> None:
    try:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps({'version': EMB_CACHE_VERSION, 'model_id': model_id, 'clips': clips}, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, path)
    except OSError as e:  # 読み取り専用の場所でも動くようにする（毎回計算し直すだけ）
        logger.warning(f'[enrolled] cannot write the embedding cache {path}: {e}')


def profile_centroid(folder: Path, refs: Sequence[ProfileRef], embedder, model_key: str = '') -> Tuple[Optional[np.ndarray], int]:
    """
    その人の「平均の声の特徴」（長さ1の単位ベクトル）と、使った wav の数。
    wav ごとの埋め込みを <フォルダ>/enrolled_emb_<モデル>.json に保存し、wav の中身（SHA-256）とモデルが同じなら再利用する。
    """
    cp = cache_path(folder, model_key or getattr(embedder, 'key', 'model'))
    model_id = getattr(embedder, 'model_id', model_key)
    old = _load_cache(cp, model_id)
    new: Dict[str, dict] = {}
    embs = []
    for p in profile_clip_files(folder, refs):
        try:
            rel = p.resolve().relative_to(folder.resolve()).as_posix()
        except ValueError:
            rel = p.resolve().as_posix()
        digest = file_sha256(p)
        hit = old.get(rel)
        if hit and hit.get('sha256') == digest and isinstance(hit.get('emb'), list):
            v = np.asarray(hit['emb'], dtype=np.float64)
        else:
            v = embedder.embed(read_audio_16k(p))
            if v is None:
                logger.warning(f'[enrolled] cannot embed {p}, skipped')
                continue
            v = np.asarray(v, dtype=np.float64)
        new[rel] = {'sha256': digest, 'emb': [round(float(x), 7) for x in v]}
        embs.append(unit(v))
    if new != old and (new or old):
        _save_cache(cp, model_id, new)
    if not embs:
        return None, 0
    return unit(np.mean(embs, axis=0)), len(embs)


def enrolled_centroids(profile_dir: Union[str, Path], embedder, model_key: str = '') -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """登録者の名前（登録フォルダの名前。voice_profile.VoiceProfiles と同じ）、平均の特徴 (人数, 次元)、人ごとの wav の数。"""
    vp = VoiceProfiles.load(profile_dir)
    names, cents, counts = [], [], {}
    for name, refs in vp.profiles.items():
        c, n = profile_centroid(Path(profile_dir) / name, refs, embedder, model_key)
        if c is None:
            logger.warning(f'[enrolled] profile {name}: no usable clip, skipped')
            continue
        names.append(name)
        cents.append(c)
        counts[name] = n
    if not names:
        raise ValueError(f'登録した声から特徴を計算できませんでした: {profile_dir}')
    return names, np.stack(cents), counts


# ---------------------------------------------------------------------------
# 行の中の窓判定（話者の途中交代の目印）
# ---------------------------------------------------------------------------
def window_spans(start_ms: int, end_ms: int, win_ms: int = WIN_MS, hop_ms: int = HOP_MS) -> List[Tuple[int, int]]:
    """[start_ms, end_ms) を win_ms の窓で hop_ms ずつずらして覆う（最後の窓は終わりに合わせる）。短い行は全体で1つ。"""
    if end_ms - start_ms <= win_ms:
        return [(int(start_ms), int(end_ms))]
    spans, a = [], start_ms
    while a + win_ms < end_ms:
        spans.append((int(a), int(a + win_ms)))
        a += hop_ms
    spans.append((int(end_ms - win_ms), int(end_ms)))
    return spans


def pieces_from_windows(spans: Sequence[Tuple[int, int]], sim_w: np.ndarray, start_ms: int, end_ms: int,
                        hop_ms: int = HOP_MS, min_piece_ms: int = MIN_PIECE_MS) -> List[Tuple[int, int, int]]:
    """
    窓ごとの類似度 (窓数, 人数) から、行の中を話者ごとの部分 [(開始ms, 終了ms, 登録者の番号)] に分ける。
    行を hop_ms の升目に分け、升目の中心を覆う窓の平均で決め → 前後1升の多数決で整え → 同じ人の続きをまとめ
    → min_piece_ms 未満の部分は隣の長い方に吸収する（meeting/src/meetingbench/enroll_assign.py の timeline_from_windows と同じ）。
    """
    cuts = list(range(int(start_ms), int(end_ms), hop_ms)) + [int(end_ms)]
    cells = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    sp = np.asarray(spans)
    lab = []
    for a, b in cells:
        mid = (a + b) / 2
        cover = np.nonzero((sp[:, 0] <= mid) & (mid <= sp[:, 1]))[0]
        if len(cover) == 0:
            cover = np.array([int(np.argmin(np.abs((sp[:, 0] + sp[:, 1]) / 2 - mid)))])
        lab.append(int(np.argmax(sim_w[cover].mean(axis=0))))
    sm = list(lab)
    for i in range(1, len(lab) - 1):
        trio = lab[i - 1:i + 2]
        top = max(set(trio), key=trio.count)
        if trio.count(top) >= 2:
            sm[i] = top
    pieces: List[list] = []
    for (a, b), k in zip(cells, sm):
        if pieces and pieces[-1][2] == k:
            pieces[-1][1] = b
        else:
            pieces.append([a, b, k])
    while len(pieces) > 1:
        short = [i for i, p in enumerate(pieces) if p[1] - p[0] < min_piece_ms]
        if not short:
            break
        i = min(short, key=lambda j: pieces[j][1] - pieces[j][0])
        j = max([k for k in (i - 1, i + 1) if 0 <= k < len(pieces)], key=lambda k: pieces[k][1] - pieces[k][0])
        pieces[j][0], pieces[j][1] = min(pieces[j][0], pieces[i][0]), max(pieces[j][1], pieces[i][1])
        del pieces[i]
        merged: List[list] = []
        for p in pieces:
            if merged and merged[-1][2] == p[2]:
                merged[-1][1] = p[1]
            else:
                merged.append(p)
        pieces = merged
    return [(int(a), int(b), int(k)) for a, b, k in pieces]


# ---------------------------------------------------------------------------
# 字幕の行ごとの割り当て
# ---------------------------------------------------------------------------
def label_segments(audio: np.ndarray, subtitles: Sequence[Sequence[int]], names: Sequence[str], centroids: np.ndarray, embedder,
                   min_sim: Optional[float] = None, window_check: bool = True, min_seg_sec: float = MIN_SEG_SEC,
                   low_margin: float = LOW_MARGIN) -> Tuple[List[str], dict]:
    """
    audio: 16kHz モノラルの波形。subtitles: [[開始ms, 終了ms], ...]（字幕の行順）。
    戻り値: (行ごとの話者名, 診断 dict)。埋め込める行が1つもなければ ValueError。
    """
    n = len(subtitles)
    cents = unit(np.asarray(centroids, dtype=np.float64))
    names = list(names)
    best_idx: List[Optional[int]] = [None] * n
    sims: List[Optional[np.ndarray]] = [None] * n
    for i, sub in enumerate(subtitles):
        try:
            s_ms, e_ms = int(sub[0]), int(sub[1])
        except (TypeError, ValueError, IndexError):
            continue
        if e_ms <= s_ms or (e_ms - s_ms) / 1000.0 < min_seg_sec:
            continue
        seg = audio[max(s_ms, 0) * SAMPLE_RATE // 1000: max(e_ms, 0) * SAMPLE_RATE // 1000]
        if len(seg) < min_seg_sec * SAMPLE_RATE * 0.5:  # 音声の終わりより後ろにある行
            continue
        v = embedder.embed(seg)
        if v is None:
            continue
        sims[i] = cents @ unit(v)
        best_idx[i] = int(np.argmax(sims[i]))

    first = next((i for i in range(n) if best_idx[i] is not None), None)
    if first is None:
        raise ValueError('埋め込める字幕の行が1つもありませんでした（音声が短すぎる、または字幕の時刻が音声とずれています）')

    labels: List[str] = []
    rows: List[dict] = []
    prev = None
    for i in range(n):
        s_ms, e_ms = (int(subtitles[i][0]), int(subtitles[i][1])) if len(subtitles[i]) >= 2 else (0, 0)
        row = {'index': i, 'start_ms': s_ms, 'end_ms': e_ms, 'speaker': None, 'best': None, 'second': None, 'second_speaker': None,
               'margin': None, 'inherited': False, 'flags': []}
        if sims[i] is not None:
            order = np.argsort(-sims[i])
            best = float(sims[i][order[0]])
            row['best'] = round(best, 4)
            if len(order) > 1:
                second = float(sims[i][order[1]])
                row['second'], row['second_speaker'] = round(second, 4), names[int(order[1])]
                row['margin'] = round(best - second, 4)
                if best - second < low_margin:
                    row['flags'].append('low_margin')
            if min_sim is not None and best < min_sim:
                row['speaker'] = UNKNOWN
                row['flags'].append('unknown')
            else:
                row['speaker'] = names[int(order[0])]
        else:
            row['inherited'] = True
            row['flags'].append('inherited')
            row['speaker'] = prev if prev is not None else None  # 先頭が埋め込めないときは、あとで最初の行の話者にする
        prev = row['speaker']
        rows.append(row)
    for row in rows[:first]:  # 先頭の埋め込めない行 = 最初に埋め込めた行の話者
        row['speaker'] = rows[first]['speaker']
    labels = [r['speaker'] for r in rows]

    if window_check:
        for i, row in enumerate(rows):
            if sims[i] is None:
                continue
            spans = window_spans(row['start_ms'], row['end_ms'])
            if len(spans) < 2:
                continue
            sim_w = np.zeros((len(spans), len(names)))
            ok = True
            for k, (a, b) in enumerate(spans):
                v = embedder.embed(audio[max(a, 0) * SAMPLE_RATE // 1000: max(b, 0) * SAMPLE_RATE // 1000])
                if v is None:
                    ok = False
                    break
                sim_w[k] = cents @ unit(v)
            if not ok:
                continue
            pieces = pieces_from_windows(spans, sim_w, row['start_ms'], row['end_ms'])
            if len(pieces) > 1:
                row['flags'].append('mid_change')
                row['pieces'] = [{'start_ms': a, 'end_ms': b, 'speaker': names[k]} for a, b, k in pieces]

    counts = {f: sum(1 for r in rows if f in r['flags']) for f in FLAG_JA}
    diag = {
        'version': 1,
        'speakers': names,
        'min_sim': min_sim,
        'low_margin': low_margin,
        'window_check': bool(window_check),
        'summary': {'lines': n, **counts, 'review': sum(1 for r in rows if r['flags'])},
        'segments': rows,
    }
    return labels, diag


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------
def write_diagnostics(diag: dict, speak_file: Union[str, Path]) -> Tuple[Path, Path]:
    """speak_file の隣に speaker_diag.json（確認用の全情報）と speaker_diag.csv（Excel で開ける表）を書く。"""
    out = Path(speak_file).parent
    jp, cp = out / DIAG_JSON, out / DIAG_CSV
    jp.write_text(json.dumps(diag, ensure_ascii=False, indent=1), encoding='utf-8')
    with open(cp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['行', '開始(秒)', '終了(秒)', '話者', '最高点', '2位の話者', '2位の点', '差', '要確認', '理由', '途中で替わった候補'])
        for r in diag['segments']:
            pieces = ' / '.join(f"{p['start_ms'] / 1000:.1f}-{p['end_ms'] / 1000:.1f}s {p['speaker']}" for p in r.get('pieces', []))
            w.writerow([r['index'] + 1, f"{r['start_ms'] / 1000:.2f}", f"{r['end_ms'] / 1000:.2f}", r['speaker'],
                        '' if r['best'] is None else r['best'], r['second_speaker'] or '', '' if r['second'] is None else r['second'],
                        '' if r['margin'] is None else r['margin'], '●' if r['flags'] else '',
                        ' ／ '.join(FLAG_JA[x] for x in r['flags']), pieces])
    return jp, cp


def copy_review_files(cache_folder: Union[str, Path], target_dir: Union[str, Path]) -> None:
    """処理用フォルダ（終了後に消える）にある診断ファイルを出力フォルダへコピーする。失敗しても処理は続ける。"""
    for name in (DIAG_JSON, DIAG_CSV):
        src = Path(cache_folder) / name
        try:
            if src.is_file():
                shutil.copy2(src, Path(target_dir) / name)
        except OSError as e:
            logger.warning(f'[enrolled] cannot copy {src}: {e}')


def enrolled_speakers(*, input_file, subtitles_file: str, speak_file: str, profile_dir, model: str = DEFAULT_MODEL, min_sim=None,
                      num_speakers=-1, window_check: bool = True, num_threads: int = 1, embedder=None, **kw):
    """
    登録した声との照合で、字幕の行ごとに話者名を付ける。built_speakers と同じ呼び方・同じ戻り値（成功 (True, None)、失敗 (False, 理由)）。

    subtitles_file: [[開始ms, 終了ms], ...]（字幕の行順）。speak_file: 行ごとの話者名の JSON 配列（built と同じ形式）。
    num_speakers は使わない（人数は登録した人数）。embedder はテスト用（本番では sherpa-onnx の onnx を使う）。
    """
    try:
        _st = time.time()
        min_sim = parse_min_sim(min_sim)
        problem = profile_dir_problem(profile_dir)
        if problem:
            raise ValueError(problem)
        subtitles = json.loads(Path(subtitles_file).read_text(encoding='utf-8'))
        if embedder is None:
            key, path = model_path(model)
            embedder = SherpaEmbedder(key, path, num_threads)
        else:
            key = getattr(embedder, 'key', str(model))
        names, centroids, counts = enrolled_centroids(profile_dir, embedder, key)
        if len(names) < 2:
            logger.warning(f'[enrolled] only {len(names)} enrolled speaker: every line gets the same name')
        if num_speakers and num_speakers > 0 and num_speakers != len(names):
            logger.debug(f'[enrolled] num_speakers={num_speakers} is ignored (enrolled speakers: {len(names)})')
        logger.debug(f'[enrolled] speakers={counts} model={key} min_sim={min_sim}')
        labels, diag = label_segments(read_audio_16k(input_file), subtitles, names, centroids, embedder, min_sim=min_sim,
                                      window_check=window_check)
        diag.update({'model': key, 'model_id': getattr(embedder, 'model_id', key), 'clips_per_speaker': counts})
        if not labels:
            return False, '0 speakers'
        Path(speak_file).write_text(json.dumps(labels), encoding='utf-8')  # built_speakers と同じ書き方
        write_diagnostics(diag, speak_file)
        s = diag['summary']
        logger.debug(f'[enrolled] done: {s} in {int(time.time() - _st)}s')
        return True, None
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(f'[enrolled] failed: {e}', exc_info=True)
        return False, f'{e}{msg}'
