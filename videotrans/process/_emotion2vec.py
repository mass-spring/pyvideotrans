# -*- coding: utf-8 -*-
"""
emotion_backend = 'emotion2vec'（音声から行ごとの感情を調べる。emotion2vec+ large / base）

字幕の1行ごとに、元の音声の [開始, 終了) を切り出して emotion2vec+ にかけ、emotion.json の1件（EmotionEntry）にする。
SenseVoice のタグ（ほぼ EMO_UNKNOWN になる）と違い、9分類のスコアが必ず得られる。測定結果と使い方は docs/meeting_emotion_ja.md。

- モデル：iic/emotion2vec_plus_large（既定）／iic/emotion2vec_plus_base。ModelScope から models/ へ取得（SenseVoice と同じ
  util/help_down.py の check_and_down_ms。通信の証明書は検証する）。ライセンスの法務確認は PLAN Q5 で未了。
- 9分類：angry disgusted fearful happy neutral other sad surprised unknown(<unk>)。
- ラベルの決め方（setting emotion2vec_label_mode）
    core4（既定）：平常・喜び・怒り・悲しみの4つだけから、スコアが最大のものを選ぶ（PLAN 5.3 の4分類。JVNV＋FLEURS で large 76.8%）
    all          ：9分類すべてから最大を選ぶ（恐れ・嫌悪・驚きも出る。other / unknown が勝った行は neutral・信頼度0）
  どちらでも scores（9分類）を emotion.json に残す。confidence = 選んだラベルの元のスコア（4つに割り直さない：
  「恐れ 0.6・悲しみ 0.3」のような行を、悲しみ 100% と見せないため）。intensity も同じ値（平常は 0）。
- 0.5 秒未満の行、音声の範囲外の行、モデルが失敗した行は neutral・強さ0・信頼度0・source='none'（翻訳の指示文・TTS には使われない）。
- 重いモデルは別プロセス（_base.py::_new_process）で動かす。emotion2vec_lines がその入口。
- 1行ずつ（granularity='utterance'、extract_embedding=False）。30 秒を超える行は 30 秒以下に等分して、長さで重みを付けて平均する。
"""
import json
import math
import os
import time
import traceback
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from videotrans.configure.config import ROOT_DIR, logger
from videotrans.process._utils import _write_log
from videotrans.util.emotion import SOURCE_EMOTION2VEC, SOURCE_NONE, EmotionBackendError, EmotionEntry

SAMPLE_RATE = 16000
MODEL_IDS = {'large': 'iic/emotion2vec_plus_large', 'base': 'iic/emotion2vec_plus_base'}
DEFAULT_MODEL = 'large'
LABEL_MODES = ('core4', 'all')
DEFAULT_LABEL_MODE = 'core4'
# emotion2vec+ の9分類（tokens.txt: 生气/angry 厌恶/disgusted 恐惧/fearful 开心/happy 中立/neutral 其他/other 难过/sad 吃惊/surprised <unk>）
E2V_CLASSES = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown')
CORE4 = ('neutral', 'happy', 'angry', 'sad')  # PLAN 5.3 の4分類（同点のときは左を優先）
NO_EMOTION_CLASSES = ('other', 'unknown')  # これが勝った行は感情なし（neutral・信頼度0）
MIN_LINE_SEC = 0.5  # これより短い行はモデルにかけない
MAX_CHUNK_SEC = 30.0  # これを超える行は等分して平均する
MODEL_FILES = ('model.pt', 'config.yaml')  # 取得後にこの2つが無ければ使えない
AUTO_THREADS = 4  # emotion2vec_threads=0 のとき使うスレッド数の上限
SCORE_DIGITS = 6  # emotion.json に残すスコアの桁数（測定用の emocheck と同じ。ラベルもこの丸めた値から選ぶ）


# ---------------------------------------------------------------------------
# 設定値の読み取りと確認（メッセージは利用者向けの日本語）
# ---------------------------------------------------------------------------
def parse_model(value) -> str:
    """'large' / 'base'（空は既定の large）。それ以外は EmotionBackendError。"""
    key = str(value or '').strip().lower() or DEFAULT_MODEL
    if key not in MODEL_IDS:
        raise EmotionBackendError(f'emotion2vec_model は {" / ".join(MODEL_IDS)} のどちらかにしてください: {value!r}')
    return key


def parse_label_mode(value) -> str:
    """'core4' / 'all'（空は既定の core4）。それ以外は EmotionBackendError。"""
    key = str(value or '').strip().lower() or DEFAULT_LABEL_MODE
    if key not in LABEL_MODES:
        raise EmotionBackendError(f'emotion2vec_label_mode は {" / ".join(LABEL_MODES)} のどちらかにしてください: {value!r}')
    return key


def parse_threads(value) -> int:
    """スレッド数。0・空は自動（CPU の数、最大 AUTO_THREADS）。負の数・整数でない値は EmotionBackendError。"""
    text = str(value if value is not None else '').strip()
    if text == '':
        n = 0
    else:
        try:
            n = int(float(text))
        except ValueError:
            raise EmotionBackendError(f'emotion2vec_threads は 0（自動）か 1 以上の整数にしてください: {value!r}') from None
    if n < 0:
        raise EmotionBackendError(f'emotion2vec_threads は 0（自動）か 1 以上の整数にしてください: {value!r}')
    return n if n > 0 else max(1, min(AUTO_THREADS, os.cpu_count() or 1))


def info_string(size: str, mode: str) -> str:
    """EmotionEntry.info に入れる『作り方の控え』。設定を変えたら emotion.json を作り直すための目印。"""
    return f'emotion2vec_plus_{size} {mode}'


def default_model_dir(size: str) -> Path:
    return Path(ROOT_DIR) / 'models' / f'emotion2vec_plus_{size}'


def _missing_model_files(folder: Path) -> List[str]:
    return [n for n in MODEL_FILES if not (folder / n).is_file()]


def ensure_model(size: str, callback=None) -> Path:
    """モデルのフォルダ（models/emotion2vec_plus_<size>）を返す。無ければ ModelScope から取得する（SenseVoice などと同じ方法）。
    取得できない・ファイルがそろわないときは EmotionBackendError（日本語）。"""
    folder = default_model_dir(size)
    model_id = MODEL_IDS[size]
    try:
        from videotrans.util.help_down import check_and_down_ms
        check_and_down_ms(model_id=model_id, callback=callback, local_dir=folder.as_posix())
    except Exception as e:
        logger.warning(f'emotion2vec+ のモデルの取得に失敗: {model_id}: {e}')
        raise EmotionBackendError(
            f'emotion2vec+ のモデル（{model_id}）を用意できませんでした。通信できる状態でもう一度実行するか、'
            f'https://modelscope.cn/models/{model_id}/files の中身を {folder.as_posix()} に置いてください（{e}）') from e
    missing = _missing_model_files(folder)
    if missing:
        raise EmotionBackendError(f'emotion2vec+ のモデルのファイルが足りません: {folder.as_posix()} に {" / ".join(missing)} がありません')
    return folder


# ---------------------------------------------------------------------------
# 9分類のスコア -> ラベル・強さ・信頼度
# ---------------------------------------------------------------------------
def canonical_class(name) -> str:
    """'生气/angry' '<unk>' など -> 'angry' 'unknown' のような英語の小文字名。読めなければ ValueError。"""
    n = str(name or '').strip().lower()
    if n in ('<unk>', 'unk', 'unknown', '未知'):
        return 'unknown'
    n = n.rsplit('/', 1)[-1].strip()
    if n not in E2V_CLASSES:
        raise ValueError(f'emotion2vec+ の分類名が読めません: {name!r}（期待: {", ".join(E2V_CLASSES)}）')
    return n


def class_scores(labels: Sequence, values: Sequence) -> Dict[str, float]:
    """モデルの出力（labels と scores）-> {分類名: スコア}。個数が合わない・数値でない・有限でないときは RuntimeError。"""
    labels, values = list(labels or []), list(values if values is not None else [])
    if not labels or len(labels) != len(values):
        raise RuntimeError(f'emotion2vec+ の出力が読めません（分類 {len(labels)} 個 / スコア {len(values)} 個）')
    out: Dict[str, float] = {}
    for lab, v in zip(labels, values):
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise RuntimeError(f'emotion2vec+ のスコアが数値ではありません: {v!r}') from None
        if not math.isfinite(x):
            raise RuntimeError(f'emotion2vec+ のスコアが有限の数ではありません: {v!r}')
        out[canonical_class(lab)] = x
    return out


def choose_label(scores: Dict[str, float], mode: str = DEFAULT_LABEL_MODE) -> Tuple[str, float, float]:
    """9分類のスコア -> (label, intensity, confidence)。util/emotion.py の LABELS のどれか。

    mode='core4'：neutral/happy/angry/sad だけから最大を選ぶ。mode='all'：全分類から最大を選ぶ（other / unknown が勝てば neutral・信頼度0）。
    confidence は選んだ分類の元のスコア。intensity は同じ値（neutral は 0）。
    """
    mode = parse_label_mode(mode)
    pool = CORE4 if mode == 'core4' else E2V_CLASSES
    present = [c for c in pool if c in scores]
    if not present:
        return 'neutral', 0.0, 0.0
    best = max(present, key=lambda c: scores[c])  # 同点は pool の並びで先のもの
    if best in NO_EMOTION_CLASSES:
        return 'neutral', 0.0, 0.0
    score = round(min(1.0, max(0.0, scores[best])), SCORE_DIGITS)
    return best, (0.0 if best == 'neutral' else score), score


def make_entry(start_ms: int, end_ms: int, scores: Dict[str, float], mode: str, info: str) -> EmotionEntry:
    # スコアは SCORE_DIGITS 桁に丸めて保存する。ラベルも丸めた値から選ぶ（emotion.json の scores から同じラベルを再現でき、
    # 4つとも 0.000000 のように区別できないときは先頭の neutral になる。meeting/src/emocheck と同じ丸め方・同点の扱い）
    rounded = {k: round(v, SCORE_DIGITS) for k, v in scores.items()}
    label, intensity, confidence = choose_label(rounded, mode)
    return EmotionEntry(int(start_ms), int(end_ms), label, intensity, confidence, SOURCE_EMOTION2VEC, scores=rounded, info=info)


def none_entry(start_ms: int, end_ms: int, info: str = '') -> EmotionEntry:
    """調べられなかった行：neutral・強さ0・信頼度0（翻訳の指示文・TTS には使われない）"""
    return EmotionEntry(int(start_ms), int(end_ms), 'neutral', 0.0, 0.0, SOURCE_NONE, info=info)


# ---------------------------------------------------------------------------
# 音声の読み込み（行ごとに必要な部分だけ。全体は読み込まない）
# ---------------------------------------------------------------------------
def _decode_pcm(raw: bytes, width: int, channels: int) -> np.ndarray:
    """整数 PCM のバイト列 -> float32 (フレーム数, チャンネル数)"""
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
        raise ValueError(f'未対応の WAV 形式です（{width * 8} ビット）')
    return y.reshape(-1, channels)


def _resample(y: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target or len(y) == 0:
        return y
    try:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), int(target))
        return np.asarray(resample_poly(y, int(target) // g, int(sr) // g), dtype=np.float32)
    except Exception:  # scipy が無い環境では線形補間（感情の判定には十分）
        n = int(round(len(y) * target / sr))
        return np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)


class SegmentReader:
    """音声ファイルから [開始ms, 終了ms) を 16kHz・モノラル・float32 で取り出す（pyVideoTrans の source_wav は 16kHz モノラル wav）。

    soundfile があればそれを使い（wav 以外も読める）、無ければ標準ライブラリで wav だけ読む。範囲が音声の外なら空の配列。
    """

    def __init__(self, path):
        self.path = str(path)
        self._sf = None
        self._wav = None
        try:
            import soundfile as sf
        except ImportError:
            sf = None
        try:
            if sf is not None:
                self._sf = sf.SoundFile(self.path)
                self.sr, self.channels, self.frames = int(self._sf.samplerate), int(self._sf.channels), int(len(self._sf))
            else:
                self._wav = wave.open(self.path, 'rb')
                self.sr, self.channels, self.frames = self._wav.getframerate(), self._wav.getnchannels(), self._wav.getnframes()
        except Exception as e:
            self.close()
            raise EmotionBackendError(f'感情を調べる音声ファイルを開けません: {self.path} ({e})') from e

    @property
    def duration_ms(self) -> int:
        return int(self.frames * 1000 / self.sr) if self.sr else 0

    def read(self, start_ms: int, end_ms: int) -> np.ndarray:
        a = max(0, int(start_ms * self.sr / 1000))
        b = min(self.frames, int(end_ms * self.sr / 1000))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        if self._sf is not None:
            self._sf.seek(a)
            y = self._sf.read(b - a, dtype='float32', always_2d=True)
        else:
            self._wav.setpos(a)
            y = _decode_pcm(self._wav.readframes(b - a), self._wav.getsampwidth(), self.channels)
        y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
        return np.ascontiguousarray(_resample(np.asarray(y, dtype=np.float32), self.sr), dtype=np.float32)

    def close(self) -> None:
        for f in (self._sf, self._wav):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._sf = self._wav = None


# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
def pick_device(device_name: str = 'auto', is_cuda: bool = False, device_index: int = 0) -> str:
    """stt_funasr.py と同じ決め方：device_name が auto なら GPU 指定のときだけ cuda、他は device_name のまま。"""
    name = str(device_name or 'auto')
    return name if name != 'auto' else (f'cuda:{device_index}' if is_cuda else 'cpu')


def load_model(folder, device: str = 'cpu', threads: int = 2):
    """funasr の AutoModel で emotion2vec+ を読み込む。

    ncpu を必ず渡す：渡さないと funasr が torch のスレッド数を 4 に上書きし、他の処理と同時だと極端に遅くなる
    （meeting/docs/research.md 21章）。
    """
    try:
        import torch
        from funasr import AutoModel
    except ImportError as e:
        raise EmotionBackendError(
            f'funasr / torch が入っていません（emotion2vec+ に必要です。SenseVoice で文字起こしできる環境には入っています）: {e}') from e
    threads = max(1, int(threads))
    model = AutoModel(model=str(folder), hub='ms', device=device, ncpu=threads, disable_update=True, disable_pbar=True,
                      disable_log=True)
    torch.set_num_threads(threads)
    return model


def predict_scores(model, wav: np.ndarray) -> Dict[str, float]:
    """1行ぶんの波形（16kHz モノラル）-> 9分類のスコア。長い行は等分して、長さで重みを付けて平均する。"""
    max_len = int(MAX_CHUNK_SEC * SAMPLE_RATE)
    if len(wav) <= max_len:
        return _run_model(model, wav)
    k = int(math.ceil(len(wav) / max_len))
    bounds = [round(i * len(wav) / k) for i in range(k + 1)]
    total: Dict[str, float] = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        w = (b - a) / len(wav)
        for name, v in _run_model(model, wav[a:b]).items():
            total[name] = total.get(name, 0.0) + w * v
    return total


def _run_model(model, wav: np.ndarray) -> Dict[str, float]:
    res = model.generate(input=wav, granularity='utterance', extract_embedding=False, disable_pbar=True)
    if not res:
        raise RuntimeError('emotion2vec+ が結果を返しませんでした')
    return class_scores(res[0].get('labels'), res[0].get('scores'))


# ---------------------------------------------------------------------------
# 字幕の行ごとの感情
# ---------------------------------------------------------------------------
def analyse_lines(spans: Sequence[Sequence[int]], reader, model, mode: str = DEFAULT_LABEL_MODE, info: str = '',
                  progress: Optional[Callable[[int, int], None]] = None) -> List[EmotionEntry]:
    """spans=[[開始ms, 終了ms], ...]（字幕の行順）-> 同じ順・同じ件数の EmotionEntry。

    0.5 秒未満・音声の範囲外・モデルが失敗した行は none_entry。調べようとした行が全部失敗したときは、最初のエラーで RuntimeError。
    """
    mode = parse_label_mode(mode)
    out: List[EmotionEntry] = []
    tried = failed = 0
    first_error: Optional[Exception] = None
    for i, (start, end) in enumerate(spans):
        start, end = int(start), int(end)
        entry = none_entry(start, end, info)
        if (end - start) / 1000.0 >= MIN_LINE_SEC:
            wav = reader.read(start, end)
            if len(wav) >= MIN_LINE_SEC * SAMPLE_RATE * 0.5:  # 音声の終わりより後ろにある行は調べない
                tried += 1
                try:
                    entry = make_entry(start, end, predict_scores(model, wav), mode, info)
                except Exception as e:
                    failed += 1
                    first_error = first_error or e
                    logger.warning(f'[emotion2vec] 行 {i + 1} ({start}-{end}ms) の感情を調べられませんでした: {e}')
        out.append(entry)
        if progress:
            progress(i + 1, len(spans))
    if tried and failed == tried:
        raise RuntimeError(f'emotion2vec+ が全ての行で失敗しました: {first_error}') from first_error
    return out


def emotion2vec_lines(*, input_file, subtitles_file: str, model_dir: str = '', model_size: str = DEFAULT_MODEL,
                      label_mode: str = DEFAULT_LABEL_MODE, num_threads: int = 0, logs_file=None, model=None,
                      is_cuda: bool = False, device_name: str = 'auto', device_index: int = 0, **kw):
    """別プロセスで動かす入口（_base.py::_new_process の callback）。

    input_file: 元の音声（16kHz モノラル wav）。subtitles_file: [[開始ms, 終了ms], ...] の JSON（字幕の行順）。
    成功は ([EmotionEntry.to_dict() の配列, 字幕の行と同じ順・件数], None)、失敗は (False, 理由)。
    model はテスト用（本番は model_dir から funasr で読み込む）。
    """
    reader = None
    try:
        _st = time.time()
        size, mode = parse_model(model_size), parse_label_mode(label_mode)
        threads = parse_threads(num_threads)
        spans = json.loads(Path(subtitles_file).read_text(encoding='utf-8'))
        reader = SegmentReader(input_file)
        if model is None:
            folder = Path(model_dir) if model_dir else default_model_dir(size)
            missing = _missing_model_files(folder)
            if missing:
                raise EmotionBackendError(f'emotion2vec+ のモデルのファイルが足りません: {folder.as_posix()} に {" / ".join(missing)} がありません')
            device = pick_device(device_name, is_cuda, device_index)
            _write_log(logs_file, json.dumps({'type': 'logs', 'text': f'Load emotion2vec+ {size} on {device} ({threads} threads)'}))
            model = load_model(folder, device, threads)
            logger.debug(f'[emotion2vec] model loaded in {time.time() - _st:.1f}s: {folder} device={device} threads={threads}')

        def _progress(done, total):
            _write_log(logs_file, json.dumps({'type': 'logs', 'text': f'emotion2vec+ {done}/{total}'}))

        entries = analyse_lines(spans, reader, model, mode, info_string(size, mode), _progress)
        logger.debug(f'[emotion2vec] done: {len(entries)} lines in {time.time() - _st:.1f}s '
                     f'(none: {sum(1 for e in entries if e.source == SOURCE_NONE)})')
        return [e.to_dict() for e in entries], None
    except Exception as e:
        logger.exception(f'[emotion2vec] failed: {e}', exc_info=True)
        return False, f'{e}{traceback.format_exc()}'
    finally:
        if reader is not None:
            reader.close()
