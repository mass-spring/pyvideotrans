import shutil
from pathlib import Path
from typing import List

from videotrans.configure.config import settings, logger
from videotrans.util.emotion import (
    EMOTION_FILE, EMOTION_RAW_FILE, SOURCE_NONE, SOURCE_SENSEVOICE,
    EmotionEntry, build_entries_from_sensevoice, covers_lines, is_aligned, load_emotion, load_raw_capture, save_emotion,
)
from videotrans.util.help_srt import get_subtitle_from_srt

# setting `emotion_backend` に指定できる値。'none' は検出しない
# 'emotion2vec' は未実装(下の _emotion_from_emotion2vec が入口)
EMOTION_BACKENDS = ('sensevoice', 'emotion2vec', 'none')


class EmotionMixin:
    """行ごとの感情を emotion.json に用意する工程(翻訳の直前に呼ぶ)。

    setting `enable_emotion`(既定 False)が True のときだけ動く。失敗しても翻訳は止めない。
    """

    def _ensure_emotion(self, srt_list=None) -> List[EmotionEntry]:
        """最終の原文字幕に対応する emotion.json を用意し、全行分の EmotionEntry を返す。

        - 字幕の編集が終わったあと(= 翻訳の開始時)に呼ぶ。srt_list を渡さなければ source_sub を読む
        - 何度呼んでも同じ結果(冪等)。既存の emotion.json が最終字幕と1対1で対応していれば作り直さない。
          字幕の統合・分割・時刻修正で対応しなくなったら、文字起こしの控え(emotion_raw.json)から作り直す
        - 無効・backend='none'・検出できない・失敗のときは [] を返す(翻訳は感情ヒントなしで続く)
        """
        if not settings.get('enable_emotion', False):
            return []
        backend = str(settings.get('emotion_backend', 'sensevoice') or '').strip().lower()
        if backend in ('', 'none'):
            return []
        if backend not in EMOTION_BACKENDS:
            logger.warning(f'emotion_backend={backend!r} は未対応のため、感情ヒントなしで続行します。対応値: {EMOTION_BACKENDS}')
            return []
        try:
            lines = srt_list or get_subtitle_from_srt(self.cfg.source_sub, is_file=True)
            cache_json = Path(self.cfg.cache_folder) / EMOTION_FILE
            target_json = Path(self.cfg.target_dir) / EMOTION_FILE if self.cfg.target_dir else None

            # キャッシュが消えていても前回の結果を使う(speaker.json と同じ扱い)
            if not cache_json.exists() and target_json and target_json.exists():
                shutil.copy2(target_json, cache_json)

            existing = load_emotion(cache_json.as_posix())
            if existing and {e.source for e in existing} - {backend, SOURCE_NONE}:
                existing = []  # 別の backend で作ったもの
            raw_file = Path(self.cfg.cache_folder) / EMOTION_RAW_FILE
            # ASR をやり直して控えのほうが新しいときは作り直す
            stale = raw_file.exists() and cache_json.exists() and raw_file.stat().st_mtime > cache_json.stat().st_mtime
            if existing and not stale and is_aligned(existing, lines):
                logger.debug(f'emotion.json は最終字幕と対応しているので再利用: {len(existing)}件')
                return existing

            if backend == 'sensevoice':
                entries = self._emotion_from_sensevoice(lines)
            else:
                entries = self._emotion_from_emotion2vec(lines)
            if not entries:
                # 作り直す材料(控え)が無い。古い emotion.json でも全行に時刻で対応が付くならそれを使う
                if existing and covers_lines(existing, lines):
                    logger.debug('emotion.json は字幕と完全には一致しないが、全行に時刻で対応が付くので再利用')
                    return existing
                return []
            save_emotion(entries, self.cfg.cache_folder, self.cfg.target_dir)
            logger.debug(f'emotion.json を作成: backend={backend}, {len(entries)}件')
            return entries
        except NotImplementedError as e:
            logger.warning(f'{e}。感情ヒントなしで続行します')
        except Exception as e:
            logger.exception(f'感情の用意に失敗、感情ヒントなしで続行します: {e}', exc_info=True)
        return []

    def _emotion_from_sensevoice(self, lines) -> List[EmotionEntry]:
        """文字起こし時に控えた SenseVoice の感情タグ(emotion_raw.json)を、字幕行の開始時刻で対応付ける。

        暫定: SenseVoice は確率を出さないため、強さ=0.5(平常は0)・信頼度=0.6 の固定値(util/emotion.py)。
        """
        raw = load_raw_capture(self.cfg.cache_folder)
        if not raw:
            logger.warning(f'{EMOTION_RAW_FILE} がありません。文字起こしが SenseVoice 以外か、字幕を外部から読み込んだ可能性があります')
            return []
        entries = build_entries_from_sensevoice(lines, raw)
        n_hit = sum(1 for e in entries if e.source == SOURCE_SENSEVOICE)
        logger.debug(f'SenseVoice の感情タグを対応付け: {n_hit}/{len(entries)}行')
        return entries

    def _emotion_from_emotion2vec(self, lines) -> List[EmotionEntry]:
        """TODO(E2): emotion2vec+ で音声から感情を検出する。まだ実装していない(モデルのダウンロードもしない)。

        実装するときの約束:
        - 入力: lines=最終の原文字幕(start_time / end_time は ms)、音声=self.cfg.source_wav(16kHz wav)
        - 出力: lines と同じ順・同じ件数の EmotionEntry(source='emotion2vec')。
          label は util/emotion.py の LABELS の7種、intensity / confidence は 0..1(モデルの確率から作る)
        - 重いモデルは別プロセス(self._new_process)で動かす(docs/design.md 3章: process/_audio_emotion.py)
        - 重みのライセンス(FunASR MODEL_LICENSE)の法務確認が済むまで使わない(docs/PLAN.md Q5)
        """
        raise NotImplementedError('emotion_backend=emotion2vec は未実装です')
