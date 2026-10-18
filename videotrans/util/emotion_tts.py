# -*- coding: utf-8 -*-
"""
Emotion -> TTS parameters (EMOTION module, part 2 = E4).

Pure helpers (no network, no model): turn a line's emotion (util/emotion.py) into what each TTS engine understands.
The dubbing stage (task/_stage_dubbing.py) puts `emotion` / `emotion_intensity` / `emotion_confidence` into the per-line
queue item; the engines (tts/_voxcpm.py, tts/_chatterbox.py) read them through the functions below.

A line has "no emotion" (= upstream behaviour) when it is neutral, its confidence is below 0.5, it has no entry in emotion.json,
or `enable_emotion` / `emotion_tts` is off. Details and limits: docs/meeting_emotion_tts_ja.md
"""
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from videotrans.util.emotion import EMOTION_FILE, HINT_MIN_CONFIDENCE, EmotionIndex, load_emotion

# emotion label -> short English style (VoxCPM2 control instruction). 'neutral' is not here = no instruction
VOXCPM_STYLES: Dict[str, str] = {
    'angry': 'angry, tense',
    'happy': 'cheerful',
    'sad': 'sad, subdued',
    'fearful': 'anxious',
    'surprised': 'surprised',
    'disgusted': 'disgusted, displeased',
}

# intensity -> wording: < LOW "slightly", LOW..HIGH (inclusive) no word, > HIGH "very"
INTENSITY_LOW = 0.4
INTENSITY_HIGH = 0.7
DEFAULT_INTENSITY = 0.5  # used when a queue item has no (or a broken) intensity

# ChatterBox: emotional line -> exaggeration = base + GAIN * intensity, at most MAX (the official range is about 0.3..0.9)
CHATTERBOX_GAIN = 0.4
CHATTERBOX_MAX_EXAGGERATION = 0.9


def _to_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ------------------------------------------------------------------ data side (used by the dubbing stage)

def is_enabled(settings) -> bool:
    """Emotion is applied to TTS only while both `enable_emotion` and `emotion_tts` are on (the second one defaults to True)."""
    return bool(settings.get('enable_emotion', False)) and bool(settings.get('emotion_tts', True))


def load_index(settings, cache_folder: Optional[str], target_dir: Optional[str] = None) -> Optional[EmotionIndex]:
    """EmotionIndex of the task's emotion.json, or None (feature off / no file / empty = upstream behaviour).

    emotion.json is made in the cache folder before translation (task/_stage_emotion.py) and copied to the output folder;
    when the cache folder was cleaned (re-dubbing an old task), the copy in the output folder is used.
    """
    if not is_enabled(settings):
        return None
    for folder in (cache_folder, target_dir):
        if not folder:
            continue
        p = Path(folder) / EMOTION_FILE
        if p.is_file():
            entries = load_emotion(p.as_posix())
            if entries:
                return EmotionIndex(entries)
    return None


def find_line_emotion(index: Optional[EmotionIndex], start_time_source) -> Dict:
    """Emotion for the line whose SOURCE start time (ms) is `start_time_source`, matched by time (+-300 ms), never by line number.

    Returns {'emotion', 'emotion_intensity', 'emotion_confidence'} to merge into the queue item, or {} for "no emotion"
    (no index, no match, neutral, unknown label, confidence < 0.5).
    """
    entry = index.find(start_time_source) if index is not None else None
    if entry is None or entry.label not in VOXCPM_STYLES or entry.confidence < HINT_MIN_CONFIDENCE:
        return {}
    return {'emotion': entry.label, 'emotion_intensity': entry.intensity, 'emotion_confidence': entry.confidence}


def cache_suffix(emo: Dict) -> str:
    """Text added to the dubbing-cache key of a line that has an emotion (label + intensity; confidence does not change the audio)."""
    return f"emo-{emo['emotion']}-{_to_float(emo.get('emotion_intensity'), DEFAULT_INTENSITY):.2f}"


def emotion_engine_types() -> frozenset:
    """tts_type ids of the engines that carry the emotion: VoxCPM (control instruction) and ChatterBox (exaggeration).

    Other engines ignore it, so their dubbing-cache keys are left unchanged (no needless re-dubbing).
    """
    from videotrans.tts import _constants as c
    return frozenset({c.VOXCPM_TTS, c.CHATTERBOX_TTS})


def engine_uses_emotion(tts_type) -> bool:
    return tts_type in emotion_engine_types()


# ------------------------------------------------------------------ engine side (read a queue item)

def item_emotion(item: Dict) -> Optional[Tuple[str, float]]:
    """(label, intensity) of a queue item, or None when it has no usable emotion.

    The dubbing stage only sets the keys for usable emotions; the checks are repeated here because the queue can also come
    from a saved file (re-dubbing). A missing confidence counts as "already checked".
    """
    label = str(item.get('emotion') or '').strip().lower()
    if label not in VOXCPM_STYLES:
        return None
    if _to_float(item.get('emotion_confidence'), 1.0) < HINT_MIN_CONFIDENCE:
        return None
    return label, _clip01(_to_float(item.get('emotion_intensity'), DEFAULT_INTENSITY))


def intensity_word(intensity) -> str:
    """'slightly' (< 0.4) / '' (0.4..0.7) / 'very' (> 0.7)"""
    v = _clip01(_to_float(intensity, DEFAULT_INTENSITY))
    if v < INTENSITY_LOW:
        return 'slightly'
    if v > INTENSITY_HIGH:
        return 'very'
    return ''


def voxcpm_instruction(label: str, intensity) -> str:
    """VoxCPM2 control instruction for an emotion, e.g. ('angry', 0.9) -> 'very angry, tense'. '' for neutral / unknown labels."""
    style = VOXCPM_STYLES.get(str(label or '').strip().lower())
    if not style:
        return ''
    return f'{intensity_word(intensity)} {style}'.strip()


def sanitize_instruction(text: str) -> str:
    """Remove parentheses (the model reads "(instruction)text" literally; VoxCPM's own app does the same) and squeeze spaces."""
    return re.sub(r'\s+', ' ', re.sub(r'[()（）]', '', text or '')).strip()


def voxcpm_control_instruction(item: Dict, priority: str = 'emotion') -> str:
    """Value for the `control_instruction` field of the VoxCPM Gradio app. '' = no emotion (upstream).

    The VoxCPM app drops the instruction while transcript-guided cloning (use_prompt_text) is on, so an emotional line
    must give up the transcript. priority='similarity' keeps transcript-guided cloning and sends no instruction instead
    (setting `voxcpm_emotion_priority`; which is better is to be measured with meeting/src/voicecheck).
    """
    if str(priority).strip().lower() == 'similarity':
        return ''
    emo = item_emotion(item)
    return sanitize_instruction(voxcpm_instruction(*emo)) if emo else ''


def same_language(a: Optional[str], b: Optional[str]) -> bool:
    """Compare primary language subtags: 'zh-cn' == 'zh', 'JA' == 'ja'. Empty/unknown on either side counts as "same" (= leave settings alone)."""
    pa = str(a or '').strip().lower().replace('_', '-').split('-')[0]
    pb = str(b or '').strip().lower().replace('_', '-').split('-')[0]
    return not pa or not pb or pa == pb


def chatterbox_params(item: Dict, base_exaggeration: float, base_cfg_weight: float, target_lang: Optional[str],
                      crosslingual_cfg0: bool = True, has_ref: bool = True) -> Tuple[float, float]:
    """(exaggeration, cfg_weight) for one ChatterBox line.

    - exaggeration: the user's value for lines without emotion; for emotional lines
      min(0.9, base + 0.4 * intensity) (never below the user's value, in case it is already above 0.9).
    - cfg_weight: the user's value, except 0 when `crosslingual_cfg0` is on and the clone reference is in another language than
      the target (ChatterBox README: otherwise the output may inherit the accent of the reference). `has_ref=False`
      (built-in voice) never changes it.
    """
    exaggeration = _to_float(base_exaggeration, 0.5)
    cfg_weight = _to_float(base_cfg_weight, 0.5)
    emo = item_emotion(item)
    if emo:
        exaggeration = max(exaggeration, min(CHATTERBOX_MAX_EXAGGERATION, exaggeration + CHATTERBOX_GAIN * emo[1]))
    if crosslingual_cfg0 and has_ref and not same_language(item.get('ref_language'), target_lang):
        cfg_weight = 0.0
    return exaggeration, cfg_weight
