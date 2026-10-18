# -*- coding: utf-8 -*-
"""
Speaker voice profiles: clean, human-checked reference clips used as the voice-clone reference
instead of each subtitle line's own (short, noisy) audio.  See docs/meeting_voice_profile_ja.md.

Layout of the profile directory (setting `voice_profile_dir`):

    <dir>/speaker_map.json            {"spk0": "Aさん", "spk1": "Bさん"}      (optional)
    <dir>/<name>/profile.json         {"refs": [{"wav": "refs/a_01.wav", "text": "...", "lang": "ja",
                                                 "sec": 14.2, "snr_db": 32.1}, ...]}
    <dir>/<name>/refs/a_01.wav        (`wav` is relative to <dir>/<name>/)
"""
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from videotrans.configure.config import logger

PROFILE_JSON = 'profile.json'
SPEAKER_MAP_JSON = 'speaker_map.json'
# prefer a reference close to 12 s; anything longer than 20 s is scored as 20 s
REF_TARGET_SEC = 12.0
REF_MAX_SEC = 20.0
DEFAULT_CLONE_REF_SAMPLE_RATE = 44100


def lang_code(lang) -> str:
    """'ja-JP' / 'ja_JP' / 'JA' -> 'ja'"""
    return re.split(r'[-_]', str(lang or '').strip().lower())[0]


def _to_float(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class ProfileRef:
    profile: str  # profile name (folder name)
    wav: str  # absolute path of the reference clip
    text: str = ''  # human-checked transcript of the clip
    lang: str = ''  # language of the clip, e.g. 'ja'
    sec: Optional[float] = None
    snr_db: Optional[float] = None

    def cache_key(self) -> str:
        """Changes when the clip or its transcript is replaced, so dubbing cached with an older reference is not reused."""
        try:
            st = os.stat(self.wav)
            stamp = f'{st.st_size}-{st.st_mtime_ns}'
        except OSError:
            stamp = ''
        return f'{self.wav}-{stamp}-{self.text}'


def select_ref(refs: List[ProfileRef], lang: str = '') -> Optional[ProfileRef]:
    """Same language as the source speech first, then highest SNR, then length closest to 12 s (capped at 20 s)."""
    lang = lang_code(lang)

    def rank(r: ProfileRef):
        return (
            0 if lang and r.lang == lang else 1,
            -r.snr_db if r.snr_db is not None else math.inf,
            abs(min(r.sec, REF_MAX_SEC) - REF_TARGET_SEC) if r.sec is not None else math.inf,
        )

    return min(refs, key=rank, default=None)  # min() is stable: ties keep the order in profile.json


def load_profile(folder: Union[str, Path]) -> List[ProfileRef]:
    """Read <folder>/profile.json. Broken entries and missing wav files are skipped with a warning."""
    folder = Path(folder)
    try:
        data = json.loads((folder / PROFILE_JSON).read_text(encoding='utf-8-sig'))
        items = data.get('refs') or []
        if not isinstance(items, list):
            raise ValueError('"refs" must be a list')
    except (OSError, ValueError, AttributeError) as e:
        logger.warning(f'[voice_profile] cannot read {folder / PROFILE_JSON}: {e}')
        return []
    refs = []
    for r in items:
        wav = (folder / str(r.get('wav') or '')) if isinstance(r, dict) else None
        if wav is None or not wav.is_file():
            logger.warning(f'[voice_profile] reference wav not found, skipped: {folder.name}: {r}')
            continue
        sec = _to_float(r.get('sec'))
        refs.append(ProfileRef(
            profile=folder.name,
            wav=os.path.abspath(wav),
            text=str(r.get('text') or '').strip(),
            lang=lang_code(r.get('lang')),
            sec=sec if sec and sec > 0 else None,
            snr_db=_to_float(r.get('snr_db')),
        ))
    return refs


def load_speaker_map(profile_dir: Union[str, Path]) -> Dict[str, str]:
    """<dir>/speaker_map.json: diarization label -> profile name. Missing or broken file = empty map."""
    p = Path(profile_dir) / SPEAKER_MAP_JSON
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as e:
        logger.warning(f'[voice_profile] cannot read {p}: {e}')
        return {}
    return {str(k): str(v) for k, v in data.items() if v} if isinstance(data, dict) else {}


def read_speaker_labels(speaker_json: Union[str, Path]) -> List[str]:
    """speaker.json written by diarization: one label per subtitle line (same order as the source srt)."""
    try:
        data = json.loads(Path(speaker_json).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    return ['' if x is None else str(x).strip() for x in data] if isinstance(data, list) else []


@dataclass
class VoiceProfiles:
    profiles: Dict[str, List[ProfileRef]] = field(default_factory=dict)
    speaker_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, profile_dir: Union[str, Path]) -> 'VoiceProfiles':
        profiles = {}
        for sub in sorted(Path(profile_dir).iterdir()):
            if sub.is_dir() and (sub / PROFILE_JSON).is_file():
                refs = load_profile(sub)
                if refs:
                    profiles[sub.name] = refs
        return cls(profiles=profiles, speaker_map=load_speaker_map(profile_dir))

    def profile_name(self, label: str) -> Optional[str]:
        """speaker_map entry if there is one, otherwise a profile folder named exactly like the label."""
        label = (label or '').strip()
        if not label:
            return None
        mapped = self.speaker_map.get(label)
        if mapped is None:  # tolerate 'SPK0' / 'spk 0'
            norm = re.sub(r'\s+', '', label).lower()
            mapped = next((v for k, v in self.speaker_map.items() if re.sub(r'\s+', '', k).lower() == norm), None)
        if mapped is None:
            return label if label in self.profiles else None
        if mapped not in self.profiles:
            logger.warning(f'[voice_profile] speaker_map: {label} -> {mapped}, but no usable profile of that name')
            return None
        return mapped

    def ref_for(self, label: str, lang: str = '') -> Optional[ProfileRef]:
        name = self.profile_name(label)
        return select_ref(self.profiles[name], lang) if name else None


def resolve_clone_ref_mode(mode, profile_dir) -> str:
    """'line' (upstream: each line's own audio) or 'profile'. Unset mode = 'profile' if a profile dir is set, else 'line'."""
    mode = str(mode or '').strip().lower()
    profile_dir = str(profile_dir or '').strip()
    if mode not in ('line', 'profile'):
        mode = 'profile' if profile_dir else 'line'
    return 'profile' if mode == 'profile' and profile_dir else 'line'


def load_for_dubbing(settings) -> Optional[VoiceProfiles]:
    """Profiles to use as clone references, or None to keep the upstream per-line references."""
    profile_dir = str(settings.get('voice_profile_dir', '') or '').strip()
    if resolve_clone_ref_mode(settings.get('clone_ref_mode', ''), profile_dir) != 'profile':
        return None
    if not Path(profile_dir).is_dir():
        logger.warning(f'[voice_profile] voice_profile_dir is not a folder, using per-line references: {profile_dir}')
        return None
    vp = VoiceProfiles.load(profile_dir)
    if not vp.profiles:
        logger.warning(f'[voice_profile] no usable profile in {profile_dir}, using per-line references')
        return None
    logger.debug(f'[voice_profile] profiles={list(vp.profiles)} speaker_map={vp.speaker_map}')
    return vp


def choose_clone_ref(vprofiles: Optional[VoiceProfiles], speakers: List[str], index: int, lang: str = '') -> Optional[ProfileRef]:
    """
    Reference for subtitle line `index` (spoken in `lang`).
    None means: use that line's own audio (upstream behaviour) - no profiles, no speaker labels,
    or the speaker has no profile.
    """
    if not vprofiles or not 0 <= index < len(speakers):
        return None
    return vprofiles.ref_for(speakers[index], lang)


def stage_ref_wav(ref: ProfileRef, cache_folder: Union[str, Path]) -> Optional[str]:
    """
    Copy the profile clip into the task cache and return the copy's path (None if copying fails).
    Some engines trim the reference in place (e.g. GPT-SoVITS cuts it to 10 s); that must never touch the profile file.
    """
    dst = Path(cache_folder) / f'profile-ref-{hashlib.md5(ref.wav.encode("utf-8")).hexdigest()[:10]}{Path(ref.wav).suffix}'
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != os.path.getsize(ref.wav):
            shutil.copy2(ref.wav, dst)
    except OSError as e:
        logger.warning(f'[voice_profile] cannot copy {ref.wav}, using per-line reference: {e}')
        return None
    return dst.as_posix()


def clone_ref_sample_rate(settings) -> int:
    """Sampling rate (Hz) for per-line clone references cut from the source audio (setting `clone_ref_sample_rate`)."""
    try:
        rate = int(float(settings.get('clone_ref_sample_rate', DEFAULT_CLONE_REF_SAMPLE_RATE)))
    except (TypeError, ValueError):
        return DEFAULT_CLONE_REF_SAMPLE_RATE
    return rate if 8000 <= rate <= 192000 else DEFAULT_CLONE_REF_SAMPLE_RATE
