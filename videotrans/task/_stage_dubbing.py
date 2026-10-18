import copy
import re
import shutil
import time
from pathlib import Path

from videotrans.configure._paths import DUBBING_CACHE
from videotrans.configure.config import tr, app_cfg, settings, logger
from videotrans.configure.excepts import DubbingSrtError
from videotrans.tts import run as run_tts, SUPPORT_CLONE
from videotrans.util.help_misc import get_md5, vail_file
from videotrans.util import voice_profile, emotion_tts
from videotrans.util.help_srt import get_subtitle_from_srt, delete_punc


class DubbingMixin:

    def dubbing(self) -> None:
        _st=time.time()
        if self._exit() or self.cfg.app_mode == 'tiqu':
            return
        if self.should_dubbing:
            self.signal(text=tr('kaishipeiyin'))
        self.precent += 3
        self._tts()
        
        if  Path(self.cfg.source_sub).exists():
            logger.debug('配音结束后，移除原始字幕中所有标点')
            subs = get_subtitle_from_srt(self.cfg.source_sub)
            for it in subs:
                if self.cfg.fix_punc==2:
                    it['text']=delete_punc(it['text'])
                it['text']=it['text'].strip('...').strip('…').strip()
            self._save_srt_target(subs, self.cfg.source_sub)
        if self.should_dubbing:
            self.signal(text=tr('The dubbing is finished'))
            logger.debug(f'[语音合成阶段结束耗时]:{time.time()-_st}s')

    def _tts(self) -> None:
        if not self.should_dubbing:
            self.signal(text='Skip tts')
            return
        queue_tts = []
        subs = get_subtitle_from_srt(self.cfg.target_sub)
        source_subs = get_subtitle_from_srt(self.cfg.source_sub)
        if len(subs) < 1:
            raise DubbingSrtError(f"SRT file error:{self.cfg.target_sub}")
        try:
            rate = int(str(self.cfg.voice_rate).replace('%', ''))
        except (ValueError,TypeError):
            rate = 0

        rate = f"+{rate}%" if rate >= 0 else f"{rate}%"

        line_roles = app_cfg.line_roles
        voice_role = self.cfg.voice_role
        logger.debug(f'{line_roles=}')
        _lang=self.cfg.detect_language.split('-')[0]
        # clone reference from speaker profiles (None = upstream: each line's own audio); speaker.json is in subtitle line order
        vprofiles = voice_profile.load_for_dubbing(settings)
        speakers = voice_profile.read_speaker_labels(f'{self.cfg.cache_folder}/speaker.json') if vprofiles else []
        # per-line emotion for the TTS (None = upstream: no emotion); matched by the SOURCE line start time, never by line number
        emo_index = emotion_tts.load_index(settings, self.cfg.cache_folder, self.cfg.target_dir)
        for i, it in enumerate(subs):
            if it['end_time'] < it['start_time'] or not it['text'].strip():
                continue
            voice = line_roles.get(f'{it["line"]}', voice_role) if line_roles else voice_role

            _key = get_md5(f"{self.cfg.target_language_code}-{it['text']}-{voice}-{rate}-{self.cfg.volume}-{self.cfg.pitch}-{self.cfg.tts_type}-{settings.get('remove_dubb_silence', True)}-{settings.get('remove_dubb_all_silence', True)}-{settings.get('remove_dubb_silence_level', 'default')}")

            is_clone = str(voice).strip().lower() == 'clone' and self.cfg.tts_type in SUPPORT_CLONE
            ref = voice_profile.choose_clone_ref(vprofiles, speakers, i, _lang) if is_clone else None
            ref_wav = voice_profile.stage_ref_wav(ref, self.cfg.cache_folder) if ref else None
            if ref_wav:
                # add the chosen reference to the cache key, so dubbing made with another reference is not reused
                _key = get_md5(f"{_key}-{ref.cache_key()}")

            start_source = source_subs[i]['start_time'] if source_subs and i < len(source_subs) else it['start_time']
            emo = emotion_tts.find_line_emotion(emo_index, start_source)
            if emo and emotion_tts.engine_uses_emotion(self.cfg.tts_type):
                # emotion changes the audio of VoxCPM / ChatterBox only: other engines keep their cache keys
                _key = get_md5(f"{_key}-{emotion_tts.cache_suffix(emo)}")

            tmp_dict = {
                "text": it['text'],
                "line": it['line'],
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "startraw": it['startraw'],
                "endraw": it['endraw'],
                "ref_text": source_subs[i]['text'] if source_subs and i < len(source_subs) else '',
                "start_time_source": start_source,
                "end_time_source": source_subs[i]['end_time'] if source_subs and i < len(source_subs) else it[
                    'end_time'],
                "role": voice,
                "rate": rate,
                "volume": self.cfg.volume,
                "pitch": self.cfg.pitch,
                "tts_type": self.cfg.tts_type,
                "filename": f"{self.cfg.cache_folder}/{i}-{_key}.wav"
            }
            tmp_dict.update(emo)  # emotion, emotion_intensity, emotion_confidence (nothing for "no emotion")
            _dubbing_cache=f'{DUBBING_CACHE}/{_key}.wav'
            if vail_file(_dubbing_cache):
                # 直接使用缓存
                shutil.copy2(_dubbing_cache,tmp_dict['filename'])
            if is_clone:
                tmp_dict['ref_wav'] = ref_wav or f"{self.cfg.cache_folder}/clone-{i}.wav"
                tmp_dict['ref_language'] = _lang
                if ref_wav:
                    # profile reference: use its own transcript and language; no per-line clip is cut (ref_profile)
                    tmp_dict.update(ref_text=ref.text, ref_language=ref.lang or _lang, ref_profile=ref.profile)
            queue_tts.append(tmp_dict)

        if vprofiles:
            logger.debug(f'clone refs from speaker profiles: {len([d for d in queue_tts if d.get("ref_profile")])}/{len([d for d in queue_tts if d.get("ref_wav")])} lines')
        if emo_index is not None:
            logger.debug(f'emotion for TTS: {len([d for d in queue_tts if d.get("emotion")])}/{len(queue_tts)} lines (tts_type={self.cfg.tts_type}, engine uses it: {emotion_tts.engine_uses_emotion(self.cfg.tts_type)})')
        self.queue_tts = copy.deepcopy(queue_tts)

        if not self.queue_tts or len(self.queue_tts) < 1:
            raise RuntimeError(f'字幕长度为0，无法继续配音')

        if len([it.get("ref_wav") for it in self.queue_tts if it.get("ref_wav") and not it.get("ref_profile")]) > 0:
            self._create_ref_from_vocal()

        run_tts(
            queue_tts=self.queue_tts,
            language=self.cfg.target_language_code,
            uuid=self.uuid,
            tts_type=self.cfg.tts_type,
            is_cuda=self.cfg.is_cuda
        )
        outname=None
        if settings.get('save_segment_audio', False):
            outname = self.cfg.target_dir + f'/segment_audio_{self.cfg.noextname}'
            Path(outname).mkdir(parents=True, exist_ok=True)
        for it in self.queue_tts:
            it['text']=it['text'].strip('...').strip('…').strip()
            if self.cfg.fix_punc==2:
                it['text']=delete_punc(it['text'])
            if Path(it['filename']).exists():
                # 保存缓存
                shutil.copy2(it['filename'],f'{DUBBING_CACHE}/'+Path(it['filename']).name.split('-')[-1])
                if outname:
                    text = re.sub(r'["\'*?\\/|:<>\r\n\t]+', '', it['text'], flags=re.I | re.S)
                    name = f'{outname}/{it["line"]}-{text[:60]}.wav'
                    shutil.copy2(it['filename'], name)
        
        self._save_srt_target(self.queue_tts, self.cfg.target_sub)