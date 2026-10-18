"""
Wiring tests for the voice-clone reference choice:
  cut_from_audio(sample_rate) / RecognMixin._create_ref_from_vocal / DubbingMixin._tts / AppSettings defaults.
ffmpeg, the TTS engines and the diarization are mocked; only the choice of ref_wav / ref_text / ref_language
and the dubbing-cache key are checked.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from videotrans.tts import SUPPORT_CLONE


class FakeSettings(dict):
    """dict with the .get(key, default) of AppSettings."""


def make_profile(root: Path, name: str, text: str, wav_bytes: bytes = b'RIFFfake', lang='ja') -> None:
    folder = root / name / 'refs'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'r.wav').write_bytes(wav_bytes)
    (root / name / 'profile.json').write_text(json.dumps({'refs': [
        {'wav': 'refs/r.wav', 'text': text, 'lang': lang, 'sec': 12.0, 'snr_db': 30.0}]}, ensure_ascii=False), encoding='utf-8')


# ---------------------------------------------------------------- cut_from_audio
class TestCutFromAudio:
    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        from videotrans.util import _ffmpeg_audio
        calls = []
        monkeypatch.setattr(_ffmpeg_audio, 'runffmpeg', lambda cmd, *a, **kw: calls.append(cmd) or True)
        src = tmp_path / 'in.wav'
        src.write_bytes(b'x')

        def _run(**kw):
            _ffmpeg_audio.cut_from_audio(ss='00:00:01,000', to='00:00:02,000', audio_file=str(src), out_file=str(tmp_path / 'o' / 'out.wav'), **kw)
            return calls[-1]

        return _run

    def test_default_is_16k_as_upstream(self, run):
        cmd = run()
        assert cmd[cmd.index('-ar') + 1] == '16000'

    def test_sample_rate_is_configurable(self, run):
        cmd = run(sample_rate=44100)
        assert cmd[cmd.index('-ar') + 1] == '44100'
        assert cmd.count('-ar') == 1

    def test_still_exported_from_help_ffmpeg(self):
        import inspect
        from videotrans.util.help_ffmpeg import cut_from_audio
        assert inspect.signature(cut_from_audio).parameters['sample_rate'].default == 16000


# ---------------------------------------------------------------- _create_ref_from_vocal
class TestCreateRefFromVocal:
    def _mixin(self, tmp_path):
        from videotrans.task._stage_recogn import RecognMixin
        m = RecognMixin()
        vocal = tmp_path / 'vocal.wav'
        vocal.write_bytes(b'x')
        m.clone_ref = str(vocal)
        m.cfg = SimpleNamespace(source_wav=str(vocal), cache_folder=str(tmp_path), name=str(vocal))
        return m, vocal

    def test_cuts_only_line_refs_at_configured_rate(self, monkeypatch, tmp_path):
        from videotrans.task import _stage_recogn
        cut = MagicMock(return_value=True)
        monkeypatch.setattr(_stage_recogn, 'cut_from_audio', cut)
        monkeypatch.setattr(_stage_recogn, 'settings', FakeSettings(clone_ref_sample_rate=48000))
        m, vocal = self._mixin(tmp_path)
        m.queue_tts = [
            {'ref_wav': str(tmp_path / 'profile-ref-1.wav'), 'ref_profile': 'Aさん', 'startraw': '00:00:00,000', 'endraw': '00:00:01,000'},
            {'ref_wav': str(tmp_path / 'clone-1.wav'), 'startraw': '00:00:01,000', 'endraw': '00:00:03,000'},
            {'startraw': '00:00:03,000', 'endraw': '00:00:04,000'},  # not a clone line
        ]
        m._create_ref_from_vocal()
        assert cut.call_count == 1
        kw = cut.call_args.kwargs
        assert kw['out_file'] == str(tmp_path / 'clone-1.wav')
        assert kw['sample_rate'] == 48000
        assert kw['audio_file'] == str(vocal)

    def test_default_rate_is_44100(self, monkeypatch, tmp_path):
        from videotrans.task import _stage_recogn
        cut = MagicMock(return_value=True)
        monkeypatch.setattr(_stage_recogn, 'cut_from_audio', cut)
        monkeypatch.setattr(_stage_recogn, 'settings', FakeSettings())
        m, _ = self._mixin(tmp_path)
        m.queue_tts = [{'ref_wav': str(tmp_path / 'clone-0.wav'), 'startraw': '00:00:00,000', 'endraw': '00:00:01,000'}]
        m._create_ref_from_vocal()
        assert cut.call_args.kwargs['sample_rate'] == 44100


# ---------------------------------------------------------------- DubbingMixin._tts
SRC = [
    {'line': 1, 'start_time': 0, 'end_time': 2000, 'startraw': '00:00:00,000', 'endraw': '00:00:02,000', 'text': 'こんにちは'},
    {'line': 2, 'start_time': 2000, 'end_time': 4000, 'startraw': '00:00:02,000', 'endraw': '00:00:04,000', 'text': 'はい'},
    {'line': 3, 'start_time': 4000, 'end_time': 6000, 'startraw': '00:00:04,000', 'endraw': '00:00:06,000', 'text': 'そうですね'},
]
TGT = [dict(it, text=t) for it, t in zip(SRC, ['Hello', 'Yes', 'I see'])]


class TestDubbingRefChoice:
    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        from videotrans.task import _stage_dubbing
        from videotrans.configure.config import app_cfg

        cache = tmp_path / 'cache'
        cache.mkdir()
        dubbing_cache = tmp_path / 'dubbing_cache'
        dubbing_cache.mkdir()
        profiles = tmp_path / 'profiles'
        profiles.mkdir()

        settings = FakeSettings(voice_profile_dir='', clone_ref_mode='', clone_ref_sample_rate=44100)
        srt = {'tgt.srt': [dict(d) for d in TGT], 'src.srt': [dict(d) for d in SRC]}
        monkeypatch.setattr(_stage_dubbing, 'settings', settings)
        monkeypatch.setattr(_stage_dubbing, 'DUBBING_CACHE', str(dubbing_cache))
        monkeypatch.setattr(_stage_dubbing, 'get_subtitle_from_srt', lambda p, *a, **kw: [dict(d) for d in srt[Path(p).name]])
        monkeypatch.setattr(app_cfg, 'line_roles', {})
        run_tts = MagicMock()
        monkeypatch.setattr(_stage_dubbing, 'run_tts', run_tts)

        m = _stage_dubbing.DubbingMixin()
        m.should_dubbing = True
        m.uuid = 'u'
        m.cfg = SimpleNamespace(
            target_sub=str(tmp_path / 'tgt.srt'), source_sub=str(tmp_path / 'src.srt'), voice_rate='0', voice_role='clone',
            detect_language='ja', target_language_code='en', volume='+0%', pitch='+0Hz', tts_type=SUPPORT_CLONE[0],
            is_cuda=False, cache_folder=str(cache), fix_punc=0, target_dir=str(tmp_path), noextname='x')
        m._save_srt_target = MagicMock()
        m._create_ref_from_vocal = MagicMock()
        return SimpleNamespace(m=m, settings=settings, cache=cache, dubbing_cache=dubbing_cache, profiles=profiles,
                               run_tts=run_tts, tmp=tmp_path, mod=_stage_dubbing)

    def _speakers(self, env, labels):
        (env.cache / 'speaker.json').write_text(json.dumps(labels), encoding='utf-8')

    def _queue(self, env):
        env.m._tts()
        return env.m.queue_tts

    def test_upstream_behaviour_when_profiles_are_not_configured(self, env):
        self._speakers(env, ['spk0', 'spk1', 'spk0'])
        q = self._queue(env)
        assert [it['ref_wav'] for it in q] == [f"{env.cache}/clone-{i}.wav" for i in range(3)]
        assert [it['ref_text'] for it in q] == ['こんにちは', 'はい', 'そうですね']
        assert all(it['ref_language'] == 'ja' and 'ref_profile' not in it for it in q)
        env.m._create_ref_from_vocal.assert_called_once()

    def test_profile_refs_replace_line_refs(self, env):
        make_profile(env.profiles, 'Aさん', 'Aの手本です')
        make_profile(env.profiles, 'Bさん', 'Bの手本です', lang='en')
        (env.profiles / 'speaker_map.json').write_text(json.dumps({'spk0': 'Aさん', 'spk1': 'Bさん'}), encoding='utf-8')
        env.settings['voice_profile_dir'] = str(env.profiles)
        self._speakers(env, ['spk0', 'spk1', 'spk0'])
        q = self._queue(env)

        assert [it['ref_text'] for it in q] == ['Aの手本です', 'Bの手本です', 'Aの手本です']
        assert [it['ref_language'] for it in q] == ['ja', 'en', 'ja']
        assert [it['ref_profile'] for it in q] == ['Aさん', 'Bさん', 'Aさん']
        # the engines get a copy in the task cache, never the profile file itself
        for it in q:
            assert Path(it['ref_wav']).parent == env.cache and Path(it['ref_wav']).is_file()
            assert 'clone-' not in it['ref_wav']
        assert q[0]['ref_wav'] == q[2]['ref_wav'] != q[1]['ref_wav']
        # no line needs a clip cut from the source audio
        env.m._create_ref_from_vocal.assert_not_called()

    def test_mode_line_keeps_upstream_even_with_a_profile_dir(self, env):
        make_profile(env.profiles, 'spk0', 'x')
        env.settings.update(voice_profile_dir=str(env.profiles), clone_ref_mode='line')
        self._speakers(env, ['spk0', 'spk0', 'spk0'])
        q = self._queue(env)
        assert all(it['ref_wav'].endswith(f'clone-{i}.wav') for i, it in enumerate(q))
        assert [it['ref_text'] for it in q] == ['こんにちは', 'はい', 'そうですね']

    def test_speakers_without_profile_or_label_fall_back_to_line_refs(self, env):
        make_profile(env.profiles, 'Aさん', 'Aの手本です')
        (env.profiles / 'speaker_map.json').write_text(json.dumps({'spk0': 'Aさん'}), encoding='utf-8')
        env.settings['voice_profile_dir'] = str(env.profiles)
        self._speakers(env, ['spk0', 'spk1'])  # line 3 has no label, spk1 has no profile
        q = self._queue(env)

        assert q[0]['ref_profile'] == 'Aさん' and q[0]['ref_text'] == 'Aの手本です'
        for i in (1, 2):
            assert 'ref_profile' not in q[i]
            assert q[i]['ref_wav'] == f"{env.cache}/clone-{i}.wav"
            assert q[i]['ref_text'] == SRC[i]['text']
            assert q[i]['ref_language'] == 'ja'
        env.m._create_ref_from_vocal.assert_called_once()  # still needed for the fallback lines

    def test_no_speaker_json_falls_back_to_line_refs(self, env):
        make_profile(env.profiles, 'spk0', 'x')
        env.settings['voice_profile_dir'] = str(env.profiles)
        q = self._queue(env)  # no speaker.json written
        assert all('ref_profile' not in it for it in q)
        assert all(it['ref_wav'].endswith(f'clone-{i}.wav') for i, it in enumerate(q))

    def test_non_clone_voice_gets_no_reference(self, env):
        make_profile(env.profiles, 'spk0', 'x')
        env.settings['voice_profile_dir'] = str(env.profiles)
        env.m.cfg.voice_role = 'some-voice'
        self._speakers(env, ['spk0'] * 3)
        q = self._queue(env)
        assert all('ref_wav' not in it and 'ref_profile' not in it for it in q)
        env.m._create_ref_from_vocal.assert_not_called()

    def test_cache_key_contains_the_chosen_reference(self, env):
        self._speakers(env, ['spk0'] * 3)
        make_profile(env.profiles, 'spk0', 'v1', wav_bytes=b'RIFF-old')
        key = lambda q: Path(q[0]['filename']).name.split('-', 1)[1]  # '{i}-{key}.wav'

        line_key = key(self._queue(env))  # upstream: line reference
        env.settings['voice_profile_dir'] = str(env.profiles)
        old_ref_key = key(self._queue(env))
        assert old_ref_key != line_key

        # replace the clip with another recording: the key must change again
        (env.profiles / 'spk0' / 'refs' / 'r.wav').write_bytes(b'RIFF-new-and-longer')
        new_ref_key = key(self._queue(env))
        assert new_ref_key not in (line_key, old_ref_key)

        # a different profile (different path) also gives a different key
        make_profile(env.profiles, 'other', 'v1', wav_bytes=b'RIFF-new-and-longer')
        (env.profiles / 'speaker_map.json').write_text(json.dumps({'spk0': 'other'}), encoding='utf-8')
        assert key(self._queue(env)) != new_ref_key

    def test_cached_dubbing_made_with_the_line_reference_is_not_reused(self, env):
        self._speakers(env, ['spk0'] * 3)
        q = self._queue(env)
        stale = env.dubbing_cache / (Path(q[0]['filename']).name.split('-', 1)[1])
        stale.write_bytes(b'dubbed with the old line reference')

        q = self._queue(env)  # same settings: cache hit as in upstream
        assert Path(q[0]['filename']).read_bytes() == b'dubbed with the old line reference'

        make_profile(env.profiles, 'spk0', 'x')
        env.settings['voice_profile_dir'] = str(env.profiles)
        q = self._queue(env)  # profile reference now: the stale file must not be picked up
        assert not Path(q[0]['filename']).exists()

    def test_line_mode_cache_key_is_unchanged_from_upstream(self, env):
        from videotrans.util.help_misc import get_md5
        self._speakers(env, ['spk0'] * 3)
        q = self._queue(env)
        s = env.settings
        expected = get_md5(f"en-Hello-clone-+0%-+0%-+0Hz-{env.m.cfg.tts_type}-{s.get('remove_dubb_silence', True)}-"
                           f"{s.get('remove_dubb_all_silence', True)}-{s.get('remove_dubb_silence_level', 'default')}")
        assert q[0]['filename'] == f"{env.cache}/0-{expected}.wav"


# ---------------------------------------------------------------- AppSettings
class TestAppSettings:
    @pytest.fixture
    def make(self, tmp_path):
        from videotrans.configure._app_settings import AppSettings

        def _make(data=None):
            p = tmp_path / 'cfg.json'
            if data is not None:
                p.write_text(json.dumps(data), encoding='utf-8')
            return AppSettings(_json_path=str(p))

        return _make

    def test_new_defaults(self, make):
        s = make()
        assert s.get('voice_profile_dir') == ''
        assert s.get('clone_ref_mode') == ''
        assert int(s.get('clone_ref_sample_rate')) == 44100
        assert s.get('max_audio_speed_rate') == 50  # upstream default is unchanged

    @pytest.mark.parametrize('raw', [1.3, '1.3'])
    def test_recommended_max_audio_speed_rate_is_not_truncated_to_1(self, make, raw):
        assert make({'max_audio_speed_rate': raw}).get('max_audio_speed_rate') == pytest.approx(1.3)

    def test_integer_speed_rates_still_work(self, make):
        assert make({'max_audio_speed_rate': 20}).get('max_audio_speed_rate') == 20
        assert float(make({'max_audio_speed_rate': '3'}).get('max_audio_speed_rate', 50)) == 3.0
