"""
EMOTION module part 2 (E4): the line's emotion reaches the TTS.

  util/emotion_tts.py           pure mapping (label/intensity -> VoxCPM instruction, ChatterBox parameters, lookup by start time)
  task/_stage_dubbing.py        queue item gets emotion / emotion_intensity / emotion_confidence; cache key changes only with emotion
  tts/_voxcpm.py                request parameters per line (Gradio client mocked)
  tts/_chatterbox.py            exaggeration / cfg_weight per line (model mocked)
  configure/_app_settings.py    new defaults

No model, network or ffmpeg is used. gradio_client / soundfile / chatterbox are replaced by stubs (they are not test dependencies).
"""
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from videotrans.util import emotion as emo
from videotrans.util import emotion_tts as et
from videotrans.util.help_misc import get_md5


class FakeSettings(dict):
    """dict with the .get(key, default) of AppSettings."""


def entry(start, label='angry', intensity=0.5, confidence=0.6, end=None):
    return emo.EmotionEntry(start, end or start + 800, label, intensity, confidence, emo.SOURCE_SENSEVOICE)


# ---------------------------------------------------------------- mapping (pure functions)
class TestIntensityWord:
    @pytest.mark.parametrize('value, word', [
        (0.0, 'slightly'), (0.1, 'slightly'), (0.39, 'slightly'),
        (0.4, ''), (0.5, ''), (0.7, ''),
        (0.71, 'very'), (0.9, 'very'), (1.0, 'very'),
    ])
    def test_levels(self, value, word):
        assert et.intensity_word(value) == word

    @pytest.mark.parametrize('value', [None, '', 'abc'])
    def test_broken_value_is_the_middle_level(self, value):
        assert et.intensity_word(value) == ''

    def test_out_of_range_is_clipped(self):
        assert et.intensity_word(-3) == 'slightly'
        assert et.intensity_word(7) == 'very'


class TestVoxcpmInstruction:
    @pytest.mark.parametrize('label, style', [
        ('angry', 'angry, tense'), ('happy', 'cheerful'), ('sad', 'sad, subdued'),
        ('fearful', 'anxious'), ('surprised', 'surprised'), ('disgusted', 'disgusted, displeased'),
    ])
    def test_label_styles(self, label, style):
        assert et.voxcpm_instruction(label, 0.5) == style
        assert et.voxcpm_instruction(label, 0.1) == f'slightly {style}'
        assert et.voxcpm_instruction(label, 0.9) == f'very {style}'

    def test_every_emotion_label_has_a_style(self):
        assert set(et.VOXCPM_STYLES) == set(emo.LABELS) - {'neutral'}

    @pytest.mark.parametrize('label', ['neutral', 'unknown', '', None])
    def test_no_instruction_for_neutral_or_unknown(self, label):
        assert et.voxcpm_instruction(label, 0.9) == ''

    def test_label_is_case_insensitive(self):
        assert et.voxcpm_instruction(' ANGRY ', 0.5) == 'angry, tense'

    def test_parentheses_are_removed(self):
        assert et.sanitize_instruction('(very  angry), tense') == 'very angry, tense'
        assert et.sanitize_instruction('（cheerful）') == 'cheerful'


class TestItemEmotion:
    def test_usable(self):
        assert et.item_emotion({'emotion': 'happy', 'emotion_intensity': 0.3, 'emotion_confidence': 0.6}) == ('happy', 0.3)

    @pytest.mark.parametrize('item', [
        {},
        {'emotion': 'neutral', 'emotion_intensity': 0.9, 'emotion_confidence': 0.9},
        {'emotion': 'angry', 'emotion_intensity': 0.9, 'emotion_confidence': 0.49},
        {'emotion': 'bored', 'emotion_intensity': 0.9, 'emotion_confidence': 0.9},
        {'emotion': None},
    ])
    def test_not_usable(self, item):
        assert et.item_emotion(item) is None
        assert et.voxcpm_control_instruction(item) == ''

    def test_confidence_exactly_half_is_usable(self):
        assert et.item_emotion({'emotion': 'sad', 'emotion_confidence': 0.5}) == ('sad', 0.5)

    def test_missing_intensity_is_the_middle_level(self):
        assert et.voxcpm_control_instruction({'emotion': 'angry'}) == 'angry, tense'


class TestChatterboxParams:
    def test_no_emotion_keeps_the_user_values(self):
        assert et.chatterbox_params({}, 0.5, 0.5, 'en') == (0.5, 0.5)
        assert et.chatterbox_params({'emotion': 'neutral', 'emotion_intensity': 1.0}, 0.6, 0.4, 'en') == (0.6, 0.4)

    @pytest.mark.parametrize('base, intensity, expected', [
        (0.5, 0.5, 0.7), (0.5, 1.0, 0.9), (0.5, 0.0, 0.5), (0.3, 0.5, 0.5), (0.7, 1.0, 0.9), (0.8, 0.5, 0.9),
    ])
    def test_exaggeration_of_an_emotional_line(self, base, intensity, expected):
        item = {'emotion': 'angry', 'emotion_intensity': intensity, 'emotion_confidence': 0.6}
        assert et.chatterbox_params(item, base, 0.5, 'en')[0] == pytest.approx(expected)

    def test_exaggeration_never_goes_below_the_users_value(self):
        item = {'emotion': 'angry', 'emotion_intensity': 1.0}
        assert et.chatterbox_params(item, 1.2, 0.5, 'en')[0] == pytest.approx(1.2)

    def test_cfg_weight_is_zero_when_the_languages_differ(self):
        assert et.chatterbox_params({'ref_language': 'ja'}, 0.5, 0.5, 'en')[1] == 0.0

    @pytest.mark.parametrize('ref, target', [('ja', 'ja'), ('zh-cn', 'zh'), ('ZH', 'zh-tw'), ('en_US', 'en')])
    def test_same_language_keeps_cfg_weight(self, ref, target):
        assert et.chatterbox_params({'ref_language': ref}, 0.5, 0.35, target)[1] == 0.35

    def test_unknown_reference_language_keeps_cfg_weight(self):
        assert et.chatterbox_params({}, 0.5, 0.5, 'en')[1] == 0.5
        assert et.chatterbox_params({'ref_language': ''}, 0.5, 0.5, 'en')[1] == 0.5

    def test_setting_off_keeps_cfg_weight(self):
        assert et.chatterbox_params({'ref_language': 'ja'}, 0.5, 0.5, 'en', crosslingual_cfg0=False)[1] == 0.5

    def test_built_in_voice_keeps_cfg_weight(self):
        assert et.chatterbox_params({'ref_language': 'ja'}, 0.5, 0.5, 'en', has_ref=False)[1] == 0.5

    def test_string_values_from_the_settings_are_accepted(self):
        assert et.chatterbox_params({}, '0.5', '0.5', 'en') == (0.5, 0.5)


class TestLookupByStartTime:
    ENTRIES = [entry(0, 'neutral', 0.0, 0.6), entry(1000, 'angry', 0.8), entry(2000, 'happy', 0.3, 0.4), entry(3000, 'sad', 0.5, 0.5)]

    def idx(self):
        return emo.EmotionIndex(self.ENTRIES)

    def test_match_within_300ms(self):
        assert et.find_line_emotion(self.idx(), 1000) == {'emotion': 'angry', 'emotion_intensity': 0.8, 'emotion_confidence': 0.6}
        assert et.find_line_emotion(self.idx(), 1290)['emotion'] == 'angry'
        assert et.find_line_emotion(self.idx(), 710)['emotion'] == 'angry'

    def test_no_match_beyond_300ms(self):
        assert et.find_line_emotion(self.idx(), 1301) == {}
        assert et.find_line_emotion(self.idx(), 5000) == {}

    def test_neutral_low_confidence_and_missing_are_no_emotion(self):
        assert et.find_line_emotion(self.idx(), 0) == {}      # neutral
        assert et.find_line_emotion(self.idx(), 2000) == {}   # confidence 0.4
        assert et.find_line_emotion(None, 1000) == {}
        assert et.find_line_emotion(emo.EmotionIndex([]), 1000) == {}

    def test_confidence_exactly_half_is_used(self):
        assert et.find_line_emotion(self.idx(), 3000)['emotion'] == 'sad'

    def test_unknown_label_is_rounded_to_neutral_by_the_loader(self):
        idx = emo.EmotionIndex([{'start_time': 0, 'label': 'bored', 'intensity': 1, 'confidence': 1}])
        assert et.find_line_emotion(idx, 0) == {}


class TestCacheSuffixAndEngines:
    def test_suffix_has_label_and_intensity_not_confidence(self):
        a = {'emotion': 'angry', 'emotion_intensity': 0.5, 'emotion_confidence': 0.6}
        assert et.cache_suffix(a) == 'emo-angry-0.50'
        assert et.cache_suffix(dict(a, emotion_confidence=0.9)) == et.cache_suffix(a)
        assert et.cache_suffix(dict(a, emotion_intensity=0.8)) != et.cache_suffix(a)
        assert et.cache_suffix(dict(a, emotion='happy')) != et.cache_suffix(a)

    def test_only_voxcpm_and_chatterbox_carry_emotion(self):
        from videotrans.tts import _constants as c
        assert et.emotion_engine_types() == {c.VOXCPM_TTS, c.CHATTERBOX_TTS}
        assert et.engine_uses_emotion(c.VOXCPM_TTS) and et.engine_uses_emotion(c.CHATTERBOX_TTS)
        for other in (c.COSYVOICE_TTS, c.EDGE_TTS, c.GPTSOVITS_TTS, c.INDEX_TTS, c.OPENAI_TTS):
            assert not et.engine_uses_emotion(other)


class TestLoadIndex:
    def write(self, folder, entries):
        Path(folder).mkdir(parents=True, exist_ok=True)
        emo.save_emotion(entries, str(folder))

    def test_off_by_default_and_when_either_switch_is_off(self, tmp_path):
        self.write(tmp_path / 'c', [entry(0)])
        assert et.load_index(FakeSettings(), str(tmp_path / 'c')) is None
        assert et.load_index(FakeSettings(enable_emotion=False), str(tmp_path / 'c')) is None
        assert et.load_index(FakeSettings(enable_emotion=True, emotion_tts=False), str(tmp_path / 'c')) is None
        assert len(et.load_index(FakeSettings(enable_emotion=True), str(tmp_path / 'c'))) == 1  # emotion_tts defaults to True

    def test_cache_folder_first_then_output_folder(self, tmp_path):
        self.write(tmp_path / 'out', [entry(0, 'sad'), entry(1000, 'sad')])
        on = FakeSettings(enable_emotion=True)
        assert len(et.load_index(on, str(tmp_path / 'cache'), str(tmp_path / 'out'))) == 2   # cache folder was cleaned
        self.write(tmp_path / 'cache', [entry(0, 'happy')])
        assert len(et.load_index(on, str(tmp_path / 'cache'), str(tmp_path / 'out'))) == 1

    def test_missing_or_broken_file_is_no_index(self, tmp_path):
        on = FakeSettings(enable_emotion=True)
        assert et.load_index(on, str(tmp_path), None) is None
        (tmp_path / emo.EMOTION_FILE).write_text('not json', encoding='utf-8')
        assert et.load_index(on, str(tmp_path), None) is None
        assert et.load_index(on, None, None) is None


# ---------------------------------------------------------------- DubbingMixin._tts
SRC = [
    {'line': 1, 'start_time': 0, 'end_time': 2000, 'startraw': '00:00:00,000', 'endraw': '00:00:02,000', 'text': 'こんにちは'},
    {'line': 2, 'start_time': 2000, 'end_time': 4000, 'startraw': '00:00:02,000', 'endraw': '00:00:04,000', 'text': 'ふざけるな'},
    {'line': 3, 'start_time': 4000, 'end_time': 6000, 'startraw': '00:00:04,000', 'endraw': '00:00:06,000', 'text': 'やった'},
]
TGT = [dict(it, text=t) for it, t in zip(SRC, ['Hello', 'Stop messing around', 'We did it'])]


class TestDubbingEmotion:
    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        from videotrans.task import _stage_dubbing
        from videotrans.configure.config import app_cfg
        from videotrans.tts import VOXCPM_TTS

        cache = tmp_path / 'cache'
        cache.mkdir()
        out = tmp_path / 'out'
        out.mkdir()
        dubbing_cache = tmp_path / 'dubbing_cache'
        dubbing_cache.mkdir()

        settings = FakeSettings(voice_profile_dir='', clone_ref_mode='', clone_ref_sample_rate=44100,
                                enable_emotion=True, emotion_backend='sensevoice')
        srt = {'tgt.srt': [dict(d) for d in TGT], 'src.srt': [dict(d) for d in SRC]}
        monkeypatch.setattr(_stage_dubbing, 'settings', settings)
        monkeypatch.setattr(_stage_dubbing, 'DUBBING_CACHE', str(dubbing_cache))
        monkeypatch.setattr(_stage_dubbing, 'get_subtitle_from_srt', lambda p, *a, **kw: [dict(d) for d in srt[Path(p).name]])
        monkeypatch.setattr(app_cfg, 'line_roles', {})
        monkeypatch.setattr(_stage_dubbing, 'run_tts', MagicMock())

        m = _stage_dubbing.DubbingMixin()
        m.should_dubbing = True
        m.uuid = 'u'
        m.cfg = SimpleNamespace(
            target_sub=str(tmp_path / 'tgt.srt'), source_sub=str(tmp_path / 'src.srt'), voice_rate='0', voice_role='clone',
            detect_language='ja', target_language_code='en', volume='+0%', pitch='+0Hz', tts_type=VOXCPM_TTS,
            is_cuda=False, cache_folder=str(cache), fix_punc=0, target_dir=str(out), noextname='x')
        m._save_srt_target = MagicMock()
        m._create_ref_from_vocal = MagicMock()
        return SimpleNamespace(m=m, settings=settings, cache=cache, out=out, srt=srt, tmp=tmp_path)

    def queue(self, env):
        env.m._tts()
        return env.m.queue_tts

    @staticmethod
    def key_of(item):
        return Path(item['filename']).name.split('-', 1)[1]  # '{i}-{key}.wav'

    def write(self, env, entries, folder=None):
        emo.save_emotion(entries, str(folder or env.cache))

    ENTRIES = [entry(0, 'neutral', 0.0, 0.6), entry(2000, 'angry', 0.8), entry(4000, 'happy', 0.3)]

    def test_items_get_the_emotion_of_their_line(self, env):
        self.write(env, self.ENTRIES)
        q = self.queue(env)
        assert 'emotion' not in q[0]  # neutral = no emotion
        assert (q[1]['emotion'], q[1]['emotion_intensity'], q[1]['emotion_confidence']) == ('angry', 0.8, 0.6)
        assert (q[2]['emotion'], q[2]['emotion_intensity'], q[2]['emotion_confidence']) == ('happy', 0.3, 0.6)

    def test_emotion_is_found_by_source_start_time_not_by_index(self, env):
        # emotion.json holds only 2 of the 3 lines and is not aligned by index: entry #0 belongs to line 3
        self.write(env, [entry(4000, 'sad', 0.5), entry(2000, 'angry', 0.9)])
        q = self.queue(env)
        assert 'emotion' not in q[0]
        assert q[1]['emotion'] == 'angry'
        assert q[2]['emotion'] == 'sad'

    def test_source_time_is_used_even_when_the_translation_times_differ(self, env):
        # the translated SRT was re-timed; the emotion still follows the SOURCE start time
        env.srt['tgt.srt'] = [dict(d, start_time=d['start_time'] + 5000, end_time=d['end_time'] + 5000) for d in env.srt['tgt.srt']]
        self.write(env, self.ENTRIES)
        q = self.queue(env)
        assert [it['start_time'] for it in q] == [5000, 7000, 9000]
        assert [it['start_time_source'] for it in q] == [0, 2000, 4000]
        assert q[1]['emotion'] == 'angry' and q[2]['emotion'] == 'happy'

    def test_tolerance_300ms(self, env):
        self.write(env, [entry(2250, 'angry', 0.8), entry(4400, 'happy', 0.8)])
        q = self.queue(env)
        assert q[1]['emotion'] == 'angry'
        assert 'emotion' not in q[2]

    def test_low_confidence_is_no_emotion(self, env):
        self.write(env, [entry(2000, 'angry', 0.8, confidence=0.49)])
        assert 'emotion' not in self.queue(env)[1]

    def test_missing_file_is_upstream(self, env):
        q = self.queue(env)
        assert all('emotion' not in it for it in q)

    def test_output_folder_copy_is_used_when_the_cache_folder_has_none(self, env):
        self.write(env, self.ENTRIES, folder=env.out)
        assert self.queue(env)[1]['emotion'] == 'angry'

    @pytest.mark.parametrize('off', [dict(enable_emotion=False), dict(emotion_tts=False)])
    def test_switches_off_is_upstream(self, env, off):
        self.write(env, self.ENTRIES)
        env.settings.update(off)
        q = self.queue(env)
        assert all('emotion' not in it for it in q)
        assert all(not k.startswith('emotion') for it in q for k in it)

    def test_cache_key_is_unchanged_without_emotion(self, env):
        s = env.settings
        upstream = lambda text: get_md5(f"en-{text}-clone-+0%-+0%-+0Hz-{env.m.cfg.tts_type}-{s.get('remove_dubb_silence', True)}-"
                                         f"{s.get('remove_dubb_all_silence', True)}-{s.get('remove_dubb_silence_level', 'default')}")
        self.write(env, [entry(2000, 'neutral', 0.0, 0.6), entry(4000, 'happy', 0.3, confidence=0.3)])  # nothing usable
        q = self.queue(env)
        assert [self.key_of(it) for it in q] == [f'{upstream(t)}.wav' for t in ('Hello', 'Stop messing around', 'We did it')]

    def test_cache_key_changes_only_for_lines_with_emotion(self, env):
        before = [self.key_of(it) for it in self.queue(env)]  # no emotion.json
        self.write(env, self.ENTRIES)
        after = [self.key_of(it) for it in self.queue(env)]
        assert after[0] == before[0]
        assert after[1] != before[1] and after[2] != before[2]
        assert after[1] != after[2]

    def test_cache_key_follows_label_and_intensity(self, env):
        self.write(env, [entry(2000, 'angry', 0.8)])
        k_angry = self.key_of(self.queue(env)[1])
        self.write(env, [entry(2000, 'angry', 0.3)])
        k_weak = self.key_of(self.queue(env)[1])
        self.write(env, [entry(2000, 'sad', 0.8)])
        k_sad = self.key_of(self.queue(env)[1])
        self.write(env, [entry(2000, 'angry', 0.8, confidence=0.9)])
        k_conf = self.key_of(self.queue(env)[1])
        assert len({k_angry, k_weak, k_sad}) == 3
        assert k_conf == k_angry  # confidence is a gate only

    def test_dubbing_made_without_emotion_is_not_reused(self, env, monkeypatch):
        from videotrans.task import _stage_dubbing
        q = self.queue(env)
        stale = Path(_stage_dubbing.DUBBING_CACHE) / self.key_of(q[1])
        stale.write_bytes(b'dubbed without emotion')
        assert Path(self.queue(env)[1]['filename']).read_bytes() == b'dubbed without emotion'  # same settings: cache hit
        self.write(env, self.ENTRIES)
        assert not Path(self.queue(env)[1]['filename']).exists()

    @pytest.mark.parametrize('engine', ['COSYVOICE_TTS', 'GPTSOVITS_TTS', 'F5_TTS', 'INDEX_TTS'])
    def test_engines_without_emotion_support_keep_their_cache_keys(self, env, engine):
        from videotrans.tts import _constants
        env.m.cfg.tts_type = getattr(_constants, engine)
        before = [self.key_of(it) for it in self.queue(env)]
        self.write(env, self.ENTRIES)
        q = self.queue(env)
        assert [self.key_of(it) for it in q] == before
        assert q[1]['emotion'] == 'angry'  # the data is still there (harmless; e.g. for later reports)

    def test_chatterbox_also_carries_emotion(self, env):
        from videotrans.tts import CHATTERBOX_TTS
        env.m.cfg.tts_type = CHATTERBOX_TTS
        before = [self.key_of(it) for it in self.queue(env)]
        self.write(env, self.ENTRIES)
        after = [self.key_of(it) for it in self.queue(env)]
        assert after[0] == before[0] and after[1] != before[1]

    def test_does_not_disturb_the_other_fields(self, env):
        self.write(env, self.ENTRIES)
        q = self.queue(env)
        assert [it['text'] for it in q] == ['Hello', 'Stop messing around', 'We did it']
        assert q[1]['ref_text'] == 'ふざけるな' and q[1]['ref_language'] == 'ja'
        assert q[1]['filename'].startswith(f"{env.cache}/1-")


# ---------------------------------------------------------------- engines (stubs for the heavy packages)
@pytest.fixture
def heavy_stubs(monkeypatch):
    """gradio_client / soundfile / chatterbox as stubs; the TTS modules are imported fresh with them and dropped afterwards."""
    gradio_client = types.ModuleType('gradio_client')
    gradio_client.handle_file = lambda p: {'file': p}
    gradio_client.Client = MagicMock(name='Client')

    soundfile = types.ModuleType('soundfile')
    soundfile.write = lambda path, data, sr, format=None: Path(path).write_bytes(b'RIFFstub')

    mtl = types.ModuleType('chatterbox.mtl_tts')
    mtl.ChatterboxMultilingualTTS = MagicMock(name='ChatterboxMultilingualTTS')
    pkg = types.ModuleType('chatterbox')
    pkg.mtl_tts = mtl

    names = ('videotrans.tts._gradio', 'videotrans.tts._voxcpm', 'videotrans.tts._chatterbox')
    for n, mod in (('gradio_client', gradio_client), ('soundfile', soundfile), ('chatterbox', pkg), ('chatterbox.mtl_tts', mtl)):
        monkeypatch.setitem(sys.modules, n, mod)
    for n in names:
        sys.modules.pop(n, None)
    try:
        yield SimpleNamespace(gradio_client=gradio_client, mtl=mtl, names=names)
    finally:
        for n in names:
            sys.modules.pop(n, None)


def _clone_item(tmp_path, text='Stop messing around', **extra):
    ref = tmp_path / 'ref.wav'
    ref.write_bytes(b'RIFFref')
    item = {'text': text, 'role': 'clone', 'ref_wav': str(ref), 'ref_text': 'ふざけるな', 'ref_language': 'ja',
            'filename': str(tmp_path / '1-key.wav'), 'start_time_source': 2000}
    item.update(extra)
    return item


ANGRY = dict(emotion='angry', emotion_intensity=0.8, emotion_confidence=0.6)


class TestVoxcpm:
    @pytest.fixture
    def tts(self, heavy_stubs, monkeypatch, tmp_path):
        mod = importlib.import_module('videotrans.tts._voxcpm')
        monkeypatch.setattr('videotrans.tts._gradio.params', {'voxcpm_url': 'http://127.0.0.1:7860'})
        monkeypatch.setattr(mod, 'params', {'voxcpmtts_version': 'v2'})
        obj = mod.VoxCPMTTS(queue_tts=[{'text': 'x', 'role': 'clone', 'filename': str(tmp_path / 'q.wav')}], language='en', uuid='u')
        obj.versions = mod.params
        obj.sent = []
        monkeypatch.setattr(obj, '_send', lambda kwargs, item: obj.sent.append(kwargs))
        return obj

    def run(self, tts, item, version='v2'):
        tts.versions['voxcpmtts_version'] = version
        tts._run(item)
        return tts.sent[-1]

    def test_emotional_line_gets_a_control_instruction_and_drops_the_prompt_text(self, tts, tmp_path):
        kw = self.run(tts, _clone_item(tmp_path, **ANGRY))
        assert kw['control_instruction'] == 'very angry, tense'
        assert kw['use_prompt_text'] is False
        assert kw['text'] == 'Stop messing around'  # the text itself is untouched: the app builds "(instruction)text"
        assert kw['ref_wav'] == {'file': str(tmp_path / 'ref.wav')}
        assert kw['api_name'] == '/generate'

    @pytest.mark.parametrize('intensity, instruction', [(0.2, 'slightly cheerful'), (0.5, 'cheerful'), (0.9, 'very cheerful')])
    def test_intensity_wording(self, tts, tmp_path, intensity, instruction):
        item = _clone_item(tmp_path, emotion='happy', emotion_intensity=intensity, emotion_confidence=0.6)
        assert self.run(tts, item)['control_instruction'] == instruction

    def test_line_without_emotion_is_exactly_upstream(self, tts, tmp_path):
        kw = self.run(tts, _clone_item(tmp_path))
        assert kw == {
            'do_normalize': False, 'denoise': False, 'api_name': '/generate',
            'text': 'Stop messing around', 'control_instruction': '', 'use_prompt_text': True,
            'ref_wav': {'file': str(tmp_path / 'ref.wav')}, 'dit_steps': 10, 'cfg_value': 2.0, 'prompt_text_value': 'ふざけるな',
        }

    @pytest.mark.parametrize('extra', [
        dict(emotion='neutral', emotion_intensity=0.9, emotion_confidence=0.9),
        dict(emotion='angry', emotion_intensity=0.9, emotion_confidence=0.3),
    ])
    def test_neutral_or_low_confidence_is_upstream(self, tts, tmp_path, extra):
        kw = self.run(tts, _clone_item(tmp_path, **extra))
        assert kw['control_instruction'] == '' and kw['use_prompt_text'] is True

    def test_emotional_line_without_a_transcript(self, tts, tmp_path):
        kw = self.run(tts, _clone_item(tmp_path, ref_text='', **ANGRY))
        assert kw['control_instruction'] == 'very angry, tense' and kw['use_prompt_text'] is False

    def test_hf_version_uses_the_same_fields(self, tts, tmp_path):
        kw = self.run(tts, _clone_item(tmp_path, **ANGRY), version='hf')
        assert kw['control_instruction'] == 'very angry, tense' and kw['use_prompt_text'] is False
        assert kw['text_input'] == 'Stop messing around' and 'text' not in kw
        kw = self.run(tts, _clone_item(tmp_path), version='hf')
        assert kw['control_instruction'] == '' and kw['use_prompt_text'] is True

    def test_voxcpm1_has_no_instruction_field(self, tts, tmp_path):
        kw = self.run(tts, _clone_item(tmp_path, **ANGRY), version='v1')
        assert 'control_instruction' not in kw and 'use_prompt_text' not in kw
        assert kw['prompt_text_input'] == 'ふざけるな'

    def test_request_reaches_the_gradio_client(self, heavy_stubs, monkeypatch, tmp_path):
        mod = importlib.import_module('videotrans.tts._voxcpm')
        monkeypatch.setattr('videotrans.tts._gradio.params', {'voxcpm_url': 'http://127.0.0.1:7860'})
        monkeypatch.setattr(mod, 'params', {'voxcpmtts_version': 'v2'})
        out = tmp_path / 'out.wav'
        out.write_bytes(b'RIFFout')
        client = MagicMock()
        client.predict.return_value = (str(out), 123)
        obj = mod.VoxCPMTTS(queue_tts=[{'text': 'x', 'role': 'clone', 'filename': str(tmp_path / 'q.wav')}], language='en', uuid='u')
        monkeypatch.setattr(obj, 'get_thread_client', lambda: client)
        monkeypatch.setattr(obj, 'convert_to_wav', lambda src, dst: Path(dst).write_bytes(b'RIFFdone'))
        item = _clone_item(tmp_path, **ANGRY)
        assert obj._run(item) is None
        client.predict.assert_called_once()
        sent = client.predict.call_args.kwargs
        assert sent['control_instruction'] == 'very angry, tense' and sent['use_prompt_text'] is False
        assert Path(item['filename']).read_bytes() == b'RIFFdone'


class TestChatterbox:
    @pytest.fixture
    def make(self, heavy_stubs, monkeypatch, tmp_path):
        mod = importlib.import_module('videotrans.tts._chatterbox')
        monkeypatch.setattr(mod, 'params', {'chatterbox_exaggeration': '0.5', 'chatterbox_cfg_weight': '0.5'})
        monkeypatch.setattr(mod.ChatterBoxTTS, 'convert_to_wav', lambda self, src, dst: Path(dst).write_bytes(b'RIFFdone'))
        model = MagicMock()
        model.sr = 24000
        model.generate.return_value.detach.return_value.cpu.return_value = MagicMock(ndim=1)
        heavy_stubs.mtl.ChatterboxMultilingualTTS.from_local.return_value = model

        def _make(items, language='en', **settings):
            monkeypatch.setattr(mod, 'settings', FakeSettings(device_name='cpu', **settings))
            obj = mod.ChatterBoxTTS(queue_tts=items, language=language, uuid='u')
            obj.local_dir = str(tmp_path)
            return obj, model

        return _make

    def items(self, tmp_path):
        return [
            _clone_item(tmp_path, text='Hello', filename=str(tmp_path / '0.wav')),
            _clone_item(tmp_path, text='Stop messing around', filename=str(tmp_path / '1.wav'), **ANGRY),
            _clone_item(tmp_path, text='We did it', filename=str(tmp_path / '2.wav'), emotion='happy', emotion_intensity=0.3, emotion_confidence=0.6),
        ]

    def calls(self, model):
        return [(c.args[0], c.kwargs['exaggeration'], c.kwargs['cfg_weight']) for c in model.generate.call_args_list]

    def test_per_line_exaggeration_and_crosslingual_cfg0(self, make, tmp_path):
        obj, model = make(self.items(tmp_path), enable_emotion=True)
        obj._exec()
        got = self.calls(model)
        assert [g[0] for g in got] == ['Hello', 'Stop messing around', 'We did it']
        assert got[0][1] == pytest.approx(0.5)            # no emotion: the user's value
        assert got[1][1] == pytest.approx(0.82)           # 0.5 + 0.4 * 0.8
        assert got[2][1] == pytest.approx(0.62)           # 0.5 + 0.4 * 0.3
        assert [g[2] for g in got] == [0.0, 0.0, 0.0]     # reference ja, target en

    def test_same_language_keeps_the_users_cfg_weight(self, make, tmp_path):
        items = [dict(it, ref_language='en') for it in self.items(tmp_path)]
        obj, model = make(items, enable_emotion=True)
        obj._exec()
        assert [g[2] for g in self.calls(model)] == [0.5, 0.5, 0.5]

    def test_crosslingual_cfg0_can_be_switched_off(self, make, tmp_path):
        obj, model = make(self.items(tmp_path), enable_emotion=True, chatterbox_crosslingual_cfg0=False)
        obj._exec()
        got = self.calls(model)
        assert [g[2] for g in got] == [0.5, 0.5, 0.5]
        assert got[1][1] == pytest.approx(0.82)  # exaggeration still follows the emotion

    def test_emotion_tts_off_is_upstream(self, make, tmp_path):
        obj, model = make(self.items(tmp_path), enable_emotion=True, emotion_tts=False)
        obj._exec()
        # cross-lingual cfg0 belongs to the emotion mode; the per-line exaggeration is data-driven (the dubbing stage
        # does not set the emotion keys while emotion_tts is off, see TestDubbingEmotion)
        assert [g[2] for g in self.calls(model)] == [0.5, 0.5, 0.5]

    def test_enable_emotion_off_is_upstream(self, make, tmp_path):
        items = [{k: v for k, v in it.items() if not k.startswith('emotion')} for it in self.items(tmp_path)]
        obj, model = make(items)  # enable_emotion absent = False
        obj._exec()
        assert [(g[1], g[2]) for g in self.calls(model)] == [(0.5, 0.5)] * 3

    def test_user_values_from_the_dialog_are_the_base(self, make, heavy_stubs, tmp_path, monkeypatch):
        mod = sys.modules['videotrans.tts._chatterbox']
        monkeypatch.setattr(mod, 'params', {'chatterbox_exaggeration': '0.3', 'chatterbox_cfg_weight': '0.35'})
        items = [dict(it, ref_language='en') for it in self.items(tmp_path)]
        obj, model = make(items, enable_emotion=True)
        obj._exec()
        got = self.calls(model)
        assert got[0][1:] == (pytest.approx(0.3), 0.35)
        assert got[1][1:] == (pytest.approx(0.62), 0.35)  # 0.3 + 0.4 * 0.8


# ---------------------------------------------------------------- AppSettings
class TestSettings:
    def test_new_defaults(self, tmp_path):
        from videotrans.configure._app_settings import AppSettings
        s = AppSettings(_json_path=str(tmp_path / 'cfg.json'))
        assert s.get('emotion_tts') is True
        assert s.get('chatterbox_crosslingual_cfg0') is True
        assert s.get('enable_emotion') is False  # emotion stays opt-in

    def test_values_in_cfg_json_win(self, tmp_path):
        from videotrans.configure._app_settings import AppSettings
        p = tmp_path / 'cfg.json'
        p.write_text(json.dumps({'emotion_tts': False, 'chatterbox_crosslingual_cfg0': 'false'}), encoding='utf-8')
        s = AppSettings(_json_path=str(p))
        assert s.get('emotion_tts') is False
        assert s.get('chatterbox_crosslingual_cfg0') is False


def test_voxcpm_similarity_priority_sends_no_instruction():
    """voxcpm_emotion_priority='similarity' keeps transcript-guided cloning: no instruction even for an emotional line"""
    from videotrans.util.emotion_tts import voxcpm_control_instruction
    item = {'emotion': 'angry', 'emotion_intensity': 0.8, 'emotion_confidence': 0.9}
    assert voxcpm_control_instruction(item) != ''
    assert voxcpm_control_instruction(item, 'emotion') == voxcpm_control_instruction(item)
    assert voxcpm_control_instruction(item, 'similarity') == ''
    assert voxcpm_control_instruction({}, 'emotion') == ''
