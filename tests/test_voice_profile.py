"""
Tests for videotrans/util/voice_profile.py (speaker voice profiles used as the clone reference).
Pure file/JSON logic only: no audio decoding, no models.
"""
import json
from pathlib import Path

import pytest

from videotrans.util import voice_profile as vp
from videotrans.util.voice_profile import ProfileRef, VoiceProfiles


def make_profile(root: Path, name: str, refs: list, write_wavs: bool = True) -> Path:
    """<root>/<name>/profile.json with refs (wav paths relative to the profile folder)."""
    folder = root / name
    (folder / 'refs').mkdir(parents=True, exist_ok=True)
    if write_wavs:
        for r in refs:
            (folder / r['wav']).write_bytes(b'RIFF-fake-' + r['wav'].encode())
    (folder / 'profile.json').write_text(json.dumps({'name': name, 'refs': refs}, ensure_ascii=False), encoding='utf-8')
    return folder


def ref(name='a', wav='refs/a.wav', lang='ja', sec=12.0, snr_db=30.0, text='テキスト'):
    return ProfileRef(profile=name, wav=wav, text=text, lang=lang, sec=sec, snr_db=snr_db)


class TestLangCode:
    @pytest.mark.parametrize('raw,expected', [('ja', 'ja'), ('ja-JP', 'ja'), ('zh_CN', 'zh'), ('EN', 'en'), ('', ''), (None, '')])
    def test_primary_subtag(self, raw, expected):
        assert vp.lang_code(raw) == expected


class TestLoadProfile:
    def test_relative_paths_become_absolute(self, tmp_path):
        folder = make_profile(tmp_path, 'Aさん', [
            {'wav': 'refs/a_01.wav', 'text': ' こんにちは ', 'lang': 'ja-JP', 'sec': 14.2, 'snr_db': 32.1}])
        refs = vp.load_profile(folder)
        assert len(refs) == 1
        r = refs[0]
        assert Path(r.wav).is_absolute()
        assert Path(r.wav).samefile(folder / 'refs/a_01.wav')
        assert (r.profile, r.text, r.lang, r.sec, r.snr_db) == ('Aさん', 'こんにちは', 'ja', 14.2, 32.1)

    def test_missing_wav_is_skipped(self, tmp_path):
        folder = make_profile(tmp_path, 'A', [{'wav': 'refs/ok.wav'}], write_wavs=True)
        data = json.loads((folder / 'profile.json').read_text(encoding='utf-8'))
        data['refs'].append({'wav': 'refs/gone.wav', 'text': 'x'})
        (folder / 'profile.json').write_text(json.dumps(data), encoding='utf-8')
        assert [Path(r.wav).name for r in vp.load_profile(folder)] == ['ok.wav']

    def test_optional_fields_may_be_missing_or_invalid(self, tmp_path):
        folder = make_profile(tmp_path, 'A', [{'wav': 'refs/a.wav', 'sec': 'x', 'snr_db': None, 'lang': None}])
        r = vp.load_profile(folder)[0]
        assert (r.text, r.lang, r.sec, r.snr_db) == ('', '', None, None)

    def test_non_positive_or_nan_values_are_dropped(self, tmp_path):
        folder = make_profile(tmp_path, 'A', [{'wav': 'refs/a.wav', 'sec': 0, 'snr_db': float('nan')}])
        r = vp.load_profile(folder)[0]
        assert r.sec is None and r.snr_db is None

    @pytest.mark.parametrize('content', ['{not json', '[]', '{"refs": 5}', '{"refs": ["x", 3]}', ''])
    def test_broken_profile_json_gives_no_refs(self, tmp_path, content):
        folder = tmp_path / 'A'
        folder.mkdir()
        (folder / 'profile.json').write_text(content, encoding='utf-8')
        assert vp.load_profile(folder) == []

    def test_profile_json_missing(self, tmp_path):
        assert vp.load_profile(tmp_path / 'nothing') == []


class TestSelectRef:
    def test_empty(self):
        assert vp.select_ref([], 'ja') is None

    def test_language_match_beats_snr(self):
        en_clean = ref('A', 'en.wav', lang='en', snr_db=45)
        ja_noisy = ref('A', 'ja.wav', lang='ja', snr_db=20)
        assert vp.select_ref([en_clean, ja_noisy], 'ja') is ja_noisy
        assert vp.select_ref([ja_noisy, en_clean], 'en-US') is en_clean

    def test_snr_beats_duration_within_same_language(self):
        good_snr_far = ref(wav='far.wav', snr_db=35, sec=5)
        low_snr_near = ref(wav='near.wav', snr_db=30, sec=12)
        assert vp.select_ref([low_snr_near, good_snr_far], 'ja') is good_snr_far

    def test_duration_closest_to_12s_on_snr_tie(self):
        a = ref(wav='a.wav', sec=8)
        b = ref(wav='b.wav', sec=13)
        c = ref(wav='c.wav', sec=19)
        assert vp.select_ref([a, c, b], 'ja') is b

    def test_duration_is_capped_at_20s(self):
        # 20 s and 60 s are both scored as 20 s (distance 8): a tie, so the first one wins
        x = ref(wav='x.wav', sec=60)
        y = ref(wav='y.wav', sec=20)
        assert vp.select_ref([x, y], 'ja') is x
        assert vp.select_ref([y, x], 'ja') is y
        # ...but a 5 s clip (distance 7) is still closer to 12 s than any clip of 20 s or more
        z = ref(wav='z.wav', sec=5)
        assert vp.select_ref([x, y, z], 'ja') is z

    def test_unknown_snr_and_sec_rank_last(self):
        unknown = ref(wav='u.wav', snr_db=None, sec=None)
        known = ref(wav='k.wav', snr_db=10, sec=30)
        assert vp.select_ref([unknown, known], 'ja') is known
        same_snr_unknown_sec = ref(wav='s.wav', snr_db=10, sec=None)
        assert vp.select_ref([same_snr_unknown_sec, known], 'ja') is known

    def test_unknown_source_language_skips_language_criterion(self):
        a = ref(wav='a.wav', lang='en', snr_db=20)
        b = ref(wav='b.wav', lang='ja', snr_db=30)
        assert vp.select_ref([a, b], '') is b
        assert vp.select_ref([a, b], 'auto') is b

    def test_no_language_match_still_returns_best_remaining(self):
        a = ref(wav='a.wav', lang='fr', snr_db=20)
        b = ref(wav='b.wav', lang='en', snr_db=30)
        assert vp.select_ref([a, b], 'ja') is b


class TestSpeakerMapAndLookup:
    def _vps(self, tmp_path, names=('Aさん', 'Bさん', 'spk2'), speaker_map=None):
        for n in names:
            make_profile(tmp_path, n, [{'wav': 'refs/x.wav', 'text': n, 'lang': 'ja', 'sec': 12, 'snr_db': 30}])
        if speaker_map is not None:
            (tmp_path / 'speaker_map.json').write_text(json.dumps(speaker_map, ensure_ascii=False), encoding='utf-8')
        return VoiceProfiles.load(tmp_path)

    def test_load_finds_profile_folders_only(self, tmp_path):
        (tmp_path / 'not_a_profile').mkdir()
        (tmp_path / 'readme.txt').write_text('x')
        vps = self._vps(tmp_path)
        assert sorted(vps.profiles) == ['Aさん', 'Bさん', 'spk2']

    def test_folder_without_usable_refs_is_not_a_profile(self, tmp_path):
        make_profile(tmp_path, 'Empty', [])
        make_profile(tmp_path, 'NoWav', [{'wav': 'refs/x.wav'}], write_wavs=False)
        assert VoiceProfiles.load(tmp_path).profiles == {}

    def test_speaker_map_entry(self, tmp_path):
        vps = self._vps(tmp_path, speaker_map={'spk0': 'Aさん', 'spk1': 'Bさん'})
        assert vps.profile_name('spk0') == 'Aさん'
        assert vps.ref_for('spk1', 'ja').profile == 'Bさん'

    def test_fallback_to_folder_named_like_label(self, tmp_path):
        vps = self._vps(tmp_path, speaker_map={'spk0': 'Aさん'})
        assert vps.profile_name('spk2') == 'spk2'
        assert vps.ref_for('spk2', 'ja').profile == 'spk2'

    def test_fallback_without_any_speaker_map_file(self, tmp_path):
        vps = self._vps(tmp_path)
        assert vps.speaker_map == {}
        assert vps.profile_name('spk2') == 'spk2'
        assert vps.profile_name('spk0') is None

    def test_unknown_label_has_no_ref(self, tmp_path):
        vps = self._vps(tmp_path, speaker_map={'spk0': 'Aさん'})
        assert vps.ref_for('spk9', 'ja') is None

    @pytest.mark.parametrize('label', ['', '   ', None])
    def test_empty_label_has_no_ref(self, tmp_path, label):
        vps = self._vps(tmp_path, speaker_map={'spk0': 'Aさん'})
        assert vps.ref_for(label, 'ja') is None

    def test_map_entry_pointing_to_missing_profile_does_not_fall_back_to_label_folder(self, tmp_path):
        # 'spk2' is mapped to a profile that does not exist, although a folder named 'spk2' does exist
        vps = self._vps(tmp_path, speaker_map={'spk2': 'Zさん'})
        assert vps.profile_name('spk2') is None

    def test_label_case_and_spaces_are_tolerated_for_the_map(self, tmp_path):
        vps = self._vps(tmp_path, speaker_map={'spk0': 'Aさん'})
        assert vps.profile_name('SPK0') == 'Aさん'
        assert vps.profile_name('spk 0') == 'Aさん'

    @pytest.mark.parametrize('content', ['{broken', '["spk0"]', '"x"', ''])
    def test_broken_speaker_map_is_ignored(self, tmp_path, content):
        (tmp_path / 'speaker_map.json').write_text(content, encoding='utf-8')
        assert vp.load_speaker_map(tmp_path) == {}

    def test_ref_for_uses_language_aware_selection(self, tmp_path):
        make_profile(tmp_path, 'A', [
            {'wav': 'refs/en.wav', 'lang': 'en', 'sec': 12, 'snr_db': 40},
            {'wav': 'refs/ja.wav', 'lang': 'ja', 'sec': 12, 'snr_db': 25},
        ])
        vps = VoiceProfiles.load(tmp_path)
        assert Path(vps.ref_for('A', 'ja').wav).name == 'ja.wav'
        assert Path(vps.ref_for('A', 'en').wav).name == 'en.wav'


class TestReadSpeakerLabels:
    def test_reads_list(self, tmp_path):
        p = tmp_path / 'speaker.json'
        p.write_text(json.dumps(['spk0', 'spk1', None, ' spk0 ']), encoding='utf-8')
        assert vp.read_speaker_labels(p) == ['spk0', 'spk1', '', 'spk0']

    @pytest.mark.parametrize('content', ['{broken', '{"a": 1}', '5'])
    def test_broken_or_wrong_shape(self, tmp_path, content):
        p = tmp_path / 'speaker.json'
        p.write_text(content, encoding='utf-8')
        assert vp.read_speaker_labels(p) == []

    def test_missing_file(self, tmp_path):
        assert vp.read_speaker_labels(tmp_path / 'nope.json') == []


class TestResolveMode:
    @pytest.mark.parametrize('mode,profile_dir,expected', [
        ('', '', 'line'),  # nothing set: upstream behaviour
        ('', '/p', 'profile'),  # dir set, mode unset: profile
        (None, '/p', 'profile'),
        ('auto', '/p', 'profile'),  # unknown value = unset
        ('line', '/p', 'line'),  # explicit line wins even when a dir is set
        ('profile', '/p', 'profile'),
        ('PROFILE', '/p', 'profile'),
        ('profile', '', 'line'),  # profile mode without a dir cannot work
        ('line', '', 'line'),
        ('', '   ', 'line'),
    ])
    def test_matrix(self, mode, profile_dir, expected):
        assert vp.resolve_clone_ref_mode(mode, profile_dir) == expected


class TestLoadForDubbing:
    def _dir(self, tmp_path):
        make_profile(tmp_path, 'A', [{'wav': 'refs/a.wav', 'lang': 'ja', 'sec': 12, 'snr_db': 30}])
        return str(tmp_path)

    def test_disabled_by_default(self):
        assert vp.load_for_dubbing({}) is None
        assert vp.load_for_dubbing({'voice_profile_dir': '', 'clone_ref_mode': ''}) is None

    def test_enabled_when_dir_is_set(self, tmp_path):
        got = vp.load_for_dubbing({'voice_profile_dir': self._dir(tmp_path)})
        assert isinstance(got, VoiceProfiles) and 'A' in got.profiles

    def test_line_mode_disables_even_with_dir(self, tmp_path):
        assert vp.load_for_dubbing({'voice_profile_dir': self._dir(tmp_path), 'clone_ref_mode': 'line'}) is None

    def test_dir_that_does_not_exist(self, tmp_path):
        assert vp.load_for_dubbing({'voice_profile_dir': str(tmp_path / 'missing')}) is None

    def test_dir_without_usable_profiles(self, tmp_path):
        assert vp.load_for_dubbing({'voice_profile_dir': str(tmp_path)}) is None


class TestChooseCloneRef:
    @pytest.fixture
    def vps(self, tmp_path):
        make_profile(tmp_path, 'Aさん', [{'wav': 'refs/a.wav', 'text': 'A text', 'lang': 'ja', 'sec': 12, 'snr_db': 30}])
        make_profile(tmp_path, 'Bさん', [{'wav': 'refs/b.wav', 'text': 'B text', 'lang': 'ja', 'sec': 12, 'snr_db': 30}])
        (tmp_path / 'speaker_map.json').write_text(json.dumps({'spk0': 'Aさん', 'spk1': 'Bさん'}), encoding='utf-8')
        return VoiceProfiles.load(tmp_path)

    def test_picks_profile_of_the_line_speaker(self, vps):
        speakers = ['spk0', 'spk1', 'spk0']
        assert vp.choose_clone_ref(vps, speakers, 0, 'ja').profile == 'Aさん'
        assert vp.choose_clone_ref(vps, speakers, 1, 'ja').profile == 'Bさん'
        assert vp.choose_clone_ref(vps, speakers, 2, 'ja').text == 'A text'

    def test_no_profiles_means_line_reference(self):
        assert vp.choose_clone_ref(None, ['spk0'], 0, 'ja') is None

    def test_missing_labels_mean_line_reference(self, vps):
        assert vp.choose_clone_ref(vps, [], 0, 'ja') is None  # no speaker.json (no diarization)

    def test_index_beyond_labels_means_line_reference(self, vps):
        assert vp.choose_clone_ref(vps, ['spk0'], 1, 'ja') is None
        assert vp.choose_clone_ref(vps, ['spk0'], -1, 'ja') is None

    def test_blank_label_means_line_reference(self, vps):
        assert vp.choose_clone_ref(vps, ['', 'spk0'], 0, 'ja') is None

    def test_speaker_without_profile_means_line_reference(self, vps):
        assert vp.choose_clone_ref(vps, ['spk5'], 0, 'ja') is None


class TestStageRefWav:
    def test_copies_into_cache_and_original_is_untouched(self, tmp_path):
        src = tmp_path / 'profiles' / 'A' / 'a.wav'
        src.parent.mkdir(parents=True)
        src.write_bytes(b'0123456789')
        r = ProfileRef(profile='A', wav=str(src))
        cache = tmp_path / 'cache'
        out = vp.stage_ref_wav(r, cache)
        assert out and Path(out).parent == cache and Path(out).suffix == '.wav'
        assert Path(out).read_bytes() == b'0123456789'
        # an engine that trims the reference in place (GPT-SoVITS) only damages the copy
        Path(out).write_bytes(b'0123')
        assert src.read_bytes() == b'0123456789'
        # staging again restores the full copy
        assert vp.stage_ref_wav(r, cache) == out
        assert Path(out).read_bytes() == b'0123456789'

    def test_same_clip_shares_one_copy_and_different_clips_do_not(self, tmp_path):
        a, b = tmp_path / 'a.wav', tmp_path / 'b.wav'
        a.write_bytes(b'aa')
        b.write_bytes(b'bb')
        cache = tmp_path / 'cache'
        assert vp.stage_ref_wav(ProfileRef('A', str(a)), cache) == vp.stage_ref_wav(ProfileRef('A', str(a)), cache)
        assert vp.stage_ref_wav(ProfileRef('A', str(a)), cache) != vp.stage_ref_wav(ProfileRef('B', str(b)), cache)

    def test_copy_failure_returns_none(self, tmp_path):
        assert vp.stage_ref_wav(ProfileRef('A', str(tmp_path / 'gone.wav')), tmp_path / 'cache') is None


class TestCacheKey:
    def test_contains_path_and_changes_with_content_and_text(self, tmp_path):
        f = tmp_path / 'a.wav'
        f.write_bytes(b'one')
        k1 = ProfileRef('A', str(f), text='t').cache_key()
        assert str(f) in k1
        f.write_bytes(b'one-longer')
        k2 = ProfileRef('A', str(f), text='t').cache_key()
        assert k2 != k1
        assert ProfileRef('A', str(f), text='t2').cache_key() != k2

    def test_different_paths_give_different_keys(self, tmp_path):
        a, b = tmp_path / 'a.wav', tmp_path / 'b.wav'
        a.write_bytes(b'x')
        b.write_bytes(b'x')
        assert ProfileRef('A', str(a)).cache_key() != ProfileRef('A', str(b)).cache_key()

    def test_missing_file_does_not_raise(self, tmp_path):
        assert ProfileRef('A', str(tmp_path / 'none.wav')).cache_key()


class TestCloneRefSampleRate:
    @pytest.mark.parametrize('value,expected', [
        (44100, 44100), ('48000', 48000), (24000.0, 24000), ('16000', 16000),
        ('abc', 44100), (None, 44100), ('', 44100), (100, 44100), (10 ** 7, 44100),
    ])
    def test_values(self, value, expected):
        assert vp.clone_ref_sample_rate({'clone_ref_sample_rate': value}) == expected

    def test_default_when_unset(self):
        assert vp.clone_ref_sample_rate({}) == 44100
