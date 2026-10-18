"""EMOTION module part 1: data model, sidecar I/O, matching by start time, SenseVoice capture."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from videotrans.util import emotion as emo
from videotrans.util.emotion import EmotionEntry, EmotionIndex


def _line(start, end=None):
    return {'start_time': start, 'end_time': end if end is not None else start + 1000}


# ---------------------------------------------------------------- tag extraction / mapping

@pytest.mark.parametrize('tag,label', [
    ('NEUTRAL', 'neutral'), ('HAPPY', 'happy'), ('ANGRY', 'angry'), ('SAD', 'sad'),
    ('FEARFUL', 'fearful'), ('DISGUSTED', 'disgusted'), ('SURPRISED', 'surprised'),
])
def test_extract_and_map_every_sensevoice_emotion_tag(tag, label):
    from videotrans.process._stt_utils import _extract_emotion_tag
    text = f'<|ja|><|{tag}|><|Speech|><|withitn|>こんにちは'
    assert _extract_emotion_tag(text) == tag
    mapped, intensity, confidence = emo.map_sensevoice_tag(tag)
    assert mapped == label
    if label == 'neutral':
        assert (intensity, confidence) == (0.0, 0.6)
    else:
        assert (intensity, confidence) == (0.5, 0.6)


@pytest.mark.parametrize('text', [
    'こんにちは',                                  # no tags at all
    '<|ja|><|Speech|><|withitn|>こんにちは',        # language / event / itn tags only
    '<|en|><|Laughter|><|woitn|>ha ha',            # event tag must not be mistaken for an emotion
    '',
    None,
])
def test_extract_returns_empty_when_no_emotion_tag(text):
    from videotrans.process._stt_utils import _extract_emotion_tag
    assert _extract_emotion_tag(text) == ''


def test_extract_is_case_insensitive_and_takes_first_emotion_tag():
    from videotrans.process._stt_utils import _extract_emotion_tag
    assert _extract_emotion_tag('<|zh|><|happy|><|Speech|>x') == 'HAPPY'
    assert _extract_emotion_tag('<|zh|><|SAD|>a<|ANGRY|>b') == 'SAD'


@pytest.mark.parametrize('tag', ['EMO_UNKNOWN', 'SOMETHING_NEW', '', None])
def test_unknown_or_missing_tag_maps_to_neutral_with_zero_confidence(tag):
    assert emo.map_sensevoice_tag(tag) == ('neutral', 0.0, 0.0)


def test_remove_unwanted_characters_output_is_unchanged():
    # the transcript text must stay exactly as before the capture was added
    from videotrans.process._stt_utils import _remove_unwanted_characters
    assert _remove_unwanted_characters('<|ja|><|HAPPY|><|Speech|><|withitn|>こんにちは。') == 'こんにちは。'


def _install_fake_funasr(monkeypatch, texts):
    """Fake funasr / modelscope so funasr_mlt can run without models."""
    fake_model = MagicMock(return_value=[{'key': str(i), 'text': t} for i, t in enumerate(texts)])
    modelscope = types.ModuleType('modelscope')
    pipelines = types.ModuleType('modelscope.pipelines')
    pipelines.pipeline = MagicMock(return_value=fake_model)
    utils = types.ModuleType('modelscope.utils')
    constant = types.ModuleType('modelscope.utils.constant')
    constant.Tasks = types.SimpleNamespace(auto_speech_recognition='asr')
    funasr = types.ModuleType('funasr')
    funasr.AutoModel = MagicMock()
    for name, mod in {'funasr': funasr, 'modelscope': modelscope, 'modelscope.pipelines': pipelines,
                      'modelscope.utils': utils, 'modelscope.utils.constant': constant}.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_funasr_sensevoice_keeps_text_and_saves_raw_tags(monkeypatch, tmp_path):
    texts = ['<|ja|><|NEUTRAL|><|Speech|><|withitn|>おはようございます',
             '<|ja|><|ANGRY|><|Speech|><|withitn|>なんでそうなるの',
             '<|ja|><|Speech|><|withitn|>タグなし']
    _install_fake_funasr(monkeypatch, texts)
    from videotrans.process.stt_funasr import funasr_mlt
    cut = [{'filename': f'{i}.wav', 'text': '', 'start_time': 1000 * i, 'end_time': 1000 * i + 800} for i in range(3)]

    srts, err = funasr_mlt(cut_audio_list=cut, detect_language='ja', model_name='iic/SenseVoiceSmall',
                           logs_file=None, hotword='', cache_folder=str(tmp_path))

    assert err is None
    assert [s['text'] for s in srts] == ['おはようございます', 'なんでそうなるの', 'タグなし']
    raw = json.loads((tmp_path / 'emotion_raw.json').read_text(encoding='utf-8'))
    assert raw == [
        {'start_time': 0, 'end_time': 800, 'tag': 'NEUTRAL'},
        {'start_time': 1000, 'end_time': 1800, 'tag': 'ANGRY'},
        {'start_time': 2000, 'end_time': 2800, 'tag': ''},
    ]


def test_funasr_other_models_do_not_write_raw_tags(monkeypatch, tmp_path):
    _install_fake_funasr(monkeypatch, ['<|ja|><|HAPPY|>x'])
    import videotrans.process.stt_funasr as mod
    sys.modules['funasr'].AutoModel.return_value.generate.return_value = [{'text': '<|HAPPY|>x'}]
    monkeypatch.setattr(mod, 'ROOT_DIR', str(tmp_path), raising=False)
    cut = [{'filename': '0.wav', 'text': '', 'start_time': 0, 'end_time': 800}]
    srts, err = mod.funasr_mlt(cut_audio_list=cut, detect_language='ja', model_name='FunAudioLLM/Fun-ASR-Nano-2512',
                               logs_file=None, hotword='', cache_folder=str(tmp_path))
    assert err is None and srts[0]['text'] == 'x'
    assert not (tmp_path / 'emotion_raw.json').exists()


def test_funasr_capture_failure_does_not_break_transcription(monkeypatch, tmp_path):
    _install_fake_funasr(monkeypatch, ['<|ja|><|HAPPY|><|Speech|>やった'])
    monkeypatch.setattr('videotrans.util.emotion.save_raw_capture', MagicMock(side_effect=OSError('disk full')))
    from videotrans.process.stt_funasr import funasr_mlt
    cut = [{'filename': '0.wav', 'text': '', 'start_time': 0, 'end_time': 800}]
    srts, err = funasr_mlt(cut_audio_list=cut, detect_language='ja', model_name='iic/SenseVoiceSmall',
                           logs_file=None, hotword='', cache_folder=str(tmp_path))
    assert err is None and srts[0]['text'] == 'やった'


# ---------------------------------------------------------------- emotion.json load / save

def test_save_and_load_roundtrip_copies_to_target_dir(tmp_path):
    cache, target = tmp_path / 'cache', tmp_path / 'out'
    target.mkdir()
    entries = [EmotionEntry(12340, 15880, 'angry', 0.5, 0.6, 'sensevoice'),
               EmotionEntry(1000, 2000, 'neutral', 0.0, 0.6, 'sensevoice')]

    path = emo.save_emotion(entries, str(cache), str(target))

    assert Path(path) == cache / 'emotion.json'
    assert (target / 'emotion.json').read_text(encoding='utf-8') == (cache / 'emotion.json').read_text(encoding='utf-8')
    loaded = emo.load_emotion(path)
    assert [e.start_time for e in loaded] == [1000, 12340]      # sorted by start time
    assert loaded[1] == entries[0]
    assert set(json.loads((cache / 'emotion.json').read_text(encoding='utf-8'))[0]) == {
        'start_time', 'end_time', 'label', 'intensity', 'confidence', 'source'}


def test_save_without_target_dir_or_missing_target_dir(tmp_path):
    emo.save_emotion([EmotionEntry(0, 1)], str(tmp_path / 'c'))
    emo.save_emotion([EmotionEntry(0, 1)], str(tmp_path / 'c'), str(tmp_path / 'does-not-exist'))
    assert not (tmp_path / 'does-not-exist').exists()


@pytest.mark.parametrize('content', [None, '', 'not json', '{"a": 1}', '[1, "x", null]'])
def test_load_missing_or_broken_file_gives_empty_list(tmp_path, content):
    p = tmp_path / 'emotion.json'
    if content is not None:
        p.write_text(content, encoding='utf-8')
    assert emo.load_emotion(str(p)) == []


def test_load_sanitizes_rows(tmp_path):
    p = tmp_path / 'emotion.json'
    p.write_text(json.dumps([
        {'start_time': 0, 'end_time': 10, 'label': 'ANGRY', 'intensity': 3, 'confidence': 0.9},
        {'start_time': 20, 'end_time': 30, 'label': 'bored', 'intensity': 0.5, 'confidence': 0.9},
        {'start_time': 40, 'label': 'happy', 'intensity': 'x', 'confidence': None},
    ]), encoding='utf-8')
    a, b, c = emo.load_emotion(str(p))
    assert (a.label, a.intensity, a.confidence, a.source) == ('angry', 1.0, 0.9, 'none')   # lower-cased, clipped
    assert (b.label, b.confidence) == ('neutral', 0.0)                                      # unknown label -> neutral/0
    assert (c.intensity, c.confidence, c.end_time) == (0.0, 0.0, 0)


# ---------------------------------------------------------------- match by start time

def test_find_matches_by_start_time_within_tolerance():
    idx = EmotionIndex([EmotionEntry(1000, 2000, 'happy'), EmotionEntry(5000, 6000, 'sad')])
    assert idx.find(1000).label == 'happy'
    assert idx.find(1299).label == 'happy'          # +299 ms
    assert idx.find(1300).label == 'happy'          # exactly the tolerance
    assert idx.find(1301) is None                   # just outside
    assert idx.find(4750).label == 'sad'            # before
    assert idx.find(3000) is None                   # between, too far from both


def test_find_picks_nearest_and_does_not_depend_on_order():
    entries = [EmotionEntry(2000, 2500, 'sad'), EmotionEntry(1000, 1500, 'happy'), EmotionEntry(1200, 1700, 'angry')]
    idx = EmotionIndex(entries)                      # deliberately unsorted input
    assert idx.find(1190).label == 'angry'
    assert idx.find(1090).label == 'happy'
    assert idx.find(1100).label == 'happy'           # tie (100 ms either side) -> earlier entry


def test_find_empty_index_and_dict_entries():
    assert EmotionIndex([]).find(0) is None
    assert not EmotionIndex([])
    idx = EmotionIndex([{'start_time': 10, 'end_time': 20, 'label': 'angry', 'confidence': 0.7}])
    assert idx.find(10).label == 'angry' and len(idx) == 1


def test_match_is_by_time_not_by_index():
    # the subtitle was edited: the first two lines were merged, so line indexes no longer line up
    entries = [EmotionEntry(0, 900, 'happy'), EmotionEntry(1000, 1900, 'neutral'), EmotionEntry(3000, 3900, 'angry')]
    final_lines = [_line(0, 1900), _line(3000, 3900)]
    idx = EmotionIndex(entries)
    assert [idx.find(ln['start_time']).label for ln in final_lines] == ['happy', 'angry']   # index 1 is NOT 'neutral'


def test_is_aligned_requires_one_to_one():
    entries = [EmotionEntry(0, 100), EmotionEntry(1000, 1100)]
    assert emo.is_aligned(entries, [_line(0), _line(1200)])                  # within tolerance
    assert not emo.is_aligned(entries, [_line(0)])                           # a line was merged away -> stale extra entry
    assert not emo.is_aligned(entries, [_line(0), _line(1400)])              # moved beyond tolerance
    assert not emo.is_aligned(entries, [_line(0), _line(1000), _line(2000)])  # a line was split
    assert not emo.is_aligned([], [_line(0)]) and not emo.is_aligned(entries, [])


def test_covers_lines():
    entries = [EmotionEntry(0, 100), EmotionEntry(1000, 1100)]
    assert emo.covers_lines(entries, [_line(0), _line(1100)])
    assert not emo.covers_lines(entries, [_line(0), _line(2000)])
    assert not emo.covers_lines(entries, [])
    assert not emo.covers_lines([], [_line(0)])


# ---------------------------------------------------------------- build from the SenseVoice capture

def test_build_entries_from_sensevoice_matches_by_time():
    raw = [{'start_time': 0, 'end_time': 800, 'tag': 'NEUTRAL'},
           {'start_time': 1000, 'end_time': 1800, 'tag': 'ANGRY'},
           {'start_time': 5000, 'end_time': 5800, 'tag': 'EMO_UNKNOWN'},
           {'start_time': 9000, 'end_time': 9800, 'tag': 'HAPPY'}]
    # the final subtitle dropped the segment at 1000 and shifted the one at 9000 by +200 ms
    lines = [_line(0, 800), _line(5000, 5800), _line(9200, 10000), _line(20000, 21000)]

    out = emo.build_entries_from_sensevoice(lines, raw)

    assert len(out) == len(lines)
    assert [(e.start_time, e.end_time) for e in out] == [(0, 800), (5000, 5800), (9200, 10000), (20000, 21000)]
    assert [e.label for e in out] == ['neutral', 'neutral', 'happy', 'neutral']
    assert [(e.intensity, e.confidence, e.source) for e in out] == [
        (0.0, 0.6, 'sensevoice'),    # NEUTRAL
        (0.0, 0.0, 'sensevoice'),    # EMO_UNKNOWN -> neutral / confidence 0
        (0.5, 0.6, 'sensevoice'),    # HAPPY, matched despite the 200 ms shift
        (0.0, 0.0, 'none'),          # nothing captured near this line
    ]


def test_build_entries_without_raw_is_all_unmatched():
    out = emo.build_entries_from_sensevoice([_line(0), _line(5000)], [])
    assert all(e.source == 'none' and e.label == 'neutral' and e.confidence == 0 for e in out)


def test_raw_capture_roundtrip(tmp_path):
    assert emo.load_raw_capture(str(tmp_path)) == []
    assert emo.save_raw_capture(str(tmp_path), []) is None
    emo.save_raw_capture(str(tmp_path), [{'start_time': 1, 'end_time': 2, 'tag': 'SAD'}])
    assert emo.load_raw_capture(str(tmp_path)) == [{'start_time': 1, 'end_time': 2, 'tag': 'SAD'}]


# ---------------------------------------------------------------- hint rows / table

def test_select_hint_rows_keeps_only_non_neutral_confident_matched_lines():
    entries = [EmotionEntry(0, 900, 'angry', 0.5, 0.6),
               EmotionEntry(1000, 1900, 'neutral', 0.0, 0.6),
               EmotionEntry(2000, 2900, 'sad', 0.5, 0.49),        # confidence just below 0.5
               EmotionEntry(3000, 3900, 'happy', 0.5, 0.5),       # exactly 0.5 -> kept
               EmotionEntry(4000, 4900, 'neutral', 0.0, 0.0)]
    keyed = [(1, 0), (2, 1000), (3, 2000), (4, 3000), (5, 4000), (6, 9999)]   # key 6 has no entry
    assert emo.select_hint_rows(keyed, entries) == [(1, 'angry', 0.5), (4, 'happy', 0.5)]
    assert emo.select_hint_rows(keyed, entries, min_confidence=0.0) == [
        (1, 'angry', 0.5), (3, 'sad', 0.5), (4, 'happy', 0.5)]                # neutral never listed


def test_format_hint_table():
    assert emo.format_hint_table([]) == ''
    assert emo.format_hint_table([(3, 'angry', 0.5), (10, 'sad', 0.7)]).splitlines() == [
        '| Line | Emotion | Intensity |', '| --- | --- | --- |', '| 3 | angry | 0.5 |', '| 10 | sad | 0.7 |']
