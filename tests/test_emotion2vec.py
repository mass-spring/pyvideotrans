"""
EMOTION module: emotion_backend = 'emotion2vec' (videotrans/process/_emotion2vec.py + EmotionMixin._emotion_from_emotion2vec).

No model is downloaded or loaded: a fake model "recognises" the emotion from the level of a constant signal, so the cutting of
lines, the label rules (core4 / all), short lines, the emotion.json round trip, the choice of backend by setting and the failure
path (model missing -> no emotion, translation goes on) can be checked on tiny synthetic audio.
"""
import json
import logging
import pickle
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from videotrans.process import _emotion2vec as e2v
from videotrans.task._stage_emotion import EmotionMixin
from videotrans.util import emotion as emo
from videotrans.util import emotion_tts
from videotrans.util.emotion import EmotionBackendError, EmotionEntry

SR = 16000
# the 9 classes in the order funasr returns them (tokens.txt of emotion2vec_plus_*)
E2V_LABELS = ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']


def raw_scores(**kw):
    """{class: score} -> the 9 values in E2V_LABELS order (the classes not given are 0)."""
    names = [lab.rsplit('/', 1)[-1] for lab in E2V_LABELS[:-1]] + ['unknown']
    return [float(kw.get(n, 0.0)) for n in names]


# the fake model reads the level of the signal: level k (mean |x| = 0.1 * k) -> the k-th score vector
LEVEL_SCORES = {
    1: raw_scores(neutral=0.90, sad=0.05, happy=0.05),
    2: raw_scores(happy=0.80, neutral=0.10, surprised=0.10),
    3: raw_scores(angry=0.70, neutral=0.20, disgusted=0.10),
    4: raw_scores(sad=0.60, neutral=0.30, fearful=0.10),
    5: raw_scores(fearful=0.60, sad=0.30, neutral=0.10),     # core4 picks sad (0.30), all picks fearful (0.60)
    6: raw_scores(other=0.70, neutral=0.20, happy=0.10),     # all: 'other' wins -> no emotion
    7: raw_scores(surprised=0.50, happy=0.40, neutral=0.10),
}


class FakeModel:
    def __init__(self, fail_levels=()):
        self.calls = []
        self.fail_levels = set(fail_levels)

    def generate(self, input=None, **kw):
        self.calls.append((input, kw))
        level = int(round(float(np.abs(input).mean()) * 10))
        if level in self.fail_levels:
            raise RuntimeError(f'boom at level {level}')
        return [{'key': 'rand_key_1', 'labels': list(E2V_LABELS), 'scores': list(LEVEL_SCORES[level])}]


def level(k, sec):
    return np.full(int(sec * SR), 0.1 * k, dtype=np.float32)


def write_wav(path, y, sr=SR, channels=1, width=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(y, dtype=np.float64), -1, 1)
    if width == 2:
        raw = (y * 32767).astype('<i2')
    else:  # 24 bit
        v = (y * 8388607).astype('<i4')
        raw = np.stack([(v >> 0) & 255, (v >> 8) & 255, (v >> 16) & 255], axis=1).astype(np.uint8)
    if channels > 1:
        raw = np.repeat(raw.reshape(len(y), -1), channels, axis=1)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(raw.tobytes())
    return path


def meeting_audio(tmp_path):
    """5 lines of 2 s each with 0.5 s of silence between them. Levels: neutral, happy, angry, sad, fearful+sad."""
    parts, spans, t = [], [], 0
    for k in (1, 2, 3, 4, 5):
        parts += [level(k, 2.0), np.zeros(SR // 2, dtype=np.float32)]
        spans.append([t, t + 2000])
        t += 2500
    return write_wav(tmp_path / 'ja.wav', np.concatenate(parts)), spans


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


# ---------------------------------------------------------------------------- settings
@pytest.mark.parametrize('value,expected', [('large', 'large'), ('BASE', 'base'), (' base ', 'base'), ('', 'large'), (None, 'large')])
def test_parse_model(value, expected):
    assert e2v.parse_model(value) == expected


@pytest.mark.parametrize('value', ['huge', 'iic/emotion2vec_plus_large', 'plus'])
def test_parse_model_rejects_unknown_with_japanese_message(value):
    with pytest.raises(EmotionBackendError, match='emotion2vec_model'):
        e2v.parse_model(value)


@pytest.mark.parametrize('value,expected', [('core4', 'core4'), ('ALL', 'all'), ('', 'core4'), (None, 'core4')])
def test_parse_label_mode(value, expected):
    assert e2v.parse_label_mode(value) == expected


def test_parse_label_mode_rejects_unknown():
    with pytest.raises(EmotionBackendError, match='emotion2vec_label_mode'):
        e2v.parse_label_mode('core3')


def test_parse_threads():
    assert e2v.parse_threads(2) == 2 and e2v.parse_threads('3') == 3 and e2v.parse_threads(' 1 ') == 1
    for auto in (0, '', None, '0'):
        assert 1 <= e2v.parse_threads(auto) <= e2v.AUTO_THREADS
    for bad in (-1, 'many'):
        with pytest.raises(EmotionBackendError, match='emotion2vec_threads'):
            e2v.parse_threads(bad)


def test_app_settings_defaults_keep_upstream_behaviour():
    from videotrans.configure._app_settings import AppSettings
    d = AppSettings._get_defaults(AppSettings.__new__(AppSettings))     # the defaults, without reading or writing cfg.json
    assert d['enable_emotion'] is False and d['emotion_backend'] == 'sensevoice'
    assert (d['emotion2vec_model'], d['emotion2vec_label_mode'], d['emotion2vec_threads']) == ('large', 'core4', 0)


# ---------------------------------------------------------------------------- class names and labels
@pytest.mark.parametrize('name,expected', [('生气/angry', 'angry'), ('中立/neutral', 'neutral'), ('<unk>', 'unknown'), ('Happy', 'happy'),
                                           ('其他/other', 'other'), ('吃惊/surprised', 'surprised')])
def test_canonical_class(name, expected):
    assert e2v.canonical_class(name) == expected


def test_canonical_class_rejects_unknown_name():
    with pytest.raises(ValueError):
        e2v.canonical_class('怒/rage')


def test_class_scores_maps_the_nine_classes():
    s = e2v.class_scores(E2V_LABELS, LEVEL_SCORES[3])
    assert set(s) == set(e2v.E2V_CLASSES) and s['angry'] == 0.7 and s['unknown'] == 0.0


@pytest.mark.parametrize('labels,values', [([], []), (E2V_LABELS, [0.1] * 8), (E2V_LABELS[:2], [0.5, float('nan')]), (E2V_LABELS[:1], ['x'])])
def test_class_scores_rejects_broken_output(labels, values):
    with pytest.raises(RuntimeError):
        e2v.class_scores(labels, values)


def test_core4_picks_the_best_of_the_four_plan_classes():
    s = e2v.class_scores(E2V_LABELS, LEVEL_SCORES[5])           # fearful 0.60 > sad 0.30 > neutral 0.10
    assert e2v.choose_label(s, 'core4') == ('sad', 0.3, 0.3)    # honest low confidence: below the 0.5 hint threshold
    assert e2v.choose_label(s, 'all') == ('fearful', 0.6, 0.6)


def test_core4_ignores_a_winning_surprised_class_and_all_keeps_it():
    s = e2v.class_scores(E2V_LABELS, LEVEL_SCORES[7])           # surprised 0.50 > happy 0.40
    assert e2v.choose_label(s, 'core4') == ('happy', 0.4, 0.4)
    assert e2v.choose_label(s, 'all') == ('surprised', 0.5, 0.5)


def test_neutral_has_zero_intensity_and_its_score_as_confidence():
    s = e2v.class_scores(E2V_LABELS, LEVEL_SCORES[1])
    assert e2v.choose_label(s, 'core4') == ('neutral', 0.0, 0.9)
    assert e2v.choose_label(s, 'all') == ('neutral', 0.0, 0.9)


def test_other_or_unknown_winning_in_all_mode_means_no_emotion():
    s = e2v.class_scores(E2V_LABELS, LEVEL_SCORES[6])           # other 0.70
    assert e2v.choose_label(s, 'all') == ('neutral', 0.0, 0.0)
    assert e2v.choose_label(s, 'core4') == ('neutral', 0.0, 0.2)
    assert e2v.choose_label({'unknown': 1.0}, 'all') == ('neutral', 0.0, 0.0)


def test_ties_go_to_neutral_first_and_a_missing_class_counts_as_zero():
    assert e2v.choose_label({'neutral': 0.4, 'happy': 0.4, 'angry': 0.4, 'sad': 0.4}, 'core4')[0] == 'neutral'
    assert e2v.choose_label({'sad': 0.2}, 'core4')[0] == 'sad'
    assert e2v.choose_label({}, 'core4') == ('neutral', 0.0, 0.0)


def test_labels_are_always_labels_the_module_knows():
    for k, vals in LEVEL_SCORES.items():
        s = e2v.class_scores(E2V_LABELS, vals)
        for mode in e2v.LABEL_MODES:
            assert e2v.choose_label(s, mode)[0] in emo.LABELS


def test_make_entry_keeps_the_scores_and_how_it_was_made():
    e = e2v.make_entry(1000, 3000, e2v.class_scores(E2V_LABELS, LEVEL_SCORES[3]), 'core4', 'emotion2vec_plus_large core4')
    assert (e.start_time, e.end_time, e.label, e.intensity, e.confidence, e.source) == (1000, 3000, 'angry', 0.7, 0.7, 'emotion2vec')
    assert e.scores['angry'] == 0.7 and set(e.scores) == set(e2v.E2V_CLASSES) and e.info == 'emotion2vec_plus_large core4'


def test_label_is_chosen_from_the_stored_rounded_scores():
    # surprised wins by far and the four PLAN classes are all ~0: after rounding to 6 digits they tie -> neutral, confidence 0
    # (so the label can always be re-derived from emotion.json, and no arbitrary "angry" is made out of 1e-7 differences)
    s = {'neutral': 1e-7, 'happy': 4e-7, 'angry': 2e-7, 'sad': 0.0, 'surprised': 0.999999}
    e = e2v.make_entry(0, 1000, s, 'core4', 'i')
    assert (e.label, e.intensity, e.confidence) == ('neutral', 0.0, 0.0)
    assert e.scores['happy'] == 0.0 and e.scores['surprised'] == 0.999999
    assert e2v.make_entry(0, 1000, s, 'all', 'i').label == 'surprised'
    for mode in e2v.LABEL_MODES:                                      # re-deriving from the stored scores gives the same label
        e = e2v.make_entry(0, 1000, e2v.class_scores(E2V_LABELS, LEVEL_SCORES[5]), mode, 'i')
        assert e2v.choose_label(e.scores, mode)[0] == e.label


# ---------------------------------------------------------------------------- emotion.json
def test_emotion_json_round_trip_keeps_scores_and_info(tmp_path):
    entries = [e2v.make_entry(0, 2000, e2v.class_scores(E2V_LABELS, LEVEL_SCORES[3]), 'core4', 'emotion2vec_plus_base core4'),
               e2v.none_entry(2500, 2700, 'emotion2vec_plus_base core4')]
    emo.save_emotion(entries, (tmp_path / 'cache').as_posix(), None)
    loaded = emo.load_emotion((tmp_path / 'cache' / 'emotion.json').as_posix())
    assert loaded == entries
    rows = json.loads((tmp_path / 'cache' / 'emotion.json').read_text(encoding='utf-8'))
    assert 'scores' in rows[0] and 'scores' not in rows[1]          # the "not detected" line carries no scores
    assert rows[1]['source'] == 'none' and rows[1]['confidence'] == 0.0


def test_entries_without_the_new_fields_look_and_load_as_before():
    sv = EmotionEntry(12340, 15880, 'angry', 0.5, 0.6, 'sensevoice')
    assert set(sv.to_dict()) == {'start_time', 'end_time', 'label', 'intensity', 'confidence', 'source'}
    old = {'start_time': 1, 'end_time': 2, 'label': 'sad', 'intensity': 0.5, 'confidence': 0.6, 'source': 'sensevoice'}
    e = EmotionEntry.from_dict(old)
    assert e.scores is None and e.info == '' and e == EmotionEntry(1, 2, 'sad', 0.5, 0.6, 'sensevoice')


@pytest.mark.parametrize('bad', ['x', [1, 2], 5, {'a': 'b', 'c': True, 'd': float('nan')}, {}])
def test_broken_scores_in_emotion_json_are_dropped(bad):
    e = EmotionEntry.from_dict({'start_time': 0, 'end_time': 1, 'label': 'happy', 'scores': bad})
    assert e.scores is None and e.label == 'happy'


def test_scores_from_emotion_json_are_numbers():
    e = EmotionEntry.from_dict({'label': 'happy', 'scores': {'happy': 0.8, 'junk': 'x', 'sad': 1}})
    assert e.scores == {'happy': 0.8, 'sad': 1.0}


def test_translation_hints_and_tts_use_the_entries_unchanged(tmp_path):
    spans = [[0, 2000], [2500, 4500], [5000, 7000], [7500, 9500], [10000, 12000]]
    model = FakeModel()
    wav, _ = meeting_audio(tmp_path)
    reader = e2v.SegmentReader(wav)
    entries = e2v.analyse_lines(spans, reader, model, 'core4', 'i')
    reader.close()
    assert [e.label for e in entries] == ['neutral', 'happy', 'angry', 'sad', 'sad']

    # translation prompt: neutral and low-confidence (sad 0.30) lines are left out
    rows = emo.select_hint_rows([(1, 0), (2, 2500), (3, 5000), (4, 7500), (5, 10000)], entries)
    assert rows == [(2, 'happy', 0.8), (3, 'angry', 0.7), (4, 'sad', 0.6)]
    assert '| 3 | angry | 0.7 |' in emo.format_hint_table(rows)

    # TTS: same rule (confidence >= 0.5, label with a style)
    idx = emo.EmotionIndex(entries)
    assert emotion_tts.find_line_emotion(idx, 5000) == {'emotion': 'angry', 'emotion_intensity': 0.7, 'emotion_confidence': 0.7}
    assert emotion_tts.find_line_emotion(idx, 0) == {} and emotion_tts.find_line_emotion(idx, 10000) == {}


# ---------------------------------------------------------------------------- reading the audio
@pytest.fixture(params=['soundfile', 'stdlib'])
def reader_backend(request, monkeypatch):
    if request.param == 'soundfile':
        pytest.importorskip('soundfile')
    else:
        monkeypatch.setitem(sys.modules, 'soundfile', None)         # `import soundfile` raises ImportError -> wave module
    return request.param


def test_segment_reader_cuts_16k_mono(tmp_path, reader_backend):
    y = (0.5 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)).astype(np.float32)
    r = e2v.SegmentReader(write_wav(tmp_path / 'a.wav', y))
    assert (r.sr, r.channels, r.frames, r.duration_ms) == (SR, 1, SR * 3, 3000)
    seg = r.read(500, 1500)
    assert seg.dtype == np.float32 and seg.shape == (SR,)
    assert np.allclose(seg, y[SR // 2:SR // 2 + SR], atol=1e-3)
    assert len(r.read(2500, 9000)) == SR // 2                        # runs past the end -> what exists
    assert len(r.read(5000, 6000)) == 0 and len(r.read(1000, 1000)) == 0
    r.close()


def test_segment_reader_downmixes_and_resamples(tmp_path, reader_backend):
    y = (0.5 * np.sin(2 * np.pi * 200 * np.arange(8000 * 2) / 8000)).astype(np.float32)
    r = e2v.SegmentReader(write_wav(tmp_path / 'b.wav', y, sr=8000, channels=2))
    seg = r.read(0, 1000)
    assert seg.ndim == 1 and abs(len(seg) - SR) <= 1
    assert abs(float(np.abs(seg).max()) - 0.5) < 0.05
    r.close()


def test_segment_reader_reads_24_bit(tmp_path):
    y = np.full(SR, 0.25, dtype=np.float32)
    seg = e2v.SegmentReader(write_wav(tmp_path / 'c.wav', y, width=3)).read(0, 500)
    assert len(seg) == SR // 2 and abs(float(seg.mean()) - 0.25) < 1e-3


def test_segment_reader_missing_file_is_a_japanese_backend_error(tmp_path):
    with pytest.raises(EmotionBackendError, match='音声ファイル'):
        e2v.SegmentReader(tmp_path / 'nope.wav')


# ---------------------------------------------------------------------------- the per-line analysis
def test_lines_are_cut_from_the_audio_and_sent_with_the_agreed_arguments(tmp_path):
    wav, spans = meeting_audio(tmp_path)
    model = FakeModel()
    reader = e2v.SegmentReader(wav)
    entries = e2v.analyse_lines(spans, reader, model, 'core4', 'emotion2vec_plus_large core4')
    reader.close()

    assert len(model.calls) == 5
    for (data, kw), (s, e) in zip(model.calls, spans):
        assert isinstance(data, np.ndarray) and data.dtype == np.float32 and data.ndim == 1
        assert len(data) == (e - s) * SR // 1000
        assert kw == {'granularity': 'utterance', 'extract_embedding': False, 'disable_pbar': True}
    assert [(x.start_time, x.end_time) for x in entries] == [tuple(s) for s in spans]
    assert [x.label for x in entries] == ['neutral', 'happy', 'angry', 'sad', 'sad']
    assert {x.source for x in entries} == {'emotion2vec'}


def test_core4_and_all_differ_only_where_other_classes_win(tmp_path):
    wav, spans = meeting_audio(tmp_path)
    model = FakeModel()
    reader = e2v.SegmentReader(wav)
    core4 = e2v.analyse_lines(spans, reader, model, 'core4', 'x')
    allc = e2v.analyse_lines(spans, reader, model, 'all', 'x')
    reader.close()
    assert [e.label for e in core4] == ['neutral', 'happy', 'angry', 'sad', 'sad']
    assert [e.label for e in allc] == ['neutral', 'happy', 'angry', 'sad', 'fearful']
    assert [e.scores for e in core4] == [e.scores for e in allc]     # same scores, only the rule differs


def test_short_lines_are_neutral_with_zero_confidence_and_do_not_reach_the_model(tmp_path):
    wav = write_wav(tmp_path / 'ja.wav', np.concatenate([level(3, 2.0)]))
    model = FakeModel()
    reader = e2v.SegmentReader(wav)
    entries = e2v.analyse_lines([[0, 400], [500, 999], [1000, 1500]], reader, model, 'core4', 'i')
    reader.close()
    assert len(model.calls) == 1                                      # only the 0.5 s line is long enough
    assert [(e.label, e.confidence, e.source) for e in entries[:2]] == [('neutral', 0.0, 'none')] * 2
    assert entries[2].label == 'angry' and entries[2].source == 'emotion2vec'
    assert entries[0].scores is None and entries[0].info == 'i'


def test_lines_outside_the_audio_are_neutral_and_not_analysed(tmp_path):
    wav = write_wav(tmp_path / 'ja.wav', level(1, 2.0))
    model = FakeModel()
    reader = e2v.SegmentReader(wav)
    entries = e2v.analyse_lines([[0, 2000], [30000, 32000]], reader, model, 'core4', 'i')
    reader.close()
    assert len(model.calls) == 1 and entries[1].source == 'none' and entries[1].confidence == 0.0


def test_a_failing_line_becomes_neutral_and_the_others_survive(tmp_path, caplog):
    wav, spans = meeting_audio(tmp_path)
    reader = e2v.SegmentReader(wav)
    with caplog.at_level(logging.WARNING):
        entries = e2v.analyse_lines(spans, reader, FakeModel(fail_levels=[3]), 'core4', 'i')
    reader.close()
    assert [e.label for e in entries] == ['neutral', 'happy', 'neutral', 'sad', 'sad']
    assert entries[2].source == 'none' and entries[2].confidence == 0.0
    assert '行 3' in caplog.text


def test_all_lines_failing_is_an_error(tmp_path):
    wav, spans = meeting_audio(tmp_path)
    reader = e2v.SegmentReader(wav)
    with pytest.raises(RuntimeError, match='全ての行で失敗'):
        e2v.analyse_lines(spans, reader, FakeModel(fail_levels=[1, 2, 3, 4, 5]), 'core4', 'i')
    reader.close()


def test_progress_is_reported_for_every_line(tmp_path):
    wav, spans = meeting_audio(tmp_path)
    seen = []
    reader = e2v.SegmentReader(wav)
    e2v.analyse_lines(spans, reader, FakeModel(), 'core4', 'i', progress=lambda d, t: seen.append((d, t)))
    reader.close()
    assert seen == [(i, 5) for i in range(1, 6)]


def test_long_lines_are_split_and_the_scores_averaged(tmp_path):
    # 70 s = 3 chunks of 23.3 s: the first two are "happy", the last one "angry" -> happy 2/3, angry 1/3
    y = np.concatenate([level(2, 46.6667), level(3, 23.3333)])
    model = FakeModel()
    scores = e2v.predict_scores(model, y)
    assert len(model.calls) == 3 and all(len(c[0]) <= e2v.MAX_CHUNK_SEC * SR for c in model.calls)
    assert sum(len(c[0]) for c in model.calls) == len(y)
    assert scores['happy'] == pytest.approx(0.8 * 2 / 3, abs=0.02) and scores['angry'] == pytest.approx(0.7 / 3, abs=0.02)
    one = FakeModel()
    e2v.predict_scores(one, level(1, 30.0))
    assert len(one.calls) == 1                                        # exactly 30 s is not split


# ---------------------------------------------------------------------------- the subprocess entry
def _run(tmp_path, spans, **kw):
    wav, _ = meeting_audio(tmp_path)
    lines = tmp_path / 'lines.json'
    lines.write_text(json.dumps(spans), encoding='utf-8')
    base = dict(input_file=wav.as_posix(), subtitles_file=lines.as_posix(), model_size='large', label_mode='core4', num_threads=2,
                logs_file=(tmp_path / 'p.log').as_posix(), model=FakeModel())
    base.update(kw)
    return e2v.emotion2vec_lines(**base)


def test_entry_function_returns_json_ready_rows_in_line_order(tmp_path):
    rows, err = _run(tmp_path, [[0, 2000], [2500, 4500], [5000, 7000]])
    assert err is None and len(rows) == 3
    json.dumps(rows)                                                  # picklable / JSON-able plain dicts
    assert [r['label'] for r in rows] == ['neutral', 'happy', 'angry']
    assert rows[1]['source'] == 'emotion2vec' and rows[1]['info'] == 'emotion2vec_plus_large core4' and 'scores' in rows[1]
    assert json.loads((tmp_path / 'p.log').read_text(encoding='utf-8'))['text'].startswith('emotion2vec+')   # progress for the UI


def test_entry_function_label_mode_all(tmp_path):
    rows, err = _run(tmp_path, [[10000, 12000]], label_mode='all', model_size='base')
    assert err is None and rows[0]['label'] == 'fearful' and rows[0]['info'] == 'emotion2vec_plus_base all'


@pytest.mark.parametrize('kw,needle', [
    ({'model_size': 'huge'}, 'emotion2vec_model'),
    ({'label_mode': 'core3'}, 'emotion2vec_label_mode'),
    ({'input_file': 'missing.wav'}, '音声ファイル'),
    ({'model': None}, 'モデルのファイルが足りません'),                 # no model object and an empty model folder
])
def test_entry_function_failures_come_back_as_false_plus_reason(tmp_path, kw, needle):
    if kw.get('model', 1) is None:
        kw['model_dir'] = (tmp_path / 'empty').as_posix()
    if 'input_file' in kw:
        kw['input_file'] = (tmp_path / kw['input_file']).as_posix()
    rows, err = _run(tmp_path, [[0, 2000]], **kw)
    assert rows is False and needle in err


def test_device_choice_follows_stt_funasr():
    assert e2v.pick_device('auto', False) == 'cpu' and e2v.pick_device('auto', True, 1) == 'cuda:1'
    assert e2v.pick_device('cuda:0', False) == 'cuda:0' and e2v.pick_device('', False) == 'cpu'


def test_load_model_passes_ncpu_and_the_local_folder(tmp_path, monkeypatch):
    seen = {}
    fake_funasr = types.ModuleType('funasr')
    fake_funasr.AutoModel = lambda **kw: seen.update(kw) or 'MODEL'
    fake_torch = types.ModuleType('torch')
    fake_torch.set_num_threads = lambda n: seen.update(torch_threads=n)
    monkeypatch.setitem(sys.modules, 'funasr', fake_funasr)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    assert e2v.load_model(tmp_path, 'cpu', 2) == 'MODEL'
    assert seen['model'] == str(tmp_path) and seen['ncpu'] == 2 and seen['torch_threads'] == 2
    assert seen['device'] == 'cpu' and seen['disable_update'] is True and seen['hub'] == 'ms'


def test_load_model_without_funasr_is_a_japanese_backend_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setitem(sys.modules, 'funasr', None)
    with pytest.raises(EmotionBackendError, match='funasr'):
        e2v.load_model('x', 'cpu', 1)


# ---------------------------------------------------------------------------- model download
def _fake_help_down(monkeypatch, fn):
    mod = types.ModuleType('videotrans.util.help_down')
    mod.check_and_down_ms = fn
    monkeypatch.setitem(sys.modules, 'videotrans.util.help_down', mod)


def test_ensure_model_uses_the_project_modelscope_downloader_for_the_right_id(tmp_path, monkeypatch):
    monkeypatch.setattr(e2v, 'ROOT_DIR', tmp_path.as_posix())
    calls = []

    def fake_down(model_id, callback=None, local_dir=None, allow_patterns=None):
        calls.append((model_id, local_dir))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for n in e2v.MODEL_FILES:
            (Path(local_dir) / n).write_text('x')
        return True

    _fake_help_down(monkeypatch, fake_down)
    assert e2v.ensure_model('base') == tmp_path / 'models' / 'emotion2vec_plus_base'
    assert e2v.ensure_model('large') == tmp_path / 'models' / 'emotion2vec_plus_large'
    assert calls == [('iic/emotion2vec_plus_base', (tmp_path / 'models' / 'emotion2vec_plus_base').as_posix()),
                     ('iic/emotion2vec_plus_large', (tmp_path / 'models' / 'emotion2vec_plus_large').as_posix())]


def test_ensure_model_download_failure_is_a_japanese_backend_error(tmp_path, monkeypatch):
    monkeypatch.setattr(e2v, 'ROOT_DIR', tmp_path.as_posix())

    def down(**kw):
        raise ConnectionError('network is down')

    _fake_help_down(monkeypatch, down)
    with pytest.raises(EmotionBackendError, match='モデル.*用意できませんでした.*network is down'):
        e2v.ensure_model('large')


def test_ensure_model_incomplete_download_is_a_backend_error(tmp_path, monkeypatch):
    monkeypatch.setattr(e2v, 'ROOT_DIR', tmp_path.as_posix())
    _fake_help_down(monkeypatch, lambda **kw: True)                   # "succeeds" but leaves no files
    with pytest.raises(EmotionBackendError, match='model.pt'):
        e2v.ensure_model('large')


# ---------------------------------------------------------------------------- the stage chooses the backend by setting
class _Task(EmotionMixin):
    """EmotionMixin with the task machinery faked: _new_process runs the real entry function in-process with a fake model."""

    def __init__(self, tmp_path, audio=None):
        self.cache = tmp_path / 'cache'
        self.target = tmp_path / 'out'
        self.cache.mkdir()
        self.target.mkdir()
        self.cfg = types.SimpleNamespace(cache_folder=self.cache.as_posix(), target_dir=self.target.as_posix(),
                                         source_sub=(self.target / 'ja.srt').as_posix(),
                                         source_wav=Path(audio).as_posix() if audio else None, is_cuda=False)
        self.process_calls = []
        self.model = FakeModel()

    def _new_process(self, callback=None, title='', is_cuda=False, kwargs=None):
        self.process_calls.append({'callback': callback, 'title': title, 'is_cuda': is_cuda, 'kwargs': dict(kwargs)})
        data, err = callback(**kwargs, model=self.model)
        if err or not data:
            raise RuntimeError(err)
        return data


def _line(span, text='x'):
    return {'start_time': span[0], 'end_time': span[1], 'text': text}


@pytest.fixture
def stage(tmp_path, monkeypatch):
    wav, spans = meeting_audio(tmp_path)
    task = _Task(tmp_path, wav)
    lines = [_line(s) for s in spans]
    models = []

    def set_(**values):
        values.setdefault('enable_emotion', True)
        monkeypatch.setattr('videotrans.task._stage_emotion.settings', _Settings(**values))

    def fake_ensure(size, callback=None):
        models.append(size)
        return tmp_path / f'models-{size}'

    monkeypatch.setattr(e2v, 'ensure_model', fake_ensure)
    return types.SimpleNamespace(task=task, lines=lines, spans=spans, set=set_, models=models)


def test_emotion2vec_backend_builds_emotion_json_from_the_audio(stage):
    stage.set(emotion_backend='emotion2vec')
    entries = stage.task._ensure_emotion(stage.lines)

    assert [e.label for e in entries] == ['neutral', 'happy', 'angry', 'sad', 'sad']
    assert [(e.start_time, e.end_time) for e in entries] == [tuple(s) for s in stage.spans]
    assert {e.source for e in entries} == {'emotion2vec'} and all(e.scores for e in entries)
    saved = emo.load_emotion((stage.task.cache / 'emotion.json').as_posix())
    assert saved == entries
    assert (stage.task.target / 'emotion.json').read_text(encoding='utf-8') == (stage.task.cache / 'emotion.json').read_text(encoding='utf-8')
    assert stage.models == ['large']


def test_process_is_started_with_the_agreed_arguments(stage):
    stage.set(emotion_backend='emotion2vec', emotion2vec_model='base', emotion2vec_label_mode='all', emotion2vec_threads=3)
    stage.task._ensure_emotion(stage.lines)
    call, = stage.task.process_calls
    kw = call['kwargs']
    assert call['callback'] is e2v.emotion2vec_lines and 'emotion2vec+' in call['title'] and call['is_cuda'] is False
    assert (kw['model_size'], kw['label_mode'], kw['num_threads'], kw['is_cuda']) == ('base', 'all', 3, False)
    assert kw['input_file'] == stage.task.cfg.source_wav and kw['model_dir'].endswith('models-base')
    assert not list(stage.task.cache.glob('emotion2vec-lines-*.json'))        # temporary line list is removed
    assert pickle.loads(pickle.dumps(call['callback'])) is e2v.emotion2vec_lines and pickle.loads(pickle.dumps(kw)) == kw   # the process pool pickles both
    assert stage.models == ['base']


def test_default_settings_use_large_and_core4(stage):
    stage.set(emotion_backend='emotion2vec')
    stage.task._ensure_emotion(stage.lines)
    kw = stage.task.process_calls[0]['kwargs']
    assert (kw['model_size'], kw['label_mode']) == ('large', 'core4') and 1 <= kw['num_threads'] <= 4


def test_sensevoice_backend_does_not_touch_the_model(stage):
    emo.save_raw_capture(stage.task.cache.as_posix(), [{'start_time': s[0], 'end_time': s[1], 'tag': 'ANGRY'} for s in stage.spans])
    stage.set(emotion_backend='sensevoice')
    entries = stage.task._ensure_emotion(stage.lines)
    assert {e.source for e in entries} == {'sensevoice'} and {e.label for e in entries} == {'angry'}
    assert not stage.task.process_calls and not stage.models
    assert 'scores' not in json.loads((stage.task.cache / 'emotion.json').read_text(encoding='utf-8'))[0]
    stage.set()                                                       # nothing set: sensevoice stays the default
    assert stage.task._ensure_emotion(stage.lines)[0].source == 'sensevoice' and not stage.task.process_calls


def test_disabled_or_none_does_nothing(stage):
    stage.set(emotion_backend='emotion2vec', enable_emotion=False)
    assert stage.task._ensure_emotion(stage.lines) == []
    stage.set(emotion_backend='none')
    assert stage.task._ensure_emotion(stage.lines) == []
    assert not stage.task.process_calls and not (stage.task.cache / 'emotion.json').exists()


def test_second_call_reuses_emotion_json(stage):
    stage.set(emotion_backend='emotion2vec')
    first = stage.task._ensure_emotion(stage.lines)
    edited = [_line(s, text='edited') for s in stage.spans]           # text edits keep the times
    assert stage.task._ensure_emotion(edited) == first
    assert len(stage.task.process_calls) == 1


def test_changed_model_or_label_mode_rebuilds_but_same_settings_do_not(stage):
    stage.set(emotion_backend='emotion2vec', emotion2vec_label_mode='core4')
    assert stage.task._ensure_emotion(stage.lines)[4].label == 'sad'
    stage.set(emotion_backend='emotion2vec', emotion2vec_label_mode='core4')
    stage.task._ensure_emotion(stage.lines)
    assert len(stage.task.process_calls) == 1
    stage.set(emotion_backend='emotion2vec', emotion2vec_label_mode='all')
    assert stage.task._ensure_emotion(stage.lines)[4].label == 'fearful'
    assert len(stage.task.process_calls) == 2
    stage.set(emotion_backend='emotion2vec', emotion2vec_label_mode='all', emotion2vec_model='base')
    stage.task._ensure_emotion(stage.lines)
    assert len(stage.task.process_calls) == 3 and stage.models == ['large', 'large', 'base']


def test_switching_backend_rebuilds_with_the_new_one(stage):
    emo.save_raw_capture(stage.task.cache.as_posix(), [{'start_time': s[0], 'end_time': s[1], 'tag': 'HAPPY'} for s in stage.spans])
    stage.set(emotion_backend='emotion2vec')
    assert {e.source for e in stage.task._ensure_emotion(stage.lines)} == {'emotion2vec'}
    stage.set(emotion_backend='sensevoice')
    assert {e.source for e in stage.task._ensure_emotion(stage.lines)} == {'sensevoice'}
    stage.set(emotion_backend='emotion2vec')
    assert {e.source for e in stage.task._ensure_emotion(stage.lines)} == {'emotion2vec'}
    assert len(stage.task.process_calls) == 2


def test_edited_subtitle_times_are_analysed_again_from_the_audio(stage):
    stage.set(emotion_backend='emotion2vec')
    stage.task._ensure_emotion(stage.lines)
    merged = [_line([0, 4500]), _line([5000, 7000])]                  # lines 1-2 merged
    entries = stage.task._ensure_emotion(merged)
    assert [(e.start_time, e.end_time) for e in entries] == [(0, 4500), (5000, 7000)]
    assert len(stage.task.process_calls) == 2


def test_no_lines_gives_nothing_and_no_process(stage):
    stage.set(emotion_backend='emotion2vec')
    assert stage.task._emotion_from_emotion2vec([]) == [] and not stage.task.process_calls


# ---- failure paths: no emotion, translation goes on, the reason is in the log (Japanese)
def test_missing_model_falls_back_to_no_emotion(tmp_path, monkeypatch, caplog):
    wav, spans = meeting_audio(tmp_path)
    task = _Task(tmp_path, wav)
    monkeypatch.setattr('videotrans.task._stage_emotion.settings', _Settings(enable_emotion=True, emotion_backend='emotion2vec'))
    monkeypatch.setattr(e2v, 'ROOT_DIR', tmp_path.as_posix())

    def down(**kw):
        raise ConnectionError('offline')

    _fake_help_down(monkeypatch, down)
    with caplog.at_level(logging.WARNING):
        assert task._ensure_emotion([_line(s) for s in spans]) == []
    assert 'emotion2vec+ のモデル' in caplog.text and '感情ヒントなしで続行' in caplog.text
    assert not task.process_calls and not (task.cache / 'emotion.json').exists()
    assert 'Traceback' not in caplog.text                              # an expected condition, not a crash


@pytest.mark.parametrize('audio', [None, 'missing.wav'])
def test_missing_audio_falls_back_to_no_emotion(tmp_path, monkeypatch, caplog, audio):
    task = _Task(tmp_path, tmp_path / audio if audio else None)
    monkeypatch.setattr('videotrans.task._stage_emotion.settings', _Settings(enable_emotion=True, emotion_backend='emotion2vec'))
    monkeypatch.setattr(e2v, 'ensure_model', lambda *a, **k: pytest.fail('must not download without audio'))
    with caplog.at_level(logging.WARNING):
        assert task._ensure_emotion([_line([0, 2000])]) == []
    assert '音声ファイルがありません' in caplog.text and '感情ヒントなしで続行' in caplog.text


@pytest.mark.parametrize('values,needle', [({'emotion2vec_model': 'huge'}, 'emotion2vec_model'),
                                           ({'emotion2vec_label_mode': 'x'}, 'emotion2vec_label_mode'),
                                           ({'emotion2vec_threads': -2}, 'emotion2vec_threads')])
def test_wrong_setting_value_falls_back_to_no_emotion_with_a_message(stage, caplog, values, needle):
    stage.set(emotion_backend='emotion2vec', **values)
    with caplog.at_level(logging.WARNING):
        assert stage.task._ensure_emotion(stage.lines) == []
    assert needle in caplog.text and not stage.task.process_calls


def test_process_failure_falls_back_to_no_emotion(stage, caplog):
    stage.set(emotion_backend='emotion2vec')
    stage.task.model = FakeModel(fail_levels=[1, 2, 3, 4, 5])
    with caplog.at_level(logging.WARNING):
        assert stage.task._ensure_emotion(stage.lines) == []
    assert '感情ヒントなしで続行' in caplog.text and not (stage.task.cache / 'emotion.json').exists()


def test_user_abort_returns_nothing(stage):
    stage.set(emotion_backend='emotion2vec')
    stage.task._new_process = lambda **kw: None                      # _new_process returns None when the user stops the task
    assert stage.task._ensure_emotion(stage.lines) == [] and not (stage.task.cache / 'emotion.json').exists()


def test_emotion_backend_error_is_not_a_crash_even_if_raised_late(stage, caplog):
    stage.set(emotion_backend='emotion2vec')
    stage.task._new_process = lambda **kw: (_ for _ in ()).throw(EmotionBackendError('funasr / torch が入っていません'))
    with caplog.at_level(logging.WARNING):
        assert stage.task._ensure_emotion(stage.lines) == []
    assert 'funasr' in caplog.text


def test_backend_names_include_emotion2vec():
    from videotrans.task._stage_emotion import EMOTION_BACKENDS
    assert set(EMOTION_BACKENDS) == {'sensevoice', 'emotion2vec', 'none'}
