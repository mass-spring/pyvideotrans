"""EMOTION module part 1: EmotionMixin._ensure_emotion (idempotent emotion.json build) and its call from trans()."""
import json
import os
import types
from pathlib import Path

import pytest

from videotrans.task._stage_emotion import EmotionMixin
from videotrans.task._stage_translate import TranslateMixin
from videotrans.util import emotion as emo

RAW = [{'start_time': 0, 'end_time': 800, 'tag': 'NEUTRAL'},
       {'start_time': 1000, 'end_time': 1800, 'tag': 'ANGRY'},
       {'start_time': 2000, 'end_time': 2800, 'tag': 'HAPPY'}]


def _line(start, end=None, text='x'):
    return {'start_time': start, 'end_time': end or start + 800, 'text': text}


LINES = [_line(0), _line(1000), _line(2000)]


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class _Task(EmotionMixin):
    def __init__(self, tmp_path):
        self.cache = tmp_path / 'cache'
        self.target = tmp_path / 'out'
        self.cache.mkdir()
        self.target.mkdir()
        self.cfg = types.SimpleNamespace(cache_folder=self.cache.as_posix(), target_dir=self.target.as_posix(),
                                         source_sub=(self.target / 'ja.srt').as_posix())


@pytest.fixture
def setting(monkeypatch):
    def _set(**values):
        monkeypatch.setattr('videotrans.task._stage_emotion.settings', _Settings(**values))
    return _set


@pytest.fixture
def task(tmp_path):
    t = _Task(tmp_path)
    emo.save_raw_capture(t.cache.as_posix(), RAW)
    return t


ON = dict(enable_emotion=True, emotion_backend='sensevoice')


def test_mixin_is_part_of_trans_create():
    from videotrans.task.trans_create import TransCreate
    assert issubclass(TransCreate, EmotionMixin) and issubclass(TransCreate, TranslateMixin)
    assert callable(getattr(TransCreate, '_ensure_emotion'))


def test_disabled_by_default_does_nothing(task, setting):
    setting()                                                        # no enable_emotion at all
    assert task._ensure_emotion(LINES) == []
    setting(enable_emotion=False, emotion_backend='sensevoice')
    assert task._ensure_emotion(LINES) == []
    assert not (task.cache / 'emotion.json').exists()


@pytest.mark.parametrize('backend', ['none', '', 'NONE', 'unknown-backend'])
def test_backend_none_or_unknown_does_nothing(task, setting, backend):
    setting(enable_emotion=True, emotion_backend=backend)
    assert task._ensure_emotion(LINES) == []
    assert not (task.cache / 'emotion.json').exists()


def test_sensevoice_backend_builds_emotion_json_and_copies_to_target(task, setting):
    setting(**ON)
    entries = task._ensure_emotion(LINES)

    assert [(e.start_time, e.label, e.intensity, e.confidence, e.source) for e in entries] == [
        (0, 'neutral', 0.0, 0.6, 'sensevoice'),
        (1000, 'angry', 0.5, 0.6, 'sensevoice'),
        (2000, 'happy', 0.5, 0.6, 'sensevoice')]
    saved = json.loads((task.cache / 'emotion.json').read_text(encoding='utf-8'))
    assert [r['label'] for r in saved] == ['neutral', 'angry', 'happy']
    assert (task.target / 'emotion.json').read_text(encoding='utf-8') == (task.cache / 'emotion.json').read_text(encoding='utf-8')


def test_default_backend_is_sensevoice(task, setting):
    setting(enable_emotion=True)
    assert len(task._ensure_emotion(LINES)) == 3


def test_ensure_is_idempotent(task, setting, monkeypatch):
    setting(**ON)
    first = task._ensure_emotion(LINES)
    mtime = (task.cache / 'emotion.json').stat().st_mtime_ns
    monkeypatch.setattr(task, '_emotion_from_sensevoice', lambda lines: pytest.fail('must not rebuild'))

    second = task._ensure_emotion(LINES)

    assert second == first
    assert (task.cache / 'emotion.json').stat().st_mtime_ns == mtime


def test_text_edit_with_same_times_reuses_file(task, setting, monkeypatch):
    setting(**ON)
    task._ensure_emotion(LINES)
    monkeypatch.setattr(task, '_emotion_from_sensevoice', lambda lines: pytest.fail('must not rebuild'))
    edited = [_line(0, text='edited'), _line(1000, text='edited'), _line(2000, text='edited')]
    assert len(task._ensure_emotion(edited)) == 3


def test_edited_subtitle_times_rebuild_for_the_final_subtitles(task, setting):
    setting(**ON)
    task._ensure_emotion(LINES)
    # the user merged lines 1-2 and shifted the last one: the final subtitles are different now
    final = [_line(0, 1800), _line(2150, 3000)]

    entries = task._ensure_emotion(final)

    assert [(e.start_time, e.end_time, e.label) for e in entries] == [(0, 1800, 'neutral'), (2150, 3000, 'happy')]
    on_disk = emo.load_emotion((task.cache / 'emotion.json').as_posix())
    assert [e.start_time for e in on_disk] == [0, 2150]


def test_edited_subtitles_without_raw_capture_reuse_the_old_file_if_every_line_matches(task, setting):
    setting(**ON)
    task._ensure_emotion(LINES)
    (task.cache / 'emotion_raw.json').unlink()                         # nothing to rebuild from

    merged = [_line(0, 1800), _line(2100, 3000)]                       # lines 1-2 merged, last one shifted 100 ms
    assert [e.label for e in task._ensure_emotion(merged)] == ['neutral', 'angry', 'happy']   # old entries, matched by time
    assert task._ensure_emotion([_line(0), _line(50000)]) == []        # a line with no counterpart -> no hints at all


def test_file_made_by_another_backend_is_not_reused(task, setting):
    setting(**ON)
    emo.save_emotion([emo.EmotionEntry(l['start_time'], l['end_time'], 'sad', 0.5, 0.9, 'emotion2vec') for l in LINES],
                     task.cache.as_posix())
    assert {e.source for e in task._ensure_emotion(LINES)} == {'sensevoice'}


def test_new_asr_capture_triggers_rebuild(task, setting):
    setting(**ON)
    task._ensure_emotion(LINES)
    emo.save_raw_capture(task.cache.as_posix(), [dict(r, tag='SAD') for r in RAW])
    future = (task.cache / 'emotion.json').stat().st_mtime + 10
    os.utime(task.cache / 'emotion_raw.json', (future, future))        # raw is newer than emotion.json

    assert {e.label for e in task._ensure_emotion(LINES)} == {'sad'}


def test_no_raw_capture_gives_no_hints_and_no_file(task, setting):
    (task.cache / 'emotion_raw.json').unlink()
    setting(**ON)
    assert task._ensure_emotion(LINES) == []
    assert not (task.cache / 'emotion.json').exists()


def test_restores_from_target_dir_when_cache_is_gone(task, setting):
    setting(**ON)
    task._ensure_emotion(LINES)
    (task.cache / 'emotion.json').unlink()
    (task.cache / 'emotion_raw.json').unlink()                         # cache cleared, like a re-run from cached subtitles

    entries = task._ensure_emotion(LINES)

    assert [e.label for e in entries] == ['neutral', 'angry', 'happy']
    assert (task.cache / 'emotion.json').exists()


def test_reads_final_source_subtitles_when_not_given(task, setting):
    setting(**ON)
    Path(task.cfg.source_sub).write_text(
        '1\n00:00:00,000 --> 00:00:00,800\nhello\n\n2\n00:00:01,000 --> 00:00:01,800\nstop it\n', encoding='utf-8')
    entries = task._ensure_emotion()
    assert [(e.start_time, e.label) for e in entries] == [(0, 'neutral'), (1000, 'angry')]


def test_emotion2vec_is_a_stub_that_does_not_break_translation(task, setting):
    setting(enable_emotion=True, emotion_backend='emotion2vec')
    assert task._ensure_emotion(LINES) == []
    with pytest.raises(NotImplementedError):
        task._emotion_from_emotion2vec(LINES)
    assert not (task.cache / 'emotion.json').exists()


def test_errors_are_swallowed(task, setting, monkeypatch):
    setting(**ON)
    monkeypatch.setattr(task, '_emotion_from_sensevoice', lambda lines: 1 / 0)
    assert task._ensure_emotion(LINES) == []


# ---------------------------------------------------------------- called from trans()

class _Translator(TranslateMixin, EmotionMixin):
    """TranslateMixin.trans with the surrounding task machinery faked."""

    def __init__(self, tmp_path):
        self.cache = tmp_path / 'cache'
        self.target = tmp_path / 'out'
        self.cache.mkdir()
        self.target.mkdir()
        self.uuid = 'u'
        self.precent = 0
        self.should_trans = True
        self.should_dubbing = True
        self.signals = []
        src = self.target / 'ja.srt'
        src.write_text('1\n00:00:00,000 --> 00:00:00,800\nhello\n\n2\n00:00:01,000 --> 00:00:01,800\nstop it\n',
                       encoding='utf-8')
        self.cfg = types.SimpleNamespace(
            cache_folder=self.cache.as_posix(), target_dir=self.target.as_posix(), source_sub=src.as_posix(),
            target_sub=(self.target / 'en.srt').as_posix(), translate_type=3, source_language_code='ja',
            target_language_code='en', is_cuda=False, fix_punc=0, app_mode='', output_srt=0)

    def _exit(self):
        return False

    def signal(self, **kw):
        self.signals.append(kw)

    def check_target_sub(self, raw, target):
        return target

    def _save_srt_target(self, srt, path):
        self.saved = (srt, path)


def test_trans_builds_emotion_before_translating_and_passes_it(tmp_path, setting, monkeypatch):
    setting(**ON)
    t = _Translator(tmp_path)
    emo.save_raw_capture(t.cache.as_posix(), RAW[:2])
    order, got = [], {}

    def fake_run_trans(**kw):
        order.append('translate')
        got.update(kw)
        return kw['text_list']

    real = t._ensure_emotion
    t._ensure_emotion = lambda *a, **k: order.append('emotion') or real(*a, **k)
    monkeypatch.setattr('videotrans.task._stage_translate.run_trans', fake_run_trans)

    t.trans()

    assert order == ['emotion', 'translate']
    assert [e.label for e in got['emotion_entries']] == ['neutral', 'angry']
    assert (t.cache / 'emotion.json').exists() and (t.target / 'emotion.json').exists()


def test_trans_without_emotion_passes_empty_list(tmp_path, setting, monkeypatch):
    setting(enable_emotion=False)
    t = _Translator(tmp_path)
    got = {}
    monkeypatch.setattr('videotrans.task._stage_translate.run_trans', lambda **kw: got.update(kw) or kw['text_list'])
    t.trans()
    assert got['emotion_entries'] == []
    assert not (t.cache / 'emotion.json').exists()
