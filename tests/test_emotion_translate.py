"""EMOTION module part 1: emotion hints in LLM translation prompts, per-batch hint tables, cache key."""
import hashlib
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from videotrans.configure.config import ROOT_DIR
from videotrans.task.taskcfg import SrtItem
from videotrans.translator._base import BaseTrans
from videotrans.util.emotion import EmotionEntry, EmotionIndex

PROMPT_DIR = Path(ROOT_DIR) / 'videotrans' / 'prompts'

# Lines 1..5 start at 0, 1000, 2000, 3000, 4000 ms. Emotion per line (see EMOTIONS):
#   1 neutral, 2 angry (0.5 / conf 0.6), 3 sad but low confidence, 4 happy, 5 no entry at all
EMOTIONS = [
    EmotionEntry(0, 900, 'neutral', 0.0, 0.6, 'sensevoice'),
    EmotionEntry(1000, 1900, 'angry', 0.5, 0.6, 'sensevoice'),
    EmotionEntry(2000, 2900, 'sad', 0.5, 0.4, 'sensevoice'),
    EmotionEntry(3000, 3900, 'happy', 0.5, 0.6, 'sensevoice'),
]


def _items(n=5, first_line=1):
    out = []
    for i in range(n):
        s = i * 1000
        out.append(SrtItem(text=f'src{first_line + i}', line=first_line + i, start_time=s, end_time=s + 900,
                           startraw=f'00:00:{i:02d},000', endraw=f'00:00:{i:02d},900',
                           time=f'00:00:{i:02d},000 --> 00:00:{i:02d},900'))
    return out


class _FakeSettings:
    """Minimal stand-in for `settings`, so the tests do not depend on the local cfg.json."""
    values = {'aisendsrt': False, 'trans_thread': 5, 'aitrans_thread': 20, 'aitrans_context': False}

    def get(self, key, default=None):
        return self.values.get(key, default)


@pytest.fixture
def make_trans(monkeypatch, tmp_path):
    monkeypatch.setattr('videotrans.translator._base.settings', _FakeSettings())
    monkeypatch.setattr('videotrans.translator._base.TEMP_ROOT', str(tmp_path))
    (tmp_path / 'translate_cache').mkdir()

    def _make(srt_mode, emotions=EMOTIONS, n=5, thread=2, ainame='chatgpt', target='fr'):
        bt = BaseTrans(text_list=_items(n), source_code='ja', target_code='French', common_target_code=target,
                       emotion=EmotionIndex(emotions) if emotions else None)
        bt.ainame = ainame
        bt.aisendsrt = srt_mode
        bt.trans_thread = thread
        bt.wait_sec = 0
        bt.signal = lambda **kw: None
        bt.prompt = bt._set_context()
        return bt

    return _make


def _batches(bt):
    if bt.aisendsrt:
        return [bt.text_list[i:i + bt.trans_thread] for i in range(0, len(bt.text_list), bt.trans_thread)]
    texts = [t['text'] for t in bt.text_list]
    return [texts[i:i + bt.trans_thread] for i in range(0, len(texts), bt.trans_thread)]


def _run_recording(bt):
    """Run the real _run_srt/_run_text with a fake LLM; return [(hint, rendered_prompt), ...] per batch."""
    seen = []

    def fake_item_task(data):
        seen.append((bt._emotion_info, bt._render_prompt(data)))
        return data if bt.aisendsrt else "\n".join(f'T:{x}' for x in data.split("\n"))

    bt._item_task = fake_item_task
    (bt._run_srt if bt.aisendsrt else bt._run_text)(_batches(bt))
    return seen


# ---------------------------------------------------------------- templates

def test_every_template_has_emotion_info_once_right_after_context_info():
    files = sorted(PROMPT_DIR.glob('srt/*.txt')) + sorted(PROMPT_DIR.glob('text/*.txt'))
    assert len(files) == 36
    for f in files:
        s = f.read_text(encoding='utf-8')
        assert s.count('{emotion_info}') == 1, f
        assert s.count('{context_info}{emotion_info}') == 1, f      # same spot in every template
        assert s.count('{batch_input}') == 1, f


def test_hint_templates_exist_and_carry_the_rules():
    for mode in ('srt', 'text'):
        s = (PROMPT_DIR / 'emotion' / f'{mode}.txt').read_text(encoding='utf-8')
        assert s.count('{emotion_table}') == 1
        low = s.lower()
        for phrase in ('only to choose wording', 'do not add emotions', 'do not output', 'do not change the number'):
            assert phrase in low, (mode, phrase)
    assert 'timestamps' in (PROMPT_DIR / 'emotion' / 'srt.txt').read_text(encoding='utf-8')


# ---------------------------------------------------------------- prompt rendering

@pytest.mark.parametrize('srt_mode', [True, False])
@pytest.mark.parametrize('ainame', ['chatgpt', 'gemini', 'bailian', 'localllm', 'deepseek'])
def test_prompt_without_emotion_is_unchanged(make_trans, srt_mode, ainame):
    bt = make_trans(srt_mode, emotions=None, ainame=ainame)
    rendered = bt._render_prompt('BATCH')
    # identical to the template as it was before {emotion_info} existed
    assert '{emotion_info}' not in rendered and 'EMOTION HINTS' not in rendered
    original = bt.prompt.replace('{emotion_info}', '').replace('{batch_input}', 'BATCH')
    assert rendered == original
    assert not re.search(r'[{}]', rendered)                  # no leftover placeholder/braces


@pytest.mark.parametrize('srt_mode', [True, False])
def test_prompt_with_emotion_fills_placeholder_and_leaves_no_braces(make_trans, srt_mode):
    bt = make_trans(srt_mode)
    bt._emotion_info = bt._make_emotion_info([(2, 1000), (4, 3000)])
    rendered = bt._render_prompt('BATCH-TEXT')
    assert '{emotion_info}' not in rendered and '{emotion_table}' not in rendered and '{batch_input}' not in rendered
    assert not re.search(r'[{}]', rendered)
    assert '# EMOTION HINTS' in rendered
    assert '| 2 | angry | 0.5 |' in rendered and '| 4 | happy | 0.5 |' in rendered
    assert rendered.index('# EMOTION HINTS') < rendered.rindex('<INPUT>')      # hints come before the input block
    assert rendered.rstrip().endswith('BATCH-TEXT\n</INPUT>')                   # input is still last


def test_subtitle_text_that_looks_like_a_placeholder_is_not_expanded(make_trans):
    bt = make_trans(False)
    bt._emotion_info = 'HINT'
    rendered = bt._render_prompt('please keep {emotion_info} literally')
    input_block = rendered[rendered.rindex('<INPUT>'):]
    assert 'please keep {emotion_info} literally' in input_block
    assert rendered.count('HINT') == 1                                    # only at the template's own placeholder


def test_non_llm_translator_never_builds_hints():
    bt = BaseTrans(text_list=_items(), emotion=EmotionIndex(EMOTIONS))   # no .prompt, like Google/DeepL
    assert bt._make_emotion_info([(2, 1000)]) == ''


def test_custom_prompt_without_placeholder_gets_no_hint(make_trans):
    bt = make_trans(True)
    bt.prompt = bt.prompt.replace('{emotion_info}', '')                  # user-edited older template
    assert bt._make_emotion_info([(2, 1000)]) == ''


# ---------------------------------------------------------------- hint table per batch

def test_srt_mode_hints_use_srt_index(make_trans):
    bt = make_trans(True, thread=2)
    # 5 lines, 2 per batch -> batches (1,2) (3,4) (5)
    seen = _run_recording(bt)
    assert len(seen) == 3
    hint1, hint2, hint3 = (h for h, _ in seen)
    assert '| 2 | angry | 0.5 |' in hint1 and '| 1 |' not in hint1             # neutral line 1 is not listed
    assert '| 4 | happy | 0.5 |' in hint2 and '| 3 |' not in hint2             # line 3: confidence 0.4 < 0.5
    assert hint3 == ''                                                          # line 5 has no entry


def test_text_mode_hints_use_position_in_batch(make_trans):
    bt = make_trans(False, thread=2)
    seen = _run_recording(bt)
    assert len(seen) == 3
    hint1, hint2, hint3 = (h for h, _ in seen)
    assert '| 2 | angry | 0.5 |' in hint1                                        # position 2 of batch 1
    # line 4 (happy) is position 2 of batch 2 -- NOT 4
    assert '| 2 | happy | 0.5 |' in hint2 and '| 4 |' not in hint2
    assert hint3 == ''


def test_srt_mode_numbers_follow_the_srt_not_the_list_position(make_trans):
    bt = make_trans(True, thread=5, n=3)
    for it, first in zip(bt.text_list, (10, 11, 12)):
        it['line'] = first                                                      # e.g. a re-numbered subtitle
    seen = _run_recording(bt)
    assert '| 11 | angry | 0.5 |' in seen[0][0]


def test_matching_uses_start_time_not_index(make_trans):
    # subtitle edited: emotion entries were made for a different segmentation (time shifted by 120 ms)
    shifted = [EmotionEntry(120, 900, 'sad', 0.5, 0.6), EmotionEntry(1120, 1900, 'angry', 0.5, 0.6)]
    bt = make_trans(False, emotions=shifted, thread=5, n=2)
    seen = _run_recording(bt)
    assert '| 1 | sad | 0.5 |' in seen[0][0] and '| 2 | angry | 0.5 |' in seen[0][0]


@pytest.mark.parametrize('srt_mode', [True, False])
def test_emotion_disabled_gives_empty_hints_and_same_prompt(make_trans, srt_mode):
    on = _run_recording(make_trans(srt_mode, thread=5))
    off = _run_recording(make_trans(srt_mode, emotions=None, thread=5))
    assert off[0][0] == '' and '{emotion_info}' not in off[0][1]
    assert on[0][0] != '' and on[0][1] != off[0][1]
    # the only difference is the inserted hint block
    assert on[0][1].replace(on[0][0], '') == off[0][1]


def test_text_list_without_start_time_still_works_when_emotion_is_off(make_trans):
    bt = make_trans(False, emotions=None, thread=5, n=2)
    bt.text_list = [{'text': 'a'}, {'text': 'b'}]                                # plain dicts, like TestSrtTrans
    bt._item_task = lambda data: data
    assert [it['text'] for it in bt._run_text([['a', 'b']])] == ['a', 'b']


# ---------------------------------------------------------------- cache key

def test_cache_key_is_unchanged_without_hint(make_trans):
    bt = make_trans(False, emotions=None)
    legacy = hashlib.md5(
        f'{bt.translate_type}-{bt.api_url}-{bt.aisendsrt}-{bt.model_name}-{bt.source_code}-{bt.target_code}-"x"'.encode()
    ).hexdigest()
    assert bt._get_key('x') == legacy


def test_cache_key_includes_the_hint(make_trans):
    bt = make_trans(False)
    plain = bt._get_key('x')
    bt._emotion_info = 'HINT A'
    with_a = bt._get_key('x')
    bt._emotion_info = 'HINT B'
    with_b = bt._get_key('x')
    assert len({plain, with_a, with_b}) == 3


def test_translation_is_cached_per_hint_in_text_mode(make_trans):
    calls = []

    def run(emotions):
        bt = make_trans(False, emotions=emotions, thread=5)
        bt._item_task = lambda data: calls.append(data) or "\n".join(f'T:{x}' for x in data.split("\n"))
        bt._run_text(_batches(bt))

    run(EMOTIONS)
    assert len(calls) == 1
    run(EMOTIONS)                      # same lines, same hints -> cache hit
    assert len(calls) == 1
    run([EmotionEntry(1000, 1900, 'sad', 0.5, 0.6)])      # different hints -> must translate again
    assert len(calls) == 2
    run(None)                          # no hints at all -> another key
    assert len(calls) == 3


# ---------------------------------------------------------------- runner / channels

def test_runner_passes_emotion_entries_as_index(monkeypatch):
    import videotrans.translator._runner as runner
    created = {}

    class _Fake:
        def __init__(self, **kw):
            created.update(kw)

        def run(self):
            return 'ok'

    monkeypatch.setattr(runner, 'get_class', lambda *a, **k: _Fake)
    from videotrans.translator import CHATGPT_INDEX
    assert runner.run(translate_type=CHATGPT_INDEX, text_list=_items(), source_code='ja', target_code='en',
                      emotion_entries=EMOTIONS) == 'ok'
    assert isinstance(created['emotion'], EmotionIndex) and len(created['emotion']) == 4
    runner.run(translate_type=CHATGPT_INDEX, text_list=_items(), source_code='ja', target_code='en')
    assert created['emotion'] is None
    runner.run(translate_type=CHATGPT_INDEX, text_list=_items(), source_code='ja', target_code='en', emotion_entries=[])
    assert created['emotion'] is None


def test_openai_compatible_channel_sends_hint_in_user_message(make_trans, monkeypatch):
    from videotrans.translator._openaicompat import OpenAICampat
    bt = make_trans(False, ainame='localllm')
    bt.__class__ = type('LocalLike', (OpenAICampat,), {})           # real _item_task, no network
    bt.api_url, bt.model_name, bt.max_tokens = 'http://localhost:1234/v1', 'm', 100
    bt.extra_body = None
    bt.temperature = 0.1
    sent = {}

    def fake_completion(kwargs):
        sent.update(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='<TRANSLATE_TEXT>a\nb</TRANSLATE_TEXT>'))])

    bt._create_completion = fake_completion
    bt._emotion_info = bt._make_emotion_info([(2, 1000)])
    assert bt._item_task.__wrapped__(bt, 'src1\nsrc2') == 'a\nb'    # bypass tenacity's retry wrapper
    user = sent['messages'][1]['content']
    assert '| 2 | angry | 0.5 |' in user and 'src1\nsrc2' in user
    assert not re.search(r'\{(emotion_info|batch_input|emotion_table)\}', user)


def test_gemini_channel_sends_hint(make_trans, monkeypatch):
    # the google SDK is not needed to test prompt building: stub it
    google = types.ModuleType('google')
    genai = MagicMock()
    google.genai = genai
    monkeypatch.setitem(sys.modules, 'google', google)
    for name in ('google.genai', 'google.genai.types', 'google.genai.errors'):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.delitem(sys.modules, 'videotrans.translator._gemini', raising=False)
    from videotrans.translator._gemini import Gemini
    bt = make_trans(False, ainame='gemini')
    bt.__class__ = Gemini
    client = MagicMock()
    client.interactions.create.return_value = types.SimpleNamespace(output_text='<TRANSLATE_TEXT>a\nb</TRANSLATE_TEXT>')
    monkeypatch.setattr('videotrans.translator._gemini.genai.Client', MagicMock(return_value=client))
    bt._emotion_info = bt._make_emotion_info([(2, 1000)])

    assert bt._item_task.__wrapped__(bt, 'src1\nsrc2') == 'a\nb'
    message = client.interactions.create.call_args.kwargs['input']
    assert '| 2 | angry | 0.5 |' in message and not re.search(r'\{(emotion_info|batch_input)\}', message)


def test_bailian_llm_channel_sends_hint(make_trans, monkeypatch):
    monkeypatch.setitem(sys.modules, 'dashscope', MagicMock())
    monkeypatch.delitem(sys.modules, 'videotrans.translator._bailian', raising=False)
    from videotrans.translator import _bailian
    bt = make_trans(False, ainame='bailian')
    bt.__class__ = _bailian.BianlianMT
    bt.api_url = 'https://dashscope.aliyuncs.com/api/v1'
    sent = {}

    class _FakeOpenAI:
        def __init__(self, **kw):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.update(kw)
                    msg = types.SimpleNamespace(content='<TRANSLATE_TEXT>a\nb</TRANSLATE_TEXT>')
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(_bailian, 'OpenAI', _FakeOpenAI)
    bt._emotion_info = bt._make_emotion_info([(2, 1000)])
    assert bt._openai('src1\nsrc2') == 'a\nb'
    assert '| 2 | angry | 0.5 |' in sent['messages'][1]['content']


# ---------------------------------------------------------------- direct translation between ja / en / fr / zh

_LANGS = ['ja', 'en', 'fr', 'zh-cn']


@pytest.mark.parametrize('src,tgt', [(s, t) for s in _LANGS for t in _LANGS if s != t])
def test_translation_between_ja_en_fr_zh_is_direct(monkeypatch, src, tgt):
    """One channel call per pair, with the real source and target; English is never an intermediate."""
    import videotrans.translator._runner as runner
    from videotrans.translator import CHATGPT_INDEX, GOOGLE_INDEX, M2M100_INDEX, DEEPL_INDEX, BAILIAN_INDEX
    from videotrans.translator._lang_codes import LANG_CODE
    for channel in (CHATGPT_INDEX, GOOGLE_INDEX, M2M100_INDEX, DEEPL_INDEX, BAILIAN_INDEX):
        calls = []

        class _Fake:
            def __init__(self, **kw):
                calls.append(kw)

            def run(self):
                return 'ok'

        monkeypatch.setattr(runner, 'get_class', lambda *a, **k: _Fake)
        monkeypatch.setattr(runner, '_check_gorm', lambda name='google': 200)
        monkeypatch.setitem(sys.modules, 'videotrans.translator._google', types.SimpleNamespace(Google=_Fake))
        runner.run(translate_type=channel, text_list=_items(), source_code=src, target_code=tgt)
        assert len(calls) == 1, channel                                   # a single call: no second hop
        kw = calls[0]
        assert kw['common_target_code'] == tgt
        assert kw['source_code'] != kw['target_code']
        if 'en' not in (src, tgt):
            english = {'en', 'EN', 'EN-US', 'English', '__en__'}
            assert kw['source_code'] not in english and kw['target_code'] not in english, (channel, kw)
        if channel == CHATGPT_INDEX:                                      # LLM channels get language names
            assert kw['source_code'] == LANG_CODE[src][7] and kw['target_code'] == LANG_CODE[tgt][7]
