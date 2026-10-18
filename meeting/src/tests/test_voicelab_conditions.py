"""voicelab：条件の展開（Job の一覧）が設計どおりか（1回に1つだけ変える）。"""
from __future__ import annotations

from collections import Counter

from test_voicelab_helpers import SHORT_TARGETS, make_cfg_dict
from voicelab.conditions import RefInfo, build_jobs, condition_sort_key, seed_for
from voicelab.config import config_from_dict
from voicelab.texts import read_targets

SPK = [{"name": "A", "profile": "profiles/A"}]


def _refs():
    line = RefInfo("line", "/o/line.wav", "/src/e1.wav", "短い手本", "ja", 3.0, 16000, "e1")
    prof = RefInfo("profile", "/o/prof.wav", "/src/r1.wav", "登録の手本", "ja", 12.0, 24000, "r1")
    return [line], [prof]


def _jobs(tmp_path, **over):
    (tmp_path / "t.csv").write_text(SHORT_TARGETS, encoding="utf-8")
    cfg = config_from_dict(make_cfg_dict(SPK, targets="t.csv", **over), tmp_path)
    texts = read_targets(tmp_path / "t.csv", cfg.target_langs)
    line, prof = _refs()
    return cfg, build_jobs(cfg, "A", line, prof, texts)


def test_job_counts_for_default_conditions(tmp_path):
    cfg, jobs = _jobs(tmp_path, engines=["voxcpm2", "qwen3tts"], repeats=2)
    c = Counter(j.condition for j in jobs)
    # 会議文 2 本（ja, en）× 繰り返し 2 × エンジン 2 = 8 件ずつ
    assert c["line_ref_16k"] == 8 and c["profile_ref"] == 8
    assert c["stretch_x1.3"] == 8 and c["stretch_x2.0"] == 8
    # 感情文 2 本 × 繰り返し 2 × 3 モード（voxcpm2 だけ）
    assert c["emotion_instruct"] == 4 and c["emotion_refonly"] == 4 and c["emotion_similarity"] == 4
    assert {j.engine for j in jobs if j.condition.startswith("emotion")} == {"voxcpm2"}
    assert len(jobs) == 8 * 4 + 12


def test_meeting_and_emotional_texts_do_not_mix(tmp_path):
    _, jobs = _jobs(tmp_path)
    for j in jobs:
        assert (j.kind == "emotional") == j.condition.startswith("emotion")


def test_c0_and_c1_differ_only_in_the_reference(tmp_path):
    """C0 と C1 は、同じ エンジン・文・繰り返し・乱数の種。違うのは手本だけ（1回に1つだけ変える）。"""
    _, jobs = _jobs(tmp_path, repeats=2)
    by = {(j.engine, j.text_id, j.repeat, j.condition): j for j in jobs}
    for (e, t, r, c), j0 in by.items():
        if c != "line_ref_16k":
            continue
        j1 = by[(e, t, r, "profile_ref")]
        assert (j0.text, j0.lang, j0.seed, j0.emotion_arg) == (j1.text, j1.lang, j1.seed, j1.emotion_arg)
        assert j0.ref.kind == "line" and j1.ref.kind == "profile"
        assert j0.ref.sr == 16000 and j1.ref.sr == 24000


def test_stretch_is_derived_from_profile_ref_output(tmp_path):
    _, jobs = _jobs(tmp_path)
    ids = {j.job_id: j for j in jobs}
    st = [j for j in jobs if j.condition.startswith("stretch_x")]
    assert {j.factor for j in st} == {1.3, 2.0} and {j.condition for j in st} == {"stretch_x1.3", "stretch_x2.0"}
    for j in st:
        parent = ids[j.parent]
        assert parent.condition == "profile_ref" and parent.engine == j.engine and parent.text_id == j.text_id
        assert parent.repeat == j.repeat and j.is_derived


def test_emotion_modes_use_the_right_arguments(tmp_path):
    _, jobs = _jobs(tmp_path)
    emo = {j.condition: j for j in jobs if j.condition.startswith("emotion") and j.text_id == "e1-ja" and j.repeat == 0}
    assert emo["emotion_similarity"].ref_text_arg == "登録の手本" and emo["emotion_similarity"].emotion_arg is None
    assert emo["emotion_refonly"].ref_text_arg is None and emo["emotion_refonly"].emotion_arg is None
    assert emo["emotion_instruct"].emotion_arg == "angry, tense, slightly faster"
    assert len({j.seed for j in emo.values()}) == 1  # 3 つとも同じ種＝対で比べられる
    assert all(j.ref.kind == "profile" for j in emo.values())


def test_seed_differs_per_repeat_and_text_but_not_per_condition():
    assert seed_for(1, "m1-ja", 0) != seed_for(1, "m1-ja", 1)
    assert seed_for(1, "m1-ja", 0) != seed_for(1, "m2-ja", 0)
    assert seed_for(1, "m1-ja", 0) == seed_for(1, "m1-ja", 0)


def test_cosyvoice_instruct2_only_in_emotion_condition(tmp_path):
    _, jobs = _jobs(tmp_path, engines=["cosyvoice3:zero_shot"], emotion_engines=["cosyvoice3:instruct2", "voxcpm2"])
    inst = [j for j in jobs if j.engine == "cosyvoice3:instruct2"]
    assert inst and {j.condition for j in inst} == {"emotion_instruct"}
    assert {j.condition for j in jobs if j.engine == "cosyvoice3:zero_shot"} >= {"line_ref_16k", "profile_ref"}
    # instruct2 は基準の条件（C0/C1）には出ない
    cfg, jobs2 = _jobs(tmp_path, engines=["cosyvoice3:instruct2"])
    assert not [j for j in jobs2 if j.condition in ("line_ref_16k", "profile_ref")]


def test_unsupported_language_is_skipped_not_dropped(tmp_path):
    (tmp_path / "t.csv").write_text("id,lang,text\nx1,ar,مرحبا\nx2,ja,はい。\n", encoding="utf-8")
    cfg = config_from_dict(make_cfg_dict(SPK, targets="t.csv", target_langs=["ar", "ja"], engines=["qwen3tts"],
                                         conditions=["line_ref_16k"]), tmp_path)
    line, prof = _refs()
    jobs = build_jobs(cfg, "A", line, prof, read_targets(tmp_path / "t.csv", cfg.target_langs))
    skipped = [j for j in jobs if j.skip_reason]
    assert [j.lang for j in skipped] == ["ar"] and [j.lang for j in jobs if not j.skip_reason] == ["ja"]


def test_profile_ref_each_makes_one_condition_per_reference(tmp_path):
    (tmp_path / "t.csv").write_text(SHORT_TARGETS, encoding="utf-8")
    cfg = config_from_dict(make_cfg_dict(SPK, targets="t.csv", profile_ref_select="each", engines=["voxcpm2"],
                                         conditions=["profile_ref", "stretch"]), tmp_path)
    line, prof = _refs()
    prof2 = RefInfo("profile", "/o/prof2.wav", "/src/r2.wav", "登録2", "ja", 11.0, 24000, "r2")
    jobs = build_jobs(cfg, "A", line, prof + [prof2], read_targets(tmp_path / "t.csv", ["ja", "en"]))
    conds = {j.condition for j in jobs}
    assert {"profile_ref_r1", "profile_ref_r2"} <= conds
    assert {j.parent.split("|")[2] for j in jobs if j.is_derived} == {"profile_ref_r1"}  # 早回しは最初の手本だけ


def test_output_paths_and_job_ids_are_unique(tmp_path):
    _, jobs = _jobs(tmp_path, repeats=3, engines=["voxcpm2", "cosyvoice3:zero_shot", "cosyvoice3:cross_lingual"])
    assert len({j.job_id for j in jobs}) == len(jobs)
    assert len({j.out_rel for j in jobs}) == len(jobs)
    assert all(j.out_rel.endswith(".wav") and ":" not in j.out_rel for j in jobs)


def test_condition_sort_order():
    names = ["emotion_instruct", "stretch_x2.0", "profile_ref", "emotion_similarity", "stretch_x1.3", "line_ref_16k", "emotion_refonly"]
    assert sorted(names, key=condition_sort_key) == [
        "line_ref_16k", "profile_ref", "stretch_x1.3", "stretch_x2.0", "emotion_similarity", "emotion_refonly", "emotion_instruct"]
