"""voicelab：設定ファイルと targets.csv の読み込み。"""
from __future__ import annotations

import json

import pytest

from test_voicelab_helpers import make_cfg_dict
from voicelab.config import ConfigError, config_from_dict, engine_options_for, load_config
from voicelab.texts import DEFAULT_TARGETS, read_targets, write_targets

SPK = [{"name": "A", "profile": "profiles/A"}]


def test_minimal_config_gets_defaults():
    cfg = config_from_dict({"speakers": SPK, "targets": "t.csv"})
    assert cfg.repeats == 2 and cfg.target_langs == ["ja", "en", "fr", "zh"]
    assert cfg.conditions == ["line_ref_16k", "profile_ref", "stretch", "emotion"]
    assert cfg.stretch_factors == [1.3, 2.0] and cfg.emotion_engines == ["voxcpm2"]
    assert "voxcpm2" in cfg.engines and "cosyvoice3:zero_shot" in cfg.engines


def test_unknown_keys_are_reported_with_the_allowed_list():
    with pytest.raises(ConfigError) as e:
        config_from_dict({"speakers": SPK, "targets": "t.csv", "repeat": 3})
    assert "repeat" in str(e.value) and "repeats" in str(e.value)
    with pytest.raises(ConfigError):
        config_from_dict({"speakers": [{"name": "A", "profil": "x"}], "targets": "t.csv"})


def test_speaker_needs_exactly_one_of_profile_or_folder():
    with pytest.raises(ConfigError):
        config_from_dict({"speakers": [{"name": "A"}], "targets": "t.csv"})
    with pytest.raises(ConfigError):
        config_from_dict({"speakers": [{"name": "A", "profile": "p", "folder": "f"}], "targets": "t.csv"})
    with pytest.raises(ConfigError):  # 名前の重複
        config_from_dict({"speakers": [{"name": "A", "profile": "p"}, {"name": "A", "profile": "q"}], "targets": "t.csv"})
    with pytest.raises(ConfigError):
        config_from_dict({"speakers": [], "targets": "t.csv"})


def test_name_defaults_to_folder_name():
    cfg = config_from_dict({"speakers": [{"profile": "profiles/Bさん"}], "targets": "t.csv"})
    assert cfg.speakers[0].name == "Bさん"


def test_stretch_pulls_in_profile_ref_and_conditions_keep_order():
    cfg = config_from_dict(make_cfg_dict(SPK, conditions=["stretch", "line_ref_16k"]))
    assert cfg.conditions == ["line_ref_16k", "profile_ref", "stretch"]


@pytest.mark.parametrize("bad", [
    {"engines": ["nosuch"]}, {"engines": []}, {"engines": ["cosyvoice3:nope"]}, {"conditions": ["C9"]},
    {"repeats": 0}, {"stretch_factors": [0.8]}, {"profile_ref_select": "all"}, {"stretch_method": "sox"},
    {"target_langs": []}, {"emotion_engines": ["voxcpm3"]}, {"max_texts_per_lang": 0},
])
def test_bad_values_are_friendly_errors(bad):
    with pytest.raises(ConfigError):
        config_from_dict(make_cfg_dict(SPK, **bad))


@pytest.mark.parametrize("name", ["f5tts", "F5-TTS", "omnivoice", "higgs", "vevo", "indextts"])
def test_non_commercial_engines_are_refused(name):
    with pytest.raises(ConfigError) as e:
        config_from_dict(make_cfg_dict(SPK, engines=[name]))
    assert "使えません" in str(e.value)


def test_engine_options_merge_name_and_full_spec():
    cfg = config_from_dict(make_cfg_dict(SPK, engines=["cosyvoice3:zero_shot"], engine_options={
        "cosyvoice3": {"fp16": True, "ja_katakana": True}, "cosyvoice3:zero_shot": {"ja_katakana": False}}))
    assert engine_options_for(cfg, "cosyvoice3:zero_shot") == {"fp16": True, "ja_katakana": False}
    assert engine_options_for(cfg, "cosyvoice3:cross_lingual") == {"fp16": True, "ja_katakana": True}


def test_load_json_resolves_relative_paths_from_the_config_folder(tmp_path):
    p = tmp_path / "sub" / "exp.json"
    p.parent.mkdir()
    p.write_text(json.dumps(make_cfg_dict(SPK)), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.base_dir == str(p.parent.resolve())


def test_load_yaml(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "exp.yaml"
    p.write_text(
        "name: y\nspeakers:\n  - name: A\n    profile: profiles/A\ntargets: t.csv\nengines: [voxcpm2, \"cosyvoice3:zero_shot\"]\n"
        "repeats: 3\n# コメント\nstretch_factors: [1.3, 2.0]\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.name == "y" and cfg.repeats == 3 and cfg.engines == ["voxcpm2", "cosyvoice3:zero_shot"]


def test_load_config_errors_are_friendly(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nothing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(bad)
    assert "書き方" in str(e.value)
    txt = tmp_path / "exp.txt"
    txt.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(txt)


# ---- targets.csv --------------------------------------------------------------------------------------
def test_default_targets_cover_four_languages_and_three_emotions(tmp_path):
    p = write_targets(tmp_path / "t.csv")
    ts = read_targets(p)
    assert {t.lang for t in ts} == {"ja", "en", "fr", "zh"}
    emo = [t for t in ts if t.kind == "emotional"]
    assert len({t.id.split("-")[0] for t in emo}) == 3 and all(t.emotion for t in emo)
    assert len(ts) == len(DEFAULT_TARGETS) and len({t.id for t in ts}) == len(ts)
    assert {t.lang for t in emo} == {"ja", "en", "fr", "zh"}


def test_targets_filters_and_limits(tmp_path):
    p = write_targets(tmp_path / "t.csv")
    ts = read_targets(p, ["ja", "en"], max_per_lang=1)
    assert sorted((t.lang, t.kind) for t in ts) == [("en", "emotional"), ("en", "meeting"), ("ja", "emotional"), ("ja", "meeting")]
    with pytest.raises(ConfigError):
        read_targets(p, ["de"])


def test_targets_japanese_headers_and_cp932(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes("id,言語,種類,感情,文\na1,JA,meeting,,こんにちは。\n".encode("cp932"))
    t = read_targets(p)[0]
    assert (t.id, t.lang, t.text, t.kind) == ("a1", "ja", "こんにちは。", "meeting")


@pytest.mark.parametrize("csv_text,msg", [
    ("id,lang\na,ja\n", "必要な列"),
    ("id,lang,text\na,ja,\n", "空"),
    ("id,lang,text,kind\na,ja,x,loud\n", "kind"),
    ("id,lang,text,kind\na,ja,x,emotional\n", "emotion"),
    ("id,lang,text\na,ja,x\na,en,y\n", "重複"),
])
def test_targets_errors(tmp_path, csv_text, msg):
    p = tmp_path / "t.csv"
    p.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        read_targets(p)
    assert msg in str(e.value)
