"""voicelab：話者の読み込み・役割の分担・pairs.csv（評価用の実声が TTS の手本と重ならないこと）。"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from test_voicelab_helpers import make_cfg_dict, make_profile, write_clip
from voicelab.config import config_from_dict
from voicelab.runner import prepare_speakers
from voicelab.scoring import (
    PairsLeakError,
    build_pairs,
    check_no_leak,
    write_pairs_csv,
)
from voicelab.speakers import SpeakerError, load_speaker, plan_speaker


def _cfg(tmp_path, speakers, **over):
    return config_from_dict(make_cfg_dict(speakers, targets="t.csv", **over), tmp_path)


def _plan(tmp_path, **over):
    d = make_profile(tmp_path / "profiles", "A", 0)
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}], **over)
    return cfg, plan_speaker(load_speaker(cfg.speakers[0], cfg), cfg)


def test_voiceprofile_format_is_read(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0)
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}])
    data = load_speaker(cfg.speakers[0], cfg)
    assert len(data.profile_refs) == 3 and len(data.eval_clips) == 5
    assert data.profile_refs[0].lang == "ja" and data.profile_refs[0].text.startswith("登録")
    assert data.profile_refs[0].sr == 24000  # 手本は元のサンプルレート


def test_plain_list_eval_format_is_also_read(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0, voiceprofile_format=False)
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}])
    assert len(load_speaker(cfg.speakers[0], cfg).eval_clips) == 5


def test_roles_are_disjoint_and_follow_the_design(tmp_path):
    cfg, plan = _plan(tmp_path)
    # C1 の手本：10〜15 秒の範囲で、点数が一番高い 1 本
    assert len(plan.profile_refs) == 1 and Path(plan.profile_refs[0].path).name == "ref_ja_01.wav"
    # C0 の手本の元：評価用の実声から、2〜4 秒の範囲に入るもの
    assert len(plan.line_refs) == 1 and 2.0 <= plan.line_refs[0].duration <= 4.0
    # 評価の ref：TTS の手本に使うものを含まない
    used = plan.tts_ref_paths
    assert used and not (used & {c.path for c in plan.score_refs})
    assert len(plan.score_refs) == 4 and len(plan.ceiling_clips) == 4


def test_pairs_never_use_tts_references(tmp_path):
    cfg, plan = _plan(tmp_path)
    out = tmp_path / "out"
    prepared = prepare_speakers(cfg, out)  # C0/C1 の手本ファイルを作る
    ps = prepared[0]
    # 手本ファイルは out/refs の下、元ファイルは ref_source
    assert Path(ps.line_refs[0].path).name.startswith("line_ref_16k_") and ps.line_refs[0].sr == 16000
    assert ps.profile_refs[0].sr == 24000

    audio = out / "audio" / "A" / "voxcpm2" / "profile_ref" / "x.wav"
    write_clip(audio, 0, 1, 2.0)
    manifest = {"j1": {"status": "ok", "speaker": "A", "out_wav": "audio/A/voxcpm2/profile_ref/x.wav", "engine_label": "voxcpm2",
                       "condition": "profile_ref", "lang": "en", "text_id": "m1-en", "repeat": "0"}}
    rows = build_pairs(cfg, [ps.plan], manifest, out)
    gen = [r for r in rows if r["engine"] != "real"]
    assert len(gen) == 1 and gen[0]["label"] == "voxcpm2|profile_ref|en" and gen[0]["test"] == "audio/A/voxcpm2/profile_ref/x.wav"
    forbidden = {str(Path(p).resolve()) for p in ps.plan.tts_ref_paths}
    for r in rows:
        refs = {str(Path(p).resolve()) for p in r["ref"].split(";")}
        assert not (refs & forbidden), "評価の ref に TTS の手本が混ざっている"
        assert str(Path(r["test"]).resolve()) not in refs
    # 実声どうしの上限：test の実声は ref から外れている
    real = [r for r in rows if r["engine"] == "real"]
    assert len(real) == 4 and all(r["label"] == "real|heldout_loo|ja" for r in real)
    assert all(r["test"] not in r["ref"].split(";") for r in real)


def test_leak_check_stops_when_a_tts_reference_is_in_ref(tmp_path):
    cfg, plan = _plan(tmp_path)
    leaked = plan.profile_refs[0].path
    bad = [{"speaker": "A", "ref": ";".join([leaked, plan.score_refs[0].path]), "test": "x.wav", "label": "v|profile_ref|ja",
            "engine": "voxcpm2"}]
    with pytest.raises(PairsLeakError):
        check_no_leak(bad, [plan])
    leaked2 = plan.line_refs[0].path
    bad2 = [{"speaker": "A", "ref": leaked2, "test": "x.wav", "label": "v|line_ref_16k|ja", "engine": "voxcpm2"}]
    with pytest.raises(PairsLeakError):
        check_no_leak(bad2, [plan])


def test_pairs_csv_is_readable_by_voicecheck(tmp_path):
    from voicecheck.pairs import read_pairs

    cfg, plan = _plan(tmp_path)
    rows = [{"speaker": "A", "ref": ";".join(c.path for c in plan.score_refs), "test": "audio/a.wav", "label": "voxcpm2|profile_ref|ja",
             "text_id": "m1-ja", "repeat": "0", "engine": "voxcpm2", "condition": "profile_ref", "lang": "ja"}]
    p = write_pairs_csv(tmp_path / "out" / "pairs.csv", rows)
    got = read_pairs(p)
    assert len(got) == 1 and got[0].label == "voxcpm2|profile_ref|ja" and len(got[0].refs) == 4
    with p.open(encoding="utf-8-sig") as f:
        assert next(csv.reader(f))[:4] == ["speaker", "ref", "test", "label"]


def test_not_enough_eval_clips_is_a_friendly_error(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0, eval_durs=(3.0,))
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}])
    with pytest.raises(SpeakerError) as e:
        plan_speaker(load_speaker(cfg.speakers[0], cfg), cfg)
    assert "足りません" in str(e.value)


def test_profile_without_eval_is_an_error(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0)
    (d / "profile.json").write_text('{"name": "A", "refs": [{"wav": "refs/ref_ja_01.wav", "text": "x", "lang": "ja"}]}', encoding="utf-8")
    (d / "eval" / "eval.json").unlink()
    for p in (d / "eval").glob("*.wav"):
        p.unlink()
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}])
    with pytest.raises(SpeakerError) as e:
        load_speaker(cfg.speakers[0], cfg)
    assert "評価用の実声" in str(e.value)


def test_eval_file_that_is_also_a_ref_is_removed_from_eval(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0)
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}])
    data = load_speaker(cfg.speakers[0], cfg)
    data.eval_clips.append(data.profile_refs[0])  # 取り違えを再現
    plan = plan_speaker(data, cfg)
    assert any("外しました" in n for n in plan.notes)
    assert data.profile_refs[0].path not in {c.path for c in plan.score_refs}


def test_line_ref_longer_than_limit_is_trimmed_and_loses_its_text(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0, eval_durs=(9.0, 8.0, 7.0, 6.0))
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}])
    ps = prepare_speakers(cfg, tmp_path / "out")[0]
    ref = ps.line_refs[0]
    assert ref.sr == 16000 and ref.sec == pytest.approx(4.0, abs=0.05) and ref.text == "" and "切った" in ref.note


def test_ref_lang_filters_refs(tmp_path):
    d = make_profile(tmp_path / "profiles", "A", 0)
    cfg = _cfg(tmp_path, [{"name": "A", "profile": str(d)}], ref_lang="fr")
    with pytest.raises(SpeakerError):
        plan_speaker(load_speaker(cfg.speakers[0], cfg), cfg)


# ---- wav フォルダ方式 ---------------------------------------------------------------------------------------
def test_folder_mode_with_transcripts_csv_and_sidecar_txt(tmp_path):
    d = tmp_path / "enroll" / "B"
    for k, dur in enumerate([12.0, 13.0, 11.0, 3.0, 4.0, 3.5, 2.5]):
        write_clip(d / f"u{k}.wav", 1, k, dur)
    (d / "transcripts.csv").write_text("file,text,lang\n" + "".join(f"u{k}.wav,書き起こし{k},ja\n" for k in range(3)), encoding="utf-8")
    (d / "u3.txt").write_text("短い文", encoding="utf-8")  # 同名 .txt
    cfg = _cfg(tmp_path, [{"name": "B", "folder": str(d)}])
    data = load_speaker(cfg.speakers[0], cfg)
    assert len(data.profile_refs) == 3 and len(data.eval_clips) == 4  # 書き起こしのある長い 3 本が手本
    assert {Path(c.path).name for c in data.profile_refs} == {"u0.wav", "u1.wav", "u2.wav"}
    assert any(c.text == "短い文" for c in data.eval_clips)
    plan = plan_speaker(data, cfg)
    assert not ({c.path for c in plan.score_refs} & plan.tts_ref_paths)


def test_folder_mode_role_column_overrides(tmp_path):
    d = tmp_path / "enroll" / "C"
    for k, dur in enumerate([12.0, 12.0, 3.0, 4.0, 3.5]):
        write_clip(d / f"u{k}.wav", 2, k, dur)
    (d / "transcripts.csv").write_text(
        "file,text,lang,role\nu0.wav,a,ja,eval\nu1.wav,b,ja,ref\nu2.wav,c,ja,eval\nu3.wav,d,ja,eval\nu4.wav,e,ja,eval\n", encoding="utf-8")
    cfg = _cfg(tmp_path, [{"name": "C", "folder": str(d)}])
    data = load_speaker(cfg.speakers[0], cfg)
    assert [Path(c.path).name for c in data.profile_refs] == ["u1.wav"] and len(data.eval_clips) == 4
