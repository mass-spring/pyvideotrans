"""run（クライアント側）: ノートブックとの一致、入力の組み立て、SSM に送る内容、ログの追跡、結果の回収。"""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import pytest

from test_cloud_aws_helpers import Args, REGION, aws, client_error, do_launch, do_setup, mocked, state  # noqa: F401

from meeting.cloud.aws import common, recipes, runner
from meeting.cloud.aws.common import NeedConfirmation, ToolError, bucket_name
from meeting.cloud.aws.runner import (
    build_plan,
    download_results,
    follow_command,
    remote_command_lines,
    run_fetch,
    run_run,
    validate_exp,
)
from meeting.cloud.aws.ssmrun import LogTail

BUCKET = bucket_name("123456789012", REGION)
NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "voice_baseline_colab.ipynb"


def rargs(**kw):
    base = dict(exp="exp01", engines="voxcpm2,qwen3tts,chatterbox", dry_run_first=False, config=None, speaker=None, targets=None,
                public_test=False, mode="quick", embedder=None, ref_lang=None, code="git", instance_id=None, out_dir=None,
                skip_audio=False, attach=False, command_id=None, resume=False, fresh=False, auto_terminate=False, yes=True)
    base.update(kw)
    return Args(**base)


def make_profile(tmp_path, name="A"):
    d = tmp_path / "profiles" / name
    (d / "refs").mkdir(parents=True)
    (d / "eval").mkdir()
    (d / "profile.json").write_text(json.dumps({"refs": [], "eval": {"index": "eval/eval.json", "n": 0}}), encoding="utf-8")
    (d / "refs" / "ref_ja_01.wav").write_bytes(b"RIFF" + b"0" * 100)
    (d / ".DS_Store").write_text("junk")
    return d


def plan_for(tmp_path, **kw):
    tmp = tmp_path / "plan-tmp"
    tmp.mkdir(exist_ok=True)
    return build_plan(rargs(**kw), bucket=BUCKET, region=REGION, tmp=tmp)


# ---------------------------------------------------------------------------
# ノートブック（環境づくりの正本）との一致
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def notebook_src():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def all_envs():
    return [recipes.SCORE_ENV] + [g.env for g in recipes.ENGINE_GROUPS.values()]


def test_every_pinned_package_appears_in_the_notebook_exactly(notebook_src):
    for env in all_envs():
        for pkg in env.packages:
            assert f'"{pkg}"' in notebook_src, f"{env.name}: {pkg} がノートブックにありません"
        for args in env.pre + env.optional_after:
            for a in args:
                assert f'"{a}"' in notebook_src, f"{env.name}: {a}"


def test_python_versions_and_import_checks_match_the_notebook(notebook_src):
    assert re.search(r'make_env\(\s*"score",\s*"3\.12"', notebook_src)
    for name, g in recipes.ENGINE_GROUPS.items():
        assert re.search(rf'make_env\("{name}",\s*"{re.escape(g.env.python)}"', notebook_src), name
        if g.env.check_import:
            assert f'check_import("{name}", "{g.env.check_import}")' in notebook_src


def test_tool_versions_clone_urls_and_cosyvoice_filters_match_the_notebook(notebook_src):
    assert f"uv=={recipes.UV_VERSION}" in notebook_src
    assert recipes.REPO_URL in notebook_src and f'"{recipes.REPO_BRANCH}"' in notebook_src
    assert recipes.COSY_REPO_URL in notebook_src
    for p in recipes.ENGINE_GROUPS["cosyvoice3"].env.drop_prefixes:
        assert f'"{p}"' in notebook_src
    for pkg in recipes.APT_PACKAGES[:4]:
        assert pkg in notebook_src
    assert '"--engines", env' in notebook_src          # generate は仮想環境名（= グループ名）を --engines に渡す


def test_engines_that_conflict_get_separate_environments():
    names = [g.env.name for g in recipes.ENGINE_GROUPS.values()]
    assert len(set(names)) == len(names)
    pins = {n: set(g.env.packages) for n, g in recipes.ENGINE_GROUPS.items()}
    assert "chatterbox-tts==0.1.7" in pins["chatterbox"] and "qwen-tts==0.1.1" in pins["qwen3tts"]
    assert not (pins["chatterbox"] & {"qwen-tts==0.1.1"})


def test_parse_engines_and_forbidden_ones():
    gs = recipes.parse_engines("voxcpm2, qwen3tts,chatterbox,voxcpm2")
    assert [g.name for g in gs] == ["voxcpm2", "qwen3tts", "chatterbox"]
    assert recipes.config_engines(recipes.parse_engines("cosyvoice3")) == ["cosyvoice3:zero_shot", "cosyvoice3:cross_lingual"]
    for bad in ("f5tts", "indextts", "omnivoice", "higgs", "vevo"):
        with pytest.raises(ToolError) as e:
            recipes.parse_engines(f"voxcpm2,{bad}")
        assert "使いません" in str(e.value)
    with pytest.raises(ToolError):
        recipes.parse_engines("nonsense")
    with pytest.raises(ToolError):
        recipes.parse_engines("cosyvoice3:zero_shot")
    with pytest.raises(ToolError):
        recipes.parse_engines("")


# ---------------------------------------------------------------------------
# 入力の組み立て
# ---------------------------------------------------------------------------
def test_exp_name_validation():
    assert validate_exp("exp01") == "exp01" and validate_exp("a.b_c-1") == "a.b_c-1"
    for bad in ("", "../x", "a b", "-x", "x" * 41, "実験", "a/b", "a;rm"):
        with pytest.raises(ToolError):
            validate_exp(bad)


def test_speaker_profile_is_tarred_without_hidden_files_and_rewritten_to_remote_paths(tmp_path):
    prof = make_profile(tmp_path)
    plan = plan_for(tmp_path, speaker=[f"A={prof}"])
    tars = [(p, k) for p, k in plan.uploads if k.endswith(".tar.gz") and "spk0" in k]
    assert len(tars) == 1
    with tarfile.open(tars[0][0]) as tf:
        names = tf.getnames()
    assert any(n.endswith("profile.json") for n in names) and any(n.endswith("refs/ref_ja_01.wav") for n in names)
    assert not any(".DS_Store" in n for n in names)
    sp = plan.config["speakers"][0]
    assert sp == {"name": "A", "profile": "/opt/voicelab/inputs/exp01/spk0"}
    assert plan.job["inputs"][0] == {"key": "exp/exp01/inputs/spk0.tar.gz", "dest": "/opt/voicelab/inputs/exp01/spk0", "kind": "tar"}


def test_wav_folder_is_detected_as_folder_kind(tmp_path):
    d = tmp_path / "enroll" / "B"
    d.mkdir(parents=True)
    (d / "u0.wav").write_bytes(b"RIFF0000")
    (d / "transcripts.csv").write_text("file,text,lang\nu0.wav,こんにちは,ja\n", encoding="utf-8")
    plan = plan_for(tmp_path, speaker=[f"B={d}"])
    assert plan.config["speakers"][0] == {"name": "B", "folder": "/opt/voicelab/inputs/exp01/spk0"}


def test_speaker_errors(tmp_path):
    with pytest.raises(ToolError):
        plan_for(tmp_path, speaker=["no-equals-sign"])
    with pytest.raises(ToolError):
        plan_for(tmp_path, speaker=[f"A={tmp_path / 'nothing'}"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ToolError):
        plan_for(tmp_path, speaker=[f"A={empty}"])
    with pytest.raises(ToolError) as e:
        plan_for(tmp_path)
    assert "--public-test" in str(e.value)


def test_public_test_uses_no_local_voice_files(tmp_path):
    plan = plan_for(tmp_path, public_test=True)
    keys = [k for _, k in plan.uploads]
    assert not any("spk" in k for k in keys)
    assert plan.config["speakers"] == [{"name": "public1272", "folder": "/opt/voicelab/inputs/exp01/public1272"}]
    assert plan.job["public_test"]["dest"].endswith("public1272") and "exp/exp01/job/prepare_public.py" in keys
    assert plan.job["default_targets"] == "/opt/voicelab/inputs/exp01/targets.csv"
    with pytest.raises(ToolError):
        plan_for(tmp_path, public_test=True, speaker=["A=" + str(make_profile(tmp_path))])


def test_quick_and_full_modes(tmp_path):
    q = plan_for(tmp_path, public_test=True, mode="quick").config
    f = plan_for(tmp_path, public_test=True, mode="full").config
    assert (q["repeats"], q["max_texts_per_lang"]) == (1, 1) and (f["repeats"], f["max_texts_per_lang"]) == (2, None)
    assert q["target_langs"] == ["ja", "en", "fr", "zh"] and q["embedder"] == "ecapa" and q["seed"] == 1234
    assert q["conditions"] == ["line_ref_16k", "profile_ref", "stretch", "emotion"] and q["stretch_factors"] == [1.3, 2.0]


def test_engines_flow_into_the_config_and_emotion_engine_follows_voxcpm2(tmp_path):
    cfg = plan_for(tmp_path, public_test=True, engines="qwen3tts,cosyvoice3").config
    assert cfg["engines"] == ["qwen3tts", "cosyvoice3:zero_shot", "cosyvoice3:cross_lingual"] and cfg["emotion_engines"] == []
    cfg = plan_for(tmp_path, public_test=True, engines="voxcpm2,chatterbox").config
    assert cfg["engines"] == ["voxcpm2", "chatterbox"] and cfg["emotion_engines"] == ["voxcpm2"]


def test_options_embedder_ref_lang_dry_run_auto_terminate(tmp_path):
    plan = plan_for(tmp_path, public_test=True, embedder="none", ref_lang="ja", dry_run_first=True, auto_terminate=True)
    assert plan.config["embedder"] == "none" and plan.config["ref_lang"] == "ja"
    assert plan.job["dry_run_first"] is True and plan.job["shutdown_when_done"] is True and plan.job["embedder"] == "none"
    plan = plan_for(tmp_path, public_test=True)
    assert plan.job["dry_run_first"] is False and plan.job["shutdown_when_done"] is False


def test_targets_csv_is_uploaded_and_config_points_to_the_remote_copy(tmp_path):
    t = tmp_path / "my_targets.csv"
    t.write_text("id,lang,kind,emotion,text\nm1-ja,ja,meeting,,はい。\n", encoding="utf-8")
    plan = plan_for(tmp_path, public_test=True, targets=str(t))
    assert plan.config["targets"] == "/opt/voicelab/inputs/exp01/targets.csv"
    assert (t, "exp/exp01/inputs/targets.csv") in plan.uploads and plan.job["default_targets"] is None
    with pytest.raises(ToolError):
        plan_for(tmp_path, public_test=True, targets=str(tmp_path / "missing.csv"))


def test_config_file_relative_paths_resolve_from_the_config_folder(tmp_path):
    prof = make_profile(tmp_path, "Aさん")
    (tmp_path / "t.csv").write_text("id,lang,kind,emotion,text\nm1-ja,ja,meeting,,はい。\n", encoding="utf-8")
    cfgdir = tmp_path / "my_exp"
    cfgdir.mkdir()
    (cfgdir / "exp.json").write_text(json.dumps({
        "speakers": [{"name": "Aさん", "profile": "../profiles/Aさん"}], "targets": "../t.csv", "repeats": 3, "ref_lang": "ja",
        "engines": ["ignored"], "max_texts_per_lang": 2}, ensure_ascii=False), encoding="utf-8")
    plan = plan_for(tmp_path, config=str(cfgdir / "exp.json"))
    assert plan.config["speakers"] == [{"name": "Aさん", "profile": "/opt/voicelab/inputs/exp01/spk0"}]     # 日本語の名前は設定の中だけ。パスは ASCII
    assert plan.config["repeats"] == 3 and plan.config["ref_lang"] == "ja" and plan.config["max_texts_per_lang"] == 2
    assert plan.config["engines"] == ["voxcpm2", "qwen3tts", "chatterbox"]                              # 強制項目は --engines が勝つ
    assert any(k == "exp/exp01/inputs/targets.csv" for _, k in plan.uploads)
    assert prof.exists()


def test_config_typos_are_caught_before_spending_gpu_time(tmp_path):
    cfg = tmp_path / "exp.json"
    cfg.write_text(json.dumps({"repeat": 3}), encoding="utf-8")
    with pytest.raises(ToolError) as e:
        plan_for(tmp_path, public_test=True, config=str(cfg))
    assert "repeat" in str(e.value) and "repeats" in str(e.value)


def test_uploaded_config_has_no_local_paths(tmp_path):
    prof = make_profile(tmp_path)
    plan = plan_for(tmp_path, speaker=[f"A={prof}"])
    cfg_file = [p for p, k in plan.uploads if k.endswith("job/exp.json")][0]
    text = cfg_file.read_text(encoding="utf-8")
    assert str(tmp_path) not in text and "/opt/voicelab/" in text


def test_code_mode_git_clones_the_branch_and_local_uploads_meeting_src(tmp_path):
    plan = plan_for(tmp_path, public_test=True, code="git")
    assert plan.job["code"] == {"mode": "git", "url": "https://github.com/mass-spring/pyvideotrans", "branch": "meeting-translator"}
    plan = plan_for(tmp_path, public_test=True, code="local")
    assert plan.job["code"]["mode"] == "s3" and plan.job["code"]["key"] == "exp/exp01/inputs/code.tar.gz"
    tar = [p for p, k in plan.uploads if k.endswith("code.tar.gz")][0]
    with tarfile.open(tar) as tf:
        names = tf.getnames()
    assert any(n.endswith("voicelab/runner.py") for n in names) and any(n.endswith("voicecheck/__init__.py") for n in names)
    assert not any("__pycache__" in n or "/tests/" in n or n.startswith("./tests") for n in names)


def test_upload_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MAX_UPLOAD_MB", 0)
    with pytest.raises(ToolError) as e:
        plan_for(tmp_path, public_test=True)
    assert "アップロード" in str(e.value)


def test_job_json_carries_the_notebook_recipes(tmp_path):
    job = plan_for(tmp_path, public_test=True, engines="voxcpm2,cosyvoice3").job
    assert job["score_env"]["python"] == "3.12" and "speechbrain==1.0.3" in job["score_env"]["packages"]
    g = {x["name"]: x for x in job["engine_groups"]}
    assert g["voxcpm2"]["env"]["packages"][2] == "voxcpm==2.0.3"
    assert g["cosyvoice3"]["env"]["python"] == "3.10" and g["cosyvoice3"]["env"]["clone"]["recursive"] is True
    assert job["uv_version"] == "0.11.32" and job["bucket"] == BUCKET and job["prefix"] == "exp/exp01"
    json.dumps(job)       # JSON にできる


# ---------------------------------------------------------------------------
# SSM に送る内容
# ---------------------------------------------------------------------------
def test_remote_command_is_short_posix_safe_and_free_of_secrets():
    lines = remote_command_lines(BUCKET, "exp/exp01", REGION)
    text = "\n".join(lines)
    assert len(text) < 2000                                              # SendCommand の文書は 64KB まで。大きな中身は S3 経由
    assert f"--bucket {BUCKET}" in text and "--prefix exp/exp01" in text and f"AWS_DEFAULT_REGION={REGION}" in text
    assert "exec python3 /opt/voicelab/remote_job.py" in text
    assert "AKIA" not in text and "SECRET" not in text.upper()


# ---------------------------------------------------------------------------
# ログの追跡
# ---------------------------------------------------------------------------
def put(aws, key, body):
    aws.client("s3").put_object(Bucket=BUCKET, Key=key, Body=body if isinstance(body, bytes) else body.encode("utf-8"))


@pytest.fixture
def bucket_ready(aws, state):
    do_setup(aws, state)
    return aws


def test_log_tail_returns_only_new_complete_lines(bucket_ready):
    tail = LogTail(bucket_ready, BUCKET, "exp/e/logs/run.log")
    assert tail.poll() == ""                                             # まだ無い
    put(bucket_ready, "exp/e/logs/run.log", "一行目\nふたつ目\n途中")
    assert tail.poll() == "一行目\nふたつ目\n"
    assert tail.poll() == ""
    put(bucket_ready, "exp/e/logs/run.log", "一行目\nふたつ目\n途中の続き\n")
    assert tail.poll() == "途中の続き\n"
    put(bucket_ready, "exp/e/logs/run.log", "一行目\nふたつ目\n途中の続き\nさいご")
    assert tail.poll() == "" and tail.poll(final=True) == "さいご"


def test_log_tail_does_not_split_multibyte_characters(bucket_ready):
    tail = LogTail(bucket_ready, BUCKET, "exp/e/logs/run.log")
    data = "あいうえお\n".encode("utf-8")
    put(bucket_ready, "exp/e/logs/run.log", data[:4])               # 「あ」の途中でちぎれた
    assert tail.poll() == ""
    put(bucket_ready, "exp/e/logs/run.log", data)
    assert tail.poll() == "あいうえお\n"


def test_log_tail_restarts_if_the_log_was_recreated(bucket_ready):
    tail = LogTail(bucket_ready, BUCKET, "exp/e/logs/run.log")
    put(bucket_ready, "exp/e/logs/run.log", "x" * 100 + "\n")
    tail.poll()
    put(bucket_ready, "exp/e/logs/run.log", "new\n")
    assert tail.poll() == ""                # 範囲外 → 先頭に戻す
    assert tail.poll() == "new\n"


def test_follow_streams_log_chunks_until_done(bucket_ready, capsys):
    ssm = bucket_ready.client("ssm")
    cid = ssm.send_command(InstanceIds=["i-1"], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": ["python3 remote_job.py"]})["Command"]["CommandId"]
    ssm.plans[cid] = ["InProgress"]
    chunks = ["[ec2run] sysinfo\n", "[ec2run] sysinfo\n$ apt-get\n", "[ec2run] sysinfo\n$ apt-get\n完了\n"]
    steps = {"n": 0}

    def sleeper(sec):
        i = steps["n"]
        steps["n"] += 1
        if i < len(chunks):
            put(bucket_ready, "exp/e/logs/run.log", chunks[i])
            put(bucket_ready, "exp/e/logs/status.json", json.dumps({"phase": f"p{i}", "elapsed_s": 5 * i, "finished": False}))
        if i == len(chunks) - 1:
            put(bucket_ready, "exp/e/logs/done.json", json.dumps({"ok": True, "exit_code": 0}))

    lines = []
    fr = follow_command(bucket_ready, bucket=BUCKET, prefix="exp/e", command_id=cid, instance_id="i-1", poll=1, sleep=sleeper,
                        monotonic=lambda: 0.0, out=lines.append)
    assert fr.reason == "done" and fr.done == {"ok": True, "exit_code": 0}
    shown = [ln for ln in lines if ln.startswith("  | ")]
    assert shown == ["  | [ec2run] sysinfo", "  | $ apt-get", "  | 完了"]
    assert any("[段階] p0" in ln for ln in lines) and any("[段階] p2" in ln for ln in lines)


def test_follow_stops_when_ssm_reports_failure(bucket_ready):
    ssm = bucket_ready.client("ssm")
    cid = ssm.send_command(InstanceIds=["i-1"], DocumentName="AWS-RunShellScript", Parameters={"commands": ["remote_job.py"]})["Command"]["CommandId"]
    ssm.plans[cid] = ["InProgress", "Failed"]
    fr = follow_command(bucket_ready, bucket=BUCKET, prefix="exp/e", command_id=cid, instance_id="i-1", poll=1,
                        sleep=lambda s: None, monotonic=lambda: 0.0, out=lambda s: None)
    assert fr.reason == "ssm" and fr.ssm_status == "Failed" and fr.done is None


def test_follow_notices_a_terminated_instance(bucket_ready, state):
    iid = do_launch(bucket_ready, state)
    bucket_ready.client("ec2").terminate_instances(InstanceIds=[iid])
    ssm = bucket_ready.client("ssm")
    cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript", Parameters={"commands": ["remote_job.py"]})["Command"]["CommandId"]
    ssm.plans[cid] = ["InProgress"]
    lines = []
    fr = follow_command(bucket_ready, bucket=BUCKET, prefix="exp/e", command_id=cid, instance_id=iid, poll=1,
                        sleep=lambda s: None, monotonic=lambda: 0.0, out=lines.append)
    assert fr.reason == "instance-gone" and any("自動停止" in ln for ln in lines)


# ---------------------------------------------------------------------------
# 結果の回収
# ---------------------------------------------------------------------------
def fill_results(aws, exp="exp01", with_done=True):
    put(aws, f"exp/{exp}/result/voicelab_result.xlsx", b"PK-xlsx")
    put(aws, f"exp/{exp}/result/manifest.csv", "engine,status\nvoxcpm2,ok\n")
    put(aws, f"exp/{exp}/result/audio/a.wav", b"RIFF-a")
    put(aws, f"exp/{exp}/result/refs/r.wav", b"RIFF-r")
    put(aws, f"exp/{exp}/logs/run.log", "log line\n")
    if with_done:
        put(aws, f"exp/{exp}/logs/done.json", json.dumps({"ok": True, "exit_code": 0, "elapsed_s": 100, "engines": {"voxcpm2": {"env_ok": True, "generate_rc": 0}},
                                                          "manifest": {"voxcpm2": {"ok": 3}}}))


def test_download_results_layout_and_skip_audio(bucket_ready, tmp_path):
    fill_results(bucket_ready)
    out = tmp_path / "outputs" / "exp01"
    s = download_results(bucket_ready, BUCKET, "exp/exp01", out)
    assert (out / "voicelab_result.xlsx").read_bytes() == b"PK-xlsx" and (out / "audio" / "a.wav").exists()
    assert (out / "ec2run_logs" / "run.log").read_text() == "log line\n" and (out / "ec2run_logs" / "done.json").exists()
    assert s.files == 6
    out2 = tmp_path / "outputs" / "exp01b"
    s2 = download_results(bucket_ready, BUCKET, "exp/exp01", out2, skip_audio=True)
    assert not (out2 / "audio").exists() and (out2 / "voicelab_result.xlsx").exists() and s2.skipped == 2


def test_download_never_writes_outside_the_destination(bucket_ready, tmp_path):
    put(bucket_ready, "exp/exp01/result/../../evil.txt", b"x")
    put(bucket_ready, "exp/exp01/result/ok.txt", b"ok")
    out = tmp_path / "dest"
    download_results(bucket_ready, BUCKET, "exp/exp01", out)
    assert (out / "ok.txt").exists() and not (tmp_path / "evil.txt").exists() and not list(tmp_path.rglob("evil.txt"))


def test_default_output_folder_is_meeting_outputs_exp():
    assert runner.out_dir_for("exp07", None) == common.MEETING_DIR / "outputs" / "exp07"
    assert str(runner.out_dir_for("e", "/x/y")) == "/x/y"


def test_fetch_downloads_without_touching_instances(bucket_ready, state, tmp_path, capsys):
    fill_results(bucket_ready)
    bucket_ready.calls.clear()
    assert run_fetch(bucket_ready, state, Args(exp="exp01", out_dir=str(tmp_path / "o"), skip_audio=False)) == 0
    assert (tmp_path / "o" / "voicelab_result.xlsx").exists()
    assert "結果: 成功" in capsys.readouterr().out
    assert bucket_ready.ops("ssm") == [] and bucket_ready.mutating() == []
    with pytest.raises(ToolError):
        run_fetch(bucket_ready, state, Args(exp="nothing", out_dir=str(tmp_path / "o2"), skip_audio=False))


# ---------------------------------------------------------------------------
# run 全体（偽のインスタンス）
# ---------------------------------------------------------------------------
def long_commands(aws):
    """実験を起動する SendCommand（launch の確認用の短いコマンドは除く）。"""
    return [s for s in aws.client("ssm").sent if any("remote_job.py" in c for c in s["Parameters"]["commands"])]


@pytest.fixture
def running(aws, state):
    do_setup(aws, state)
    iid = do_launch(aws, state)
    aws.calls.clear()
    return aws, iid


def test_run_refuses_without_yes_and_uploads_nothing(running, state, tmp_path, capsys):
    aws, iid = running
    with pytest.raises(NeedConfirmation) as e:
        run_run(aws, state, rargs(public_test=True, yes=False, out_dir=str(tmp_path / "o")), sleep=lambda s: None)
    out = capsys.readouterr().out
    assert "データの規則" in out and "自分の声" in out and "同意" in out and "公開データ" in out and "社外秘" in out or "会社の会議音声" in out
    assert "--yes" in str(e.value) and "データの規則" in str(e.value)
    assert aws.mutating() == []
    assert aws.ops("ssm") == []
    assert aws.client("s3").list_objects_v2(Bucket=BUCKET).get("KeyCount", 0) == 0


def test_full_run_sends_the_command_follows_the_log_and_downloads(running, state, tmp_path, capsys):
    aws, iid = running
    ssm = aws.client("ssm")

    def on_send(kw, cid):                         # 偽のインスタンスが S3 に書くもの
        fill_results(aws)

    ssm.on_send = on_send
    out = tmp_path / "out"
    rc = run_run(aws, state, rargs(public_test=True, dry_run_first=True, out_dir=str(out)), sleep=lambda s: None, poll=1)
    assert rc == 0
    sent = [s for s in ssm.sent if any("remote_job.py" in c for c in s["Parameters"]["commands"])]
    assert len(sent) == 1
    s = sent[0]
    assert s["DocumentName"] == "AWS-RunShellScript" and s["InstanceIds"] == [iid]
    assert s["Parameters"]["commands"][0] == "#!/bin/bash"
    assert s["OutputS3BucketName"] == BUCKET and s["OutputS3KeyPrefix"] == "exp/exp01/ssm-output"
    assert len(s["Comment"]) <= 100 and s["TimeoutSeconds"] >= 30
    assert 900 <= int(s["Parameters"]["executionTimeout"][0]) <= 3 * 3600                 # 自動停止の時刻までに収まる
    assert len(json.dumps(s["Parameters"])) < 4000
    keys = [o["Key"] for o in aws.client("s3").list_objects_v2(Bucket=BUCKET, Prefix="exp/exp01/job/")["Contents"]]
    assert {"exp/exp01/job/job.json", "exp/exp01/job/remote_job.py", "exp/exp01/job/exp.json", "exp/exp01/job/prepare_public.py"} <= set(keys)
    job = json.loads(aws.client("s3").get_object(Bucket=BUCKET, Key="exp/exp01/job/job.json")["Body"].read())
    assert job["dry_run_first"] is True and job["bucket"] == BUCKET
    assert (out / "voicelab_result.xlsx").exists() and (out / "ec2run_info.json").exists()
    info = json.loads((out / "ec2run_info.json").read_text())
    assert info["instance_id"] == iid and info["command_id"] == s["CommandId"]
    text = capsys.readouterr().out
    assert "結果の Excel" in text and "teardown --yes" in text and "データの規則" in text
    assert aws.settings.secret_access_key not in text and "123456789012" not in text
    rec = common.State(state.path).last_run("exp01")
    assert rec["command_id"] == s["CommandId"] and rec["status"] == "done"


def test_run_survives_without_done_marker_but_reports_it(running, state, tmp_path, capsys):
    aws, iid = running
    ssm = aws.client("ssm")
    ssm.on_send = lambda kw, cid: fill_results(aws, with_done=False)
    ssm.long_statuses = ["InProgress", "Failed"]
    ssm.stderr = "Traceback: boom"
    rc = run_run(aws, state, rargs(public_test=True, out_dir=str(tmp_path / "o")), sleep=lambda s: None, poll=1)
    assert rc == 1
    out = capsys.readouterr().out
    assert "done.json" in out and "boom" in out
    assert (tmp_path / "o" / "voicelab_result.xlsx").exists()                      # 途中結果でも回収する


def test_existing_results_need_resume_or_fresh(running, state, tmp_path):
    aws, iid = running
    fill_results(aws)
    with pytest.raises(ToolError) as e:
        run_run(aws, state, rargs(public_test=True, out_dir=str(tmp_path / "o")), sleep=lambda s: None)
    assert "--resume" in str(e.value) and "--fresh" in str(e.value)
    assert long_commands(aws) == []


def test_fresh_deletes_old_s3_data_and_resume_keeps_it(running, state, tmp_path):
    aws, iid = running
    ssm = aws.client("ssm")
    fill_results(aws)
    put(aws, "exp/exp01/result/stale.txt", "old")
    run_run(aws, state, rargs(public_test=True, resume=True, out_dir=str(tmp_path / "o1")), sleep=lambda s: None, poll=1)
    s3 = aws.client("s3")
    assert "stale.txt" in {k["Key"].rsplit("/", 1)[-1] for k in s3.list_objects_v2(Bucket=BUCKET, Prefix="exp/exp01/result/")["Contents"]}
    ssm.on_send = lambda kw, cid: put(aws, "exp/exp01/logs/done.json", json.dumps({"ok": True, "exit_code": 0}))
    run_run(aws, state, rargs(public_test=True, fresh=True, out_dir=str(tmp_path / "o2")), sleep=lambda s: None, poll=1)
    keys = {k["Key"] for k in s3.list_objects_v2(Bucket=BUCKET, Prefix="exp/exp01/")["Contents"]}
    assert "exp/exp01/result/stale.txt" not in keys and "exp/exp01/result/voicelab_result.xlsx" not in keys


def test_logs_from_a_previous_attempt_are_cleared_before_a_new_run(running, state, tmp_path):
    aws, iid = running
    put(aws, "exp/exp01/logs/done.json", json.dumps({"ok": True}))         # 前回の完了の印が残っていると、すぐ終わったと誤解する
    ssm = aws.client("ssm")
    seen = {}

    def on_send(kw, cid):
        seen["done_at_send"] = aws.client("s3").list_objects_v2(Bucket=BUCKET, Prefix="exp/exp01/logs/done.json").get("KeyCount", 0)
        put(aws, "exp/exp01/logs/done.json", json.dumps({"ok": True, "exit_code": 0}))

    ssm.on_send = on_send
    run_run(aws, state, rargs(public_test=True, out_dir=str(tmp_path / "o")), sleep=lambda s: None, poll=1)
    assert seen["done_at_send"] == 0


def test_attach_reconnects_without_uploading(running, state, tmp_path, capsys):
    aws, iid = running
    ssm = aws.client("ssm")
    cid = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript", Parameters={"commands": ["remote_job.py"]})["Command"]["CommandId"]
    state.record_run("exp01", command_id=cid, instance_id=iid)
    ssm.plans[cid] = ["InProgress"]
    fill_results(aws)
    aws.calls.clear()
    rc = run_run(aws, state, rargs(attach=True, out_dir=str(tmp_path / "o3")), sleep=lambda s: None, poll=1)
    assert rc == 0 and (tmp_path / "o3" / "voicelab_result.xlsx").exists()
    assert [c for c in aws.calls if c[1] in ("PutObject", "SendCommand")] == []
    with pytest.raises(ToolError):
        run_run(aws, state, rargs(attach=True, exp="unknown"), sleep=lambda s: None)


def test_run_needs_exactly_one_live_instance(aws, state):
    do_setup(aws, state)
    with pytest.raises(ToolError) as e:
        run_run(aws, state, rargs(public_test=True), sleep=lambda s: None)
    assert "launch" in str(e.value)
    do_launch(aws, state)
    from meeting.cloud.aws.launch import run_launch
    from test_cloud_aws_helpers import launch_args

    run_launch(aws, state, launch_args(allow_multiple=True), sleep=lambda s: None)
    with pytest.raises(ToolError) as e:
        run_run(aws, state, rargs(public_test=True), sleep=lambda s: None)
    assert "--instance-id" in str(e.value)


def test_run_does_not_target_an_untagged_instance(running, state):
    aws, iid = running
    foreign = aws.client("ec2").run_instances(ImageId=aws.ami_id, InstanceType="t3.micro", MinCount=1, MaxCount=1)["Instances"][0]["InstanceId"]
    with pytest.raises(ToolError) as e:
        run_run(aws, state, rargs(public_test=True, instance_id=foreign), sleep=lambda s: None)
    assert "このツールで起動した" in str(e.value)
    assert long_commands(aws) == []


# ---------------------------------------------------------------------------
# 通信が一時的に切れても、実験の追跡を続ける
# ---------------------------------------------------------------------------
def test_follow_survives_transient_network_errors(bucket_ready, monkeypatch):
    ssm = bucket_ready.client("ssm")
    cid = ssm.send_command(InstanceIds=["i-1"], DocumentName="AWS-RunShellScript", Parameters={"commands": ["remote_job.py"]})["Command"]["CommandId"]
    ssm.plans[cid] = ["InProgress"]
    real = runner.read_json_object
    calls = {"n": 0}

    def flaky(aws, bucket, key):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise ConnectionError("proxy hiccup")
        if key.endswith("done.json"):
            return {"ok": True, "exit_code": 0}
        return real(aws, bucket, key)

    monkeypatch.setattr(runner, "read_json_object", flaky)
    lines = []
    fr = follow_command(bucket_ready, bucket=BUCKET, prefix="exp/e", command_id=cid, instance_id="i-1", poll=1,
                        sleep=lambda s: None, monotonic=lambda: 0.0, out=lines.append)
    assert fr.reason == "done" and any("一時的に失敗" in ln for ln in lines)


def test_follow_gives_up_after_ten_consecutive_failures_and_points_to_attach(bucket_ready, monkeypatch):
    monkeypatch.setattr(runner, "read_json_object", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(ToolError) as e:
        follow_command(bucket_ready, bucket=BUCKET, prefix="exp/exp01", command_id="c", instance_id="i-1", poll=1,
                       sleep=lambda s: None, monotonic=lambda: 0.0, out=lambda s: None)
    assert "--attach" in str(e.value) and "exp01" in str(e.value)


def test_follow_does_not_hide_permission_errors(bucket_ready, monkeypatch):
    def denied(*a, **k):
        raise client_error("AccessDenied", "no", "GetObject")

    monkeypatch.setattr(runner, "read_json_object", denied)
    with pytest.raises(ToolError) as e:
        follow_command(bucket_ready, bucket=BUCKET, prefix="exp/e", command_id="c", instance_id="i-1", poll=1,
                       sleep=lambda s: None, monotonic=lambda: 0.0, out=lambda s: None)
    assert "権限が足りません" in str(e.value)
