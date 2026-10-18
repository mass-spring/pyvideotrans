"""remote_job.py（インスタンス側の実行役）: 偽のコマンド実行役と moto の S3 で、手順の順番・失敗時の動き・保存を確かめる。

実際の uv / TTS モデルの実行は GPU インスタンスでしか確かめられない。ここで確かめるのは「何をどの順で実行するか」。
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from test_cloud_aws_helpers import Args, REGION, aws, do_setup, mocked, state  # noqa: F401
from test_cloud_aws_run import BUCKET, make_profile, plan_for, rargs  # noqa: F401

from meeting.cloud.aws import recipes, remote_job
from meeting.cloud.aws.remote_job import CommandRunner, Fatal, Job, LogUploader, RemoteLog, S3Store, filter_requirements, safe_extract
from meeting.cloud.aws.runner import upload_plan

MANIFEST_HEAD = "job_id,speaker,engine,status,error\n"


class FakeRunner(CommandRunner):
    """コマンドを実行せず、記録して、必要なファイルだけ作る。fail(cmd) が True のコマンドは失敗させる。"""

    def __init__(self, log, work: Path, *, fail=None, gpu=True, cosy_requirements: str | None = None):
        super().__init__(log, {})
        self.work = work
        self.cmds: list[tuple[list[str] | str, dict]] = []
        self.fail = fail or (lambda cmd: False)
        self.gpu = gpu
        self.cosy_requirements = cosy_requirements

    @staticmethod
    def text(cmd):
        return cmd if isinstance(cmd, str) else " ".join(cmd)

    def capture(self, cmd):
        self.cmds.append((cmd, {}))
        if cmd[0] == "nvidia-smi":
            return (0, "Tesla T4, 15360 MiB, 595.91.07") if self.gpu else (9, "no gpu")
        if cmd[:2] == ["git", "-C"] or "rev-parse" in cmd:
            return 0, "0123456789abcdef0123456789abcdef01234567\n"
        return 0, ""

    def run(self, cmd, *, env=None, cwd=None):
        self.cmds.append((cmd, dict(env or {})))
        self.log.line("$ " + self.text(cmd))
        if self.fail(cmd):
            return 1
        if isinstance(cmd, str):
            if "install.sh" in cmd:
                (self.work / "bin").mkdir(parents=True, exist_ok=True)
                (self.work / "bin" / "uv").write_text("#!/bin/sh\n")
            return 0
        if cmd[0].endswith("/uv") and cmd[1] == "venv":
            py = Path(cmd[-1]) / "bin" / "python"
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        elif cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            if dest.name == "CosyVoice":
                (dest / "cosyvoice").mkdir(parents=True, exist_ok=True)
                (dest / "requirements.txt").write_text(self.cosy_requirements or "onnxruntime-gpu==1.18.0\n")
            else:
                (dest / "meeting" / "src").mkdir(parents=True, exist_ok=True)
        elif "voicelab" in cmd and "-m" in cmd:
            self.simulate_voicelab(cmd, env or {})
        return 0

    def simulate_voicelab(self, cmd, env):
        i = cmd.index("voicelab")
        sub = cmd[i + 1]
        out = Path(cmd[cmd.index("--out") + 1]) if "--out" in cmd else None
        if sub == "generate" and out is not None:
            eng = cmd[cmd.index("--engines") + 1]
            out.mkdir(parents=True, exist_ok=True)
            man = out / "manifest.csv"
            if not man.exists():
                man.write_text(MANIFEST_HEAD, encoding="utf-8-sig")
            with man.open("a", encoding="utf-8-sig") as f:
                f.write(f"j-{eng}-1,A,{eng},ok,\nj-{eng}-2,A,{eng},failed,boom\n")
            (out / "audio").mkdir(exist_ok=True)
            (out / "audio" / f"{eng}.wav").write_bytes(b"RIFF" + eng.encode())
        elif sub == "score" and out is not None:
            (out / "voicelab_result.xlsx").write_bytes(b"PK-xlsx")
            (out / "pairs.csv").write_text("speaker,ref,test,label\n")


def labels(runner: FakeRunner) -> list[str]:
    """実行されたコマンドを、読みやすい短い名前の並びにする。"""
    out = []
    for cmd, _env in runner.cmds:
        t = runner.text(cmd)
        if t.startswith("nvidia-smi"):
            continue
        if "apt-get" in t and "install" in t:
            out.append("apt-install")
        elif "install.sh" in t:
            out.append("uv-install")
        elif t.startswith("git clone") and "pyvideotrans" in t:
            out.append("clone-repo")
        elif t.startswith("git clone") and "CosyVoice" in t:
            out.append("clone-cosy")
        elif " venv " in t:
            out.append("venv:" + Path(cmd[-1]).name)
        elif " pip install " in t:
            py = cmd[cmd.index("--python") + 1]
            out.append("pip:" + Path(py).parent.parent.name)
        elif "prepare_public.py" in t:
            out.append("prepare-public")
        elif "write_targets" in t:
            out.append("default-targets")
        elif " -m voicelab " in t:
            parts = t.split(" -m voicelab ")[1].split()
            sub = parts[0]
            if sub == "run" and "--dry-run" in parts:
                out.append("voicelab-dry-run")
            elif sub == "generate":
                out.append("generate:" + parts[parts.index("--engines") + 1])
            else:
                out.append("voicelab-" + sub)
        elif "import " in t:
            out.append("check-import:" + Path(cmd[0]).parent.parent.name)
    return out


class Env:
    """1 回分のジョブ（偽のインスタンス）。"""

    def __init__(self, aws, state, tmp_path, **plan_kw):
        self.aws = aws
        do_setup(aws, state)
        self.s3 = S3Store(BUCKET, REGION, client=aws.client("s3"))
        self.work = tmp_path / "vl"
        plan_kw.setdefault("public_test", True)
        plan = plan_for(tmp_path, **plan_kw)
        self.plan = plan
        upload_plan(aws, BUCKET, plan)
        self.spec = json.loads(json.dumps(plan.job))
        # 本物のパス（/opt/voicelab）を、このテスト用の場所に付け替える
        self.spec["workdir"] = str(self.work)
        for k, v in self.spec["paths"].items():
            self.spec["paths"][k] = v.replace("/opt/voicelab", str(self.work))
        for inp in self.spec["inputs"]:
            inp["dest"] = inp["dest"].replace("/opt/voicelab", str(self.work))
        if self.spec.get("public_test"):
            self.spec["public_test"]["dest"] = self.spec["public_test"]["dest"].replace("/opt/voicelab", str(self.work))
        if self.spec.get("default_targets"):
            self.spec["default_targets"] = self.spec["default_targets"].replace("/opt/voicelab", str(self.work))
        self.log = RemoteLog(tmp_path / "run.log", echo=False)

    def job(self, **runner_kw):
        self.runner = FakeRunner(self.log, self.work, **runner_kw)
        return Job(self.spec, self.s3, self.runner, self.log)

    def s3_keys(self, prefix=""):
        resp = self.aws.client("s3").list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        return {o["Key"] for o in resp.get("Contents", [])}

    def done(self):
        return json.loads(self.aws.client("s3").get_object(Bucket=BUCKET, Key="exp/exp01/logs/done.json")["Body"].read())


@pytest.fixture
def env(aws, state, tmp_path):
    return Env(aws, state, tmp_path, engines="voxcpm2,qwen3tts,chatterbox", dry_run_first=True)


# ---------------------------------------------------------------------------
# 手順の順番
# ---------------------------------------------------------------------------
def test_happy_path_runs_the_notebook_steps_in_order(env):
    job = env.job()
    assert job.run() == 0
    seq = labels(env.runner)
    expected = ["apt-install", "uv-install", "clone-repo", "venv:score", "pip:score", "pip:score", "pip:score", "pip:score", "prepare-public", "default-targets",
                "voicelab-plan", "voicelab-dry-run",
                "venv:voxcpm2", "pip:voxcpm2", "check-import:voxcpm2", "generate:voxcpm2",
                "venv:qwen3tts", "pip:qwen3tts", "check-import:qwen3tts", "generate:qwen3tts",
                "venv:chatterbox", "pip:chatterbox", "check-import:chatterbox", "generate:chatterbox",
                "voicelab-score"]
    assert seq == expected, seq
    assert env.done()["ok"] is True and env.done()["exit_code"] == 0


def test_dry_run_happens_before_any_engine_environment_is_built(env):
    env.job().run()
    seq = labels(env.runner)
    assert seq.index("voicelab-dry-run") < seq.index("venv:voxcpm2")


def test_without_dry_run_first_there_is_no_dry_run(aws, state, tmp_path):
    e = Env(aws, state, tmp_path, engines="voxcpm2", dry_run_first=False)
    e.job().run()
    assert "voicelab-dry-run" not in labels(e.runner)


def exact(runner: FakeRunner, needle: str):
    return [runner.text(c) for c, _ in runner.cmds if needle in runner.text(c)]


def test_environment_commands_match_the_notebook_pins(env):
    env.job().run()
    w = str(env.work)
    uv = f"{w}/bin/uv"
    r = env.runner
    assert f"{uv} venv --python 3.12 {w}/envs/score" in exact(r, " venv ")
    assert f"{uv} pip install --python {w}/envs/score/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0" in exact(r, "pip install")
    assert (f"{uv} pip install --python {w}/envs/score/bin/python numpy scipy soundfile librosa openpyxl pandas praat-parselmouth "
            "speechbrain==1.0.3") in exact(r, "pip install")
    assert f"{uv} pip install --python {w}/envs/score/bin/python setuptools<81 pyworld" in exact(r, "pip install")
    assert f"{uv} venv --python 3.11 {w}/envs/chatterbox" in exact(r, " venv ")
    assert f"{uv} pip install --python {w}/envs/chatterbox/bin/python chatterbox-tts==0.1.7 soundfile scipy" in exact(r, "pip install")
    assert (f"{uv} pip install --python {w}/envs/voxcpm2/bin/python torch==2.6.0 torchaudio==2.6.0 voxcpm==2.0.3 numpy soundfile scipy librosa"
            in exact(r, "pip install"))
    assert any("astral.sh/uv/0.11.32/install.sh" in t and "UV_UNMANAGED_INSTALL" in t for t in exact(r, "install.sh"))
    assert any("rubberband-cli sox libsox-dev ffmpeg" in t for t in exact(r, "apt-get"))


def test_voicelab_is_run_per_engine_with_the_notebook_environment(env):
    env.job().run()
    gens = [(c, e) for c, e in env.runner.cmds if isinstance(c, list) and "generate" in c]
    assert len(gens) == 3
    src = env.spec["paths"]["src"]
    for cmd, e in gens:
        assert cmd[1:4] == ["-u", "-m", "voicelab"] and cmd[-2] == "--engines"
        assert Path(cmd[0]).parent.parent.name == cmd[-1]                    # 仮想環境の名前 = --engines の値
        assert e["PYTHONPATH"] == src and e["HF_HOME"].endswith("hf_cache") and e["COSYVOICE_REPO"].endswith("CosyVoice")
    score = [c for c, _ in env.runner.cmds if isinstance(c, list) and "score" in c and "voicelab" in c and c[c.index("voicelab") + 1] == "score"][0]
    assert score[score.index("--embedder") + 1] == "ecapa" and Path(score[0]).parent.parent.name == "score"


def test_dry_run_config_is_the_notebooks_dry_config(env):
    env.job().run()
    dry = json.loads((env.work / "work" / "exp01" / "dry_exp.json").read_text())
    assert dry["engines"] == ["voxcpm2"] and dry["repeats"] == 1 and dry["max_texts_per_lang"] == 1 and dry["embedder"] == "none"
    cmd = [c for c, _ in env.runner.cmds if isinstance(c, list) and "--dry-run" in c][0]
    assert cmd[cmd.index("voicelab") + 1:cmd.index("voicelab") + 3] == ["run", "--config"] and "--embedder" in cmd


# ---------------------------------------------------------------------------
# 失敗したとき
# ---------------------------------------------------------------------------
def test_dry_run_failure_stops_before_spending_gpu_time(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and "--dry-run" in cmd)
    assert job.run() == 7
    seq = labels(env.runner)
    assert not any(s.startswith("venv:voxcpm2") or s.startswith("generate") for s in seq)
    d = env.done()
    assert d["ok"] is False and "ダミー" in d["error"]


def test_one_engine_failing_does_not_stop_the_others(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and "qwen-tts==0.1.1" in cmd)
    assert job.run() == 0
    seq = labels(env.runner)
    assert "generate:voxcpm2" in seq and "generate:chatterbox" in seq and "generate:qwen3tts" not in seq
    d = env.done()
    assert d["engines"]["qwen3tts"]["env_ok"] is False and d["engines"]["voxcpm2"]["generate_rc"] == 0
    assert d["ok"] is True


def test_generate_failure_is_recorded_and_the_rest_continues(env):
    env.job(fail=lambda cmd: isinstance(cmd, list) and "generate" in cmd and cmd[-1] == "voxcpm2").run()
    d = env.done()
    assert d["engines"]["voxcpm2"]["generate_rc"] == 1
    assert "generate:chatterbox" in labels(env.runner) and "voicelab-score" in labels(env.runner)


def test_no_gpu_stops_immediately(env):
    job = env.job(gpu=False)
    assert job.run() == 3
    assert labels(env.runner) == []
    assert "GPU" in env.done()["error"]


def test_score_failure_is_reported_but_results_are_saved(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and "voicelab" in cmd and cmd[cmd.index("voicelab") + 1] == "score")
    assert job.run() == 1
    d = env.done()
    assert d["ok"] is False and d["score_rc"] == 1
    keys = env.s3_keys("exp/exp01/result/")
    assert "exp/exp01/result/manifest.csv" in keys and "exp/exp01/result/audio/voxcpm2.wav" in keys


def test_plan_failure_is_a_config_problem_and_stops(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and "voicelab" in cmd and cmd[cmd.index("voicelab") + 1] == "plan")
    assert job.run() == 5
    assert "plan" in env.done()["error"]
    assert not any(s.startswith("venv:voxcpm2") for s in labels(env.runner))


def test_score_env_failure_is_fatal(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and "speechbrain==1.0.3" in cmd)
    assert job.run() == 4


def test_unexpected_exception_still_saves_log_and_done(env, monkeypatch):
    job = env.job()
    monkeypatch.setattr(job, "get_code", lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
    assert job.run() == 1
    d = env.done()
    assert "kaboom" in d["error"] and d["ok"] is False


def test_pyworld_failure_is_tolerated(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and "pyworld" in cmd)
    assert job.run() == 0


# ---------------------------------------------------------------------------
# 保存と再開
# ---------------------------------------------------------------------------
def test_results_are_saved_to_s3_after_each_engine_and_at_the_end(env):
    saved = []
    orig = env.s3.sync_up

    def spy(local, prefix):
        n = orig(local, prefix)
        saved.append((prefix, sorted(p.name for p in local.rglob("*.wav"))))
        return n

    env.s3.sync_up = spy
    env.job().run()
    prefixes = [p for p, _ in saved]
    assert prefixes.count("exp/exp01/result") == 4                      # 3 エンジン + 最後
    assert saved[0][1] == ["voxcpm2.wav"] and saved[1][1] == ["qwen3tts.wav", "voxcpm2.wav"]
    keys = env.s3_keys("exp/exp01/result/")
    assert {"exp/exp01/result/voicelab_result.xlsx", "exp/exp01/result/manifest.csv", "exp/exp01/result/pairs.csv"} <= keys


def test_rerun_restores_previous_results_from_s3(env):
    env.aws.client("s3").put_object(Bucket=BUCKET, Key="exp/exp01/result/audio/old.wav", Body=b"RIFFold")
    env.job().run()
    assert (env.work / "work" / "exp01" / "result" / "audio" / "old.wav").read_bytes() == b"RIFFold"


def test_status_and_done_markers_are_written(env):
    env.job().run()
    st = json.loads(env.aws.client("s3").get_object(Bucket=BUCKET, Key="exp/exp01/logs/status.json")["Body"].read())
    assert st["finished"] is True
    d = env.done()
    assert [t["phase"] for t in d["timeline"]][:3] == ["sysinfo", "system", "code"]
    assert d["manifest"]["voxcpm2"] == {"ok": 1, "failed": 1}


def test_sync_up_skips_unchanged_wavs_but_reuploads_changed_files(aws, state, tmp_path):
    do_setup(aws, state)
    s3 = S3Store(BUCKET, REGION, client=aws.client("s3"))
    d = tmp_path / "res"
    (d / "audio").mkdir(parents=True)
    (d / "audio" / "a.wav").write_bytes(b"1234")
    (d / "manifest.csv").write_text("a\n")
    assert s3.sync_up(d, "exp/x/result") == 2
    assert s3.sync_up(d, "exp/x/result") == 1                          # wav は飛ばし、manifest は上げ直す
    (d / "audio" / "a.wav").write_bytes(b"12345")
    assert s3.sync_up(d, "exp/x/result") == 2


# ---------------------------------------------------------------------------
# 入力・コード
# ---------------------------------------------------------------------------
def test_speaker_tar_and_config_are_downloaded_and_extracted(aws, state, tmp_path):
    prof = make_profile(tmp_path)
    e = Env(aws, state, tmp_path, public_test=False, speaker=[f"A={prof}"], engines="voxcpm2")
    e.job().run()
    spk = e.work / "inputs" / "exp01" / "spk0"
    assert (spk / "profile.json").exists() and (spk / "refs" / "ref_ja_01.wav").exists()
    cfg = json.loads((e.work / "work" / "exp01" / "exp.json").read_text())
    assert cfg["speakers"][0]["name"] == "A"
    seq = labels(e.runner)
    assert "prepare-public" not in seq and "default-targets" in seq


def test_public_test_prepares_the_public_speaker(env):
    env.job().run()
    c = [c for c, _ in env.runner.cmds if isinstance(c, list) and any("prepare_public.py" in x for x in c)][0]
    assert Path(c[0]).parent.parent.name == "score" and "--src" in c and c[c.index("--out") + 1].endswith("inputs/exp01/public1272")
    pip_public = [t for t in exact(env.runner, "pip install") if "pyarrow" in t]
    assert pip_public and "huggingface_hub" in pip_public[0]


def test_local_code_mode_extracts_the_uploaded_source(aws, state, tmp_path):
    e = Env(aws, state, tmp_path, engines="voxcpm2", code="local")
    e.job().run()
    assert "clone-repo" not in labels(e.runner)
    assert (Path(e.spec["paths"]["src"]) / "voicelab" / "runner.py").exists()


def test_git_mode_clones_the_meeting_translator_branch(env):
    env.job().run()
    clone = [c for c, _ in env.runner.cmds if isinstance(c, list) and c[:2] == ["git", "clone"] and "pyvideotrans" in " ".join(c)][0]
    assert clone[:6] == ["git", "clone", "--depth", "1", "--branch", "meeting-translator"]
    assert "https://github.com/mass-spring/pyvideotrans" in clone


def test_git_clone_failure_suggests_local_mode(env):
    job = env.job(fail=lambda cmd: isinstance(cmd, list) and cmd[:2] == ["git", "clone"])
    assert job.run() == 5
    assert "--code local" in env.done()["error"]


def test_cosyvoice_env_clones_filters_requirements_and_adds_extras(aws, state, tmp_path):
    reqs = "--extra-index-url https://x\n# comment\ndeepspeed==0.15.1\nonnxruntime-gpu==1.18.0\nGradio==5.4\ntorch==2.3.1\n\nuvicorn\npyworld==0.3.4\nnumpy==1.26.4\n"
    e = Env(aws, state, tmp_path, engines="cosyvoice3")
    e.job(cosy_requirements=reqs).run()
    seq = labels(e.runner)
    assert "clone-cosy" in seq and seq.index("clone-cosy") < seq.index("venv:cosyvoice3")
    pip = [t for t in exact(e.runner, "envs/cosyvoice3/bin/python") if "pip install" in t][0]
    pkgs = pip.split("bin/python ")[1].split()
    assert pkgs == ["onnxruntime-gpu==1.18.0", "torch==2.3.1", "numpy==1.26.4", "soundfile", "scipy", "huggingface_hub", "pykakasi==2.3.0"]
    assert "venv --python 3.10" in " ".join(exact(e.runner, "envs/cosyvoice3"))
    gens = [c for c, _ in e.runner.cmds if isinstance(c, list) and "generate" in c]
    assert gens and gens[0][-1] == "cosyvoice3"                          # グループ名を --engines に渡す（ノートブックと同じ）
    cfg = json.loads((e.work / "work" / "exp01" / "exp.json").read_text())
    assert cfg["engines"] == ["cosyvoice3:zero_shot", "cosyvoice3:cross_lingual"]


def test_filter_requirements_like_the_notebook():
    text = "# c\n--index-url x\n  deepspeed==1\nGradio\nonnxruntime\n\ntensorrt-cu12==10\nfastapi==0.1\ngrpcio==1\ntensorboard\nTORCH==2\nuvicorn\npyworld\n"
    assert filter_requirements(text, recipes.ENGINE_GROUPS["cosyvoice3"].env.drop_prefixes) == ["onnxruntime", "TORCH==2"]


def test_safe_extract_blocks_path_traversal_and_links(tmp_path):
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tf:
        ti = tarfile.TarInfo("../outside.txt")
        ti.size = 1
        import io

        tf.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(Fatal):
        safe_extract(evil, tmp_path / "dest")
    assert not (tmp_path / "outside.txt").exists()
    link = tmp_path / "link.tar"
    with tarfile.open(link, "w") as tf:
        ti = tarfile.TarInfo("l")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    with pytest.raises(Fatal):
        safe_extract(link, tmp_path / "dest2")


# ---------------------------------------------------------------------------
# ログ送信と main
# ---------------------------------------------------------------------------
def test_log_uploader_pushes_and_stops_cleanly(aws, state, tmp_path):
    do_setup(aws, state)
    s3 = S3Store(BUCKET, REGION, client=aws.client("s3"))
    log = RemoteLog(tmp_path / "x.log", echo=False)
    up = LogUploader(log, s3, "exp/x/logs/run.log", interval=0.05)
    up.start()
    log.line("hello")
    up.stop()
    up.join(timeout=5)
    assert not up.is_alive()
    body = aws.client("s3").get_object(Bucket=BUCKET, Key="exp/x/logs/run.log")["Body"].read().decode()
    assert "hello" in body


def test_main_writes_done_marker_after_the_final_log_upload(env, monkeypatch, tmp_path):
    order = []
    real = env.s3
    s3 = S3Store(BUCKET, REGION, client=env.aws.client("s3"))
    orig_up, orig_put = s3.upload, s3.put_text
    s3.upload = lambda p, k: (order.append(("upload", k)), orig_up(p, k))[1]
    s3.put_text = lambda k, t: (order.append(("put", k)), orig_put(k, t))[1]
    monkeypatch.setattr(remote_job, "S3Store", lambda bucket, region: s3)
    created = {}

    def runner_factory(log, base_env):
        created["r"] = FakeRunner(log, env.work)
        return created["r"]

    monkeypatch.setattr(remote_job, "CommandRunner", runner_factory)
    monkeypatch.setattr(remote_job, "LOG_INTERVAL", 0.05)
    # job.json の置き場所（workdir）をテスト用に付け替えて、S3 に置き直す
    env.aws.client("s3").put_object(Bucket=BUCKET, Key="exp/exp01/job/job.json", Body=json.dumps(env.spec).encode())
    rc = remote_job.main(["--bucket", BUCKET, "--prefix", "exp/exp01", "--region", REGION, "--workdir", str(env.work)])
    assert rc == 0
    puts = [i for i, o in enumerate(order) if o == ("put", "exp/exp01/logs/done.json")]
    log_uploads = [i for i, o in enumerate(order) if o == ("upload", "exp/exp01/logs/run.log")]
    assert puts and log_uploads and log_uploads[-1] < puts[-1]
    assert real is not None


def test_auto_terminate_runs_shutdown_after_everything_is_saved(env, monkeypatch):
    env.spec["shutdown_when_done"] = True
    s3 = S3Store(BUCKET, REGION, client=env.aws.client("s3"))
    monkeypatch.setattr(remote_job, "S3Store", lambda bucket, region: s3)
    monkeypatch.setattr(remote_job, "CommandRunner", lambda log, base_env: FakeRunner(log, env.work))
    env.aws.client("s3").put_object(Bucket=BUCKET, Key="exp/exp01/job/job.json", Body=json.dumps(env.spec).encode())
    calls = []
    monkeypatch.setattr(remote_job.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    remote_job.main(["--bucket", BUCKET, "--prefix", "exp/exp01", "--region", REGION, "--workdir", str(env.work)])
    assert calls and calls[-1][:3] == ["shutdown", "-h", "+1"]
    assert "exp/exp01/logs/done.json" in env.s3_keys("exp/exp01/logs/")


def test_no_auto_terminate_by_default(env, monkeypatch):
    s3 = S3Store(BUCKET, REGION, client=env.aws.client("s3"))
    monkeypatch.setattr(remote_job, "S3Store", lambda bucket, region: s3)
    monkeypatch.setattr(remote_job, "CommandRunner", lambda log, base_env: FakeRunner(log, env.work))
    env.aws.client("s3").put_object(Bucket=BUCKET, Key="exp/exp01/job/job.json", Body=json.dumps(env.spec).encode())
    calls = []
    monkeypatch.setattr(remote_job.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    remote_job.main(["--bucket", BUCKET, "--prefix", "exp/exp01", "--region", REGION, "--workdir", str(env.work)])
    assert calls == []


def test_remote_job_is_standard_library_only_at_import_time():
    src = Path(remote_job.__file__).read_text(encoding="utf-8")
    top_imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    allowed = {"argparse", "csv", "json", "os", "shutil", "subprocess", "sys", "tarfile", "threading", "time", "traceback", "collections",
               "pathlib", "typing", "__future__"}
    for ln in top_imports:
        mod = ln.split()[1].split(".")[0]
        assert mod in allowed, ln


def test_remote_job_and_helper_scripts_parse_on_python_3_10_syntax():
    import ast

    for name in ("remote_job.py", "prepare_public.py"):
        ast.parse((Path(remote_job.__file__).parent / name).read_text(encoding="utf-8"), feature_version=(3, 10))
