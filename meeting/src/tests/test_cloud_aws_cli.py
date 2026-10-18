"""コマンドライン: サブコマンド、--yes の強制、鍵の伏せ字、入口（python -m / スクリプト）。"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from test_cloud_aws_helpers import REPO_ROOT, Args, aws, do_launch, do_setup, mocked, state  # noqa: F401

from meeting.cloud.aws import cli
from meeting.cloud.aws.cli import build_parser, main


def run_cli(aws, state, *argv):
    return main(list(argv), aws=aws, state=state)


# ---------------------------------------------------------------------------
# --yes が無ければ作らない・起動しない・消さない
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("argv", [
    ["setup"],
    ["launch", "--type", "g4dn.xlarge", "--hours", "3"],
    ["launch", "--type", "g4dn.xlarge", "--spot", "--hours", "1"],
])
def test_create_commands_refuse_without_yes_and_change_nothing(aws, state, argv, capsys):
    if argv[0] == "launch":
        do_setup(aws, state)
        aws.calls.clear()
    rc = run_cli(aws, state, *argv)
    out = capsys.readouterr().out
    assert rc == 2 and "--yes" in out
    assert aws.mutating() == []
    assert not [c for c in aws.calls if c[1] in ("RunInstances", "CreateBucket", "CreateRole", "CreateSecurityGroup", "CreateInstanceProfile")]


def test_run_refuses_without_yes(aws, state, capsys, tmp_path):
    do_setup(aws, state)
    do_launch(aws, state)
    aws.calls.clear()
    rc = run_cli(aws, state, "run", "--exp", "exp01", "--engines", "voxcpm2,qwen3tts,chatterbox", "--public-test", "--dry-run-first",
                 "--out-dir", str(tmp_path / "o"))
    assert rc == 2 and "--yes" in capsys.readouterr().out
    assert aws.mutating() == [] and aws.ops("ssm") == []


def test_teardown_refuses_without_yes_when_there_is_something_to_stop(aws, state, capsys):
    do_setup(aws, state)
    do_launch(aws, state)
    aws.calls.clear()
    rc = run_cli(aws, state, "teardown")
    assert rc == 2 and "TerminateInstances" not in [c[1] for c in aws.calls]
    rc = run_cli(aws, state, "teardown", "--delete-bucket", "--confirm-bucket", "wrong-name", "--yes")
    assert rc == 2 and "--confirm-bucket" in capsys.readouterr().out
    assert "DeleteBucket" not in [c[1] for c in aws.calls]


def test_read_only_commands_work_without_yes(aws, state, capsys):
    assert run_cli(aws, state, "preflight") == 0
    assert run_cli(aws, state, "status") == 0
    assert aws.mutating() == []


def test_yes_short_flag_is_accepted(aws, state):
    assert run_cli(aws, state, "setup", "-y") == 0


# ---------------------------------------------------------------------------
# 鍵
# ---------------------------------------------------------------------------
def test_missing_keys_exit_code_2_with_instructions(monkeypatch, capsys):
    for k in ("EC2_ACCESS_KEY_ID", "EC2_SECRET_ACCESS_KEY", "EC2_REGION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "PLACEHOLDER")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "PLACEHOLDER")
    rc = main(["preflight"])
    out = capsys.readouterr().out
    assert rc == 2 and "EC2_ACCESS_KEY_ID" in out and "PLACEHOLDER" not in out


def test_keys_are_read_from_ec2_variables_and_region_option_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("EC2_ACCESS_KEY_ID", "AKIAFROMENVFROMENV12")
    monkeypatch.setenv("EC2_SECRET_ACCESS_KEY", "secretFromEnvSecretFromEnv")
    monkeypatch.setenv("EC2_REGION", "us-west-2")
    monkeypatch.setenv("EC2RUN_STATE_FILE", str(tmp_path / "s.json"))
    seen = {}

    class FakeAws:
        def __init__(self, settings):
            seen["settings"] = settings
            self.settings = settings

    monkeypatch.setattr(cli, "Aws", FakeAws)
    monkeypatch.setattr(cli, "dispatch", lambda args, aws, state: 0)
    assert main(["preflight"]) == 0
    assert seen["settings"].region == "us-west-2" and seen["settings"].access_key_id == "AKIAFROMENVFROMENV12"
    assert main(["--region", "eu-west-1", "preflight"]) == 0
    assert seen["settings"].region == "eu-west-1"


def test_secrets_in_unexpected_errors_are_redacted(aws, state, monkeypatch, capsys):
    secret = aws.settings.secret_access_key

    def boom(args, a, s):
        raise RuntimeError(f"failed with {secret} and {aws.settings.access_key_id}")

    monkeypatch.setattr(cli, "dispatch", boom)
    rc = run_cli(aws, state, "status")
    out = capsys.readouterr().out
    assert rc == 1 and secret not in out and aws.settings.access_key_id not in out and "***" in out


def test_aws_errors_are_explained_not_dumped(aws, state, monkeypatch, capsys):
    from test_cloud_aws_helpers import client_error

    def boom(args, a, s):
        raise client_error("AccessDenied", "User: arn:aws:iam::123456789012:user/x is not authorized", "ListBuckets")

    monkeypatch.setattr(cli, "dispatch", boom)
    rc = run_cli(aws, state, "status")
    out = capsys.readouterr().out
    assert rc == 1 and "権限が足りません" in out and "Traceback" not in out


def test_debug_env_reraises(aws, state, monkeypatch):
    monkeypatch.setenv("EC2RUN_DEBUG", "1")
    monkeypatch.setattr(cli, "dispatch", lambda a, b, c: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(ValueError):
        run_cli(aws, state, "status")


# ---------------------------------------------------------------------------
# 引数の既定値
# ---------------------------------------------------------------------------
def test_launch_defaults():
    a = build_parser().parse_args(["launch"])
    assert (a.type, a.hours, a.os_version, a.volume_gb, a.spot, a.yes) == ("g4dn.xlarge", 3.0, "22.04", 100, False, False)


def test_run_defaults_match_the_notebook_first_run():
    a = build_parser().parse_args(["run"])
    assert a.exp == "exp01" and a.engines == "voxcpm2,qwen3tts,chatterbox" and a.mode == "quick" and a.code == "git"
    assert not (a.dry_run_first or a.public_test or a.yes or a.attach or a.resume or a.fresh or a.auto_terminate)
    a = build_parser().parse_args(["run", "--speaker", "A=x", "--speaker", "B=y", "--dry-run-first", "--mode", "full"])
    assert a.speaker == ["A=x", "B=y"] and a.dry_run_first and a.mode == "full"


def test_teardown_flags():
    a = build_parser().parse_args(["teardown", "--all", "--delete-bucket", "--confirm-bucket", "b", "--delete-setup", "--yes"])
    assert a.all and a.delete_bucket and a.confirm_bucket == "b" and a.delete_setup and a.yes


def test_all_seven_subcommands_exist():
    sub = [a for a in build_parser()._actions if a.dest == "command"][0].choices
    assert set(sub) == {"preflight", "setup", "launch", "run", "fetch", "status", "teardown"}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def clean_env():
    env = dict(os.environ)
    for k in ("EC2_ACCESS_KEY_ID", "EC2_SECRET_ACCESS_KEY", "EC2_REGION"):
        env.pop(k, None)
    return env


def test_python_dash_m_entry_point_works_from_the_repo_root():
    p = subprocess.run([sys.executable, "-m", "meeting.cloud.aws", "--help"], cwd=REPO_ROOT, capture_output=True, text=True,
                       env=clean_env(), timeout=60)
    assert p.returncode == 0 and "preflight" in p.stdout and "teardown" in p.stdout


def test_script_entry_point_works_from_any_folder(tmp_path):
    script = REPO_ROOT / "meeting" / "cloud" / "aws" / "ec2run.py"
    p = subprocess.run([sys.executable, str(script), "launch", "--help"], cwd=tmp_path, capture_output=True, text=True,
                       env=clean_env(), timeout=60)
    assert p.returncode == 0 and "--hours" in p.stdout and "--spot" in p.stdout
    p = subprocess.run([sys.executable, str(script), "preflight"], cwd=tmp_path, capture_output=True, text=True,
                       env=clean_env(), timeout=60)
    assert p.returncode == 2 and "EC2_ACCESS_KEY_ID" in p.stdout


def test_importing_the_package_does_not_import_boto3():
    code = ("import sys; sys.path.insert(0, %r); import meeting.cloud.aws.cli, meeting.cloud.aws.launch, meeting.cloud.aws.runner, "
            "meeting.cloud.aws.teardown, meeting.cloud.aws.preflight, meeting.cloud.aws.provision; "
            "print('boto3' in sys.modules, 'botocore' in sys.modules)") % str(REPO_ROOT)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and p.stdout.strip() == "False False", p.stderr
