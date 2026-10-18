"""teardown / status: タグが付いた自分のインスタンスだけを終了する。--yes、--all、--delete-bucket の確認、残りものの報告。"""
from __future__ import annotations

import datetime as dt

import pytest

from test_cloud_aws_helpers import Args, REGION, aws, client_error, do_launch, do_setup, mocked, state  # noqa: F401

from meeting.cloud.aws import common, instances
from meeting.cloud.aws.common import NeedConfirmation, ToolError, bucket_name
from meeting.cloud.aws.teardown import run_status, run_teardown

BUCKET = bucket_name("123456789012", REGION)


def targs(**kw):
    base = dict(all=False, delete_bucket=False, confirm_bucket=None, delete_setup=False, yes=True)
    base.update(kw)
    return Args(**base)


def foreign_instance(aws, tags):
    return aws.client("ec2").run_instances(ImageId=aws.ami_id, InstanceType="t3.micro", MinCount=1, MaxCount=1,
                                           TagSpecifications=[{"ResourceType": "instance", "Tags": tags}] if tags else [])["Instances"][0]["InstanceId"]


def state_of(aws, iid):
    return aws.client("ec2").describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]


@pytest.fixture
def ready(aws, state):
    do_setup(aws, state)
    return aws


def run_td(aws, state, **kw):
    return run_teardown(aws, state, targs(**kw), sleep=lambda s: None)


def test_without_yes_lists_the_instance_and_terminates_nothing(ready, state, capsys):
    iid = do_launch(ready, state)
    ready.calls.clear()
    with pytest.raises(NeedConfirmation):
        run_td(ready, state, yes=False)
    assert iid in capsys.readouterr().out
    assert state_of(ready, iid) == "running"
    assert "TerminateInstances" not in [c[1] for c in ready.calls]


def test_terminates_our_instance_and_verifies_terminated(ready, state, capsys):
    iid = do_launch(ready, state)
    assert run_td(ready, state) == 0
    out = capsys.readouterr().out
    assert state_of(ready, iid) == "terminated"
    assert f"{iid}: terminated" in out and "課金は止まりました" in out
    assert common.State(state.path).instances()[iid]["state"] == "terminated"


def test_only_tagged_instances_are_touched(ready, state):
    mine = do_launch(ready, state)
    untagged = foreign_instance(ready, None)
    project_only = foreign_instance(ready, [{"Key": "Project", "Value": "meeting-translator"}])      # ManagedBy が無い
    other_project = foreign_instance(ready, common.base_tags() and [{"Key": "Project", "Value": "something-else"}, {"Key": "ManagedBy", "Value": "ec2run"}])
    run_td(ready, state, all=True)
    assert state_of(ready, mine) == "terminated"
    for other in (untagged, project_only, other_project):
        assert state_of(ready, other) == "running", other


def test_instance_tagged_by_us_but_missing_from_state_needs_all(ready, state, capsys):
    iid = do_launch(ready, state)
    lost = common.State(state.path.with_name("lost.json"))                 # 状態ファイルが消えた状況
    assert run_td(ready, lost) == 0
    out = capsys.readouterr().out
    assert state_of(ready, iid) == "running" and "--all" in out
    assert run_td(ready, lost, all=True) == 0
    assert state_of(ready, iid) == "terminated"


def test_tags_are_rechecked_right_before_terminating(ready, state):
    iid = do_launch(ready, state)
    ready.client("ec2").delete_tags(Resources=[iid], Tags=[{"Key": "ManagedBy"}])        # 誰かがタグを外した
    run_td(ready, state)
    assert state_of(ready, iid) == "running"


def test_nothing_to_do_needs_no_yes(ready, state, capsys):
    assert run_td(ready, state, yes=False) == 0
    assert "終了させるインスタンスはありません" in capsys.readouterr().out


def test_terminate_error_is_explained(ready, state):
    do_launch(ready, state)

    def denied(**kw):
        raise client_error("UnauthorizedOperation", "denied", "TerminateInstances")

    ready.client("ec2").terminate_instances = denied
    with pytest.raises(ToolError) as e:
        run_td(ready, state)
    assert "権限が足りません" in str(e.value)


def test_leftovers_report_mentions_volumes_bucket_and_cost(ready, state, capsys):
    do_launch(ready, state)
    ready.client("s3").put_object(Bucket=BUCKET, Key="exp/exp01/result/voicelab_result.xlsx", Body=b"x" * 2_000_000)
    ec2 = ready.client("ec2")
    vol = ec2.create_volume(AvailabilityZone="ap-northeast-1a", Size=100, VolumeType="gp3",
                            TagSpecifications=[{"ResourceType": "volume", "Tags": common.base_tags()}])["VolumeId"]
    run_td(ready, state)
    out = capsys.readouterr().out
    assert "[残っているもの]" in out
    assert vol in out and "100 GB" in out and "$9.60/月" in out                     # 100GB × 0.096
    assert BUCKET in out and "1 オブジェクト" in out and "2.0 MB" in out and "14 日で自動削除" in out
    assert "無料です" in out


def test_delete_bucket_needs_yes_and_the_bucket_name(ready, state):
    do_launch(ready, state)
    run_td(ready, state)                                                  # まず終了
    ready.client("s3").put_object(Bucket=BUCKET, Key="exp/a", Body=b"1")
    with pytest.raises(ToolError) as e:
        run_td(ready, state, delete_bucket=True)
    assert "--confirm-bucket" in str(e.value) and BUCKET in str(e.value)
    with pytest.raises(ToolError):
        run_td(ready, state, delete_bucket=True, confirm_bucket="some-other-bucket")
    with pytest.raises(NeedConfirmation):
        run_td(ready, state, delete_bucket=True, confirm_bucket=BUCKET, yes=False)
    assert ready.client("s3").head_bucket(Bucket=BUCKET)                   # まだある


def test_delete_bucket_removes_objects_and_bucket(ready, state, capsys):
    ready.client("s3").put_object(Bucket=BUCKET, Key="exp/a", Body=b"1")
    ready.client("s3").put_object(Bucket=BUCKET, Key="exp/b/c", Body=b"2")
    assert run_td(ready, state, delete_bucket=True, confirm_bucket=BUCKET) == 0
    names = [b["Name"] for b in ready.client("s3").list_buckets()["Buckets"]]
    assert BUCKET not in names
    assert "オブジェクト 2 件" in capsys.readouterr().out
    assert common.State(state.path).get("bucket") is None


def test_delete_bucket_that_is_already_gone_is_not_an_error(ready, state):
    run_td(ready, state, delete_bucket=True, confirm_bucket=BUCKET)
    assert run_td(ready, state, delete_bucket=True, confirm_bucket=BUCKET) == 0


def test_delete_setup_removes_role_profile_and_group(ready, state):
    assert run_td(ready, state, delete_setup=True) == 0
    from meeting.cloud.aws.provision import discover

    d = discover(ready)
    assert d.role_exists is False and d.profile_exists is False and d.sg_id is None
    assert common.State(state.path).get("security_group_id") is None


def test_does_not_wait_forever_if_aws_never_reports_terminated(ready, state, capsys):
    do_launch(ready, state)
    ec2 = ready.client("ec2")
    real = ec2.describe_instances

    def stuck(**kw):
        resp = real(**kw)
        if kw.get("InstanceIds") and not kw.get("Filters"):
            for r in resp["Reservations"]:
                for i in r["Instances"]:
                    i["State"]["Name"] = "shutting-down"
        return resp

    ec2.describe_instances = stuck
    clock = iter(range(0, 100000, 100))
    rc = run_teardown(ready, state, targs(), sleep=lambda s: None, monotonic=lambda: next(clock))
    assert rc == 0 and "terminated を確認できない" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def test_status_shows_elapsed_time_cost_and_time_left(ready, state, capsys, monkeypatch):
    iid = do_launch(ready, state)
    inst = ready.client("ec2").describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    later = inst["LaunchTime"] + dt.timedelta(hours=1, minutes=30)
    monkeypatch.setattr(instances, "utcnow", lambda: later)
    assert run_status(ready, state, Args()) == 0
    out = capsys.readouterr().out
    assert iid in out and "経過 1:30:00" in out and "$1.09" in out and "自動停止まで 1:29:" in out or "自動停止まで 1:30:" in out
    assert "teardown --yes" in out
    assert BUCKET in out


def test_status_warns_when_past_the_scheduled_stop(ready, state, capsys, monkeypatch):
    iid = do_launch(ready, state)
    inst = ready.client("ec2").describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    later = inst["LaunchTime"] + dt.timedelta(hours=4)
    monkeypatch.setattr(instances, "utcnow", lambda: later)
    monkeypatch.setattr("meeting.cloud.aws.teardown.utcnow", lambda: later)
    run_status(ready, state, Args())
    out = capsys.readouterr().out
    assert "過ぎています" in out and "予定時刻を過ぎても動いている" in out


def test_status_ignores_foreign_instances_and_is_read_only(ready, state, capsys):
    foreign = foreign_instance(ready, None)
    ready.calls.clear()
    run_status(ready, state, Args())
    out = capsys.readouterr().out
    assert foreign not in out and "動いているものはありません" in out
    assert ready.mutating() == []


def test_status_before_setup_does_not_crash(aws, state, capsys):
    assert run_status(aws, state, Args()) == 0
    assert "setup 前" in capsys.readouterr().out
