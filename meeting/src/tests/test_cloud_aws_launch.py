"""launch: RunInstances の引数（IMDSv2・終了時の動作・タグ・user-data の自動停止）、--yes、見積、再試行、SSM 待ち。"""
from __future__ import annotations

import base64
import datetime as dt

import pytest

from test_cloud_aws_helpers import Args, FakeQuotas, REGION, aws, client_error, do_launch, do_setup, launch_args, mocked, state  # noqa: F401

from meeting.cloud.aws import common, launch
from meeting.cloud.aws.common import NeedConfirmation, ToolError
from meeting.cloud.aws.launch import AmiInfo, build_run_params, build_user_data, run_launch

NOW = dt.datetime(2026, 10, 18, 12, 0, 0, tzinfo=dt.timezone.utc)


def capture_run_params(aws):
    """RunInstances に渡された引数（boto3 が加工する前）を記録する。"""
    seen: list[dict] = []
    aws.client("ec2").meta.events.register("provide-client-params.ec2.RunInstances", lambda params, **kw: seen.append(dict(params)))
    return seen


def tags_of(spec):
    return common.tags_to_dict(spec["Tags"])


@pytest.fixture
def ready(aws, state):
    do_setup(aws, state)
    aws.calls.clear()
    return aws


# ---------------------------------------------------------------------------
# 純粋な部品
# ---------------------------------------------------------------------------
def test_ami_parameter_paths_are_the_documented_public_ssm_names():
    assert launch.ami_parameter("22.04") == "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
    assert launch.ami_parameter("24.04") == "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
    with pytest.raises(ToolError):
        launch.ami_parameter("20.04")


@pytest.mark.parametrize("hours,minutes", [(3, 180), (1.5, 90), (0.5, 30), (24, 1440)])
def test_user_data_schedules_shutdown_in_minutes(hours, minutes):
    ud = build_user_data(hours)
    assert ud.startswith("#!/bin/bash")
    assert f"shutdown -h +{minutes} " in ud
    assert "terminate" in ud          # 説明コメント


def test_build_run_params_on_demand():
    ami = AmiInfo("ami-0abc", "DLAMI", "/dev/sda1", launch.ami_parameter("22.04"))
    p = build_run_params(ami=ami, instance_type="g4dn.xlarge", spot=False, hours=3, subnet_id="subnet-1", sg_id="sg-1",
                         profile_name="prof", volume_gb=100, now=NOW, rate=0.71)
    assert p["ImageId"] == "ami-0abc" and p["InstanceType"] == "g4dn.xlarge" and p["MinCount"] == p["MaxCount"] == 1
    assert p["MetadataOptions"]["HttpTokens"] == "required"                      # IMDSv2 必須
    assert p["InstanceInitiatedShutdownBehavior"] == "terminate"                 # 停止 = 終了
    assert "shutdown -h +180" in p["UserData"]
    assert p["IamInstanceProfile"] == {"Name": "prof"}
    ebs = p["BlockDeviceMappings"][0]
    assert ebs["DeviceName"] == "/dev/sda1" and ebs["Ebs"] == {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True}
    ni = p["NetworkInterfaces"][0]
    assert ni["Groups"] == ["sg-1"] and ni["SubnetId"] == "subnet-1" and ni["AssociatePublicIpAddress"] is True
    assert "KeyName" not in p and "SecurityGroupIds" not in p                    # SSH 用の鍵は付けない
    assert "InstanceMarketOptions" not in p
    by_type = {s["ResourceType"]: tags_of(s) for s in p["TagSpecifications"]}
    assert set(by_type) == {"instance", "volume", "network-interface"}
    for t in by_type.values():
        assert t["Project"] == "meeting-translator" and t["ManagedBy"] == "ec2run"
    inst = by_type["instance"]
    assert inst["Hours"] == "3" and inst["ExpiresAt"] == "2026-10-18T15:00:00Z" and inst["Purchase"] == "on-demand"
    assert inst["Name"] == "meeting-translator-g4dn.xlarge" and inst["HourlyRate"] == "0.71"


def test_build_run_params_spot_is_one_time_and_terminates():
    ami = AmiInfo("ami-0abc", "DLAMI", "/dev/sda1", "p")
    p = build_run_params(ami=ami, instance_type="g4dn.xlarge", spot=True, hours=2, subnet_id="s", sg_id="g", profile_name="x",
                         volume_gb=100, now=NOW, rate=0.71)
    assert p["InstanceMarketOptions"] == {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time",
                                                                              "InstanceInterruptionBehavior": "terminate"}}
    assert {s["ResourceType"] for s in p["TagSpecifications"]} == {"instance", "volume", "network-interface", "spot-instances-request"}
    assert "shutdown -h +120" in p["UserData"] and p["InstanceInitiatedShutdownBehavior"] == "terminate"


@pytest.mark.parametrize("hours", [0, -1, 24.5])
def test_hours_out_of_range_is_rejected(hours):
    with pytest.raises(ToolError):
        launch.validate_launch_args(launch_args(hours=hours))


def test_type_not_in_price_table_needs_an_explicit_rate():
    with pytest.raises(ToolError) as e:
        launch.validate_launch_args(launch_args(type="g6e.xlarge"))
    assert "--hourly-rate" in str(e.value)
    assert launch.validate_launch_args(launch_args(type="g6e.xlarge", hourly_rate=2.5))[2] == 2.5


# ---------------------------------------------------------------------------
# --yes と見積
# ---------------------------------------------------------------------------
def test_refuses_without_yes_but_prints_the_estimated_max_cost(ready, state, capsys):
    with pytest.raises(NeedConfirmation):
        run_launch(ready, state, launch_args(yes=False), sleep=lambda s: None)
    out = capsys.readouterr().out
    assert "費用の上限の見積" in out and "0.71$/時間" in out and "3時間" in out
    assert "approximate, check AWS pricing" in out
    assert "RunInstances" not in [c[1] for c in ready.calls]
    assert ready.mutating() == []


def test_estimate_is_hours_times_rate_plus_small_extras(ready, state, capsys):
    with pytest.raises(NeedConfirmation):
        run_launch(ready, state, launch_args(yes=False, hours=10), sleep=lambda s: None)
    out = capsys.readouterr().out
    est = common.estimate_cost(0.71, 10)
    assert f"${est.total:,.2f}" in out and f"${est.compute:,.2f}" in out and 7.1 < est.total < 7.5


# ---------------------------------------------------------------------------
# 本番（moto）
# ---------------------------------------------------------------------------
def test_launch_passes_the_safety_parameters_to_aws(ready, state):
    seen = capture_run_params(ready)
    iid = do_launch(ready, state)
    assert len(seen) == 1
    p = seen[0]
    assert p["MetadataOptions"]["HttpTokens"] == "required"
    assert p["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert "shutdown -h +180" in p["UserData"]
    assert p["IamInstanceProfile"]["Name"] == common.PROFILE_NAME
    assert p["NetworkInterfaces"][0]["Groups"] == [state.get("security_group_id")]
    assert p["ImageId"] == ready.ami_id
    # moto 側で見ても、その設定になっている
    ec2 = ready.client("ec2")
    assert ec2.describe_instance_attribute(InstanceId=iid, Attribute="instanceInitiatedShutdownBehavior")["InstanceInitiatedShutdownBehavior"]["Value"] == "terminate"
    ud = base64.b64decode(ec2.describe_instance_attribute(InstanceId=iid, Attribute="userData")["UserData"]["Value"]).decode()
    assert "shutdown -h +180" in ud
    inst = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    tags = common.tags_to_dict(inst["Tags"])
    assert common.is_ours(tags) and tags["Hours"] == "3"
    assert inst["InstanceType"] == "g4dn.xlarge"


def test_launch_records_the_instance_in_the_state_file(ready, state):
    iid = do_launch(ready, state)
    st = common.State(state.path)
    rec = st.instances()[iid]
    assert rec["type"] == "g4dn.xlarge" and rec["hours"] == 3.0 and rec["spot"] is False and rec["state"] == "ready"
    assert rec["auto_shutdown_verified"] is True and st.get("last_instance_id") == iid


def test_launch_waits_for_ssm_and_checks_the_shutdown_is_scheduled(ready, state, capsys):
    ssm = ready.client("ssm")
    ssm.online_after_polls = 2
    do_launch(ready, state)
    assert ssm.polls >= 3
    text = capsys.readouterr().out
    assert "準備できました" in text and "確認できました" in text and "Tesla T4" in text


def test_if_the_shutdown_is_not_scheduled_the_tool_schedules_it_again(ready, state, capsys):
    ssm = ready.client("ssm")

    def responder(cmds):
        text = "\n".join(cmds)
        if "NOT_SCHEDULED" in text:
            return "Success", "NOT_SCHEDULED\n", ""
        return "Success", "SCHEDULED\n", ""

    ssm.responder = responder
    do_launch(ready, state)
    resched = [s for s in ssm.sent if any("shutdown -h +" in c for c in s["Parameters"]["commands"]) and not any("NOT_SCHEDULED" in c for c in s["Parameters"]["commands"])]
    assert resched, "再予約のコマンドが送られていない"
    minutes = int([c for c in resched[0]["Parameters"]["commands"] if c.startswith("shutdown")][0].split("+")[1].split()[0])
    assert 170 <= minutes <= 180
    assert "予約し直します" in capsys.readouterr().out


def test_ssm_never_online_reports_failure_but_does_not_terminate(ready, state, capsys):
    ssm = ready.client("ssm")
    ssm.online_ids = set()
    from meeting.cloud.aws.launch import run_launch as rl

    rc = rl(ready, state, launch_args(ssm_timeout=0), sleep=lambda s: None)
    assert rc == 1
    assert "TerminateInstances" not in [c[1] for c in ready.calls]       # 勝手に消さない（自動停止が効く）
    assert "Online になりませんでした" in capsys.readouterr().out
    assert state.instances()[state.get("last_instance_id")]["state"] == "running-no-ssm"


def test_no_wait_returns_right_after_the_request(ready, state):
    from meeting.cloud.aws.launch import run_launch as rl

    assert rl(ready, state, launch_args(no_wait=True), sleep=lambda s: None) == 0
    assert ready.ops("ssm") == ["GetParameter"]


def test_second_launch_is_refused_while_one_is_alive(ready, state):
    do_launch(ready, state)
    with pytest.raises(ToolError) as e:
        run_launch(ready, state, launch_args(), sleep=lambda s: None)
    assert "すでに 1 台" in str(e.value) and "--allow-multiple" in str(e.value)
    assert run_launch(ready, state, launch_args(allow_multiple=True), sleep=lambda s: None) == 0


def test_launch_before_setup_says_to_run_setup(aws, state):
    with pytest.raises(ToolError) as e:
        run_launch(aws, state, launch_args(), sleep=lambda s: None)
    assert "setup --yes" in str(e.value)
    assert "RunInstances" not in [c[1] for c in aws.calls]


def test_quota_zero_stops_before_run_instances(ready, state):
    ready._clients["service-quotas"] = FakeQuotas(ready.calls, {common.QUOTA_ONDEMAND_G: 0.0, common.QUOTA_SPOT_G: 0.0})
    with pytest.raises(ToolError) as e:
        run_launch(ready, state, launch_args(), sleep=lambda s: None)
    assert "Service Quotas" in str(e.value) and "引き上げ" in str(e.value)
    assert "RunInstances" not in [c[1] for c in ready.calls]


def test_spot_launch_uses_the_spot_quota_and_market_options(ready, state):
    seen = capture_run_params(ready)
    do_launch(ready, state, spot=True)
    assert seen[0]["InstanceMarketOptions"]["SpotOptions"]["InstanceInterruptionBehavior"] == "terminate"
    ready._clients["service-quotas"] = FakeQuotas(ready.calls, {common.QUOTA_ONDEMAND_G: 8.0, common.QUOTA_SPOT_G: 0.0})
    with pytest.raises(ToolError):
        run_launch(ready, state, launch_args(spot=True, allow_multiple=True), sleep=lambda s: None)


def test_new_instance_profile_propagation_is_retried(ready, state, capsys):
    ec2 = ready.client("ec2")
    real = ec2.run_instances
    attempts = []

    def flaky(**kw):
        attempts.append(1)
        if len(attempts) < 3:
            raise client_error("InvalidParameterValue", "Value (meeting-translator-ec2-profile) for parameter iamInstanceProfile.name is invalid. Invalid IAM Instance Profile name", "RunInstances")
        return real(**kw)

    ec2.run_instances = flaky
    sleeps = []
    assert run_launch(ready, state, launch_args(), sleep=sleeps.append) == 0
    assert len(attempts) == 3 and sleeps.count(10) >= 2
    assert "反映待ち" in capsys.readouterr().out


def test_capacity_error_moves_on_to_the_next_az(ready, state, capsys):
    ec2 = ready.client("ec2")
    real = ec2.run_instances
    subnets_tried = []

    def capacity(**kw):
        subnets_tried.append(kw["NetworkInterfaces"][0]["SubnetId"])
        if len(subnets_tried) == 1:
            raise client_error("InsufficientInstanceCapacity", "no capacity", "RunInstances")
        return real(**kw)

    ec2.run_instances = capacity
    assert run_launch(ready, state, launch_args(), sleep=lambda s: None) == 0
    assert len(set(subnets_tried)) == 2
    assert "次の AZ" in capsys.readouterr().out


def test_unexpected_aws_error_is_explained_after_one_attempt(ready, state):
    ec2 = ready.client("ec2")
    n = []

    def denied(**kw):
        n.append(1)
        raise client_error("UnauthorizedOperation", "You are not authorized to perform this operation.", "RunInstances")

    ec2.run_instances = denied
    with pytest.raises(ToolError) as e:
        run_launch(ready, state, launch_args(), sleep=lambda s: None)
    assert len(n) == 1 and "権限が足りません" in str(e.value)


def test_missing_ami_parameter_explains(ready, state):
    ready.client("ssm").params.clear()
    with pytest.raises(ToolError) as e:
        run_launch(ready, state, launch_args(), sleep=lambda s: None)
    assert "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id" in str(e.value)


def test_ubuntu_2404_uses_the_other_parameter(ready, state):
    do_launch(ready, state, os_version="24.04")
    gets = [c for c in ready.calls if c == ("ssm", "GetParameter")]
    assert gets


def test_launch_never_prints_secrets(ready, state, capsys):
    do_launch(ready, state)
    out = capsys.readouterr().out
    assert ready.settings.secret_access_key not in out and ready.settings.access_key_id not in out
    assert "123456789012" not in out
