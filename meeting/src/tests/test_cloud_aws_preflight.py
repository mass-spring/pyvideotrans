"""preflight: 読み取りだけ。アカウント ID の伏せ字、枠（クォータ）の判定と案内、提供状況、起動中の一覧、料金表。"""
from __future__ import annotations

import pytest

from test_cloud_aws_helpers import Args, FakeQuotas, REGION, aws, client_error, mocked, state  # noqa: F401

from meeting.cloud.aws import preflight
from meeting.cloud.aws.common import QUOTA_ONDEMAND_G, QUOTA_SPOT_G, base_tags
from meeting.cloud.aws.preflight import fetch_quota, quota_guidance, run_preflight

MOTO_ACCOUNT = "123456789012"


def run(aws, capsys, **kw):
    args = Args(type=kw.pop("type", "g4dn.xlarge"), spot=kw.pop("spot", False))
    rc = run_preflight(aws, args)
    return rc, capsys.readouterr().out


def make_instance(aws, *, tagged=True, itype="t3.micro", extra_tags=None):
    ec2 = aws.client("ec2")
    ami = aws.ami_id
    tags = base_tags(extra_tags) if tagged else [{"Key": "Project", "Value": "meeting-translator"}]   # Project だけ（ManagedBy 無し）
    r = ec2.run_instances(ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
                          TagSpecifications=[{"ResourceType": "instance", "Tags": tags}])
    return r["Instances"][0]["InstanceId"]


def test_preflight_ok_masks_account_and_shows_region_quota_price(aws, capsys):
    rc, out = run(aws, capsys)
    assert rc == 0
    assert "********9012" in out and MOTO_ACCOUNT not in out          # アカウント ID は下4桁だけ
    assert REGION in out
    assert "L-DB2E81BA" in out and "L-3819A6DF" in out and "Running On-Demand G and VT instances" in out
    assert "8 vCPU" in out
    assert "g4dn.xlarge" in out and "g5.xlarge" in out and "0.710" in out and "1.459" in out
    assert "approximate, check AWS pricing" in out
    assert "ap-northeast-1a" in out                                  # 提供している AZ
    assert make_settings_secret() not in out


def make_settings_secret():
    from test_cloud_aws_helpers import make_settings

    return make_settings().secret_access_key


def test_preflight_creates_nothing(aws, capsys):
    run(aws, capsys)
    assert aws.mutating() == []
    assert set(aws.ops("sts")) == {"GetCallerIdentity"}


def test_quota_zero_prints_japanese_increase_guidance_and_fails(aws, capsys):
    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {QUOTA_ONDEMAND_G: 0.0, QUOTA_SPOT_G: 0.0})
    rc, out = run(aws, capsys)
    assert rc == 1
    assert "0 vCPU" in out and "不足" in out
    for needle in ("Service Quotas", "Running On-Demand G and VT instances", "L-DB2E81BA", "引き上げ", "ap-northeast-1", "4"):
        assert needle in out
    assert "https://ap-northeast-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA" in out
    assert "request-service-quota-increase" in out


def test_spot_flag_judges_by_the_spot_quota(aws, capsys):
    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {QUOTA_ONDEMAND_G: 8.0, QUOTA_SPOT_G: 0.0})
    rc, out = run(aws, capsys, spot=True)
    assert rc == 1 and "All G and VT Spot Instance Requests" in out and "L-3819A6DF" in out
    rc, out = run(aws, capsys, spot=False)
    assert rc == 0


def test_quota_smaller_than_needed_but_nonzero_is_still_insufficient(aws, capsys):
    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {QUOTA_ONDEMAND_G: 2.0, QUOTA_SPOT_G: 2.0})
    rc, out = run(aws, capsys)
    assert rc == 1 and "2 vCPU" in out


def test_default_value_is_used_when_no_applied_value_exists(aws, capsys):
    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {QUOTA_ONDEMAND_G: 0.0, QUOTA_SPOT_G: 0.0}, source="default")
    q = fetch_quota(aws, QUOTA_ONDEMAND_G)
    assert q.value == 0.0 and q.source == "既定値" and q.error is None
    ops = aws.ops("servicequotas")
    assert "GetServiceQuota" in ops and "GetAWSDefaultServiceQuota" in ops


def test_pending_increase_request_is_mentioned_instead_of_telling_to_apply_again(aws, capsys):
    pending = {QUOTA_ONDEMAND_G: [{"Status": "PENDING", "DesiredValue": 4.0}]}
    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {QUOTA_ONDEMAND_G: 0.0, QUOTA_SPOT_G: 0.0}, pending=pending)
    rc, out = run(aws, capsys)
    assert rc == 1 and "申請中" in out and "4 vCPU" in out
    assert "リクエスト」→ 新しい値" not in out


def test_quota_lookup_denied_is_reported_but_does_not_block(aws, capsys):
    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {})
    rc, out = run(aws, capsys)
    assert rc == 0 and "取得できませんでした" in out


def test_instance_type_not_offered_blocks(aws, capsys):
    rc, out = run(aws, capsys, type="g99.xlarge")
    assert rc == 1 and "提供していません" in out


def test_only_instances_tagged_by_this_tool_are_listed(aws, capsys):
    ours = make_instance(aws)
    foreign = make_instance(aws, tagged=False)
    rc, out = run(aws, capsys)
    assert ours in out and foreign not in out
    assert "teardown" in out


def test_invalid_keys_are_explained_and_nothing_else_is_called(aws, capsys):
    class BadSts:
        def get_caller_identity(self):
            raise client_error("InvalidClientTokenId", "The security token included in the request is invalid.", "GetCallerIdentity")

    aws._clients["sts"] = BadSts()
    rc, out = run(aws, capsys)
    assert rc == 1 and "鍵が無効" in out and "EC2_ACCESS_KEY_ID" in out
    assert aws.settings.secret_access_key not in out and aws.settings.access_key_id not in out
    assert aws.ops("ec2") == [] and aws.ops("s3") == []               # 鍵が使えないなら他を呼ばない
    assert "概算" in out                                               # 料金表は鍵なしでも見られる


def test_setup_status_is_shown_for_the_next_step(aws, capsys):
    rc, out = run(aws, capsys)
    assert "[新規]" in out and "meeting-translator-ec2-role" in out


def test_guidance_function_mentions_4_vcpu_for_g4dn_xlarge():
    q = preflight.QuotaResult(QUOTA_ONDEMAND_G, "Running On-Demand G and VT instances", 0.0, "既定値")
    text = "\n".join(quota_guidance(q, 4, "ap-northeast-1"))
    assert "4（g4dn.xlarge 1台分 = 4 vCPU）" in text


def test_check_ready_for_launch_stops_early_when_quota_is_short(aws):
    from meeting.cloud.aws.common import ToolError

    aws._clients["service-quotas"] = FakeQuotas(aws.calls, {QUOTA_ONDEMAND_G: 0.0, QUOTA_SPOT_G: 0.0})
    with pytest.raises(ToolError) as e:
        preflight.check_ready_for_launch(aws, "g4dn.xlarge", spot=False)
    assert "引き上げ" in str(e.value)
