"""setup: S3 バケット・IAM ロール＋プロファイル・セキュリティグループを、冪等に作る。--yes が無ければ何も作らない。"""
from __future__ import annotations

import json

import pytest

from test_cloud_aws_helpers import Args, REGION, aws, do_setup, mocked, state  # noqa: F401

from meeting.cloud.aws import common, provision
from meeting.cloud.aws.common import NeedConfirmation, ToolError, bucket_name
from meeting.cloud.aws.provision import discover, instance_s3_policy, run_setup

MOTO_ACCOUNT = "123456789012"


def test_refuses_without_yes_prints_the_plan_and_creates_nothing(aws, state, capsys):
    with pytest.raises(NeedConfirmation) as e:
        run_setup(aws, state, Args(yes=False))
    out = capsys.readouterr().out
    assert "--yes" in str(e.value)
    assert "[新規]" in out and bucket_name(MOTO_ACCOUNT, REGION) in out and "インバウンドの許可は 0 件" in out and "14日" in out
    assert aws.mutating() == []
    assert state.get("bucket") is None
    # 本当に何も無い
    assert discover(aws).bucket_exists is False and discover(aws).role_exists is False


def test_creates_a_private_encrypted_expiring_bucket_tagged_as_ours(aws, state):
    do_setup(aws, state, keep_policy=True)
    name = bucket_name(MOTO_ACCOUNT, REGION)
    s3 = aws.client("s3")
    assert s3.get_bucket_location(Bucket=name)["LocationConstraint"] == REGION
    pab = s3.get_public_access_block(Bucket=name)["PublicAccessBlockConfiguration"]
    assert pab == {"BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True}
    enc = s3.get_bucket_encryption(Bucket=name)["ServerSideEncryptionConfiguration"]["Rules"][0]["ApplyServerSideEncryptionByDefault"]
    assert enc["SSEAlgorithm"] == "AES256"
    rules = s3.get_bucket_lifecycle_configuration(Bucket=name)["Rules"]
    assert rules[0]["Expiration"]["Days"] == 14 and rules[0]["Status"] == "Enabled"
    tags = common.tags_to_dict(s3.get_bucket_tagging(Bucket=name)["TagSet"])
    assert common.is_ours(tags)
    policy = json.loads(s3.get_bucket_policy(Bucket=name)["Policy"])
    st = policy["Statement"][0]
    assert st["Effect"] == "Deny" and st["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}


def test_role_trusts_ec2_has_ssm_core_and_s3_for_this_bucket_only(aws, state):
    do_setup(aws, state)
    iam = aws.client("iam")
    role = iam.get_role(RoleName=common.ROLE_NAME)["Role"]
    trust = role["AssumeRolePolicyDocument"]
    assert trust["Statement"][0]["Principal"] == {"Service": "ec2.amazonaws.com"}
    assert common.is_ours(common.tags_to_dict(role["Tags"]))
    attached = [p["PolicyArn"] for p in iam.list_attached_role_policies(RoleName=common.ROLE_NAME)["AttachedPolicies"]]
    assert attached == ["arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"]
    inline = iam.get_role_policy(RoleName=common.ROLE_NAME, PolicyName=common.INSTANCE_S3_POLICY_NAME)["PolicyDocument"]
    bucket = bucket_name(MOTO_ACCOUNT, REGION)
    resources = []
    for st in inline["Statement"]:
        r = st["Resource"]
        resources += r if isinstance(r, list) else [r]
    assert set(resources) == {f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"}
    prof = iam.get_instance_profile(InstanceProfileName=common.PROFILE_NAME)["InstanceProfile"]
    assert [r["RoleName"] for r in prof["Roles"]] == [common.ROLE_NAME]


def test_instance_s3_policy_has_no_wildcard_resource_and_no_delete():
    doc = instance_s3_policy("meeting-translator-abc-ap-northeast-1")
    actions = [a for st in doc["Statement"] for a in (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])]
    assert all(a.startswith("s3:") and "*" not in a for a in actions)
    assert not any("Delete" in a or "Put" in a and "Bucket" in a for a in actions)
    for st in doc["Statement"]:
        assert st["Resource"].startswith("arn:aws:s3:::meeting-translator-abc-ap-northeast-1") and st["Resource"] != "*"


def test_security_group_has_no_inbound_rules_and_is_tagged(aws, state):
    do_setup(aws, state)
    sg = aws.client("ec2").describe_security_groups(GroupIds=[state.get("security_group_id")])["SecurityGroups"][0]
    assert sg["GroupName"] == common.SG_NAME and sg["IpPermissions"] == []
    assert common.is_ours(common.tags_to_dict(sg["Tags"]))


def test_state_file_records_the_created_resources(aws, state):
    do_setup(aws, state)
    again = common.State(state.path)
    assert again.get("bucket") == bucket_name(MOTO_ACCOUNT, REGION)
    assert again.get("role_name") == common.ROLE_NAME and again.get("instance_profile") == common.PROFILE_NAME
    assert again.get("security_group_id", "").startswith("sg-") and again.get("vpc_id", "").startswith("vpc-")
    assert again.get("region") == REGION
    assert MOTO_ACCOUNT not in state.path.read_text(encoding="utf-8")        # アカウント ID 全体は保存しない
    assert again.get("account_last4") == "9012"


def test_running_setup_twice_creates_nothing_new(aws, state, capsys):
    do_setup(aws, state)
    sg1 = state.get("security_group_id")
    first_creates = [c for c in aws.calls if c[1].startswith("Create")]
    assert {c[1] for c in first_creates} >= {"CreateBucket", "CreateRole", "CreateInstanceProfile", "CreateSecurityGroup"}
    aws.calls.clear()
    capsys.readouterr()
    do_setup(aws, state)
    out = capsys.readouterr().out
    assert [c for c in aws.calls if c[1].startswith("Create")] == []
    assert state.get("security_group_id") == sg1
    assert out.count("すでにあります") >= 4
    assert len(aws.client("ec2").describe_security_groups(Filters=[{"Name": "group-name", "Values": [common.SG_NAME]}])["SecurityGroups"]) == 1


def test_second_run_shows_existing_resources_in_the_plan(aws, state, capsys):
    do_setup(aws, state)
    capsys.readouterr()
    with pytest.raises(NeedConfirmation):
        run_setup(aws, state, Args(yes=False))
    assert "[ある]" in capsys.readouterr().out


def test_a_foreign_role_with_the_same_name_is_not_overwritten(aws, state):
    aws.client("iam").create_role(RoleName=common.ROLE_NAME, AssumeRolePolicyDocument=json.dumps(provision.trust_policy()))
    with pytest.raises(ToolError) as e:
        run_setup(aws, state, Args(yes=True))
    assert "このツールが作ったものではありません" in str(e.value)


def test_inbound_rules_found_on_our_group_are_removed(aws, state):
    do_setup(aws, state)
    sg = state.get("security_group_id")
    aws.client("ec2").authorize_security_group_ingress(GroupId=sg, IpPermissions=[
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
    do_setup(aws, state)
    assert aws.client("ec2").describe_security_groups(GroupIds=[sg])["SecurityGroups"][0]["IpPermissions"] == []


def test_missing_default_vpc_gives_clear_guidance(aws, state):
    ec2 = aws.client("ec2")
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    for sn in ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]:
        ec2.delete_subnet(SubnetId=sn["SubnetId"])
    for sg in ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]:
        if sg["GroupName"] != "default":
            ec2.delete_security_group(GroupId=sg["GroupId"])
    for ig in ec2.describe_internet_gateways()["InternetGateways"]:
        ec2.detach_internet_gateway(InternetGatewayId=ig["InternetGatewayId"], VpcId=vpc)
        ec2.delete_internet_gateway(InternetGatewayId=ig["InternetGatewayId"])
    ec2.delete_vpc(VpcId=vpc)
    with pytest.raises(ToolError) as e:
        run_setup(aws, state, Args(yes=True))
    assert "デフォルト VPC" in str(e.value)
    assert [c for c in aws.calls if c[1] in ("CreateBucket", "CreateRole")] == []        # 作り始める前に止まる


def test_resolve_resources_finds_everything_even_if_the_state_file_is_lost(aws, state, tmp_path):
    do_setup(aws, state)
    lost = common.State(tmp_path / "other-state.json")
    res = provision.resolve_resources(aws, lost)
    assert res["bucket"] == bucket_name(MOTO_ACCOUNT, REGION) and res["security_group_id"] == state.get("security_group_id")


def test_resolve_resources_before_setup_tells_what_to_do(aws, state):
    with pytest.raises(ToolError) as e:
        provision.resolve_resources(aws, state)
    assert "setup --yes" in str(e.value)


def test_delete_setup_removes_role_profile_and_group(aws, state):
    do_setup(aws, state)
    done = provision.delete_setup_resources(aws)
    assert done
    d = discover(aws)
    assert d.role_exists is False and d.profile_exists is False and d.sg_id is None
