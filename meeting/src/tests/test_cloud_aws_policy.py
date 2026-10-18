"""iam_user_policy.json（ユーザーに付ける最小の権限）: 形式・サイズ・最小性・ツールが実際に呼ぶ操作を全部カバーしているか。

「ツールが呼ぶ操作」は、moto 上で全コマンド（preflight → setup → launch → run → status → fetch → teardown）を実際に流して、
すべての AWS 呼び出しを記録し、IAM のアクション名に直して、ポリシーで許可されているかを確かめる。
"""
from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

from test_cloud_aws_helpers import Args, REGION, RecordingAws, aws, do_launch, do_setup, mocked, state  # noqa: F401
from test_cloud_aws_run import BUCKET, fill_results, rargs

from meeting.cloud.aws import common
from meeting.cloud.aws.preflight import run_preflight
from meeting.cloud.aws.provision import instance_s3_policy
from meeting.cloud.aws.runner import run_fetch, run_run
from meeting.cloud.aws.teardown import run_status, run_teardown

POLICY_PATH = Path(common.AWS_DIR) / "iam_user_policy.json"

# S3 は「操作名」と「IAM のアクション名」が違うものがある
S3_OP_TO_ACTION = {
    "HeadObject": "GetObject", "HeadBucket": "ListBucket", "ListObjectsV2": "ListBucket", "ListObjects": "ListBucket",
    "DeleteObjects": "DeleteObject", "CreateMultipartUpload": "PutObject", "UploadPart": "PutObject",
    "CompleteMultipartUpload": "PutObject", "PutBucketEncryption": "PutEncryptionConfiguration",
    "GetBucketEncryption": "GetEncryptionConfiguration", "PutBucketLifecycleConfiguration": "PutLifecycleConfiguration",
    "PutPublicAccessBlock": "PutBucketPublicAccessBlock", "ListMultipartUploads": "ListBucketMultipartUploads",
}
IAM_PREFIX = {"service-quotas": "servicequotas"}
PERMISSION_ONLY_ACTIONS = {"iam:PassRole"}          # API の操作名ではなく、権限だけの名前（RunInstances が内部で使う）
NO_PERMISSION_NEEDED = {"sts:GetCallerIdentity"}    # この操作は IAM の許可がなくても呼べる（AWS の仕様）


def policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def statements():
    return policy()["Statement"]


def as_list(x):
    return x if isinstance(x, list) else [x]


def allowed_actions():
    return [a for st in statements() if st["Effect"] == "Allow" for a in as_list(st["Action"])]


def action_for(service: str, op: str) -> str:
    service = IAM_PREFIX.get(service, service)
    if service == "s3":
        op = S3_OP_TO_ACTION.get(op, op)
    return f"{service}:{op}"


def is_allowed(action: str) -> bool:
    return any(fnmatch.fnmatchcase(action.lower(), pat.lower()) for pat in allowed_actions())


def real_operation_names(service: str) -> set[str]:
    import botocore.session

    return set(botocore.session.get_session().get_service_model(service).operation_names)


def test_policy_is_valid_json_with_unique_sids():
    p = policy()
    assert p["Version"] == "2012-10-17"
    sids = [s["Sid"] for s in p["Statement"]]
    assert len(sids) == len(set(sids)) and all(re.fullmatch(r"[A-Za-z0-9]+", s) for s in sids)
    assert all(s["Effect"] == "Allow" for s in p["Statement"])


def test_policy_fits_the_customer_managed_policy_size_limit():
    # IAM のカスタマー管理ポリシーは 6,144 文字まで（空白は数えない）。ユーザーのインライン（2,048）には入らない → 管理ポリシーとして作る
    compact = re.sub(r"\s+", "", POLICY_PATH.read_text(encoding="utf-8"))
    assert len(compact) <= 6144, len(compact)


def test_no_wildcard_actions_and_no_broad_services():
    for a in allowed_actions():
        assert "*" not in a, a
        assert a.split(":")[0] in {"ec2", "ssm", "iam", "s3", "servicequotas"}


def test_wide_resources_are_only_for_read_only_calls():
    read_only = re.compile(r"^(ec2:Describe|ssm:(Get|List|Describe|Cancel)|servicequotas:(Get|List))")
    for st in statements():
        if "*" in as_list(st["Resource"]):
            for a in as_list(st["Action"]):
                assert read_only.match(a), (st["Sid"], a)


def test_every_action_is_a_real_api_operation():
    special_reverse = {v: k for k, v in S3_OP_TO_ACTION.items()}
    for a in allowed_actions() + [x for st in instance_s3_policy("b")["Statement"] for x in as_list(st["Action"])]:
        service, op = a.split(":")
        if a in PERMISSION_ONLY_ACTIONS:
            continue
        names = real_operation_names("service-quotas" if service == "servicequotas" else service)
        ok = op in names or (service == "s3" and special_reverse.get(op) in names)
        assert ok, f"{a} は存在する操作ではありません（タイプミス？）"


def test_instances_must_be_tagged_when_created_and_only_tagged_ones_are_terminated():
    by_sid = {s["Sid"]: s for s in statements()}
    run_new = by_sid["Ec2RunInstancesNewResourcesMustBeTagged"]
    assert run_new["Condition"]["StringEquals"]["aws:RequestTag/Project"] == "meeting-translator"
    assert any(r.endswith(":instance/*") for r in run_new["Resource"])
    term = by_sid["Ec2TerminateOnlyOurInstances"]
    assert term["Action"] == "ec2:TerminateInstances"
    assert term["Condition"]["StringEquals"]["ec2:ResourceTag/Project"] == "meeting-translator"
    send = by_sid["SsmSendCommandOnlyOurInstances"]
    assert send["Condition"]["StringEquals"]["ssm:resourceTag/Project"] == "meeting-translator"
    doc = by_sid["SsmSendCommandDocument"]
    assert doc["Resource"].endswith("document/AWS-RunShellScript")
    tags = by_sid["Ec2TagOnlyAtCreation"]
    assert tags["Condition"]["StringEquals"]["ec2:CreateAction"] == ["RunInstances", "CreateSecurityGroup"]


def test_ec2_and_ssm_resources_are_pinned_to_tokyo_and_iam_to_our_names():
    for st in statements():
        for a in as_list(st["Action"]):
            res = [r for r in as_list(st["Resource"]) if r != "*"]
            if a.startswith(("ec2:Run", "ec2:Terminate", "ec2:CreateTags", "ssm:SendCommand", "ssm:GetParameter")):
                assert res and all(f":{REGION}:" in r for r in res), (st["Sid"], res)
            if a.startswith("iam:") and a != "iam:CreateServiceLinkedRole":
                assert res and all("meeting-translator-" in r for r in res), (st["Sid"], res)
            if a.startswith("s3:"):
                assert res and all("meeting-translator-*" in r for r in res), (st["Sid"], res)


def test_iam_policy_attachment_is_limited_to_the_ssm_managed_policy_and_passrole_to_ec2():
    by_sid = {s["Sid"]: s for s in statements()}
    att = by_sid["IamAttachOnlyTheSsmManagedPolicy"]
    assert att["Condition"]["ArnEquals"]["iam:PolicyARN"] == common.SSM_CORE_POLICY_ARN
    pr = by_sid["IamPassRoleToEc2Only"]
    assert pr["Condition"]["StringEquals"]["iam:PassedToService"] == "ec2.amazonaws.com"
    sl = by_sid["IamSpotServiceLinkedRoleFirstUse"]
    assert sl["Resource"].endswith("AWSServiceRoleForEC2Spot") and sl["Condition"]["StringEquals"]["iam:AWSServiceName"] == "spot.amazonaws.com"


def test_the_policy_does_not_allow_opening_inbound_ports_or_ssh_keys():
    acts = set(allowed_actions())
    assert not {"ec2:AuthorizeSecurityGroupIngress", "ec2:CreateKeyPair", "ec2:ImportKeyPair", "ec2:CreateImage", "ec2:StartInstances"} & acts
    assert "ec2:RevokeSecurityGroupIngress" in acts


# ---------------------------------------------------------------------------
# ツールが実際に呼ぶ操作が全部許可されているか
# ---------------------------------------------------------------------------
def test_every_api_call_the_tool_makes_is_allowed_by_the_policy(aws, state, tmp_path):
    """全コマンドを流して記録した呼び出しが、すべてポリシーで許可されること（足りない権限があると、ここで分かる）。"""
    assert run_preflight(aws, Args(type="g4dn.xlarge", spot=False)) == 0
    do_setup(aws, state)
    do_launch(aws, state)
    aws.client("ssm").on_send = lambda kw, cid: fill_results(aws)
    assert run_run(aws, state, rargs(public_test=True, dry_run_first=True, out_dir=str(tmp_path / "o")), sleep=lambda s: None, poll=1) == 0
    assert run_run(aws, state, rargs(public_test=True, fresh=True, out_dir=str(tmp_path / "o2")), sleep=lambda s: None, poll=1) == 0
    assert run_fetch(aws, state, Args(exp="exp01", out_dir=str(tmp_path / "o3"), skip_audio=False)) == 0
    assert run_status(aws, state, Args()) == 0
    td = Args(all=True, delete_bucket=True, confirm_bucket=BUCKET, delete_setup=True, yes=True)
    assert run_teardown(aws, state, td, sleep=lambda s: None) == 0

    seen = sorted({action_for(s, o) for s, o in aws.calls})
    missing = [a for a in seen if not is_allowed(a) and a not in NO_PERMISSION_NEEDED]
    assert missing == [], f"ポリシーに足りない操作: {missing}"
    # 主要な操作が本当に流れている（テスト自体が空振りしていない）
    for must in ("ec2:RunInstances", "ec2:TerminateInstances", "ssm:SendCommand", "ssm:GetCommandInvocation", "ssm:GetParameter",
                 "ssm:DescribeInstanceInformation", "s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:CreateBucket", "s3:DeleteBucket",
                 "s3:PutEncryptionConfiguration", "s3:PutLifecycleConfiguration", "s3:PutBucketPublicAccessBlock", "s3:ListBucket",
                 "iam:CreateRole", "iam:CreateInstanceProfile", "iam:PutRolePolicy", "iam:AttachRolePolicy",
                 "servicequotas:GetServiceQuota", "ec2:CreateSecurityGroup", "ec2:DescribeInstanceTypeOfferings", "ec2:DescribeImages",
                 "ec2:DescribeVolumes", "iam:DeleteRole", "ec2:DeleteSecurityGroup"):
        assert must in seen, must


def test_the_instance_role_policy_is_a_subset_of_what_the_user_can_pass():
    """インスタンスに渡す S3 権限は、ユーザーのポリシーが許すバケット（meeting-translator-*）の中だけ。"""
    doc = instance_s3_policy("meeting-translator-abc-ap-northeast-1")
    for st in doc["Statement"]:
        assert fnmatch.fnmatchcase(st["Resource"], "arn:aws:s3:::meeting-translator-*")
