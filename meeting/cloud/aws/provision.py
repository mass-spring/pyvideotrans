"""setup: 実験用の AWS の土台を、何度実行しても同じ結果になるように作る。

作るもの（すべて Project=meeting-translator のタグ付き）:
  1. S3 バケット    ... 入力と成果物の置き場。非公開・SSE-S3 で暗号化・14日で自動削除
  2. IAM ロール＋インスタンスプロファイル ... AmazonSSMManagedInstanceCore と「このバケットだけ」の S3 権限
  3. セキュリティグループ ... インバウンド（外からの接続）は1つも許可しない

実行前に作る内容を必ず表示し、--yes が無ければ何も作らない。
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from .common import (
    PROFILE_NAME,
    ROLE_NAME,
    SG_NAME,
    SSM_CORE_POLICY_ARN,
    INSTANCE_S3_POLICY_NAME,
    Aws,
    State,
    ToolError,
    base_tags,
    bucket_name,
    error_code,
    explain_client_error,
    iso,
    mask_account,
    require_yes,
    say,
    tag_spec,
    tags_to_dict,
    utcnow,
    is_ours,
    warn,
)

BUCKET_EXPIRE_DAYS = 14


# ---------------------------------------------------------------------------
# ポリシー文書（テストで中身を確かめる）
# ---------------------------------------------------------------------------
def trust_policy() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }


def instance_s3_policy(bucket: str) -> dict[str, Any]:
    """インスタンスに与える S3 権限。このバケットの中だけ。削除権限・他のバケットへの権限は無い。
    GetEncryptionConfiguration は SSM Run Command の出力を S3 に書くときに確認される (要検証)。"""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "BucketLevel", "Effect": "Allow",
             "Action": ["s3:ListBucket", "s3:GetBucketLocation", "s3:GetEncryptionConfiguration",
                        "s3:ListBucketMultipartUploads"],
             "Resource": f"arn:aws:s3:::{bucket}"},
            {"Sid": "ObjectLevel", "Effect": "Allow",
             "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
        ],
    }


def tls_only_bucket_policy(bucket: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }],
    }


# ---------------------------------------------------------------------------
# 現状の確認（読み取りだけ）
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SetupStatus:
    region: str
    bucket: str
    bucket_exists: bool | None = None
    role_exists: bool | None = None
    profile_exists: bool | None = None
    profile_has_role: bool | None = None
    vpc_id: str | None = None
    sg_id: str | None = None
    sg_inbound_rules: int | None = None
    notes: list[str] = dataclasses.field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.bucket_exists and self.role_exists and self.profile_exists and self.profile_has_role and self.sg_id)


def default_vpc_id(aws: Aws) -> str | None:
    resp = aws.client("ec2").describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpcs = resp.get("Vpcs", [])
    return vpcs[0]["VpcId"] if vpcs else None


def find_sg(aws: Aws, vpc_id: str) -> dict[str, Any] | None:
    resp = aws.client("ec2").describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc_id]}])
    groups = resp.get("SecurityGroups", [])
    return groups[0] if groups else None


def discover(aws: Aws) -> SetupStatus:
    """作る前の状態を調べる。権限が無い項目は None（不明）のままにして、止まらない。"""
    st = SetupStatus(region=aws.region, bucket=bucket_name(aws.account_id, aws.region))
    s3, iam = aws.client("s3"), aws.client("iam")
    try:
        s3.head_bucket(Bucket=st.bucket)
        st.bucket_exists = True
    except Exception as e:  # noqa: BLE001
        code = error_code(e)
        if code in ("404", "NoSuchBucket", "NotFound"):
            st.bucket_exists = False
        elif code in ("403", "AccessDenied", "Forbidden"):
            st.notes.append("バケット名がすでに他人のものか、確認する権限がありません（403）")
        else:
            st.notes.append(f"バケットの確認に失敗: {code or e}")
    try:
        iam.get_role(RoleName=ROLE_NAME)
        st.role_exists = True
    except Exception as e:  # noqa: BLE001
        if error_code(e) == "NoSuchEntity":
            st.role_exists = False
        else:
            st.notes.append(f"IAM ロールの確認に失敗: {error_code(e) or e}")
    try:
        prof = iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)["InstanceProfile"]
        st.profile_exists = True
        st.profile_has_role = any(r.get("RoleName") == ROLE_NAME for r in prof.get("Roles", []))
    except Exception as e:  # noqa: BLE001
        if error_code(e) == "NoSuchEntity":
            st.profile_exists = False
            st.profile_has_role = False
        else:
            st.notes.append(f"インスタンスプロファイルの確認に失敗: {error_code(e) or e}")
    try:
        st.vpc_id = default_vpc_id(aws)
        if st.vpc_id is None:
            st.notes.append("デフォルト VPC がありません（`aws ec2 create-default-vpc` で作れます）")
        else:
            sg = find_sg(aws, st.vpc_id)
            if sg:
                st.sg_id = sg["GroupId"]
                st.sg_inbound_rules = len(sg.get("IpPermissions", []))
    except Exception as e:  # noqa: BLE001
        st.notes.append(f"セキュリティグループの確認に失敗: {error_code(e) or e}")
    return st


def plan_lines(st: SetupStatus) -> list[str]:
    def mark(exists: bool | None) -> str:
        return "[ある]" if exists else ("[新規]" if exists is False else "[不明]")

    return [
        f"{mark(st.bucket_exists)} S3 バケット {st.bucket}\n"
        f"        非公開（パブリックアクセスを全部ブロック）／SSE-S3 で暗号化／{BUCKET_EXPIRE_DAYS}日たつと自動で削除／HTTPS 以外は拒否",
        f"{mark(st.role_exists)} IAM ロール {ROLE_NAME}\n"
        f"        AmazonSSMManagedInstanceCore ＋ 上のバケットの中だけの S3 権限（他のバケットは触れない）",
        f"{mark(None if st.profile_exists is None else bool(st.profile_exists and st.profile_has_role))} "
        f"インスタンスプロファイル {PROFILE_NAME}（上のロールを入れる）",
        f"{mark(st.sg_id is not None if st.vpc_id else None)} セキュリティグループ {SG_NAME}（デフォルト VPC。インバウンドの許可は 0 件）",
    ]


# ---------------------------------------------------------------------------
# 作る（どれも「あれば作らない」「設定は毎回当て直す」）
# ---------------------------------------------------------------------------
def ensure_bucket(aws: Aws, name: str) -> bool:
    """作ったら True、すでにあれば False。"""
    s3 = aws.client("s3")
    created = False
    try:
        s3.head_bucket(Bucket=name)
    except Exception as e:  # noqa: BLE001
        code = error_code(e)
        if code in ("404", "NoSuchBucket", "NotFound"):
            kwargs: dict[str, Any] = {"Bucket": name}
            if aws.region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": aws.region}
            try:
                s3.create_bucket(**kwargs)
                created = True
            except Exception as e2:  # noqa: BLE001
                if error_code(e2) != "BucketAlreadyOwnedByYou":
                    raise
        elif code in ("403", "AccessDenied", "Forbidden"):
            raise ToolError(f"バケット {name} はすでに存在し、このアカウントからはアクセスできません。"
                            "別のアカウントが同じ名前を使っている可能性があります。") from e
        else:
            raise
    s3.put_public_access_block(Bucket=name, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(Bucket=name, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
    s3.put_bucket_lifecycle_configuration(Bucket=name, LifecycleConfiguration={"Rules": [{
        "ID": f"expire-after-{BUCKET_EXPIRE_DAYS}-days", "Status": "Enabled", "Filter": {"Prefix": ""},
        "Expiration": {"Days": BUCKET_EXPIRE_DAYS},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}]})
    s3.put_bucket_tagging(Bucket=name, Tagging={"TagSet": base_tags()})
    s3.put_bucket_policy(Bucket=name, Policy=json.dumps(tls_only_bucket_policy(name)))
    return created


def ensure_role(aws: Aws, bucket: str) -> bool:
    iam = aws.client("iam")
    created = False
    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        if not is_ours(tags_to_dict(role.get("Tags"))):
            raise ToolError(f"同じ名前の IAM ロール {ROLE_NAME} がすでにあり、このツールが作ったものではありません（タグがありません）。"
                            "上書きしないので、コンソールで名前を確認してください。")
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001
        if error_code(e) != "NoSuchEntity":
            raise
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust_policy()),
                        Description="meeting-translator ec2run: SSM + 専用バケットだけ", Tags=base_tags())
        created = True
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=SSM_CORE_POLICY_ARN)
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName=INSTANCE_S3_POLICY_NAME,
                        PolicyDocument=json.dumps(instance_s3_policy(bucket)))
    return created


def ensure_instance_profile(aws: Aws) -> bool:
    iam = aws.client("iam")
    created = False
    try:
        prof = iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)["InstanceProfile"]
        roles = [r.get("RoleName") for r in prof.get("Roles", [])]
    except Exception as e:  # noqa: BLE001
        if error_code(e) != "NoSuchEntity":
            raise
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME, Tags=base_tags())
        created, roles = True, []
    if ROLE_NAME not in roles:
        try:
            iam.add_role_to_instance_profile(InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME)
        except Exception as e:  # noqa: BLE001
            if error_code(e) != "LimitExceeded":   # すでに別のロールが入っている
                raise
            raise ToolError(f"インスタンスプロファイル {PROFILE_NAME} には、すでに別のロールが入っています。コンソールで確認してください。") from e
    return created


def ensure_security_group(aws: Aws, vpc_id: str) -> tuple[str, bool]:
    ec2 = aws.client("ec2")
    sg = find_sg(aws, vpc_id)
    created = False
    if sg is None:
        resp = ec2.create_security_group(
            GroupName=SG_NAME, Description="meeting-translator ec2run: no inbound (SSM only)", VpcId=vpc_id,
            TagSpecifications=[tag_spec("security-group", {"Name": SG_NAME})])
        sg_id = resp["GroupId"]
        created = True
        inbound: list[dict[str, Any]] = []
    else:
        sg_id = sg["GroupId"]
        if not is_ours(tags_to_dict(sg.get("Tags"))):
            raise ToolError(f"同じ名前のセキュリティグループ {SG_NAME} がすでにあり、このツールが作ったものではありません。上書きしません。")
        inbound = sg.get("IpPermissions", [])
    if inbound:   # このツールのグループにインバウンドがあるのは想定外なので、外す
        ec2.revoke_security_group_ingress(GroupId=sg_id, IpPermissions=inbound)
        warn(f"セキュリティグループ {sg_id} にあったインバウンドのルール {len(inbound)} 件を削除しました（外からの接続は許可しません）")
    return sg_id, created


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run_setup(aws: Aws, state: State, args: Any) -> int:
    st = discover(aws)
    say("== setup: 作る AWS リソース ==")
    say(f"リージョン: {aws.region} / アカウント: {mask_account(aws.account_id)}")
    for line in plan_lines(st):
        say("  " + line)
    for n in st.notes:
        warn(n)
    say("  （すべてに タグ Project=meeting-translator / ManagedBy=ec2run を付けます。S3・IAM・セキュリティグループ自体は無料です）")
    if st.vpc_id is None:
        raise ToolError("デフォルト VPC を使えません。`aws ec2 create-default-vpc` で作るか、コンソールの VPC で作成してください"
                        "（権限不足で確認できなかった場合は README の IAM ポリシーを確認）。")
    if st.bucket_exists is None:
        raise ToolError("S3 バケットの状態を確認できません（名前が他人のものか、権限不足）。上の注意を確認してください。")
    require_yes(args, "上のリソースを作成します（すでにあるものは作り直さず、設定だけ当て直します）。")

    say("")
    try:
        created_bucket = ensure_bucket(aws, st.bucket)
        say(f"  S3 バケット: {'作成しました' if created_bucket else 'すでにあります（設定を当て直しました）'}: {st.bucket}")
        created_role = ensure_role(aws, st.bucket)
        say(f"  IAM ロール: {'作成しました' if created_role else 'すでにあります（権限を当て直しました）'}: {ROLE_NAME}")
        created_prof = ensure_instance_profile(aws)
        say(f"  インスタンスプロファイル: {'作成しました' if created_prof else 'すでにあります'}: {PROFILE_NAME}")
        sg_id, created_sg = ensure_security_group(aws, st.vpc_id)
        say(f"  セキュリティグループ: {'作成しました' if created_sg else 'すでにあります'}: {sg_id}（インバウンド 0 件）")
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001
        if hasattr(e, "response"):
            raise ToolError(explain_client_error(e, "setup")) from e
        raise

    state.set(region=aws.region, account_last4=aws.account_id[-4:], bucket=st.bucket, role_name=ROLE_NAME,
              instance_profile=PROFILE_NAME, security_group_id=sg_id, vpc_id=st.vpc_id, setup_at=iso(utcnow()))
    say("")
    say(f"状態ファイル {state.path.name} に記録しました。")
    if created_prof:
        say("メモ: 作ったばかりのインスタンスプロファイルは AWS 内で反映されるまで数十秒かかることがあります（launch が自動で待ち直します）。")
    say("次: python -m meeting.cloud.aws launch --type g4dn.xlarge --hours 3   （費用の見積を見て、よければ --yes を付けて実行）")
    return 0


def resolve_resources(aws: Aws, state: State) -> dict[str, str]:
    """launch / run が使う ID。状態ファイルに無くても、決まった名前で探して見つける（環境が消えても復旧できる）。"""
    bucket = state.get("bucket") or bucket_name(aws.account_id, aws.region)
    sg_id = state.get("security_group_id")
    vpc_id = state.get("vpc_id")
    if not sg_id or not vpc_id:
        vpc_id = default_vpc_id(aws)
        sg = find_sg(aws, vpc_id) if vpc_id else None
        sg_id = sg["GroupId"] if sg else None
    if not sg_id:
        raise ToolError("セキュリティグループがありません。先に `setup --yes` を実行してください。")
    try:
        aws.client("s3").head_bucket(Bucket=bucket)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"S3 バケット {bucket} を確認できません（{error_code(e) or e}）。先に `setup --yes` を実行してください。") from e
    return {"bucket": bucket, "security_group_id": sg_id, "vpc_id": vpc_id or "",
            "instance_profile": state.get("instance_profile") or PROFILE_NAME}


# ---------------------------------------------------------------------------
# 片付け（teardown --delete-setup から呼ぶ）
# ---------------------------------------------------------------------------
def delete_setup_resources(aws: Aws) -> list[str]:
    """ロール・インスタンスプロファイル・セキュリティグループを消す。バケットは別（--delete-bucket）。"""
    done: list[str] = []
    iam, ec2 = aws.client("iam"), aws.client("ec2")

    def tolerant(label: str, fn) -> None:
        try:
            fn()
            done.append(label)
        except Exception as e:  # noqa: BLE001
            code = error_code(e)
            if code == "NoSuchEntity":
                return
            if code in ("InvalidGroup.NotFound",):
                return
            warn(f"{label} を削除できませんでした: {code or e}")

    tolerant("インスタンスプロファイルからロールを外す", lambda: iam.remove_role_from_instance_profile(
        InstanceProfileName=PROFILE_NAME, RoleName=ROLE_NAME))
    tolerant(f"インスタンスプロファイル {PROFILE_NAME}", lambda: iam.delete_instance_profile(InstanceProfileName=PROFILE_NAME))
    tolerant("ロールの S3 権限", lambda: iam.delete_role_policy(RoleName=ROLE_NAME, PolicyName=INSTANCE_S3_POLICY_NAME))
    tolerant("ロールの SSM 権限", lambda: iam.detach_role_policy(RoleName=ROLE_NAME, PolicyArn=SSM_CORE_POLICY_ARN))
    tolerant(f"IAM ロール {ROLE_NAME}", lambda: iam.delete_role(RoleName=ROLE_NAME))
    vpc = default_vpc_id(aws)
    sg = find_sg(aws, vpc) if vpc else None
    if sg and is_ours(tags_to_dict(sg.get("Tags"))):
        tolerant(f"セキュリティグループ {sg['GroupId']}", lambda: ec2.delete_security_group(GroupId=sg["GroupId"]))
    return done
