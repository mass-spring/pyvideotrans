"""ec2run 共通部品: 鍵の読み込み（環境変数）、伏せ字、状態ファイル、料金、名前、エラー表示。"""
from __future__ import annotations

import json
import os
import stat

import pytest

from test_cloud_aws_helpers import REGION, client_error, make_settings  # noqa: F401  (REPO_ROOT を sys.path に足す副作用も含む)

from meeting.cloud.aws import common
from meeting.cloud.aws.common import (
    Aws,
    NeedConfirmation,
    State,
    ToolError,
    bucket_name,
    estimate_cost,
    explain_client_error,
    load_settings,
    mask_account,
    mask_text,
    require_yes,
)
from meeting.cloud.aws.ssmrun import parse_s3_url


# ---------------------------------------------------------------------------
# 鍵（EC2_*）
# ---------------------------------------------------------------------------
def test_settings_come_from_ec2_variables_not_aws_placeholders():
    env = {"AWS_ACCESS_KEY_ID": "PLACEHOLDER", "AWS_SECRET_ACCESS_KEY": "PLACEHOLDER2",
           "EC2_ACCESS_KEY_ID": "AKIAREALREALREAL1234", "EC2_SECRET_ACCESS_KEY": "realsecretrealsecret", "EC2_REGION": "us-west-2"}
    s = load_settings(env)
    assert s.access_key_id == "AKIAREALREALREAL1234" and s.region == "us-west-2"


def test_region_defaults_to_tokyo_and_cli_region_wins():
    env = {"EC2_ACCESS_KEY_ID": "AKIAREALREALREAL1234", "EC2_SECRET_ACCESS_KEY": "realsecretrealsecret"}
    assert load_settings(env).region == "ap-northeast-1"
    assert load_settings(env, region="eu-west-1").region == "eu-west-1"


def test_missing_keys_explain_without_touching_aws_variables():
    with pytest.raises(ToolError) as e:
        load_settings({"AWS_ACCESS_KEY_ID": "PLACEHOLDER", "AWS_SECRET_ACCESS_KEY": "PLACEHOLDER"})
    msg = str(e.value)
    assert "EC2_ACCESS_KEY_ID" in msg and "EC2_SECRET_ACCESS_KEY" in msg and "README" in msg
    assert "PLACEHOLDER" not in msg


def test_pasted_keys_with_spaces_or_quotes_are_rejected_without_echoing_them():
    env = {"EC2_ACCESS_KEY_ID": "AKIA REAL", "EC2_SECRET_ACCESS_KEY": "secretsecretsecret"}
    with pytest.raises(ToolError) as e:
        load_settings(env)
    assert "AKIA REAL" not in str(e.value)
    env = {"EC2_ACCESS_KEY_ID": "AKIAREALREALREAL1234", "EC2_SECRET_ACCESS_KEY": '"quoted"'}
    with pytest.raises(ToolError):
        load_settings(env)


def test_trailing_newline_from_copy_paste_is_stripped():
    env = {"EC2_ACCESS_KEY_ID": "AKIAREALREALREAL1234\n", "EC2_SECRET_ACCESS_KEY": "realsecretrealsecret\r\n"}
    s = load_settings(env)
    assert s.access_key_id.endswith("1234") and s.secret_access_key.endswith("secret")


def test_bad_region_name_is_rejected():
    env = {"EC2_ACCESS_KEY_ID": "AKIAREALREALREAL1234", "EC2_SECRET_ACCESS_KEY": "realsecretrealsecret", "EC2_REGION": "tokyo"}
    with pytest.raises(ToolError):
        load_settings(env)


def test_secrets_never_appear_in_repr_or_redacted_text():
    s = make_settings()
    assert s.secret_access_key not in repr(s) and s.access_key_id not in repr(s)
    text = f"boom {s.secret_access_key} and {s.access_key_id}"
    red = s.redact(text)
    assert s.secret_access_key not in red and s.access_key_id not in red and "***" in red


def test_session_is_built_from_explicit_keys_ignoring_aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAPLACEHOLDERXXXXXX")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "placeholder")
    aws = Aws(make_settings(), ca_bundle="")
    creds = aws.session.get_credentials()
    assert creds.access_key == "AKIATESTTESTTESTTEST"
    assert aws.session.region_name == REGION


def test_client_uses_ca_bundle_instead_of_disabling_verification(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("dummy")
    aws = Aws(make_settings(), ca_bundle=str(ca))
    c = aws.client("sts")
    assert getattr(c._endpoint.http_session, "_verify", None) == str(ca)


def test_find_ca_bundle_prefers_aws_ca_bundle(tmp_path):
    a, b = tmp_path / "a.crt", tmp_path / "b.crt"
    a.write_text("x")
    b.write_text("y")
    assert common.find_ca_bundle({"AWS_CA_BUNDLE": str(a), "SSL_CERT_FILE": str(b)}) == str(a)
    assert common.find_ca_bundle({"SSL_CERT_FILE": str(b)}) == str(b)


# ---------------------------------------------------------------------------
# 伏せ字・名前・料金
# ---------------------------------------------------------------------------
def test_account_id_is_masked_except_last4():
    assert mask_account("123456789012") == "********9012"
    assert "123456789012" not in mask_text("arn:aws:iam::123456789012:user/x", "123456789012")
    assert mask_text("arn:aws:iam::123456789012:user/x", "123456789012").endswith("9012:user/x")


def test_bucket_name_is_deterministic_valid_and_hides_the_account():
    n = bucket_name("123456789012", "ap-northeast-1")
    assert n == bucket_name("123456789012", "ap-northeast-1")
    assert n != bucket_name("123456789013", "ap-northeast-1") and n != bucket_name("123456789012", "us-east-1")
    assert n.startswith("meeting-translator-") and 3 <= len(n) <= 63 and n == n.lower()
    assert "123456789012" not in n and "_" not in n


def test_price_table_has_the_two_requested_types_with_verified_tokyo_rates():
    assert common.INSTANCE_SPECS["g4dn.xlarge"].usd_per_hour == pytest.approx(0.71)
    assert common.INSTANCE_SPECS["g5.xlarge"].usd_per_hour == pytest.approx(1.459)
    assert common.INSTANCE_SPECS["g4dn.xlarge"].vcpu == 4 and common.INSTANCE_SPECS["g5.xlarge"].vcpu == 4
    assert "概算" in common.PRICE_NOTE and "料金ページ" in common.PRICE_NOTE


def test_estimate_cost_adds_disk_and_ipv4():
    e = estimate_cost(0.71, 3.0, 100)
    assert e.compute == pytest.approx(2.13)
    assert e.ebs == pytest.approx(100 * 0.096 / 730 * 3)
    assert e.ipv4 == pytest.approx(0.015)
    assert e.total == pytest.approx(e.compute + e.ebs + e.ipv4) and 2.1 < e.total < 2.3


def test_rate_for_unknown_type_needs_override():
    assert common.rate_for("p5.48xlarge") is None
    assert common.rate_for("p5.48xlarge", 55.0) == 55.0


def test_quota_codes_are_the_documented_ones():
    assert common.QUOTA_ONDEMAND_G == "L-DB2E81BA" and common.QUOTA_SPOT_G == "L-3819A6DF"


def test_tags_mark_everything_as_ours():
    tags = common.tags_to_dict(common.base_tags({"Name": "x"}))
    assert tags["Project"] == "meeting-translator" and tags["ManagedBy"] == "ec2run" and tags["Name"] == "x"
    assert common.is_ours(tags) and not common.is_ours({"Project": "meeting-translator"}) and not common.is_ours({})


# ---------------------------------------------------------------------------
# --yes
# ---------------------------------------------------------------------------
def test_require_yes_refuses_without_the_flag():
    class A:
        yes = False

    with pytest.raises(NeedConfirmation) as e:
        require_yes(A(), "作ります")
    assert "--yes" in str(e.value) and e.value.exit_code == 2
    A.yes = True
    require_yes(A(), "作ります")


# ---------------------------------------------------------------------------
# 状態ファイル
# ---------------------------------------------------------------------------
def test_state_roundtrip_and_private_permissions(tmp_path):
    p = tmp_path / "sub" / ".state.json"
    st = State(p)
    st.set(bucket="b1", security_group_id="sg-1")
    st.record_instance("i-0123456789abcdef0", type="g4dn.xlarge", hours=3)
    st.record_run("exp01", command_id="c1", status="running")
    st2 = State(p)
    assert st2.get("bucket") == "b1" and st2.instances()["i-0123456789abcdef0"]["type"] == "g4dn.xlarge"
    assert st2.last_run("exp01")["command_id"] == "c1"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert not list(p.parent.glob("*.tmp"))


def test_state_same_experiment_is_updated_not_duplicated(tmp_path):
    st = State(tmp_path / "s.json")
    st.record_run("exp01", command_id="c1")
    st.record_run("exp01", status="done")
    st.record_run("exp02", command_id="c2")
    assert [r["exp"] for r in st.runs()] == ["exp01", "exp02"] and st.last_run("exp01") == {"exp": "exp01", "command_id": "c1", "status": "done"}


def test_corrupt_state_is_set_aside_and_does_not_crash(tmp_path, capsys):
    p = tmp_path / ".state.json"
    p.write_text("{not json", encoding="utf-8")
    st = State(p)
    assert st.data == {"version": 1}
    assert list(tmp_path.glob(".state.json.corrupt-*")) and not p.exists()
    assert "空の状態" in capsys.readouterr().out


def test_state_file_location_can_be_overridden_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EC2RUN_STATE_FILE", str(tmp_path / "custom.json"))
    st = State()
    st.set(x=1)
    assert json.loads((tmp_path / "custom.json").read_text())["x"] == 1


def test_state_file_is_gitignored_in_this_folder():
    gi = (common.AWS_DIR / ".gitignore").read_text(encoding="utf-8")
    assert ".state.json" in gi.split()


def test_default_state_path_is_inside_the_aws_folder(monkeypatch):
    monkeypatch.delenv("EC2RUN_STATE_FILE", raising=False)
    assert common.state_path() == common.AWS_DIR / ".state.json"


# ---------------------------------------------------------------------------
# エラーの説明・URL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code,needle", [
    ("InvalidClientTokenId", "鍵が無効"),
    ("AuthFailure", "鍵が無効"),
    ("ExpiredToken", "期限"),
    ("UnauthorizedOperation", "権限"),
    ("AccessDenied", "iam_user_policy.json"),
    ("VcpuLimitExceeded", "preflight"),
    ("InsufficientInstanceCapacity", "g5.xlarge"),
])
def test_client_errors_are_explained_in_japanese(code, needle):
    msg = explain_client_error(client_error(code, "denied"), "テスト")
    assert needle in msg and code in msg


def test_parse_s3_url_forms():
    assert parse_s3_url("s3://b/k/x.txt") == ("b", "k/x.txt")
    assert parse_s3_url("https://s3.ap-northeast-1.amazonaws.com/bkt/exp/a/stdout") == ("bkt", "exp/a/stdout")
    assert parse_s3_url("https://bkt.s3.ap-northeast-1.amazonaws.com/exp/a/stdout") == ("bkt", "exp/a/stdout")
    with pytest.raises(ToolError):
        parse_s3_url("https://example.com/x")
