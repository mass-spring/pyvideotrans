"""meeting/cloud/aws（ec2run）のテスト用の部品（テスト本体は無い）。他の test_cloud_aws_*.py が import して使う。

- S3 / IAM / EC2 / STS は moto（本物そっくりの偽の AWS。通信しない）
- SSM と Service Quotas は moto が未対応の操作があるので、小さな偽物（FakeSSM / FakeQuotas）
- すべての AWS 呼び出しを記録して、「作る・消す系の操作が呼ばれていないか」「IAM ポリシーで足りるか」を確かめられる
実行: cd meeting && uv run --with boto3 --with moto pytest -q src/tests/test_cloud_aws_*.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))      # import meeting.cloud.aws を可能にする（meeting/ は名前空間パッケージ）

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from botocore.exceptions import ClientError  # noqa: E402

from meeting.cloud.aws import common  # noqa: E402
from meeting.cloud.aws.common import Aws, Settings, State  # noqa: E402

REGION = "ap-northeast-1"
DLAMI_PARAM_22 = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
DLAMI_PARAM_24 = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"

# 作る・消す・変える系の操作（名前の先頭）。「何もしないはずのコマンド」で呼ばれていないことの確認に使う
MUTATING_PREFIXES = ("Create", "Put", "Delete", "Run", "Terminate", "Attach", "Detach", "Add", "Remove", "Revoke",
                     "Authorize", "SendCommand", "Tag", "Start", "Stop", "Request", "Cancel", "Upload", "CompleteMultipart",
                     "AbortMultipart")


def client_error(code: str, message: str = "", op: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message or code}}, op)


# ---------------------------------------------------------------------------
# 偽の SSM / Service Quotas
# ---------------------------------------------------------------------------
class FakeBase:
    service = "?"

    def __init__(self, calls: list[tuple[str, str]]):
        self.calls = calls

    def rec(self, op: str) -> None:
        self.calls.append((self.service, op))


class FakeSSM(FakeBase):
    service = "ssm"

    def __init__(self, calls, ami_id: str, *, online_after_polls: int = 0):
        super().__init__(calls)
        self.params = {DLAMI_PARAM_22: ami_id, DLAMI_PARAM_24: ami_id}
        self.online_after_polls = online_after_polls
        self.polls = 0
        self.online_ids: set[str] | None = None        # None なら、問い合わせた ID は Online
        self.sent: list[dict[str, Any]] = []
        self.responder: Callable[[list[str]], tuple[str, str, str]] = self.default_responder
        self.on_send: Callable[[dict[str, Any], str], None] | None = None
        self.plans: dict[str, list[str]] = {}          # CommandId → これから返す Status の列
        self.long_statuses = ["InProgress", "Success"]
        self.stderr = ""
        self._result: dict[str, tuple[str, str, str]] = {}

    @staticmethod
    def default_responder(cmds: list[str]) -> tuple[str, str, str]:
        text = "\n".join(cmds)
        if "NOT_SCHEDULED" in text:
            return "Success", "SCHEDULED\nTesla T4, 15360 MiB, 595.91.07\n/dev/root 97G 30G 67G 31% /\n", ""
        if "shutdown -h +" in text:
            return "Success", "SCHEDULED\n", ""
        return "Success", "ok\n", ""

    def get_parameter(self, Name: str, **kw):
        self.rec("GetParameter")
        if Name not in self.params:
            raise client_error("ParameterNotFound", Name, "GetParameter")
        return {"Parameter": {"Name": Name, "Value": self.params[Name]}}

    def describe_instance_information(self, Filters=None, **kw):
        self.rec("DescribeInstanceInformation")
        ids = (Filters or [{}])[0].get("Values", [])
        self.polls += 1
        if self.polls <= self.online_after_polls:
            return {"InstanceInformationList": []}
        if self.online_ids is not None:
            ids = [i for i in ids if i in self.online_ids]
        return {"InstanceInformationList": [{"InstanceId": i, "PingStatus": "Online"} for i in ids]}

    def send_command(self, **kw):
        self.rec("SendCommand")
        cid = str(uuid.uuid4())
        self.sent.append({"CommandId": cid, **kw})
        is_long = any("remote_job.py" in c for c in kw["Parameters"]["commands"])
        if is_long:
            self.plans[cid] = list(self.long_statuses)
            self._result[cid] = ("Success", "", self.stderr)
            if self.on_send:
                self.on_send(kw, cid)
        else:
            self.plans[cid] = ["Success"]
            self._result[cid] = self.responder(kw["Parameters"]["commands"])
        return {"Command": {"CommandId": cid, "Status": "Pending"}}

    def get_command_invocation(self, CommandId: str, InstanceId: str, **kw):
        self.rec("GetCommandInvocation")
        plan = self.plans.get(CommandId)
        if plan is None:
            raise client_error("InvocationDoesNotExist", "", "GetCommandInvocation")
        status = plan.pop(0) if len(plan) > 1 else plan[0]
        final = status in ("Success", "Failed", "TimedOut", "Cancelled")
        st, out, err = self._result[CommandId]
        if status == "Success":
            status = st
        return {"CommandId": CommandId, "InstanceId": InstanceId, "Status": status,
                "StandardOutputContent": out if final else "", "StandardErrorContent": err if final else "",
                "StandardOutputUrl": ""}

    def cancel_command(self, **kw):
        self.rec("CancelCommand")
        return {}


class FakeQuotas(FakeBase):
    service = "servicequotas"

    def __init__(self, calls, values: dict[str, float | None] | None = None, *, source: str = "applied",
                 pending: dict[str, list[dict]] | None = None):
        super().__init__(calls)
        self.values = values if values is not None else {common.QUOTA_ONDEMAND_G: 8.0, common.QUOTA_SPOT_G: 8.0}
        self.source = source
        self.pending = pending or {}

    def get_service_quota(self, ServiceCode: str, QuotaCode: str, **kw):
        self.rec("GetServiceQuota")
        v = self.values.get(QuotaCode)
        if v is None or self.source == "default":
            raise client_error("NoSuchResourceException", "no applied value", "GetServiceQuota")
        return {"Quota": {"QuotaCode": QuotaCode, "Value": v}}

    def get_aws_default_service_quota(self, ServiceCode: str, QuotaCode: str, **kw):
        self.rec("GetAWSDefaultServiceQuota")
        v = self.values.get(QuotaCode)
        if v is None:
            raise client_error("AccessDeniedException", "denied", "GetAWSDefaultServiceQuota")
        return {"Quota": {"QuotaCode": QuotaCode, "Value": v}}

    def list_requested_service_quota_change_history_by_quota(self, ServiceCode: str, QuotaCode: str, **kw):
        self.rec("ListRequestedServiceQuotaChangeHistoryByQuota")
        return {"RequestedQuotas": self.pending.get(QuotaCode, [])}


# ---------------------------------------------------------------------------
# 記録つきの Aws
# ---------------------------------------------------------------------------
class RecordingAws(Aws):
    """moto で作った本物の boto3 クライアントに、呼び出しの記録を付ける。"""

    def __init__(self, settings: Settings, session: Any, calls: list[tuple[str, str]], clients: dict[str, Any]):
        super().__init__(settings, session=session, clients=clients, ca_bundle="")
        self.calls = calls

    def _make_client(self, service: str):
        c = super()._make_client(service)

        def record(model, **kw):
            name = model.service_model.service_name
            self.calls.append(("servicequotas" if name == "service-quotas" else name, model.name))

        c.meta.events.register("before-call.*.*", record)
        return c

    def mutating(self) -> list[tuple[str, str]]:
        return [c for c in self.calls if c[1].startswith(MUTATING_PREFIXES)]

    def ops(self, service: str) -> list[str]:
        return [o for s, o in self.calls if s == service]


def make_settings(region: str = REGION) -> Settings:
    return Settings("AKIATESTTESTTESTTEST", "testSecretKeyDoNotPrint0123456789abcdef", region)


_SSM_POLICY_JSON: str | None = None


def _small_managed_policies() -> str:
    """moto の AWS 管理ポリシー（数百件）のうち、使う 1 件だけ。全部読み込むと 1 テストごとに 1 秒以上かかる。"""
    global _SSM_POLICY_JSON
    if _SSM_POLICY_JSON is None:
        import json

        from moto.iam.aws_managed_policies import aws_managed_policies_data

        _SSM_POLICY_JSON = json.dumps({"AmazonSSMManagedInstanceCore": json.loads(aws_managed_policies_data)["AmazonSSMManagedInstanceCore"]})
    return _SSM_POLICY_JSON


@pytest.fixture
def mocked(monkeypatch):
    from moto import mock_aws

    monkeypatch.setenv("MOTO_IAM_LOAD_MANAGED_POLICIES", "true")    # AmazonSSMManagedInstanceCore を使えるようにする
    monkeypatch.setattr("moto.iam.models.aws_managed_policies_data", _small_managed_policies())
    monkeypatch.setenv("MOTO_EC2_LOAD_DEFAULT_AMIS", "false")      # 既定の AMI 1,300 件を読むと遅い。必要な 1 件だけ自分で登録する
    with mock_aws():
        yield


@pytest.fixture
def aws(mocked, monkeypatch, tmp_path) -> RecordingAws:
    """moto の上で動く Aws。SSM と Service Quotas は偽物。"""
    monkeypatch.setenv("EC2RUN_STATE_FILE", str(tmp_path / "state.json"))
    session = boto3.session.Session(aws_access_key_id="testing", aws_secret_access_key="testing", region_name=REGION)
    calls: list[tuple[str, str]] = []
    ami_id = session.client("ec2").register_image(
        Name="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261009", RootDeviceName="/dev/sda1",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 75, "VolumeType": "gp3"}}],
        Architecture="x86_64", VirtualizationType="hvm")["ImageId"]
    clients = {"ssm": FakeSSM(calls, ami_id), "service-quotas": FakeQuotas(calls)}
    a = RecordingAws(make_settings(), session, calls, clients)
    a.ami_id = ami_id
    return a


@pytest.fixture
def state(tmp_path) -> State:
    return State(tmp_path / "state.json")


class Args:
    """argparse の Namespace の代わり（足りない項目は False / None）。"""

    def __init__(self, **kw: Any):
        self.__dict__.update(kw)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def do_setup(aws: Aws, state: State, *, keep_policy: bool = False) -> None:
    """setup --yes を実行する。

    moto は「HTTPS 以外を拒否する」バケットポリシーの Condition を見ずに全部拒否してしまう（moto の制限）ので、
    テストでオブジェクトを出し入れできるよう、setup の後にバケットポリシーだけ外す（記録に残らない別のクライアントで）。
    ポリシーの中身そのものは test_cloud_aws_setup.py で keep_policy=True にして確かめている。
    """
    from meeting.cloud.aws.provision import run_setup

    assert run_setup(aws, state, Args(yes=True)) == 0
    if not keep_policy:
        aws.session.client("s3").delete_bucket_policy(Bucket=state.get("bucket"))


def launch_args(**kw: Any) -> Args:
    base = dict(type="g4dn.xlarge", spot=False, hours=3.0, os_version="22.04", volume_gb=100, hourly_rate=None,
                allow_multiple=False, no_wait=False, ssm_timeout=60, yes=True)
    base.update(kw)
    return Args(**base)


def do_launch(aws: Aws, state: State, **kw: Any) -> str:
    from meeting.cloud.aws.launch import run_launch

    assert run_launch(aws, state, launch_args(**kw), sleep=lambda s: None) == 0
    return state.get("last_instance_id")


def _noop(*a: Any, **k: Any) -> None:
    return None
