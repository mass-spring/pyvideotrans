"""共通部品: 設定（環境変数）、boto3 の接続、状態ファイル、料金表、表示の補助。

- boto3 は使う時まで import しない（このファイルを import しただけでは boto3 が無くてもよい）。
- 鍵は環境変数 EC2_ACCESS_KEY_ID / EC2_SECRET_ACCESS_KEY / EC2_REGION から読む。
  （この開発環境にある AWS_ACCESS_KEY_ID などは作業環境が入れた無効な仮の値なので、見に行かない）
- 鍵の値は絶対に表示しない（repr も、エラーメッセージも伏せる）。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
AWS_DIR = Path(__file__).resolve().parent            # meeting/cloud/aws
MEETING_DIR = AWS_DIR.parents[1]                     # meeting/
REPO_ROOT = MEETING_DIR.parent
DEFAULT_OUT_ROOT = MEETING_DIR / "outputs"           # 結果の置き場所（Git に入れない）
STATE_ENV = "EC2RUN_STATE_FILE"
DEFAULT_STATE_PATH = AWS_DIR / ".state.json"

DEFAULT_REGION = "ap-northeast-1"

# このツールが作るものすべてに付ける印。teardown / status はこの印がついたものしか触らない。
PROJECT_TAG_KEY, PROJECT_TAG_VALUE = "Project", "meeting-translator"
TOOL_TAG_KEY, TOOL_TAG_VALUE = "ManagedBy", "ec2run"

ROLE_NAME = "meeting-translator-ec2-role"
PROFILE_NAME = "meeting-translator-ec2-profile"
SG_NAME = "meeting-translator-sg"
SSM_CORE_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
INSTANCE_S3_POLICY_NAME = "meeting-translator-s3-bucket-only"

# Service Quotas（EC2）。値の単位は vCPU。
# https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
QUOTA_SERVICE = "ec2"
QUOTA_ONDEMAND_G = "L-DB2E81BA"   # Running On-Demand G and VT instances
QUOTA_SPOT_G = "L-3819A6DF"       # All G and VT Spot Instance Requests
QUOTA_NAMES = {
    QUOTA_ONDEMAND_G: "Running On-Demand G and VT instances",
    QUOTA_SPOT_G: "All G and VT Spot Instance Requests",
}

# 料金（概算）。ap-northeast-1（東京）・Linux・共有テナント・オンデマンド。
# 2026-10-15 公開の AWS 価格表（Price List API の AmazonEC2 / ap-northeast-1）から取った値。
# 価格は変わる。実行前に必ず AWS の料金ページで確認すること。
PRICE_NOTE = "概算 (approximate, check AWS pricing)：2026-10-15 時点の AWS 公開価格表・東京リージョン・Linux・オンデマンド。必ず AWS の料金ページで確認してください"
PRICE_REGION = "ap-northeast-1"
EBS_GP3_USD_GB_MONTH = 0.096       # 東京の gp3（価格表より）
PUBLIC_IPV4_USD_HOUR = 0.005       # 公開 IPv4 アドレス（AWS の発表値。価格表の別サービスなので (要確認)）
S3_STANDARD_USD_GB_MONTH = 0.025   # S3 Standard 東京（概算 (要確認)）
HOURS_PER_MONTH = 730
DEFAULT_VOLUME_GB = 100
MAX_HOURS = 24.0


@dataclasses.dataclass(frozen=True)
class InstanceSpec:
    vcpu: int
    gpu: str
    usd_per_hour: float


INSTANCE_SPECS: dict[str, InstanceSpec] = {
    "g4dn.xlarge": InstanceSpec(4, "NVIDIA T4 (16GB)", 0.71),
    "g5.xlarge": InstanceSpec(4, "NVIDIA A10G (24GB)", 1.459),
    "g4dn.2xlarge": InstanceSpec(8, "NVIDIA T4 (16GB)", 1.015),
    "g5.2xlarge": InstanceSpec(8, "NVIDIA A10G (24GB)", 1.75776),
    "g6.xlarge": InstanceSpec(4, "NVIDIA L4 (24GB)", 1.1672),
}

DATA_RULE = (
    "【データの規則】クラウドに上げてよい音声は、自分の声／同意を得た模擬会議／公開データだけです。\n"
    "  会社の会議音声・社外秘の音声は、この AWS アカウントが会社に承認されたものでない限り上げません。\n"
    "  声の複製は本人の同意が必要です（CLAUDE.md・PLAN.md D4）。"
)


# ---------------------------------------------------------------------------
# エラー
# ---------------------------------------------------------------------------
class ToolError(Exception):
    """利用者向けの日本語メッセージを持つエラー。終了コード 2。"""

    exit_code = 2


class NeedConfirmation(ToolError):
    """--yes が無いので実行しなかった。"""


def require_yes(args: Any, what: str, say_fn: Callable[[str], None] | None = None) -> None:
    """--yes が無ければ NeedConfirmation。呼び出し側は AWS を変更する前にこれを通す。"""
    if getattr(args, "yes", False):
        return
    raise NeedConfirmation(f"{what}\n→ 上の内容でよければ、同じコマンドに --yes を付けて実行してください（付けないと何も作りません）。")


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------
def say(msg: str = "") -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"! {msg}", flush=True)


def mask_account(account_id: str) -> str:
    """アカウント ID は下4桁以外を伏せる。"""
    account_id = str(account_id or "")
    if len(account_id) <= 4:
        return "*" * len(account_id)
    return "*" * (len(account_id) - 4) + account_id[-4:]


def mask_text(text: str, account_id: str | None) -> str:
    """文章の中のアカウント ID（12桁）を伏せる。"""
    if account_id:
        text = text.replace(account_id, mask_account(account_id))
    return text


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 費用の見積
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CostEstimate:
    hours: float
    rate_per_hour: float      # インスタンス本体
    compute: float
    ebs: float
    ipv4: float
    total: float


def extras_per_hour(volume_gb: float) -> float:
    return volume_gb * EBS_GP3_USD_GB_MONTH / HOURS_PER_MONTH + PUBLIC_IPV4_USD_HOUR


def estimate_cost(rate_per_hour: float, hours: float, volume_gb: float = DEFAULT_VOLUME_GB) -> CostEstimate:
    compute = rate_per_hour * hours
    ebs = volume_gb * EBS_GP3_USD_GB_MONTH / HOURS_PER_MONTH * hours
    ipv4 = PUBLIC_IPV4_USD_HOUR * hours
    return CostEstimate(hours, rate_per_hour, compute, ebs, ipv4, compute + ebs + ipv4)


def rate_for(instance_type: str, override: float | None = None) -> float | None:
    if override is not None:
        return float(override)
    spec = INSTANCE_SPECS.get(instance_type)
    return spec.usd_per_hour if spec else None


def vcpu_for(instance_type: str) -> int | None:
    spec = INSTANCE_SPECS.get(instance_type)
    return spec.vcpu if spec else None


# ---------------------------------------------------------------------------
# 設定（環境変数）と boto3
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Settings:
    access_key_id: str = dataclasses.field(repr=False)
    secret_access_key: str = dataclasses.field(repr=False)
    region: str = DEFAULT_REGION
    session_token: str | None = dataclasses.field(default=None, repr=False)

    def secrets(self) -> list[str]:
        return [s for s in (self.access_key_id, self.secret_access_key, self.session_token) if s]

    def redact(self, text: str) -> str:
        """文章の中に鍵が紛れ込んでいても伏せる。"""
        for s in self.secrets():
            if len(s) >= 8:
                text = text.replace(s, "***")
        return text


def load_settings(env: Mapping[str, str] | None = None, region: str | None = None) -> Settings:
    """EC2_ACCESS_KEY_ID / EC2_SECRET_ACCESS_KEY / EC2_REGION（任意 EC2_SESSION_TOKEN）を読む。"""
    env = os.environ if env is None else env
    ak = (env.get("EC2_ACCESS_KEY_ID") or "").strip()
    sk = (env.get("EC2_SECRET_ACCESS_KEY") or "").strip()
    missing = [n for n, v in (("EC2_ACCESS_KEY_ID", ak), ("EC2_SECRET_ACCESS_KEY", sk)) if not v]
    if missing:
        raise ToolError(
            "AWS の鍵が環境変数に設定されていません: " + ", ".join(missing) + "\n"
            "  Claude Code の環境設定（Environment variables）に次の3つを登録してください（値をチャットに貼らないこと）。\n"
            "    EC2_ACCESS_KEY_ID / EC2_SECRET_ACCESS_KEY / EC2_REGION（省略時 ap-northeast-1）\n"
            "  手順は meeting/cloud/aws/README.md の「2. 鍵を登録する」を見てください。\n"
            "  （この環境にもともとある AWS_ACCESS_KEY_ID などは仮の値なので使いません）"
        )
    for name, v in (("EC2_ACCESS_KEY_ID", ak), ("EC2_SECRET_ACCESS_KEY", sk)):
        if re.search(r"\s|[\"']", v):
            raise ToolError(f"{name} に空白や引用符が含まれています。コピーのとき余計な文字が入っていないか確認してください（値は表示しません）。")
    token = (env.get("EC2_SESSION_TOKEN") or "").strip() or None
    reg = (region or env.get("EC2_REGION") or DEFAULT_REGION).strip()
    if not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", reg):
        raise ToolError(f"リージョン名が正しくありません: {reg!r}（例: ap-northeast-1）")
    return Settings(ak, sk, reg, token)


def find_ca_bundle(env: Mapping[str, str] | None = None) -> str | None:
    """この環境の通信はプロキシ経由で、CA 証明書の束を指定する必要がある。TLS の検証は絶対に切らない。"""
    env = os.environ if env is None else env
    for name in ("AWS_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        p = env.get(name)
        if p and Path(p).is_file():
            return p
    p = "/root/.ccr/ca-bundle.crt"
    return p if Path(p).is_file() else None


def _import_boto3():
    try:
        import boto3  # noqa: WPS433 (遅延 import)
    except ImportError as e:  # pragma: no cover - 環境による
        raise ToolError(
            "boto3 が入っていません。次のように --with boto3 を付けて実行してください:\n"
            "  uv run --no-project --with boto3 python -m meeting.cloud.aws preflight\n"
            "  （meeting/pyproject.toml には boto3 を入れていません）"
        ) from e
    return boto3


class Aws:
    """boto3 の Session とクライアントをまとめて持つ。テストでは clients に偽物を差し込める。"""

    def __init__(self, settings: Settings, session: Any = None, clients: dict[str, Any] | None = None,
                 ca_bundle: str | None = None):
        self.settings = settings
        self._session = session
        self._clients: dict[str, Any] = dict(clients or {})
        self.ca_bundle = ca_bundle if ca_bundle is not None else find_ca_bundle()
        self._account: str | None = None
        self._identity: dict | None = None

    @property
    def region(self) -> str:
        return self.settings.region

    @property
    def session(self):
        if self._session is None:
            boto3 = _import_boto3()
            self._session = boto3.session.Session(
                aws_access_key_id=self.settings.access_key_id,
                aws_secret_access_key=self.settings.secret_access_key,
                aws_session_token=self.settings.session_token,
                region_name=self.settings.region,
            )
        return self._session

    def _make_client(self, service: str):
        from botocore.config import Config  # 遅延 import

        cfg = Config(retries={"max_attempts": 6, "mode": "standard"}, connect_timeout=15, read_timeout=90,
                     user_agent_extra="meeting-translator-ec2run")
        kwargs: dict[str, Any] = {"region_name": self.settings.region, "config": cfg}
        if self.ca_bundle:
            kwargs["verify"] = self.ca_bundle
        return self.session.client(service, **kwargs)

    def client(self, service: str):
        if service not in self._clients:
            self._clients[service] = self._make_client(service)
        return self._clients[service]

    @property
    def identity(self) -> dict:
        if self._identity is None:
            self._identity = self.client("sts").get_caller_identity()
        return self._identity

    @property
    def account_id(self) -> str:
        if self._account is None:
            self._account = str(self.identity["Account"])
        return self._account


def error_code(exc: BaseException) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", "")) if hasattr(exc, "response") else ""


def explain_client_error(exc: BaseException, doing: str = "") -> str:
    """boto3 の ClientError を、利用者向けの日本語にする（鍵は含めない）。"""
    code = error_code(exc)
    msg = str(getattr(exc, "response", {}).get("Error", {}).get("Message", "")) if hasattr(exc, "response") else str(exc)
    head = f"{doing}に失敗しました" if doing else "AWS の呼び出しに失敗しました"
    if code in {"InvalidClientTokenId", "AuthFailure", "SignatureDoesNotMatch", "UnrecognizedClientException",
                "InvalidAccessKeyId"}:
        hint = "鍵が無効です。EC2_ACCESS_KEY_ID / EC2_SECRET_ACCESS_KEY を確認してください（この環境にある AWS_* の仮の値ではなく EC2_* を使います）。"
    elif code in {"ExpiredToken", "ExpiredTokenException", "RequestExpired"}:
        hint = "鍵（セッショントークン）の期限が切れています。新しい鍵に取り替えてください。"
    elif code in {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation", "UnauthorizedAccess"}:
        hint = ("権限が足りません。README の IAM ポリシー（iam_user_policy.json）が、このユーザーに付いているか確認してください。"
                "どの操作が拒否されたかはメッセージに出ます。")
    elif code in {"VcpuLimitExceeded", "InstanceLimitExceeded", "MaxSpotInstanceCountExceeded"}:
        hint = "vCPU の枠（クォータ）が足りません。`preflight` の案内に従って Service Quotas で引き上げを申請してください。"
    elif code == "InsufficientInstanceCapacity":
        hint = "いま AWS 側にこのインスタンスの空きがありません。時間をおくか、別のタイプ（--type g5.xlarge）を試してください。"
    else:
        hint = ""
    parts = [f"{head}（{code or type(exc).__name__}）"]
    if msg:
        parts.append(f"  AWS のメッセージ: {msg}")
    if hint:
        parts.append(f"  → {hint}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 名前・タグ
# ---------------------------------------------------------------------------
def bucket_name(account_id: str, region: str) -> str:
    """バケット名は世界で一意。アカウント ID は名前に出さず、ハッシュで決める（同じ入力なら必ず同じ名前）。
    状態ファイルが消えても同じバケットを見つけられる。"""
    h = hashlib.sha256(f"meeting-translator:{account_id}:{region}".encode()).hexdigest()[:10]
    return f"meeting-translator-{h}-{region}"


def base_tags(extra: Mapping[str, str] | None = None) -> list[dict[str, str]]:
    tags = {PROJECT_TAG_KEY: PROJECT_TAG_VALUE, TOOL_TAG_KEY: TOOL_TAG_VALUE}
    tags.update(extra or {})
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def tag_spec(resource_type: str, extra: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {"ResourceType": resource_type, "Tags": base_tags(extra)}


def tags_to_dict(tags: Iterable[Mapping[str, str]] | None) -> dict[str, str]:
    return {t["Key"]: t["Value"] for t in (tags or [])}


def our_tag_filters() -> list[dict[str, Any]]:
    return [
        {"Name": f"tag:{PROJECT_TAG_KEY}", "Values": [PROJECT_TAG_VALUE]},
        {"Name": f"tag:{TOOL_TAG_KEY}", "Values": [TOOL_TAG_VALUE]},
    ]


def is_ours(tags: Mapping[str, str]) -> bool:
    return tags.get(PROJECT_TAG_KEY) == PROJECT_TAG_VALUE and tags.get(TOOL_TAG_KEY) == TOOL_TAG_VALUE


# ---------------------------------------------------------------------------
# 状態ファイル（作ったリソースの ID を覚える。Git に入れない）
# ---------------------------------------------------------------------------
def state_path(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get(STATE_ENV)
    return Path(env) if env else DEFAULT_STATE_PATH


class State:
    """meeting/cloud/aws/.state.json。壊れていたら退避して空から始める（止まらない）。"""

    VERSION = 1

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = state_path(path)
        self.data: dict[str, Any] = {"version": self.VERSION}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not an object")
            self.data = data
            self.data.setdefault("version", self.VERSION)
        except (OSError, ValueError) as e:
            backup = self.path.with_name(f"{self.path.name}.corrupt-{utcnow().strftime('%Y%m%d%H%M%S')}")
            try:
                self.path.replace(backup)
                where = f"{backup.name} に退避しました"
            except OSError:
                where = "退避できませんでした"
            warn(f"状態ファイル {self.path.name} を読めませんでした（{e}）。{where}。空の状態から始めます。")
            self.data = {"version": self.VERSION}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # --- 取り出し・書き込みの補助 ---
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, **kwargs: Any) -> None:
        self.data.update(kwargs)
        self.save()

    def instances(self) -> dict[str, dict[str, Any]]:
        return self.data.setdefault("instances", {})

    def record_instance(self, instance_id: str, **fields: Any) -> None:
        self.instances().setdefault(instance_id, {}).update(fields)
        self.save()

    def runs(self) -> list[dict[str, Any]]:
        return self.data.setdefault("runs", [])

    def record_run(self, exp: str, **fields: Any) -> dict[str, Any]:
        """exp 名ごとに1件。同じ exp の再実行は上書き。"""
        runs = self.runs()
        for r in runs:
            if r.get("exp") == exp:
                r.update(fields)
                self.save()
                return r
        r = {"exp": exp, **fields}
        runs.append(r)
        self.save()
        return r

    def last_run(self, exp: str) -> dict[str, Any] | None:
        for r in reversed(self.runs()):
            if r.get("exp") == exp:
                return r
        return None


def short_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)
