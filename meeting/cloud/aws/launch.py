"""launch: GPU インスタンスを1台起動し、SSM が使えるようになるまで待つ。

費用の上限を二重にかけている:
  1. user-data で `shutdown -h +<分>` を予約する
  2. InstanceInitiatedShutdownBehavior=terminate なので、OS の停止 = インスタンスの終了（ディスクも削除）になる
さらに起動後、SSM 経由で「停止の予約が本当に入っているか」を確かめ、入っていなければ入れ直す。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import time
import uuid
from typing import Any, Callable

from .common import (
    DEFAULT_VOLUME_GB,
    INSTANCE_SPECS,
    MAX_HOURS,
    PRICE_NOTE,
    Aws,
    State,
    ToolError,
    error_code,
    estimate_cost,
    explain_client_error,
    fmt_hms,
    fmt_usd,
    iso,
    rate_for,
    require_yes,
    say,
    tag_spec,
    utcnow,
    warn,
)
from .instances import RUNNING_STATES, describe_line, find_instances
from .preflight import check_ready_for_launch, offered_azs
from .provision import resolve_resources
from .ssmrun import run_quick, wait_ssm_online

# AWS Deep Learning Base GPU AMI（OSS Nvidia ドライバ）の最新 AMI ID を返す公開 SSM パラメータ。
# 出典（Ubuntu 22.04 / 24.04 それぞれのリリースノート「SSM Parameter Query」）:
#   https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-22-04.html
#   https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html
#   形式の説明: https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html
# 対応インスタンスに G4dn・G5・G6 を含む。SSM エージェント入り、既定のログインユーザーは ubuntu。
DLAMI_PARAM_TEMPLATE = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-{ver}/latest/ami-id"
OS_VERSIONS = ("22.04", "24.04")

CAPACITY_ERRORS = {"InsufficientInstanceCapacity", "Unsupported", "SpotMaxPriceTooLow", "InsufficientFreeAddressesInSubnet"}
SHUTDOWN_FILE = "/run/systemd/shutdown/scheduled"


def ami_parameter(os_version: str) -> str:
    if os_version not in OS_VERSIONS:
        raise ToolError(f"--os-version は {', '.join(OS_VERSIONS)} のどちらかにしてください: {os_version}")
    return DLAMI_PARAM_TEMPLATE.format(ver=os_version)


@dataclasses.dataclass
class AmiInfo:
    image_id: str
    name: str
    root_device: str
    parameter: str


def resolve_ami(aws: Aws, os_version: str) -> AmiInfo:
    param = ami_parameter(os_version)
    try:
        value = aws.client("ssm").get_parameter(Name=param)["Parameter"]["Value"]
    except Exception as e:  # noqa: BLE001
        if error_code(e) == "ParameterNotFound":
            raise ToolError(f"AMI の公開パラメータが見つかりません: {param}\n"
                            f"  このリージョン（{aws.region}）に無いか、AWS 側で名前が変わった可能性があります。"
                            "  https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html で最新の名前を確認してください。") from e
        raise ToolError(explain_client_error(e, "AMI の検索（SSM パラメータ）")) from e
    try:
        img = aws.client("ec2").describe_images(ImageIds=[value])["Images"][0]
    except Exception as e:  # noqa: BLE001
        raise ToolError(explain_client_error(e, f"AMI {value} の確認") if hasattr(e, "response") else f"AMI {value} を確認できません") from e
    return AmiInfo(value, img.get("Name", "?"), img.get("RootDeviceName", "/dev/sda1"), param)


def build_user_data(hours: float) -> str:
    """起動時に走る。hours 時間後に OS を停止する予約を入れる（= 費用の上限）。"""
    minutes = max(1, int(round(hours * 60)))
    return (
        "#!/bin/bash\n"
        "# meeting-translator ec2run: 費用の上限。時間が来たら OS を停止する。\n"
        "# InstanceInitiatedShutdownBehavior=terminate なので、停止 = インスタンスの終了（ディスクも削除）になる。\n"
        "mkdir -p /var/log/ec2run\n"
        f"echo \"hours={hours} minutes={minutes}\" > /var/log/ec2run/schedule.txt\n"
        f"shutdown -h +{minutes} \"ec2run: auto shutdown after {minutes} min (cost cap)\" >> /var/log/ec2run/schedule.txt 2>&1\n"
    )


def build_run_params(*, ami: AmiInfo, instance_type: str, spot: bool, hours: float, subnet_id: str, sg_id: str,
                     profile_name: str, volume_gb: int, now: dt.datetime, rate: float | None) -> dict[str, Any]:
    """RunInstances に渡す引数。テストで1つずつ確かめられるよう、通信しない純粋な関数にしてある。"""
    expires = now + dt.timedelta(hours=hours)
    extra = {"Name": f"meeting-translator-{instance_type}", "Hours": f"{hours:g}", "ExpiresAt": iso(expires),
             "LaunchedAt": iso(now), "Purchase": "spot" if spot else "on-demand"}
    if rate is not None:
        extra["HourlyRate"] = f"{rate:g}"
    specs = [tag_spec("instance", extra), tag_spec("volume", {"Name": extra["Name"]}),
             tag_spec("network-interface", {"Name": extra["Name"]})]
    params: dict[str, Any] = {
        "ImageId": ami.image_id,
        "InstanceType": instance_type,
        "MinCount": 1, "MaxCount": 1,
        "IamInstanceProfile": {"Name": profile_name},
        "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet_id, "Groups": [sg_id],
                               "AssociatePublicIpAddress": True, "DeleteOnTermination": True}],
        "BlockDeviceMappings": [{"DeviceName": ami.root_device,
                                 "Ebs": {"VolumeSize": int(volume_gb), "VolumeType": "gp3", "DeleteOnTermination": True}}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled", "HttpPutResponseHopLimit": 1},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "UserData": build_user_data(hours),
        "TagSpecifications": specs,
    }
    if spot:
        params["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
            "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
        params["TagSpecifications"].append(tag_spec("spot-instances-request", {"Name": extra["Name"]}))
    return params


def validate_launch_args(args: Any) -> tuple[str, float, float | None]:
    itype = args.type
    hours = float(args.hours)
    if not (0 < hours <= MAX_HOURS):
        raise ToolError(f"--hours は 0 より大きく {MAX_HOURS:g} 以下にしてください（費用の上限を決める数字です）: {args.hours}")
    rate = rate_for(itype, getattr(args, "hourly_rate", None))
    if rate is None:
        raise ToolError(f"{itype} の料金が表にありません。表にあるタイプ: {', '.join(INSTANCE_SPECS)}。"
                        "別のタイプを使うなら --hourly-rate <USD/時間> を付けてください（見積に使います）。")
    return itype, hours, rate


def default_subnets(aws: Aws, vpc_id: str) -> list[dict[str, str]]:
    resp = aws.client("ec2").describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "default-for-az", "Values": ["true"]}])
    subs = [{"SubnetId": s["SubnetId"], "AZ": s["AvailabilityZone"]} for s in resp.get("Subnets", [])]
    return sorted(subs, key=lambda s: s["AZ"])


def _is_profile_propagation(e: Exception) -> bool:
    msg = str(getattr(e, "response", {}).get("Error", {}).get("Message", ""))
    low = msg.lower()
    return error_code(e) in {"InvalidParameterValue", "InvalidParameter"} and ("instance profile" in low or "iaminstanceprofile" in low)


def launch_instance(aws: Aws, params_for_subnet: Callable[[str], dict[str, Any]], subnets: list[dict[str, str]],
                    sleep: Callable[[float], None] = time.sleep) -> tuple[str, str]:
    """サブネット（AZ）を順に試して RunInstances する。(instance_id, az) を返す。"""
    ec2 = aws.client("ec2")
    last: Exception | None = None
    for sub in subnets:
        for attempt in range(8):      # 作りたてのインスタンスプロファイルは反映に時間がかかることがある
            params = params_for_subnet(sub["SubnetId"])
            params["ClientToken"] = uuid.uuid4().hex
            try:
                resp = ec2.run_instances(**params)
                return resp["Instances"][0]["InstanceId"], sub["AZ"]
            except Exception as e:  # noqa: BLE001
                if not hasattr(e, "response"):
                    raise
                last = e
                if _is_profile_propagation(e) and attempt < 7:
                    say(f"  インスタンスプロファイルの反映待ち… ({attempt + 1}/7)")
                    sleep(10)
                    continue
                if error_code(e) in CAPACITY_ERRORS:
                    warn(f"{sub['AZ']} では起動できませんでした（{error_code(e)}）。次の AZ を試します。")
                    break
                raise ToolError(explain_client_error(e, "インスタンスの起動")) from e
    raise ToolError((explain_client_error(last, "インスタンスの起動") if last is not None else "起動できる AZ がありません")
                    + "\n  すべての AZ で起動できませんでした。時間をおく／別のタイプ（--type g5.xlarge）を試す／--spot を外す、のいずれかを試してください。")


def verify_auto_shutdown(aws: Aws, instance_id: str, remaining_minutes: int) -> bool:
    """停止の予約が実際に入っているかを SSM で確かめ、入っていなければ入れ直す。確認できたら True。"""
    # SSM が Online になっても、user-data（停止の予約）がまだ走っていないことがある。cloud-init の完了を待ってから確認する
    cmd = ["timeout 300 cloud-init status --wait > /dev/null 2>&1 || true",
           f"if [ -e {SHUTDOWN_FILE} ]; then echo SCHEDULED; else echo NOT_SCHEDULED; fi",
           "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2",
           "df -h / | tail -1"]
    status, out, _err = run_quick(aws, instance_id, cmd, comment="ec2run check", timeout=420)
    if status != "Success":
        warn(f"起動後の確認コマンドが成功しませんでした（{status}）。`status` で状態を確認してください。")
        return False
    first = out.strip().splitlines()[0] if out.strip() else ""
    for ln in out.strip().splitlines()[1:]:
        say(f"  {ln}")
    if first == "SCHEDULED":
        return True
    warn("自動停止の予約が見つかりません。SSM で予約し直します。")
    mins = max(1, remaining_minutes)
    status, out2, _ = run_quick(aws, instance_id, [f"shutdown -h +{mins} 'ec2run: re-scheduled (cost cap)'", f"[ -e {SHUTDOWN_FILE} ] && echo SCHEDULED"],
                                comment="ec2run reschedule", timeout=120)
    ok = status == "Success" and "SCHEDULED" in out2
    if not ok:
        warn("自動停止を予約できませんでした！ すぐに `teardown --yes` で止めるか、時間を決めて手動で止めてください。")
    return ok


def run_launch(aws: Aws, state: State, args: Any, *, sleep: Callable[[float], None] = time.sleep,
               monotonic: Callable[[], float] = time.monotonic, now_fn: Callable[[], dt.datetime] = utcnow) -> int:
    itype, hours, rate = validate_launch_args(args)
    spot = bool(args.spot)
    volume_gb = int(getattr(args, "volume_gb", DEFAULT_VOLUME_GB))
    est = estimate_cost(rate, hours, volume_gb)
    spec = INSTANCE_SPECS.get(itype)

    say("== launch: GPU インスタンスの起動 ==")
    say(f"  タイプ: {itype}{f'（{spec.gpu}, {spec.vcpu} vCPU）' if spec else ''} / 購入方法: {'スポット' if spot else 'オンデマンド'} / リージョン: {aws.region}")
    say(f"  ディスク: gp3 {volume_gb}GB（終了時に削除）/ IMDSv2 必須 / インバウンド 0 / SSH なし（SSM で操作）")
    say(f"  自動停止: {hours:g} 時間後に OS を停止 → インスタンスごと終了（費用の上限）")
    say(f"  費用の上限の見積: {fmt_usd(est.total)}"
        f"（本体 {fmt_usd(est.compute)} = {rate:g}$/時間 × {hours:g}時間 ＋ ディスク {fmt_usd(est.ebs)} ＋ 公開IPv4 {fmt_usd(est.ipv4)}）")
    if spot:
        say("  ※スポットは空きがあれば安くなりますが、上の上限はオンデマンド料金で計算しています。AWS 側の都合で止められることがあります。")
    say(f"  {PRICE_NOTE}")
    require_yes(args, f"{itype} を最大 {hours:g} 時間、上限 約{fmt_usd(est.total)} で起動します。")

    res = resolve_resources(aws, state)
    live = find_instances(aws, states=RUNNING_STATES + ["stopping", "stopped"])
    if live and not getattr(args, "allow_multiple", False):
        lines = "\n".join(describe_line(i) for i in live)
        raise ToolError(f"このツールで起動したインスタンスがすでに {len(live)} 台あります（二重に課金されます）:\n{lines}\n"
                        "  使うなら `run --instance-id ...`、不要なら `teardown --yes`。本当にもう1台起動するなら --allow-multiple を付けてください。")
    check_ready_for_launch(aws, itype, spot)

    ami = resolve_ami(aws, args.os_version)
    say(f"  AMI: {ami.image_id}（{ami.name}）")
    azs = offered_azs(aws, itype)
    subnets = [s for s in default_subnets(aws, res["vpc_id"]) if s["AZ"] in azs] if res.get("vpc_id") else []
    if not subnets:
        raise ToolError(f"{itype} を提供している AZ にデフォルトサブネットがありません（提供 AZ: {', '.join(azs) or 'なし'}）。")

    now = now_fn()

    def params_for(subnet_id: str) -> dict[str, Any]:
        return build_run_params(ami=ami, instance_type=itype, spot=spot, hours=hours, subnet_id=subnet_id,
                                sg_id=res["security_group_id"], profile_name=res["instance_profile"],
                                volume_gb=volume_gb, now=now, rate=rate)

    say("  起動を依頼します…")
    instance_id, az = launch_instance(aws, params_for, subnets, sleep=sleep)
    expires = now + dt.timedelta(hours=hours)
    state.record_instance(instance_id, type=itype, spot=spot, hours=hours, hourly_rate=rate, launched_at=iso(now),
                          expires_at=iso(expires), az=az, ami=ami.image_id, state="launched", volume_gb=volume_gb)
    state.set(last_instance_id=instance_id)
    say(f"  起動を受け付けました: {instance_id}（{az}）。{iso(expires)} ごろに自動で終了します。")

    if getattr(args, "no_wait", False):
        say("  --no-wait のため、ここで終わります。`status` で状態を確認してください。")
        return 0

    say("  running になるのを待っています…")
    try:
        ec2 = aws.client("ec2")
        ec2.get_waiter("instance_exists").wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 2, "MaxAttempts": 30})   # 依頼直後は一覧に出ないことがある
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60})
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"{instance_id} が running になりませんでした: {type(e).__name__}。`status` で確認し、不要なら `teardown --yes`。") from e

    say("  SSM エージェントが Online になるのを待っています（通常 2〜4 分）…")
    ticks: list[int] = []

    def on_wait(elapsed: float) -> None:
        if int(elapsed) // 30 > (ticks[-1] if ticks else -1):
            ticks.append(int(elapsed) // 30)
            say(f"    …{int(elapsed)} 秒")

    ok = wait_ssm_online(aws, instance_id, timeout=float(getattr(args, "ssm_timeout", 600)), sleep=sleep,
                         monotonic=monotonic, on_wait=on_wait)
    if not ok:
        warn("SSM が Online になりませんでした。考えられる原因: インスタンスプロファイルの反映待ち／公開 IP が付いていない／ロールの権限不足。")
        warn(f"少し待ってから `status` を確認してください。使わないなら `teardown --yes`（自動停止は {hours:g} 時間後に効きます）。")
        state.record_instance(instance_id, state="running-no-ssm")
        return 1
    remaining = int((expires - now_fn()).total_seconds() // 60)
    shut_ok = verify_auto_shutdown(aws, instance_id, remaining)
    state.record_instance(instance_id, state="ready", auto_shutdown_verified=bool(shut_ok), ready_at=iso(now_fn()))
    say("")
    say(f"準備できました: {instance_id}（{itype}, {az}）")
    say(f"  自動停止の予約: {'確認できました' if shut_ok else '確認できていません（上の警告を見てください）'} / 残り約 {fmt_hms(remaining * 60)}")
    say("次: python -m meeting.cloud.aws run --exp exp01 --engines voxcpm2,qwen3tts,chatterbox --public-test --dry-run-first --yes")
    say("   （--public-test は公開データだけで動かす最初の確認。自分の声を使うときは --speaker 名前=フォルダ）")
    return 0
