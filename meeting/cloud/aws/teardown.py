"""status / teardown: 動いているものを確認し、止めて、残りものと費用の見込みを報告する。

触ってよいのは、タグ Project=meeting-translator かつ ManagedBy=ec2run の両方が付いたものだけ。
他のインスタンスやバケットは、名前が似ていても触らない。
"""
from __future__ import annotations

import time
from typing import Any, Callable

from .common import (
    EBS_GP3_USD_GB_MONTH,
    S3_STANDARD_USD_GB_MONTH,
    Aws,
    State,
    ToolError,
    bucket_name,
    error_code,
    explain_client_error,
    fmt_usd,
    is_ours,
    iso,
    our_tag_filters,
    require_yes,
    say,
    tags_to_dict,
    utcnow,
    warn,
)
from .instances import LIVE_STATES, describe_line, find_instances
from .provision import delete_setup_resources
from .runner import s3_delete_prefix


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def bucket_usage(aws: Aws, bucket: str, limit_keys: int = 20000) -> tuple[int, int, bool]:
    """(オブジェクト数, 合計バイト, 途中で打ち切ったか)"""
    s3 = aws.client("s3")
    n = size = 0
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            n += 1
            size += int(o["Size"])
        if not resp.get("IsTruncated"):
            return n, size, False
        if n >= limit_keys:
            return n, size, True
        token = resp.get("NextContinuationToken")


def run_status(aws: Aws, state: State, args: Any) -> int:
    say("== status ==")
    say(f"リージョン: {aws.region}")
    try:
        infos = find_instances(aws)
    except Exception as e:  # noqa: BLE001
        raise ToolError(explain_client_error(e, "インスタンスの一覧") if hasattr(e, "response") else str(e)) from e
    live = [i for i in infos if i.state in LIVE_STATES]
    recent = [i for i in infos if i.state not in LIVE_STATES][-5:]
    say("\n[インスタンス（このツールが起動したもの）]")
    if live:
        for i in live:
            say(describe_line(i))
    else:
        say("  動いているものはありません")
    if recent:
        say("  最近終了したもの:")
        for i in recent:
            say("  " + describe_line(i).strip())
    over = [i for i in live if i.expires_at and i.expires_at < utcnow() and i.state == "running"]
    if over:
        warn("自動停止の予定時刻を過ぎても動いているものがあります。`teardown --yes` で止めてください。")
    if live:
        warn("動いている間は課金されます。使い終わったら `teardown --yes`。")

    bucket = state.get("bucket") or bucket_name(aws.account_id, aws.region)
    say("\n[S3 バケット]")
    try:
        n, size, cut = bucket_usage(aws, bucket)
        say(f"  {bucket}: {n}{'+' if cut else ''} オブジェクト / {size / 1e6:.1f} MB"
            f"（保管の目安 {fmt_usd(size / 1e9 * S3_STANDARD_USD_GB_MONTH)}/月。14 日で自動削除）")
    except Exception as e:  # noqa: BLE001
        code = error_code(e)
        say("  バケットはありません（setup 前）" if code in ("NoSuchBucket", "404", "NotFound") else f"  確認できませんでした: {code or e}")

    runs = state.runs()
    if runs:
        say("\n[実験の記録（状態ファイル）]")
        for r in runs[-5:]:
            say(f"  {r.get('exp')}: {r.get('status', '?')} / エンジン {','.join(r.get('engines', []))} / 結果 {r.get('out_dir', '-')}")
    return 0


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------
def select_targets(aws: Aws, state: State, all_tagged: bool):
    """終了させる候補。通常は状態ファイルに記録のあるもの。--all ならタグで探したもの全部。どちらもタグを確かめる。"""
    live = find_instances(aws, states=LIVE_STATES)
    if all_tagged:
        return live, live
    recorded = set(state.instances())
    return [i for i in live if i.instance_id in recorded], live


def wait_terminated(aws: Aws, ids: list[str], *, timeout: float = 600, poll: float = 10,
                    sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic) -> dict[str, str]:
    """全部 terminated になるまで待ち、{ID: 最後の状態} を返す。"""
    ec2 = aws.client("ec2")
    start = monotonic()
    states: dict[str, str] = {}
    while True:
        resp = ec2.describe_instances(InstanceIds=ids)
        states = {i["InstanceId"]: i["State"]["Name"] for r in resp["Reservations"] for i in r["Instances"]}
        if all(s == "terminated" for s in states.values()) and len(states) == len(ids):
            return states
        if monotonic() - start > timeout:
            return states
        sleep(poll)


def leftover_report(aws: Aws, bucket: str, state: State) -> list[str]:
    """残っている資源と、その費用の見込み。"""
    lines: list[str] = []
    ec2 = aws.client("ec2")
    try:
        vols = ec2.describe_volumes(Filters=our_tag_filters())["Volumes"]
    except Exception as e:  # noqa: BLE001
        vols = []
        lines.append(f"EBS ボリュームを確認できませんでした: {error_code(e) or e}")
    keep = [v for v in vols if is_ours(tags_to_dict(v.get("Tags")))]
    if keep:
        total = sum(v["Size"] for v in keep)
        lines.append(f"EBS ボリューム {len(keep)} 個（計 {total} GB）が残っています → 約 {fmt_usd(total * EBS_GP3_USD_GB_MONTH)}/月 かかり続けます。"
                     "不要なら削除してください: " + ", ".join(f"{v['VolumeId']}({v['State']})" for v in keep))
    else:
        lines.append("EBS ボリューム: 残っていません（インスタンス終了で削除されました）")
    try:
        n, size, cut = bucket_usage(aws, bucket)
        lines.append(f"S3 バケット {bucket}: {n}{'+' if cut else ''} オブジェクト / {size / 1e6:.1f} MB → 約 {fmt_usd(size / 1e9 * S3_STANDARD_USD_GB_MONTH)}/月"
                     "（14 日で自動削除。今すぐ消すなら --delete-bucket）")
    except Exception as e:  # noqa: BLE001
        if error_code(e) in ("NoSuchBucket", "404", "NotFound"):
            lines.append("S3 バケット: ありません")
        else:
            lines.append(f"S3 バケットを確認できませんでした: {error_code(e) or e}")
    lines.append("IAM ロール・インスタンスプロファイル・セキュリティグループ: 無料です（消すなら --delete-setup）")
    return lines


def delete_bucket(aws: Aws, bucket: str) -> int | None:
    """中身を空にしてからバケットを消す。バケットが無ければ None。"""
    try:
        n = s3_delete_prefix(aws, bucket, "")
        aws.client("s3").delete_bucket(Bucket=bucket)
    except Exception as e:  # noqa: BLE001
        if error_code(e) in ("NoSuchBucket", "404", "NotFound"):
            return None
        raise
    return n


def run_teardown(aws: Aws, state: State, args: Any, *, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> int:
    say("== teardown ==")
    targets, all_live = select_targets(aws, state, bool(args.all))
    bucket = state.get("bucket") or bucket_name(aws.account_id, aws.region)
    if targets:
        say("終了させるインスタンス（タグ Project=meeting-translator / ManagedBy=ec2run のもの）:")
        for i in targets:
            say(describe_line(i))
    else:
        say("終了させるインスタンスはありません。")
    others = [i for i in all_live if i not in targets]
    if others:
        say("  （状態ファイルに記録が無いので、今回は触りません。これも終了するなら --all を付ける）:")
        for i in others:
            say("  " + describe_line(i).strip())
    plan_extra = []
    if args.delete_bucket:
        plan_extra.append(f"S3 バケット {bucket} を、中身ごと削除")
    if args.delete_setup:
        plan_extra.append("IAM ロール・インスタンスプロファイル・セキュリティグループを削除")
    for p in plan_extra:
        say(f"追加で行う: {p}")

    if args.delete_bucket and args.confirm_bucket != bucket:
        raise ToolError(f"バケットの削除は確認が要ります。--confirm-bucket {bucket} を付けてください（実験の結果を回収済みか確かめてから）。")
    if targets or plan_extra:
        require_yes(args, "上のものを終了／削除します。")

    if targets:
        ids = [i.instance_id for i in targets]
        # 念のため終了の直前にもタグを確かめ直す（他人のインスタンスを止めない）
        recheck = {i.instance_id for i in find_instances(aws, states=LIVE_STATES, instance_ids=ids)}
        ids = [i for i in ids if i in recheck]
        if ids:
            try:
                aws.client("ec2").terminate_instances(InstanceIds=ids)
            except Exception as e:  # noqa: BLE001
                raise ToolError(explain_client_error(e, "インスタンスの終了") if hasattr(e, "response") else str(e)) from e
            say(f"終了を依頼しました: {', '.join(ids)}。終了するのを待っています…")
            states = wait_terminated(aws, ids, sleep=sleep, monotonic=monotonic)
            for iid in ids:
                s = states.get(iid, "?")
                say(f"  {iid}: {s}{'  ← まだ終わっていません。`status` で確認してください' if s != 'terminated' else ''}")
                if iid in state.instances():
                    state.record_instance(iid, state=s, terminated_at=iso(utcnow()))
            if all(states.get(i) == "terminated" for i in ids):
                say("確認しました: すべて terminated です（課金は止まりました）。")
            else:
                warn("terminated を確認できないものがあります。")

    if args.delete_bucket:
        n = delete_bucket(aws, bucket)
        say(f"S3 バケット {bucket} を削除しました（オブジェクト {n} 件）" if n is not None else f"S3 バケット {bucket} は、もうありません")
        state.data.pop("bucket", None)
        state.save()
    if args.delete_setup:
        done = delete_setup_resources(aws)
        for d in done:
            say(f"  削除しました: {d}")
        for k in ("role_name", "instance_profile", "security_group_id", "vpc_id", "setup_at"):
            state.data.pop(k, None)
        state.save()

    say("\n[残っているもの]")
    for ln in leftover_report(aws, bucket, state):
        say("  " + ln)
    return 0
