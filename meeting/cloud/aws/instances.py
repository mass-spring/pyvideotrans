"""このツールが起動したインスタンスの検索と表示。

触ってよいのは「Project=meeting-translator かつ ManagedBy=ec2run」の両方のタグが付いたものだけ。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

from .common import (
    INSTANCE_SPECS,
    Aws,
    estimate_cost,
    fmt_hms,
    fmt_usd,
    is_ours,
    our_tag_filters,
    parse_iso,
    rate_for,
    tags_to_dict,
    utcnow,
)

LIVE_STATES = ["pending", "running", "stopping", "stopped", "shutting-down"]
RUNNING_STATES = ["pending", "running"]


@dataclasses.dataclass
class InstanceInfo:
    instance_id: str
    instance_type: str
    state: str
    az: str
    launch_time: dt.datetime | None
    spot: bool
    tags: dict[str, str]

    @property
    def hours(self) -> float | None:
        try:
            return float(self.tags.get("Hours", ""))
        except ValueError:
            return None

    @property
    def expires_at(self) -> dt.datetime | None:
        return parse_iso(self.tags.get("ExpiresAt"))

    def elapsed_seconds(self, now: dt.datetime | None = None) -> float:
        if not self.launch_time:
            return 0.0
        return max(0.0, ((now or utcnow()) - self.launch_time).total_seconds())

    def hourly_rate(self) -> float | None:
        override = self.tags.get("HourlyRate")
        if override:
            try:
                return float(override)
            except ValueError:
                pass
        return rate_for(self.instance_type)


def _to_info(i: dict[str, Any]) -> InstanceInfo:
    lt = i.get("LaunchTime")
    if isinstance(lt, dt.datetime) and lt.tzinfo is None:
        lt = lt.replace(tzinfo=dt.timezone.utc)
    return InstanceInfo(
        instance_id=i["InstanceId"],
        instance_type=i.get("InstanceType", "?"),
        state=i.get("State", {}).get("Name", "?"),
        az=i.get("Placement", {}).get("AvailabilityZone", "?"),
        launch_time=lt,
        spot=i.get("InstanceLifecycle") == "spot",
        tags=tags_to_dict(i.get("Tags")),
    )


def find_instances(aws: Aws, states: list[str] | None = None, instance_ids: list[str] | None = None) -> list[InstanceInfo]:
    """タグ（Project と ManagedBy の両方）で探す。states を省略すると終了済みを含む全部。"""
    ec2 = aws.client("ec2")
    filters = our_tag_filters()
    if states:
        filters.append({"Name": "instance-state-name", "Values": list(states)})
    kwargs: dict[str, Any] = {"Filters": filters}
    if instance_ids:
        kwargs["InstanceIds"] = list(instance_ids)
    out: list[InstanceInfo] = []
    token = None
    while True:
        if token:
            kwargs["NextToken"] = token
        resp = ec2.describe_instances(**kwargs)
        for r in resp.get("Reservations", []):
            for i in r.get("Instances", []):
                info = _to_info(i)
                if is_ours(info.tags):      # 念のため、返ってきたものも自分で確かめる
                    out.append(info)
        token = resp.get("NextToken")
        if not token:
            break
    out.sort(key=lambda x: (x.launch_time or dt.datetime.min.replace(tzinfo=dt.timezone.utc), x.instance_id))
    return out


def describe_line(info: InstanceInfo, now: dt.datetime | None = None) -> str:
    """1行の説明: ID・タイプ・状態・経過時間・費用の見積・自動停止までの残り。"""
    now = now or utcnow()
    el = info.elapsed_seconds(now)
    parts = [f"{info.instance_id}", f"{info.instance_type}{' (spot)' if info.spot else ''}", info.state, info.az]
    if info.state in ("running", "pending"):
        rate = info.hourly_rate()
        parts.append(f"経過 {fmt_hms(el)}")
        if rate is not None:
            parts.append(f"ここまでの費用の目安 {fmt_usd(estimate_cost(rate, el / 3600).total)}")
        exp = info.expires_at
        if exp:
            left = (exp - now).total_seconds()
            parts.append(f"自動停止まで {fmt_hms(left)}" if left > 0 else f"自動停止の予定時刻を {fmt_hms(-left)} 過ぎています（要確認）")
    return "  " + " / ".join(parts)


def known_types() -> str:
    return ", ".join(INSTANCE_SPECS)
