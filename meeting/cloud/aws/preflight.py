"""preflight: 何も作らず、AWS で GPU を動かせる状態かを確かめる（読み取りだけ）。

確認すること:
  1. 鍵が有効か（STS）。アカウント ID は下4桁以外を伏せて表示
  2. GPU インスタンスの vCPU の枠（Service Quotas）。オンデマンド L-DB2E81BA とスポット L-3819A6DF
  3. 使いたいインスタンスタイプが、そのリージョン／AZ で提供されているか
  4. このツールで起動中のインスタンス
  5. 料金の目安（概算の固定表）
  6. setup が済んでいるか
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .common import (
    INSTANCE_SPECS,
    PRICE_NOTE,
    PRICE_REGION,
    QUOTA_NAMES,
    QUOTA_ONDEMAND_G,
    QUOTA_SERVICE,
    QUOTA_SPOT_G,
    Aws,
    ToolError,
    error_code,
    estimate_cost,
    explain_client_error,
    fmt_usd,
    mask_account,
    mask_text,
    say,
    vcpu_for,
    warn,
)
from .instances import LIVE_STATES, describe_line, find_instances
from .provision import discover


@dataclasses.dataclass
class QuotaResult:
    code: str
    name: str
    value: float | None = None
    source: str = ""            # "適用値" / "既定値"
    error: str | None = None
    pending: list[str] = dataclasses.field(default_factory=list)   # 申請中の希望値など


def fetch_quota(aws: Aws, code: str) -> QuotaResult:
    """適用中の値を取り、無ければ AWS の既定値を取る。どちらも取れなければ error に理由を入れる。"""
    res = QuotaResult(code=code, name=QUOTA_NAMES.get(code, code))
    sq = aws.client("service-quotas")
    try:
        res.value = float(sq.get_service_quota(ServiceCode=QUOTA_SERVICE, QuotaCode=code)["Quota"]["Value"])
        res.source = "適用値"
    except Exception as e:  # noqa: BLE001
        if error_code(e) == "NoSuchResourceException":
            try:
                res.value = float(sq.get_aws_default_service_quota(ServiceCode=QUOTA_SERVICE, QuotaCode=code)["Quota"]["Value"])
                res.source = "既定値"
            except Exception as e2:  # noqa: BLE001
                res.error = explain_client_error(e2, "枠（既定値）の取得") if hasattr(e2, "response") else str(e2)
        else:
            res.error = explain_client_error(e, "枠の取得") if hasattr(e, "response") else str(e)
    try:
        hist = sq.list_requested_service_quota_change_history_by_quota(ServiceCode=QUOTA_SERVICE, QuotaCode=code)
        for r in hist.get("RequestedQuotas", []):
            if r.get("Status") in ("PENDING", "CASE_OPENED"):
                res.pending.append(f"{r.get('DesiredValue'):g} vCPU（{r.get('Status')}）")
    except Exception:  # noqa: BLE001  (権限が無くても preflight は続ける)
        pass
    return res


def quota_console_url(region: str, code: str) -> str:
    return f"https://{region}.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/{code}"


def quota_guidance(q: QuotaResult, need: int, region: str) -> list[str]:
    """枠が足りないときの手順（日本語）。"""
    lines = [
        f"  ▶ 「{q.name}」({q.code}) が {q.value:g} vCPU で、必要な {need} vCPU に足りません。",
    ]
    if q.pending:
        lines.append(f"    すでに引き上げを申請中です: {', '.join(q.pending)}。承認の連絡（メール）を待ってください（数分〜数日）。")
        return lines
    lines += [
        "    新しいアカウントでは GPU（G インスタンス）の枠が 0 のことが多く、そのままでは起動できません。引き上げを申請します。",
        f"    1) 画面右上のリージョンを「{region}」にする（枠はリージョンごと）",
        "    2) AWS コンソール → Service Quotas → AWS のサービス → Amazon EC2 → "
        f"「{q.name}」を開く",
        f"       直接のリンク: {quota_console_url(region, q.code)}",
        f"    3) 「アカウントレベルでの引き上げをリクエスト」→ 新しい値に {max(need, 4)}（g4dn.xlarge 1台分 = 4 vCPU）以上を入力 → 送信",
        "    4) 承認されるまで数分〜数日かかります。メールで届きます。承認後に preflight をもう一度実行してください。",
        "       （理由の欄が出たら「機械学習の音声合成の実験用」などと書く。個人のアカウントでも通ることが多いです）",
        "    CLI で申請する場合（IAM に servicequotas:RequestServiceQuotaIncrease が必要。通常はコンソールでの申請が簡単）:",
        f"      aws service-quotas request-service-quota-increase --service-code ec2 --quota-code {q.code} "
        f"--desired-value {max(need, 4)} --region {region}",
    ]
    return lines


def offered_azs(aws: Aws, instance_type: str) -> list[str]:
    resp = aws.client("ec2").describe_instance_type_offerings(
        LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": [instance_type]}])
    return sorted({o["Location"] for o in resp.get("InstanceTypeOfferings", [])})


def required_vcpu(aws: Aws, instance_type: str) -> int | None:
    v = vcpu_for(instance_type)
    if v:
        return v
    try:
        resp = aws.client("ec2").describe_instance_types(InstanceTypes=[instance_type])
        return int(resp["InstanceTypes"][0]["VCpuInfo"]["DefaultVCpus"])
    except Exception:  # noqa: BLE001
        return None


def price_table_lines(highlight: str | None = None) -> list[str]:
    lines = [f"  {'type':14s} {'vCPU':>4s}  {'GPU':20s} {'$/hour':>8s}  {'3h cap*':>16s}"]
    for t, spec in INSTANCE_SPECS.items():
        est = estimate_cost(spec.usd_per_hour, 3.0)
        mark = " ←" if t == highlight else ""
        lines.append(f"  {t:14s} {spec.vcpu:>4d}  {spec.gpu:20s} {spec.usd_per_hour:>8.3f}  {fmt_usd(est.total):>16s}{mark}")
    lines.append(f"  {PRICE_NOTE}")
    lines.append(f"  * 見積 = 本体 × 時間 ＋ EBS 100GB(gp3) ＋ 公開 IPv4。為替・消費税・データ転送は含みません。値は {PRICE_REGION} のもので、他のリージョンは違います。")
    lines.append("  スポット（--spot）は空きがあれば安いこともありますが変動します。見積の上限はオンデマンド料金で計算します。")
    return lines


def run_preflight(aws: Aws, args: Any) -> int:
    itype = getattr(args, "type", None) or "g4dn.xlarge"
    use_spot = bool(getattr(args, "spot", False))
    blocking: list[str] = []
    say("== preflight（何も作りません。読み取りだけ） ==")

    # 1. 鍵とアカウント
    say("\n[1] 接続とアカウント")
    account = None
    try:
        ident = aws.identity
        account = str(ident["Account"])
        say(f"  アカウント: {mask_account(account)}")
        say(f"  呼び出し元: {mask_text(ident.get('Arn', '?'), account)}")
        say(f"  リージョン: {aws.region}")
    except Exception as e:  # noqa: BLE001
        msg = explain_client_error(e, "鍵の確認（sts get-caller-identity）") if hasattr(e, "response") else \
            f"AWS に接続できませんでした: {type(e).__name__}"
        say("  " + aws.settings.redact(msg).replace("\n", "\n  "))
        say(f"  リージョン: {aws.region}")
        blocking.append("鍵が使えません")

    if account:
        # 2. 枠
        need = required_vcpu(aws, itype)
        say(f"\n[2] GPU インスタンスの vCPU の枠（Service Quotas）… {itype} は {need if need else '?'} vCPU 必要")
        results = {c: fetch_quota(aws, c) for c in (QUOTA_ONDEMAND_G, QUOTA_SPOT_G)}
        for code, q in results.items():
            if q.value is None:
                say(f"  {q.name} ({code}): 取得できませんでした")
                if q.error:
                    say("    " + q.error.replace("\n", "\n    "))
                continue
            ok = need is not None and q.value >= need
            flag = "OK" if ok else ("不足" if need is not None else "?")
            say(f"  {q.name} ({code}): {q.value:g} vCPU（{q.source}） → {flag}")
        chosen = results[QUOTA_SPOT_G if use_spot else QUOTA_ONDEMAND_G]
        if need is not None and chosen.value is not None and chosen.value < need:
            for ln in quota_guidance(chosen, need, aws.region):
                say(ln)
            blocking.append(f"{chosen.name} の枠が {chosen.value:g} vCPU（必要 {need}）")
        elif chosen.value is None:
            warn("枠を確認できなかったので、起動できるかは launch してみないと分かりません。")

        # 3. 提供状況
        say(f"\n[3] {itype} の提供状況（{aws.region}）")
        try:
            azs = offered_azs(aws, itype)
            if azs:
                say(f"  提供している AZ: {', '.join(azs)}")
            else:
                say(f"  {aws.region} では {itype} を提供していません。リージョンかタイプを変えてください。")
                blocking.append(f"{itype} がこのリージョンで提供されていません")
        except Exception as e:  # noqa: BLE001
            say("  確認できませんでした: " + (explain_client_error(e, "提供状況の確認") if hasattr(e, "response") else str(e)))

        # 4. 起動中
        say("\n[4] このツールで起動したインスタンス（タグ Project=meeting-translator / ManagedBy=ec2run）")
        try:
            live = find_instances(aws, states=LIVE_STATES)
            if live:
                for info in live:
                    say(describe_line(info))
                warn(f"{len(live)} 台が残っています。使わないなら `teardown --yes` で止めてください（動いている間は課金されます）。")
            else:
                say("  なし")
        except Exception as e:  # noqa: BLE001
            say("  確認できませんでした: " + (explain_client_error(e, "インスタンスの一覧") if hasattr(e, "response") else str(e)))

    # 5. 料金
    say("\n[5] 料金の目安（オンデマンド・概算）")
    for ln in price_table_lines(itype):
        say(ln)

    # 6. setup
    if account:
        say("\n[6] setup（S3・IAM・セキュリティグループ）の状況")
        try:
            st = discover(aws)
            from .provision import plan_lines

            for ln in plan_lines(st):
                say("  " + ln)
            for n in st.notes:
                warn(n)
        except Exception as e:  # noqa: BLE001
            say("  確認できませんでした: " + (explain_client_error(e, "setup の確認") if hasattr(e, "response") else str(e)))

    say("\n== まとめ ==")
    if blocking:
        for b in blocking:
            say(f"  × {b}")
        say("  → 上の案内を済ませてから、もう一度 preflight を実行してください。")
        return 1
    say("  ○ 進められます。次: setup（内容確認）→ setup --yes → launch --type g4dn.xlarge --hours 3 → launch に --yes")
    return 0


def check_ready_for_launch(aws: Aws, instance_type: str, spot: bool) -> None:
    """launch の前に、枠が足りないと分かっていたら早めに止める（分からなければ通す）。"""
    need = required_vcpu(aws, instance_type)
    if need is None:
        return
    q = fetch_quota(aws, QUOTA_SPOT_G if spot else QUOTA_ONDEMAND_G)
    if q.value is not None and q.value < need:
        raise ToolError("\n".join(quota_guidance(q, need, aws.region)))
