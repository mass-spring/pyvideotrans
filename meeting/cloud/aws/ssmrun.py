"""SSM Run Command の補助（SSH は使わない）。

- 短いコマンドは SendCommand の結果をそのまま読む（GetCommandInvocation の標準出力は先頭 24,000 文字、
  標準エラーは先頭 8,000 文字まで。実行が終わるまでは空）。
  https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_GetCommandInvocation.html
- 長い実験は、実行中の様子を S3 のログ（リモート側が定期的にアップロード）から少しずつ読む。
  SSM の出力そのものも OutputS3BucketName で S3 に残す（サイズ制限を避ける）。
- SendCommand の文書は 64KB まで。大きな中身は S3 に置き、コマンドは数行にする。
"""
from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from .common import Aws, ToolError, error_code, explain_client_error

TERMINAL_STATUSES = {"Success", "Failed", "TimedOut", "Cancelled"}
DOCUMENT = "AWS-RunShellScript"
BASH = "#!/bin/bash"   # Run Command の既定のシェルは sh のことがあるので、先頭行で bash を指定する


def wait_ssm_online(aws: Aws, instance_id: str, *, timeout: float = 600, poll: float = 10,
                    sleep: Callable[[float], None] = time.sleep,
                    monotonic: Callable[[], float] = time.monotonic,
                    on_wait: Callable[[float], None] | None = None) -> bool:
    """SSM エージェントが Online になるまで待つ。時間切れなら False。"""
    ssm = aws.client("ssm")
    start = monotonic()
    while True:
        try:
            resp = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [instance_id]}])
            infos = resp.get("InstanceInformationList", [])
            if infos and infos[0].get("PingStatus") == "Online":
                return True
        except Exception as e:  # noqa: BLE001
            if not hasattr(e, "response") or error_code(e) in {"AccessDeniedException", "AccessDenied"}:
                raise ToolError(explain_client_error(e, "SSM の状態確認")) from e
        elapsed = monotonic() - start
        if elapsed >= timeout:
            return False
        if on_wait:
            on_wait(elapsed)
        sleep(poll)


def send_script(aws: Aws, instance_id: str, lines: list[str], *, comment: str, execution_timeout_s: int,
                output_bucket: str | None = None, output_prefix: str | None = None,
                delivery_timeout_s: int = 300) -> str:
    """AWS-RunShellScript でスクリプトを送る。CommandId を返す。"""
    params = {"commands": [BASH, *lines], "executionTimeout": [str(int(execution_timeout_s))]}
    kwargs: dict[str, Any] = {
        "InstanceIds": [instance_id], "DocumentName": DOCUMENT, "Comment": comment[:100],
        "Parameters": params, "TimeoutSeconds": max(30, int(delivery_timeout_s)),
    }
    if output_bucket:
        kwargs["OutputS3BucketName"] = output_bucket
        if output_prefix:
            kwargs["OutputS3KeyPrefix"] = output_prefix
    try:
        resp = aws.client("ssm").send_command(**kwargs)
    except Exception as e:  # noqa: BLE001
        if hasattr(e, "response"):
            raise ToolError(explain_client_error(e, "SSM でコマンドを送る操作")) from e
        raise
    return resp["Command"]["CommandId"]


def get_invocation(aws: Aws, command_id: str, instance_id: str, *, retries: int = 8, wait: float = 2.0,
                   sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """GetCommandInvocation。送った直後は InvocationDoesNotExist になることがあるので数回やり直す。"""
    ssm = aws.client("ssm")
    for attempt in range(retries):
        try:
            return ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        except Exception as e:  # noqa: BLE001
            if error_code(e) == "InvocationDoesNotExist" and attempt < retries - 1:
                sleep(wait)
                continue
            raise
    raise ToolError("コマンドの状態を取得できませんでした")  # pragma: no cover


def run_quick(aws: Aws, instance_id: str, lines: list[str], *, comment: str = "ec2run quick", timeout: float = 300,
              poll: float = 3.0, sleep: Callable[[float], None] = time.sleep,
              monotonic: Callable[[], float] = time.monotonic) -> tuple[str, str, str]:
    """短いコマンドを送って終わるまで待つ。(Status, stdout, stderr)。出力は SSM が返す先頭分だけ。"""
    cid = send_script(aws, instance_id, lines, comment=comment, execution_timeout_s=max(60, int(timeout)))
    start = monotonic()
    while True:
        inv = get_invocation(aws, cid, instance_id, sleep=sleep)
        if inv.get("Status") in TERMINAL_STATUSES:
            return inv["Status"], inv.get("StandardOutputContent", ""), inv.get("StandardErrorContent", "")
        if monotonic() - start > timeout:
            return "TimedOut", inv.get("StandardOutputContent", ""), "ec2run: 待ち時間切れ"
        sleep(poll)


def parse_s3_url(url: str) -> tuple[str, str]:
    """GetCommandInvocation の StandardOutputUrl などから (bucket, key) を取り出す。"""
    u = urlparse(url)
    if u.scheme == "s3":
        return u.netloc, unquote(u.path.lstrip("/"))
    host, path = u.netloc, unquote(u.path.lstrip("/"))
    if host.startswith("s3.") or host.startswith("s3-") or host == "s3.amazonaws.com":   # パス形式: s3.<region>.amazonaws.com/bucket/key
        bucket, _, key = path.partition("/")
        return bucket, key
    if ".s3." in host or ".s3-" in host:                                                  # 仮想ホスト形式: bucket.s3.<region>.amazonaws.com/key
        return host.split(".s3", 1)[0], path
    raise ToolError(f"S3 の URL を解釈できません: {url}")


def read_s3_text(aws: Aws, bucket: str, key: str, limit: int = 200_000) -> str:
    body = aws.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read(limit)
    return body.decode("utf-8", errors="replace")


def full_output(aws: Aws, inv: dict[str, Any], which: str = "stdout", limit: int = 200_000) -> str:
    """SSM が返す先頭分では足りないとき、S3 に保存された全文を読む（OutputS3BucketName を指定していた場合）。"""
    inline = inv.get("StandardOutputContent" if which == "stdout" else "StandardErrorContent", "")
    url = inv.get("StandardOutputUrl" if which == "stdout" else "StandardErrorUrl", "")
    if not url:
        return inline
    try:
        b, k = parse_s3_url(url)
        return read_s3_text(aws, b, k, limit)
    except Exception:  # noqa: BLE001
        return inline


class LogTail:
    """S3 上のログ（リモートが十数秒ごとに上書きアップロード）を、前回の続きから読む。"""

    def __init__(self, aws: Aws, bucket: str, key: str):
        self.aws, self.bucket, self.key = aws, bucket, key
        self.offset = 0

    def poll(self, final: bool = False) -> str:
        """新しく増えた分（行の区切りまで）を返す。final=True なら、改行で終わっていない最後の行も返す。"""
        s3 = self.aws.client("s3")
        try:
            r = s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={self.offset}-")
        except Exception as e:  # noqa: BLE001
            code = error_code(e)
            if code in ("NoSuchKey", "404", "NotFound"):
                return ""
            if code in ("InvalidRange", "416"):
                try:
                    size = int(s3.head_object(Bucket=self.bucket, Key=self.key)["ContentLength"])
                    if size < self.offset:      # ログが作り直された
                        self.offset = 0
                except Exception:  # noqa: BLE001
                    pass
                return ""
            raise
        data = r["Body"].read()
        if not data:
            return ""
        cut = len(data) - 1 if final else data.rfind(b"\n")
        if cut < 0:            # まだ1行が書き終わっていない
            return ""
        self.offset += cut + 1
        return data[: cut + 1].decode("utf-8", errors="replace")
