"""コマンドライン入口: python -m meeting.cloud.aws <サブコマンド> / python meeting/cloud/aws/ec2run.py <サブコマンド>

  preflight  何も作らず、AWS で GPU を動かせるか確認（鍵・vCPU の枠・提供状況・料金の目安）
  setup      S3 バケット・IAM ロール・セキュリティグループを作る（--yes が要る）
  launch     GPU インスタンスを起動（--yes が要る。自動停止付き）
  run        実験を動かし、結果を meeting/outputs/<実験名>/ に回収（--yes が要る）
  fetch      S3 にある結果を回収だけする
  status     動いているインスタンスと経過時間・費用の見込み
  teardown   インスタンスを終了し、残りものを報告する（--yes が要る）

鍵は環境変数 EC2_ACCESS_KEY_ID / EC2_SECRET_ACCESS_KEY / EC2_REGION から読みます（値は表示しません）。
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .common import (
    DEFAULT_REGION,
    DEFAULT_VOLUME_GB,
    MAX_HOURS,
    Aws,
    NeedConfirmation,
    State,
    ToolError,
    explain_client_error,
    load_settings,
    say,
)
from .launch import OS_VERSIONS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ec2run",
        description="AWS EC2（GPU）で voicelab の比較実験を動かす道具。SSH は使わず、SSM と S3 だけで操作します。",
    )
    p.add_argument("--version", action="version", version=f"ec2run {__version__}")
    p.add_argument("--region", default=None, help=f"リージョン（省略時は環境変数 EC2_REGION、無ければ {DEFAULT_REGION}）")
    p.add_argument("--state-file", default=None, help="状態ファイルの場所（省略時は meeting/cloud/aws/.state.json）")
    sub = p.add_subparsers(dest="command", required=True, metavar="{preflight,setup,launch,run,fetch,status,teardown}")

    def yes(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--yes", "-y", action="store_true", help="内容を確認したうえで実行する（無いと何も作らない・消さない）")

    s = sub.add_parser("preflight", help="事前確認（何も作らない）")
    s.add_argument("--type", default="g4dn.xlarge", help="使う予定のインスタンスタイプ")
    s.add_argument("--spot", action="store_true", help="スポットの枠を基準に判定する")

    s = sub.add_parser("setup", help="S3・IAM・セキュリティグループを作る")
    yes(s)

    s = sub.add_parser("launch", help="GPU インスタンスを起動する（自動停止付き）")
    s.add_argument("--type", default="g4dn.xlarge", help="インスタンスタイプ（既定 g4dn.xlarge）")
    s.add_argument("--spot", action="store_true", help="スポットで起動（安いが、AWS 側の都合で止まることがある）")
    s.add_argument("--hours", type=float, default=3.0, help=f"この時間がたったら自動で停止・終了する（費用の上限。最大 {MAX_HOURS:g}）")
    s.add_argument("--os-version", choices=OS_VERSIONS, default="22.04", help="DLAMI の Ubuntu のバージョン")
    s.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB, help="ルートディスク（gp3）の大きさ GB")
    s.add_argument("--hourly-rate", type=float, default=None, help="料金表にないタイプのときの 1 時間あたりの料金（USD。見積用）")
    s.add_argument("--allow-multiple", action="store_true", help="すでに起動中のものがあっても、もう 1 台起動する")
    s.add_argument("--no-wait", action="store_true", help="起動を依頼したらすぐ終わる（SSM の準備を待たない）")
    s.add_argument("--ssm-timeout", type=int, default=600, help="SSM が使えるようになるまで待つ秒数")
    yes(s)

    s = sub.add_parser("run", help="実験を動かして結果を回収する")
    s.add_argument("--exp", default="exp01", help="実験名（S3 のフォルダ名と、結果の保存先 meeting/outputs/<実験名>/ になる）")
    s.add_argument("--engines", default="voxcpm2,qwen3tts,chatterbox",
                   help="使うエンジン（カンマ区切り: voxcpm2, qwen3tts, chatterbox, cosyvoice3）")
    s.add_argument("--dry-run-first", action="store_true", help="最初にダミー音声で流れを確認する（GPU の時間を使う前に間違いを見つける）")
    s.add_argument("--config", default=None, help="voicelab の設定ファイル（.json / .yaml）。パスはインスタンス用に自動で書き換える")
    s.add_argument("--speaker", action="append", metavar="名前=フォルダ",
                   help="話者フォルダ（voiceprofile の出力、または wav を入れたフォルダ）。複数指定できる")
    s.add_argument("--targets", default=None, help="合成する文の CSV（省略時は既定の文）")
    s.add_argument("--public-test", action="store_true", help="公開データ（LibriSpeech 由来）だけで動作確認する。自分の声は使わない")
    s.add_argument("--mode", choices=["quick", "full"], default="quick", help="quick=繰り返し1回・言語ごとに1文 / full=繰り返し2回・全部の文")
    s.add_argument("--embedder", default=None, help="voicecheck の埋め込み（ecapa / resemblyzer / all / none）。省略時 ecapa")
    s.add_argument("--ref-lang", default=None, help="手本の言語を決め打ちする（例 ja）")
    s.add_argument("--code", choices=["git", "local"], default="git",
                   help="git=GitHub の meeting-translator ブランチを clone / local=このパソコンの meeting/src をアップロード（push 前の変更や非公開のとき）")
    s.add_argument("--instance-id", default=None, help="対象のインスタンス（省略時は、実行中の 1 台）")
    s.add_argument("--out-dir", default=None, help="結果の保存先（省略時 meeting/outputs/<実験名>）")
    s.add_argument("--skip-audio", action="store_true", help="wav をダウンロードしない（Excel と csv だけ）")
    s.add_argument("--attach", action="store_true", help="すでに動いている実験の実行を追いかけ直す（アップロードはしない）")
    s.add_argument("--command-id", default=None, help="--attach のとき、SSM の CommandId を直接指定する")
    s.add_argument("--resume", action="store_true", help="S3 に残っている同じ実験名の途中結果から続きを作る")
    s.add_argument("--fresh", action="store_true", help="S3 に残っている同じ実験名のデータを消して最初から作る")
    s.add_argument("--auto-terminate", action="store_true", help="終わったら 1 分後にインスタンスを自動で終了する（費用の節約）")
    yes(s)

    s = sub.add_parser("fetch", help="S3 にある結果を回収する")
    s.add_argument("--exp", default="exp01")
    s.add_argument("--out-dir", default=None)
    s.add_argument("--skip-audio", action="store_true")

    sub.add_parser("status", help="動いているものと費用の見込みを表示する")

    s = sub.add_parser("teardown", help="インスタンスを終了し、残りものを報告する")
    s.add_argument("--all", action="store_true", help="状態ファイルに記録が無くても、タグが付いたインスタンスを全部終了する")
    s.add_argument("--delete-bucket", action="store_true", help="S3 バケットを中身ごと削除する（--confirm-bucket 名前 が要る）")
    s.add_argument("--confirm-bucket", default=None, help="削除するバケット名（間違い防止の確認）")
    s.add_argument("--delete-setup", action="store_true", help="IAM ロール・インスタンスプロファイル・セキュリティグループも削除する")
    yes(s)
    return p


def dispatch(args: argparse.Namespace, aws: Aws, state: State) -> int:
    cmd = args.command
    if cmd == "preflight":
        from .preflight import run_preflight

        return run_preflight(aws, args)
    if cmd == "setup":
        from .provision import run_setup

        return run_setup(aws, state, args)
    if cmd == "launch":
        from .launch import run_launch

        return run_launch(aws, state, args)
    if cmd == "run":
        from .runner import run_run

        return run_run(aws, state, args)
    if cmd == "fetch":
        from .runner import run_fetch

        return run_fetch(aws, state, args)
    if cmd == "status":
        from .teardown import run_status

        return run_status(aws, state, args)
    if cmd == "teardown":
        from .teardown import run_teardown

        return run_teardown(aws, state, args)
    raise ToolError(f"知らないサブコマンド: {cmd}")   # pragma: no cover


def main(argv: list[str] | None = None, *, aws: Aws | None = None, state: State | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    debug = bool(os.environ.get("EC2RUN_DEBUG"))
    secrets_redact = (lambda t: t)
    try:
        if aws is None:
            aws = Aws(load_settings(region=args.region))
        secrets_redact = aws.settings.redact
        state = state or State(args.state_file)
        return int(dispatch(args, aws, state))
    except NeedConfirmation as e:
        say(secrets_redact(str(e)))
        return e.exit_code
    except ToolError as e:
        say(f"エラー: {secrets_redact(str(e))}")
        return e.exit_code
    except KeyboardInterrupt:
        say("中断しました。")
        return 130
    except Exception as e:  # noqa: BLE001
        if debug:
            raise
        if hasattr(e, "response"):
            say(secrets_redact(explain_client_error(e)))
        else:
            say(secrets_redact(f"エラー: {type(e).__name__}: {e}"))
        say("（詳しい情報: 環境変数 EC2RUN_DEBUG=1 を付けて実行）")
        return 1


if __name__ == "__main__":
    sys.exit(main())
