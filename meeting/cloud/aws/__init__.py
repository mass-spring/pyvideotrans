"""AWS EC2（GPU）で voicelab の比較実験を動かす道具（ec2run）。

このクラウド開発環境から、AWS の API だけ（SSM Run Command と S3）で
GPU インスタンスの起動 → 実験の実行 → 結果の回収 → 停止までを行う。SSH は使わない。

使い方は README.md を参照。
    python -m meeting.cloud.aws preflight        （リポジトリのルートで実行）
    python meeting/cloud/aws/ec2run.py preflight
"""
__version__ = "0.1.0"
