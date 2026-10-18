"""python meeting/cloud/aws/ec2run.py ... で起動する入口（python -m meeting.cloud.aws と同じ）。"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # スクリプトとして直接実行された場合: リポジトリのルートを import 先頭に置く
    # （このフォルダ自身を先頭に置くと、common.py などが他のライブラリ名と衝突しうるので外す）
    sys.path[0:1] = [str(Path(__file__).resolve().parents[3])]
    from meeting.cloud.aws.cli import main  # noqa: E402
else:  # pragma: no cover - python -m で来たとき
    from .cli import main

if __name__ == "__main__":
    sys.exit(main())
