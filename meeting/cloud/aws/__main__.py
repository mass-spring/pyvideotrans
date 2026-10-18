"""python -m meeting.cloud.aws ... で起動する。"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
