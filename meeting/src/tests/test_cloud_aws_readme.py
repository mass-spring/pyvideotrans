"""README.md が、実際のコードと食い違っていないか（貼り付ける IAM ポリシー・コマンド・名前）。"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from test_cloud_aws_helpers import REPO_ROOT  # noqa: F401  (sys.path を整える)

from meeting.cloud.aws import common, launch
from meeting.cloud.aws.cli import build_parser

README = Path(common.AWS_DIR) / "README.md"


@pytest.fixture(scope="module")
def text():
    return README.read_text(encoding="utf-8")


def test_the_json_in_the_readme_is_exactly_the_policy_file(text):
    blocks = re.findall(r"```json\n(.*?)\n```", text, flags=re.S)
    assert len(blocks) == 1
    assert json.loads(blocks[0]) == json.loads((common.AWS_DIR / "iam_user_policy.json").read_text(encoding="utf-8"))


def test_every_command_example_parses_with_the_real_cli(text):
    cmds = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("$EC2 "):
            line = re.sub(r"\s+#.*$", "", line)                       # 行末の説明
            line = line.replace("<バケット名>", "some-bucket")
            cmds.append(shlex.split(line)[1:])
    assert len(cmds) >= 12
    parser = build_parser()
    seen = set()
    for argv in cmds:
        args = parser.parse_args(argv)
        seen.add(args.command)
    assert seen == {"preflight", "setup", "launch", "run", "status", "teardown", "fetch"}


def test_readme_names_the_environment_variables_and_the_step_order(text):
    for name in ("EC2_ACCESS_KEY_ID", "EC2_SECRET_ACCESS_KEY", "EC2_REGION"):
        assert name in text
    order = [text.index(f"### 手順{i}") for i in range(1, 6)]
    assert order == sorted(order)
    assert text.index("手順1　preflight") < text.index("手順2　setup") < text.index("手順3　launch") < text.index("手順4　run") < text.index("手順5　teardown")


def test_readme_covers_the_console_tasks_the_user_must_do(text):
    for needle in ("L-DB2E81BA", "L-3819A6DF", "Running On-Demand G and VT instances", "ap-northeast-1", "Budgets", "Create access key",
                   "Create policy", "コンソールには入れない", "チャットに貼らない", "4 vCPU", "ルートユーザーの鍵は**絶対に使いません**"):
        assert needle in text, needle


def test_readme_states_the_data_rule_and_cost_caveats(text):
    for needle in ("自分の声", "同意を得た模擬会議", "公開データ", "社外秘", "approximate, check AWS pricing", "--yes", "未実測"):
        assert needle in text, needle


def test_price_table_in_the_readme_matches_the_code(text):
    for itype, spec in common.INSTANCE_SPECS.items():
        row = [ln for ln in text.splitlines() if ln.startswith(f"| {itype}") or ln.startswith(f"| **{itype}**")]
        assert row, itype
        assert f"| {spec.usd_per_hour:g} |" in row[0] or f"| {spec.usd_per_hour:.3f} |" in row[0] or f"| {round(spec.usd_per_hour, 3):g} |" in row[0], row[0]
        est = common.estimate_cost(spec.usd_per_hour, 3.0).total
        assert f"${est:.2f}" in row[0], row[0]


def test_documented_ami_parameter_matches_the_code(text):
    assert launch.ami_parameter("22.04") in text
