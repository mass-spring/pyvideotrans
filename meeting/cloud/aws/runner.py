"""run / fetch: インスタンスで voicelab の実験を動かし、結果をこのパソコン（開発環境）に回収する。

流れ:
  1. 入力（話者フォルダ・targets.csv・設定）を作る → S3 にアップロード（SSH は使わない）
  2. SSM Run Command で remote_job.py を起動（実行時間の上限は SSM の executionTimeout）
  3. 実行中は、インスタンスが S3 に上げ続けるログを読んで表示（SSM の出力は終わるまで空なので）
  4. 結果（Excel・csv・wav）を S3 から meeting/outputs/<実験名>/ にダウンロード
"""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from . import recipes
from .common import (
    DATA_RULE,
    DEFAULT_OUT_ROOT,
    MEETING_DIR,
    Aws,
    State,
    ToolError,
    error_code,
    explain_client_error,
    fmt_hms,
    iso,
    require_yes,
    say,
    utcnow,
    warn,
)
from .instances import InstanceInfo, describe_line, find_instances
from .provision import resolve_resources
from .ssmrun import TERMINAL_STATUSES, LogTail, full_output, get_invocation, send_script

EXP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
REMOTE_JOB_PY = Path(__file__).with_name("remote_job.py")
PREPARE_PUBLIC_PY = Path(__file__).with_name("prepare_public.py")
MAX_UPLOAD_MB = 500
PUBLIC_SPEAKER = "public1272"


def validate_exp(name: str) -> str:
    if not EXP_RE.match(name or ""):
        raise ToolError(f"--exp は英数字と . _ - だけ（40 文字まで、先頭は英数字）にしてください: {name!r}")
    return name


# ---------------------------------------------------------------------------
# 入力の組み立て（通信しない）
# ---------------------------------------------------------------------------
def load_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ToolError(f"設定ファイルがありません: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ToolError("YAML を読むには pyyaml が要ります。同じ内容を .json で書くか、`uv run --with pyyaml ...` で実行してください。") from e
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text.lstrip("﻿"))
        except json.JSONDecodeError as e:
            raise ToolError(f"設定ファイルを JSON として読めません: {p}（{e}）") from e
    if not isinstance(data, dict):
        raise ToolError(f"設定ファイルの中身が辞書ではありません: {p}")
    return data


@dataclasses.dataclass
class SpeakerInput:
    name: str
    kind: str          # "profile" | "folder"
    local_dir: Path


def detect_kind(d: Path) -> str:
    if (d / "profile.json").is_file():
        return "profile"
    if any(d.glob("*.wav")):
        return "folder"
    raise ToolError(f"{d} は話者フォルダではありません（profile.json か wav がありません）")


def collect_speakers(speaker_args: list[str] | None, base_cfg: dict[str, Any], base_dir: Path) -> list[SpeakerInput]:
    out: list[SpeakerInput] = []
    for s in speaker_args or []:
        name, sep, d = s.partition("=")
        if not sep or not name.strip() or not d.strip():
            raise ToolError(f"--speaker は 名前=フォルダ の形で指定してください（例: --speaker A=meeting/profiles/A）: {s!r}")
        p = Path(d).expanduser()
        if not p.is_dir():
            raise ToolError(f"話者フォルダがありません: {p}")
        out.append(SpeakerInput(name.strip(), detect_kind(p), p))
    if out:
        return out
    for sp in base_cfg.get("speakers", []) or []:
        if not isinstance(sp, dict) or not sp.get("name"):
            raise ToolError("設定ファイルの speakers の各項目には name が要ります")
        key = "profile" if sp.get("profile") else ("folder" if sp.get("folder") else None)
        if key is None:
            raise ToolError(f"話者 {sp['name']} に profile か folder がありません")
        p = Path(sp[key]).expanduser()
        p = p if p.is_absolute() else (base_dir / p)
        if not p.is_dir():
            raise ToolError(f"話者 {sp['name']} のフォルダがありません: {p}")
        out.append(SpeakerInput(str(sp["name"]), key, p))
    return out


def _tar_filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(ti.name).parts
    if any(p.startswith(".") or p == "__pycache__" for p in parts if p != "."):
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def make_tar(src_dir: Path, dest: Path) -> int:
    """フォルダの中身を tar.gz にする（中身が tar の直下に来る）。大きさ（バイト）を返す。"""
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(str(src_dir), arcname=".", filter=_tar_filter)
    return dest.stat().st_size


def local_commit() -> str:
    """パソコン側のリポジトリの commit（参考情報）。"""
    try:
        h = subprocess.run(["git", "-C", str(MEETING_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(MEETING_DIR), "status", "--porcelain", "--", "src"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return (h[:10] or "unknown") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclasses.dataclass
class RunPlan:
    exp: str
    prefix: str
    groups: list[recipes.EngineGroup]
    config: dict[str, Any]
    job: dict[str, Any]
    uploads: list[tuple[Path, str]]      # (ローカルのファイル, S3 のキー)
    notes: list[str]
    upload_bytes: int


def build_plan(args: Any, *, bucket: str, region: str, tmp: Path) -> RunPlan:
    """入力を作る。tmp の中に tar などを書く。AWS には触らない。"""
    exp = validate_exp(args.exp)
    groups = recipes.parse_engines(args.engines)
    prefix = f"exp/{exp}"
    work = recipes.WORKDIR
    base_cfg: dict[str, Any] = {}
    base_dir = Path.cwd()
    if args.config:
        base_cfg = load_config_file(args.config)
        base_dir = Path(args.config).resolve().parent
    public = bool(args.public_test)
    notes: list[str] = []
    uploads: list[tuple[Path, str]] = []
    inputs: list[dict[str, str]] = []

    # --- 話者 ---
    speakers_cfg: list[dict[str, str]] = []
    if public:
        if args.speaker:
            raise ToolError("--public-test と --speaker は同時に使えません")
        speakers_cfg.append({"name": PUBLIC_SPEAKER, "folder": f"{work}/inputs/{exp}/{PUBLIC_SPEAKER}"})
        notes.append("話者: 公開データ（LibriSpeech 由来・CC-BY-4.0）を、インスタンス上で取得します（あなたの声は使いません）")
    else:
        spk = collect_speakers(args.speaker, base_cfg, base_dir)
        if not spk:
            raise ToolError("話者がありません。--speaker 名前=フォルダ（例: --speaker A=meeting/profiles/A）、設定ファイルの speakers、"
                            "または --public-test（公開データだけで動作確認）を指定してください。")
        for i, s in enumerate(spk):
            tar = tmp / f"spk{i}.tar.gz"
            size = make_tar(s.local_dir, tar)
            key = f"{prefix}/inputs/spk{i}.tar.gz"
            uploads.append((tar, key))
            dest = f"{work}/inputs/{exp}/spk{i}"
            inputs.append({"key": key, "dest": dest, "kind": "tar"})
            speakers_cfg.append({"name": s.name, s.kind: dest})
            notes.append(f"話者 {s.name}: {s.local_dir}（{s.kind}）→ アップロード {size / 1e6:.1f} MB")

    # --- targets ---
    targets_local: Path | None = None
    if args.targets:
        targets_local = Path(args.targets).expanduser()
    elif base_cfg.get("targets"):
        t = Path(str(base_cfg["targets"])).expanduser()
        targets_local = t if t.is_absolute() else base_dir / t
    default_targets = None
    if targets_local is not None:
        if not targets_local.is_file():
            raise ToolError(f"targets.csv がありません: {targets_local}")
        key = f"{prefix}/inputs/targets.csv"
        uploads.append((targets_local, key))
        remote_targets = f"{work}/inputs/{exp}/targets.csv"
        inputs.append({"key": key, "dest": remote_targets, "kind": "file"})
        notes.append(f"文: {targets_local} をアップロード")
    else:
        remote_targets = f"{work}/inputs/{exp}/targets.csv"
        default_targets = remote_targets
        notes.append("文: 既定の targets.csv（会議風3文＋感情3文 × 4言語）をインスタンス上で作ります")

    # --- 設定 ---
    quick = args.mode == "quick"
    cfg: dict[str, Any] = {
        "target_langs": ["ja", "en", "fr", "zh"],
        "conditions": ["line_ref_16k", "profile_ref", "stretch", "emotion"],
        "repeats": 1 if quick else 2,
        "seed": 1234,
        "stretch_factors": [1.3, 2.0],
        "max_texts_per_lang": 1 if quick else None,
        "ref_lang": "",
        "embedder": "ecapa",
    }
    for k, v in base_cfg.items():                 # 設定ファイルの値で上書き（下の強制項目を除く）
        if k not in {"name", "speakers", "targets", "engines"}:
            cfg[k] = v
    if args.ref_lang is not None:
        cfg["ref_lang"] = args.ref_lang
    if args.embedder:
        cfg["embedder"] = args.embedder
    cfg["emotion_engines"] = base_cfg.get("emotion_engines", ["voxcpm2"] if any(g.name == "voxcpm2" for g in groups) else [])
    cfg.update({"name": exp, "speakers": speakers_cfg, "targets": remote_targets, "engines": recipes.config_engines(groups)})
    cfg_local = tmp / "exp.json"
    cfg_local.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    uploads.append((cfg_local, f"{prefix}/job/exp.json"))
    validate_config_locally(cfg)

    # --- コード ---
    commit = local_commit()
    if args.code == "git":
        code = {"mode": "git", "url": recipes.REPO_URL, "branch": recipes.REPO_BRANCH}
        notes.append(f"コード: {recipes.REPO_URL} の {recipes.REPO_BRANCH}（push 済みの内容。パソコン側の commit は {commit}）")
    else:
        tar = tmp / "code.tar.gz"
        size = make_tar_code(tar)
        key = f"{prefix}/inputs/code.tar.gz"
        uploads.append((tar, key))
        code = {"mode": "s3", "key": key, "client_commit": commit}
        notes.append(f"コード: このパソコンの meeting/src をアップロード（{size / 1e6:.1f} MB、commit {commit}）")

    # --- ジョブ ---
    job: dict[str, Any] = {
        "version": 1, "exp": exp, "region": region, "bucket": bucket, "prefix": prefix, "workdir": work,
        "paths": {"repo": f"{work}/pyvideotrans", "src": f"{work}/pyvideotrans/meeting/src",
                  "result": f"{work}/work/{exp}/result", "config": f"{work}/work/{exp}/exp.json",
                  "hf_home": f"{work}/hf_cache", "cosy_repo": f"{work}/CosyVoice"},
        "apt": recipes.APT_PACKAGES, "uv_version": recipes.UV_VERSION, "code": code,
        "score_env": recipes.SCORE_ENV.to_dict(),
        "engine_groups": [g.to_dict() for g in groups],
        "inputs": inputs, "default_targets": default_targets,
        "public_test": ({"helper_key": f"{prefix}/job/prepare_public.py", "dest": f"{work}/inputs/{exp}/{PUBLIC_SPEAKER}"} if public else None),
        "config_key": f"{prefix}/job/exp.json",
        "dry_run_first": bool(args.dry_run_first), "embedder": cfg["embedder"],
        "shutdown_when_done": bool(args.auto_terminate), "allow_no_gpu": False,
    }
    job_local = tmp / "job.json"
    job_local.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    uploads.append((job_local, f"{prefix}/job/job.json"))
    uploads.append((REMOTE_JOB_PY, f"{prefix}/job/remote_job.py"))
    if public:
        uploads.append((PREPARE_PUBLIC_PY, f"{prefix}/job/prepare_public.py"))
    total = sum(p.stat().st_size for p, _ in uploads)
    if total > MAX_UPLOAD_MB * 1_000_000:
        raise ToolError(f"アップロードが {total / 1e6:.0f} MB あります（上限の目安 {MAX_UPLOAD_MB} MB）。話者フォルダの中身を減らしてください。")
    return RunPlan(exp, prefix, groups, cfg, job, uploads, notes, total)


def make_tar_code(dest: Path) -> int:
    src = MEETING_DIR / "src"
    if not src.is_dir():
        raise ToolError(f"{src} がありません")

    def flt(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        if "tests" in parts[:2] or any(p == "__pycache__" or p.startswith(".") for p in parts if p != "."):
            return None
        return _tar_filter(ti)

    with tarfile.open(dest, "w:gz") as tf:
        tf.add(str(src), arcname=".", filter=flt)
    return dest.stat().st_size


def validate_config_locally(cfg: dict[str, Any]) -> None:
    """voicelab の設定検査を、パソコン側で先に通す（GPU の時間を使う前に書き間違いを見つける）。
    voicelab が import できない環境では飛ばす。"""
    import sys

    src = str(MEETING_DIR / "src")
    added = src not in sys.path
    if added:
        sys.path.insert(0, src)
    try:
        from voicelab.config import ConfigError, config_from_dict   # type: ignore
    except Exception:  # noqa: BLE001
        return
    finally:
        if added and src in sys.path:
            sys.path.remove(src)
    try:
        config_from_dict(dict(cfg))
    except ConfigError as e:
        raise ToolError(f"設定に誤りがあります（voicelab の検査）: {e}") from e


def remote_command_lines(bucket: str, prefix: str, region: str) -> list[str]:
    """SendCommand で送る数行。大きな中身は S3 に置いてあるので、ここは短い（SendCommand の文書は 64KB まで）。"""
    wd = recipes.WORKDIR
    return [
        "set -u",
        f"export HOME=/root DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}",
        f"mkdir -p {wd} && cd {wd}",
        # DLAMI の標準 Python には boto3 が入っている。無ければ入れる
        "python3 -c 'import boto3' 2>/dev/null || python3 -m pip install -q boto3 || python3 -m pip install -q --break-system-packages boto3",
        f"python3 -c \"import boto3; boto3.client('s3', region_name='{region}').download_file('{bucket}', '{prefix}/job/remote_job.py', '{wd}/remote_job.py')\"",
        f"exec python3 {wd}/remote_job.py --bucket {bucket} --prefix {prefix} --region {region} --workdir {wd}",
    ]


# ---------------------------------------------------------------------------
# S3 の出し入れ
# ---------------------------------------------------------------------------
def s3_has_objects(aws: Aws, bucket: str, prefix: str) -> bool:
    resp = aws.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(resp.get("KeyCount"))


def s3_delete_prefix(aws: Aws, bucket: str, prefix: str) -> int:
    s3 = aws.client("s3")
    n = 0
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys = [{"Key": o["Key"]} for o in resp.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
            n += len(keys)
        if not resp.get("IsTruncated"):
            return n
        token = resp.get("NextContinuationToken")


def upload_plan(aws: Aws, bucket: str, plan: RunPlan) -> None:
    s3 = aws.client("s3")
    for path, key in plan.uploads:
        s3.upload_file(str(path), bucket, key)


def read_json_object(aws: Aws, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        body = aws.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        if error_code(e) in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


@dataclasses.dataclass
class DownloadSummary:
    files: int = 0
    bytes: int = 0
    skipped: int = 0
    dest: Path | None = None


def _is_audio(rel: str) -> bool:
    return rel.lower().endswith(".wav")


def download_results(aws: Aws, bucket: str, prefix: str, dest: Path, *, skip_audio: bool = False) -> DownloadSummary:
    """S3 の result/ と logs/ を dest にダウンロードする（同じ大きさのファイルは飛ばす）。"""
    s3 = aws.client("s3")
    dest.mkdir(parents=True, exist_ok=True)
    summ = DownloadSummary(dest=dest)
    for sub, local_sub in (("result", ""), ("logs", "ec2run_logs")):
        p = f"{prefix}/{sub}/"
        token = None
        while True:
            kw: dict[str, Any] = {"Bucket": bucket, "Prefix": p}
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                rel = o["Key"][len(p):]
                if not rel or rel.endswith("/") or ".." in Path(rel).parts:
                    continue
                if skip_audio and _is_audio(rel):
                    summ.skipped += 1
                    continue
                target = dest / local_sub / rel if local_sub else dest / rel
                if target.exists() and target.stat().st_size == int(o["Size"]) and not rel.endswith((".xlsx", ".csv", ".json", ".log")):
                    summ.skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, o["Key"], str(target))
                summ.files += 1
                summ.bytes += int(o["Size"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    return summ


def out_dir_for(exp: str, override: str | None) -> Path:
    return Path(override).expanduser() if override else DEFAULT_OUT_ROOT / exp


# ---------------------------------------------------------------------------
# 実行を追いかける
# ---------------------------------------------------------------------------
def _is_transient(e: BaseException) -> bool:
    """一時的な失敗（ネットワーク・混雑）か。権限エラーなどは一時的ではない。"""
    if hasattr(e, "response"):
        return error_code(e) in {"Throttling", "ThrottlingException", "RequestLimitExceeded", "InternalError", "InternalServerError",
                                 "ServiceUnavailable", "SlowDown", "RequestTimeout", "RequestTimeTooSkewed"}
    return isinstance(e, (OSError, ConnectionError, TimeoutError)) or type(e).__name__ in {
        "EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError", "ProxyConnectionError", "SSLError"}


@dataclasses.dataclass
class FollowResult:
    ssm_status: str
    done: dict[str, Any] | None
    reason: str          # "done" | "ssm" | "instance-gone" | "timeout"


def follow_command(aws: Aws, *, bucket: str, prefix: str, command_id: str, instance_id: str, poll: float = 10.0,
                   sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
                   max_seconds: float | None = None, out: Callable[[str], None] = say,
                   heartbeat: float = 120.0) -> FollowResult:
    tail = LogTail(aws, bucket, f"{prefix}/logs/run.log")
    start = monotonic()
    last_phase = None
    last_output = monotonic()
    polls = 0
    ssm_status = "InProgress"
    done: dict[str, Any] | None = None
    reason = "ssm"

    def emit(text: str) -> None:
        nonlocal last_output
        for ln in text.splitlines():
            out(f"  | {ln}")
        if text:
            last_output = monotonic()

    failures = 0
    while True:
        try:
            emit(tail.poll())
            st = read_json_object(aws, bucket, f"{prefix}/logs/status.json")
            if st and st.get("phase") != last_phase:
                last_phase = st.get("phase")
                out(f"  [段階] {last_phase}（経過 {fmt_hms(st.get('elapsed_s', 0))}）")
            done = read_json_object(aws, bucket, f"{prefix}/logs/done.json")
            if done is not None:
                reason = "done"
                break
            try:
                inv = get_invocation(aws, command_id, instance_id, retries=2, wait=1.0, sleep=sleep)
                ssm_status = inv.get("Status", ssm_status)
                if ssm_status in TERMINAL_STATUSES:
                    reason = "ssm"
                    break
            except Exception as e:  # noqa: BLE001
                if error_code(e) not in {"InvalidInstanceId", "InvocationDoesNotExist"}:
                    raise
            failures = 0
        except Exception as e:  # noqa: BLE001
            if not _is_transient(e):
                if hasattr(e, "response"):
                    raise ToolError(explain_client_error(e, "実行状況の確認")) from e
                raise
            failures += 1
            out(f"  ! AWS との通信が一時的に失敗しました（{failures}/10）。やり直します…")
            if failures >= 10:
                raise ToolError("AWS との通信が続けて失敗しました。実験はインスタンスで続いています。"
                                f"通信が戻ったら `run --exp {prefix.split('/', 1)[-1]} --attach` で追いかけ直せます。") from e
            sleep(min(30.0, poll * failures))
            continue
        polls += 1
        if polls % 6 == 0:
            try:
                infos = find_instances(aws, instance_ids=[instance_id])
                if infos and infos[0].state in ("shutting-down", "terminated", "stopping", "stopped"):
                    out(f"  ! インスタンス {instance_id} は {infos[0].state} です（自動停止の時間が来た可能性）。S3 にある分を回収します。")
                    reason = "instance-gone"
                    break
            except Exception:  # noqa: BLE001
                pass
        if max_seconds is not None and monotonic() - start > max_seconds:
            reason = "timeout"
            break
        if monotonic() - last_output > heartbeat:
            out(f"  …実行中（段階 {last_phase or '?'}、経過 {fmt_hms(monotonic() - start)}）")
            last_output = monotonic()
        sleep(poll)

    for _ in range(3):      # 最後のログの取りこぼしを拾う（終了直後はアップロードが少し遅れる）
        emit(tail.poll(final=True))
        if done is not None or reason != "ssm":
            sleep(min(poll, 3.0))
        else:
            break
    if done is None:
        done = read_json_object(aws, bucket, f"{prefix}/logs/done.json")
    return FollowResult(ssm_status, done, reason)


def summarize_done(done: dict[str, Any] | None) -> list[str]:
    if not done:
        return ["完了の記録（done.json）がありません。途中で止まった可能性があります。ログ（ec2run_logs/run.log）を確認してください。"]
    lines = [f"結果: {'成功' if done.get('ok') else '失敗または一部失敗'}（終了コード {done.get('exit_code')}、所要 {fmt_hms(done.get('elapsed_s', 0))}）"]
    if done.get("error"):
        lines.append(f"  中止の理由: {done['error']}")
    for name, info in (done.get("engines") or {}).items():
        env = "環境OK" if info.get("env_ok") else "環境NG"
        lines.append(f"  エンジン {name}: {env} / generate 終了コード {info.get('generate_rc')} / {info.get('seconds', '-')}秒")
    for eng, counts in (done.get("manifest") or {}).items():
        lines.append(f"  生成 {eng}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if done.get("commit"):
        lines.append(f"  コードの commit: {done['commit']}")
    return lines


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def pick_instance(aws: Aws, state: State, instance_id: str | None) -> InstanceInfo:
    live = find_instances(aws, states=["running", "pending"])
    if instance_id:
        for i in live:
            if i.instance_id == instance_id:
                return i
        raise ToolError(f"{instance_id} は、このツールで起動した実行中のインスタンスではありません（タグ Project/ManagedBy が無い、または停止済み）。")
    if len(live) == 1:
        return live[0]
    if not live:
        raise ToolError("実行中のインスタンスがありません。先に `launch --yes` で起動してください。")
    raise ToolError("実行中のインスタンスが複数あります。--instance-id で選んでください:\n" + "\n".join(describe_line(i) for i in live))


def run_run(aws: Aws, state: State, args: Any, *, sleep: Callable[[float], None] = time.sleep,
            monotonic: Callable[[], float] = time.monotonic, poll: float = 10.0) -> int:
    exp = validate_exp(args.exp)
    out = out_dir_for(exp, args.out_dir)
    res = resolve_resources(aws, state)
    bucket = res["bucket"]
    prefix = f"exp/{exp}"

    if args.attach:
        rec = state.last_run(exp)
        if not rec or not rec.get("command_id"):
            raise ToolError(f"実験 {exp} の実行記録が状態ファイルにありません。--command-id と --instance-id を指定してください。")
        cid = args.command_id or rec["command_id"]
        iid = args.instance_id or rec["instance_id"]
        say(f"== run --attach: 実験 {exp} の実行を追いかけます（command {cid} / {iid}）==")
        return _follow_and_fetch(aws, state, exp, bucket, cid, iid, out, args, sleep, monotonic, poll)

    say("== run: 実験をインスタンスで動かす ==")
    say(DATA_RULE)
    inst = pick_instance(aws, state, args.instance_id)
    with tempfile.TemporaryDirectory(prefix="ec2run-") as td:
        plan = build_plan(args, bucket=bucket, region=aws.region, tmp=Path(td))
        say(f"実験名: {exp} / エンジン: {', '.join(g.name for g in plan.groups)} / モード: {args.mode}"
            f"{' / 最初にダミー確認' if args.dry_run_first else ''}")
        say(f"インスタンス: {describe_line(inst).strip()}")
        for n in plan.notes:
            say("  - " + n)
        say(f"  - S3 にアップロードするのは {len(plan.uploads)} ファイル（計 {plan.upload_bytes / 1e6:.1f} MB）: s3://{bucket}/{prefix}/")
        say(f"  - 結果の保存先: {out}")
        if inst.expires_at:
            left = (inst.expires_at - utcnow()).total_seconds()
            say(f"  - 自動停止まで {fmt_hms(left)}（それまでに終わらなければ、途中結果は S3 に残るので `run --resume` で続きから作れます）")
            if left < 1800:
                warn("自動停止まで 30 分を切っています。環境づくりだけで 20〜40 分かかることがあります。")
        require_yes(args, "上の内容を S3（あなたの AWS アカウントの専用バケット）にアップロードし、インスタンスで実行します。\n"
                          "--yes を付けることは、上の【データの規則】を満たしていると確認した、という意味です。")

        exists = s3_has_objects(aws, bucket, f"{prefix}/result/")
        if exists and not (args.resume or args.fresh):
            raise ToolError(f"実験名 {exp} の結果が S3 に残っています。続きから作る: --resume / 作り直す: --fresh / 別の名前: --exp 別名")
        if args.fresh:
            n = s3_delete_prefix(aws, bucket, f"{prefix}/")
            say(f"  前回の S3 上のデータ {n} 件を削除しました（--fresh）")
        else:
            s3_delete_prefix(aws, bucket, f"{prefix}/logs/")      # ログと完了の印だけは必ず新しくする
        say("  アップロード中…")
        upload_plan(aws, bucket, plan)

    exec_s = 3600 * 6
    if inst.expires_at:
        exec_s = max(900, int((inst.expires_at - utcnow()).total_seconds()))
    say("  SSM でコマンドを送ります…")
    cid = send_script(aws, inst.instance_id, remote_command_lines(bucket, prefix, aws.region), comment=f"ec2run {exp}",
                      execution_timeout_s=exec_s, output_bucket=bucket, output_prefix=f"{prefix}/ssm-output")
    state.record_run(exp, command_id=cid, instance_id=inst.instance_id, bucket=bucket, started_at=iso(utcnow()),
                     engines=[g.name for g in plan.groups], mode=args.mode, out_dir=str(out), status="running")
    say(f"  送りました: command {cid}。以下はインスタンスのログです（切れても実験は続きます。再接続: run --exp {exp} --attach）")
    return _follow_and_fetch(aws, state, exp, bucket, cid, inst.instance_id, out, args, sleep, monotonic, poll,
                             info={"engines": [g.name for g in plan.groups], "mode": args.mode,
                                   "code": plan.job["code"], "config": plan.config})


def _follow_and_fetch(aws: Aws, state: State, exp: str, bucket: str, cid: str, iid: str, out: Path, args: Any,
                      sleep: Callable[[float], None], monotonic: Callable[[], float], poll: float,
                      info: dict[str, Any] | None = None) -> int:
    prefix = f"exp/{exp}"
    try:
        fr = follow_command(aws, bucket=bucket, prefix=prefix, command_id=cid, instance_id=iid, poll=poll, sleep=sleep,
                            monotonic=monotonic)
    except KeyboardInterrupt:
        say(f"\n切断しました。インスタンスの実験は続いています。再接続: python -m meeting.cloud.aws run --exp {exp} --attach")
        return 130
    state.record_run(exp, status="done" if fr.done and fr.done.get("ok") else "finished-with-problems", ssm_status=fr.ssm_status,
                     finished_at=iso(utcnow()))
    say("")
    for ln in summarize_done(fr.done):
        say(ln)
    if fr.ssm_status != "Success" and fr.ssm_status != "InProgress":
        try:
            inv = get_invocation(aws, cid, iid, retries=1, sleep=sleep)
            err = full_output(aws, inv, "stderr", 4000).strip()
            if err:
                say("  SSM の標準エラー（先頭）:\n    " + err[:1500].replace("\n", "\n    "))
        except Exception:  # noqa: BLE001
            pass
    dl = download_results(aws, bucket, prefix, out, skip_audio=bool(getattr(args, "skip_audio", False)))
    write_run_info(out, exp=exp, bucket=bucket, command_id=cid, instance_id=iid, done=fr.done, info=info or {}, region=aws.region)
    say(f"結果をダウンロードしました: {out}（{dl.files} ファイル、{dl.bytes / 1e6:.1f} MB、飛ばした {dl.skipped}）")
    xlsx = out / "voicelab_result.xlsx"
    if xlsx.exists():
        say(f"  ★ 結果の Excel: {xlsx}")
        say("    見る順番は meeting/src/voicelab/README.md の6章（「現状値」シート → 「条件比較」シート）")
    else:
        warn("voicelab_result.xlsx がありません。ログ（ec2run_logs/run.log）の最後を確認してください。")
    say("")
    say("インスタンスはまだ動いています（動いている間は課金されます）。使い終わったら: python -m meeting.cloud.aws teardown --yes")
    return 0 if fr.done and fr.done.get("ok") else 1


def write_run_info(out: Path, **fields: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    info = {k: v for k, v in fields.items()}
    info["downloaded_at"] = iso(utcnow())
    (out / "ec2run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_fetch(aws: Aws, state: State, args: Any) -> int:
    exp = validate_exp(args.exp)
    res = resolve_resources(aws, state)
    out = out_dir_for(exp, args.out_dir)
    prefix = f"exp/{exp}"
    if not s3_has_objects(aws, res["bucket"], f"{prefix}/"):
        raise ToolError(f"S3 に実験 {exp} のデータがありません（s3://{res['bucket']}/{prefix}/）。バケットの保存期間は 14 日です。")
    done = read_json_object(aws, res["bucket"], f"{prefix}/logs/done.json")
    for ln in summarize_done(done):
        say(ln)
    dl = download_results(aws, res["bucket"], prefix, out, skip_audio=bool(args.skip_audio))
    say(f"ダウンロードしました: {out}（{dl.files} ファイル、{dl.bytes / 1e6:.1f} MB）")
    return 0
