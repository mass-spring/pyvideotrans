#!/usr/bin/env python3
"""ec2run のインスタンス側の実行役（GPU インスタンスの上で動く。パソコン側では使わない）。

流れ（ノートブック voice_baseline_colab.ipynb の手順と同じ）:
  sysinfo → system（apt・uv）→ code → resume → env:score → inputs → plan → [dry-run]
  → engine ごと（env 作成 → generate）→ score（測定・Excel）→ done

- 標準ライブラリだけで動く（S3 のために boto3 を使う。DLAMI の標準 Python に入っている）。
- ログは 15 秒ごとに S3（<prefix>/logs/run.log）へ上げる。パソコン側がそれを読んで表示する。
- 途中結果は engine ごとに S3（<prefix>/result/）へ上げる。同じ実験名でやり直すと続きから作る（voicelab の再開機能）。
- 使うのは SSM Run Command だけ。SSH は使わない。
Python 3.10 以上で動く。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable

LOG_INTERVAL = 15.0


class Fatal(Exception):
    """続けられない失敗。exit_code で終わる。"""

    def __init__(self, msg: str, exit_code: int = 1):
        super().__init__(msg)
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# ログ
# ---------------------------------------------------------------------------
class RemoteLog:
    def __init__(self, path: Path, echo: bool = True):
        self.path = path
        self.echo = echo
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8", errors="replace")   # 同じインスタンスで再実行しても、前回のログを混ぜない

    def write(self, text: str) -> None:
        with self._lock:
            self._f.write(text)
            self._f.flush()
            if self.echo:
                sys.stdout.write(text)
                sys.stdout.flush()

    def line(self, msg: str) -> None:
        self.write(f"[ec2run {time.strftime('%H:%M:%S')}] {msg}\n")

    def close(self) -> None:
        with self._lock:
            self._f.close()


class S3Store:
    """S3 の出し入れ。boto3 は最初に使うときに import する。テストでは client を差し込める。"""

    def __init__(self, bucket: str, region: str, client: Any = None):
        self.bucket, self.region = bucket, region
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3  # noqa: WPS433

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def download(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))

    def upload(self, path: Path, key: str) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def put_text(self, key: str, text: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=text.encode("utf-8"))

    def list(self, prefix: str):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                yield o["Key"], int(o["Size"])

    def sync_down(self, prefix: str, local_dir: Path) -> int:
        n = 0
        for key, size in self.list(prefix.rstrip("/") + "/"):
            rel = key[len(prefix.rstrip("/")) + 1:]
            if not rel or rel.endswith("/") or ".." in Path(rel).parts:
                continue
            dst = local_dir / rel
            if dst.exists() and dst.stat().st_size == size:
                continue
            self.download(key, dst)
            n += 1
        return n

    def sync_up(self, local_dir: Path, prefix: str) -> int:
        """大きさか中身が変わったものだけ上げる（毎回全部は上げない）。"""
        if not local_dir.exists():
            return 0
        remote = {k: s for k, s in self.list(prefix.rstrip("/") + "/")}
        n = 0
        for p in sorted(local_dir.rglob("*")):
            if not p.is_file():
                continue
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}"
            # 同じ大きさでも、実行のたびに書き換わるファイル（Excel・csv・json）は上げ直す
            if remote.get(key) == p.stat().st_size and p.suffix.lower() == ".wav":
                continue
            self.upload(p, key)
            n += 1
        return n


class LogUploader(threading.Thread):
    def __init__(self, log: RemoteLog, s3: S3Store, key: str, interval: float = LOG_INTERVAL):
        super().__init__(daemon=True)
        self.log, self.s3, self.key, self.interval = log, s3, key, interval
        self._halt = threading.Event()
        self._last = -1

    def push(self) -> None:
        try:
            size = self.log.path.stat().st_size
            if size != self._last:
                self.s3.upload(self.log.path, self.key)
                self._last = size
        except Exception as e:  # noqa: BLE001  ログの送信で本体を止めない
            sys.stderr.write(f"log upload failed: {e}\n")

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.push()

    def stop(self) -> None:
        self._halt.set()
        self.push()


class CommandRunner:
    """コマンドを実行し、出力をそのままログに流す。"""

    def __init__(self, log: RemoteLog, base_env: dict[str, str] | None = None):
        self.log = log
        self.base_env = dict(base_env or {})

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        e = os.environ.copy()
        e.update(self.base_env)
        e.update(extra or {})
        return e

    def run(self, cmd: list[str] | str, *, env: dict[str, str] | None = None, cwd: str | None = None) -> int:
        shown = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.log.line(f"$ {shown}")
        try:
            p = subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", env=self.env(env), cwd=cwd, bufsize=1)
        except OSError as e:
            self.log.line(f"実行できません: {e}")
            return 127
        assert p.stdout is not None
        for line in p.stdout:
            self.log.write(line)
        rc = p.wait()
        if rc != 0:
            self.log.line(f"終了コード {rc}")
        return rc

    def capture(self, cmd: list[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=self.env())
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except OSError as e:
            return 127, str(e)


def safe_extract(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if root != target and root not in target.parents:
                raise Fatal(f"tar に不正なパスがあります: {m.name}")
            if m.issym() or m.islnk():
                raise Fatal(f"tar にリンクがあります: {m.name}")
        tf.extractall(dest)   # noqa: S202 (上で検査済み)


def filter_requirements(text: str, drop_prefixes: list[str]) -> list[str]:
    """ノートブックの CosyVoice の手順と同じ: 空行と drop_prefixes で始まる行を除く。"""
    drop = tuple(p.lower() for p in drop_prefixes)
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if s and not s.lower().startswith(drop):
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, spec: dict[str, Any], s3: S3Store, runner: CommandRunner, log: RemoteLog,
                 clock: Callable[[], float] = time.time):
        self.spec, self.s3, self.run_cmd, self.log, self.clock = spec, s3, runner, log, clock
        self.work = Path(spec["workdir"])
        self.prefix: str = spec["prefix"]
        p = spec["paths"]
        self.repo = Path(p["repo"])
        self.src = Path(p["src"])
        self.result = Path(p["result"])
        self.cfg_path = Path(p["config"])
        self.hf_home = p["hf_home"]
        self.cosy_repo = p["cosy_repo"]
        self.envs_dir = self.work / "envs"
        self.uv = str(self.work / "bin" / "uv")
        self.started = clock()
        self.timeline: list[dict[str, Any]] = []
        self.engines: dict[str, dict[str, Any]] = {}
        self.commit: str | None = None
        self.done: dict[str, Any] = {}

    # --- 補助 ---
    def phase(self, name: str, **extra: Any) -> None:
        self.log.line(f"===== {name} =====")
        rec = {"phase": name, "t": round(self.clock() - self.started, 1), **extra}
        self.timeline.append(rec)
        self.write_status(finished=False, **extra)

    def write_status(self, finished: bool, **extra: Any) -> None:
        cur = self.timeline[-1]["phase"] if self.timeline else ""
        body = {"phase": cur, "finished": finished, "elapsed_s": round(self.clock() - self.started), **extra}
        try:
            self.s3.put_text(f"{self.prefix}/logs/status.json", json.dumps(body, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self.log.line(f"status を書けませんでした: {e}")

    def env_py(self, name: str) -> str:
        return str(self.envs_dir / name / "bin" / "python")

    def fmt(self, s: str) -> str:
        return s.replace("{cosy_repo}", self.cosy_repo).replace("{work}", str(self.work)).replace("{src}", str(self.src))

    def vl(self, env: str, *args: str) -> int:
        """voicelab を、仮想環境 env の Python で実行する（ノートブックの vl() と同じ）。"""
        return self.run_cmd.run([self.env_py(env), "-u", "-m", "voicelab", *args],
                                env={"PYTHONPATH": str(self.src), "HF_HOME": self.hf_home, "COSYVOICE_REPO": self.cosy_repo,
                                     "PYTHONUNBUFFERED": "1"})

    # --- 段階 ---
    def sysinfo(self) -> bool:
        rc, out = self.run_cmd.capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        has_gpu = rc == 0
        self.log.line(f"GPU: {out.strip() if has_gpu else '見つかりません'}")
        self.run_cmd.run("df -h / | tail -1; python3 --version; uname -r")
        return has_gpu

    def system_setup(self) -> None:
        apt = self.spec.get("apt", [])
        if apt:
            self.run_cmd.run(["apt-get", "-o", "DPkg::Lock::Timeout=600", "-qq", "update"])
            rc = self.run_cmd.run(["apt-get", "-o", "DPkg::Lock::Timeout=600", "-qq", "install", "-y", *apt])
            if rc != 0:
                self.log.line("apt のインストールに失敗しました。rubberband が無い場合、早回しは librosa で代用されます。続けます。")
        (self.work / "bin").mkdir(parents=True, exist_ok=True)
        if not Path(self.uv).exists():
            ver = self.spec["uv_version"]
            rc = self.run_cmd.run(f"curl -LsSf https://astral.sh/uv/{ver}/install.sh | env UV_UNMANAGED_INSTALL={self.work}/bin sh")
            if rc != 0 or not Path(self.uv).exists():
                self.log.line("uv の公式インストーラが使えませんでした。pip で入れます。")
                rc = self.run_cmd.run(f"python3 -m pip install -q uv=={ver} || python3 -m pip install -q --break-system-packages uv=={ver}")
                found = shutil.which("uv")
                if rc != 0 or not found:
                    raise Fatal("uv を入れられませんでした", 4)
                self.uv = found
        self.run_cmd.base_env.update({
            "UV_HTTP_TIMEOUT": "300", "UV_CACHE_DIR": str(self.work / "uv-cache"),
            "PATH": f"{self.work / 'bin'}:{os.environ.get('PATH', '')}", "HF_HOME": self.hf_home})
        self.run_cmd.run([self.uv, "--version"])

    def get_code(self) -> None:
        code = self.spec["code"]
        if code["mode"] == "git":
            if (self.repo / ".git").is_dir():
                self.run_cmd.run(["git", "-C", str(self.repo), "fetch", "origin", code["branch"]])
                self.run_cmd.run(["git", "-C", str(self.repo), "checkout", code["branch"]])
                rc = self.run_cmd.run(["git", "-C", str(self.repo), "pull", "--ff-only", "origin", code["branch"]])
            else:
                rc = self.run_cmd.run(["git", "clone", "--depth", "1", "--branch", code["branch"], code["url"], str(self.repo)])
            if rc != 0 or not self.src.is_dir():
                raise Fatal("リポジトリを取得できませんでした（非公開の場合は、パソコン側で --code local を付けて実行してください）", 5)
            rc, out = self.run_cmd.capture(["git", "-C", str(self.repo), "rev-parse", "HEAD"])
            self.commit = out.strip() if rc == 0 else None
        else:
            tmp = self.work / "code.tar.gz"
            self.s3.download(code["key"], tmp)
            self.src.parent.mkdir(parents=True, exist_ok=True)
            safe_extract(tmp, self.src)
            self.commit = code.get("client_commit")
            if not self.src.is_dir():
                raise Fatal("コードの展開に失敗しました", 5)
        self.log.line(f"コード: {code['mode']} / commit {self.commit or '不明'} / {self.src}")

    def restore_results(self) -> None:
        n = self.s3.sync_down(f"{self.prefix}/result", self.result)
        if n:
            self.log.line(f"前回の途中結果 {n} 件を S3 から戻しました（できている音声は作り直しません）")

    def make_env(self, spec: dict[str, Any]) -> bool:
        """ノートブックの make_env と同じ。成功すれば True。"""
        name = spec["name"]
        py = self.env_py(name)
        if spec.get("clone"):
            c = spec["clone"]
            dest = Path(self.fmt(c["dest"]))
            if not (dest / c.get("marker", "")).is_dir():
                cmd = ["git", "clone"] + (["--recursive"] if c.get("recursive") else []) + [c["url"], str(dest)]
                if self.run_cmd.run(cmd) != 0:
                    self.log.line(f"× {name}: リポジトリを取得できませんでした")
                    return False
        if not Path(py).exists():
            if self.run_cmd.run([self.uv, "venv", "--python", spec["python"], str(self.envs_dir / name)]) != 0:
                self.log.line(f"× 仮想環境 {name} を作れませんでした")
                return False
        for args in spec.get("pre", []):
            if self.run_cmd.run([self.uv, "pip", "install", "--python", py, *args]) != 0:
                self.log.line(f"× {name}: {args} を入れられませんでした")
                return False
        packages = list(spec["packages"])
        if spec.get("requirements_file"):
            req = Path(self.fmt(spec["requirements_file"]))
            try:
                packages = filter_requirements(req.read_text(encoding="utf-8"), spec.get("drop_prefixes", [])) + packages
            except OSError as e:
                self.log.line(f"× {name}: requirements を読めません: {e}")
                return False
        if self.run_cmd.run([self.uv, "pip", "install", "--python", py, *packages]) != 0:
            self.log.line(f"× {name}: インストールに失敗しました。上のメッセージを確認してください")
            return False
        for args in spec.get("optional_after", []):
            if self.run_cmd.run([self.uv, "pip", "install", "--python", py, *args]) != 0:
                self.log.line(f"{name}: {args} は入りませんでした（なくても動きます。測定が遅くなることがあります）")
        self.log.line(f"○ 環境 {name} ができました")
        return True

    def check_import(self, name: str, module: str) -> bool:
        rc = self.run_cmd.run([self.env_py(name), "-c", f"import {module}; print('import {module}: OK')"])
        return rc == 0

    def prepare_inputs(self) -> None:
        for inp in self.spec.get("inputs", []):
            dest = Path(self.fmt(inp["dest"]))
            if inp["kind"] == "tar":
                tmp = self.work / "tmp" / Path(inp["key"]).name
                self.s3.download(inp["key"], tmp)
                safe_extract(tmp, dest)
            else:
                self.s3.download(inp["key"], dest)
            self.log.line(f"入力を用意しました: {dest}")
        py = self.env_py("score")
        if self.spec.get("public_test"):
            pt = self.spec["public_test"]
            helper = self.work / "prepare_public.py"
            self.s3.download(pt["helper_key"], helper)
            self.run_cmd.run([self.uv, "pip", "install", "--python", py, "pyarrow", "huggingface_hub", "soundfile"])
            rc = self.run_cmd.run([py, str(helper), "--src", str(self.src), "--out", self.fmt(pt["dest"])])
            if rc != 0:
                raise Fatal("公開データを用意できませんでした", 6)
        if self.spec.get("default_targets"):
            dest = self.fmt(self.spec["default_targets"])
            code = f"from voicelab.texts import write_targets; write_targets({dest!r})"
            if self.run_cmd.run([py, "-c", code], env={"PYTHONPATH": str(self.src)}) != 0:
                raise Fatal("既定の targets.csv を作れませんでした", 6)
        self.s3.download(self.spec["config_key"], self.cfg_path)

    def dry_run(self) -> None:
        cfg = json.loads(self.cfg_path.read_text(encoding="utf-8"))
        cfg.update({"engines": cfg["engines"][:1], "repeats": 1, "max_texts_per_lang": 1, "embedder": "none"})
        dry = self.cfg_path.with_name("dry_exp.json")
        dry.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        out = self.cfg_path.parent / "dryrun"
        rc = self.vl("score", "run", "--config", str(dry), "--out", str(out), "--dry-run", "--embedder", "none")
        if rc != 0:
            raise Fatal("ダミーでの確認に失敗しました。設定か声のデータの間違いです（GPU の時間を使う前に止めました）", 7)
        self.log.line("ダミーでの確認: 成功")

    def generate_all(self) -> None:
        for g in self.spec["engine_groups"]:
            name = g["name"]
            info = self.engines.setdefault(name, {})
            self.phase(f"engine:{name}")
            t0 = self.clock()
            if not self.make_env(g["env"]):
                info.update(env_ok=False, generate_rc=None)
                continue
            info["env_ok"] = True
            if g["env"].get("check_import"):
                info["import_ok"] = self.check_import(name, g["env"]["check_import"])
            rc = self.vl(name, "generate", "--config", str(self.cfg_path), "--out", str(self.result), "--engines", name)
            info["generate_rc"] = rc
            info["seconds"] = round(self.clock() - t0)
            self.run_cmd.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
            n = self.s3.sync_up(self.result, f"{self.prefix}/result")      # エンジンが終わるごとに保存
            self.log.line(f"{name}: 途中結果を S3 に保存しました（{n} 件）")

    def score(self) -> int:
        rc = self.vl("score", "score", "--config", str(self.cfg_path), "--out", str(self.result), "--embedder", self.spec["embedder"])
        return rc

    def manifest_counts(self) -> dict[str, dict[str, int]]:
        man = self.result / "manifest.csv"
        out: dict[str, Counter] = {}
        if man.exists():
            with man.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    out.setdefault(row.get("engine", "?"), Counter())[row.get("status", "?")] += 1
        return {k: dict(v) for k, v in out.items()}

    # --- 全体 ---
    def publish_done(self) -> None:
        """done.json（完了の印）。ログの最終アップロードより後に書く（パソコン側はこれを見て終わりと判断する）。"""
        try:
            self.s3.put_text(f"{self.prefix}/logs/done.json", json.dumps(self.done, ensure_ascii=False, indent=1))
        except Exception as e:  # noqa: BLE001
            self.log.line(f"done.json を書けませんでした: {e}")

    def run(self, publish_done: bool = True) -> int:
        exit_code = 1
        ok = False
        score_rc: int | None = None
        error: str | None = None
        try:
            self.phase("sysinfo")
            has_gpu = self.sysinfo()
            if not has_gpu and not self.spec.get("allow_no_gpu"):
                raise Fatal("GPU が見つかりません（nvidia-smi が使えません）。GPU 付きのインスタンスか確認してください", 3)
            self.phase("system")
            self.system_setup()
            self.phase("code")
            self.get_code()
            self.phase("resume")
            self.restore_results()
            self.phase("env:score")
            if not self.make_env(self.spec["score_env"]):
                raise Fatal("測定用の環境（score）を作れませんでした", 4)
            self.phase("inputs")
            self.prepare_inputs()
            self.phase("plan")
            if self.vl("score", "plan", "--config", str(self.cfg_path), "--out", str(self.result)) != 0:
                raise Fatal("設定の確認（plan）に失敗しました。設定ファイルか声のデータを見直してください", 5)
            if self.spec.get("dry_run_first"):
                self.phase("dry-run")
                self.dry_run()
            self.generate_all()
            self.phase("score")
            score_rc = self.score()
            counts = self.manifest_counts()
            generated = sum(c.get("ok", 0) for c in counts.values())
            ok = score_rc == 0 and generated > 0
            exit_code = 0 if ok else 1
        except Fatal as f:
            error = str(f)
            exit_code = f.exit_code
            self.log.line(f"中止: {f}")
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            self.log.line("予期しないエラー:\n" + traceback.format_exc())
            exit_code = 1
        finally:
            try:
                n = self.s3.sync_up(self.result, f"{self.prefix}/result")
                self.log.line(f"最終結果を S3 に保存しました（{n} 件）: s3://{self.s3.bucket}/{self.prefix}/result/")
            except Exception as e:  # noqa: BLE001
                self.log.line(f"最終結果の保存に失敗: {e}")
            self.done = {"ok": ok, "exit_code": exit_code, "error": error, "score_rc": score_rc, "commit": self.commit,
                         "engines": self.engines, "manifest": self.manifest_counts() if self.result.exists() else {},
                         "elapsed_s": round(self.clock() - self.started), "timeline": self.timeline}
            self.log.line(f"結果: {'成功' if ok else '失敗または一部失敗'}（所要 {self.done['elapsed_s']} 秒）")
            if self.spec.get("shutdown_when_done"):
                self.log.line("--auto-terminate: このあと 1 分でインスタンスを停止（= 終了）します")
            self.write_status(finished=True)
            if publish_done:
                self.publish_done()
        return exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ec2run のインスタンス側の実行役")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True, help="S3 のキー接頭辞（例 exp/exp01）")
    ap.add_argument("--region", required=True)
    ap.add_argument("--workdir", default="/opt/voicelab")
    args = ap.parse_args(argv)

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    s3 = S3Store(args.bucket, args.region)
    spec_path = work / "job.json"
    s3.download(f"{args.prefix}/job/job.json", spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    log = RemoteLog(work / "run.log")
    uploader = LogUploader(log, s3, f"{args.prefix}/logs/run.log")
    uploader.start()
    runner = CommandRunner(log, {"HOME": os.environ.get("HOME", "/root"), "DEBIAN_FRONTEND": "noninteractive",
                                 "AWS_DEFAULT_REGION": args.region})
    try:
        job = Job(spec, s3, runner, log)
        rc = job.run(publish_done=False)
    finally:
        uploader.stop()          # ログの最終アップロード
        if "job" in locals():
            job.publish_done()   # そのあとに完了の印を書く
        log.close()
    if spec.get("shutdown_when_done"):
        subprocess.run(["shutdown", "-h", "+1", "ec2run: finished (auto-terminate)"], check=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
