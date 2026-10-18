"""コマンドライン入口: voicecheck compare / calibrate / features"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .analysis import Analyzer
from .audio import AUDIO_EXTS
from .calibrate import CalibError, find_speaker_dirs, run_calibrate
from .compare import run_compare, summarize
from .embedders import CHOICES, load_embedders
from .features import FEATURE_KEYS, backend_status
from .pairs import PairsError, read_pairs
from .report import (
    base_info,
    read_calibration_thresholds,
    write_calibration,
    write_compare,
    write_features,
)

HERE_ROOT = "（直下）"


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# しきい値の指定: "0.5" / "ecapa=0.5,resemblyzer=0.8" / --calib calib.xlsx
# ---------------------------------------------------------------------------
def parse_threshold(arg: str | None, names: list[str]) -> dict[str, float]:
    if not arg:
        return {}
    arg = arg.strip()
    try:
        value = float(arg)
    except ValueError:
        out: dict[str, float] = {}
        for part in arg.split(","):
            key, sep, val = part.partition("=")
            try:
                if not sep:
                    raise ValueError
                out[key.strip().lower()] = float(val)
            except ValueError:
                raise ValueError(
                    f"--threshold の書き方が違います: '{arg}'\n"
                    "  例:  --threshold 0.45   または   --threshold ecapa=0.45,resemblyzer=0.8"
                ) from None
        return out
    if len(names) > 1:
        say(f"注意: --threshold {value} を {', '.join(names)} の全部に使います。"
            "埋め込みごとに値が違うので、ecapa=0.45,resemblyzer=0.8 のように分けて指定するのがおすすめです。")
    return {n: value for n in names}


def resolve_thresholds(arg: str | None, calib: str | None, names: list[str]) -> tuple[dict[str, float], str]:
    """(使うしきい値 {埋め込み名: 値}, 出どころの説明) を返す。--threshold が --calib より優先。"""
    th: dict[str, float] = {}
    src: list[str] = []
    if calib:
        got = read_calibration_thresholds(calib)
        th.update({k: v for k, v in got.items() if k in names})
        src.append(f"較正ファイル {calib}")
    if arg:
        th.update({k: v for k, v in parse_threshold(arg, names).items() if k in names})
        src.append("--threshold")
    missing = [n for n in names if n not in th]
    if th and missing:
        say(f"注意: {', '.join(missing)} のしきい値がないので、その列の赤塗り・合格率は出ません")
    return th, " + ".join(src) if src else "(指定なし)"


def _fmt_thresholds(th: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in th.items()) if th else "(指定なし: 赤塗り・合格率は出ません)"


def _embedder_info(embedders, failed: dict[str, str]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = [("使った埋め込み", ", ".join(e.name for e in embedders) or "(なし: 特徴量だけ)")]
    for name, why in failed.items():
        items.append((f"使えなかった埋め込み: {name}", why))
    bs = backend_status()
    if not bs["pyworld"]:
        items.append(("使えなかった機能: pyworld", "声の高さは librosa.pyin で代用しました (遅く、結果が少し違います)"))
    if not bs["parselmouth"]:
        items.append(("使えなかった機能: parselmouth", "CPPS は空欄です"))
    return items


def _load_embedders(spec: str):
    embedders, failed = load_embedders(spec)
    for name, why in failed.items():
        say(f"警告: 埋め込み {name} を使えません。この部分は空欄になります。\n  理由: {why}")
    return embedders, failed


def _f0_counts(analyzer: Analyzer) -> str:
    c = Counter(str(cl.features.get("f0_method", "")) for cl in analyzer.clips() if cl.ok and cl.features)
    return ", ".join(f"{k}: {v}ファイル" for k, v in c.items() if k) or "-"


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def cmd_compare(args: argparse.Namespace) -> int:
    rows = read_pairs(args.pairs)
    if not rows:
        raise PairsError("pairs.csv に評価する行がありません")
    say(f"pairs.csv: {len(rows)} 行を読みました。モデルを準備しています...")
    embedders, failed = _load_embedders(args.embedder)
    names = [e.name for e in embedders]
    thresholds, th_src = resolve_thresholds(args.threshold, args.calib, names)

    analyzer = Analyzer(embedders, with_features=True, f0_method=args.f0)
    results = run_compare(rows, analyzer, names)
    summary = summarize(results, names, thresholds)

    n_notes = sum(1 for r in results if r.notes)
    info = base_info(
        f"voicecheck compare --pairs {args.pairs} --out {args.out} --embedder {args.embedder}",
        [
            ("入力 pairs.csv", str(Path(args.pairs).resolve())),
            ("行数", len(rows)),
            ("備考のある行", n_notes),
            *_embedder_info(embedders, failed),
            ("しきい値", _fmt_thresholds(thresholds)),
            ("しきい値の出どころ", th_src),
            ("F0の方式", _f0_counts(analyzer)),
        ],
    )
    out = write_compare(args.out, results, summary, names, thresholds, info)

    say("")
    say(f"完了: {out.resolve()}")
    say(f"  行数 {len(results)} / 備考のある行 {n_notes}")
    for n in names:
        v = [r.sims[n] for r in results if n in r.sims]
        if v:
            msg = f"  SIM({n}): 平均 {np.mean(v):.3f} / 最小 {np.min(v):.3f} / 最大 {np.max(v):.3f}"
            if n in thresholds:
                msg += f" / しきい値 {thresholds[n]:.3f} 以上の割合 {np.mean([x >= thresholds[n] for x in v]):.0%}"
            say(msg)
    if not names:
        say("  (埋め込みを使えなかったので、SIM は出していません。特徴量の差だけ書きました)")
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
def cmd_calibrate(args: argparse.Namespace) -> int:
    find_speaker_dirs(args.enroll_dir)  # フォルダの形が違えば、モデルを読み込む前にエラーにする
    say("モデルを準備しています...")
    embedders, failed = _load_embedders(args.embedder)
    if not embedders:
        raise CalibError("使える埋め込みがありません (calibrate は埋め込みが必要です)。上の警告を確認してください")
    calib = run_calibrate(args.enroll_dir, embedders)
    info = base_info(
        f"voicecheck calibrate --enroll-dir {args.enroll_dir} --out {args.out} --embedder {args.embedder}",
        [("登録フォルダ", str(Path(args.enroll_dir).resolve())), *_embedder_info(embedders, failed)],
    )
    out = write_calibration(args.out, calib, info)

    say("")
    say(f"完了: {out.resolve()}")
    for c in calib.per_embedder:
        if c.eer is None:
            say(f"  [{c.name}] EER を出せませんでした: {c.note}")
            continue
        say(
            f"  [{c.name}] 話者 {c.n_speakers} / 音声 {c.n_files} / 同じ人ペア {c.n_same} / 別の人ペア {c.n_diff}\n"
            f"           同じ人 SIM 平均 {c.same_mean:.3f}, 別の人 SIM 平均 {c.diff_mean:.3f}\n"
            f"           EER {c.eer.eer:.2%}  おすすめしきい値 {c.eer.threshold:.3f}  (厳しめ {c.strict_threshold:.3f})"
        )
    th = " ".join(f"{c.name}={c.eer.threshold:.3f}" for c in calib.per_embedder if c.eer)
    if th:
        say(f"次は: voicecheck compare --pairs pairs.csv --calib {args.out}   (または --threshold {th.replace(' ', ',')})")
    return 0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _collect_audio(target: Path) -> list[tuple[str, Path]]:
    """(話者フォルダ名, ファイル) のリスト。フォルダ直下のファイルは『（直下）』。"""
    if target.is_file():
        return [(HERE_ROOT, target)]
    if not target.is_dir():
        raise FileNotFoundError(f"見つかりません: {target}")
    files = sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    out = []
    for p in files:
        rel = p.relative_to(target)
        out.append((rel.parts[0] if len(rel.parts) > 1 else HERE_ROOT, p))
    return out


def cmd_features(args: argparse.Namespace) -> int:
    items = _collect_audio(Path(args.input))
    if not items:
        raise FileNotFoundError(f"音声ファイル ({', '.join(sorted(AUDIO_EXTS))}) が見つかりません: {args.input}")
    say(f"{len(items)} ファイルの特徴量を測ります...")
    analyzer = Analyzer([], with_features=True, f0_method=args.f0)
    analyzer.prefetch([str(p) for _, p in items])

    rows: list[dict] = []
    for no, (spk, p) in enumerate(items, start=1):
        clip = analyzer.get(str(p))
        row: dict[str, Any] = {"no": no, "speaker": spk, "file": p.name}
        row.update(clip.features)
        row["notes"] = clip.error if not clip.ok else "\n".join(clip.notes)
        rows.append(row)

    speaker_rows = []
    for spk in dict.fromkeys(r["speaker"] for r in rows):
        group = [r for r in rows if r["speaker"] == spk and "f0_median_hz" in r]
        s: dict[str, Any] = {"speaker": spk, "n_files": len(group), "total_s": sum(r.get("duration_s", 0) or 0 for r in group)}
        for k in FEATURE_KEYS:
            if k == "duration_s":
                continue
            vals = [r[k] for r in group if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
            s[k] = float(np.mean(vals)) if vals else None
        speaker_rows.append(s)

    info = base_info(
        f"voicecheck features --in {args.input} --out {args.out}",
        [("入力", str(Path(args.input).resolve())), ("ファイル数", len(items)),
         *_embedder_info([], {}), ("F0の方式", _f0_counts(analyzer))],
    )
    out = write_features(args.out, rows, speaker_rows, info)
    say("")
    say(f"完了: {out.resolve()}  ({len(rows)} ファイル)")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicecheck",
        description="声の分析カルテ: 合成音声が本人の声にどれだけ似ているかと、声の特徴を測って Excel に出します。",
    )
    p.add_argument("--version", action="version", version=f"voicecheck {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{compare,calibrate,features}")

    emb_help = "話者埋め込み: ecapa / resemblyzer / all (両方、既定) / none (特徴量だけ)"
    c = sub.add_parser("compare", help="手本と評価音声を比べる (SIM と特徴量の差)")
    c.add_argument("--pairs", required=True, help="pairs.csv (列: speaker, ref, test, 任意: source, label, emotion_src, emotion_test)")
    c.add_argument("--out", default="report.xlsx", help="出力する Excel (既定 report.xlsx)")
    c.add_argument("--embedder", default="all", choices=CHOICES, help=emb_help)
    c.add_argument("--threshold", help="合格ライン。0.45 または ecapa=0.45,resemblyzer=0.8")
    c.add_argument("--calib", help="calibrate が作った Excel。EER しきい値を自動で読む (--threshold が優先)")
    c.add_argument("--f0", default="harvest", choices=["harvest", "pyin"], help="声の高さの測り方 (既定 harvest)")
    c.set_defaults(func=cmd_compare)

    k = sub.add_parser("calibrate", help="登録音声から合格ライン (EER しきい値) を決める")
    k.add_argument("--enroll-dir", required=True, help="話者ごとのサブフォルダに音声(wav 等)を入れたフォルダ")
    k.add_argument("--out", default="calib.xlsx", help="出力する Excel (既定 calib.xlsx)")
    k.add_argument("--embedder", default="all", choices=CHOICES, help=emb_help)
    k.set_defaults(func=cmd_calibrate)

    f = sub.add_parser("features", help="声の特徴量だけを測る")
    f.add_argument("--in", dest="input", required=True, help="wav ファイル、または音声の入ったフォルダ")
    f.add_argument("--out", default="features.xlsx", help="出力する Excel (既定 features.xlsx)")
    f.add_argument("--f0", default="harvest", choices=["harvest", "pyin"], help="声の高さの測り方 (既定 harvest)")
    f.set_defaults(func=cmd_features)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows のコンソールで文字化けしても落ちないように
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    debug = bool(os.environ.get("VOICECHECK_DEBUG"))  # 1 にすると想定外のエラーでも詳細を出す
    try:
        return int(args.func(args))
    except (PairsError, CalibError, FileNotFoundError, ValueError) as e:
        if debug:
            raise
        say(f"エラー: {e}")
        return 2
    except PermissionError as e:
        say(f"エラー: ファイルに書き込めません ({e})。Excel で開いている場合は閉じてからやり直してください")
        return 2
    except KeyboardInterrupt:
        say("中断しました")
        return 130


if __name__ == "__main__":
    sys.exit(main())
