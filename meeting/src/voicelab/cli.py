"""コマンドライン入口: python -m voicelab run / generate / score / plan / init / engines

  python -m voicelab run --config exp.yaml --out outputs/exp01          生成 → 測定 → Excel まで通し
  python -m voicelab run --config exp.yaml --out outputs/exp01 --dry-run  モデルなし（ダミー音声）で流れだけ確認
  python -m voicelab generate --config exp.json --out OUT --engines voxcpm2   1 エンジンだけ生成（GPU）
  python -m voicelab score --config exp.json --out OUT                  生成済みの音声を測って Excel を作る（CPU）
  python -m voicelab plan --config exp.yaml --out outputs/exp01          何件作るかだけ表示
  python -m voicelab init --out my_exp                                   exp.yaml と targets.csv のひな形を作る
  python -m voicelab engines                                             エンジンの一覧と、入っているかの確認
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, load_config
from .engines import FORBIDDEN, EngineError, available_engines, engine_class
from .runner import (
    build_all_jobs,
    load_texts,
    prepare_speakers,
    run_generate,
    summarize_plan,
)
from .scoring import PairsLeakError, run_score
from .speakers import SpeakerError
from .templates import write_templates


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def _split(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _cfg(args):
    cfg = load_config(args.config)
    return cfg


def cmd_run(args) -> int:
    cfg = _cfg(args)
    out = Path(args.out)
    run_generate(cfg, out, dry_run=args.dry_run, only_engines=_split(args.engines), force=args.force)
    if args.no_score:
        say("測定は飛ばしました（--no-score）。あとで `python -m voicelab score` を実行してください")
        return 0
    run_score(cfg, out, embedder=args.embedder, calib=args.calib, threshold=args.threshold, dry_run=args.dry_run)
    return 0


def cmd_generate(args) -> int:
    cfg = _cfg(args)
    run_generate(cfg, Path(args.out), dry_run=args.dry_run, only_engines=_split(args.engines), force=args.force)
    return 0


def cmd_score(args) -> int:
    cfg = _cfg(args)
    run_score(cfg, Path(args.out), embedder=args.embedder, calib=args.calib, threshold=args.threshold, dry_run=args.dry_run)
    return 0


def cmd_plan(args) -> int:
    cfg = _cfg(args)
    out = Path(args.out)
    prepared = prepare_speakers(cfg, out, write=False)
    texts = load_texts(cfg)
    jobs = build_all_jobs(cfg, prepared, texts, args.dry_run)
    p = summarize_plan(jobs)
    say(f"Job：合計 {p['total']}（生成 {p['generate']} / 早回しの後処理 {p['derived']} / 対応外で飛ばす {p['skipped']}）")
    say("エンジン別の生成数：")
    for e, n in p["by_engine"].items():
        say(f"  {e:28s} {n}")
    say("条件別（早回し含む）：")
    for c, n in sorted(p["by_condition"].items()):
        say(f"  {c:28s} {n}")
    for ps in prepared:
        say(f"話者 {ps.plan.name}：手本 {len(ps.profile_refs)} 本 / C0 の手本 {len(ps.line_refs)} 本 / "
            f"評価用の実声 {len(ps.plan.score_refs)} 本（実声どうしの上限 {len(ps.plan.ceiling_clips)} 本）")
    say("目安：1 件あたり GPU で 5〜20 秒（エンジン・文の長さによる）。測定は音声 1 秒あたり CPU で約 0.5 秒（4 コア）。")
    return 0


def cmd_init(args) -> int:
    files = write_templates(args.out)
    say("ひな形を作りました:")
    for f in files:
        say(f"  {f}")
    say("exp.yaml の speakers を自分の登録フォルダに直してから、`python -m voicelab run --config exp.yaml --out outputs/exp01 --dry-run` で流れを確認してください")
    return 0


def cmd_engines(args) -> int:
    probes = {
        "voxcpm2": ("voxcpm", "pip install voxcpm==2.0.3"),
        "qwen3tts": ("qwen_tts", "pip install qwen-tts==0.1.1"),
        "cosyvoice3": ("cosyvoice", "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git（pip では入らない）"),
        "chatterbox": ("chatterbox", "pip install chatterbox-tts==0.1.7"),
    }
    for name in available_engines():
        cls = engine_class(name)
        mod, how = probes.get(name, (name, ""))
        has = importlib.util.find_spec(mod) is not None
        if name == "cosyvoice3":
            has = has or os.path.isdir(os.environ.get("COSYVOICE_REPO", "/content/CosyVoice"))
        variants = ", ".join(v for v in cls.VARIANTS if v) or "-"
        say(f"{name:12s} {'入っている' if has else '入っていない'}  variant: {variants}   {'' if has else '→ ' + how}")
    say("使わないもの（ライセンス）: " + ", ".join(sorted({v.split(' ')[0] for v in FORBIDDEN.values()})))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicelab",
        description="声の似かた（本人らしさ）の比較実験：条件を1つずつ変えて TTS を動かし、voicecheck で測って Excel にまとめます。",
    )
    p.add_argument("--version", action="version", version=f"voicelab {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{run,generate,score,plan,init,engines}")

    def common(sp, needs_config=True):
        if needs_config:
            sp.add_argument("--config", required=True, help="実験の設定ファイル（.yaml / .json）")
        sp.add_argument("--out", required=True, help="結果を入れるフォルダ（例 outputs/exp01）")
        sp.add_argument("--dry-run", action="store_true", help="モデルを使わずダミー音声で流れだけ確認する")

    def scoring(sp):
        sp.add_argument("--embedder", default=None, help="voicecheck の埋め込み（ecapa / resemblyzer / all / none）。省略時は設定の embedder")
        sp.add_argument("--calib", default=None, help="voicecheck calibrate の Excel（合格ライン）")
        sp.add_argument("--threshold", default=None, help="合格ラインを直接指定（例 0.45 または ecapa=0.45）")

    r = sub.add_parser("run", help="生成 → 測定 → Excel まで通しで実行")
    common(r)
    r.add_argument("--engines", default=None, help="このエンジンだけ生成（カンマ区切り。例 voxcpm2,chatterbox）")
    r.add_argument("--force", action="store_true", help="できている音声も作り直す")
    r.add_argument("--no-score", action="store_true", help="生成だけして測定は飛ばす")
    scoring(r)
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("generate", help="音声の生成だけ（GPU が要る部分）")
    common(g)
    g.add_argument("--engines", default=None, help="このエンジンだけ生成（カンマ区切り）")
    g.add_argument("--force", action="store_true", help="できている音声も作り直す")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="生成済みの音声を測って Excel を作る（CPU でよい）")
    common(s)
    scoring(s)
    s.set_defaults(func=cmd_score)

    pl = sub.add_parser("plan", help="何件作るかだけ表示（何も生成しない）")
    common(pl)
    pl.set_defaults(func=cmd_plan)

    i = sub.add_parser("init", help="exp.yaml と targets.csv のひな形を作る")
    i.add_argument("--out", required=True, help="ひな形を書くフォルダ")
    i.set_defaults(func=cmd_init)

    e = sub.add_parser("engines", help="エンジンの一覧と、入っているかの確認")
    e.set_defaults(func=cmd_engines)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    debug = bool(os.environ.get("VOICELAB_DEBUG"))
    try:
        return int(args.func(args))
    except (ConfigError, SpeakerError, EngineError, PairsLeakError, FileNotFoundError) as e:
        if debug:
            raise
        say(f"エラー: {e}")
        return 2
    except RuntimeError as e:
        if debug:
            raise
        say(f"エラー: {e}")
        return 2
    except PermissionError as e:
        say(f"エラー: ファイルに書き込めません（{e}）。Excel で開いている場合は閉じてからやり直してください")
        return 2
    except KeyboardInterrupt:
        say("中断しました（できた分は manifest.csv に残っています。同じコマンドで続きから再開できます）")
        return 130


if __name__ == "__main__":
    sys.exit(main())
