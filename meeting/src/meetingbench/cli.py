"""コマンドライン入口: python -m meetingbench build / diarize / asr / report"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd_build(args: argparse.Namespace) -> int:
    from .build import BuildParams, build_pseudo_meeting
    from .jvnv import load_index

    clips = load_index(args.jvnv)
    params = BuildParams(seed=args.seed, minutes_min=args.minutes[0], minutes_max=args.minutes[1],
                         overlap_ratio=args.overlap_ratio)
    m = build_pseudo_meeting(clips, args.out, params, variants=tuple(args.variants), keep_48k=args.keep_48k, progress=say)
    for v, st in m["variants"].items():
        say(f"[{v}] {st['audio_sec'] / 60:.1f} 分 / 発言 {st['turns']} 件 / 重ねた発言 {st['overlapped_turns']} 件 "
            f"({st['overlap_ratio']:.1%}) / 話者ごとの話す時間(分) {st['speech_min_per_speaker']}")
    say(f"完了: {Path(args.out).resolve()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from . import diarize as dz
    from . import pipeline as pl

    lay = pl.Layout(args.data)
    for v in args.variant:
        if args.stage in ("seg", "all"):
            pl.stage_segments(lay, v, args.seg, force=args.force, say=say)
        if args.stage in ("asr", "all"):
            pl.stage_asr(lay, v, args.seg, force=args.force, threads=args.threads, say=say)
        if args.stage in ("diar", "all"):
            for cfg in args.backend:
                for n in args.num_speakers:
                    pl.stage_diar(lay, v, cfg, n, force=args.force, say=say)
        if args.stage in ("assign", "all"):
            pl.stage_assign(lay, v, say=say)
    return 0


def cmd_sanity(args: argparse.Namespace) -> int:
    import json

    from . import pipeline as pl
    from .sanity import ecapa_check

    lay = pl.Layout(args.data)
    for v in args.variant:
        res = ecapa_check(lay.wav(v), lay.ref(v), k=args.k)
        out = lay.runs(v) / "sanity_ecapa.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        say(f"[{v}] ECAPA で正解の発言をクラスタリング（人数 {args.k}）: 発言単位の正解率 {res['accuracy']:.1%}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from . import model_fetch as mf

    mf.fetch_builtin(say)
    for k in args.only or list(mf.MS_MODELS):
        mf.fetch_modelscope(k, say)
    say("モデルの準備ができました")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import excel, pipeline as pl, reportinfo, scoring

    lay = pl.Layout(args.data)
    variants = [v for v in args.variant if (lay.runs(v) / "cond").is_dir()]
    if not variants:
        raise ValueError("評価用の結果がありません。先に run --stage assign（または all）を実行してください")
    conds, segrows = [], []
    for v in variants:
        say(f"[{v}] 採点しています（asrcheck）...")
        conds += scoring.score_variant(lay, v)
        segrows += scoring.segment_report(lay, v)
    manifest = reportinfo.load_manifest(lay)
    recs = {v: {"minutes": manifest["variants"][v]["audio_sec"] / 60, "turns": manifest["variants"][v]["turns"],
                "overlapped": manifest["variants"][v]["overlapped_turns"]} for v in variants if v in manifest.get("variants", {})}
    cmd = "python -m meetingbench report " + " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "python -m meetingbench report"
    info = reportinfo.collect_info(lay, cmd, manifest, conds)
    names = {c.name for c in conds}
    detail = [n for v in variants for n in (f"{v}:silero/built-ja/n4", f"{v}:oracle/built-ja/n4", f"{v}:silero/cam/n4") if n in names]
    out = excel.write_workbook(args.out, conds, segrows, info, excel.best_lines(conds, scoring.DEFAULT_KPIS), recs, detail_for=detail,
                               not_measured=reportinfo.not_measured(conds, variants))
    say(f"完了: {Path(out).resolve()}")
    return 0


def cmd_enroll_embed(args: argparse.Namespace) -> int:
    import os

    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, str(args.threads))      # CPU を取り合う環境では 1 本に固定する（モデルを読み込む前に設定）
    from . import enroll_embed as ee
    from . import pipeline as pl

    lay = pl.Layout(args.data)
    for name in args.embedder:
        ee.run_embedding(name, lay, args.jvnv, say=say, threads=args.threads, pool_sec=args.pool_sec, windows=args.windows)
    return 0


def cmd_enroll_report(args: argparse.Namespace) -> int:
    from . import enroll_report as er
    from . import pipeline as pl

    cmd = "python -m meetingbench enroll-report " + " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "python -m meetingbench enroll-report"
    er.run_report(pl.Layout(args.data), args.out, embedders=args.embedder, secs=tuple(args.secs), seeds=range(args.seeds), say=say,
                  force=args.force, command=cmd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .diarize import BACKENDS, NUM_SPEAKERS
    from .segments import SEG_KINDS

    p = argparse.ArgumentParser(prog="meetingbench",
                                description="公開データ（JVNV）の疑似会議で、pyVideoTrans の話者分離と文字起こしを測る。")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="疑似会議（A: 重ならない / B: 重なる）と正解の書き起こしを作る")
    b.add_argument("--jvnv", default="data/testdata/jvnv", help="JVNV のフォルダ（F1/F2/M1/M2 と transcription_no_nv.csv）")
    b.add_argument("--out", default="data/testdata/pseudo_meeting", help="出力フォルダ")
    b.add_argument("--seed", type=int, default=20261019, help="乱数の種（同じ種なら同じ会議ができる）")
    b.add_argument("--minutes", type=float, nargs=2, default=[6.0, 8.0], metavar=("最小", "最大"), help="1人が話す時間（分）の範囲")
    b.add_argument("--overlap-ratio", type=float, default=0.125, help="B で重ねる発言の割合（既定 0.125 = 12.5%%）")
    b.add_argument("--variants", nargs="+", default=["A", "B"], choices=["A", "B"])
    b.add_argument("--keep-48k", action="store_true", help="48kHz の複製も残す（1本約 180MB。既定は 16kHz だけ。同じ種で作り直せる）")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="測定の段階を実行する（seg=区間, asr=文字起こし, diar=話者分離, assign=評価用に整える, all=全部）")
    r.add_argument("--stage", required=True, choices=["seg", "asr", "diar", "assign", "all"])
    r.add_argument("--data", default="data/testdata/pseudo_meeting", help="build の出力フォルダ")
    r.add_argument("--variant", nargs="+", default=["A", "B"], choices=["A", "B"])
    r.add_argument("--seg", nargs="+", default=list(SEG_KINDS), choices=list(SEG_KINDS), help="区間の種類")
    r.add_argument("--backend", nargs="+", default=list(BACKENDS), choices=list(BACKENDS), help="話者分離の条件")
    r.add_argument("--num-speakers", nargs="+", default=list(NUM_SPEAKERS), choices=list(NUM_SPEAKERS), help="n4=4人と指定 / auto=自動")
    r.add_argument("--threads", type=int, default=2, help="torch の CPU スレッド数（文字起こし）")
    r.add_argument("--force", action="store_true", help="結果があっても作り直す")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("models", help="必要なモデルを取得する（あれば飛ばす。合計 約1.4GB。TLS は検証する）")
    m.add_argument("--only", nargs="+", choices=["cam", "sensevoice", "fsmn_vad"], help="ModelScope のモデルのうち取得するもの（built 用 onnx は常に確認）")
    m.set_defaults(func=cmd_models)

    y = sub.add_parser("sanity", help="参考値: 正解の発言を ECAPA 埋め込み＋素朴なクラスタリングで分けた正解率（voicecheck の環境で実行: uv run）")
    y.add_argument("--data", default="data/testdata/pseudo_meeting")
    y.add_argument("--variant", nargs="+", default=["A", "B"], choices=["A", "B"])
    y.add_argument("-k", type=int, default=4, help="クラスタの数（人数）")
    y.set_defaults(func=cmd_sanity)

    e = sub.add_parser("enroll-embed", help="登録音声との照合の準備: 登録候補と字幕の区間の話者埋め込みを作って保存（モデルを動かす。続きから再開できる）")
    e.add_argument("--data", default="data/testdata/pseudo_meeting")
    e.add_argument("--jvnv", default="data/testdata/jvnv", help="JVNV のフォルダ（登録に使う発話を選ぶ）")
    e.add_argument("--embedder", nargs="+", required=True, choices=["ecapa", "titanet", "eres2net", "campp"],
                   help="ecapa は meeting の uv 環境（speechbrain）、titanet / eres2net / campp は測定用環境（sherpa-onnx）で実行")
    e.add_argument("--threads", type=int, default=1, help="CPU スレッド数（他の処理と同時なら 1）")
    e.add_argument("--pool-sec", type=float, default=150.0, help="話者×感情ごとの登録候補の量（秒）")
    e.add_argument("--windows", action="store_true", help="silero の区間の中を1.5秒の窓に分けた埋め込みも作る（区間の切り直しの見積もり用。時間が約1.5倍かかる）")
    e.set_defaults(func=cmd_enroll_embed)

    t = sub.add_parser("enroll-report", help="登録音声との照合を条件ごとに採点して Excel を作る（enroll-embed の後。採点は数分。結果は保存され、続きから再開できる）")
    t.add_argument("--data", default="data/testdata/pseudo_meeting")
    t.add_argument("--out", default="outputs/enroll_assign.xlsx")
    t.add_argument("--embedder", nargs="+", choices=["ecapa", "titanet", "eres2net", "campp"], help="省略すると、埋め込みがある全部")
    t.add_argument("--secs", nargs="+", type=int, default=[10, 30, 60, 180], help="登録の長さ（秒/人）")
    t.add_argument("--seeds", type=int, default=5, help="登録発話の選び方を変える回数")
    t.add_argument("--force", action="store_true", help="保存済みの結果があっても採点し直す")
    t.set_defaults(func=cmd_enroll_report)

    q = sub.add_parser("report", help="採点して Excel（まとめ・全指標・区間の診断・話者混同・発言別・実行情報）を作る")
    q.add_argument("--data", default="data/testdata/pseudo_meeting")
    q.add_argument("--variant", nargs="+", default=["A", "B"], choices=["A", "B"])
    q.add_argument("--out", default="outputs/pseudo_meeting_asr.xlsx")
    q.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as e:
        say(f"エラー: {e}")
        return 2
    except KeyboardInterrupt:
        say("中断しました")
        return 130


if __name__ == "__main__":
    sys.exit(main())
