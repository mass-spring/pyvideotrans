"""コマンドライン入口: emocheck run / prepare-jvnv"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

from asrcheck.ioformats import InputError
from asrcheck.textnorm import NormOptions

from .analysis import cer_tables, emotion_stats
from .dataset import DATASET_PRESETS, build_jvnv_labels, describe_counts, read_labels, stratified_subset
from .labels import JA
from .report import ReportData, write_report
from .runners import SV_DEFAULT_MODEL, WHISPER_DEFAULT_MODEL, ModelRun, SenseVoiceRecognizer, WhisperRecognizer, run_model

MODELS = ("sensevoice", "whisper")


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="emocheck",
        description="感情の聞き分け (SenseVoice の感情タグ) と日本語の文字起こし (CER) の現状値を、ラベル付き音声フォルダで測って Excel に出す",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("prepare-jvnv", help="JVNV コーパスのフォルダから ラベル CSV (file,emotion,text,speaker,style) を作る")
    j.add_argument("--jvnv-dir", required=True, help="JVNV のフォルダ (F1/ F2/ M1/ M2/ と transcription_no_nv.csv がある所)")
    j.add_argument("--out", default="data/testdata/jvnv_labels.csv", help="出力するラベル CSV (既定 data/testdata/jvnv_labels.csv)")
    j.set_defaults(func=cmd_prepare_jvnv)

    r = sub.add_parser("run", help="モデルを動かして、結果を Excel にまとめる")
    r.add_argument("--audio-dir", required=True, help="音声フォルダ (ラベル CSV の file 列は、ここからの相対パス)")
    r.add_argument("--labels", required=True, help="ラベル CSV (file,emotion,text。話者 speaker・区分 style の列は任意)")
    r.add_argument("--models", nargs="+", choices=MODELS, default=["sensevoice"], help="動かすモデル (既定 sensevoice。whisper も比べるなら: sensevoice whisper)")
    r.add_argument("--dataset", choices=sorted(DATASET_PRESETS), help="データの出典・ライセンスを Excel に載せる (jvnv)")
    r.add_argument("--data-note", help="--dataset を使わないとき、データの出典・ライセンスを自分で書く")
    r.add_argument("--sv-model", default=SV_DEFAULT_MODEL, help=f"SenseVoice のモデル名/フォルダ (既定 {SV_DEFAULT_MODEL})")
    r.add_argument("--sv-language", default="auto", choices=("auto", "ja", "zh", "en", "yue", "ko"),
                   help="SenseVoice に渡す言語 (既定 auto。pyVideoTrans の SenseVoiceSmall は言語を指定しない)")
    r.add_argument("--sv-subset", type=int, default=0, metavar="N", help="SenseVoice は『話者×感情』ごとに N 件だけ処理 (既定 0 = 全件)")
    r.add_argument("--whisper-model", default=WHISPER_DEFAULT_MODEL, help=f"faster-whisper のモデル名 (既定 {WHISPER_DEFAULT_MODEL}。small など小さいものも可)")
    r.add_argument("--whisper-subset", type=int, default=10, metavar="N", help="Whisper は『話者×感情』ごとに N 件だけ処理 (既定 10。0 = 全件)")
    r.add_argument("--seed", type=int, default=0, help="件を選ぶ乱数の種 (同じ値なら毎回同じ件。既定 0)")
    r.add_argument("--threads", type=int, default=2, help="CPU のスレッド数 (既定 2。ほかの重い処理と同時に動かすときは小さく)")
    r.add_argument("--cache-dir", help="途中結果の保存場所 (既定: 出力 Excel のフォルダ/emocheck_cache)。止まっても続きから再開できる")
    r.add_argument("--no-cache", action="store_true", help="途中結果を使わず、保存もしない")
    r.add_argument("--out", default="outputs/jvnv_emotion_asr.xlsx", help="出力する Excel (既定 outputs/jvnv_emotion_asr.xlsx)")
    r.set_defaults(func=cmd_run)
    return p


def cmd_prepare_jvnv(args: argparse.Namespace) -> int:
    stats = build_jvnv_labels(args.jvnv_dir, args.out)
    say(f"ラベル CSV を作りました: {Path(args.out).resolve()}")
    say(f"  音声 {stats['wavs']} 件 / 書き起こしが無い音声 {stats['no_text']} 件 / 使われなかった書き起こし {stats['unused_text']} 行")
    say("次は:")
    say(f"  python -m emocheck run --audio-dir {args.jvnv_dir} --labels {args.out} --dataset jvnv --models sensevoice whisper")
    return 0


def _print_headline(runs: list[ModelRun], d: ReportData) -> None:
    for st in [emotion_stats(d.clips, r) for r in runs if r.has_emotion]:
        say(f"[{st.model}] 感情の正解率 {st.accuracy:.1%} ({st.correct}/{st.n}) / 平均再現率 {st.macro_recall:.1%} / "
            f"平常と判定 {st.neutral_rate:.1%} / 不明 {st.unknown_rate:.1%} / タグなし {st.notag_rate:.1%}")
        for row in st.per_emotion:
            say(f"    {JA[row.label]}: {row.recall:.1%} ({row.correct}/{row.n})")
    for t in cer_tables(d.clips, runs, d.opts):
        say(f"[{t.model}] CER {t.overall.cer:.1%}  {t.scope}")


def cmd_run(args: argparse.Namespace) -> int:
    threads = max(1, args.threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # モデルを読み込む前に設定する (faster-whisper / torch が見る)
        os.environ[var] = str(threads)
    t_all = time.perf_counter()
    clips = read_labels(args.labels, args.audio_dir)
    say(f"音声 {len(clips)} 件を読みました（{describe_counts(clips)}）")
    out = Path(args.out)
    cache_dir = None if args.no_cache else Path(args.cache_dir) if args.cache_dir else out.parent / "emocheck_cache"
    preset = DATASET_PRESETS.get(args.dataset or "", {})
    runs: list[ModelRun] = []
    subsets: dict[str, str] = {}
    for m in args.models:
        per = args.sv_subset if m == "sensevoice" else args.whisper_subset
        use = stratified_subset(clips, per, args.seed)
        subsets[m] = (f"全件（{len(use)} 件）" if len(use) == len(clips)
                      else f"『話者×感情』ごとに {per} 件ずつ抽出（乱数の種 {args.seed}）→ {len(use)} 件 / 全 {len(clips)} 件中")
        say(f"\n== {m}: {len(use)} 件を処理します ==")
        t0 = time.perf_counter()
        if m == "sensevoice":
            rec = SenseVoiceRecognizer(args.sv_model, args.sv_language, "cpu", threads)
        else:
            rec = WhisperRecognizer(args.whisper_model, "ja", "cpu", threads)
        load_s = time.perf_counter() - t0
        say(f"  モデルの準備 {load_s:.1f} 秒")
        run = run_model(rec, use, (cache_dir / f"{rec.key}.jsonl") if cache_dir else None, load_s)
        runs.append(run)
        if run.n_errors:
            say(f"  注意: {run.n_errors} 件で失敗しました（Excel の『1件ごとの結果』に理由が出ます）")
    notes = [f"{r.label}: {r.n_errors} 件の音声で認識に失敗しました（失敗した件は集計から除いています）" for r in runs if r.n_errors]
    d = ReportData(
        clips=clips, runs=runs, opts=NormOptions(lang="ja"),
        dataset_name=preset.get("name", ""), dataset_info=list(preset.get("info", [])) + ([("データの出典（自己申告）", args.data_note)] if args.data_note else []),
        dataset_notes=list(preset.get("notes", [])),
        command="python -m emocheck " + " ".join(shlex.quote(a) for a in args.argv), labels_csv=str(args.labels), audio_dir=str(args.audio_dir),
        subsets=subsets, elapsed_sec=time.perf_counter() - t_all, notes=notes,
    )
    path = write_report(out, d)
    say("")
    _print_headline(runs, d)
    say(f"\n完了: {path.resolve()}  ({d.elapsed_sec:.0f} 秒)")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows のコンソールで文字化けしても落ちないように
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]
    debug = bool(os.environ.get("EMOCHECK_DEBUG"))
    try:
        return int(args.func(args))
    except (InputError, FileNotFoundError, ValueError) as e:
        if debug:
            raise
        say(f"エラー: {e}")
        return 2
    except PermissionError as e:
        say(f"エラー: ファイルに書き込めません ({e})。Excel で開いている場合は閉じてからやり直してください")
        return 2
    except KeyboardInterrupt:
        say("中断しました（途中までの結果は保存してあるので、同じコマンドをもう一度実行すると続きから再開します）")
        return 130


if __name__ == "__main__":
    sys.exit(main())
