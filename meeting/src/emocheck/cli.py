"""コマンドライン入口: emocheck run / compare / prepare-jvnv / prepare-fleurs"""
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
from .compare import compare_models
from .compare_report import write_compare_report
from .dataset import (DATASET_PRESETS, FLEURS_SPLITS, build_fleurs, build_jvnv_labels, describe_counts, read_labels, read_sets,
                      stratified_subset)
from .labels import JA
from .report import ReportData, write_report
from .runners import (E2V_BASE_MODEL, E2V_LARGE_MODEL, SV_DEFAULT_MODEL, WHISPER_DEFAULT_MODEL, Emotion2VecRecognizer, ModelRun,
                      SenseVoiceRecognizer, WhisperRecognizer, run_model, trim_memory)

MODELS = ("sensevoice", "sensevoice-nounk", "whisper", "emotion2vec-large", "emotion2vec-base")
COMPARE_MODELS = ("sensevoice", "sensevoice-nounk", "emotion2vec-large", "emotion2vec-base")


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

    f = sub.add_parser("prepare-fleurs", help="FLEURS 日本語（平常の読み上げ）から N 件を取り出し、ラベル CSV を作る（音声の tar.gz は取り出した後に消す）")
    f.add_argument("--split", default="dev", choices=FLEURS_SPLITS, help="どの分割か（既定 dev = 266 件・約 180 MB。test は 650 件・約 450 MB）")
    f.add_argument("--max-clips", type=int, default=300, help="取り出す最大件数（既定 300。0 = 全件）")
    f.add_argument("--seed", type=int, default=0, help="件を選ぶ乱数の種（既定 0）")
    f.add_argument("--out-dir", default="data/testdata/fleurs_ja", help="音声の出力先（既定 data/testdata/fleurs_ja）")
    f.add_argument("--labels-out", default="data/testdata/fleurs_labels.csv", help="ラベル CSV の出力先（既定 data/testdata/fleurs_labels.csv）")
    f.add_argument("--tsv", help="手元の FLEURS の tsv を使う（指定しなければダウンロード）")
    f.add_argument("--tarball", help="手元の FLEURS の音声 tar.gz を使う（指定しなければダウンロードし、取り出した後に消す。指定したファイルは消さない）")
    f.set_defaults(func=cmd_prepare_fleurs)

    c = sub.add_parser("compare", help="複数のデータ（JVNV + FLEURS など）を複数のモデルで測り、4分類の KPI を含む比較表を 1 つの Excel にまとめる")
    c.add_argument("--set", nargs=3, action="append", metavar=("名前", "音声フォルダ", "ラベルCSV"), required=True,
                   help="測るデータ。名前は jvnv / fleurs なら出典とライセンスが Excel に載る。2 つ以上、繰り返して指定できる")
    c.add_argument("--models", nargs="+", choices=COMPARE_MODELS, default=list(COMPARE_MODELS),
                   help="動かすモデル（既定: すべて。emotion2vec-large は重み約 2 GB、-base は約 1.1 GB）")
    c.add_argument("--sv-model", default=SV_DEFAULT_MODEL, help=f"SenseVoice のモデル名/フォルダ（既定 {SV_DEFAULT_MODEL}）")
    c.add_argument("--sv-language", default="auto", choices=("auto", "ja", "zh", "en", "yue", "ko"), help="SenseVoice に渡す言語（既定 auto）")
    c.add_argument("--e2v-large-model", default=E2V_LARGE_MODEL, help=f"emotion2vec+ large のモデル名（既定 {E2V_LARGE_MODEL}。Hugging Face なら emotion2vec/emotion2vec_plus_large）")
    c.add_argument("--e2v-base-model", default=E2V_BASE_MODEL, help=f"emotion2vec+ base のモデル名（既定 {E2V_BASE_MODEL}）")
    c.add_argument("--e2v-hub", default="auto", choices=("auto", "ms", "hf"), help="emotion2vec+ の取得元（既定 auto = iic/ は ModelScope、emotion2vec/ は Hugging Face）")
    c.add_argument("--subset", type=int, default=0, metavar="N", help="試運転用: 各データから『話者×感情』ごとに N 件だけ処理（既定 0 = 全件）")
    c.add_argument("--seed", type=int, default=0, help="--subset で件を選ぶ乱数の種（既定 0）")
    c.add_argument("--threads", type=int, default=2, help="CPU のスレッド数（既定 2）")
    c.add_argument("--cache-dir", help="途中結果の保存場所（既定: 出力 Excel のフォルダ/emocheck_cache）。止まっても続きから再開できる")
    c.add_argument("--no-cache", action="store_true", help="途中結果を使わず、保存もしない")
    c.add_argument("--cache-only", action="store_true", help="モデルを動かして途中結果を保存するだけで、Excel は作らない（あとで同じコマンドから --cache-only を外すと、Excel だけ作る）")
    c.add_argument("--out", default="outputs/emotion_compare.xlsx", help="出力する Excel（既定 outputs/emotion_compare.xlsx）")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("run", help="モデルを動かして、結果を Excel にまとめる")
    r.add_argument("--audio-dir", required=True, help="音声フォルダ (ラベル CSV の file 列は、ここからの相対パス)")
    r.add_argument("--labels", required=True, help="ラベル CSV (file,emotion,text。話者 speaker・区分 style の列は任意)")
    r.add_argument("--models", nargs="+", choices=MODELS, default=["sensevoice"],
                   help="動かすモデル (既定 sensevoice)。sensevoice-nounk = SenseVoice に EMO_UNKNOWN を禁止して感情を選ばせた場合。例: sensevoice sensevoice-nounk whisper")
    r.add_argument("--e2v-large-model", default=E2V_LARGE_MODEL, help=f"emotion2vec-large のモデル名 (既定 {E2V_LARGE_MODEL})")
    r.add_argument("--e2v-base-model", default=E2V_BASE_MODEL, help=f"emotion2vec-base のモデル名 (既定 {E2V_BASE_MODEL})")
    r.add_argument("--e2v-hub", default="auto", choices=("auto", "ms", "hf"), help="emotion2vec+ の取得元 (既定 auto)")
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
    say(f"  python -m emocheck run --audio-dir {args.jvnv_dir} --labels {args.out} --dataset jvnv --models sensevoice sensevoice-nounk whisper")
    return 0


def cmd_prepare_fleurs(args: argparse.Namespace) -> int:
    say(f"FLEURS 日本語の {args.split} から最大 {args.max_clips or '全'} 件を取り出します（ダウンロードする音声は取り出した後に消します）")
    stats = build_fleurs(args.split, args.out_dir, args.labels_out, args.max_clips, args.seed, args.tsv, args.tarball)
    say(f"音声を作りました: {Path(args.out_dir).resolve()}  （{stats['extracted']} 件 / {args.split} 全 {stats['rows']} 件中。男性 {stats['male']}・女性 {stats['female']}）")
    say(f"ラベル CSV を作りました: {Path(args.labels_out).resolve()}  （感情はすべて neutral = 平常）")
    say("次は:")
    say(f"  python -m emocheck compare --set jvnv data/testdata/jvnv data/testdata/jvnv_labels.csv --set fleurs {args.out_dir} {args.labels_out}")
    return 0


def make_recognizer(name: str, args: argparse.Namespace, threads: int):
    """モデル名 → 認識器（読み込みは、使うときに初めて行う）。"""
    if name in ("sensevoice", "sensevoice-nounk"):
        return SenseVoiceRecognizer(args.sv_model, args.sv_language, "cpu", threads, ban_unk=(name == "sensevoice-nounk"))
    if name == "emotion2vec-large":
        return Emotion2VecRecognizer(args.e2v_large_model, "cpu", threads, args.e2v_hub)
    if name == "emotion2vec-base":
        return Emotion2VecRecognizer(args.e2v_base_model, "cpu", threads, args.e2v_hub)
    return WhisperRecognizer(args.whisper_model, "ja", "cpu", threads)


def _set_threads(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # モデルを読み込む前に設定する (faster-whisper / torch が見る)
        os.environ[var] = str(threads)


def cmd_compare(args: argparse.Namespace) -> int:
    threads = max(1, args.threads)
    _set_threads(threads)
    t_all = time.perf_counter()
    sets = [(n, a, lab) for n, a, lab in args.set]
    clips = read_sets(sets)
    if args.subset > 0:
        clips = [c for n, _, _ in sets for c in stratified_subset([x for x in clips if x.dataset == n], args.subset, args.seed)]
    for n, _, _ in sets:
        part = [c for c in clips if c.dataset == n]
        say(f"{n}: 音声 {len(part)} 件（{describe_counts(part)}）")
    out = Path(args.out)
    cache_dir = None if args.no_cache else Path(args.cache_dir) if args.cache_dir else out.parent / "emocheck_cache"
    runs: list[ModelRun] = []
    subsets: dict[str, str] = {}
    for m in args.models:
        rec = make_recognizer(m, args, threads)
        say(f"\n== {m}: {len(clips)} 件を処理します ==")
        run = run_model(rec, clips, (cache_dir / f"{rec.key}.jsonl") if cache_dir else None)
        say(f"  モデルの準備 {run.load_sec:.1f} 秒（前回の結果だけで足りるときは 0 秒）")
        del rec
        trim_memory()      # 次のモデルを読み込む前に、使い終わったモデルのメモリを OS に返す
        subsets[run.key] = f"全件（{len(clips)} 件）" if args.subset <= 0 else f"試運転: 『話者×感情』ごとに {args.subset} 件ずつ（乱数の種 {args.seed}）→ {len(clips)} 件"
        runs.append(run)
        if run.n_errors:
            say(f"  注意: {run.n_errors} 件で失敗しました（Excel の『1件ごとの結果』に理由が出ます）")
    if args.cache_only:
        say(f"\n途中結果を保存しました（{cache_dir}）。Excel は作っていません。--cache-only を外して同じコマンドを実行すると、Excel を作ります")
        return 0
    info: list[tuple[str, str]] = []
    notes: list[str] = []
    names: list[str] = []
    for n, _, _ in sets:
        preset = DATASET_PRESETS.get(n, {})
        names.append(preset.get("name", n))
        info += [(f"【{preset.get('name', n)}】{k}", v) for k, v in preset.get("info", [])]
        notes += list(preset.get("notes", []))
    d = ReportData(
        clips=clips, runs=runs, opts=NormOptions(lang="ja"), dataset_name=" + ".join(names), dataset_info=info, dataset_notes=notes,
        command="python -m emocheck " + " ".join(shlex.quote(a) for a in args.argv), labels_csv=" / ".join(str(x[2]) for x in sets),
        audio_dir=" / ".join(str(x[1]) for x in sets), subsets=subsets, elapsed_sec=time.perf_counter() - t_all,
        notes=[f"{r.label}: {r.n_errors} 件の音声で認識に失敗しました（失敗した件は集計から除いています）" for r in runs if r.n_errors],
    )
    path = write_compare_report(out, d)
    say("")
    _print_compare_headline(compare_models(clips, runs, d.opts))
    say(f"\n完了: {path.resolve()}  ({d.elapsed_sec:.0f} 秒)")
    return 0


def _print_compare_headline(cmp) -> None:
    from .compare import VIEW_JA

    for mc in cmp:
        for v, e in mc.four.items():
            say(f"[{mc.run.label}] 4分類の正解率（{VIEW_JA[v]}） {e.accuracy:.1%} ({e.correct}/{e.n}) / 平均再現率 {e.balanced:.1%}")
        if "strict" in mc.six:
            e = mc.six["strict"]
            say(f"[{mc.run.label}] 6感情（JVNV）の正解率 {e.accuracy:.1%} ({e.correct}/{e.n})")
        if mc.neutral:
            say(f"[{mc.run.label}] 平常の音声: 平常と判定 {mc.neutral.neutral_rate:.1%} / 感情が付かなかった {mc.neutral.no_emotion_rate:.1%} ({mc.neutral.n} 件)")


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
    _set_threads(threads)
    t_all = time.perf_counter()
    clips = read_labels(args.labels, args.audio_dir)
    say(f"音声 {len(clips)} 件を読みました（{describe_counts(clips)}）")
    out = Path(args.out)
    cache_dir = None if args.no_cache else Path(args.cache_dir) if args.cache_dir else out.parent / "emocheck_cache"
    preset = DATASET_PRESETS.get(args.dataset or "", {})
    runs: list[ModelRun] = []
    subsets: dict[str, str] = {}
    for m in args.models:
        per = args.whisper_subset if m == "whisper" else args.sv_subset
        use = stratified_subset(clips, per, args.seed)
        rec = make_recognizer(m, args, threads)
        subsets[rec.key] = (f"全件（{len(use)} 件）" if len(use) == len(clips)
                            else f"『話者×感情』ごとに {per} 件ずつ抽出（乱数の種 {args.seed}）→ {len(use)} 件 / 全 {len(clips)} 件中")
        say(f"\n== {m}: {len(use)} 件を処理します ==")
        run = run_model(rec, use, (cache_dir / f"{rec.key}.jsonl") if cache_dir else None)
        say(f"  モデルの準備 {run.load_sec:.1f} 秒（前回の結果だけで足りるときは 0 秒）")
        runs.append(run)
        if run.n_errors:
            say(f"  注意: {run.n_errors} 件で失敗しました（Excel の『1件ごとの結果』に理由が出ます）")
    notes = [f"{r.label}: {r.n_errors} 件の音声で認識に失敗しました（失敗した件は集計から除いています）" for r in runs if r.n_errors]
    if "whisper" in args.models and args.whisper_model not in (WHISPER_DEFAULT_MODEL, "turbo"):
        notes.append(f"Whisper は、比較の本命の {WHISPER_DEFAULT_MODEL} ではなく {args.whisper_model} で測っています。{args.whisper_model} のほうが小さく、"
                     f"{WHISPER_DEFAULT_MODEL} より誤りが多くなる可能性が高いので、SenseVoice との優劣の判断には使わないでください（参考値）")
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
