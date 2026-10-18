"""コマンドライン入口: asrcheck eval / run"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .ioformats import InputError, read_utts_table
from .kpi import parse_kpi_overrides
from .model import Utt, rec_key
from .textnorm import FILLER_MODES, LANGS, LONG_VOWEL_MODES, NUMBER_MODES, NormOptions


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# --hyp NAME[@REC]=PATH の読み取り
# ---------------------------------------------------------------------------
def _looks_like_path(s: str) -> bool:
    return "/" in s or "\\" in s or (len(s) == 2 and s[1] == ":")


def parse_spec(spec: str) -> tuple[str, str, str]:
    """'名前@録音=パス' → (名前, 録音, パス)。名前を省略すると、パスのファイル名 (フォルダ名) から付ける。"""
    left, sep, right = spec.partition("=")
    if not sep or _looks_like_path(left):
        p = PurePosixPath(spec.replace("\\", "/").rstrip("/"))   # Windows のパス (\ 区切り) もそのまま扱う
        return (p.stem if p.suffix else p.name) or spec, "", spec
    name, _, rec = left.partition("@")
    if not name.strip() or not right.strip():
        raise InputError(f"指定の書き方が違います: '{spec}'  例: whisper=out/whisper.csv   または   whisper@mock1=out/mock1.csv")
    return name.strip(), rec.strip(), right.strip()


def group_specs(specs: list[str]) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for s in specs:
        name, rec, path = parse_spec(s)
        out.setdefault(name, []).append((rec, path))
    return out


def parse_times(specs: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in specs or []:
        name, sep, val = s.partition("=")
        try:
            if not sep:
                raise ValueError
            out[name.strip()] = float(val)
        except ValueError:
            raise InputError(f"--time の書き方が違います: '{s}'  例: --time whisper=1800  (名前=処理にかかった秒数)") from None
    return out


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
# 重い部品 (numpy/scipy/openpyxl) は eval のときだけ import する。run だけなら openpyxl は要らない。
def load_reference(ref_path: str, audio: list[str]) -> tuple[dict[str, list[Utt]], dict[str, str], list[str], str, dict[str, float]]:
    utts, notes, enc = read_utts_table(ref_path, role="reference")
    has_rec = [bool(u.rec) for u in utts]
    if any(has_rec) and not all(has_rec):
        raise InputError("正解ファイルの file (録音) 列が一部の行だけ空です。全部の行に書くか、列ごと消してください")
    default = Path(audio[0]).stem if len(audio) == 1 else Path(ref_path).stem
    by: dict[str, list[Utt]] = {}
    names: dict[str, str] = {}
    for u in utts:
        disp = Path(u.rec).stem if u.rec else default
        key = rec_key(disp)
        by.setdefault(key, []).append(u)
        names.setdefault(key, disp)
    durations: dict[str, float] = {}
    for a in audio:
        from .evaluate import audio_duration

        d = audio_duration(a)
        key = rec_key(a)
        if len(audio) == 1 and len(by) == 1:
            key = next(iter(by))
        if key not in by:
            notes.append(f"音声 {Path(a).name} に対応する録音が正解にありません (file 列と音声のファイル名を合わせてください)")
            continue
        if d is None:
            notes.append(f"音声 {Path(a).name} の長さを調べられませんでした (処理時間の換算には --time と音声の長さが必要です)")
        else:
            durations[key] = d
    return by, names, notes, enc, durations


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import EngineResult, EvalOptions, evaluate_engine
    from .report import base_info, summary_rows, write_report

    norm = NormOptions(lang=args.lang, numbers=args.numbers, long_vowel=args.long_vowel, fillers=args.fillers,
                       kana_fold=args.kana_fold, strip_accents=args.strip_accents)
    opts = EvalOptions(norm=norm, collar=args.collar)
    kpis = parse_kpi_overrides(args.kpi)
    times = parse_times(args.time)

    ref_by_rec, rec_names, ref_notes, enc, durations = load_reference(args.ref, args.audio or [])
    for n in ref_notes[:10]:
        say(f"注意: {n}")
    if len(ref_notes) > 10:
        say(f"注意: ほか {len(ref_notes) - 10} 件 (「実行情報」シートに全部書きます)")
    n_utts = sum(len(v) for v in ref_by_rec.values())
    spk = {u.speaker for v in ref_by_rec.values() for u in v if u.speaker}
    say(f"正解: 発言 {n_utts} 件 / 話者 {len(spk)} 人 / 録音 {len(ref_by_rec)} 本  (文字コード: {enc})")

    hyps = group_specs(args.hyp)
    diars = {n: p for n, lst in group_specs(args.diar or []).items() for _, p in lst[:1]}
    unknown = [n for n in diars if n not in hyps]
    if unknown:
        raise InputError(f"--diar の名前が --hyp にありません: {', '.join(unknown)}")
    for n in times:
        if n not in hyps:
            raise InputError(f"--time の名前が --hyp にありません: {n}")

    engines: list[EngineResult] = []
    for name, specs in hyps.items():
        say(f"評価しています: {name} ...")
        eng = evaluate_engine(name, ref_by_rec, rec_names, specs, diars.get(name), opts, lang=args.lang,
                              audio_s=durations, elapsed_s=times.get(name))
        engines.append(eng)

    ref_info = {"ref_name": Path(args.ref).name, "ref_utts": n_utts, "ref_speakers": len(spk), "ref_recs": len(ref_by_rec)}
    info = base_info(
        "asrcheck " + " ".join(shlex.quote(a) for a in args.argv),
        [
            ("正解ファイル", str(Path(args.ref).resolve())),
            ("正解の読み込み", f"{enc} / 発言 {n_utts} / 話者 {len(spk)} / 録音 {', '.join(rec_names.values())}"),
            ("正解の注意", "\n".join(ref_notes) or "なし"),
            ("音声の長さ(秒)", ", ".join(f"{rec_names[k]}={v:.1f}" for k, v in durations.items()) or "未指定"),
            ("言語", args.lang),
            ("文字の整え方", norm.describe()),
            ("DER の collar (KPI判定)", f"{args.collar:g} 秒 (サマリーには 0 と 0.25 の両方を載せます)"),
            ("KPI", "; ".join(f"{k.key}={k.rule[0]}/{k.rule[1]}" for k in kpis.values())),
        ] + [
            (f"認識結果: {e.name}", f"{e.source}\n形式: {e.kind}" + (f"\n処理時間: {e.elapsed_s:.1f} 秒" if e.elapsed_s else "")) for e in engines
        ],
    )
    out = write_report(args.out, engines, opts, kpis, info, ref_info)
    if args.csv:
        _write_csv(Path(args.out).with_suffix(".summary.csv"), engines, kpis, opts)

    say("")
    say(f"完了: {out.resolve()}")
    for r in summary_rows(engines, kpis, opts):
        parts = []
        for key, label in (("cer", "CER"), ("wer", "WER"), ("cpcer", "cpCER"), ("der", "DER"), ("spk_acc", "話者正解率")):
            if r.get(key) is not None:
                parts.append(f"{label} {r[key]:.1%}")
        say(f"  {r['engine']}: {' / '.join(parts) or '(計算できた指標なし)'}  → {r['verdict']}")
    return 0


def _write_csv(path: Path, engines: list, kpis: dict, opts: Any) -> None:
    from .evaluate import DER_COLLARS
    from .report import summary_cols, summary_rows

    unit = "文字" if opts.norm.lang == "ja" else "単語"

    collars = sorted({*DER_COLLARS, opts.collar})
    cols = summary_cols(collars, opts.collar, unit)
    rows = summary_rows(engines, kpis, opts)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM 付き UTF-8 (Excel で文字化けしない)
        w = csv.writer(f)
        w.writerow([c.header for c in cols])
        for r in rows:
            w.writerow(["" if r.get(c.key) is None else r.get(c.key) for c in cols])
    say(f"CSV: {path.resolve()}")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asrcheck",
        description="話者付き文字起こしの精度を測る: CER / WER / cpCER / DER / 話者正解率 を Excel にまとめます。",
    )
    p.add_argument("--version", action="version", version=f"asrcheck {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{eval,run}")

    e = sub.add_parser("eval", help="正解と認識結果を比べて Excel を作る")
    e.add_argument("--ref", required=True, help="正解の書き起こし (CSV か Excel。列: start,end,speaker,text。複数の録音なら file 列も)")
    e.add_argument("--hyp", action="append", required=True, metavar="[名前=]パス",
                   help="認識結果 (繰り返し指定できる)。pyVideoTrans のフォルダ/.srt、CSV、Excel、JSON、RTTM。"
                        " 録音が複数のときは 名前@録音名=パス")
    e.add_argument("--diar", action="append", metavar="名前=パス",
                   help="話者の区間だけの認識結果 (RTTM/CSV)。指定するとその名前の DER・話者正解率はこちらで計算 (文字起こしは --hyp のものを使う)")
    e.add_argument("--audio", action="append", help="録音ファイル (長さを調べて、処理時間を『1時間あたり』に換算する。繰り返し可)")
    e.add_argument("--lang", default="ja", choices=LANGS, help="言語。ja は CER、en/fr は WER (既定 ja)")
    e.add_argument("--numbers", default="arabic", choices=NUMBER_MODES,
                   help="数字の扱い: arabic=算用数字にそろえる(既定) / kanji=漢数字にそろえる / ignore=数字を無視 / keep=そのまま")
    e.add_argument("--long-vowel", default="keep", choices=LONG_VOWEL_MODES, help="長音「ー」: keep=残す(既定) / drop=消す")
    e.add_argument("--fillers", default="keep", choices=FILLER_MODES, help="フィラー(えーと等): keep=残す(既定) / drop=消す")
    e.add_argument("--kana-fold", action="store_true", help="カタカナをひらがなにそろえる")
    e.add_argument("--strip-accents", action="store_true", help="(en/fr) アクセント記号を消す")
    e.add_argument("--collar", type=float, default=0.0, help="DER の許容幅(秒、正解の区間の前後それぞれ)。KPI判定に使う値。既定 0 (0.25 もサマリーに併記)")
    e.add_argument("--kpi", help="KPI の上書き。例: cer=0.15/0.08,spk_acc=0.9/0.95,der=0.2/0.15,time=60/30 (最低ライン/目標)")
    e.add_argument("--time", action="append", metavar="名前=秒", help="その認識結果の処理にかかった秒数 (--audio と合わせて『1時間あたり・分』を出す)")
    e.add_argument("--out", default="asr_report.xlsx", help="出力する Excel (既定 asr_report.xlsx)")
    e.add_argument("--csv", action="store_true", help="サマリーを CSV (UTF-8 BOM付き) でも出す")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("run", help="(任意) 音声から認識結果の CSV を作る: ASR エンジン × 話者分離")
    from .runners import add_run_arguments

    add_run_arguments(r)

    def _run(args: argparse.Namespace) -> int:
        from .runners import cmd_run

        return cmd_run(args)

    r.set_defaults(func=_run)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows のコンソールで文字化けしても落ちないように
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]   # 「実行情報」シートに残すコマンド
    debug = bool(os.environ.get("ASRCHECK_DEBUG"))
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
        say("中断しました")
        return 130


if __name__ == "__main__":
    sys.exit(main())
