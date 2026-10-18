"""コマンドライン入口: voiceprofile build / map / script"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from voicecheck.embedders import CHOICES

from . import __version__
from .align import STATUS_JA
from .pipeline import BuildError, BuildOptions, BuildResult, build_profile
from .profile_io import (
    check_can_write,
    parse_set_args,
    profile_dir_for,
    write_profile,
    write_speaker_map,
)
from .report import write_report
from .script import Script, ScriptError, load_script, total_est_sec
from .text_utils import PAUSE_BETWEEN_SENTENCES_SEC, norm_lang


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


def _split_langs(spec: str) -> list[str]:
    langs = [norm_lang(x) for x in spec.split(",") if x.strip()]
    if not langs:
        raise ValueError("--lang に言語を指定してください（例: ja または ja,en）")
    return list(dict.fromkeys(langs))


def report_filename(langs: list[str], appending: bool) -> str:
    """新規作成は enroll_report.xlsx。--append で足したときは言語名を付けて、前の報告書を消さない。"""
    return "enroll_report_" + "-".join(langs) + ".xlsx" if appending else "enroll_report.xlsx"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> int:
    script: Script | None = load_script(args.script) if args.script else None
    langs = _split_langs(args.lang)
    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(f"録音ファイルが見つかりません: {p}")
    if args.skip_first is not None and args.skip_first < 0:
        raise ValueError("--skip-first は 0 以上の整数です")
    if script is None and len(langs) > 1:
        say("注意: 台本 (--script) がないので、言語は最初の1つ (" + langs[0] + ") として扱います")
        langs = langs[:1]
    opts = BuildOptions(
        name=args.name, inputs=inputs, out_dir=Path(args.out), script=script, script_path=args.script or "", langs=langs,
        skip_first=args.skip_first, consent=not args.no_consent, asr=args.asr, asr_model=args.asr_model,
        n_refs=args.n_refs, n_refs_other=args.n_refs_other, min_sec=args.min_sec, max_sec=args.max_sec,
        target_sec=args.target_sec, min_pause=args.min_pause, embedder=args.embedder, f0=args.f0, vad=args.vad,
        append=args.append, force=args.force, stats=not args.no_stats, excel=not args.no_excel,
    )
    pdir = profile_dir_for(opts.out_dir, opts.name)
    check_can_write(pdir, opts.append, opts.force)  # 重い計算の前に確認する
    if script is None:
        say("注意: 台本 (--script) がないので、書き起こしは空欄になります（--asr whisper で下書きを作れます）。聞いて書き込んでください")
    res = build_profile(opts)
    written = write_profile(res)
    report_path = None
    if opts.excel:
        command = "voiceprofile " + " ".join(_quote(a) for a in args.argv)
        report_path = write_report(res, pdir / report_filename(langs, written["appended"]), command, written)
    _print_summary(res, written, report_path)
    return 0


def _quote(a: str) -> str:
    return f'"{a}"' if (" " in a or not a.isascii()) else a


def _print_summary(res: BuildResult, written: dict, report_path: Path | None) -> None:
    say("")
    say(f"完了: {written['dir']}")
    say(f"  発話区間 {len(res.atoms)} 個 / 候補クリップ {len(res.candidates)} 個 / 手本 {len(res.refs)} 本 / 評価用 {len(res.evals)} 本")
    say("  手本（点数の高い順）:")
    for c in res.refs:
        cons = " ".join(f"{n}={v:.2f}" for n, v in {**c.consistency, **c.consistency_true}.items())
        ids = (c.slot_ids[0] + ("〜" + c.slot_ids[-1] if len(c.slot_ids) > 1 else "")) if c.slot_ids else "-"
        say(f"    {c.ref_name}  {c.dur:5.1f}秒  SNR {c.snr_db:5.1f} dB  割れ {c.clip_ratio:.2%}  "
            f"{('一貫性 ' + cons + '  ') if cons else ''}点 {c.score:5.1f}  文 {ids}  [{STATUS_JA.get(c.status, c.status)}]")
    for lang, st in res.stats.items():
        f0, iqr, rate = st.get("f0_median_hz"), st.get("f0_iqr_semitone"), st.get("rate_mora_per_sec")
        say(f"  声の特徴 ({lang}): 声の高さ {f0:.0f} Hz / 幅 {iqr:.1f} 半音" + (f" / 話速 {rate:.1f} モーラ/秒" if rate else "")
            if f0 and iqr else f"  声の特徴 ({lang}): 測れませんでした")
    if report_path:
        say(f"  報告書: {report_path}")
    if res.warnings:
        say("")
        say(f"確認してください（{len(res.warnings)} 件。全部「確認事項」シートにも書きました）:")
        for w in res.warnings[:12]:
            say("  - " + w)
        if len(res.warnings) > 12:
            say(f"  ... ほか {len(res.warnings) - 12} 件")
    say("")
    say("次にやること: refs フォルダの手本を聞き、profile.json の text（書き起こし）が合っているか確認してください。")
    say(f"             そのあと:  voiceprofile map --profiles {res.options.out_dir} --set spk0={res.options.name}")


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------
def cmd_map(args: argparse.Namespace) -> int:
    from .profile_io import SPEAKER_MAP_JSON, read_json

    pdir = Path(args.profiles)
    if args.show and not args.set:
        data = read_json(pdir / SPEAKER_MAP_JSON)
        if not isinstance(data, dict) or not data:
            say(f"{pdir / SPEAKER_MAP_JSON} はまだありません")
        else:
            for k, v in data.items():
                say(f"  {k} -> {v}")
        return 0
    if not args.set:
        raise ValueError("--set spk0=Aさん のように、話者ラベルと名前を指定してください")
    mapping = parse_set_args(args.set)
    path, content, problems = write_speaker_map(pdir, mapping, replace=args.replace, allow_missing=args.allow_missing)
    say(f"書きました: {path}")
    for k, v in content.items():
        say(f"  {k} -> {v}")
    for p in problems:
        say(f"警告: {p}")
    say("pyVideoTrans の設定 voice_profile_dir に、このフォルダの絶対パスを指定してください（README を参照）")
    return 0


# ---------------------------------------------------------------------------
# script (読む台本を端末に出す)
# ---------------------------------------------------------------------------
def cmd_script(args: argparse.Namespace) -> int:
    script = load_script(args.script)
    langs = _split_langs(args.lang)
    print(f"■ 同意文（最初に読みます。{args.script}）")
    print("  " + script.render_consent(args.speaker or "", args.org or "", args.retention or ""))
    if not (args.speaker and args.org and args.retention):
        print("  （○○の所は --speaker 名前 --org 組織名 --retention 保存期間 で埋まります）")
    for lang in langs:
        sents = script.sentences(lang)
        sec = total_est_sec(sents, PAUSE_BETWEEN_SENTENCES_SEC)
        print(f"\n■ {script.labels.get(lang, lang)}  {len(sents)} 文 / 約 {sec:.0f} 秒")
        for s in sents:
            print(f"  {s.id}  {s.text}")
            if s.yomi:
                print(f"        （{s.yomi}）")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voiceprofile",
        description="話者の声の登録ツール: 録音から、声の再現に使うきれいな手本（10〜15秒×3〜4本）と評価用の取り置きを作ります。",
    )
    p.add_argument("--version", action="version", version=f"voiceprofile {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="{build,map,script}")

    b = sub.add_parser("build", help="録音から話者プロファイル（手本・評価用・報告書）を作る")
    b.add_argument("--name", required=True, help='話者の名前（フォルダ名になります。例: "Aさん"）')
    b.add_argument("--in", dest="inputs", action="append", required=True, metavar="WAV",
                   help="録音ファイル。複数回指定できる（--in a.wav --in b.wav）。wav 推奨（48 kHz・モノラル）")
    b.add_argument("--script", help="朗読台本の JSON（enroll/script_ja.json）。付けると書き起こしを自動で付ける")
    b.add_argument("--out", default="profiles", help="プロフィールを置くフォルダ（既定: profiles）。<out>/<名前>/ ができる")
    b.add_argument("--lang", default="ja", help="録音に入っている台本の言語。読んだ順にカンマ区切り（既定 ja。例: ja,en）")
    b.add_argument("--skip-first", type=int, default=None, metavar="N",
                   help="録音の最初の N 区間を同意文として捨てる。省略すると、台本があれば同意文も台本の一部として自動で対応づける")
    b.add_argument("--no-consent", action="store_true", help="録音に同意文が入っていない（外国語だけの追加録音など）")
    b.add_argument("--asr", default="none", choices=["none", "whisper"],
                   help="各区間を faster-whisper で書き起こして台本と照合する（uv sync --group asr が必要）。台本なしなら下書きの書き起こしになる")
    b.add_argument("--asr-model", default="small", help="faster-whisper のモデル名（既定 small。tiny/base/small/medium/large-v3）")
    b.add_argument("--n-refs", type=int, default=4, help="手本の本数の上限（既定 4。取り置きが足りなければ最低 3 本まで減らす）")
    b.add_argument("--n-refs-other", type=int, default=2, help="2 つ目以降の言語の手本の本数の上限（既定 2）")
    b.add_argument("--min-sec", type=float, default=8.0, help="手本の最短の長さ（秒、既定 8）")
    b.add_argument("--max-sec", type=float, default=20.0, help="手本の最長の長さ（秒、既定 20）")
    b.add_argument("--target-sec", type=float, default=12.0, help="目標の長さ（秒、既定 12）")
    b.add_argument("--min-pause", type=float, default=0.35, help="これ未満の無音では発話区間を分けない（秒、既定 0.35）")
    b.add_argument("--embedder", default="ecapa", choices=CHOICES,
                   help="声の一貫性を測る話者埋め込み: ecapa（既定）/ resemblyzer / all / none（測らない）")
    b.add_argument("--f0", default="harvest", choices=["harvest", "pyin"], help="声の高さの測り方（既定 harvest）")
    b.add_argument("--vad", default="auto", choices=["auto", "energy", "webrtc"],
                   help="発話区間の判定: auto（音量＋webrtcvad で物音を除く）/ energy（音量だけ）/ webrtc（必須）")
    b.add_argument("--append", action="store_true", help="既存のプロフィールに足す（今回の言語の手本だけ置き換え、他の言語は残す）")
    b.add_argument("--force", action="store_true", help="既存のプロフィールを消して作り直す（refs と eval を消します）")
    b.add_argument("--no-stats", action="store_true", help="声の特徴（高さ・話速など）を測らない（速くなる）")
    b.add_argument("--no-excel", action="store_true", help="Excel の報告書を作らない")
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("map", help="話者ラベル（spk0 など）とプロフィール名の対応表 speaker_map.json を書く")
    m.add_argument("--profiles", required=True, help="プロフィールを置いたフォルダ（build の --out と同じ）")
    m.add_argument("--set", action="append", metavar="LABEL=NAME", help="対応を追加（例: --set spk0=Aさん）。複数回指定できる")
    m.add_argument("--replace", action="store_true", help="既存の対応表を捨てて、今回の --set だけにする（省略時は追加・上書き）")
    m.add_argument("--allow-missing", action="store_true", help="まだ作っていないプロフィール名でも書く")
    m.add_argument("--show", action="store_true", help="今の対応表を表示する（--set なしのとき）")
    m.set_defaults(func=cmd_map)

    s = sub.add_parser("script", help="読む台本（同意文と文）を画面に出す")
    s.add_argument("--script", default="enroll/script_ja.json", help="台本の JSON（既定 enroll/script_ja.json）")
    s.add_argument("--lang", default="ja", help="表示する言語（カンマ区切り。例: ja,en,fr,zh）")
    s.add_argument("--speaker", help="同意文に入れる名前")
    s.add_argument("--org", help="同意文に入れる組織名")
    s.add_argument("--retention", help="同意文に入れる保存期間（例: 2027年3月末）")
    s.set_defaults(func=cmd_script)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows のコンソールで文字化けしても落ちないように
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    debug = bool(os.environ.get("VOICEPROFILE_DEBUG"))  # 1 にすると想定外のエラーでも詳細を出す
    try:
        return int(args.func(args))
    except (BuildError, ScriptError, FileNotFoundError, ValueError) as e:
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
