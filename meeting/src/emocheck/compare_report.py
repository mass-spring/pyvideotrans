"""モデル比較の Excel（outputs/emotion_compare.xlsx）。シート名・見出しは日本語。

シート: まとめ / 感情の混同表 / 感情ごとの正解率 / 文字誤り率 / 1件ごとの結果 / 説明 / 実行情報
配色・部品は emocheck.report と asrcheck.report と同じものを使う。
"""
from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from asrcheck.report import BAD_FILL, BAD_FONT, BORDER, GOOD_FILL, GOOD_FONT, MID_FILL, NOTE_FILL, TITLE_FONT, _print_setup, _write_info
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from . import __version__
from .analysis import rate, wilson
from .compare import (NO_EMOTION, VIEW_FOLD, VIEW_HELP, VIEW_JA, VIEW_PICK, VIEW_STRICT, ModelCompare, ViewEval, compare_models,
                      dataset_names, predict)
from .dataset import Clip, describe_counts
from .labels import EMO_OTHER, EMOTIONS, FOUR, JA, NOTAG, ORDER, SIX, UNKNOWN, predicted_label
from .report import ReportData, _cell, _header, _note, _num, _section
from .runners import ModelRun

SHEET_SUMMARY = "まとめ"
SHEET_CONF = "感情の混同表"
SHEET_RECALL = "感情ごとの正解率"
SHEET_CER = "文字誤り率"
SHEET_CLIPS = "1件ごとの結果"
SHEET_EXPLAIN = "説明"

PLAN_TARGET = 0.70      # PLAN §5.3「感情の正解率（音声から、4分類）」の初期目標
KPI_CELL = "$B$4"       # まとめの KPI の値のセル。ここを書き換えると色分けが変わる
CF_GOOD = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")     # 条件付き書式用（背景色を start/end の両方に入れる）
CF_BAD = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")


def _kpi_color(ws: Worksheet, rng: str) -> None:
    """rng の数字を、KPI（B4）以上なら緑、未満なら赤にする。"""
    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThanOrEqual", formula=[KPI_CELL], fill=CF_GOOD, font=GOOD_FONT))
    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=[KPI_CELL], fill=CF_BAD, font=BAD_FONT))


def _pct(ws: Worksheet, r: int, c: int, v: Any, bold: bool = False) -> Any:
    return _cell(ws, r, c, v, "0.0%", bold=bold)


def _label(ev: ViewEval) -> str:
    return VIEW_JA[ev.view]


def _ja(label: str) -> str:
    return JA.get(label, label)


def _dataset_of(c: Clip) -> str:
    return c.dataset or "(指定なし)"


# ---------------------------------------------------------------------------
# 結論（数字から機械的に作る文）
# ---------------------------------------------------------------------------
def conclusions(cmp: list[ModelCompare]) -> list[str]:
    out: list[str] = []
    best: tuple[float, str, str] | None = None
    reached: list[str] = []
    for mc in cmp:
        parts = []
        for v, e in mc.four.items():
            parts.append(f"{VIEW_JA[v]} {e.accuracy:.1%}（平均再現率 {e.balanced:.1%}）")
            if best is None or e.accuracy > best[0]:
                best = (e.accuracy, mc.run.label, VIEW_JA[v])
            if e.accuracy >= PLAN_TARGET:
                reached.append(f"{mc.run.label}／{VIEW_JA[v]}")
        if parts:
            out.append(f"4分類の正解率 — {mc.run.label}: " + " ／ ".join(parts))
    if best:
        out.append(f"4分類の正解率がいちばん高かったのは {best[1]}（{best[2]}）の {best[0]:.1%}。"
                   + (f"KPI {PLAN_TARGET:.0%} 以上: {'、'.join(reached)}。" if reached else f"KPI {PLAN_TARGET:.0%} に届いた組み合わせはありません。"))
    for mc in cmp:
        if mc.neutral:
            n = mc.neutral
            out.append(f"平常の読み上げ（{n.n} 件）に感情を付けなかった率 — {mc.run.label}: {n.no_emotion_rate:.1%}（平常と判定 {n.neutral_rate:.1%}）")
    for mc in cmp:
        if "strict" in mc.six:
            e = mc.six["strict"]
            lab, cnt = e.top_pred()
            out.append(f"6感情の正解率（JVNV・そのまま） — {mc.run.label}: {e.accuracy:.1%}（最も多く出した予測は {_ja(lab)} {rate(cnt, e.n):.0%}）")
    return out


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------
NOTES = [
    "FLEURS（平常）と JVNV（感情）は、別のコーパスです。話者・マイク・部屋・1本の長さ（FLEURS 平均 約13秒、JVNV 約8秒）が違うので、"
    "モデルが『感情』ではなく『録音の違い』で見分けている可能性があります。4分類の数字は、会議の音声での正解率とは別物です（要検証。模擬会議で測り直す）。",
    "JVNV の喜び・怒り・悲しみは、演技で誇張された感情です。会議の自然な発言のほうが感情は弱いので、会議での正解率はこれより低くなる可能性があります（要検証）。",
    "FLEURS の『平常』は、感情を指定せず文章を読み上げた録音を、平常と見なしたものです（人手で感情を確認してはいません）。話者は男性が約89%で、人数も少ないです。",
    "4分類の件数は 平常（FLEURS）と 喜び・怒り・悲しみ（JVNV）で違うので、正解率（件数で重みを付けた値）と、平均再現率（4つを同じ重みで平均した値）の両方を載せています。",
    "emotion2vec+ の重み（モデル本体）のライセンスは FunASR のモデルライセンスで、『参考と学習の目的のみ』という文言があります。商用利用の可否は法務確認が未了です（PLAN Q5）。",
    "同じ文章を複数の話者が読んでいる（JVNV は4人が同じ文、FLEURS は1文を最大3人）ので、件数ほど独立なデータではありません。95%区間は実際より狭く見える可能性があります。",
]


def _write_summary(ws: Worksheet, d: ReportData, cmp: list[ModelCompare]) -> None:
    ws.title = SHEET_SUMMARY
    widths = [34, 30, 14, 14, 14, 12, 12, 16, 18, 16, 16, 16, 16, 16, 16, 16, 16]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws["A1"] = "感情の聞き分け 比較：4分類の KPI（平常・喜び・怒り・悲しみ）を測る"
    ws["A1"].font = TITLE_FONT
    names = dataset_names(d.clips)
    ws["A2"] = "音声 " + " + ".join(f"{n} {sum(1 for c in d.clips if _dataset_of(c) == n)} 件（{describe_counts([c for c in d.clips if _dataset_of(c) == n])}）" for n in names)
    ws["A3"] = ("これは『今の値（ベースライン）』を測った結果です。数字の意味は『説明』、使ったモデル・データの出典・ライセンスは『実行情報』にあります。"
                "平常の音声（FLEURS）と感情の音声（JVNV）が別のコーパスである点に注意してください（下の『読むときの注意』）。")
    ws["A3"].font = Font(italic=True)
    ws["A4"] = "KPI（PLAN §5.3）：4分類の正解率の目標"
    ws["A4"].font = Font(bold=True)
    kc = ws["B4"]
    kc.value, kc.number_format, kc.font, kc.fill, kc.border = PLAN_TARGET, "0%", Font(bold=True, size=12), MID_FILL, BORDER
    ws["C4"] = "← この数字を書き換えると、下の緑（達成）・赤（未達）の色が変わります"
    r = 6

    # ---- 結論 ----
    r = _section(ws, r, "見どころ（数字から自動で作った文）")
    for line in conclusions(cmp):
        r = _note(ws, r, "・" + line, 12, fill=NOTE_FILL)
    r += 1

    # ---- ① 4分類 ----
    r = _section(ws, r, "① 4分類の正解率（平常・喜び・怒り・悲しみ）— PLAN の KPI")
    four_any = next((e for mc in cmp for e in mc.four.values()), None)
    if four_any is None:
        r = _note(ws, r, "平常の音声（と、喜び・怒り・悲しみの音声）が無いので、4分類は測れません。--set で FLEURS などの平常の音声を足してください。", 12, 34, MID_FILL)
    else:
        parts = "・".join(f"{n} {k} 件" for n, k in four_any.datasets.items())
        r = _note(ws, r, f"対象：正解が 平常・喜び・怒り・悲しみ の音声 {four_any.n} 件（{parts}）。モデルが 恐れ・嫌悪・驚き・その他・不明・タグなし と答えたら、"
                         f"『そのまま』の見方では不正解です。見方は下の 3 種類です。", 12, 34)
        for v in (VIEW_STRICT, VIEW_FOLD, VIEW_PICK):
            r = _note(ws, r, f"　{VIEW_JA[v]}：{VIEW_HELP[v]}", 12, 20, NOTE_FILL)
        _header(ws, r, ["モデル", "見方", "4分類の正解率", "95%区間（下）", "95%区間（上）", "正解数", "件数", "KPI 判定", "平均再現率\n（4つを同じ重みで）"])
        r += 1
        first = r
        for mc in cmp:
            for v, e in mc.four.items():
                lo, hi = e.ci
                _cell(ws, r, 1, mc.run.label, wrap=True)
                _cell(ws, r, 2, VIEW_JA[v], wrap=True)
                _pct(ws, r, 3, e.accuracy, True)
                _pct(ws, r, 4, lo)
                _pct(ws, r, 5, hi)
                _cell(ws, r, 6, e.correct, "#,##0")
                _cell(ws, r, 7, e.n, "#,##0")
                _cell(ws, r, 8, "達成" if e.accuracy >= PLAN_TARGET else "未達", bold=True)
                _pct(ws, r, 9, e.balanced, True)
                r += 1
        if r > first:
            _kpi_color(ws, f"C{first}:C{r - 1}")
            _kpi_color(ws, f"I{first}:I{r - 1}")
        r += 1

        # ---- ④ クラス別 ----
        r = _section(ws, r, "② 4分類のクラス別の成績（再現率＝その感情の音声のうち当てられた割合 / 適合率＝その感情と答えた音声のうち本当にそうだった割合）")
        head = ["モデル", "見方"]
        for t in FOUR:
            head += [f"{JA[t]}\n再現率", f"{JA[t]}\n適合率"]
        head += ["いちばん多い間違い"]
        _header(ws, r, head)
        r += 1
        for mc in cmp:
            for v, e in mc.four.items():
                _cell(ws, r, 1, mc.run.label, wrap=True)
                _cell(ws, r, 2, VIEW_JA[v], wrap=True)
                tops = []
                for j, t in enumerate(FOUR):
                    row = e.row(t)
                    _pct(ws, r, 3 + 2 * j, row.recall if row else None)
                    _pct(ws, r, 4 + 2 * j, row.precision if row else None)
                    if row and row.top_wrong:
                        tops.append(f"{JA[t]}→{_ja(row.top_wrong)}（{row.top_wrong_n}件）")
                _cell(ws, r, 11, "　".join(tops), wrap=True)
                r += 1
        ws.column_dimensions["K"].width = 70
        r += 1

    # ---- ③ 平常の音声 ----
    neutral_models = [mc for mc in cmp if mc.neutral]
    r = _section(ws, r, "③ 平常の読み上げ音声（FLEURS）に、感情を付けてしまわないか")
    if not neutral_models:
        r = _note(ws, r, "平常の音声が無いので測れていません。", 12)
    else:
        r = _note(ws, r, "『感情が付かなかった率』＝ 平常・その他（other）・不明・タグなし と判定された割合（pyVideoTrans は、不明・タグなしを平常として扱うので、"
                         "感情が付かなかった＝平常として扱われる）。残りは 平常の音声に何かの感情を付けた誤りです。", 17, 34)
        head = ["モデル", "件数", "感情が付かなかった率", "95%区間（下）", "95%区間（上）", "感情を付けた率（誤り）", "4つから最大を選ぶと平常"] + [f"内訳：{_ja(x)}" for x in (*ORDER, EMO_OTHER, UNKNOWN, NOTAG)]
        _header(ws, r, head)
        r += 1
        for mc in neutral_models:
            n = mc.neutral
            lo, hi = n.ci(*NO_EMOTION)
            _cell(ws, r, 1, mc.run.label, wrap=True)
            _cell(ws, r, 2, n.n, "#,##0")
            _pct(ws, r, 3, n.no_emotion_rate, True)
            _pct(ws, r, 4, lo)
            _pct(ws, r, 5, hi)
            _pct(ws, r, 6, n.emotion_rate)
            _pct(ws, r, 7, n.pick4_rate)
            for j, x in enumerate((*ORDER, EMO_OTHER, UNKNOWN, NOTAG)):
                _pct(ws, r, 8 + j, n.share(x))
            r += 1
        r += 1

    # ---- ④ JVNV の 6 感情 ----
    six_models = [mc for mc in cmp if mc.six]
    r = _section(ws, r, "④ 感情のある音声（JVNV）の正解率（喜び・怒り・悲しみ・恐れ・嫌悪・驚きの6感情。平常は無い）")
    if not six_models:
        r = _note(ws, r, "感情のある音声が無いので測れていません。", 12)
    else:
        head = ["モデル", "見方", "6感情の正解率", "95%区間（下）", "95%区間（上）", "正解数", "件数", "平均再現率"] + [f"{JA[t]}\n再現率" for t in SIX] + ["いちばん多く出した予測", "その割合"]
        _header(ws, r, head)
        r += 1
        for mc in six_models:
            for v in (VIEW_STRICT, VIEW_PICK):
                e = mc.six.get(v)
                if e is None:
                    continue
                lo, hi = e.ci
                _cell(ws, r, 1, mc.run.label, wrap=True)
                _cell(ws, r, 2, VIEW_JA[v] if v == VIEW_STRICT else "6感情の中から最大を選ぶ", wrap=True)
                _pct(ws, r, 3, e.accuracy, True)
                _pct(ws, r, 4, lo)
                _pct(ws, r, 5, hi)
                _cell(ws, r, 6, e.correct, "#,##0")
                _cell(ws, r, 7, e.n, "#,##0")
                _pct(ws, r, 8, e.balanced)
                for j, t in enumerate(SIX):
                    row = e.row(t)
                    _pct(ws, r, 9 + j, row.recall if row else None)
                lab, cnt = e.top_pred()
                _cell(ws, r, 15, _ja(lab))
                _pct(ws, r, 16, rate(cnt, e.n))
                r += 1
        r = _note(ws, r, "6感情でのでたらめの当たりは約17%。『いちばん多く出した予測』の割合が大きいほど、そのモデルは特定の感情に偏って答えています（偏ると、その感情の再現率だけが高くなる）。", 16, 34)
        r += 1

    # ---- ⑤ 4分類の混同表 ----
    if four_any is not None:
        r = _section(ws, r, "⑤ 4分類の混同表（行＝正解、列＝モデルの予測。件数。緑＝正解）。恐れ・嫌悪・驚きの正解は 4分類の対象外なので行には出ません")
        for mc in cmp:
            for v, e in mc.four.items():
                cols = e.pred_labels
                ws.cell(r, 1, f"{mc.run.label}（{VIEW_JA[v]}）").font = Font(bold=True)
                r += 1
                _header(ws, r, ["正解 ＼ 予測", *[_ja(p) for p in cols], "合計", "再現率"])
                r += 1
                for t in e.truth:
                    n = sum(e.matrix[t].values())
                    _cell(ws, r, 1, JA[t], bold=True)
                    for j, p in enumerate(cols, start=2):
                        c = _cell(ws, r, j, e.matrix[t][p], "#,##0")
                        if p == t:
                            c.fill, c.font = GOOD_FILL, Font(bold=True)
                    _cell(ws, r, len(cols) + 2, n, "#,##0", bold=True)
                    _pct(ws, r, len(cols) + 3, e.matrix[t][t] / n if n else None, True)
                    r += 1
                r += 1

    # ---- ⑥ 速度 ----
    r = _section(ws, r, "⑥ 処理速度（CPU のみ。RTF＝処理にかかった秒 ÷ 音声の長さ。1より小さいほど実時間より速い）")
    _header(ws, r, ["モデル", "データ", "件数", "音声の長さ（分）", "処理時間（秒）", "RTF", "1件あたり（秒）", "CPU スレッド数"])
    r += 1
    for mc in cmp:
        for n in dataset_names(d.clips):
            sp = mc.speed_by_dataset.get(n)
            if not sp or not sp["n"]:
                continue
            files = {c.file for c in d.clips if _dataset_of(c) == n}
            thr = sorted({x.threads for f, x in mc.run.results.items() if f in files and x.threads})
            _cell(ws, r, 1, mc.run.label, wrap=True)
            _cell(ws, r, 2, n)
            _cell(ws, r, 3, sp["n"], "#,##0")
            _cell(ws, r, 4, sp["audio_sec"] / 60, "0.0")
            _cell(ws, r, 5, sp["proc_sec"], "#,##0")
            _cell(ws, r, 6, sp["rtf"], "0.000", bold=True)
            _cell(ws, r, 7, sp["sec_per_clip"], "0.00")
            _cell(ws, r, 8, "・".join(map(str, thr)) if thr else "(記録なし)")
            r += 1
    r = _note(ws, r, ("RTF は、別の重い処理が同時に動いている CPU（4コア）で、1 本ずつ処理して測った参考値です。SenseVoice の JVNV の分は、前回の測定（スレッド数 1）を再利用しています。"
                      "本番 PC の速さの判断には使わないでください。モデルの読み込み時間は含みません（『実行情報』にあります）。"), 12, 48)
    r += 1

    # ---- ⑦ CER ----
    cer_models = [mc for mc in cmp if mc.cer_by_dataset]
    if cer_models:
        r = _section(ws, r, "⑦ おまけ：日本語の文字起こしの文字誤り率（CER。SenseVoiceSmall。小さいほど良い）")
        _header(ws, r, ["モデル", "データ", "音声の件数", "正解の文字数", "CER", "CER（正規化なし）", "置換", "脱落", "挿入"])
        r += 1
        for mc in cer_models:
            for n, t in mc.cer_by_dataset.items():
                c = t.overall.counts
                _cell(ws, r, 1, mc.run.label)
                _cell(ws, r, 2, n)
                _cell(ws, r, 3, t.n_clips, "#,##0")
                _cell(ws, r, 4, c.n_ref, "#,##0")
                _pct(ws, r, 5, t.overall.cer, True)
                _pct(ws, r, 6, t.raw_overall.cer if t.raw_overall else None)
                _cell(ws, r, 7, c.sub, "#,##0")
                _cell(ws, r, 8, c.dele, "#,##0")
                _cell(ws, r, 9, c.ins, "#,##0")
                r += 1
        r = _note(ws, r, "CER は、正解と認識結果の両方に同じ整え方（全角半角・句読点・数字の書き方をそろえる）をかけて測っています。FLEURS の正解の文は raw_transcription（句読点つきの元の文）です。"
                         "朗読音声なので、会議（雑音・重なり・言いよどみ）の CER はこれより悪くなる見込みです。", 12, 48)
        r += 1

    # ---- 注意 ----
    r = _section(ws, r, "読むときの注意")
    for n in [*d.dataset_notes, *NOTES, *d.notes]:
        r = _note(ws, r, "・" + n, 12, fill=MID_FILL if n in d.notes else NOTE_FILL)
    ws.freeze_panes = "A5"


# ---------------------------------------------------------------------------
# 感情の混同表（全部の予測の列つき）
# ---------------------------------------------------------------------------
def _matrix_block(ws: Worksheet, r: int, title: str, e: ViewEval) -> int:
    cols = e.pred_labels
    r = _section(ws, r, title)
    _header(ws, r, ["正解 ＼ 予測", *[_ja(p) for p in cols], "合計", "正解率"], widths=[16] + [11] * (len(cols) + 2))
    r += 1
    for t in e.truth:
        n = sum(e.matrix[t].values())
        _cell(ws, r, 1, JA[t], bold=True)
        for j, p in enumerate(cols, start=2):
            c = _cell(ws, r, j, e.matrix[t][p], "#,##0")
            if p == t:
                c.fill, c.font = GOOD_FILL, Font(bold=True)
        _cell(ws, r, len(cols) + 2, n, "#,##0", bold=True)
        _pct(ws, r, len(cols) + 3, e.matrix[t][t] / n if n else None, True)
        r += 1
    _cell(ws, r, 1, "合計", bold=True)
    for j, p in enumerate(cols, start=2):
        _cell(ws, r, j, sum(e.matrix[t][p] for t in e.truth), "#,##0", bold=True)
    _cell(ws, r, len(cols) + 2, e.n, "#,##0", bold=True)
    _pct(ws, r, len(cols) + 3, e.accuracy, True)
    r += 2
    ws.cell(r - 1, 1, "各行の割合（合計＝100%。濃い色ほど多い）").font = Font(italic=True)
    _header(ws, r, ["正解 ＼ 予測", *[_ja(p) for p in cols]])
    r += 1
    first = r
    for t in e.truth:
        n = sum(e.matrix[t].values())
        _cell(ws, r, 1, JA[t], bold=True)
        for j, p in enumerate(cols, start=2):
            _pct(ws, r, j, e.matrix[t][p] / n if n else None)
        r += 1
    ws.conditional_formatting.add(f"B{first}:{get_column_letter(len(cols) + 1)}{r - 1}",
                                  ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1, end_color="5B9BD5"))
    return r + 1


def _write_conf(ws: Worksheet, cmp: list[ModelCompare]) -> None:
    ws.title = SHEET_CONF
    ws.column_dimensions["A"].width = 16
    ws["A1"] = "感情の混同表（行＝正解の感情、列＝モデルの予測。件数と、行ごとの割合）"
    ws["A1"].font = TITLE_FONT
    r = 3
    for mc in cmp:
        evs = mc.seven or mc.six
        for v in (VIEW_STRICT, VIEW_PICK):
            e = evs.get(v)
            if e is None:
                continue
            what = "7感情（平常 + 6感情）" if mc.seven else "6感情"
            title = f"{mc.run.label}：{what}（{'そのまま' if v == VIEW_STRICT else '対象の感情の中から最大を選ぶ'}）"
            r = _matrix_block(ws, r, title, e)
    if r == 3:
        ws.cell(r, 1, "感情を出すモデルを実行していません")


# ---------------------------------------------------------------------------
# 感情ごとの正解率
# ---------------------------------------------------------------------------
def _write_recall(ws: Worksheet, d: ReportData, cmp: list[ModelCompare]) -> None:
    ws.title = SHEET_RECALL
    ws["A1"] = "感情ごとの正解率（再現率）・適合率と、話者ごとの違い"
    ws["A1"].font = TITLE_FONT
    r = 3
    if not cmp:
        ws.cell(r, 1, "感情を出すモデルを実行していません")
        return
    for mc in cmp:
        evs = mc.seven or mc.six
        for v in (VIEW_STRICT, VIEW_FOLD, VIEW_PICK):
            e = evs.get(v)
            if e is None or (v == VIEW_FOLD and mc.seven is None):
                continue
            r = _section(ws, r, f"{mc.run.label}：{'7感情' if mc.seven else '6感情'}（{VIEW_JA[v] if v != VIEW_PICK else '対象の感情の中から最大を選ぶ'}）の感情ごとの成績")
            _header(ws, r, ["正解の感情", "件数", "正解数", "正解率（再現率）", "95%区間（下）", "95%区間（上）", "いちばん多い間違い先", "その感情と予測された件数", "適合率"],
                    widths=[22, 10, 10, 14, 14, 14, 30, 14, 12])
            r += 1
            for row in e.rows:
                lo, hi = row.ci
                _cell(ws, r, 1, JA[row.label])
                _cell(ws, r, 2, row.n, "#,##0")
                _cell(ws, r, 3, row.correct, "#,##0")
                _pct(ws, r, 4, row.recall, True)
                _pct(ws, r, 5, lo)
                _pct(ws, r, 6, hi)
                _cell(ws, r, 7, f"{_ja(row.top_wrong)}（{row.top_wrong_n}件）" if row.top_wrong else "", wrap=True)
                _cell(ws, r, 8, row.pred_n, "#,##0")
                _pct(ws, r, 9, row.precision)
                r += 1
            lo, hi = e.ci
            _cell(ws, r, 1, "全体", bold=True)
            _cell(ws, r, 2, e.n, "#,##0", bold=True)
            _cell(ws, r, 3, e.correct, "#,##0", bold=True)
            _pct(ws, r, 4, e.accuracy, True)
            _pct(ws, r, 5, lo)
            _pct(ws, r, 6, hi)
            r += 1
            _cell(ws, r, 1, "平均再現率", bold=True)
            _pct(ws, r, 4, e.balanced, True)
            r += 2
        # 話者ごと（そのまま）
        r = _speaker_table(ws, r, d.clips, mc.run)
    r = _note(ws, r, ("適合率は『その感情と予測した音声のうち、本当にその感情だった割合』です。何でも同じ感情と答えるモデルは、その感情の再現率だけが高くなり、適合率は低くなります。"
                      "再現率と適合率を一緒に見てください。"), 9, 34)


def _speaker_table(ws: Worksheet, r: int, clips: list[Clip], run: ModelRun) -> int:
    """話者 × 正解の感情 の正解率（そのまま）。"""
    done = [c for c in clips if c.file in run.results and not run.results[c.file].error]
    speakers = sorted({c.speaker for c in done if c.speaker})
    truths = [t for t in ORDER if any(c.emotion == t for c in done)]
    if not speakers:
        return r
    r = _section(ws, r, f"{run.label}：話者 × 感情の正解率（そのまま。空欄＝その組み合わせの音声なし）")
    _header(ws, r, ["話者", *[JA[t] for t in truths], "全体", "件数"])
    r += 1
    for s in speakers:
        _cell(ws, r, 1, s, bold=True)
        mine = [c for c in done if c.speaker == s]
        for j, t in enumerate(truths, start=2):
            sub = [c for c in mine if c.emotion == t]
            _pct(ws, r, j, rate(sum(predicted_label(run.results[c.file].tag) == t for c in sub), len(sub)) if sub else None)
        _pct(ws, r, len(truths) + 2, rate(sum(predicted_label(run.results[c.file].tag) == c.emotion for c in mine), len(mine)), True)
        _cell(ws, r, len(truths) + 3, len(mine), "#,##0")
        r += 1
    return r + 1


# ---------------------------------------------------------------------------
# 文字誤り率
# ---------------------------------------------------------------------------
def _write_cer(ws: Worksheet, cmp: list[ModelCompare]) -> None:
    ws.title = SHEET_CER
    ws["A1"] = "文字誤り率（CER）— データごと、感情ごと、話者ごと（SenseVoiceSmall。小さいほど良い）"
    ws["A1"].font = TITLE_FONT
    r = 3
    tabs = [(mc, n, t) for mc in cmp for n, t in mc.cer_by_dataset.items()]
    if not tabs:
        ws.cell(r, 1, "正解の文章（text 列）が無い、または文字起こしをするモデルを実行していないので、CER は測れませんでした")
        return
    r = _section(ws, r, "データごとの CER")
    _header(ws, r, ["モデル", "データ", "音声の件数", "正解の文字数", "誤り数（置換＋脱落＋挿入）", "CER", "CER（正規化なし）", "置換", "脱落", "挿入"], widths=[26, 14, 12, 12, 16, 10, 14, 9, 9, 9])
    r += 1
    for mc, n, t in tabs:
        c = t.overall.counts
        _cell(ws, r, 1, mc.run.label)
        _cell(ws, r, 2, n)
        _cell(ws, r, 3, t.n_clips, "#,##0")
        _cell(ws, r, 4, c.n_ref, "#,##0")
        _cell(ws, r, 5, c.errors, "#,##0")
        _pct(ws, r, 6, t.overall.cer, True)
        _pct(ws, r, 7, t.raw_overall.cer if t.raw_overall else None)
        _cell(ws, r, 8, c.sub, "#,##0")
        _cell(ws, r, 9, c.dele, "#,##0")
        _cell(ws, r, 10, c.ins, "#,##0")
        r += 1
    r += 1
    r = _section(ws, r, "感情ごと・話者ごとの CER")
    _header(ws, r, ["モデル", "データ", "区分", "グループ", "音声の件数", "正解の文字数", "CER"])
    r += 1
    for mc, n, t in tabs:
        for kind, rows in (("感情", t.by_emotion), ("話者", t.by_speaker)):
            for x in rows:
                _cell(ws, r, 1, mc.run.label)
                _cell(ws, r, 2, n)
                _cell(ws, r, 3, kind)
                _cell(ws, r, 4, JA.get(x.label, x.label) if kind == "感情" else x.label)
                _cell(ws, r, 5, x.n_clips, "#,##0")
                _cell(ws, r, 6, x.counts.n_ref, "#,##0")
                _pct(ws, r, 7, x.cer)
                r += 1


# ---------------------------------------------------------------------------
# 1件ごとの結果
# ---------------------------------------------------------------------------
def _write_clips(ws: Worksheet, d: ReportData, opts) -> None:
    from .analysis import clip_cer

    ws.title = SHEET_CLIPS
    head = ["No", "データ", "ファイル", "話者", "区分", "正解の感情", "正解の文章"]
    widths = [6, 10, 34, 12, 9, 11, 40]
    emo_runs = [r for r in d.runs if r.has_emotion]
    for run in emo_runs:
        head += [f"{run.label}\n予測（そのまま）", f"{run.label}\n一致"]
        widths += [16, 8]
        if run.has_scores:
            head += [f"{run.label}\n予測（4つから最大）", f"{run.label}\n一致（4つから）"] + [f"{run.label}\nスコア：{JA[t]}" for t in FOUR]
            widths += [16, 10] + [11] * 4
    cer_runs = [r for r in d.runs if r.report_cer]
    for run in cer_runs:
        head += [f"{run.label}\n認識結果", f"{run.label}\nCER", f"{run.label}\n処理秒"]
        widths += [40, 9, 9]
    _header(ws, 1, head, widths=widths)
    for i, c in enumerate(d.clips, start=1):
        row = i + 1
        vals: list[tuple[Any, str | None]] = [(i, None), (_dataset_of(c), None), (c.file, None), (c.speaker, None), (c.style, None), (JA[c.emotion], None), (c.text, None)]
        for run in emo_runs:
            res = run.results.get(c.file)
            if res is None or res.error:
                vals += [(None, None)] * 2
                if run.has_scores:
                    vals += [(None, None)] * 6
                continue
            pred = predicted_label(res.tag)
            vals += [(JA[pred], None), ("○" if pred == c.emotion else "×", None)]
            if run.has_scores:
                p4 = predict(res, VIEW_PICK, FOUR)
                vals += [(JA[p4] if p4 else None, None), (("○" if p4 == c.emotion else "×") if (p4 and c.emotion in FOUR) else None, None)]
                vals += [((res.scores or {}).get(t), "0.000") for t in FOUR]
        for run in cer_runs:
            res = run.results.get(c.file)
            if res is None:
                vals += [(None, None)] * 3
            elif res.error:
                vals += [(f"エラー: {res.error}", None), (None, None), (None, None)]
            else:
                cer = clip_cer(c.text, res.text, opts) if c.text.strip() else None
                vals += [(res.text, None), (cer, "0%"), (res.sec, "0.00")]
        for j, (v, fmt) in enumerate(vals, start=1):
            cell = _cell(ws, row, j, v, fmt)
            if v == "×":
                cell.font = Font(color="9C0006", bold=True)
            elif v == "○":
                cell.font = Font(color="006100", bold=True)
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(head))}{len(d.clips) + 1}"
    ws.cell(len(d.clips) + 3, 3, "※ 空欄は、そのモデルで処理していない件、または（スコアのないモデルの）4つから最大を選ぶ見方。『一致（4つから）』は正解が 平常・喜び・怒り・悲しみ の件だけ。")


# ---------------------------------------------------------------------------
# 説明
# ---------------------------------------------------------------------------
EXPLAIN = [
    ("4分類の正解率（KPI）", "正解が 平常・喜び・怒り・悲しみ の音声のうち、モデルの予測が正解と一致した件数 ÷ 件数。PLAN §5.3 の KPI（70%以上）。",
     "4つのでたらめの当たりは25%（件数の偏りで変わる）。件数は平常（FLEURS）と3感情（JVNV）で違うので、平均再現率も一緒に見る。"),
    ("平均再現率", "平常・喜び・怒り・悲しみ それぞれの再現率の平均（4つを同じ重みで）。", "件数の少ない感情も同じ重みで見る。正解率との差が大きいと、感情によって得意不得意がある。"),
    ("見方：そのまま", VIEW_HELP[VIEW_STRICT], "いちばん厳しい見方。モデルが自信なし（不明・その他）のときも不正解になる。"),
    ("見方：不明・その他を平常に読み替え", VIEW_HELP[VIEW_FOLD], "pyVideoTrans の今の作りでの結果に近い。SenseVoice は不明が 97.6% なので、ほぼ全部が平常になる。"),
    ("見方：対象の中から最大を選ぶ", VIEW_HELP[VIEW_PICK], "『4つに絞れば、どこまで当たるか』の上限に近い見方。スコアを出さない SenseVoice のタグでは計算できない。"),
    ("感情が付かなかった率", "平常の音声が 平常・その他（other）・不明・タグなし と判定された割合。", "高いほど、平常の発言に余計な感情を付けない。pyVideoTrans は不明・タグなしを平常として扱う。"),
    ("再現率", "ある感情の音声のうち、その感情と当てられた割合。", "低い感情は、別の感情や『平常』『不明』と間違えられている。混同表で行き先を見る。"),
    ("適合率", "ある感情と予測した音声のうち、本当にその感情だった割合。", "再現率だけでは、何でも同じ感情と答えるモデルが良く見える。適合率が低い感情は、ほかの感情の音声を取り込んでいる。"),
    ("混同表", "行＝正解の感情、列＝予測。対角線（左上から右下）が正解。", "対角線以外の数字が大きい所が、間違えやすい組み合わせ。"),
    ("95%区間", "件数が有限なので、本当の割合がこの範囲に入る確率が約95%という幅（Wilson の方法）。", "幅が広いのは件数が少ないから。同じ文を複数人が読むので、実際の幅はもう少し広い可能性がある。"),
    ("emotion2vec+ の 9 分類", "angry（怒り）・disgusted（嫌悪）・fearful（恐れ）・happy（喜び）・neutral（平常）・other（その他）・sad（悲しみ）・surprised（驚き）・unknown（不明）。",
     "other と unknown は『感情を判定しない』に近い出力。表では別の列に分けて数える。"),
    ("CER（文字誤り率）", "(置換 + 脱落 + 挿入) ÷ 正解の文字数。グループ全体の誤りの合計で計算する。", "小さいほど良い。0% が完全一致。"),
    ("RTF", "処理にかかった秒 ÷ 音声の長さ。モデルの読み込みは含まない。", "1より小さいと実時間より速い。他の処理と同時に動かした CPU の値なので参考。"),
    ("ドメインのずれ", "平常の音声（FLEURS）と感情の音声（JVNV）で、話者・マイク・部屋・長さが違うこと。",
     "モデルが『感情』ではなく『録音の違い』で見分ける可能性がある。会議の音声（平常も感情も同じ録音）で測り直すまで、4分類の数字は目安。"),
]


def _write_explain(ws: Worksheet) -> None:
    ws.title = SHEET_EXPLAIN
    ws["A1"] = "用語の説明"
    ws["A1"].font = TITLE_FONT
    _header(ws, 3, ["用語", "意味", "見方"], widths=[34, 70, 70])
    for i, (a, b, c) in enumerate(EXPLAIN, start=4):
        _cell(ws, i, 1, a, wrap=True, bold=True)
        _cell(ws, i, 2, b, wrap=True)
        _cell(ws, i, 3, c, wrap=True)


# ---------------------------------------------------------------------------
# 実行情報
# ---------------------------------------------------------------------------
def info_items(d: ReportData, cmp: list[ModelCompare]) -> list[tuple[str, Any]]:
    names = dataset_names(d.clips)
    items: list[tuple[str, Any]] = [
        ("ツール", f"emocheck {__version__}（compare）"),
        ("実行日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("コマンド", d.command),
        ("データ", d.dataset_name or "(指定なし)"),
        *d.dataset_info,
        ("表示（出典の書き方）", ("JVNV コーパス（arXiv:2310.06072。非言語音声を除いた版 litagin/jvnv_corpus_v1_no_nv）: CC BY-SA 4.0 / "
                              "FLEURS（Conneau et al., 2022, arXiv:2205.12446。google/fleurs ja_jp）: CC BY 4.0。"
                              "この Excel は両コーパスの文章と、モデルの出力を含む派生物なので、社外に出すときは出典の表示と、JVNV の分は同じ条件（CC BY-SA 4.0）での共有に従う")),
        ("ラベル CSV", d.labels_csv),
        ("音声フォルダ", d.audio_dir),
    ]
    for n in names:
        part = [c for c in d.clips if _dataset_of(c) == n]
        spk = Counter_str([c.speaker for c in part if c.speaker])
        items.append((f"音声の件数：{n}", f"{len(part)} 件（{describe_counts(part)}）" + (f" / 話者：{spk}" if spk else "")))
    items += [
        ("文字の整え方（CER）", d.opts.describe() + "（asrcheck と同じ）"),
        ("集計の方法", ("感情: 正解率＝一致件数÷件数、95%区間は Wilson 法。4分類＝正解が 平常・喜び・怒り・悲しみ の音声。"
                      "見方 3 種（そのまま／不明・その他を平常に読み替え／対象の中から最大を選ぶ）は『説明』シート。CER: 1件ごとに編集距離を数え、グループ内で誤り合計÷正解の文字数")),
        ("SenseVoice の読み替え", "感情タグ→内部の感情名は videotrans/util/emotion.py の SENSEVOICE_TAG_MAP と同じ。違い: EMO_UNKNOWN・タグなしを平常にせず別の列で数える（『読み替え』の見方で平常にする）"),
        ("emotion2vec+ の読み替え", "9 分類 → angry/disgusted/fearful/happy/neutral/sad/surprised はそのまま、other は『その他（other）』、<unk> は『不明』の別の列。『読み替え』の見方では other・不明を平常にする（pyVideoTrans の SenseVoice の扱いに合わせた仮の読み替え。要検討）"),
    ]
    by_key = {mc.run.key: mc for mc in cmp}
    for run in d.runs:
        sp_all = {"n": 0, "proc_sec": 0.0, "audio_sec": 0.0}
        for r in run.results.values():
            if not r.error:
                sp_all["n"] += 1
                sp_all["proc_sec"] += r.sec
        sp_all["audio_sec"] = sum(run.durations.get(f, 0.0) for f, r in run.results.items() if not r.error)
        rtf = sp_all["proc_sec"] / sp_all["audio_sec"] if sp_all["audio_sec"] else float("nan")
        items.append((f"── モデル: {run.label} ──", ""))
        items += run.info
        items.append(("処理した範囲", d.subsets.get(run.key, "全件")))
        items.append(("処理件数", f"{sp_all['n']} 件（うち前回の結果の再利用 {run.n_cached} 件 / 失敗 {run.n_errors} 件）"))
        used = sorted({r.threads for r in run.results.values() if r.threads})
        items.append(("CPU スレッド数", "・".join(map(str, used)) + "（その結果を出したときの設定）" if used else "(記録なし)"))
        items.append(("処理時間", f"モデル読み込み {run.load_sec:.1f} 秒 / 認識 合計 {sp_all['proc_sec']:.1f} 秒 / 音声 合計 {sp_all['audio_sec']:.1f} 秒 / RTF {rtf:.3f}"
                              + ("（今回の実行ぶん以外は、前回の測定の時間）" if run.n_cached else "")))
        if run.key in by_key and by_key[run.key].run.has_scores:
            items.append(("出力", "9 分類のスコア（softmax。合計 1）を 1 件ごとに保存。『1件ごとの結果』のスコアは 平常・喜び・怒り・悲しみ"))
        if run.key.startswith("emotion2vec"):
            items.append(("法務確認", "未了（PLAN Q5：emotion2vec+ の重みのライセンスの法務確認）。判断が出るまで、製品への組み込みは決めない"))
    items += [
        ("── 実行環境 ──", ""),
        ("全体の所要時間", f"{d.elapsed_sec:.1f} 秒（この実行での合計。前回の結果を再利用した分は含まない）"),
        ("計算機", f"{platform.platform()} / Python {platform.python_version()} / CPU {os.cpu_count()} コア（GPU なし）"),
        ("注意（処理速度）", "別の重い処理（スコア計算など）と同時に動かした環境で測った参考値。本番 PC の速さではない"),
        ("注意（ライセンス）", ("『1件ごとの結果』シートに JVNV・FLEURS の書き起こし文章が入っている。JVNV は CC BY-SA 4.0（表示・同じ条件での共有）、"
                            "FLEURS は CC BY 4.0（表示）に従って扱う。社外に出す前に確認する")),
    ]
    return items


def Counter_str(vals: list[str]) -> str:
    from collections import Counter

    return "・".join(f"{k} {v}" for k, v in sorted(Counter(vals).items()))


def write_compare_report(path: str | Path, d: ReportData) -> Path:
    cmp = compare_models(d.clips, d.runs, d.opts)
    wb = Workbook()
    _write_summary(wb.active, d, cmp)
    _write_conf(wb.create_sheet(), cmp)
    _write_recall(wb.create_sheet(), d, cmp)
    _write_cer(wb.create_sheet(), cmp)
    _write_clips(wb.create_sheet(), d, d.opts)
    _write_explain(wb.create_sheet())
    _write_info(wb.create_sheet(), info_items(d, cmp))
    for ws in wb.worksheets:
        _print_setup(ws)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out
