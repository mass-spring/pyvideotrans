"""登録音声との照合の結果を 1 つの Excel（日本語のシート名・見出し）にまとめる。

シート: まとめ / 全指標 / 登録時間ごとの比較 / 話者混同 / 不明と平滑化 / 説明 / 実行情報
色（緑＝目標達成 / 黄＝最低ライン達成 / 赤＝未達）と採点の中身は、meetingbench の他の Excel（asrcheck）と同じ。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from asrcheck import report as ar
from asrcheck.evaluate import DER_COLLARS, EvalOptions
from asrcheck.kpi import FAIL, MINIMUM, TARGET, Kpi
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from . import enroll_assign as ea
from .enroll_embed import EMBEDDERS
from .excel import CFG_JA, SEG_JA, VAR_JA
from .excel import _hdr, _val

EMB_SHORT = {"ecapa": "ECAPA", "titanet": "TitaNet small", "eres2net": "ERes2Net large", "campp": "CAM++"}
EMB_NOTE = {"ecapa": "英語中心(VoxCeleb)", "titanet": "英語用", "eres2net": "中国語用", "campp": "中国語用"}
MODE_SHORT = {"mixed": "感情を混ぜる", "calm": "落ち着いた感情だけ"}


def emb_label(e: str) -> str:
    return f"{EMB_SHORT.get(e, e)}（{EMB_NOTE.get(e, '')}）" if e in EMB_NOTE else EMB_SHORT.get(e, e)


def pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _style(cell, kpi: Kpi, v: float | None) -> None:
    j = kpi.judge(v)
    if j in ar.KPI_STYLE:
        cell.fill, cell.font = ar.KPI_STYLE[j]


def _key(e, sec, mode, rule, v, g, unk=None, smooth=""):
    return (e, sec, mode, rule, v, g, unk, smooth)


class Data:
    """集計済みの結果と、比べる相手（built）をまとめて渡すための入れ物。"""

    def __init__(self, agg: dict, baselines, bench, results: list[dict], extras: dict):
        self.agg, self.base, self.bench, self.results, self.extras = agg, baselines, bench, results, extras
        self.embedders = bench.embedders
        self.secs = sorted({k[1] for k in agg if k[6] is None and not k[7]})
        self.kpis = bench.kpis

    def a(self, e, sec, mode, rule, v, g, unk=None, smooth="") -> dict | None:
        return self.agg.get(_key(e, sec, mode, rule, v, g, unk, smooth))


# ---------------------------------------------------------------------------
def facts(d: Data) -> list[str]:
    """数字から機械的に書く要点（解釈は人が書く）。"""
    out = []
    k_acc, k_der = d.kpis["spk_acc"], d.kpis["der"]
    for g in ("silero", "oracle"):
        for v in ("A", "B"):
            b = d.base.conds
            parts = []
            for name, ja in ((f"{g}__built-ja__n4", "built(ja)"), (f"{g}__built-zh__n4", "built(zh)"), (f"{g}__truth__none", "上限(話者だけ正解)")):
                c = b.get((v, name))
                if c:
                    parts.append(f"{ja} {pct(c.row_sub.get('spk_acc'))}/DER {pct(c.row_sub.get('der_0'))}")
            if parts:
                out.append(f"録音{v}・区間={g}: 比較相手（教師なし）— " + "、".join(parts) + "。")
    best: dict[str, tuple] = {}
    for g in ("silero", "oracle"):
        cands = []
        for e in d.embedders:
            for mode in ("mixed", "calm"):
                for sec in d.secs:
                    for rule in ("mean", "max"):
                        a, bb = d.a(e, sec, mode, rule, "A", g), d.a(e, sec, mode, rule, "B", g)
                        if a and bb and a["spk_acc"] is not None and bb["spk_acc"] is not None:
                            cands.append(((a["spk_acc"] + bb["spk_acc"]) / 2, e, sec, mode, rule, a, bb))
        if cands:
            best[g] = max(cands, key=lambda x: x[0])
            _, e, sec, mode, rule, a, bb = best[g]
            out.append(f"区間={g}で話者正解率（A と B の平均）が最も高かった登録条件: {emb_label(e)}・登録{sec}秒・{MODE_SHORT[mode]}・{ea.RULE_JA[rule]}"
                       f" → A {pct(a['spk_acc'])}（DER {pct(a['der_0'])}）、B {pct(bb['spk_acc'])}（DER {pct(bb['der_0'])}）。"
                       f"（登録発話の選び方5通りの平均。KPI 最低ライン 話者正解率 {k_acc.fmt(k_acc.minimum)} 以上、DER {k_der.fmt(k_der.minimum)} 以下）")
    # 条件ごとの下限・上限
    for g in ("silero",):
        vals = [(a["spk_acc"], k) for k, a in d.agg.items() if k[5] == g and k[6] is None and not k[7] and a["spk_acc"] is not None]
        if vals:
            lo = min(vals)
            out.append(f"区間={g}の全条件（埋め込み器×登録時間×感情×方式×録音）で、話者正解率が最も低かったのは {pct(lo[0])}"
                       f"（{emb_label(lo[1][0])}・{lo[1][1]}秒・{MODE_SHORT[lo[1][2]]}・{ea.RULE_JA[lo[1][3]]}・録音{lo[1][4]}）。")
    secs = d.secs
    for e in d.embedders:
        parts = []
        for v in ("A", "B"):
            seq = [(sec, d.a(e, sec, "mixed", "mean", v, "silero")) for sec in secs]
            parts.append(f"録音{v} " + " → ".join(f"{sec}秒 {pct(a['spk_acc'])}" for sec, a in seq if a))
        out.append(f"{emb_label(e)}・区間=silero・感情を混ぜる・平均との類似度: 登録の長さごとの話者正解率 " + " / ".join(parts) + "。")
    big = max(secs) if secs else None
    if big:
        for e in d.embedders:
            parts = []
            for v in ("A", "B"):
                m, c = d.a(e, big, "mixed", "mean", v, "silero"), d.a(e, big, "calm", "mean", v, "silero")
                if m and c:
                    parts.append(f"録音{v} 混ぜる {pct(m['spk_acc'])} / 落ち着いた感情だけ {pct(c['spk_acc'])}")
            if parts:
                out.append(f"{emb_label(e)}・登録{big}秒・区間=silero: 登録の感情の違い — " + "、".join(parts) + "。")
    for w in d.extras.get("window", []):
        out.append(f"{emb_label(w['embedder'])}・録音{w['variant']}・登録180秒: 区間の中を窓で判定し話者の交代で切り直すと、話者正解率 {pct(w['acc_base'])} → {pct(w['acc_win'])}、"
                   f"DER(collar 0) {pct(w['der0_base'])} → {pct(w['der0_win'])}（切り直した区間は文字起こしのやり直しが必要。cpCER は未測定）。")
    nonid = [k for k, a in d.agg.items() if not a["map_identity_all"]]
    out.append("採点は asrcheck の標準（話者名の対応づけを自動で最適化）。名前をそのまま比べても同じかを確認: "
               + ("全条件で、対応づけは名前どおり（F1→F1 …）だった。" if not nonid else f"{len(nonid)} 条件で名前どおりでない対応づけがあった（全指標シートの『名前どおり』列を参照）。"))
    return out


# ---------------------------------------------------------------------------
def _summary(ws, d: Data, notes: list[str], recs: dict) -> None:
    ws.title = "まとめ"
    ws["A1"] = "登録した声との照合で話者名を付ける（PLAN §4.1 ②）— 疑似会議（JVNV 公開データ）での測定"
    ws["A1"].font = ar.TITLE_FONT
    ws["A2"] = ("録音 " + " / ".join(f"{v}: {r['minutes']:.1f}分・発言{r['turns']}件・重ねた発言{r['overlapped']}件" for v, r in recs.items())
                + "  ・話者4人（F1, F2, M1, M2）。各人の登録音声は『会議で使っていない発話』から作った（登録と会議は別の発話）。")
    ws["A3"] = ("注意: 登録の発話も会議の発話も、同じスタジオ録音を同じ処理（無音カット・音量そろえ・16kHz）に通したもので、登録と本番で録音条件が同じ"
                "（実際より有利）。雑音・反響・マイク違い・体調による声の変化がある実会議では下がる可能性が高い (要検証)。"
                "感情の幅が大きい朗読なので、『登録が感情を含むか』の影響は見える。各条件は登録発話の選び方を変えた5回の平均。")
    ws["A3"].alignment = ar.WRAP
    ws.merge_cells("A3:T3")
    ws.row_dimensions[3].height = 48
    r = 5
    ws.cell(r, 1, "KPI（PLAN §4.4 の初期案）と色").font = ar.SECTION_FONT
    for j, (k, text) in enumerate(((TARGET, "目標達成"), (MINIMUM, "最低ライン達成"), (FAIL, "未達")), start=2):
        c = ws.cell(r, j, text)
        c.fill, c.font = ar.KPI_STYLE[k]
    r += 1
    kp = d.kpis
    ws.cell(r, 1, "  話者正解率 " + " / ".join(kp["spk_acc"].rule) + "  ・  DER " + " / ".join(kp["der"].rule)
            + "  （最低ライン / 目標。DER の色は collar 0）。cpCER の KPI は未設定。")
    r += 2
    ws.cell(r, 1, "要点（数字から機械的に書いた事実。解釈は説明シートと、レポートの文章を参照）").font = ar.SECTION_FONT
    for t in notes:
        r += 1
        ws.cell(r, 1, "・" + t)
    cers = []
    for g in ("silero", "oracle"):
        for v in ("A", "B"):
            a = None
            for k, x in d.agg.items():
                if k[4] == v and k[5] == g and x["cer"] is not None:
                    a = x["cer"]
                    break
            if a is not None:
                cers.append(f"{g}・録音{v} {pct(a)}")
    r += 1
    ws.cell(r, 1, "・CER（話者に依らない。文字起こしは SenseVoiceSmall で同じ）: " + "、".join(cers) + "（KPI 最低ライン 15% 以下）。")
    r += 2
    heads = ["区間の切り方", "方式", "埋め込み", "登録\n(秒/人)", "登録の感情",
             "A 話者\n正解率", "A DER\n(collar 0)", "A DER\n(collar 0.25)", "A cpCER",
             "B 話者\n正解率", "B DER\n(collar 0)", "B DER\n(collar 0.25)", "B cpCER",
             "A 話者正解率\nの範囲(5回)", "B 話者正解率\nの範囲(5回)", "処理時間\n(1時間あたり・分)\n録音A", "RTF\n録音A", "KPI判定\n(録音A)"]
    widths = [26, 24, 26, 8, 22, 10, 10, 10, 9, 10, 10, 10, 9, 16, 16, 14, 8, 20]
    head = r
    _hdr(ws, head, heads, widths)
    first_seg = True
    for g in ("silero", "oracle"):
        r += 1
        ws.cell(r, 1, f"■ 区間の切り方: {SEG_JA[g]}").font = ar.SECTION_FONT
        # 比べる相手
        for name, ja, em in ((f"{g}__built-ja__n4", "built（教師なし）4人と指定", CFG_JA["built-ja"]),
                             (f"{g}__built-zh__n4", "built（教師なし）4人と指定", CFG_JA["built-zh"]),
                             (f"{g}__truth__none", "上限（話者だけ正解にした）", "区間の切り方の限界")):
            ca, cb = d.base.conds.get(("A", name)), d.base.conds.get(("B", name))
            if not ca or not cb:
                continue
            r += 1
            ra, rb = ca.row_sub, cb.row_sub
            vals = [SEG_JA[g], ja, em, None, None, ra.get("spk_acc"), ra.get("der_0"), ra.get("der_0.25"), ra.get("cpcer"),
                    rb.get("spk_acc"), rb.get("der_0"), rb.get("der_0.25"), rb.get("cpcer"), None, None, ra.get("minutes"),
                    ca.times.get("rtf"), ra.get("verdict")]
            _row(ws, r, vals, d.kpis, italic=True)
        for e in d.embedders:
            for mode in ("mixed", "calm"):
                for sec in d.secs:
                    a, b = d.a(e, sec, mode, "mean", "A", g), d.a(e, sec, mode, "mean", "B", g)
                    if not a or not b:
                        continue
                    r += 1
                    mins = a.get("minutes")
                    verdict = _verdict(d.kpis, a)
                    vals = [SEG_JA[g], "登録した声との照合（教師あり）", emb_label(e), sec, MODE_SHORT[mode],
                            a["spk_acc"], a["der_0"], a["der_0.25"], a["cpcer"], b["spk_acc"], b["der_0"], b["der_0.25"], b["cpcer"],
                            _rng(a), _rng(b), mins, (mins / 60 if mins is not None else None), verdict]
                    _row(ws, r, vals, d.kpis)
        first_seg = False
    ws.freeze_panes = ws.cell(head + 1, 4)
    r += 2
    ws.cell(r, 1, "読み方").font = ar.SECTION_FONT
    for t in [
        "『built（教師なし）』は、同じ録音・同じ区間・同じ文字起こしで測った pyVideoTrans 内蔵の話者分離（meetingbench の既存の測定。pseudo_meeting_asr.xlsx と同じ値）。",
        "『登録した声との照合』は、字幕の区間ごとに『登録した人の中でいちばん近い人』を選ぶ。表の数字は、登録発話の選び方を変えた5回の平均（範囲は最小〜最大）。",
        "『上限』は、字幕の区間はそのまま、話者だけ正解にした条件。1つの字幕に2人が混ざる区間があるため、どんな話者判定でもこれを超えられない（区間を切り直さない限り）。",
        "処理時間・RTF は『区間分け + 文字起こし + 話者埋め込み（照合は計算が一瞬）』。built の行は『区間分け + 文字起こし + 話者分離』。他のジョブと CPU を取り合った状態の参考値。",
    ]:
        r += 1
        ws.cell(r, 1, t)


def _rng(a: dict) -> str:
    return f"{a['spk_acc_min'] * 100:.1f}〜{a['spk_acc_max'] * 100:.1f}%" if a.get("spk_acc_min") is not None else "—"


def _verdict(kpis: dict[str, Kpi], a: dict) -> str:
    js = [kpis["spk_acc"].judge(a.get("spk_acc")), kpis["der"].judge(a.get("der_0"))]
    js = [j for j in js if j]
    names = ["話者正解率", "DER"]
    bad = [n for n, j in zip(names, js) if j == FAIL]
    if bad:
        return "未達（" + "・".join(bad) + "）"
    return "目標達成" if all(j == TARGET for j in js) else "最低ライン達成"


def _row(ws, r: int, vals: list, kpis: dict[str, Kpi], italic: bool = False) -> None:
    fmts = [None, None, None, "0", None, "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", None, None, "0.0", "0.000", None]
    for j, (v, f) in enumerate(zip(vals, fmts), start=1):
        _val(ws, r, j, v, f, wrap=j in (2, 3, 5, 18))
    _style(ws.cell(r, 6), kpis["spk_acc"], vals[5])
    _style(ws.cell(r, 7), kpis["der"], vals[6])
    _style(ws.cell(r, 10), kpis["spk_acc"], vals[9])
    _style(ws.cell(r, 11), kpis["der"], vals[10])
    v = vals[17] or ""
    vk = TARGET if v == "目標達成" else MINIMUM if v == "最低ライン達成" else FAIL if v.startswith("未達") else None
    if vk:
        ws.cell(r, 18).fill, ws.cell(r, 18).font = ar.KPI_STYLE[vk]
    if italic:
        for j in (1, 2, 3):
            ws.cell(r, j).font = Font(italic=True, color="595959")


# ---------------------------------------------------------------------------
def _all_metrics(ws, d: Data) -> None:
    ws.title = "全指標"
    collars = sorted({*DER_COLLARS, 0.0})
    cols = ar.summary_cols(collars, 0.0, "文字")
    keep = [c for c in cols if c.key not in ("engine", "wer", "verdict", "notes", "minutes")]
    extra_heads = ["名前どおりの話者正解率", "対応づけが名前どおり", "区間ごとの正解率", "不明にした区間数", "しきい値", "登録の長さ(秒/人・平均)", "登録発話の本数(平均)", "処理時間(1時間あたり・分)"]
    heads = ["録音", "区間の切り方", "埋め込み", "登録(秒/人)", "登録の感情", "照合方式", "登録発話の選び方(seed)", "不明のしきい値", "平滑化"] \
        + [c.header for c in keep] + extra_heads + ["KPI判定"]
    widths = [14, 22, 24, 9, 20, 18, 10, 12, 10] + [max(c.width, 10) for c in keep] + [12, 10, 10, 10, 9, 12, 10, 12] + [18]
    _hdr(ws, 1, heads, widths)
    order = sorted(d.results, key=lambda x: (x["spec"]["variant"], 0 if x["spec"]["seg"] == "silero" else 1, d.embedders.index(x["spec"]["embedder"]),
                                              x["spec"]["mode"], x["spec"]["rule"], x["spec"]["sec"], x["spec"]["seed"], str(x["spec"]["unk_pct"]), x["spec"]["smooth"]))
    r = 1
    for x in order:
        s, row, ex = x["spec"], x["row"], x["extra"]
        r += 1
        base = [VAR_JA[s["variant"]], SEG_JA[s["seg"]], emb_label(s["embedder"]), s["sec"], MODE_SHORT[s["mode"]], ea.RULE_JA[s["rule"]], s["seed"],
                (f"登録の本人点数の下位{s['unk_pct']:g}%" if s["unk_pct"] is not None else "使わない"),
                ({"groups": "続きの区間をまとめて判定", "short": "短い区間は前後を引き継ぐ"}.get(s["smooth"], "なし"))]
        for j, v in enumerate(base, start=1):
            _val(ws, r, j, v)
        for j, col in enumerate(keep, start=len(base) + 1):
            _val(ws, r, j, row.get(col.key), col.fmt)
        n0 = len(base) + len(keep) + 1
        import numpy as np

        vals = [ex.get("strict_acc"), "はい" if ex.get("spk_map_identity") and ex.get("der_map_identity") else "いいえ", ex.get("seg_acc"),
                ex.get("n_unknown"), ex.get("threshold"), float(np.mean(ex["enroll_sec"])), float(np.mean(ex["enroll_clips"])), row.get("minutes"), row.get("verdict")]
        fm = ["0.0%", None, "0.0%", "0", "0.000", "0", "0.0", "0.0", None]
        for j, (v, f) in enumerate(zip(vals, fm), start=n0):
            _val(ws, r, j, v, f)
    ws.freeze_panes = "J2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(heads))}{max(r, 2)}"


# ---------------------------------------------------------------------------
def _block(ws, r: int, d: Data, title: str, metric: str, g: str, rule: str, kpi_key: str | None) -> tuple[int, int, int]:
    """行=埋め込み×感情、列=録音A/Bの登録時間。戻り値: (次の行, 見出しの行, 最初のデータ行)。"""
    ws.cell(r, 1, title).font = ar.SECTION_FONT
    r += 1
    secs = d.secs
    heads = ["埋め込み・登録の感情"] + [f"A {s}秒" for s in secs] + [f"B {s}秒" for s in secs] + [f"範囲 A {s}秒" for s in secs] + [f"範囲 B {s}秒" for s in secs]
    for j, h in enumerate(heads, start=1):
        c = ws.cell(r, j, h)
        c.fill, c.font, c.border = ar.HEADER_FILL, ar.HEADER_FONT, ar.BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    head = r
    kpi = d.kpis.get(kpi_key) if kpi_key else None
    for e in d.embedders:
        for mode in ("mixed", "calm"):
            r += 1
            _val(ws, r, 1, f"{emb_label(e)} / {MODE_SHORT[mode]}")
            for i, v in enumerate(("A", "B")):
                for j, sec in enumerate(secs):
                    a = d.a(e, sec, mode, rule, v, g)
                    val = a[metric] if a else None
                    c = _val(ws, r, 2 + i * len(secs) + j, val, "0.0%")
                    if kpi:
                        _style(c, kpi, val)
                    lo, hi = (a.get(metric + "_min"), a.get(metric + "_max")) if a else (None, None)
                    _val(ws, r, 2 + (2 + i) * len(secs) + j, f"{lo * 100:.1f}〜{hi * 100:.1f}%" if lo is not None else "—")
    return r + 2, head, head + 1


def _compare(ws, d: Data) -> None:
    ws.title = "登録時間ごとの比較"
    ws["A1"] = "登録の長さ・感情・埋め込み器ごとの比較（登録発話の選び方5通りの平均。右側は最小〜最大）"
    ws["A1"].font = ar.TITLE_FONT
    ws["A2"] = "登録の長さは『1人あたり、この秒数以上になるまで発話を足した』もの（最後の1本で少し超える）。実際の長さは全指標シートの列を参照。"
    ws.column_dimensions["A"].width = 44
    for j in range(2, 2 + 4 * len(d.secs)):
        ws.column_dimensions[get_column_letter(j)].width = 13
    r = 4
    blocks = [
        ("① 話者正解率（発言単位）— 区間=pyVideoTrans 既定の区切り（silero）・方式=平均との類似度", "spk_acc", "silero", "mean", "spk_acc"),
        ("② 話者正解率（発言単位）— 区間=正解の区間（oracle）・方式=平均との類似度", "spk_acc", "oracle", "mean", "spk_acc"),
        ("③ 話者正解率（発言単位）— 区間=silero・方式=登録発話との類似度の最大", "spk_acc", "silero", "max", "spk_acc"),
        ("④ 話者正解率（発言単位）— 区間=oracle・方式=登録発話との類似度の最大", "spk_acc", "oracle", "max", "spk_acc"),
        ("⑤ DER（collar 0）— 区間=silero・方式=平均", "der_0", "silero", "mean", "der"),
        ("⑥ DER（collar 0.25）— 区間=silero・方式=平均", "der_0.25", "silero", "mean", None),
        ("⑦ cpCER — 区間=silero・方式=平均", "cpcer", "silero", "mean", None),
    ]
    first = None
    for title, metric, g, rule, kk in blocks:
        r, head, r0 = _block(ws, r, d, title, metric, g, rule, kk)
        if first is None:
            first = (head, r0)
    # 折れ線グラフ（① の A と B）
    if first:
        head, r0 = first
        n = len(d.embedders) * 2
        for i, v in enumerate(("A", "B")):
            ch = LineChart()
            ch.title = f"話者正解率（silero・平均との類似度）— 録音{v}"
            ch.y_axis.title = "話者正解率"
            ch.x_axis.title = "登録の長さ（秒/人）"
            ch.y_axis.number_format = "0%"
            ch.y_axis.scaling.min = 0.5
            ch.y_axis.scaling.max = 1.0
            ch.height, ch.width = 9, 17
            c0 = 2 + i * len(d.secs)
            data = Reference(ws, min_col=c0, max_col=c0 + len(d.secs) - 1, min_row=r0, max_row=r0 + n - 1)
            ch.add_data(data, from_rows=True, titles_from_data=False)
            for s_i, s in enumerate(ch.series):
                from openpyxl.chart.series import SeriesLabel
                from openpyxl.chart.data_source import StrRef

                s.tx = SeriesLabel(strRef=StrRef(f"'{ws.title}'!$A${r0 + s_i}"))
            ch.set_categories(Reference(ws, min_col=2, max_col=1 + len(d.secs), min_row=head))
            ch.x_axis.delete = False
            ch.y_axis.delete = False
            ws.add_chart(ch, f"{get_column_letter(2 + 4 * len(d.secs) + 1)}{4 + i * 20}")


# ---------------------------------------------------------------------------
def _confusion(wb: Workbook, engines: list) -> None:
    ws = wb.create_sheet()
    ar._write_confusion(ws, engines, EvalOptions())


def _unk_smooth(ws, d: Data) -> None:
    ws.title = "不明と平滑化"
    ws["A1"] = "『不明』のしきい値と、単純な平滑化の効果"
    ws["A1"].font = ar.TITLE_FONT
    ws["A2"] = ("しきい値は登録音声だけから決める（登録発話を1本ずつ抜いて、残りの登録と照合した『本人への点数』の下位X%）。会議の正解は使わない。"
                "不明にした区間は、アプリでは人が確認する区間になる想定。")
    ws["A2"].alignment = ar.WRAP
    ws.merge_cells("A2:L2")
    ws.row_dimensions[2].height = 34
    r = 4
    ws.cell(r, 1, "① 不明にしたときの数字（登録180秒・感情を混ぜる・平均との類似度・区間=silero。登録発話の選び方5通りの平均）").font = ar.SECTION_FONT
    r += 1
    heads = ["埋め込み", "録音", "しきい値(下位%)", "不明にした区間の割合", "受け入れた区間の正解率", "不明にした区間のうち本当は誤りだった割合",
             "誤りのうち不明で拾えた割合", "話者正解率(不明なし)", "話者正解率(不明を誤りと数える)", "DER collar 0 (不明なし)", "DER collar 0 (不明あり)"]
    for j, h in enumerate(heads, start=1):
        c = ws.cell(r, j, h)
        c.fill, c.font, c.border = ar.HEADER_FILL, ar.HEADER_FONT, ar.BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[get_column_letter(j)].width = 18 if j > 1 else 28
    ws.row_dimensions[r].height = 62
    for row in d.extras.get("reject", []):
        r += 1
        vals = [emb_label(row["embedder"]), VAR_JA[row["variant"]], row["pct"], row["rejected_pct"], row["accuracy_accepted"], row["wrong_among_rejected"],
                row["errors_caught_pct"], row["acc_plain"], row["acc_unk"], row["der_plain"], row["der_unk"]]
        for j, (v, f) in enumerate(zip(vals, [None, None, "0", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%"]), start=1):
            _val(ws, r, j, v, f)
    r += 3
    ws.cell(r, 1, "② 平滑化（登録180秒・感情を混ぜる・平均との類似度・区間=silero。5回の平均）").font = ar.SECTION_FONT
    r += 1
    heads2 = ["埋め込み", "平滑化", "録音", "話者正解率", "DER collar 0", "cpCER", "影響を受けた区間の数(目安)"]
    for j, h in enumerate(heads2, start=1):
        c = ws.cell(r, j, h)
        c.fill, c.font, c.border = ar.HEADER_FILL, ar.HEADER_FONT, ar.BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    for row in d.extras.get("smooth", []):
        r += 1
        vals = [emb_label(row["embedder"]), row["name"], VAR_JA[row["variant"]], row["spk_acc"], row["der_0"], row["cpcer"], row["note"]]
        for j, (v, f) in enumerate(zip(vals, [None, None, None, "0.0%", "0.0%", "0.0%", None]), start=1):
            _val(ws, r, j, v, f)
    r += 2
    for t in d.extras.get("smooth_notes", []):
        ws.cell(r, 1, "・" + t)
        r += 1
    win = d.extras.get("window", [])
    if win:
        r += 2
        ws.cell(r, 1, "③ 参考: 区間の中を窓で判定し、話者の交代で切り直した場合（登録180秒・感情を混ぜる・平均との類似度・区間=silero。5回の平均）").font = ar.SECTION_FONT
        r += 1
        ws.cell(r, 1, ("silero の区間ごとに 1.5 秒の窓（0.5 秒ずつ）で判定し、前後の多数決で整え、1 秒未満の部分は隣に吸収して、区間を話者ごとの部分に切り直した話者の時間割。"
                       "字幕に使うには、切り直した部分ごとに文字起こしをやり直す必要がある（ここでは文字は付けていないので cpCER は無い）。"
                       "重なった発話は1人にしか付けられない。"))
        r += 1
        heads3 = ["埋め込み", "録音", "話者正解率\n（区間ごとに1人）", "話者正解率\n（窓で切り直し）", "DER collar 0\n（区間ごとに1人）", "DER collar 0\n（窓で切り直し）",
                  "DER collar 0.25\n（区間ごとに1人）", "DER collar 0.25\n（窓で切り直し）", "字幕の区間数", "切り直し後の部分の数", "2つ以上に分かれた区間の数"]
        for j, h in enumerate(heads3, start=1):
            c = ws.cell(r, j, h)
            c.fill, c.font, c.border = ar.HEADER_FILL, ar.HEADER_FONT, ar.BORDER
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.row_dimensions[r].height = 48
        for row in win:
            r += 1
            vals = [emb_label(row["embedder"]), VAR_JA[row["variant"]], row["acc_base"], row["acc_win"], row["der0_base"], row["der0_win"], row["der25_base"],
                    row["der25_win"], row["segments"], row["pieces"], row["split_segments"]]
            for j, (v, f) in enumerate(zip(vals, [None, None, "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0.0%", "0", "0.0", "0.0"]), start=1):
                c = _val(ws, r, j, v, f)
            _style(ws.cell(r, 4), d.kpis["spk_acc"], row["acc_win"])
            _style(ws.cell(r, 6), d.kpis["der"], row["der0_win"])


def _explain(ws) -> None:
    ws.title = "説明"
    ws["A1"] = "このExcelの読み方"
    ws["A1"].font = ar.TITLE_FONT
    lines = [
        "目的: PLAN §4.1 ②『事前登録した声と照合して名前を付ける』が、教師なしの話者分離（built）より話者を当てられるかを測る。参加者は事前に約3分の声を登録する（D14）。",
        "方法: ① 登録用の発話（会議で使っていない JVNV の発話）を話者ごとに選ぶ ② 話者埋め込み（声の特徴を数百次元のベクトルにしたもの）を作る "
        "③ 字幕の区間ごとに埋め込みを作り、登録者との類似度（コサイン）がいちばん高い人の名前を付ける。同じ字幕の区間（silero / 正解）と同じ文字起こしを使う。",
        "登録の選び方: 『感情を混ぜる』は6種の感情（怒り・嫌悪・恐れ・喜び・悲しみ・驚き）を順番に回して足す。『落ち着いた感情だけ』は喜び・悲しみだけ（JVNV に平常はない）で、"
        "落ち着いた朗読で登録した場合の代わり。5回は、乱数で登録発話の選び方を変えたもの。",
        "照合の方式: 『平均との類似度』は登録発話の埋め込みの平均（重心）との類似度。『最大』は登録した各発話との類似度の最大。",
        "1. 『まとめ』: built（教師なし）と並べた主な結果と KPI の色。 2. 『全指標』: 条件×登録発話の選び方ごとの asrcheck の全列。フィルターで絞れる。",
        "3. 『登録時間ごとの比較』: 登録の長さ・感情・埋め込み器ごとの表とグラフ。 4. 『話者混同』: 誰を誰と取り違えたか。 5. 『不明と平滑化』: しきい値で不明にした効果と、単純な平滑化の効果。",
        "6. 『実行情報』: 使ったモデルとバージョン、ライセンス、JVNV の出典（CC BY-SA 4.0）、処理時間・RTF、再現コマンド。",
        "",
        "【指標】話者正解率=正解の発言ごとに、最も長く重なった認識側の話者が合っているかの割合（発言単位）。DER=話者分離誤り率（collar 0 と 0.25）。cpCER=話者ごとにつないだ文字列で測る誤り率。"
        "定義の詳細は meeting/src/asrcheck/README.md §6。asrcheck は話者名の対応づけを自動で最適化するので、『名前どおりの話者正解率』（全指標シート）で、名前を取り違えて有利になっていないことも確認している。",
        "【上限】字幕の区間のうち20〜30%は2人以上の声が混ざる（silero）。1つの字幕に1人の名前しか付けられないので、話者判定が完璧でも話者正解率は 95.3%（A）/ 90.9%（B）までしか上がらない。"
        "これを超えるには、字幕の区間を話者の交代で切り直す必要がある（本測定の対象外）。",
        "【注意】登録と会議が同じスタジオ録音・同じ処理で、実際の会議より有利。実会議で測り直すまでは上限に近い参考値と読む。",
    ]
    for i, t in enumerate(lines, start=2):
        ws.cell(i, 1, t).alignment = ar.WRAP
        ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=4)
        ws.row_dimensions[i].height = 48 if t else 12
    ws.column_dimensions["A"].width = 60
    for col in "BCD":
        ws.column_dimensions[col].width = 40


def write_enroll_workbook(path: Path | str, d: Data, info_items: list[tuple[str, Any]], recs: dict, confusion_engines: list) -> Path:
    wb = Workbook()
    _summary(wb.active, d, facts(d), recs)
    _all_metrics(wb.create_sheet(), d)
    _compare(wb.create_sheet(), d)
    _confusion(wb, confusion_engines)
    _unk_smooth(wb.create_sheet(), d)
    _explain(wb.create_sheet())
    ws = wb.create_sheet()
    ar._write_info(ws, info_items)
    ws.title = "実行情報"
    for w in wb.worksheets:
        ar._print_setup(w)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def base_info(command: str) -> list[tuple[str, Any]]:
    return [("ツール", "meetingbench enroll-report（採点は asrcheck）"), ("作成日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")), ("コマンド", command)]
