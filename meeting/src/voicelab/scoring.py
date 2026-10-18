"""測定フェーズ：pairs.csv を作り、voicecheck compare で測り、Excel に「条件比較」などのシートを足す。

- ref（本人の基準）＝その話者の「評価用の実声」のうち、**TTS の手本に使っていないもの**（plan.score_refs）。
- test ＝生成した wav。label ＝ "エンジン|条件|言語"。
- 実声どうしの上限：評価用の実声を 1 本ずつ test にして、残りを ref にした行（label = real|heldout_loo|言語）も入れる。
- voicecheck の Excel（行ごと・話者ごと・説明・実行情報）に、集計のシートを足す。
"""
from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .conditions import condition_sort_key
from .config import (
    COND_LINE,
    COND_PROFILE,
    EMO_INSTRUCT,
    EMO_REFONLY,
    EMO_SIMILARITY,
    ExperimentConfig,
)
from .runner import MANIFEST, STATUS_FAILED, STATUS_OK, STATUS_SKIPPED, prepare_speakers, read_manifest
from .speakers import SpeakerPlan

PAIRS_CSV = "pairs.csv"
RESULT_XLSX = "voicelab_result.xlsx"
REAL_ENGINE, REAL_COND = "real", "heldout_loo"

# voicecheck の Excel（行ごと）の見出し。voicecheck/report.py の定義と同じ文字列
H_LABEL = "ラベル"
H_SPEAKER = "話者"
FEATURE_HEADERS = {
    "f0_st": "声の高さの差（半音）",
    "f0_iqr_ratio": "声の高さの幅の比",
    "rate_ratio": "話速の比",
    "loud_diff": "音量の差（dB）",
    "tilt_diff": "傾きの差（dB/oct）",
}
RE_SIM = re.compile(r"^SIM（(.+)）$")
RE_JUDGE = re.compile(r"^判定（(.+)）$")


class PairsLeakError(RuntimeError):
    """評価の ref に、TTS の手本に使ったファイルが混ざった（あってはならない）。"""


def say(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# pairs.csv
# ---------------------------------------------------------------------------
def _norm(p: str) -> str:
    return str(Path(p).resolve())


def build_pairs(
    cfg: ExperimentConfig, plans: list[SpeakerPlan], manifest: dict[str, dict], out_dir: Path
) -> list[dict[str, str]]:
    by_name = {p.name: p for p in plans}
    rows: list[dict[str, str]] = []
    for r in manifest.values():
        if r.get("status") != STATUS_OK:
            continue
        plan = by_name.get(r["speaker"])
        wav = out_dir / r["out_wav"]
        if plan is None or not wav.is_file():
            continue
        refs = [c.path for c in plan.score_refs]
        rows.append({
            "speaker": plan.name, "ref": ";".join(refs), "test": r["out_wav"],
            "label": f"{r['engine_label']}|{r['condition']}|{r['lang']}",
            "text_id": r["text_id"], "repeat": r["repeat"], "engine": r["engine_label"],
            "condition": r["condition"], "lang": r["lang"],
        })
    for plan in plans:
        score = [c.path for c in plan.score_refs]
        for clip in plan.ceiling_clips:
            others = [p for p in score if p != clip.path]
            rows.append({
                "speaker": plan.name, "ref": ";".join(others), "test": clip.path,
                "label": f"{REAL_ENGINE}|{REAL_COND}|{clip.lang or '?'}",
                "text_id": Path(clip.path).stem, "repeat": "0", "engine": REAL_ENGINE,
                "condition": REAL_COND, "lang": clip.lang or "?",
            })
    check_no_leak(rows, plans)
    return rows


def check_no_leak(rows: list[dict[str, str]], plans: list[SpeakerPlan]) -> None:
    """評価の ref に、TTS の手本に使ったファイルが入っていないことを確かめる（入っていたら止める）。"""
    forbidden = {p.name: {_norm(x) for x in p.tts_ref_paths} for p in plans}
    for r in rows:
        refs = {_norm(x) for x in r["ref"].split(";") if x}
        bad = refs & forbidden.get(r["speaker"], set())
        if bad:
            raise PairsLeakError(f"評価の ref に TTS の手本が混ざっています（{r['speaker']}）: {sorted(bad)}")
        if r["engine"] != REAL_ENGINE and _norm(r["test"]) in refs:
            raise PairsLeakError(f"評価の ref に評価音声そのものが入っています: {r['test']}")
        if not refs:
            raise PairsLeakError(f"評価の ref が空です（{r['speaker']}）")
        if any(";" in x for x in r["ref"].split(";") if x):  # pragma: no cover - split 済みなので起きないが念のため
            raise PairsLeakError("パスに ; が含まれています")


def write_pairs_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    cols = ["speaker", "ref", "test", "label", "text_id", "repeat", "engine", "condition", "lang"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------------------
# voicecheck の Excel を読んで集計
# ---------------------------------------------------------------------------
@dataclass
class ScoreRow:
    speaker: str
    engine: str
    cond: str
    lang: str
    sims: dict[str, float | None]
    judge: dict[str, bool | None]
    feats: dict[str, float | None]


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def read_score_rows(xlsx: Path) -> tuple[list[ScoreRow], list[str], dict[str, bool]]:
    """voicecheck の「行ごと」シートを読む。(行, 埋め込み名の一覧, 判定列があるか) を返す。"""
    from openpyxl import load_workbook

    wb = load_workbook(xlsx, data_only=True, read_only=False)
    ws = wb["行ごと"]
    headers = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(headers) if h}
    if H_LABEL not in col:
        raise RuntimeError("voicecheck の Excel に「ラベル」列がありません（voicecheck の版が違う可能性）")
    names = [m.group(1) for h in headers if h and (m := RE_SIM.match(str(h)))]
    judge_cols = {m.group(1): i for i, h in enumerate(headers) if h and (m := RE_JUDGE.match(str(h)))}
    rows: list[ScoreRow] = []
    for rec in ws.iter_rows(min_row=2, values_only=True):
        label = rec[col[H_LABEL]]
        if not label:
            continue
        parts = str(label).split("|")
        if len(parts) != 3:
            continue
        sims = {n: _num(rec[col[f"SIM（{n}）"]]) for n in names}
        judge: dict[str, bool | None] = {}
        for n, i in judge_cols.items():
            v = rec[i]
            judge[n] = True if v == "合格" else (False if v == "不合格" else None)
        feats = {k: (_num(rec[col[h]]) if h in col else None) for k, h in FEATURE_HEADERS.items()}
        rows.append(ScoreRow(str(rec[col[H_SPEAKER]]), parts[0], parts[1], parts[2], sims, judge, feats))
    wb.close()
    return rows, names, {n: True for n in judge_cols}


def _mean(xs: list[float]) -> float | None:
    return float(statistics.fmean(xs)) if xs else None


def _std(xs: list[float]) -> float | None:
    return float(statistics.stdev(xs)) if len(xs) >= 2 else None


def group_values(rows: list[ScoreRow], getter: Callable[[ScoreRow], float | None]) -> dict[tuple[str, str], dict[str, list[float]]]:
    """{(engine, cond): {lang: [値...]}}"""
    out: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        v = getter(r)
        if v is None:
            continue
        out.setdefault((r.engine, r.cond), {}).setdefault(r.lang, []).append(v)
    return out


def baseline_of(cond: str) -> str | None:
    """その条件の「効果」を見るときの比べる相手（design.md 6章：1回に1つだけ変える）。"""
    if cond == COND_LINE or cond == EMO_SIMILARITY or cond == REAL_COND:
        return None
    if cond.startswith("stretch_x"):
        return COND_PROFILE
    if cond.startswith(COND_PROFILE):
        return COND_LINE
    if cond in (EMO_INSTRUCT, EMO_REFONLY):
        return EMO_SIMILARITY
    return None


def ordered_keys(rows: list[ScoreRow], engine_order: list[str]) -> list[tuple[str, str]]:
    keys = {(r.engine, r.cond) for r in rows}

    def ekey(e: str) -> tuple:
        if e == REAL_ENGINE:
            return (-1, 0, e)
        names = [x for x in engine_order]
        return (0, names.index(e) if e in names else 99, e)

    return sorted(keys, key=lambda k: (ekey(k[0]), condition_sort_key(k[1])))


def langs_in(rows: list[ScoreRow], preferred: list[str]) -> list[str]:
    present = {r.lang for r in rows}
    return [x for x in preferred if x in present] + sorted(present - set(preferred))


# ---------------------------------------------------------------------------
# Excel のシート
# ---------------------------------------------------------------------------
def _styles():
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    thin = Side(style="thin", color="BFBFBF")
    return {
        "hfill": PatternFill("solid", fgColor="1F4E78"),
        "hfont": Font(bold=True, color="FFFFFF"),
        "title": Font(bold=True, size=13),
        "bold": Font(bold=True),
        "note": Font(italic=True, color="595959"),
        "warn": PatternFill("solid", fgColor="FFF2CC"),
        "border": Border(left=thin, right=thin, top=thin, bottom=thin),
        "wrap": Alignment(wrap_text=True, vertical="top"),
        "real": PatternFill("solid", fgColor="E2EFDA"),
    }


def _put_block(ws, top: int, title: str, header: list[str], body: list[list[Any]], fmt: str = "0.000",
               note: str = "", value_cols: tuple[int, int] | None = None, diverging: bool = False,
               first_data_cols: int = 2, shade_rows: set[int] | None = None) -> int:
    """表を1つ書く。次に書ける行番号を返す。value_cols = 色を付ける列の範囲（1始まり、両端含む）。"""
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter

    st = _styles()
    ws.cell(top, 1, title).font = st["title"]
    r = top + 1
    if note:
        ws.cell(r, 1, note).font = st["note"]
        r += 1
    for j, h in enumerate(header, start=1):
        c = ws.cell(r, j, h)
        c.fill, c.font, c.border = st["hfill"], st["hfont"], st["border"]
    first = r + 1
    for i, rec in enumerate(body):
        for j, v in enumerate(rec, start=1):
            c = ws.cell(first + i, j, v)
            c.border = st["border"]
            if j > first_data_cols and isinstance(v, float):
                c.number_format = fmt
            if shade_rows and i in shade_rows:
                c.fill = st["real"]
    last = first + len(body) - 1
    if value_cols and body:
        a, b = get_column_letter(value_cols[0]), get_column_letter(value_cols[1])
        rng = f"{a}{first}:{b}{last}"
        if diverging:
            rule = ColorScaleRule(start_type="min", start_color="F8696B", mid_type="num", mid_value=0,
                                  mid_color="FFFFFF", end_type="max", end_color="63BE7B")
        else:
            rule = ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50,
                                  mid_color="FFEB84", end_type="max", end_color="63BE7B")
        ws.conditional_formatting.add(rng, rule)
    return last + 3


def _engine_cond_table(rows: list[ScoreRow], keys, langs, getter, agg) -> list[list[Any]]:
    gv = group_values(rows, getter)
    body = []
    for (e, c) in keys:
        per = gv.get((e, c), {})
        allv = [x for xs in per.values() for x in xs]
        vals = [agg(per.get(lg, [])) for lg in langs]
        body.append([e, c, *vals, agg(allv), len(allv)])
    return body


def _diff_table(rows: list[ScoreRow], keys, langs, name: str) -> tuple[list[list[Any]], list[str]]:
    gv = group_values(rows, lambda r: r.sims.get(name))
    body = []
    for (e, c) in keys:
        base = baseline_of(c)
        if base is None:
            continue
        bper = gv.get((e, base))
        per = gv.get((e, c))
        if not bper or not per:
            continue
        diffs = []
        for lg in langs:
            a, b = _mean(per.get(lg, [])), _mean(bper.get(lg, []))
            diffs.append(None if a is None or b is None else a - b)
        a_all = _mean([x for xs in per.values() for x in xs])
        b_all = _mean([x for xs in bper.values() for x in xs])
        body.append([e, c, base, *diffs, None if a_all is None or b_all is None else a_all - b_all])
    return body, ["エンジン", "条件", "比べる相手", *langs, "全言語"]


def add_summary_sheets(xlsx: Path, cfg: ExperimentConfig, manifest: dict[str, dict], dry_run: bool,
                       extra_info: list[tuple[str, Any]] | None = None) -> None:
    from openpyxl import load_workbook

    rows, names, judge_names = read_score_rows(xlsx)
    wb = load_workbook(xlsx)
    st = _styles()
    engine_order = list(dict.fromkeys(m["engine_label"] for m in manifest.values()))  # 設定の並び順（ドライランは dry: 付き）
    keys = ordered_keys(rows, engine_order)
    langs = langs_in(rows, cfg.target_langs)
    banner = "※ これは --dry-run のダミー音声の結果です。数値に意味はありません（流れの確認用）。" if dry_run else ""

    # ---- 条件比較 -----------------------------------------------------------
    ws = wb.create_sheet("条件比較")
    top = 1
    if banner:
        c = ws.cell(top, 1, banner)
        c.font, c.fill = st["bold"], st["warn"]
        top += 2
    if not names:
        ws.cell(top, 1, "SIM（話者類似度）が出ていません。埋め込み（ecapa など）を読み込めなかった可能性があります。"
                "「特徴量の差」シートと voicecheck の「実行情報」を見てください。").font = st["bold"]
        top += 2
    for name in names:
        header = ["エンジン", "条件", *langs, "全言語", "行数"]
        body = _engine_cond_table(rows, keys, langs, lambda r, n=name: r.sims.get(n), _mean)
        real_idx = {i for i, b in enumerate(body) if b[0] == REAL_ENGINE}
        nl = len(langs)
        top = _put_block(
            ws, top, f"SIM の平均（{name}）＝ エンジン × 条件 × 言語",
            header, body, "0.000",
            note="数字が大きいほど本人に似ている。緑＝高い／赤＝低い。real は本人の実声どうし（上限の目安）。行数＝平均に使った音声の数（話者・文・繰り返しの合計）。",
            value_cols=(3, 3 + nl), shade_rows=real_idx,
        )
        sd = _engine_cond_table(rows, keys, langs, lambda r, n=name: r.sims.get(n), _std)
        top = _put_block(
            ws, top, f"SIM のばらつき（{name}・標準偏差）",
            header, sd, "0.000",
            note="同じ条件のなかでの揺れ（繰り返し・文・話者の違いを含む）。差を見るときは、この揺れより大きい差かどうかを確認する。",
        )
        dbody, dheader = _diff_table(rows, keys, langs, name)
        if dbody:
            top = _put_block(
                ws, top, f"条件を変えたときの SIM の差（{name}）＝ その条件の平均 − 比べる相手の平均",
                dheader, dbody, "+0.000;-0.000;0.000",
                note="プラス＝その条件のほうが似ている。profile_ref は line_ref_16k と、stretch は profile_ref と、emotion_instruct/refonly は emotion_similarity と比べる（1回に1つだけ変えた比較）。",
                value_cols=(4, 4 + nl), diverging=True, first_data_cols=3,
            )
        if name in judge_names:
            pr = {}
            for r in rows:
                j = r.judge.get(name)
                if j is not None:
                    pr.setdefault((r.engine, r.cond), {}).setdefault(r.lang, []).append(1.0 if j else 0.0)
            pbody = []
            for (e, c) in keys:
                per = pr.get((e, c), {})
                allv = [x for xs in per.values() for x in xs]
                pbody.append([e, c, *[_mean(per.get(lg, [])) for lg in langs], _mean(allv), len(allv)])
            top = _put_block(ws, top, f"合格率（{name}・しきい値以上の割合）", header, pbody, "0.0%",
                             note="voicecheck に渡したしきい値（--calib / --threshold）以上だった音声の割合。",
                             value_cols=(3, 3 + nl))
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 18
    for col in "DEFGHIJ":
        ws.column_dimensions[col].width = 11
    ws.freeze_panes = "C1"

    # ---- 現状値 -------------------------------------------------------------
    wc = wb.create_sheet("現状値")
    top = 1
    if banner:
        c = wc.cell(top, 1, banner)
        c.font, c.fill = st["bold"], st["warn"]
        top += 2
    wc.cell(top, 1, "現状値（V2）＝ 何も改善しない今の方式で、どれだけ本人に似ているか").font = st["title"]
    wc.cell(top + 1, 1, "C0 line_ref_16k ＝ 上流 pyVideoTrans と同じ手本（その場の短い 1 行・16 kHz）。この行の数字が現状値。"
                         "real の行は本人の実声どうし（これ以上は期待できない上限の目安）。C1 は登録手本に替えた参考値。").font = st["note"]
    top += 3
    for name in names:
        cur = [(e, c) for (e, c) in keys if c in (COND_LINE, COND_PROFILE, REAL_COND)]
        body = _engine_cond_table(rows, cur, langs, lambda r, n=name: r.sims.get(n), _mean)
        real_idx = {i for i, b in enumerate(body) if b[0] == REAL_ENGINE}
        top = _put_block(wc, top, f"SIM の平均（{name}）", ["エンジン", "条件", *langs, "全言語", "行数"], body, "0.000",
                         value_cols=(3, 3 + len(langs)), shade_rows=real_idx)
    wc.column_dimensions["A"].width = 30
    wc.column_dimensions["B"].width = 22

    # ---- 特徴量の差 ---------------------------------------------------------
    wf = wb.create_sheet("特徴量の差")
    top = 1
    if banner:
        c = wf.cell(top, 1, banner)
        c.font, c.fill = st["bold"], st["warn"]
        top += 2
    wf.cell(top, 1, "声の特徴の差（評価用の実声の平均との差。エンジン × 条件ごとの平均）").font = st["title"]
    wf.cell(top + 1, 1, "早回し（stretch）では話速の比が倍率に近づくはず。声の高さの差は 0 に近いほど良い（半音）。").font = st["note"]
    top += 3
    hdr = ["エンジン", "条件", "声の高さの差 絶対値の平均（半音）", "声の高さの幅の比", "話速の比", "音量の差（dB）", "傾きの差（dB/oct）", "行数"]
    body = []
    for (e, c) in keys:
        sub = [r for r in rows if r.engine == e and r.cond == c]
        f0 = [abs(r.feats["f0_st"]) for r in sub if r.feats.get("f0_st") is not None]
        body.append([
            e, c, _mean(f0),
            _mean([r.feats["f0_iqr_ratio"] for r in sub if r.feats.get("f0_iqr_ratio") is not None]),
            _mean([r.feats["rate_ratio"] for r in sub if r.feats.get("rate_ratio") is not None]),
            _mean([r.feats["loud_diff"] for r in sub if r.feats.get("loud_diff") is not None]),
            _mean([r.feats["tilt_diff"] for r in sub if r.feats.get("tilt_diff") is not None]),
            len(sub),
        ])
    _put_block(wf, top, "特徴量の差", hdr, body, "0.00")
    wf.column_dimensions["A"].width = 30
    wf.column_dimensions["B"].width = 22
    for col in "CDEFGH":
        wf.column_dimensions[col].width = 18

    # ---- 実験の説明 ---------------------------------------------------------
    we = wb.create_sheet("実験の説明")
    we.column_dimensions["A"].width = 24
    we.column_dimensions["B"].width = 70
    we.column_dimensions["C"].width = 60
    r = 1
    we.cell(r, 1, "この Excel の読み方").font = st["title"]
    r += 2
    for h, j in zip(["条件", "何をしたか", "何を知るための比較か"], (1, 2, 3)):
        c = we.cell(r, j, h)
        c.fill, c.font = st["hfill"], st["hfont"]
    for cond, what, why in EXPLAIN_ROWS:
        r += 1
        for j, v in enumerate((cond, what, why), start=1):
            c = we.cell(r, j, v)
            c.alignment = st["wrap"]
    r += 2
    for line in READ_GUIDE:
        we.cell(r, 1, line).alignment = Alignment_wrap()
        we.merge_cells(start_row=r, start_column=1, end_row=r, end_column=3)
        we.row_dimensions[r].height = 32
        r += 1

    # ---- 実験設定 -----------------------------------------------------------
    wi = wb.create_sheet("実験設定")
    wi.column_dimensions["A"].width = 28
    wi.column_dimensions["B"].width = 90
    ok = sum(1 for m in manifest.values() if m["status"] == STATUS_OK)
    ng = [m for m in manifest.values() if m["status"] == STATUS_FAILED]
    sk = sum(1 for m in manifest.values() if m["status"] == STATUS_SKIPPED)
    info: list[tuple[str, Any]] = [
        ("実験名", cfg.name), ("作成日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("ドライラン（ダミー音声）", "はい（数値に意味はありません）" if dry_run else "いいえ"),
        ("話者", ", ".join(s.name for s in cfg.speakers)), ("目標言語", ", ".join(cfg.target_langs)),
        ("エンジン", ", ".join(cfg.engines)), ("感情の条件のエンジン", ", ".join(cfg.emotion_engines)),
        ("条件", ", ".join(cfg.conditions)), ("繰り返し", cfg.repeats), ("乱数の種", cfg.seed),
        ("早回しの倍率", ", ".join(f"{f:g}" for f in cfg.stretch_factors)), ("早回しの方式", cfg.stretch_method),
        ("C0 の手本の長さ", f"{cfg.line_ref_min_s:g}〜{cfg.line_ref_max_s:g} 秒"),
        ("C1 の手本の選び方", f"{cfg.profile_ref_select}（{cfg.profile_ref_min_s:g}〜{cfg.profile_ref_max_s:g} 秒）"),
        ("生成に成功", ok), ("失敗", len(ng)), ("対応外で飛ばした", sk),
        ("測定の埋め込み", ", ".join(names) or "(なし)"),
    ]
    modes = sorted({(m["engine"], m["condition"], m["mode"]) for m in manifest.values() if m["status"] == STATUS_OK and m.get("mode")})
    for e, c, mode in modes:
        info.append((f"実際の方式：{e} / {c}", mode))
    for m in ng[:30]:
        info.append((f"失敗：{m['engine']} / {m['condition']} / {m['text_id']}", m["error"][:200]))
    if len(ng) > 30:
        info.append(("失敗（以降省略）", f"ほか {len(ng) - 30} 件。manifest.csv を見てください"))
    for k, v in (extra_info or []):
        info.append((k, v))
    for i, (k, v) in enumerate(info, start=1):
        wi.cell(i, 1, k).font = st["bold"]
        c = wi.cell(i, 2, v if isinstance(v, (int, float)) else str(v))
        c.alignment = st["wrap"]

    # ---- 並べ替え：まとめを前に --------------------------------------------
    front = ["現状値", "条件比較", "特徴量の差", "実験の説明"]
    for i, n in enumerate(front):
        wb.move_sheet(n, offset=-(wb.sheetnames.index(n) - i))
    wb.save(xlsx)


def Alignment_wrap():
    from openpyxl.styles import Alignment

    return Alignment(wrap_text=True, vertical="top")


EXPLAIN_ROWS = [
    ("real（実声どうし）", "評価用の実声を1本ずつ取り出し、残りの実声と比べた SIM。", "本人の実声どうしでも SIM は 1 にならない。これが達成できる上限の目安。"),
    ("line_ref_16k（C0）", "手本＝評価用の実声から取った 2〜4 秒の 1 本を 16 kHz にしたもの。上流 pyVideoTrans が「その場の字幕1行」を手本にするのを再現。",
     "【現状値】今の方式で、どれだけ似るか。"),
    ("profile_ref（C1）", "手本＝登録した良い手本（10〜15 秒・元のサンプルレート・書き起こし付き）に替えた。他は C0 と同じ。", "手本を替えるだけで、どれだけ良くなるか（改善策①）。"),
    ("stretch_x1.3 / x2.0（C2）", "C1 で作った音声を、声の高さを変えずに 1.3 倍・2.0 倍の速さにした（rubberband があれば rubberband、無ければ librosa）。",
     "長さ合わせの早回しで、声がどれだけ変わるか（改善策④の根拠）。"),
    ("emotion_similarity（C3）", "VoxCPM2：手本＋書き起こしで複製（感情の指示なし）。似かた優先（D13 の similarity）。", "感情の文で、書き起こし付き複製がどれだけ似るか。"),
    ("emotion_refonly（C3・対照）", "VoxCPM2：手本だけで複製（書き起こしなし・指示なし）。", "書き起こしを使うことの効果（similarity との差）を切り分ける。"),
    ("emotion_instruct（C3）", "VoxCPM2：手本＋感情の指示（書き起こしは使えない）。感情優先（D13 の emotion）。",
     "感情の指示を入れると、似かたがどれだけ下がるか（refonly との差）＝ D13 の判断材料。"),
]

READ_GUIDE = [
    "1. まず「現状値」シート：line_ref_16k の行が現状値です。real の行（実声どうし）が上限の目安です。",
    "2. 「条件比較」シート：緑が高い（似ている）、赤が低い。「条件を変えたときの SIM の差」の表で、改善の効果（プラスなら改善）を見ます。",
    "3. 差が「ばらつき（標準偏差）」より小さいときは、偶然の揺れかもしれません。繰り返しを増やすか、文を増やして確かめます。",
    "4. 数字は埋め込み（ecapa など）ごとに水準が違います。合格ラインは voicecheck calibrate で自分たちの登録データから決めます。",
    "5. 声の高さ・話速など個別の差は「特徴量の差」シート。1 行ごとの詳細は「行ごと」シート（voicecheck が作成）。",
]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run_score(
    cfg: ExperimentConfig,
    out_dir: str | Path,
    *,
    embedder: str | None = None,
    calib: str | None = None,
    threshold: str | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = say,
) -> Path:
    out = Path(out_dir)
    manifest = read_manifest(out)
    if not manifest:
        raise RuntimeError(f"{out / MANIFEST} がありません。先に generate（または run）を実行してください")
    dry_run = dry_run or any(m.get("mode") == "dry_run" for m in manifest.values())  # 生成がドライランなら、測定もそう扱う
    prepared = prepare_speakers(cfg, out, write=False)
    plans = [p.plan for p in prepared]
    rows = build_pairs(cfg, plans, manifest, out)
    gen_rows = [r for r in rows if r["engine"] != REAL_ENGINE]
    if not gen_rows:
        raise RuntimeError("測る音声がありません（生成に成功した音声が 0 件です）。manifest.csv の error 列を確認してください")
    pairs = write_pairs_csv(out / PAIRS_CSV, rows)
    log(f"pairs.csv：{len(rows)} 行（生成 {len(gen_rows)} / 実声の上限 {len(rows) - len(gen_rows)}）→ {pairs}")

    try:
        from voicecheck.cli import main as vc_main
    except ImportError as e:
        raise RuntimeError(
            f"voicecheck を import できません（{e}）。meeting フォルダで `pip install -e .`、"
            "または PYTHONPATH に meeting/src を足してください"
        ) from e
    xlsx = out / RESULT_XLSX
    emb = embedder or cfg.embedder or "ecapa"
    args = ["compare", "--pairs", str(pairs), "--out", str(xlsx), "--embedder", emb]
    cal = calib or cfg.calib
    thr = threshold or cfg.threshold
    if cal:
        args += ["--calib", str(Path(cal).expanduser())]
    if thr:
        args += ["--threshold", thr]
    log("voicecheck compare を実行します（音声の数によっては時間がかかります）…")
    rc = vc_main(args)
    if rc != 0:
        raise RuntimeError(f"voicecheck compare が失敗しました（終了コード {rc}）。上のメッセージを確認してください")
    add_summary_sheets(xlsx, cfg, manifest, dry_run, [("voicecheck の引数", " ".join(args))])
    log(f"Excel を作りました：{xlsx.resolve()}")
    return xlsx
