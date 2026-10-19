"""集計: 感情の正解率・混同表・再現率、文字誤り率 (CER)。Excel に書く前の『数字だけ』の部分。

感情
  - 正解率 (accuracy) = 予測が正解の感情と一致した件数 ÷ 全件数
  - 再現率 (recall)   = ある感情の音声のうち、その感情と当てられた割合 (感情ごとの正解率)
  - 平均再現率        = 感情ごとの再現率の平均 (件数が少ない感情も同じ重みで見る)
  - 信頼区間 (95%)    = Wilson の方法。件数が少ないと幅が広い (『この数字はこの範囲でぶれうる』)
  - 3 感情の見方      = 正解が 喜び・怒り・悲しみ の音声だけを取り出し、予測を 喜び/怒り/悲しみ/その他 の4つにまとめる
文字誤り率 (CER)
  - 1件ごとに『正解 → 認識結果』の編集距離 (asrcheck と同じ) を数え、グループ内の誤りの合計 ÷ 正解の文字数 で出す。
    (1件ごとの CER の平均ではない。長い文章ほど重みが大きい)
  - 正解と認識結果には asrcheck.textnorm の正規化を同じようにかける (全角半角・句読点・数字の書き方などの表記ゆれをそろえる)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from asrcheck import editdist
from asrcheck.editdist import ErrorCounts
from asrcheck.textnorm import NormOptions, normalize_text

from .dataset import Clip
from .labels import ALL_PRED_LABELS, CORE3, EMO_OTHER, EMOTIONS, JA, NOTAG, ORDER, OTHER, UNKNOWN, core3_view, fold_to_pvt, predicted_label
from .runners import ModelRun


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """割合 k/n の 95% 信頼区間 (Wilson)。n=0 なら (nan, nan)。"""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


# ---------------------------------------------------------------------------
# 感情
# ---------------------------------------------------------------------------
@dataclass
class EmotionRow:
    """ある感情 (または全体) の成績。"""
    label: str
    n: int
    correct: int
    top_wrong: str = ""          # いちばん多い間違い先 (日本語の文字列。無ければ '')
    pred_n: int = 0              # その感情と予測された件数 (正解がどの感情でも数える)。適合率の分母

    @property
    def recall(self) -> float:
        return rate(self.correct, self.n)

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.correct, self.n)

    @property
    def precision(self) -> float:
        """適合率 = その感情と予測した音声のうち、本当にその感情だった割合。何でも同じ感情と答えると再現率だけ高くなるのを見抜くため。"""
        return rate(self.correct, self.pred_n)


@dataclass
class EmotionStats:
    model: str
    key: str
    n: int
    correct: int
    true_labels: list[str]                       # 混同表の行 (データにある感情だけ、ORDER の順)
    pred_labels: list[str]                       # 混同表の列 (PRED_LABELS の順)
    matrix: dict[str, dict[str, int]]            # matrix[正解][予測] = 件数
    per_emotion: list[EmotionRow]
    by_speaker: list[EmotionRow] = field(default_factory=list)
    by_style: list[EmotionRow] = field(default_factory=list)
    speaker_emotion: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)   # (話者, 感情) → (正解数, 件数)
    pred_counts: Counter = field(default_factory=Counter)    # 予測ごとの件数 (全件)
    core3: dict[str, dict[str, int]] = field(default_factory=dict)   # 3 感情の見方: core3[正解][喜び/怒り/悲しみ/その他]
    core3_n: int = 0
    core3_correct: int = 0
    has_neutral_truth: bool = False              # 正解に『平常』が含まれるか (含まれなければ 4 分類の評価はできない)

    @property
    def accuracy(self) -> float:
        return rate(self.correct, self.n)

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.correct, self.n)

    @property
    def macro_recall(self) -> float:
        rs = [r.recall for r in self.per_emotion if r.n]
        return sum(rs) / len(rs) if rs else float("nan")

    @property
    def core3_accuracy(self) -> float:
        return rate(self.core3_correct, self.core3_n)

    @property
    def core3_ci(self) -> tuple[float, float]:
        return wilson(self.core3_correct, self.core3_n)

    def pred_rate(self, *labels: str) -> float:
        return rate(sum(self.pred_counts[x] for x in labels), self.n)

    @property
    def neutral_rate(self) -> float:
        """平常と判定された割合 (NEUTRAL タグだけ)。"""
        return self.pred_rate("neutral")

    @property
    def unknown_rate(self) -> float:
        return self.pred_rate(UNKNOWN)

    @property
    def notag_rate(self) -> float:
        return self.pred_rate(NOTAG)

    @property
    def folded_neutral_rate(self) -> float:
        """pyVideoTrans の map_sensevoice_tag と同じ読み替え (不明・タグなしも平常) をした場合の、平常の割合。"""
        return rate(sum(c for lab, c in self.pred_counts.items() if fold_to_pvt(lab) == "neutral"), self.n)

    @property
    def emotion_detected_rate(self) -> float:
        """平常・不明・タグなし 以外の感情タグが出た割合 (『何かしらの感情あり』と判定した割合)。"""
        return rate(sum(c for lab, c in self.pred_counts.items() if fold_to_pvt(lab) != "neutral"), self.n)


def _top_wrong(row: dict[str, int], true_label: str) -> str:
    wrong = [(c, lab) for lab, c in row.items() if lab != true_label and c > 0]
    if not wrong:
        return ""
    c, lab = max(wrong, key=lambda x: (x[0], -ALL_PRED_LABELS.index(x[1])))
    return f"{JA[lab]}（{c}件）"


def _rows_by(clips: list[Clip], preds: dict[str, str], key) -> list[EmotionRow]:
    groups: dict[str, list[Clip]] = {}
    for c in clips:
        groups.setdefault(key(c), []).append(c)
    out = []
    for g in sorted(groups):
        items = groups[g]
        out.append(EmotionRow(g, len(items), sum(1 for c in items if preds[c.file] == c.emotion)))
    return out


def emotion_stats(clips: list[Clip], run: ModelRun) -> EmotionStats:
    """clips のうち run が処理したものについて、感情の成績を集計する。"""
    done = [c for c in clips if c.file in run.results and not run.results[c.file].error]
    preds = {c.file: predicted_label(run.results[c.file].tag) for c in done}
    true_labels = [e for e in ORDER if any(c.emotion == e for c in done)]
    # 予測の列。emotion2vec+ の other は、実際に出たときだけ列を足す (SenseVoice の表は今までと同じ列のまま)
    pred_labels = [p for p in ALL_PRED_LABELS if p != EMO_OTHER or any(x == EMO_OTHER for x in preds.values())]
    matrix = {t: {p: 0 for p in pred_labels} for t in true_labels}
    for c in done:
        matrix[c.emotion][preds[c.file]] += 1
    per = []
    for t in true_labels:
        n = sum(matrix[t].values())
        per.append(EmotionRow(t, n, matrix[t][t], _top_wrong(matrix[t], t), sum(matrix[x][t] for x in true_labels)))
    se: dict[tuple[str, str], tuple[int, int]] = {}
    for c in done:
        k, n = se.get((c.speaker, c.emotion), (0, 0))
        se[(c.speaker, c.emotion)] = (k + (preds[c.file] == c.emotion), n + 1)
    core3: dict[str, dict[str, int]] = {}
    for t in CORE3:
        if t in true_labels:
            core3[t] = {x: 0 for x in (*CORE3, OTHER)}
    for c in done:
        if c.emotion in core3:
            core3[c.emotion][core3_view(preds[c.file])] += 1
    core3_n = sum(sum(r.values()) for r in core3.values())
    core3_ok = sum(core3[t][t] for t in core3)
    return EmotionStats(
        model=run.label, key=run.key, n=len(done), correct=sum(1 for c in done if preds[c.file] == c.emotion),
        true_labels=true_labels, pred_labels=pred_labels, matrix=matrix, per_emotion=per,
        by_speaker=_rows_by(done, preds, lambda c: c.speaker) if any(c.speaker for c in done) else [],
        by_style=_rows_by(done, preds, lambda c: c.style) if any(c.style for c in done) else [],
        speaker_emotion=se, pred_counts=Counter(preds.values()), core3=core3, core3_n=core3_n, core3_correct=core3_ok,
        has_neutral_truth="neutral" in true_labels,
    )


# ---------------------------------------------------------------------------
# 文字誤り率
# ---------------------------------------------------------------------------
def clip_counts(ref: str, hyp: str, opts: NormOptions) -> ErrorCounts:
    """1件の (置換・脱落・挿入) の数。正解が空 (整えた後) なら n_ref=0。"""
    r, h = list(normalize_text(ref, opts)), list(normalize_text(hyp, opts))
    return editdist.count_ops(editdist.align(r, h))


@dataclass
class CerRow:
    """あるグループ (全体・感情ごと・話者ごと) の文字誤り率。"""
    label: str
    n_clips: int
    counts: ErrorCounts

    @property
    def cer(self) -> float:
        return self.counts.rate


@dataclass
class CerTable:
    """1つのモデル × 1つの範囲 (全件、または他のモデルと共通の件) の結果。"""
    model: str
    scope: str
    n_clips: int
    overall: CerRow
    by_emotion: list[CerRow]
    by_speaker: list[CerRow]
    by_style: list[CerRow]
    raw_overall: CerRow | None = None            # 正規化をほとんどしない場合 (比較用)


def _group_cer(items: list[tuple[Clip, ErrorCounts]], key, order=None) -> list[CerRow]:
    groups: dict[str, list[ErrorCounts]] = {}
    for c, cnt in items:
        groups.setdefault(key(c), []).append(cnt)
    names = sorted(groups) if order is None else [g for g in order if g in groups]
    rows = []
    for g in names:
        tot = ErrorCounts()
        for cnt in groups[g]:
            tot += cnt
        rows.append(CerRow(g, len(groups[g]), tot))
    return rows


def cer_table(clips: list[Clip], run: ModelRun, opts: NormOptions, scope: str, only: set[str] | None = None) -> CerTable:
    """run の結果のうち、正解の文章がある件 (only があればその件だけ) で CER を集計する。"""
    use = [c for c in clips if c.file in run.results and not run.results[c.file].error and c.text.strip()
           and (only is None or c.file in only)]
    ro = opts.raw()
    items = [(c, clip_counts(c.text, run.results[c.file].text, opts)) for c in use]
    raw_total = ErrorCounts()
    for c in use:
        raw_total += clip_counts(c.text, run.results[c.file].text, ro)
    tot = ErrorCounts()
    for _, cnt in items:
        tot += cnt
    return CerTable(
        model=run.label, scope=scope, n_clips=len(use), overall=CerRow("全体", len(use), tot),
        by_emotion=_group_cer(items, lambda c: c.emotion, EMOTIONS),
        by_speaker=_group_cer(items, lambda c: c.speaker) if any(c.speaker for c in use) else [],
        by_style=_group_cer(items, lambda c: c.style) if any(c.style for c in use) else [],
        raw_overall=CerRow("全体（正規化なし）", len(use), raw_total),
    )


def cer_tables(clips: list[Clip], runs: list[ModelRun], opts: NormOptions) -> list[CerTable]:
    """各モデルの『処理した全件』の表。モデルごとに処理した件が違うときは、全モデル共通の件での表も足す。"""
    tables = []
    runs = [r for r in runs if r.report_cer]
    sets = [{c.file for c in clips if c.file in r.results and not r.results[c.file].error and c.text.strip()} for r in runs]
    common = set.intersection(*sets) if sets else set()
    differ = any(s != sets[0] for s in sets)
    for r, s in zip(runs, sets):
        tables.append(cer_table(clips, r, opts, f"処理した全件（{len(s)}件）"))
    if differ and len(runs) > 1:
        for r in runs:
            tables.append(cer_table(clips, r, opts, f"全モデル共通の件（{len(common)}件）", only=common))
    return [t for t in tables if t.n_clips]     # 正解の文章が無い (処理した件が 0) 表は作らない


def clip_cer(ref: str, hyp: str, opts: NormOptions) -> float | None:
    """1件の CER (正解が空なら None)。"""
    c = clip_counts(ref, hyp, opts)
    return c.rate if c.n_ref else None


def speed(run: ModelRun) -> dict[str, float]:
    """処理の速さ。rtf = 処理にかかった秒 ÷ 音声の長さ (1より小さければ実時間より速い)。モデルの読み込み時間は含めない。"""
    proc = sum(r.sec for r in run.results.values() if not r.error)
    audio = sum(run.durations.get(f, 0.0) for f, r in run.results.items() if not r.error)
    n = sum(1 for r in run.results.values() if not r.error)
    return {
        "n": n, "proc_sec": proc, "audio_sec": audio,
        "rtf": proc / audio if audio else float("nan"), "sec_per_clip": proc / n if n else float("nan"),
    }
