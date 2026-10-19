"""モデル比較の集計: PLAN §5.3 の『4分類（平常・喜び・怒り・悲しみ）の正解率』と、その周辺の数字。

測る対象（正解の感情で決まる。データの名前には依存しない）
  4分類      正解が 平常・喜び・怒り・悲しみ の音声（JVNV の喜び・怒り・悲しみ + FLEURS の平常）
  6感情      正解が 喜び・怒り・悲しみ・恐れ・嫌悪・驚き の音声（JVNV 全体。平常は無い）
  7感情      上の両方（平常 + 6感情）
  平常の音声  正解が 平常 の音声（FLEURS）。『感情を付けてしまわないか』を見る

予測の読み方（見方 view）
  strict  モデルの出力そのまま。4つ以外（恐れ・嫌悪・驚き・その他・不明・タグなし）が出たら不正解
  fold    不明・タグなし・その他（emotion2vec+ の other）を平常に読み替える。pyVideoTrans の map_sensevoice_tag と同じ方式
  pick    対象の感情だけを候補にして、スコアが最大のものを選ぶ（emotion2vec+ のようにスコアを出すモデルだけ）
          『4分類にしぼれば、どこまで当たるか』の見方。実際の組み込みでも、この選び方はできる

正解率の注意: 4分類の件数は 平常 と 喜び・怒り・悲しみ で大きく違う（FLEURS は数百件、JVNV の3感情は約800件）ので、
正解率（件数で重みを付けた値）とは別に『平均再現率』（4つの感情を同じ重みで平均した値）も出す。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .analysis import CerTable, cer_table, rate, wilson
from .dataset import Clip
from .labels import ALL_PRED_LABELS, EMO_OTHER, EMOTIONS, FOUR, NOTAG, ORDER, SIX, UNKNOWN, fold_to_pvt, predicted_label
from .runners import ClipResult, ModelRun

SEVEN = ("neutral", *SIX)
VIEW_STRICT, VIEW_FOLD, VIEW_PICK = "strict", "fold", "pick"
VIEW_JA = {
    VIEW_STRICT: "そのまま",
    VIEW_FOLD: "不明・その他を平常に読み替え",
    VIEW_PICK: "対象の中から最大を選ぶ",
}
VIEW_HELP = {
    VIEW_STRICT: "モデルの出力どおり。対象の感情以外（恐れ・嫌悪・驚き・その他・不明・タグなし）が出たら不正解",
    VIEW_FOLD: "不明・タグなし・その他（other）を『平常』に読み替える。pyVideoTrans の今の作り（map_sensevoice_tag）と同じ方式",
    VIEW_PICK: "対象の感情だけを候補にして、スコアが最大のものを選ぶ（スコアを出す emotion2vec+ だけ）。4分類なら 平常・喜び・怒り・悲しみ の4つから選ぶ",
}
NO_EMOTION = ("neutral", EMO_OTHER, UNKNOWN, NOTAG)         # 『感情は付かなかった』と見なす予測


def predict(res: ClipResult, view: str, allowed: tuple[str, ...] | None = None) -> str | None:
    """1件の結果 → 予測の感情（見方 view に従う）。pick でスコアが無ければ None。"""
    if view == VIEW_PICK:
        if not res.scores or not allowed:
            return None
        return max(((res.scores.get(k, 0.0), -i, k) for i, k in enumerate(allowed)))[2]     # 同点なら allowed の前のほう
    lab = predicted_label(res.tag)
    return fold_to_pvt(lab) if view == VIEW_FOLD else lab


@dataclass
class ClassRow:
    label: str
    n: int               # 正解がこの感情の件数
    correct: int
    pred_n: int = 0      # この感情と予測された件数（正解がどの感情でも。ただし評価の対象にした件の中で）
    top_wrong: str = ""  # いちばん多い間違い先 (予測の感情名, 件数)
    top_wrong_n: int = 0

    @property
    def recall(self) -> float:
        return rate(self.correct, self.n)

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.correct, self.n)

    @property
    def precision(self) -> float:
        return rate(self.correct, self.pred_n)


@dataclass
class ViewEval:
    """あるモデル × ある対象（4分類など）× ある見方 の成績。"""
    model: str
    key: str
    view: str
    truth: tuple[str, ...]                       # 対象の感情
    n: int
    correct: int
    pred_labels: list[str]                       # 混同表の列
    matrix: dict[str, dict[str, int]]            # matrix[正解][予測]
    rows: list[ClassRow]
    pred_counts: Counter = field(default_factory=Counter)
    datasets: dict[str, int] = field(default_factory=dict)     # データ名ごとの件数

    @property
    def accuracy(self) -> float:
        return rate(self.correct, self.n)

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.correct, self.n)

    @property
    def balanced(self) -> float:
        """平均再現率（対象の感情を同じ重みで平均）。"""
        rs = [r.recall for r in self.rows if r.n]
        return sum(rs) / len(rs) if rs else float("nan")

    def row(self, label: str) -> ClassRow | None:
        return next((r for r in self.rows if r.label == label), None)

    def top_pred(self) -> tuple[str, int]:
        """いちばん多く出した予測とその件数。"""
        if not self.pred_counts:
            return "", 0
        order = {x: i for i, x in enumerate(ALL_PRED_LABELS)}
        lab, c = max(self.pred_counts.items(), key=lambda kv: (kv[1], -order.get(kv[0], 99)))
        return lab, c


def evaluate(clips: list[Clip], run: ModelRun, view: str, truth: tuple[str, ...], allowed: tuple[str, ...] | None = None) -> ViewEval | None:
    """clips のうち、正解が truth の感情で、run が処理できた（失敗していない）ものを、見方 view で集計する。

    見方 pick は allowed（既定 truth）から選ぶ。使えない（スコアなし・対象の件なし）なら None。
    """
    allowed = allowed or truth
    done = [c for c in clips if c.emotion in truth and c.file in run.results and not run.results[c.file].error]
    if not done:
        return None
    preds: dict[str, str] = {}
    for c in done:
        p = predict(run.results[c.file], view, allowed)
        if p is None:
            return None
        preds[c.file] = p
    if view == VIEW_PICK:
        cols = [e for e in ORDER if e in allowed]
    elif view == VIEW_FOLD:
        cols = list(ORDER)
    else:
        cols = [p for p in (*ORDER, EMO_OTHER, UNKNOWN, NOTAG) if p != EMO_OTHER or any(x == EMO_OTHER for x in preds.values())]
    true_labels = [e for e in ORDER if e in truth and any(c.emotion == e for c in done)]
    matrix = {t: {p: 0 for p in cols} for t in true_labels}
    for c in done:
        matrix[c.emotion][preds[c.file]] += 1
    rows: list[ClassRow] = []
    for t in true_labels:
        n = sum(matrix[t].values())
        wrong = [(cnt, lab) for lab, cnt in matrix[t].items() if lab != t and cnt > 0]
        tw = max(wrong, key=lambda x: (x[0], -cols.index(x[1]))) if wrong else (0, "")
        rows.append(ClassRow(t, n, matrix[t][t], sum(matrix[x][t] for x in true_labels), tw[1], tw[0]))
    return ViewEval(
        model=run.label, key=run.key, view=view, truth=tuple(true_labels), n=len(done),
        correct=sum(1 for c in done if preds[c.file] == c.emotion), pred_labels=cols, matrix=matrix, rows=rows,
        pred_counts=Counter(preds.values()), datasets=dict(Counter(c.dataset or "(指定なし)" for c in done)),
    )


@dataclass
class NeutralEval:
    """正解が『平常』の音声に、モデルが何を付けたか。"""
    model: str
    key: str
    n: int
    counts: Counter                       # 予測（strict）ごとの件数
    pick4_neutral: int | None = None      # 4つ（平常・喜び・怒り・悲しみ）から最大を選んだとき、平常になった件数（スコアがあるモデルのみ）

    def share(self, *labels: str) -> float:
        return rate(sum(self.counts[x] for x in labels), self.n)

    @property
    def neutral_rate(self) -> float:
        """平常と判定された割合。"""
        return self.share("neutral")

    @property
    def no_emotion_rate(self) -> float:
        """感情が付かなかった割合（平常・その他・不明・タグなし）。平常の音声を『感情あり』と間違えなかった率。"""
        return self.share(*NO_EMOTION)

    @property
    def emotion_rate(self) -> float:
        """平常の音声に、何かの感情（喜び・怒り・悲しみ・恐れ・嫌悪・驚き）を付けた割合。"""
        return self.share(*(e for e in EMOTIONS if e != "neutral"))

    @property
    def pick4_rate(self) -> float | None:
        return None if self.pick4_neutral is None else rate(self.pick4_neutral, self.n)

    def ci(self, *labels: str) -> tuple[float, float]:
        """labels のどれかが付いた割合の 95% 信頼区間。"""
        return wilson(sum(self.counts[x] for x in labels), self.n)


def neutral_eval(clips: list[Clip], run: ModelRun) -> NeutralEval | None:
    done = [c for c in clips if c.emotion == "neutral" and c.file in run.results and not run.results[c.file].error]
    if not done:
        return None
    counts = Counter(predicted_label(run.results[c.file].tag) for c in done)
    pick = None
    if all(run.results[c.file].scores for c in done):
        pick = sum(1 for c in done if predict(run.results[c.file], VIEW_PICK, FOUR) == "neutral")
    return NeutralEval(run.label, run.key, len(done), counts, pick)


def speed_for(run: ModelRun, clips: list[Clip]) -> dict[str, float]:
    """clips の範囲での処理速度。rtf = 処理にかかった秒 ÷ 音声の長さ（モデルの読み込みは含めない）。"""
    use = [c for c in clips if c.file in run.results and not run.results[c.file].error]
    proc = sum(run.results[c.file].sec for c in use)
    audio = sum(run.durations.get(c.file, 0.0) for c in use)
    return {"n": len(use), "proc_sec": proc, "audio_sec": audio,
            "rtf": proc / audio if audio else float("nan"), "sec_per_clip": proc / len(use) if use else float("nan")}


@dataclass
class ModelCompare:
    run: ModelRun
    four: dict[str, ViewEval]          # 見方 → 4分類の成績
    six: dict[str, ViewEval]           # 見方 → 6感情の成績（正解が平常でない音声）
    seven: dict[str, ViewEval]         # 見方 → 7感情の成績（平常があるときだけ）
    neutral: NeutralEval | None
    speed_by_dataset: dict[str, dict[str, float]]
    cer_by_dataset: dict[str, CerTable] = field(default_factory=dict)


def views_for(run: ModelRun) -> tuple[str, ...]:
    return (VIEW_STRICT, VIEW_FOLD, VIEW_PICK) if run.has_scores else (VIEW_STRICT, VIEW_FOLD)


def dataset_names(clips: list[Clip]) -> list[str]:
    """データ名（出てきた順）。名前が無い音声は '(指定なし)'。"""
    out: list[str] = []
    for c in clips:
        n = c.dataset or "(指定なし)"
        if n not in out:
            out.append(n)
    return out


def compare_models(clips: list[Clip], runs: list[ModelRun], opts=None) -> list[ModelCompare]:
    out: list[ModelCompare] = []
    names = dataset_names(clips)
    for run in runs:
        if not run.has_emotion:
            continue
        four, six, seven = {}, {}, {}
        for v in views_for(run):
            for dst, truth in ((four, FOUR), (six, SIX), (seven, SEVEN)):
                e = evaluate(clips, run, v, truth)
                if e is not None:
                    dst[v] = e
        # 4分類は、平常の音声と、平常以外（喜び・怒り・悲しみのどれか）の音声の両方が無いと測れない。7感情も同じ
        labels4 = {r.label for e in four.values() for r in e.rows}
        if "neutral" not in labels4 or not labels4 & {"happy", "angry", "sad"}:
            four, seven = {}, {}
        spd = {n: speed_for(run, [c for c in clips if (c.dataset or "(指定なし)") == n]) for n in names}
        cers: dict[str, CerTable] = {}
        if opts is not None and run.report_cer:
            for n in names:
                files = {c.file for c in clips if (c.dataset or "(指定なし)") == n}
                t = cer_table(clips, run, opts, f"{n}", only=files)
                if t.n_clips:
                    cers[n] = t
        out.append(ModelCompare(run, four, six, seven, neutral_eval(clips, run), spd, cers))
    return out
