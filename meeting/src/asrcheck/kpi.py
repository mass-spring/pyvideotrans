"""KPI (合格ライン)。PLAN §4.4 の初期案。ベースライン測定後に確定する値なので、--kpi で上書きできる。

| 指標                         | 最低ライン | 目標  |
| CER (表記ゆれを正規化した後)   | 15% 以下   | 8% 以下 |
| 話者正解率 (発言単位)          | 90% 以上   | 95% 以上 |
| DER                          | 20% 以下   | 15% 以下 |
| 処理時間 (1時間の会議, Mac)    | 60 分以内  | 30 分以内 |
| 完全オフラインで動作           | 必須       | 必須   |   ← プログラムでは判定できないので手で確認する
"""
from __future__ import annotations

from dataclasses import dataclass, replace

TARGET, MINIMUM, FAIL = "target", "min", "fail"
LABELS = {TARGET: "目標達成", MINIMUM: "最低ライン達成", FAIL: "未達"}


@dataclass(frozen=True)
class Kpi:
    key: str
    label: str
    minimum: float      # 最低ライン
    target: float       # 目標
    lower_is_better: bool
    unit: str           # "%" or "分"
    note: str = ""

    def judge(self, value: float | None) -> str | None:
        if value is None or value != value:
            return None
        if self.lower_is_better:
            return TARGET if value <= self.target else MINIMUM if value <= self.minimum else FAIL
        return TARGET if value >= self.target else MINIMUM if value >= self.minimum else FAIL

    def fmt(self, v: float) -> str:
        if self.unit == "%":
            return f"{v * 100:g}%"
        return f"{v:g}{self.unit}"

    @property
    def rule(self) -> tuple[str, str]:
        op = "以下" if self.lower_is_better else "以上"
        return f"{self.fmt(self.minimum)} {op}", f"{self.fmt(self.target)} {op}"


DEFAULT_KPIS: dict[str, Kpi] = {
    "cer": Kpi("cer", "文字誤り率 CER (表記ゆれを正規化した後)", 0.15, 0.08, True, "%", "日本語のみ。英語・フランス語の WER には目標を決めていません"),
    "spk_acc": Kpi("spk_acc", "話者正解率 (発言単位)", 0.90, 0.95, False, "%"),
    "der": Kpi("der", "話者分離誤り率 DER", 0.20, 0.15, True, "%"),
    "time": Kpi("time", "処理時間 (1時間の会議)", 60.0, 30.0, True, "分", "目標は Mac での値。別の機械で測った値は参考"),
}


def parse_kpi_overrides(spec: str | None, base: dict[str, Kpi] | None = None) -> dict[str, Kpi]:
    """'cer=0.15/0.08,der=0.2/0.15,time=60/30' → 上書きした KPI 辞書 (割合は 0〜1、時間は分)。"""
    kpis = dict(base or DEFAULT_KPIS)
    if not spec:
        return kpis
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, val = part.partition("=")
        key = key.strip().lower()
        if not sep or key not in kpis:
            raise ValueError(
                f"--kpi の書き方が違います: '{part}'\n  例: --kpi cer=0.15/0.08,spk_acc=0.9/0.95,der=0.2/0.15,time=60/30  (最低ライン/目標の順。cer・spk_acc・der は 0〜1 の割合、time は分)"
            )
        try:
            mn, tg = (float(x) for x in val.split("/"))
        except ValueError:
            raise ValueError(f"--kpi {key} の値は 最低ライン/目標 の形で数字を2つ書いてください: '{val}'") from None
        kpis[key] = replace(kpis[key], minimum=mn, target=tg)
    return kpis
