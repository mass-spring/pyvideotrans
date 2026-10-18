"""合成する文（targets.csv）の読み込みと、既定の文。

targets.csv の列:
  id       文の名前（重複なし）。例 m1-ja
  lang     言語（ja / en / fr / zh など）
  text     読み上げる文
  kind     meeting（会議風の文）か emotional（感情のある文）。省略時は meeting
  emotion  kind が emotional のとき、TTS に渡す感情の指示（英語、例 "angry, tense, slightly faster"）
Excel で作って「CSV UTF-8」で保存すればよい。見出しは日本語（言語・文・種類・感情）でも読める。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from .config import KIND_EMOTIONAL, KIND_MEETING, ConfigError

ALIASES = {
    "言語": "lang", "文": "text", "テキスト": "text", "種類": "kind", "感情": "emotion", "感情の指示": "emotion",
    "名前": "id", "番号": "id",
}


@dataclass(frozen=True)
class TargetText:
    id: str
    lang: str
    text: str
    kind: str = KIND_MEETING
    emotion: str = ""


# 会議風の文：同じ意味を 4 言語で（自分で作った文。公開データではない）
# 感情の文：怒り・喜び・悲しみ（PLAN 5.3 の 4 分類のうち平常以外）を 1 文ずつ。指示は VoxCPM2 の書き方（英語）。
DEFAULT_TARGETS: list[dict[str, str]] = [
    # --- 会議風 1：宿題の依頼
    {"id": "m1-ja", "lang": "ja", "text": "来週の水曜日までに、試験の結果をまとめて共有してください。"},
    {"id": "m1-en", "lang": "en", "text": "Please summarize the test results and share them by next Wednesday."},
    {"id": "m1-fr", "lang": "fr", "text": "Merci de résumer les résultats des essais et de les partager d'ici mercredi prochain."},
    {"id": "m1-zh", "lang": "zh", "text": "请在下周三之前整理试验结果并共享给大家。"},
    # --- 会議風 2：持ち帰り
    {"id": "m2-ja", "lang": "ja", "text": "この件については、一度持ち帰って、関係部署と相談してから回答します。"},
    {"id": "m2-en", "lang": "en", "text": "I will take this back, consult the relevant departments, and then give you an answer."},
    {"id": "m2-fr", "lang": "fr", "text": "Je vais en discuter avec les services concernés avant de vous donner une réponse."},
    {"id": "m2-zh", "lang": "zh", "text": "这件事我先带回去，和相关部门商量之后再答复您。"},
    # --- 会議風 3：目標値の決め方
    {"id": "m3-ja", "lang": "ja", "text": "目標値を決める前に、まず現状の測定値を確認しておく必要があります。"},
    {"id": "m3-en", "lang": "en", "text": "Before we set the target value, we first need to check the current measured values."},
    {"id": "m3-fr", "lang": "fr", "text": "Avant de fixer la valeur cible, il faut d'abord vérifier les valeurs mesurées actuelles."},
    {"id": "m3-zh", "lang": "zh", "text": "在确定目标值之前，我们首先需要确认当前的测量值。"},
    # --- 感情 1：怒り
    {"id": "e1-ja", "lang": "ja", "kind": "emotional", "emotion": "angry, tense, slightly faster", "text": "どうして期限を守れなかったのですか。これで三度目ですよ。"},
    {"id": "e1-en", "lang": "en", "kind": "emotional", "emotion": "angry, tense, slightly faster", "text": "Why did you miss the deadline again? This is the third time!"},
    {"id": "e1-fr", "lang": "fr", "kind": "emotional", "emotion": "angry, tense, slightly faster", "text": "Pourquoi n'avez-vous pas respecté le délai ? C'est la troisième fois !"},
    {"id": "e1-zh", "lang": "zh", "kind": "emotional", "emotion": "angry, tense, slightly faster", "text": "为什么又没有按时完成？这已经是第三次了！"},
    # --- 感情 2：喜び
    {"id": "e2-ja", "lang": "ja", "kind": "emotional", "emotion": "cheerful, bright, slightly faster", "text": "すばらしい結果です。皆さんのおかげで、目標を達成できました。"},
    {"id": "e2-en", "lang": "en", "kind": "emotional", "emotion": "cheerful, bright, slightly faster", "text": "That is wonderful news. Thanks to everyone, we achieved our goal!"},
    {"id": "e2-fr", "lang": "fr", "kind": "emotional", "emotion": "cheerful, bright, slightly faster", "text": "C'est une excellente nouvelle. Grâce à vous tous, nous avons atteint notre objectif !"},
    {"id": "e2-zh", "lang": "zh", "kind": "emotional", "emotion": "cheerful, bright, slightly faster", "text": "太棒了！多亏大家的努力，我们达成了目标！"},
    # --- 感情 3：悲しみ
    {"id": "e3-ja", "lang": "ja", "kind": "emotional", "emotion": "sad, subdued, slower", "text": "残念ですが、このプロジェクトは中止になりました。"},
    {"id": "e3-en", "lang": "en", "kind": "emotional", "emotion": "sad, subdued, slower", "text": "Unfortunately, this project has been cancelled."},
    {"id": "e3-fr", "lang": "fr", "kind": "emotional", "emotion": "sad, subdued, slower", "text": "Malheureusement, ce projet a été annulé."},
    {"id": "e3-zh", "lang": "zh", "kind": "emotional", "emotion": "sad, subdued, slower", "text": "很遗憾，这个项目被取消了。"},
]


def write_targets(path: str | Path, rows: list[dict[str, str]] | None = None) -> Path:
    """targets.csv を書く（Excel で開ける UTF-8 BOM 付き）。rows を省略すると既定の文。"""
    rows = rows if rows is not None else DEFAULT_TARGETS
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "lang", "kind", "emotion", "text"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"kind": KIND_MEETING, "emotion": "", **r})
    return p


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_targets(path: str | Path, langs: list[str] | None = None, max_per_lang: int | None = None) -> list[TargetText]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"targets.csv が見つかりません: {p}")
    reader = csv.reader(io.StringIO(_decode(p.read_bytes())))
    try:
        header = next(reader)
    except StopIteration:
        raise ConfigError(f"targets.csv が空です: {p}") from None
    cols = [ALIASES.get(h.strip(), h.strip()).lower() for h in header]
    missing = [c for c in ("id", "lang", "text") if c not in cols]
    if missing:
        raise ConfigError(
            f"targets.csv に必要な列がありません: {', '.join(missing)}\n"
            f"  見つかった列: {', '.join(h.strip() for h in header)}\n  必須: id, lang, text（任意: kind, emotion）"
        )
    idx = {c: cols.index(c) for c in ("id", "lang", "text", "kind", "emotion") if c in cols}

    def cell(rec: list[str], name: str) -> str:
        i = idx.get(name)
        return rec[i].strip() if i is not None and i < len(rec) else ""

    out: list[TargetText] = []
    seen: set[str] = set()
    for n, rec in enumerate(reader, start=2):
        if not any(c.strip() for c in rec):
            continue
        t = TargetText(
            id=cell(rec, "id"),
            lang=cell(rec, "lang").lower(),
            text=cell(rec, "text"),
            kind=(cell(rec, "kind") or KIND_MEETING).lower(),
            emotion=cell(rec, "emotion"),
        )
        if not t.id or not t.lang or not t.text:
            raise ConfigError(f"targets.csv の {n} 行目：id・lang・text のどれかが空です")
        if t.kind not in (KIND_MEETING, KIND_EMOTIONAL):
            raise ConfigError(f"targets.csv の {n} 行目：kind は meeting か emotional にしてください（'{t.kind}'）")
        if t.kind == KIND_EMOTIONAL and not t.emotion:
            raise ConfigError(f"targets.csv の {n} 行目：emotional の文には emotion（感情の指示）が必要です")
        if t.id in seen:
            raise ConfigError(f"targets.csv の id が重複しています: {t.id}")
        seen.add(t.id)
        out.append(t)
    if langs:
        out = [t for t in out if t.lang in langs]
    if max_per_lang:
        counts: dict[tuple[str, str], int] = {}
        kept = []
        for t in out:
            k = (t.lang, t.kind)
            counts[k] = counts.get(k, 0) + 1
            if counts[k] <= max_per_lang:
                kept.append(t)
        out = kept
    if not out:
        raise ConfigError(f"targets.csv に、使う言語（{', '.join(langs or [])}）の文がありません")
    return out
