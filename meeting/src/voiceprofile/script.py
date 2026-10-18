"""朗読台本 (meeting/enroll/script_ja.json) の読み込み。

台本は「同意文」と「言語ごとの文のセット」でできている。ツールは録音の発話区間を、この文の並びに順番どおり対応づける。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .text_utils import count_mora, estimate_seconds, norm_lang

PLACEHOLDERS = ("name", "organization", "retention")


class ScriptError(ValueError):
    pass


@dataclass(frozen=True)
class Slot:
    """録音の中で読まれるはずの 1 文。"""

    id: str
    text: str
    lang: str
    est_sec: float            # 普通の速さで読んだときの長さの目安 (秒)
    role: str = "speech"      # "consent" (同意文: 手本にしない) / "speech"
    yomi: str = ""
    focus: str = ""

    @property
    def mora(self) -> int | None:
        return count_mora(self.yomi) if self.yomi else None


@dataclass
class Script:
    path: Path | None
    title: str
    consent: list[Slot]
    sets: dict[str, list[Slot]]
    labels: dict[str, str] = field(default_factory=dict)

    def languages(self) -> list[str]:
        return list(self.sets)

    def sentences(self, lang: str) -> list[Slot]:
        lang = norm_lang(lang)
        if lang not in self.sets:
            raise ScriptError(f"台本に言語 '{lang}' の文がありません (あるもの: {', '.join(self.sets)})")
        return self.sets[lang]

    def slots(self, langs: list[str], with_consent: bool = True) -> list[Slot]:
        """録音に入っているはずの文の並び (同意文 → 言語ごとの文の順)。"""
        out: list[Slot] = list(self.consent) if with_consent else []
        for lang in langs:
            out.extend(self.sentences(lang))
        return out

    def render_consent(self, name: str = "", organization: str = "", retention: str = "") -> str:
        return render_consent(" ".join(s.text for s in self.consent), name, organization, retention)


def render_consent(template: str, name: str = "", organization: str = "", retention: str = "") -> str:
    """同意文の {name} {organization} {retention} を埋める。空のものは ○○ にする。"""
    values = {"name": name or "○○", "organization": organization or "○○", "retention": retention or "○○"}
    for key, val in values.items():
        template = template.replace("{" + key + "}", val)
    return template


def total_est_sec(slots: list[Slot], pause: float = 0.8) -> float:
    return sum(s.est_sec for s in slots) + pause * len(slots)


def _slot(raw: dict, lang: str, role: str, where: str) -> Slot:
    if not isinstance(raw, dict) or not str(raw.get("id") or "").strip() or not str(raw.get("text") or "").strip():
        raise ScriptError(f"台本の {where} に id と text のない文があります: {raw!r}")
    text = str(raw["text"]).strip()
    yomi = str(raw.get("yomi") or "").strip()
    est = raw.get("est_sec")
    est_sec = float(est) if isinstance(est, (int, float)) and est > 0 else estimate_seconds(text, lang, yomi)
    return Slot(id=str(raw["id"]).strip(), text=text, lang=lang, est_sec=est_sec, role=role, yomi=yomi,
                focus=str(raw.get("focus") or ""))


def load_script(path: str | Path) -> Script:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"台本が見つかりません: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except ValueError as e:
        raise ScriptError(f"台本 {p.name} を JSON として読めません: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("sets"), dict) or not data["sets"]:
        raise ScriptError(f"台本 {p.name} に 'sets' (言語ごとの文) がありません")

    consent_raw = (data.get("consent") or {}).get("sentences") or []
    consent = [_slot(r, norm_lang((data.get("consent") or {}).get("lang") or "ja"), "consent", "consent") for r in consent_raw]
    sets: dict[str, list[Slot]] = {}
    labels: dict[str, str] = {}
    for lang_key, st in data["sets"].items():
        lang = norm_lang(lang_key)
        sents = (st or {}).get("sentences") if isinstance(st, dict) else None
        if not sents:
            raise ScriptError(f"台本の言語 '{lang_key}' に文がありません")
        sets[lang] = [_slot(r, lang, "speech", lang_key) for r in sents]
        labels[lang] = str(st.get("label") or lang)

    ids = [s.id for s in consent] + [s.id for ss in sets.values() for s in ss]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ScriptError(f"台本の id が重複しています: {', '.join(dup)}")
    return Script(path=p, title=str(data.get("title") or p.stem), consent=consent, sets=sets, labels=labels)
