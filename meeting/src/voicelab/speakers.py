"""話者の登録音声（手本）と、評価用の実声（held-out）を読み込み、役割を決める。

入力は 2 通り（設定の speakers: で選ぶ）:

(A) profile: voiceprofile ツール（src/voiceprofile/profile_io.py）が作ったフォルダ
      <フォルダ>/profile.json   {"name": "Aさん", "refs": [{"wav": "refs/ref_ja_01.wav", "text": "...", "lang": "ja", "sec": 14.2, "snr_db": 32.1, ...}],
                                 "eval": {"index": "eval/eval.json", "n": 5}, ...}
      <フォルダ>/eval/eval.json {"name": "Aさん", "clips": [{"wav": "eval/eval_ja_01.wav", "text": "...", "lang": "ja", "sec": 6.1, ...}]}
      eval が評価用に取り置いた実声（手本に使っていない）。wav のパスは profile.json のあるフォルダ基準。
      ※ 手書きの profile.json で `"eval": [{"wav": ...}]`（一覧を直接書く）や、`eval_clips` / `heldout` という名前も受け付ける。
        項目が無ければ <フォルダ>/eval/（または heldout/）の wav を使う。

(B) folder: wav を入れただけのフォルダ
      <フォルダ>/*.wav と書き起こし（transcripts.csv の列: file,text,lang,role、または wav と同名の .txt）。
      role 列に ref / eval を書けばその通りに分ける。無ければ「書き起こしのある 6 秒以上の wav から、10〜15 秒に近い順に
      folder_ref_count 本」を手本にし、残りを評価用にする。

役割（plan_speaker が決める）:
  profile_refs  C1 の手本（登録した良い手本）
  line_refs     C0 の手本の元（評価用の実声から 2〜4 秒に近いものを取り出す。TTS の手本に使うので評価には使わない）
  score_refs    voicecheck の ref（評価用の実声。**TTS の手本に使ったものは絶対に入れない**）
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from .audio import duration_of
from .config import ExperimentConfig, SpeakerSpec, resolve_path

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
EVAL_KEYS = ("eval", "eval_clips", "heldout", "held_out", "holdout")
EVAL_DIRS = ("eval", "heldout", "held_out", "holdout")


class SpeakerError(ValueError):
    """話者データの不足・形式違い（メッセージは利用者向けの日本語）。"""


@dataclass
class Clip:
    path: str                   # 絶対パス
    text: str = ""
    lang: str = ""
    duration: float = 0.0
    sr: int = 0
    snr_db: float | None = None
    best: bool = False          # 登録側が「これが一番良い」と印を付けているとき
    score: float | None = None  # voiceprofile が付けた手本の点数（高いほど良い）


@dataclass
class SpeakerData:
    name: str
    profile_refs: list[Clip]
    eval_clips: list[Clip]
    source: str


@dataclass
class SpeakerPlan:
    name: str
    profile_refs: list[Clip]    # 使う手本（best なら 1 本、each なら全部）
    line_refs: list[Clip]
    score_refs: list[Clip]
    ceiling_clips: list[Clip]
    notes: list[str] = field(default_factory=list)

    @property
    def tts_ref_paths(self) -> set[str]:
        """TTS の手本に使う元ファイル（評価の ref に混ざってはいけないもの）。"""
        return {c.path for c in self.profile_refs} | {c.path for c in self.line_refs}


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def _clip(path: Path, text: str = "", lang: str = "", snr: float | None = None, best: bool = False,
          score: float | None = None) -> Clip:
    if not path.is_file():
        raise SpeakerError(f"音声ファイルが見つかりません: {path}")
    dur, sr = duration_of(path)
    return Clip(str(path.resolve()), (text or "").strip(), (lang or "").strip().lower(), float(dur), int(sr),
                None if snr is None else float(snr), bool(best), None if score is None else float(score))


def _read_text_sidecar(wav: Path) -> str:
    t = wav.with_suffix(".txt")
    if t.is_file():
        try:
            return t.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            return t.read_text(encoding="cp932", errors="replace").strip()
    return ""


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _entry_to_clip(base: Path, e) -> Clip:
    if isinstance(e, str):
        p = (base / e)
        return _clip(p, _read_text_sidecar(p))
    if not isinstance(e, dict):
        raise SpeakerError(f"profile.json の音声の項目は文字列か辞書にしてください: {e!r}")
    w = e.get("wav") or e.get("path") or e.get("file")
    if not w:
        raise SpeakerError(f"profile.json の項目に wav がありません: {e!r}")
    p = base / w
    text = e.get("text") or e.get("transcript") or _read_text_sidecar(p)
    return _clip(p, text, e.get("lang", ""), e.get("snr_db"), bool(e.get("best", False)), e.get("score"))


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _load_eval_clips(base: Path, data: dict) -> list[Clip]:
    """評価用の実声を探す。voiceprofile の形（profile.json の eval.index → eval/eval.json の clips）を最優先にする。"""
    entries = None
    for k in EVAL_KEYS:
        v = data.get(k)
        if isinstance(v, list) and v:
            entries = v
            break
        if isinstance(v, dict):
            idx = v.get("index")
            j = _read_json(base / idx) if idx else None
            if isinstance(j, dict) and j.get("clips"):
                entries = j["clips"]
                break
            if isinstance(v.get("clips"), list) and v["clips"]:
                entries = v["clips"]
                break
    if entries is None:
        j = _read_json(base / "eval" / "eval.json")
        if isinstance(j, dict) and j.get("clips"):
            entries = j["clips"]
    if entries is not None:
        return [_entry_to_clip(base, e) for e in entries]
    for d in EVAL_DIRS:
        if (base / d).is_dir():
            return [_clip(p, _read_text_sidecar(p)) for p in sorted((base / d).iterdir())
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return []


def _load_profile(spec: SpeakerSpec, cfg: ExperimentConfig) -> SpeakerData:
    pdir = resolve_path(cfg, spec.profile)
    pj = pdir if pdir.is_file() else pdir / "profile.json"
    if not pj.is_file():
        raise SpeakerError(f"profile.json が見つかりません: {pj}（voiceprofile で作ったフォルダを指定してください）")
    base = pj.parent
    try:
        data = json.loads(pj.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise SpeakerError(f"profile.json の書き方が違います（{pj}）: {e}") from None
    refs = [_entry_to_clip(base, e) for e in data.get("refs", [])]
    if not refs:
        raise SpeakerError(f"{pj} に refs（手本）がありません")
    evals = _load_eval_clips(base, data)
    if not evals:
        raise SpeakerError(
            f"{spec.name}：評価用の実声（held-out）がありません。voiceprofile build で eval を作るか、{base}/eval/ に wav を入れてください。"
            "手本に使っていない本人の実声がないと、似ているかを測れません"
        )
    name = spec.name or data.get("name") or base.name
    return SpeakerData(name, refs, evals, f"profile:{pj}")


def _load_folder(spec: SpeakerSpec, cfg: ExperimentConfig) -> SpeakerData:
    d = resolve_path(cfg, spec.folder)
    if not d.is_dir():
        raise SpeakerError(f"フォルダが見つかりません: {d}")
    meta: dict[str, dict[str, str]] = {}
    csv_path = resolve_path(cfg, spec.transcripts) if spec.transcripts else d / "transcripts.csv"
    if csv_path.is_file():
        rows = list(csv.DictReader(io.StringIO(_decode(csv_path.read_bytes()))))
        for r in rows:
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            f = r.get("file") or r.get("wav") or r.get("filename") or r.get("ファイル")
            if f:
                meta[Path(f).name] = r
    elif spec.transcripts:
        raise SpeakerError(f"書き起こしの CSV が見つかりません: {csv_path}")
    wavs = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not wavs:
        raise SpeakerError(f"音声ファイルがありません: {d}")
    clips: list[tuple[Clip, str]] = []
    for p in wavs:
        m = meta.get(p.name, {})
        text = m.get("text") or m.get("書き起こし") or _read_text_sidecar(p)
        clips.append((_clip(p, text, m.get("lang", "")), (m.get("role") or "").lower()))

    refs = [c for c, role in clips if role == "ref"]
    evals = [c for c, role in clips if role == "eval"]
    rest = [c for c, role in clips if role not in ("ref", "eval")]
    if rest:
        mid = (cfg.profile_ref_min_s + cfg.profile_ref_max_s) / 2
        cand = sorted((c for c in rest if c.text and c.duration >= 6.0), key=lambda c: abs(c.duration - mid))
        if not cand:  # 書き起こしが無い → 長い順（書き起こしなしでも動くエンジンで測れるように）
            cand = sorted(rest, key=lambda c: -c.duration)
        n = max(1, cfg.folder_ref_count) if not refs else 0
        picked = cand[:n]
        refs += picked
        evals += [c for c in rest if c not in picked]
    if not refs:
        raise SpeakerError(f"{spec.name}：手本（ref）にできる音声がありません")
    return SpeakerData(spec.name, refs, evals, f"folder:{d}")


def load_speaker(spec: SpeakerSpec, cfg: ExperimentConfig) -> SpeakerData:
    return _load_profile(spec, cfg) if spec.profile else _load_folder(spec, cfg)


# ---------------------------------------------------------------------------
# 役割の決定
# ---------------------------------------------------------------------------
def pick_profile_refs(refs: list[Clip], min_s: float, max_s: float, mode: str) -> list[Clip]:
    """C1 の手本を選ぶ。印（best）→ 書き起こしあり → 長さが範囲内 → voiceprofile の点数 → SNR → 範囲の中央に近い、の順。"""
    mid = (min_s + max_s) / 2

    def key(c: Clip):
        in_range = min_s <= c.duration <= max_s
        snr = -c.snr_db if c.snr_db is not None else float("inf")
        score = -c.score if c.score is not None else float("inf")
        return (0 if c.best else 1, 0 if c.text else 1, 0 if in_range else 1, score, snr, abs(c.duration - mid))

    ranked = sorted(refs, key=key)
    return ranked if mode == "each" else ranked[:1]


def pick_line_refs(evals: list[Clip], min_s: float, max_s: float, count: int) -> list[Clip]:
    """C0 の手本の元を、評価用の実声から選ぶ（2〜4 秒に近く、書き起こしがあるものを優先）。"""
    mid = (min_s + max_s) / 2

    def key(c: Clip):
        in_range = min_s <= c.duration <= max_s
        return (0 if in_range else 1, 0 if c.text else 1, abs(c.duration - mid), c.path)

    return sorted(evals, key=key)[: max(1, count)]


def plan_speaker(data: SpeakerData, cfg: ExperimentConfig) -> SpeakerPlan:
    notes: list[str] = []
    cand_refs = data.profile_refs
    if cfg.ref_lang:
        cand_refs = [c for c in data.profile_refs if c.lang == cfg.ref_lang]
        if not cand_refs:
            raise SpeakerError(
                f"{data.name}：ref_lang={cfg.ref_lang} の手本がありません（ある言語: "
                f"{', '.join(sorted({c.lang or '?' for c in data.profile_refs}))}）"
            )
    refs = pick_profile_refs(cand_refs, cfg.profile_ref_min_s, cfg.profile_ref_max_s, cfg.profile_ref_select)
    # 評価用の実声から、手本と同じファイルを念のため外す（登録と評価の取り違え対策）
    ref_paths = {c.path for c in data.profile_refs}
    evals = [c for c in data.eval_clips if c.path not in ref_paths]
    if len(evals) != len(data.eval_clips):
        notes.append("評価用の実声のうち、手本と同じファイルが混ざっていたので外しました")
    need = cfg.line_refs_per_speaker + 1
    if len(evals) < need:
        raise SpeakerError(
            f"{data.name}：評価用の実声が足りません（{len(evals)} 本。C0 の手本用に {cfg.line_refs_per_speaker} 本 + 評価用に最低 1 本、"
            f"合計 {need} 本以上が必要です。目安は 4 本以上）"
        )
    line_pool = [c for c in evals if c.lang == cfg.ref_lang] if cfg.ref_lang else evals
    if len(line_pool) < cfg.line_refs_per_speaker:
        line_pool = evals
        if cfg.ref_lang:
            notes.append(f"評価用の実声に {cfg.ref_lang} が足りないので、C0 の手本は別の言語の実声から選びました")
    lines = pick_line_refs(line_pool, cfg.line_ref_min_s, cfg.line_ref_max_s, cfg.line_refs_per_speaker)
    line_paths = {c.path for c in lines}
    score = [c for c in evals if c.path not in line_paths]
    if len(score) == 1:
        notes.append("評価用の実声が 1 本だけです。SIM が不安定になるので、2 本以上（目安 4 本以上）にすると安心です")
    ceiling = score[: cfg.ceiling_max_clips] if (cfg.real_ceiling and len(score) >= 2) else []
    if cfg.real_ceiling and not ceiling:
        notes.append("評価用の実声が 1 本なので、実声どうしの上限（real）は出せません")
    if not any(c.text for c in refs):
        notes.append("手本に書き起こしがありません。書き起こしを使うエンジンは、手本だけの方式に切り替わります")
    in_range = [c for c in refs if cfg.profile_ref_min_s <= c.duration <= cfg.profile_ref_max_s]
    if not in_range:
        notes.append(
            f"手本の長さが {cfg.profile_ref_min_s:g}〜{cfg.profile_ref_max_s:g} 秒の範囲外です（{refs[0].duration:.1f} 秒）"
        )
    return SpeakerPlan(data.name, refs, lines, score, ceiling, notes)
