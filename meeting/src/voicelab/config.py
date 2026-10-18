"""実験の設定（exp.yaml / exp.json）の読み込みと検査。

初級者が書き間違えやすいので、
- 知らない項目名（タイポ）はエラーにして、使える項目名を表示する
- 相対パスは「設定ファイルのあるフォルダ」基準で解決する
YAML は pyyaml が入っているときだけ読める（無ければ JSON を使う）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .engines import EngineError, parse_engine_spec, validate_engine_spec  # noqa: F401 (parse_engine_spec は再公開)

# 条件の名前（design.md 6章の「1回に1つだけ変える」に対応）
COND_LINE = "line_ref_16k"     # C0: 上流 pyVideoTrans と同じ（短い・16kHz・その場の1行）
COND_PROFILE = "profile_ref"   # C1: 登録した良い手本
COND_STRETCH = "stretch"       # C2: C1 の出力を早回し（実際の条件名は stretch_x1.3 など）
COND_EMOTION = "emotion"       # C3: VoxCPM2 の感情指示 vs 書き起こし付き複製
ALLOWED_CONDITIONS = [COND_LINE, COND_PROFILE, COND_STRETCH, COND_EMOTION]

# C3 の中身（emotion_modes）
EMO_INSTRUCT = "emotion_instruct"        # 手本＋感情の指示（書き起こしは使わない）
EMO_REFONLY = "emotion_refonly"          # 手本だけ・指示なし（書き起こしも使わない）＝対照
EMO_SIMILARITY = "emotion_similarity"    # 手本＋書き起こし付き複製（指示なし）＝似かた優先

KIND_MEETING = "meeting"
KIND_EMOTIONAL = "emotional"

DEFAULT_ENGINES = ["voxcpm2", "qwen3tts", "cosyvoice3:zero_shot", "cosyvoice3:cross_lingual", "chatterbox"]
DEFAULT_LANGS = ["ja", "en", "fr", "zh"]


class ConfigError(ValueError):
    """設定の書き方が違うときのエラー（メッセージは利用者向けの日本語）。"""


@dataclass
class SpeakerSpec:
    name: str = ""
    profile: str = ""       # voiceprofile が作ったフォルダ（profile.json がある）
    folder: str = ""        # wav を入れただけのフォルダ（+ 書き起こし CSV）
    transcripts: str = ""   # folder のとき：書き起こし CSV（省略時は folder/transcripts.csv）


@dataclass
class ExperimentConfig:
    name: str = "exp"
    speakers: list[SpeakerSpec] = field(default_factory=list)
    targets: str = ""                                   # 合成する文の CSV
    target_langs: list[str] = field(default_factory=lambda: list(DEFAULT_LANGS))
    engines: list[str] = field(default_factory=lambda: list(DEFAULT_ENGINES))
    engine_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    conditions: list[str] = field(default_factory=lambda: list(ALLOWED_CONDITIONS))
    repeats: int = 2                                    # 同じ条件の繰り返し回数（VoxCPM は実行ごとにぶれる）
    seed: int = 1234
    stretch_factors: list[float] = field(default_factory=lambda: [1.3, 2.0])
    stretch_method: str = "auto"                        # auto / rubberband / librosa
    emotion_engines: list[str] = field(default_factory=lambda: ["voxcpm2"])
    line_ref_min_s: float = 2.0                         # C0 の手本の長さ
    line_ref_max_s: float = 4.0
    line_refs_per_speaker: int = 1
    ref_lang: str = ""                                  # 手本の言語を限定（例 ja）。空なら全部から選ぶ
    profile_ref_select: str = "best"                    # best（一番良い1本）/ each（1本ずつ全部）
    profile_ref_min_s: float = 10.0                     # C1 の手本の長さ
    profile_ref_max_s: float = 15.0
    folder_ref_count: int = 3                           # wav フォルダ方式：手本に回す本数
    real_ceiling: bool = True                           # 実声どうしの SIM（上限の目安）も出す
    ceiling_max_clips: int = 6
    max_texts_per_lang: int | None = None               # お試し用：言語ごとの文の数を減らす
    embedder: str = "ecapa"                             # voicecheck の埋め込み（ecapa / resemblyzer / all / none）
    calib: str = ""                                     # voicecheck calibrate の Excel（合格ライン）
    threshold: str = ""                                 # 例 0.45 または ecapa=0.45
    # 実行時にだけ使う（ファイルには書かない）
    base_dir: str = "."


_KEYS = {f.name for f in fields(ExperimentConfig)} - {"base_dir"}
_SPEAKER_KEYS = {f.name for f in fields(SpeakerSpec)}


def _unknown(kind: str, got: set[str], allowed: set[str]) -> ConfigError:
    return ConfigError(
        f"{kind}に知らない項目があります: {', '.join(sorted(got))}\n"
        f"  使える項目: {', '.join(sorted(allowed))}"
    )


def _check_engine(spec: str) -> None:
    try:
        validate_engine_spec(spec)
    except EngineError as e:
        raise ConfigError(str(e)) from None


def config_from_dict(data: dict[str, Any], base_dir: str | Path = ".") -> ExperimentConfig:
    if not isinstance(data, dict):
        raise ConfigError("設定ファイルの中身は「項目: 値」の形にしてください")
    bad = set(data) - _KEYS
    if bad:
        raise _unknown("設定", bad, _KEYS)

    cfg = ExperimentConfig(base_dir=str(base_dir))
    simple = _KEYS - {"speakers"}
    for k in simple:
        if k in data and data[k] is not None:
            setattr(cfg, k, data[k])

    # --- speakers
    raw = data.get("speakers") or []
    if not isinstance(raw, list) or not raw:
        raise ConfigError("speakers（話者の一覧）が空です。profile: か folder: を持つ話者を1人以上書いてください")
    speakers: list[SpeakerSpec] = []
    for i, s in enumerate(raw, start=1):
        if not isinstance(s, dict):
            raise ConfigError(f"speakers の {i} 番目は「name: ..., profile: ...」の形にしてください")
        bad = set(s) - _SPEAKER_KEYS
        if bad:
            raise _unknown(f"speakers の {i} 番目", bad, _SPEAKER_KEYS)
        sp = SpeakerSpec(**{k: ("" if v is None else str(v)) for k, v in s.items()})
        if bool(sp.profile) == bool(sp.folder):
            raise ConfigError(f"speakers の {i} 番目（{sp.name or '名前なし'}）は profile: か folder: のどちらか一方だけを書いてください")
        if not sp.name:
            sp.name = Path(sp.profile or sp.folder).name
        speakers.append(sp)
    names = [s.name for s in speakers]
    if len(set(names)) != len(names):
        raise ConfigError(f"話者の name が重複しています: {names}")
    cfg.speakers = speakers

    # --- 値の検査
    if not cfg.targets:
        raise ConfigError("targets（合成する文の CSV）を書いてください")
    cfg.target_langs = [str(x).strip().lower() for x in _as_list(cfg.target_langs)]
    if not cfg.target_langs:
        raise ConfigError("target_langs が空です（例: [ja, en, fr, zh]）")
    cfg.engines = [str(x).strip() for x in _as_list(cfg.engines)]
    if not cfg.engines:
        raise ConfigError("engines が空です")
    for e in cfg.engines:
        _check_engine(e)
    cfg.emotion_engines = [str(x).strip() for x in _as_list(cfg.emotion_engines)]
    for e in cfg.emotion_engines:
        _check_engine(e)
    cfg.conditions = [str(x).strip() for x in _as_list(cfg.conditions)]
    bad = [c for c in cfg.conditions if c not in ALLOWED_CONDITIONS]
    if bad:
        raise ConfigError(f"知らない条件: {', '.join(bad)}（使える条件: {', '.join(ALLOWED_CONDITIONS)}）")
    if COND_STRETCH in cfg.conditions and COND_PROFILE not in cfg.conditions:
        # 早回しは「登録手本の出力」を加工するので、手本の条件も必ず作る
        cfg.conditions.insert(0, COND_PROFILE)
    cfg.conditions = [c for c in ALLOWED_CONDITIONS if c in cfg.conditions]  # 並びを固定
    try:
        cfg.repeats = int(cfg.repeats)
        cfg.seed = int(cfg.seed)
        cfg.stretch_factors = [float(x) for x in _as_list(cfg.stretch_factors)]
        cfg.line_refs_per_speaker = int(cfg.line_refs_per_speaker)
        cfg.folder_ref_count = int(cfg.folder_ref_count)
        cfg.ceiling_max_clips = int(cfg.ceiling_max_clips)
        for k in ("line_ref_min_s", "line_ref_max_s", "profile_ref_min_s", "profile_ref_max_s"):
            setattr(cfg, k, float(getattr(cfg, k)))
        if cfg.max_texts_per_lang is not None:
            cfg.max_texts_per_lang = int(cfg.max_texts_per_lang)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"数値の項目の書き方が違います: {e}") from None
    if cfg.repeats < 1:
        raise ConfigError("repeats は 1 以上にしてください")
    if COND_STRETCH in cfg.conditions:
        if not cfg.stretch_factors or any(f <= 1.0 or f > 4.0 for f in cfg.stretch_factors):
            raise ConfigError("stretch_factors は 1 より大きく 4 以下の数のリストにしてください（例: [1.3, 2.0]）")
    if cfg.profile_ref_select not in ("best", "each"):
        raise ConfigError("profile_ref_select は best か each にしてください")
    if cfg.stretch_method not in ("auto", "rubberband", "librosa"):
        raise ConfigError("stretch_method は auto / rubberband / librosa のどれかにしてください")
    if cfg.line_ref_min_s <= 0 or cfg.line_ref_max_s < cfg.line_ref_min_s:
        raise ConfigError("line_ref_min_s / line_ref_max_s の関係がおかしいです")
    if cfg.max_texts_per_lang is not None and cfg.max_texts_per_lang < 1:
        raise ConfigError("max_texts_per_lang は 1 以上にしてください")
    if not isinstance(cfg.engine_options, dict) or not all(isinstance(v, dict) for v in cfg.engine_options.values()):
        raise ConfigError("engine_options は「エンジン名: {項目: 値}」の形にしてください")
    cfg.ref_lang = str(cfg.ref_lang or "").strip().lower()
    cfg.calib = str(cfg.calib or "")
    cfg.threshold = str(cfg.threshold or "")
    return cfg


def _as_list(v: Any) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {p}")
    text = p.read_text(encoding="utf-8-sig")
    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ConfigError(
                    "YAML を読むには pyyaml が必要です（pip install pyyaml）。入れたくなければ、同じ内容を .json で書いてください"
                ) from None
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"設定ファイルは .yaml / .yml / .json にしてください: {p.name}")
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001 - YAML / JSON の書き方の誤り
        raise ConfigError(f"設定ファイルの書き方が違います（{p.name}）: {str(e)[:300]}") from None
    return config_from_dict(data or {}, p.resolve().parent)


def resolve_path(cfg: ExperimentConfig, p: str) -> Path:
    """設定内の相対パスを、設定ファイルのあるフォルダ基準で絶対パスにする。"""
    q = Path(p).expanduser()
    return q if q.is_absolute() else (Path(cfg.base_dir) / q).resolve()


def engine_options_for(cfg: ExperimentConfig, spec: str) -> dict[str, Any]:
    """engine_options から、エンジン名（voxcpm2）と完全な指定（cosyvoice3:zero_shot）の設定を合成する（後者が優先）。"""
    name, _ = parse_engine_spec(spec)
    out: dict[str, Any] = {}
    out.update(cfg.engine_options.get(name, {}))
    out.update(cfg.engine_options.get(spec, {}))
    return out
