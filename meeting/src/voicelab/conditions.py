"""実験の条件を「仕事（Job）の一覧」に展開する。

条件（design.md 6章：1回に1つだけ変える）:
  C0 line_ref_16k   手本＝評価用の実声から取った 2〜4 秒の 1 本を 16 kHz にしたもの（上流 pyVideoTrans の再現）
  C1 profile_ref    手本＝登録した良い手本（10〜15 秒・元のサンプルレート・書き起こし付き）
  C2 stretch_xN     C1 の出力を N 倍速にしたもの（コンピュータ上の後処理だけ。GPU 不要）
  C3 emotion_*      VoxCPM2 など：感情の指示（instruct） / 手本だけ（refonly） / 書き起こし付き複製（similarity）
各条件 × エンジン × 目標言語 × 文 × 繰り返し（repeats）が、1 つの Job になる。
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

from .config import (
    COND_EMOTION,
    COND_LINE,
    COND_PROFILE,
    COND_STRETCH,
    EMO_INSTRUCT,
    EMO_REFONLY,
    EMO_SIMILARITY,
    KIND_EMOTIONAL,
    ExperimentConfig,
)
from .engines import engine_caps
from .engines.base import MODE_INSTRUCT, MODE_REFONLY, MODE_SIMILARITY
from .texts import TargetText

def condition_sort_key(cond: str) -> tuple:
    if cond == COND_LINE:
        return (0, 0.0, cond)
    if cond.startswith(COND_PROFILE):
        return (1, 0.0, cond)
    m = re.match(r"stretch_x([0-9.]+)$", cond)
    if m:
        return (2, float(m.group(1)), cond)
    order = {EMO_SIMILARITY: 0, EMO_REFONLY: 1, EMO_INSTRUCT: 2}
    if cond in order:
        return (3, float(order[cond]), cond)
    return (9, 0.0, cond)


@dataclass
class RefInfo:
    kind: str            # "line" か "profile"
    path: str            # TTS に渡す wav（out/refs の下に作ったもの）
    source: str          # 元ファイル（評価の ref には絶対に使えない）
    text: str            # 手本の書き起こし（使えないとき ""）
    lang: str
    sec: float
    sr: int
    stem: str = ""
    note: str = ""


@dataclass
class Job:
    job_id: str
    speaker: str
    engine: str                  # 例 cosyvoice3:zero_shot
    engine_label: str            # ラベルに使う名前（ドライランは dry:... が付く）
    condition: str
    lang: str
    text_id: str
    kind: str
    repeat: int
    text: str
    emotion: str                 # 感情の指示（条件に関係なく文の側の値。使うかどうかは emotion_arg）
    ref: RefInfo | None
    ref_text_arg: str | None     # TTS に渡す書き起こし（None なら渡さない）
    emotion_arg: str | None      # TTS に渡す感情の指示
    seed: int
    out_rel: str
    parent: str = ""             # stretch：元の Job の job_id
    factor: float = 0.0          # stretch の倍率
    skip_reason: str = ""        # 空でなければ実行しない（対応外の言語など）
    tags: dict = field(default_factory=dict)

    @property
    def is_derived(self) -> bool:
        return bool(self.parent)


def fmt_factor(f: float) -> str:
    """1.3 → '1.3'、2 → '2.0'（条件名 stretch_x2.0 のため）。"""
    s = f"{f:g}"
    return s if ("." in s or "e" in s) else s + ".0"


def safe_name(s: str) -> str:
    """ファイル名に使える形にする（日本語はそのまま、記号と空白は _）。"""
    out = re.sub(r"[\\/:*?\"<>|\s;]+", "_", s.strip())
    return out or "x"


def seed_for(base: int, text_id: str, repeat: int) -> int:
    """文と繰り返し番号から決まる乱数の種。条件が違っても同じ文・同じ繰り返しなら同じ種（＝対で比べられる）。"""
    return int((base + (zlib.crc32(text_id.encode()) % 100000) * 7 + repeat * 100003) % (2**31 - 1))


def _out_rel(speaker: str, engine: str, cond: str, text_id: str, repeat: int) -> str:
    return f"audio/{safe_name(speaker)}/{safe_name(engine.replace(':', '-'))}/{cond}/{safe_name(text_id)}_r{repeat}.wav"


def _job_id(speaker: str, engine: str, cond: str, lang: str, text_id: str, repeat: int) -> str:
    return "|".join([speaker, engine, cond, lang, text_id, f"r{repeat}"])


def build_jobs(
    cfg: ExperimentConfig,
    speaker: str,
    line_refs: list[RefInfo],
    profile_refs: list[RefInfo],
    texts: list[TargetText],
    dry_run: bool = False,
) -> list[Job]:
    """1 人の話者について、全部の Job を作る。"""
    jobs: list[Job] = []
    prefix = "dry:" if dry_run else ""

    def mk(spec: str, cond: str, t: TargetText, rep: int, ref: RefInfo | None, ref_text: str | None,
           emo: str | None, skip: str = "", parent: str = "", factor: float = 0.0) -> Job:
        return Job(
            job_id=_job_id(speaker, spec, cond, t.lang, t.id, rep), speaker=speaker, engine=spec,
            engine_label=prefix + spec, condition=cond, lang=t.lang, text_id=t.id, kind=t.kind, repeat=rep,
            text=t.text, emotion=t.emotion, ref=ref, ref_text_arg=ref_text, emotion_arg=emo,
            seed=seed_for(cfg.seed, t.id, rep), out_rel=_out_rel(speaker, spec, cond, t.id, rep),
            parent=parent, factor=factor, skip_reason=skip,
        )

    meeting = [t for t in texts if t.kind != KIND_EMOTIONAL]
    emotional = [t for t in texts if t.kind == KIND_EMOTIONAL]

    for spec in cfg.engines:
        caps = engine_caps(spec)
        if caps.emotion_only:
            continue  # 感情の条件だけに使うエンジン（cosyvoice3:instruct2）は、ここには出さない

        def lang_skip(t: TargetText) -> str:
            return "" if (caps.langs is None or t.lang in caps.langs) else f"{spec} は {t.lang} に対応していません"

        for t in meeting:
            for rep in range(cfg.repeats):
                # --- C0
                if COND_LINE in cfg.conditions and line_refs:
                    ref = line_refs[rep % len(line_refs)]
                    jobs.append(mk(spec, COND_LINE, t, rep, ref, ref.text or None, None, lang_skip(t)))
                # --- C1
                if COND_PROFILE in cfg.conditions and profile_refs:
                    for k, pref in enumerate(profile_refs):
                        cond = COND_PROFILE if cfg.profile_ref_select == "best" else f"{COND_PROFILE}_{safe_name(pref.stem)}"
                        j = mk(spec, cond, t, rep, pref, pref.text or None, None, lang_skip(t))
                        jobs.append(j)
                        # --- C2：最初の（一番良い）手本の C1 だけを早回しする
                        if COND_STRETCH in cfg.conditions and k == 0:
                            for f in cfg.stretch_factors:
                                jobs.append(mk(spec, f"stretch_x{fmt_factor(f)}", t, rep, pref, None, None,
                                               j.skip_reason, parent=j.job_id, factor=f))

    # --- C3：感情
    if COND_EMOTION in cfg.conditions and profile_refs and emotional:
        pref = profile_refs[0]
        for spec in cfg.emotion_engines:
            caps = engine_caps(spec)
            for t in emotional:
                skip = "" if (caps.langs is None or t.lang in caps.langs) else f"{spec} は {t.lang} に対応していません"
                for rep in range(cfg.repeats):
                    if MODE_SIMILARITY in caps.emotion_modes:
                        jobs.append(mk(spec, EMO_SIMILARITY, t, rep, pref, pref.text or None, None, skip))
                    if MODE_REFONLY in caps.emotion_modes:
                        jobs.append(mk(spec, EMO_REFONLY, t, rep, pref, None, None, skip))
                    if MODE_INSTRUCT in caps.emotion_modes:
                        jobs.append(mk(spec, EMO_INSTRUCT, t, rep, pref, pref.text or None, t.emotion, skip))
    return jobs
