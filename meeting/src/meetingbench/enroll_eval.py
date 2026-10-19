"""登録音声との照合（enroll_assign）の結果を、asrcheck で採点する。

- 条件 = 埋め込み器 × 登録時間 × 登録の感情 × 照合の方式 × 登録発話の選び方（乱数 seed）× 録音（A/B）× 区間（正解/silero）
- 採点は meetingbench の他の条件と同じ asrcheck（CER・cpCER・DER・話者正解率）。話者ごとの名前（F1/F2/M1/M2）がそのまま出力になる。
- 速くするため、CER（話者に依らない）は区間ごとに 1 回だけ計算し、それ以外（cpCER・DER・話者正解率）を条件ごとに計算する。
  この近道が asrcheck.evaluate_recording と同じ値になることは、テストで確かめている。
- 結果は runs/enroll/results.jsonl に 1 条件 1 行で追記する（途中で止まっても続きから再開できる）。
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from asrcheck.cli import load_reference
from asrcheck.diar import compute_der, speaker_accuracy
from asrcheck.evaluate import DER_COLLARS, EngineResult, EvalOptions, RecResult, _speech_time
from asrcheck.kpi import DEFAULT_KPIS, Kpi
from asrcheck.metrics import cp_cer, text_errors
from asrcheck.model import Utt, sort_by_time
from asrcheck.report import summary_rows

from . import enroll_assign as ea
from . import scoring
from .pipeline import Layout

BASELINE_CONDS = ("silero__built-ja__n4", "silero__built-zh__n4", "silero__truth__none",
                  "oracle__built-ja__n4", "oracle__built-zh__n4", "oracle__truth__none")


@dataclass(frozen=True)
class Spec:
    embedder: str
    sec: int
    mode: str
    rule: str
    seed: int
    variant: str
    seg: str
    unk_pct: float | None = None     # 不明にするしきい値＝登録音声の本人点数の下位何 % か（None: 使わない）
    smooth: str = ""                 # "" / "groups"（隙間のない続きの区間をまとめて判定）/ "short"（短い区間は前後を引き継ぐ）

    @property
    def key(self) -> str:
        return "|".join(f"{k}={v}" for k, v in asdict(self).items())

    @property
    def cond_name(self) -> str:
        extra = (f"+不明{self.unk_pct:g}%" if self.unk_pct is not None else "") + (f"+平滑化:{self.smooth}" if self.smooth else "")
        return f"{self.variant}:{self.seg}/{self.embedder}/{self.sec}s/{self.mode}/{self.rule}/seed{self.seed}{extra}"


def make_grid(embedders: list[str], secs=(30, 60, 180), modes=("mixed", "calm"), rules=("mean", "max"), seeds=range(5),
              variants=("A", "B"), segs=("oracle", "silero")) -> list[Spec]:
    return [Spec(e, s, m, r, sd, v, g) for e in embedders for m in modes for s in secs for r in rules for sd in seeds
            for v in variants for g in segs]


# ---------------------------------------------------------------------------
# 採点（asrcheck.evaluate_recording と同じ中身。CER だけ使い回す）
# ---------------------------------------------------------------------------
def evaluate_labels(name: str, rec: str, ref: list[Utt], items: list[dict], labels: list[str], opts: EvalOptions, text_res=None,
                    audio_s: float | None = None, elapsed_s: float | None = None) -> EngineResult:
    hyp = [Utt(it["start_ms"] / 1000, it["end_ms"] / 1000, lab, it["text"]) for it, lab in zip(items, labels)]
    res = RecResult(rec=rec, ref=sort_by_time(ref), hyp=sort_by_time(hyp))
    res.ref_speech_s = _speech_time(res.ref)
    res.text = text_res if text_res is not None else text_errors(res.ref, res.hyp, opts.norm)
    res.cp = cp_cer(res.ref, res.hyp, opts.norm, force=True)
    ref_segs = [(u.start, u.end, u.speaker) for u in res.ref if u.has_time and u.speaker]
    hyp_segs = [(u.start, u.end, u.speaker) for u in sort_by_time(hyp) if u.has_time and u.speaker]
    for c in sorted({*DER_COLLARS, opts.collar}):
        res.der[c] = compute_der(ref_segs, hyp_segs, collar=c)       # type: ignore[arg-type]
    res.spk = speaker_accuracy([(u.start or 0.0, u.end or 0.0, u.speaker if u.has_time else "") for u in res.ref], hyp_segs)   # type: ignore[arg-type]
    return EngineResult(name=name, recs=[res], kind="登録音声との照合", elapsed_s=elapsed_s, audio_s=audio_s)


def extra_metrics(eng: EngineResult, truth: list[str], labels: list[str]) -> dict:
    """asrcheck の標準指標に加えて見る数字。
    strict_acc: 名前をそのまま比べた話者正解率（asrcheck は最適な対応づけをするので、名前の取り違えで有利になっていないかの確認）。
    seg_acc   : 字幕の区間ごとの正解率（その区間に最も長く重なった正解の話者と同じ名前か）。"""
    r = eng.recs[0]
    utts = r.spk.utts if r.spk else []
    strict = float(np.mean([u.dominant == u.ref_speaker for u in utts])) if utts else None
    ident = bool(r.spk and all(k == v for k, v in r.spk.mapping.items() if k != ea.UNKNOWN))
    d0 = r.der.get(0.0)
    der_ident = bool(d0 and all(v == k for k, v in d0.mapping.items() if k in d0.hyp_speakers))
    seg_acc = float(np.mean([a == b for a, b in zip(labels, truth)])) if labels else None
    return {"strict_acc": strict, "spk_map_identity": ident, "der_map_identity": der_ident, "seg_acc": seg_acc,
            "n_unknown": int(sum(1 for lab in labels if lab == ea.UNKNOWN)), "n_segments": len(labels)}


# ---------------------------------------------------------------------------
class Bench:
    """採点に要るものをまとめて持つ（正解・区間・文字起こし・埋め込み・処理時間）。"""

    def __init__(self, lay: Layout, embedders: list[str], variants=("A", "B"), segs=("oracle", "silero")):
        from .enroll_embed import meta_path, seg_items, store_path

        self.lay, self.embedders, self.variants, self.segs = lay, embedders, list(variants), list(segs)
        self.opts = scoring.default_opts()
        self.kpis: dict[str, Kpi] = DEFAULT_KPIS
        self.ref: dict[str, list[Utt]] = {}
        self.audio_s: dict[str, float] = {}
        self.items: dict[tuple[str, str], list[dict]] = {}
        self.truth: dict[tuple[str, str], list[str]] = {}
        self.text_res: dict[tuple[str, str], Any] = {}
        self.times: dict[tuple[str, str], dict] = {}
        for v in self.variants:
            ref_by, _names, _n, _e, durs = load_reference(str(lay.ref(v)), [str(lay.wav(v))])
            self.ref[v] = next(iter(ref_by.values()))
            self.audio_s[v] = float(next(iter(durs.values())))
            for g in self.segs:
                self.items[(v, g)] = seg_items(lay, v, g)
                self.truth[(v, g)] = scoring.truth_labels(self.ref[v], self.items[(v, g)])
                t: dict = {}
                if g != "oracle":
                    t["vad"] = json.loads((lay.runs(v) / f"seg_{g}.json").read_text(encoding="utf-8"))["meta"].get("elapsed_sec")
                else:
                    t["vad"] = 0.0
                t["asr"] = json.loads((lay.runs(v) / f"asr_{g}.json").read_text(encoding="utf-8"))["meta"].get("elapsed_sec")
                self.times[(v, g)] = t
        self.store: dict[str, ea.EmbeddingStore] = {}
        self.meta: dict[str, dict] = {}
        for e in embedders:
            self.store[e] = ea.EmbeddingStore.load(store_path(lay, e))
            mp = meta_path(lay, e)
            self.meta[e] = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
            if not len(self.store[e]):
                raise FileNotFoundError(f"{e} の埋め込みがありません。先に enroll-embed --embedder {e} を実行してください")
        self._seg_cache: dict[tuple, np.ndarray] = {}
        self._enr_cache: dict[tuple, ea.Enrollment] = {}
        self.check_enrollment_unused()

    def check_enrollment_unused(self) -> None:
        manifest = json.loads((self.lay.root / "manifest.json").read_text(encoding="utf-8"))
        used = ea.used_in_meeting(manifest)
        for e, st in self.store.items():
            bad = [k for k in st.vec if k.startswith(ea.KEY_ENR) and (ea.parse_clip_name(k[len(ea.KEY_ENR):])[0], k[len(ea.KEY_ENR):]) in used]
            if bad:
                raise ValueError(f"{e}: 会議で使った発話が登録候補に入っています: {bad[:3]}")

    def seg_embs(self, e: str, v: str, g: str) -> np.ndarray:
        k = (e, v, g)
        if k not in self._seg_cache:
            st = self.store[e]
            keys = [ea.seg_key(v, g, it["start_ms"], it["end_ms"]) for it in self.items[(v, g)]]
            miss = [x for x in keys if x not in st]
            if miss:
                raise ValueError(f"{e}: 区間の埋め込みが足りません（{len(miss)} 個）。enroll-embed をやり直してください")
            self._seg_cache[k] = st.matrix(keys)
        return self._seg_cache[k]

    def enrollment(self, e: str, mode: str, sec: int, seed: int) -> ea.Enrollment:
        k = (e, mode, sec, seed)
        if k not in self._enr_cache:
            self._enr_cache[k] = ea.make_enrollment(self.store[e], mode, float(sec), seed)
        return self._enr_cache[k]

    def embed_elapsed(self, e: str, v: str, g: str) -> float | None:
        d = self.meta[e].get("segments", {}).get(f"{v}:{g}")
        return d["elapsed_sec"] if d else None

    def elapsed_total(self, e: str, v: str, g: str) -> float | None:
        t, emb = self.times[(v, g)], self.embed_elapsed(e, v, g)
        if emb is None or t.get("asr") is None or t.get("vad") is None:
            return None
        return t["vad"] + t["asr"] + emb

    # ---- 1 条件 ----
    def assign(self, s: Spec) -> tuple[list[str], np.ndarray, np.ndarray, float | None]:
        enr = self.enrollment(s.embedder, s.mode, s.sec, s.seed)
        x = self.seg_embs(s.embedder, s.variant, s.seg)
        sim = ea.similarity(x, enr, s.rule)
        items = self.items[(s.variant, s.seg)]
        starts, ends = [it["start_ms"] for it in items], [it["end_ms"] for it in items]
        dur = [(b - a) / 1000.0 for a, b in zip(starts, ends)]
        if s.smooth == "groups":
            sim = ea.smooth_by_groups(sim, dur, ea.pool_continuous(x, starts, ends))
        thr = ea.threshold_from_enrollment(enr, s.unk_pct, s.rule) if s.unk_pct is not None else None
        labels, best, margin = ea.decide(sim, enr.speakers, thr)
        if s.smooth == "short":
            labels = ea.smooth_short(labels, sim, starts, ends, enr.speakers)
        return labels, best, margin, thr

    def evaluate(self, s: Spec, keep_engine: bool = False) -> dict:
        t0 = time.perf_counter()
        labels, best, margin, thr = self.assign(s)
        t_assign = time.perf_counter() - t0
        items = self.items[(s.variant, s.seg)]
        tk = (s.variant, s.seg)
        eng = evaluate_labels(s.cond_name, self.lay.wav(s.variant).name, self.ref[s.variant], items, labels, self.opts, self.text_res.get(tk),
                              self.audio_s[s.variant], self.elapsed_total(s.embedder, s.variant, s.seg))
        if tk not in self.text_res:
            self.text_res[tk] = eng.recs[0].text
        row = summary_rows([eng], self.kpis, self.opts)[0]
        enr = self.enrollment(s.embedder, s.mode, s.sec, s.seed)
        truth = self.truth[tk]
        correct = np.array([a == b for a, b in zip(self.assign_plain(s), truth)]) if thr is not None else np.array([a == b for a, b in zip(labels, truth)])
        out = {"spec": asdict(s), "row": _clean(row), "extra": {
            **extra_metrics(eng, truth, labels), "threshold": thr, "enroll_sec": [round(x, 1) for x in enr.secs],
            "enroll_clips": [len(c) for c in enr.clips], "assign_sec": t_assign,
            "median_best": float(np.median(best)), "median_margin": float(np.median(margin)),
            "seg_acc_forced": float(correct.mean())}}
        if keep_engine:
            out["engine"] = eng
        return out

    def assign_plain(self, s: Spec) -> list[str]:
        """不明を使わずに、いちばん近い登録者を選んだときのラベル（不明にした区間が本当は合っていたかを見るため）。"""
        from dataclasses import replace

        return self.assign(replace(s, unk_pct=None))[0]


def _clean(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 結果の保存と読み込み
# ---------------------------------------------------------------------------
def results_path(lay: Layout) -> Path:
    return lay.root / "runs" / "enroll" / "results.jsonl"


def load_results(lay: Layout) -> dict[str, dict]:
    p = results_path(lay)
    out: dict[str, dict] = {}
    if p.is_file():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                out[Spec(**d["spec"]).key] = d
    return out


def run_grid(bench: Bench, specs: list[Spec], say=print, force: bool = False) -> dict[str, dict]:
    done = {} if force else load_results(bench.lay)
    todo = [s for s in specs if s.key not in done]
    say(f"条件 {len(specs)} 個のうち、未実施 {len(todo)} 個を採点します")
    p = results_path(bench.lay)
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(p, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, start=1):
            r = bench.evaluate(s)
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
            f.flush()
            done[s.key] = r
            if i % 20 == 0 or i == len(todo):
                say(f"    {i}/{len(todo)}（{time.perf_counter() - t0:.0f} 秒）")
    return done


def aggregate(results: list[dict], metrics=("spk_acc", "der_0", "der_0.25", "cpcer", "cer")) -> dict[tuple, dict]:
    """(embedder, sec, mode, rule, variant, seg, unk_pct, smooth) ごとに、登録発話の選び方（seed）をまたいだ平均・最小・最大。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        s = r["spec"]
        groups[(s["embedder"], s["sec"], s["mode"], s["rule"], s["variant"], s["seg"], s["unk_pct"], s["smooth"])].append(r)
    out = {}
    for k, rs in groups.items():
        d: dict[str, Any] = {"n": len(rs)}
        for m in metrics:
            vals = [r["row"].get(m) for r in rs if r["row"].get(m) is not None]
            d[m] = float(np.mean(vals)) if vals else None
            d[m + "_min"] = float(np.min(vals)) if vals else None
            d[m + "_max"] = float(np.max(vals)) if vals else None
        for m in ("strict_acc", "seg_acc", "n_unknown", "enroll_sec_mean", "seg_acc_forced"):
            if m == "enroll_sec_mean":
                vals = [float(np.mean(r["extra"]["enroll_sec"])) for r in rs]
            else:
                vals = [r["extra"][m] for r in rs if r["extra"].get(m) is not None]
            d[m] = float(np.mean(vals)) if vals else None
        d["map_identity_all"] = all(r["extra"]["spk_map_identity"] and r["extra"]["der_map_identity"] for r in rs)
        d["minutes"] = float(np.mean([r["row"]["minutes"] for r in rs if r["row"].get("minutes") is not None])) if any(r["row"].get("minutes") is not None for r in rs) else None
        d["verdicts"] = [r["row"].get("verdict") for r in rs]
        d["seeds"] = sorted(r["spec"]["seed"] for r in rs)
        out[k] = d
    return out


@dataclass
class Baselines:
    """比べる相手: pyVideoTrans の内蔵 built（教師なし）と、区間の限界（話者だけ正解）。"""

    conds: dict[tuple[str, str], Any] = field(default_factory=dict)    # (variant, cond_dir_name) → scoring.Cond


def load_baselines(lay: Layout, variants=("A", "B")) -> Baselines:
    b = Baselines()
    for v in variants:
        for c in scoring.score_variant(lay, v, only=set(BASELINE_CONDS)):
            b.conds[(v, c.dir.name)] = c
    return b
