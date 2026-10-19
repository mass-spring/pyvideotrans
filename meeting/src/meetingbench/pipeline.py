"""測定の各段階（区間 → 文字起こし → 話者分離 → 割り当て）。結果は data/testdata/pseudo_meeting/runs/<A|B>/ の下に保存する。

runs/<v>/seg_<kind>.json            区間 {"segments": [[start_ms, end_ms]...], "meta": {...}}
runs/<v>/asr_<kind>.json            文字起こし {"items": [{start_ms,end_ms,text}...], "meta": {...}}
runs/<v>/diar/<cfg>_<n>/            話者分離の実行結果（subtitles.json, speaker.json = 正解の区間に対する本体の出力, raw_diar.json, meta.json）
runs/<v>/cond/<seg>__<cfg>__<n>/    評価用: meeting<v>.srt, speaker.json（本体と同じ形）, raw_diar.csv, meta.json
どの段階も、結果のファイルがあれば飛ばす（--force で作り直す）。途中で止まっても続きから再開できる。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import diarize as dz
from . import segments as sg

DEFAULT_ROOT = Path("data/testdata/pseudo_meeting")


class Layout:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    def wav(self, v: str) -> Path:
        return self.root / f"meeting{v}.wav"

    def ref(self, v: str) -> Path:
        return self.root / f"ref_meeting{v}.csv"

    def runs(self, v: str) -> Path:
        return self.root / "runs" / v


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audio_seconds(wav: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(wav)).duration)


def stage_segments(lay: Layout, v: str, kinds: list[str], force: bool = False, say=print) -> None:
    for k in kinds:
        out = lay.runs(v) / f"seg_{k}.json"
        if out.is_file() and not force:
            continue
        say(f"[{v}] 区間: {k} ...")
        if k == "oracle":
            segs, meta = sg.oracle_segments(lay.ref(v)), {"source": lay.ref(v).name}
        elif k == "silero":
            segs, meta = sg.silero_segments(lay.wav(v))
        elif k == "fsmn":
            segs, meta = sg.fsmn_segments(lay.wav(v))
        else:
            raise ValueError(f"区間の種類が違います: {k}")
        _dump(out, {"segments": segs, "meta": meta})
        say(f"    {len(segs)} 区間")


def stage_asr(lay: Layout, v: str, kinds: list[str], force: bool = False, threads: int = 2, say=print) -> None:
    for k in kinds:
        out = lay.runs(v) / f"asr_{k}.json"
        if out.is_file() and not force:
            continue
        seg_file = lay.runs(v) / f"seg_{k}.json"
        if not seg_file.is_file():
            stage_segments(lay, v, [k], say=say)
        segs = _load(seg_file)["segments"]
        say(f"[{v}] SenseVoiceSmall で文字起こし（区間={k}、{len(segs)} 区間）...")
        items, meta = sg.transcribe_sensevoice(lay.wav(v), segs, lay.runs(v) / "tmp" / f"asr_{k}", threads=threads)
        meta["audio_sec"] = audio_seconds(lay.wav(v))
        meta["rtf"] = meta["elapsed_sec"] / meta["audio_sec"]
        _dump(out, {"items": items, "meta": meta})
        say(f"    {meta['elapsed_sec']:.0f} 秒（RTF {meta['rtf']:.3f}）")


def diar_dir(lay: Layout, v: str, cfg: str, n: str) -> Path:
    return lay.runs(v) / "diar" / f"{cfg}_{n}"


def stage_diar(lay: Layout, v: str, cfg: str, n: str, force: bool = False, say=print) -> dict:
    d = diar_dir(lay, v, cfg, n)
    meta_file = d / "meta.json"
    if meta_file.is_file() and not force:
        return _load(meta_file)
    if not (lay.runs(v) / "seg_oracle.json").is_file():
        stage_segments(lay, v, ["oracle"], say=say)
    segs = _load(lay.runs(v) / "seg_oracle.json")["segments"]
    num = dz.NUM_SPEAKERS[n]
    say(f"[{v}] 話者分離: {cfg} / 人数={'4' if num > 0 else '自動'} ...")
    r = dz.run_backend(cfg, lay.wav(v), segs, num, d)
    audio_s = audio_seconds(lay.wav(v))
    meta = {"backend": cfg, "num_speakers_arg": num, "ok": r["ok"], "message": (r["message"] or "")[:2000],
            "elapsed_sec": r["elapsed_sec"], "audio_sec": audio_s, "rtf": r["elapsed_sec"] / audio_s,
            "raw_turns": len(r["raw"] or []), "raw_speakers": len({x[1] for x in (r["raw"] or [])}),
            "oracle_labels_speakers": len(set(r["labels"] or []))}
    if r["raw"] is not None:
        _dump(d / "raw_diar.json", r["raw"])
    _dump(meta_file, meta)
    say(f"    {'成功' if r['ok'] else '失敗: ' + str(r['message'])[:200]} / {r['elapsed_sec']:.0f} 秒（RTF {meta['rtf']:.3f}）/ 生の話者数 {meta['raw_speakers']}")
    return meta


def cond_name(seg: str, cfg: str, n: str) -> str:
    return f"{seg}__{cfg}__{n}"


def stage_assign(lay: Layout, v: str, say=print) -> list[Path]:
    """話者分離の結果ごと × 区間の種類ごと（文字起こしがあるもの）に、評価用のフォルダを作る。"""
    outs = []
    for dd in sorted((lay.runs(v) / "diar").glob("*/meta.json")):
        meta = _load(dd)
        if not meta.get("ok") or not (dd.parent / "raw_diar.json").is_file():
            continue
        raw = _load(dd.parent / "raw_diar.json")
        cfg_n = dd.parent.name
        cfg, n = cfg_n.rsplit("_", 1)
        for asr_file in sorted(lay.runs(v).glob("asr_*.json")):
            seg = asr_file.stem[len("asr_"):]
            items = [it for it in _load(asr_file)["items"] if it["text"].strip()]   # 本体と同じく、文字のない字幕は捨てる
            subs = [[it["start_ms"], it["end_ms"]] for it in items]
            labels = dz.assign_with_pvt(subs, raw)
            cd = lay.runs(v) / "cond" / cond_name(seg, cfg, n)
            cd.mkdir(parents=True, exist_ok=True)
            dz.write_srt(cd / f"meeting{v}.srt", items)
            (cd / "speaker.json").write_text(json.dumps(labels), encoding="utf-8")
            dz.write_raw_csv(cd / "raw_diar.csv", raw)
            info = {"seg": seg, "backend": cfg, "num_speakers": n, "subtitles": len(items), "labels_speakers": len(set(labels)),
                    "diar_elapsed_sec": meta["elapsed_sec"], "diar_rtf": meta["rtf"]}
            # 正解の区間に対しては、本体の関数を実際に呼んだ結果（speaker_oracle）と一致することを確かめる
            if seg == "oracle" and (dd.parent / "speaker.json").is_file():
                real = _load(dd.parent / "speaker.json")
                full = [list(x) for x in _load(lay.runs(v) / "seg_oracle.json")["segments"]]
                info["replay_equals_real_call"] = bool(dz.assign_with_pvt(full, raw) == real)
            _dump(cd / "meta.json", info)
            outs.append(cd)
    say(f"[{v}] 評価用のフォルダ {len(outs)} 個を作りました")
    return outs
