"""字幕の区間の作り方（正解の区間 / pyVideoTrans の VAD / FunASR の fsmn-vad）と、SenseVoiceSmall での文字起こし。

文字起こしは pyVideoTrans の `videotrans.process.stt_funasr.funasr_mlt`（SenseVoiceSmall の経路）をそのまま呼ぶ。
区間の切り出し（前後に 400ms ずつの無音を足して wav に書き出す）は本体の `BaseRecogn.cut_audio` と同じ手順。
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import pvt_bridge

SEG_KINDS = {
    "oracle": "正解の区間（話者の交代位置が正しく分かっている理想条件）",
    "silero": "pyVideoTrans の既定の区切り（VAD=silero、最長5秒）",
    "fsmn": "FunASR の fsmn-vad（長めの区間、最長30秒）",
}


def oracle_segments(ref_csv: Path) -> list[list[int]]:
    """正解の CSV の start,end（秒）→ [[開始ms, 終了ms]]（開始順）。"""
    rows = []
    with open(ref_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(round(float(r["start"]) * 1000)), int(round(float(r["end"]) * 1000))))
    return [list(r) for r in sorted(rows)]


def silero_segments(wav: Path) -> tuple[list[list[int]], dict]:
    """本体の _vad_split と同じ設定の読み方で、本体の VAD 関数を呼ぶ。"""
    st = pvt_bridge.settings()
    vad = pvt_bridge.vad_module()
    vad_type = st.get("vad_type", "tenvad")
    min_speech = max(int(float(st.get("min_speech_duration_ms", 1000))), 1000)
    max_speech = min(int(float(st.get("max_speech_duration_s", 6)) * 1000), 25000)
    if max_speech <= min_speech:
        max_speech = min_speech + 1000
    min_silence = max(int(st.get("min_silence_duration_ms", 600)), 100)
    kw = {"input_wav": str(wav), "threshold": float(st.get("threshold", 0.45)), "min_speech_duration_ms": min_speech,
          "max_speech_duration_ms": max_speech, "min_silent_duration_ms": min_silence}
    t0 = time.perf_counter()
    fn = vad.get_speech_timestamp if vad_type == "tenvad" else vad.get_speech_timestamp_silero
    segs = fn(**kw) or []
    return [[int(s), int(e)] for s, e in segs], {"vad_type": vad_type, "elapsed_sec": time.perf_counter() - t0, **kw}


def fsmn_segments(wav: Path) -> tuple[list[list[int]], dict]:
    from funasr import AutoModel

    t0 = time.perf_counter()
    model = AutoModel(model=str(pvt_bridge.models_dir() / "speech_fsmn_vad_zh-cn-16k-common-pytorch"), device="cpu",
                      disable_update=True, disable_pbar=True, disable_log=True)
    res = model.generate(input=str(wav), disable_pbar=True)
    segs = [[int(s), int(e)] for s, e in res[0]["value"]] if res else []
    return segs, {"elapsed_sec": time.perf_counter() - t0, "model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch (default params)"}


def cut_audio(wav: Path, segments_ms: list[list[int]], out_dir: Path) -> list[dict]:
    """本体 BaseRecogn.cut_audio と同じ: 各区間を切り出し、前後に 400ms ずつの無音を足して wav に書く。"""
    from pydub import AudioSegment

    out_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(str(wav))
    silent = AudioSegment.silent(duration=400, frame_rate=audio.frame_rate).set_channels(audio.channels).set_sample_width(audio.sample_width)
    items = []
    for i, (s, e) in enumerate(segments_ms):
        s, e = max(0, s), max(0, e)
        if e <= s:
            continue
        fn = out_dir / f"audio_{i}.wav"
        (silent + audio[s:e] + silent).export(str(fn), format="wav")
        items.append({"line": i + 1, "text": "", "start_time": s, "end_time": e, "filename": str(fn)})
    return items


def transcribe_sensevoice(wav: Path, segments_ms: list[list[int]], work_dir: Path, threads: int = 2) -> tuple[list[dict], dict]:
    """funasr_mlt（本体）で SenseVoiceSmall を CPU で動かす。戻り値: ([{start_ms,end_ms,text,emotion}], meta)。"""
    import shutil

    import torch

    torch.set_num_threads(threads)
    f = pvt_bridge.funasr_module()
    work_dir.mkdir(parents=True, exist_ok=True)
    clips = work_dir / "clips"
    items = cut_audio(wav, segments_ms, clips)
    lst = work_dir / "cut_audio_list.json"
    lst.write_text(json.dumps(items), encoding="utf-8")
    t0 = time.perf_counter()
    res, err = f.funasr_mlt(cut_audio_list=str(lst), detect_language="ja", model_name="iic/SenseVoiceSmall",
                            logs_file=str(work_dir / "funasr.log"), is_cuda=False, hotword="", cache_folder=str(work_dir))
    elapsed = time.perf_counter() - t0
    shutil.rmtree(clips, ignore_errors=True)   # 切り出した wav は不要（ディスクを空ける）
    if res is False:
        raise RuntimeError(f"funasr_mlt が失敗しました: {err}")
    out = [{"start_ms": int(r["start_time"]), "end_ms": int(r["end_time"]), "text": (r.get("text") or "").strip()} for r in res]
    return out, {"elapsed_sec": elapsed, "segments": len(out), "model": "iic/SenseVoiceSmall (pyVideoTrans funasr_mlt, CPU)"}
