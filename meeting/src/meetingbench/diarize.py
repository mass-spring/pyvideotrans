"""pyVideoTrans の話者分離（built / ali_CAM）を、本体の関数のまま呼び出して結果を保存する。

本体の task/_base.py::_diariz_common と同じ呼び方:
  built   : built_speakers(input_file, subtitles_file, speak_file, num_speakers=-1 か人数, language=<検出言語>)
  ali_CAM : cam_speakers(input_file, subtitles_file, speak_file, num_speakers, is_cuda=False)
  subtitles_file は [[開始ms, 終了ms], ...] の JSON、speak_file には「字幕1行に1人」の話者名リスト（speaker.json）が書かれる。

本体の関数は最後に出力（字幕ごとの話者）しか返さないので、`_assign_speakers` に渡される「話者分離モデルの生の出力」
（話者ごとの区間）を、呼び出しのあいだだけ横取りして保存する（本体のファイルは書き換えない。実行中だけ関数を包む）。
生の出力は「字幕の区間の切り方」に依存しないので、別の字幕の区間（ASR で切った区間）への割り当ては、
本体の `_assign_speakers` をそのまま使って後から計算できる（モデルを回し直さずに済む）。
"""
from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

from . import pvt_bridge

# 条件名 → (本体の関数, language 引数, 説明)
# built の language: 本体は『検出言語』をそのまま渡し、'zh' 以外だと英語用の埋め込み（NeMo titanet small）を使う。
# 日本語の会議を 'ja' で渡すと英語モデルになるので、'zh'（3D-Speaker ERes2Net large、中国語用）も比べる。
BACKENDS = {
    "built-ja": ("built", "ja", "built（内蔵 sherpa-onnx）言語=ja → 埋め込みは英語用 NeMo titanet small"),
    "built-zh": ("built", "zh", "built（内蔵 sherpa-onnx）言語=zh → 埋め込みは 3D-Speaker ERes2Net large（中国語用）"),
    "cam": ("ali_CAM", None, "ali_CAM（ModelScope の CAM++ 話者分離 speech_campplus_speaker-diarization_common）"),
}
NUM_SPEAKERS = {"n4": 4, "auto": -1}


@contextmanager
def capture_raw_diarization(mod):
    """mod._assign_speakers に渡される話者分離の生の出力を、呼び出しのあいだだけ記録する。"""
    holder: dict = {"raw": None}
    orig = mod._assign_speakers

    def wrapped(subtitles, diarizations):
        holder["raw"] = [[list(d[0]), d[1]] for d in diarizations]
        return orig(subtitles, diarizations)

    mod._assign_speakers = wrapped
    try:
        yield holder
    finally:
        mod._assign_speakers = orig


def run_backend(backend: str, audio: Path, segments_ms: list[list[int]], num_speakers: int, work_dir: Path) -> dict:
    """本体の関数を1回呼ぶ。戻り値: {ok, message, elapsed_sec, labels, raw}。work_dir に subtitles.json / speaker.json を書く。"""
    kind, language, _ = BACKENDS[backend]
    mod = pvt_bridge.speakers_module()
    work_dir.mkdir(parents=True, exist_ok=True)
    subs = work_dir / "subtitles.json"
    spk = work_dir / "speaker.json"
    subs.write_text(json.dumps(segments_ms), encoding="utf-8")
    if spk.exists():
        spk.unlink()
    kw = dict(input_file=str(audio), subtitles_file=str(subs), speak_file=str(spk), num_speakers=num_speakers)
    if kind == "built":
        func = mod.built_speakers
        kw["language"] = language
    else:
        func = mod.cam_speakers
        kw["is_cuda"] = False
    with capture_raw_diarization(mod) as cap:
        t0 = time.perf_counter()
        ok, msg = func(**kw)
        elapsed = time.perf_counter() - t0
    labels = json.loads(spk.read_text(encoding="utf-8")) if ok and spk.is_file() else None
    return {"ok": bool(ok), "message": msg, "elapsed_sec": elapsed, "labels": labels, "raw": cap["raw"]}


def assign_with_pvt(segments_ms: list[list[int]], raw: list) -> list[str]:
    """本体の _assign_speakers で、生の話者区間を字幕の区間に割り当てる（字幕1行ごとの話者名のリスト）。"""
    return pvt_bridge.speakers_module()._assign_speakers(segments_ms, raw)


def write_raw_csv(path: Path, raw: list) -> None:
    """話者分離の生の出力を start,end,speaker（秒）の CSV にする（asrcheck の --diar に渡せる）。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "speaker"])
        for (s, e), spk in sorted(raw, key=lambda r: (r[0][0], r[0][1])):
            w.writerow([f"{s / 1000:.3f}", f"{e / 1000:.3f}", spk])


def srt_time(ms: int) -> str:
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, items: list[dict]) -> None:
    """items: [{start_ms, end_ms, text}] → 字幕ファイル（pyVideoTrans の stt の出力と同じ形）。"""
    lines = []
    for i, it in enumerate(items, start=1):
        lines += [str(i), f"{srt_time(it['start_ms'])} --> {srt_time(it['end_ms'])}", it["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
