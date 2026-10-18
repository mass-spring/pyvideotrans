"""録音から「話している区間 (発話区間)」を見つける。

方法: 10 ms ごとの音量 (voicecheck.features.analyze_frames) から、雑音の床と話し声の大きさを推定し、
その中間にしきい値を置いて「音が出ている区間」を取る (ヒステリシス付き)。
min_pause 秒より短い無音 (言葉の中の促音・息継ぎ・読点) は区間を分けない。
webrtcvad が入っていれば、咳・物音などの「声でない区間」を落とすために使う。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from voicecheck.audio import TARGET_SR
from voicecheck.features import HOP, analyze_frames

FRAME_SEC = HOP / TARGET_SR     # 0.01 秒
MIN_SPREAD_DB = 12.0            # 話し声と雑音の差がこれ未満だと、話している所を判定できない
MIN_LOUD_FRAMES = 3             # しきい値を超えるフレームがこれ未満の山は雑音とみなす
VAD_MIN_SPEECH_FRACTION = 0.15  # webrtcvad が「声」と判定したフレームの割合がこれ未満の区間は声でないとみなす


class NoSpeechError(ValueError):
    """話し声を見つけられない (無音・音が小さすぎる・雑音と区別できない)。"""


@dataclass(frozen=True)
class Segment:
    """発話区間 (音が出ている範囲)。秒。前後の余白は含まない。"""

    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Rejected:
    seg: Segment
    reason: str


@dataclass
class Segmentation:
    segments: list[Segment]
    rejected: list[Rejected]
    frame_db: np.ndarray        # 10 ms ごとの音量 (dBFS)
    noise_db: float             # 雑音の床の推定 (下位 5%)
    speech_db: float            # 話し声の大きさの推定 (上位 10% 点)
    on_db: float                # 「話している」と判定するしきい値
    off_db: float               # 区間を終えるしきい値 (on より少し低い)
    vad_used: str               # "webrtcvad" / "energy"
    notes: list[str] = field(default_factory=list)


def _webrtc_speech_fraction(y16: np.ndarray, seg: Segment, mode: int = 2) -> float | None:
    """区間のうち webrtcvad が声と判定した 30 ms フレームの割合 (判定できなければ None)。"""
    import webrtcvad  # 呼び出し側が ImportError を扱う

    vad = webrtcvad.Vad(mode)
    a, b = int(seg.start * TARGET_SR), int(seg.end * TARGET_SR)
    pcm = (np.clip(y16[a:b], -1.0, 1.0) * 32767.0).astype(np.int16)
    n = int(0.03 * TARGET_SR)
    flags = [vad.is_speech(pcm[i:i + n].tobytes(), TARGET_SR) for i in range(0, len(pcm) - n + 1, n)]
    return float(np.mean(flags)) if flags else None


def detect_segments(
    y16: np.ndarray,
    *,
    min_pause: float = 0.35,
    min_speech: float = 0.25,
    vad: str = "auto",
) -> Segmentation:
    """16 kHz モノラル波形から発話区間を返す。

    vad: "auto" (webrtcvad があれば使う) / "energy" (音量だけ) / "webrtc" (必須: 入っていなければエラー)
    """
    if vad not in ("auto", "energy", "webrtc"):
        raise ValueError(f"vad は auto / energy / webrtc のどれかです: {vad}")
    db = ndimage.median_filter(analyze_frames(y16).db, size=3, mode="nearest")
    noise_db = float(np.percentile(db, 5))
    speech_db = float(np.percentile(db, 90))
    if speech_db < -60.0:
        raise NoSpeechError(f"音が小さすぎて話し声を見つけられません (大きい所でも {speech_db:.0f} dBFS)。録音のレベルを上げて録り直してください")
    if speech_db - noise_db < MIN_SPREAD_DB:
        raise NoSpeechError(
            f"話し声と雑音の差が小さすぎます ({speech_db - noise_db:.0f} dB)。静かな部屋で、口に近いマイクで録り直してください"
        )
    on_db = max(noise_db + 9.0, speech_db - 28.0)
    off_db = on_db - 3.0

    loud = db >= on_db
    labels, _ = ndimage.label(db >= off_db)
    regions: list[list[int]] = []
    for sl in ndimage.find_objects(labels):
        s, e = sl[0].start, sl[0].stop
        if int(loud[s:e].sum()) >= MIN_LOUD_FRAMES:
            regions.append([s, e])
    gap_frames = max(1, int(round(min_pause / FRAME_SEC)))
    merged: list[list[int]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] < gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    notes: list[str] = []
    rejected: list[Rejected] = []
    segs: list[Segment] = []
    for s, e in merged:
        seg = Segment(s * FRAME_SEC, e * FRAME_SEC)
        if seg.dur < min_speech:
            rejected.append(Rejected(seg, f"短すぎる音 ({seg.dur:.2f}秒 < {min_speech}秒): 物音・クリックの可能性"))
        else:
            segs.append(seg)

    vad_used = "energy"
    if vad != "energy":
        try:
            import webrtcvad  # noqa: F401

            kept: list[Segment] = []
            for seg in segs:
                frac = _webrtc_speech_fraction(y16, seg)
                if frac is not None and frac < VAD_MIN_SPEECH_FRACTION:
                    rejected.append(Rejected(seg, f"声ではない可能性 (webrtcvad が声と判定した割合 {frac:.0%}): 咳・物音など"))
                else:
                    kept.append(seg)
            segs, vad_used = kept, "webrtcvad"
        except ImportError:
            if vad == "webrtc":
                raise
            notes.append("webrtcvad が入っていないため、音量だけで発話区間を判定しました (咳・物音を声と誤認することがあります)")
    if not segs:
        raise NoSpeechError("話している区間が見つかりませんでした")
    return Segmentation(segs, rejected, db, noise_db, speech_db, on_db, off_db, vad_used, notes)
