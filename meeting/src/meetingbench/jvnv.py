"""日本語の公開感情音声コーパス JVNV（CC BY-SA 4.0）の索引。

フォルダの形: <root>/{F1,F2,M1,M2}/<話者>_<感情>_<regular|free>_<番号>.wav と <root>/transcription_no_nv.csv（"キー|本文" の行）。
キー = <感情>_<regular|free>_<番号>（例: anger_regular_01）。同じキーの文は4人全員が読んでいる。
出典: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus
      （非言語音声を除いた版 https://huggingface.co/datasets/litagin/jvnv_corpus_v1_no_nv）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRANSCRIPT_NAME = "transcription_no_nv.csv"


@dataclass(frozen=True)
class Clip:
    speaker: str      # F1 / F2 / M1 / M2
    key: str          # anger_regular_01
    emotion: str      # anger / disgust / fear / happy / sad / surprise
    style: str        # regular / free
    path: Path
    text: str
    duration: float   # 秒（前後の無音を含む元の長さ）


class JvnvError(ValueError):
    """JVNV のフォルダの形が想定と違うとき（メッセージは利用者向けの日本語）。"""


def read_transcripts(root: Path) -> dict[str, str]:
    p = Path(root) / TRANSCRIPT_NAME
    if not p.is_file():
        raise JvnvError(f"{p} が見つかりません。JVNV（非言語音声なし版）のフォルダを指定してください")
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        key, sep, text = line.partition("|")
        if sep and key.strip() and text.strip():
            out[key.strip()] = text.strip()
    return out


def parse_key(key: str) -> tuple[str, str]:
    """'anger_regular_01' → ('anger', 'regular')"""
    parts = key.split("_")
    if len(parts) < 3:
        raise JvnvError(f"キーの形が違います: {key}")
    return parts[0], parts[1]


def load_index(root: Path | str, speakers: list[str] | None = None, read_durations: bool = True) -> list[Clip]:
    """root の下の話者フォルダから、本文のある wav を全部拾う（並びは話者名→ファイル名順で固定）。"""
    import soundfile as sf

    root = Path(root)
    texts = read_transcripts(root)
    clips: list[Clip] = []
    spk_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if speakers:
        spk_dirs = [d for d in spk_dirs if d.name in speakers]
    for d in spk_dirs:
        for wav in sorted(d.glob("*.wav")):
            key = wav.stem[len(d.name) + 1:] if wav.stem.startswith(d.name + "_") else wav.stem
            if key not in texts:
                continue
            emo, style = parse_key(key)
            dur = float(sf.info(str(wav)).duration) if read_durations else 0.0
            clips.append(Clip(d.name, key, emo, style, wav, texts[key], dur))
    if not clips:
        raise JvnvError(f"{root} の下に、本文付きの wav が見つかりません（話者フォルダ F1/F2/M1/M2 と {TRANSCRIPT_NAME} が必要）")
    return clips
