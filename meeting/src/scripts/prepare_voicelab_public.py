"""voicelab 用の「公開データの話者フォルダ」を作る（自分の声を録る前の動作確認・現状値の測定用）。

作るもの（voicelab の folder 方式：wav ＋ transcripts.csv（列 file,text,lang,role））
  <出力>/jvnv_F1/ など … 日本語の公開感情音声 JVNV（CC BY-SA 4.0）の話者
  <出力>/libri1272/     … 英語の LibriSpeech（CC BY 4.0）話者 1272（prepare_testdata.dummy() の parquet から）

JVNV の役割の決め方（再現できるように固定）
  手本（ref）: 喜び・悲しみの regular から、10〜15 秒に近い順に 3 本（怒りなど強い感情は手本にしない）
  評価（eval）: 残りの regular から、感情ごとに同じ数（既定 4 本 × 6 感情）。長さ 4 秒以上
  JVNV には「平常」の感情がないので、評価の実声も感情のある声になる（結果の注意書きに書くこと）。

使い方（meeting フォルダで）
  uv run --with pyarrow python src/scripts/prepare_voicelab_public.py --out data/voicelab/speakers
出典: JVNV https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus
      （非言語音声を除いた版 https://huggingface.co/datasets/litagin/jvnv_corpus_v1_no_nv）
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
JVNV = ROOT / "data" / "testdata" / "jvnv"
LIBRI = ROOT / "data" / "testdata" / "librispeech_asr_dummy"
EMOTIONS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
REF_EMOTIONS = ("happy", "sad")


def _dur(p: Path) -> float:
    i = sf.info(str(p))
    return i.frames / i.samplerate


def jvnv_texts(root: Path = JVNV) -> dict[str, str]:
    out = {}
    for line in (root / "transcription_no_nv.csv").read_text(encoding="utf-8").splitlines():
        if "|" in line:
            k, t = line.split("|", 1)
            out[k.strip()] = t.strip()
    return out


def plan_jvnv(speaker: str, root: Path = JVNV, n_ref: int = 3, eval_per_emotion: int = 4,
              eval_min_s: float = 4.0) -> list[dict]:
    """JVNV の1人分の (file, text, lang, role) を決める。"""
    texts = jvnv_texts(root)
    clips = []
    for p in sorted((root / speaker).glob(f"{speaker}_*_regular_*.wav")):
        key = p.stem.split("_", 1)[1]          # anger_regular_01
        if key in texts:
            clips.append((p, key.split("_")[0], texts[key], _dur(p)))
    cand = [c for c in clips if c[1] in REF_EMOTIONS and 10.0 <= c[3] <= 15.0] \
        or [c for c in clips if c[1] in REF_EMOTIONS]
    refs = sorted(cand, key=lambda c: (abs(c[3] - 12.0), c[0].name))[:n_ref]
    used = {c[0] for c in refs}
    rows = [{"file": c[0].name, "text": c[2], "lang": "ja", "role": "ref"} for c in refs]
    for emo in EMOTIONS:
        pool = [c for c in clips if c[1] == emo and c[0] not in used and c[3] >= eval_min_s]
        for c in pool[:eval_per_emotion]:
            rows.append({"file": c[0].name, "text": c[2], "lang": "ja", "role": "eval"})
    return rows


def write_folder(out: Path, rows: list[dict], src_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        shutil.copy2(src_dir / r["file"], out / r["file"])
    with open(out / "transcripts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text", "lang", "role"])
        w.writeheader()
        w.writerows(rows)


def libri_folder(out: Path, root: Path = LIBRI) -> int:
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pf in sorted(root.glob("clean/*.parquet")):
        for r in pq.read_table(pf).to_pylist():
            data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
            name = f"{r['id']}.wav"
            sf.write(out / name, data, sr, subtype="PCM_16")
            t = r["text"].strip().lower()
            rows.append({"file": name, "text": t[:1].upper() + t[1:] + ".", "lang": "en"})
    with open(out / "transcripts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text", "lang"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="voicelab 用に公開データの話者フォルダを作る")
    ap.add_argument("--out", default=str(ROOT / "data" / "voicelab" / "speakers"))
    ap.add_argument("--jvnv", nargs="*", default=["F1", "M1"], help="JVNV の話者（F1 F2 M1 M2）")
    ap.add_argument("--no-libri", action="store_true", help="LibriSpeech の話者を作らない")
    ap.add_argument("--eval-per-emotion", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    for spk in a.jvnv:
        rows = plan_jvnv(spk, eval_per_emotion=a.eval_per_emotion)
        write_folder(out / f"jvnv_{spk}", rows, JVNV / spk)
        n_ref = sum(r["role"] == "ref" for r in rows)
        print(f"jvnv_{spk}: 手本 {n_ref} 本・評価 {len(rows) - n_ref} 本 → {out / f'jvnv_{spk}'}")
    if not a.no_libri:
        if not LIBRI.is_dir():
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            import prepare_testdata
            prepare_testdata.dummy()
        n = libri_folder(out / "libri1272")
        print(f"libri1272: {n} 発話 → {out / 'libri1272'}")


if __name__ == "__main__":
    main()
