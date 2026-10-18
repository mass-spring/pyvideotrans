#!/usr/bin/env python3
"""公開データ（LibriSpeech 由来、CC-BY-4.0）の英語話者1人分を、voicelab の folder 形式にする。

ノートブック voice_baseline_colab.ipynb の手順 7（SPEAKER_SOURCE = "public_test"）と同じ処理。
インスタンスの上で、測定用の環境（score）の Python から実行される。自分の声を使わない最初の動作確認用。

  python prepare_public.py --src <リポジトリの meeting/src> --out <出力フォルダ>
出力: <out>/*.wav と <out>/transcripts.csv（列 file,text,lang）
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="meeting/src（scripts/prepare_testdata.py がある場所）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(args.src) / "scripts"))
    import prepare_testdata as ptd   # noqa: E402  (pyarrow・huggingface_hub が要る)
    import pyarrow.parquet as pq     # noqa: E402
    import soundfile as sf           # noqa: E402

    ptd.dummy()                       # 公開音声をダウンロード（話者 1272 の 73 発話）
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for pf in sorted(glob.glob(f"{ptd.ROOT}/librispeech_asr_dummy/clean/*.parquet")):
        for r in pq.read_table(pf).to_pylist():
            data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
            name = f"{r['id']}.wav"
            sf.write(str(folder / name), data, sr, subtype="PCM_16")
            t = r["text"].strip().lower()
            rows.append({"file": name, "text": t[:1].upper() + t[1:] + ".", "lang": "en"})
    if not rows:
        print("公開データが 0 件でした", file=sys.stderr)
        return 1
    with open(folder / "transcripts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "text", "lang"])
        w.writeheader()
        w.writerows(rows)
    print(f"公開データ {len(rows)} 発話を用意しました：{folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
