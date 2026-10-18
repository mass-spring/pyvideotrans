"""動作確認用の公開音声 (LibriSpeech 由来、CC-BY-4.0) を話者フォルダ別の wav にする。

使い方 (pyarrow は一時的に足して実行する。プロジェクトの依存には入れない):
    uv run --with pyarrow --with huggingface_hub python src/scripts/prepare_testdata.py

出力 (data/testdata/ の下):
    librispeech_dummy_wavs/<話者ID>/*.wav   ← hf-internal-testing/librispeech_asr_dummy
                                              (73発話だが話者は1人だけ: 1272)
    sv_testclean_wavs/<話者ID>/*.wav         ← DynamicSuperb/SpeakerVerification_LibriSpeech-TestClean
                                              (LibriSpeech test-clean の40人、重複を除いた335発話)
"""
from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2] / "data" / "testdata"


def _save(audio_bytes: bytes, dst: Path) -> None:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, data, sr, subtype="PCM_16")


def dummy() -> None:
    src = Path(snapshot_download(repo_id="hf-internal-testing/librispeech_asr_dummy", repo_type="dataset",
                                 local_dir=ROOT / "librispeech_asr_dummy"))
    out, n = ROOT / "librispeech_dummy_wavs", 0
    for pf in sorted((src / "clean").glob("*.parquet")):
        for row in pq.read_table(pf).to_pylist():
            _save(row["audio"]["bytes"], out / str(row["speaker_id"]) / f"{row['id']}.wav")
            n += 1
    print(f"dummy: {n} 件 -> {out}")


def speaker_verification() -> None:
    src = Path(snapshot_download(repo_id="DynamicSuperb/SpeakerVerification_LibriSpeech-TestClean",
                                 repo_type="dataset", local_dir=ROOT / "sv_librispeech_testclean"))
    out, seen = ROOT / "sv_testclean_wavs", set()
    for pf in sorted((src / "data").glob("*.parquet")):
        for row in pq.read_table(pf).to_pylist():
            for key, akey in (("file", "audio"), ("file2", "audio2")):
                # 例: 6930-75918-0002-1 → 6930-75918-0002 (話者-章-発話)。末尾の -数字 付きは同じ音声の重複なので除く
                name = "-".join(Path(row[key]).stem.split("-")[:3])
                if name in seen:
                    continue
                seen.add(name)
                _save(row[akey]["bytes"], out / name.split("-")[0] / f"{name}.wav")
    print(f"sv_testclean: {len(seen)} 件 -> {out}")


if __name__ == "__main__":
    dummy()
    speaker_verification()
