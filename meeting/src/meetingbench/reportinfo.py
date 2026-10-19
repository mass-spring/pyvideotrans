"""Excel の『実行情報』シートに載せる内容（モデル・バージョン・ライセンス・出典・処理時間・環境）を集める。"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from . import model_fetch, pvt_bridge
from .pipeline import Layout

JVNV_ATTRIBUTION = (
    "元データ: JVNV コーパス（日本語の感情音声コーパス）。作成者: Detai Xin, Junfeng Jiang, Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, Hiroshi Saruwatari"
    "（公式ページ https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus で確認）。"
    "文献: \"JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and Nonverbal Expressions\", arXiv:2310.06072 (2023)。"
    "非言語音声を取り除いた版 litagin/jvnv_corpus_v1_no_nv（https://huggingface.co/datasets/litagin/jvnv_corpus_v1_no_nv）を使用。"
    "ライセンス: CC BY-SA 4.0（https://creativecommons.org/licenses/by-sa/4.0/）。\n"
    "改変: 発話の前後の無音を切り、音量（RMS -24 dBFS）をそろえ、順番につないで（A は間をあけ、B は一部を重ねて）16kHz に変換した。"
    "この疑似会議の音声と正解の書き起こしは、元データの派生物として CC BY-SA 4.0（同じ条件での共有）で扱う。"
    "（Git には入れない: meeting/data/ は .gitignore 対象。公開・再配布するときはこの表記を付ける）"
)

PKGS = ["sherpa-onnx", "modelscope", "funasr", "torch", "torchaudio", "faster-whisper", "onnxruntime", "librosa", "soundfile", "numpy",
        "scipy", "openpyxl", "pydub", "ten-vad"]


def pkg_versions() -> str:
    out = []
    for p in PKGS:
        try:
            out.append(f"{p} {metadata.version(p)}")
        except metadata.PackageNotFoundError:
            out.append(f"{p} (未インストール)")
    return ", ".join(out)


def git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "不明"


def collect_info(lay: Layout, command: str, manifest: dict, conds: list, extra_notes: list[str] | None = None) -> list[tuple[str, Any]]:
    from datetime import datetime

    repo = pvt_bridge.ensure_pvt_importable()
    st = pvt_bridge.settings()
    commit = git(["rev-parse", "--short", "HEAD"], repo)
    changed = git(["status", "--porcelain", "--", "videotrans"], repo)
    p = manifest.get("params", {})
    items: list[tuple[str, Any]] = [
        ("ツール", "meetingbench 0.1.0（採点は asrcheck）"),
        ("作成日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("コマンド", command),
        ("pyVideoTrans 本体", f"リポジトリ {repo}（commit {commit}）。本体 videotrans/ の未コミット変更: {'なし' if not changed else changed}。"
                           "本体の関数（built_speakers / cam_speakers / _assign_speakers / get_speech_timestamp_silero / funasr_mlt）を、そのまま呼び出している"),
        ("疑似会議の作り方", f"乱数の種 {p.get('seed')}、1人 {p.get('minutes_min')}〜{p.get('minutes_max')} 分、間 {p.get('gap_min')}〜{p.get('gap_max')} 秒、"
                       f"B は発言の {p.get('overlap_ratio', 0):.1%} が直前に {p.get('overlap_min')}〜{p.get('overlap_max')} 秒かぶさる、"
                       f"音量そろえ RMS {p.get('target_rms_db')} dBFS、前後の無音を {p.get('trim_db')} dB 基準で切る（余白 {p.get('pad_s')} 秒）"),
    ]
    for v, d in manifest.get("variants", {}).items():
        items.append((f"録音 {v}", f"{d['description']}。{d['audio_sec'] / 60:.1f} 分、発言 {d['turns']} 件、重ねた発言 {d['overlapped_turns']} 件（{d['overlap_ratio']:.1%}）、"
                                f"話者ごとの発言時間(分) {d['speech_min_per_speaker']}、感情別の発言数 {d['emotion_turns']}、正解の文字数 {d['ref_chars']}、"
                                f"16kHz wav の SHA-256 {d['sha256_16k'][:16]}…"))
    items.append(("データの出典・ライセンス・表記", JVNV_ATTRIBUTION))
    items += [
        ("話者分離 built（内蔵）", "sherpa-onnx の OfflineSpeakerDiarization。分割: pyannote segmentation-3.0 を onnx にした seg_model.onnx、埋め込み: language=ja なら "
                              "NeMo titanet small（英語用）、language=zh なら 3D-Speaker ERes2Net large（中国語用）、クラスタリング: FastClustering（num_clusters=人数 または -1、threshold 0.5）、"
                              "min_duration_on 0.3 / off 0.5。CPU 1 スレッド（本体の設定のまま）。本体は日本語でも language='ja'（英語用の埋め込み）を渡す。"),
        ("話者分離 ali_CAM", "ModelScope の iic/speech_campplus_speaker-diarization_common（CAM++）。pipeline(task='speaker-diarization')。CPU。人数を渡すと oracle_num として使う"),
        ("pyannote（未測定）", "pyVideoTrans の pyannote 経路は Hugging Face の利用同意（gated）とトークンが必要。この環境にはトークンがないため測っていない"),
        ("区間: pyVideoTrans 既定の区切り", f"videotrans.process.vad.get_speech_timestamp_silero（faster-whisper 同梱の silero VAD）。設定(cfg.json): vad_type={st.get('vad_type')}, "
                                       f"threshold={st.get('threshold')}, min_speech={st.get('min_speech_duration_ms')}ms, max_speech={st.get('max_speech_duration_s')}s, min_silence={st.get('min_silence_duration_ms')}ms"),
        ("区間: fsmn-vad", "FunASR の iic/speech_fsmn_vad_zh-cn-16k-common-pytorch を既定のパラメータで使用（長めの区間の比較用）"),
        ("区間: 正解", "疑似会議の正解の書き起こしの開始・終了時刻をそのまま字幕の区間にした（話者の交代位置が分かっている理想条件）"),
        ("文字起こし", "SenseVoiceSmall（iic/SenseVoiceSmall）を pyVideoTrans の funasr_mlt でそのまま実行（CPU、区間ごとに切り出して前後に 400ms の無音を足す。language は auto）。"
                    "文字のない区間は本体と同じく捨てる"),
        ("省略した条件", "faster-whisper large-v3-turbo（任意項目）は、ディスク空き容量が少ない間はモデルを新しく取得しない方針のため実施していない"),
        ("採点", "asrcheck（CER は NFKC・句読点除去・数字を算用数字にそろえる等の正規化後。DER の collar は 0 と 0.25。話者の対応は Hungarian 法）"),
        ("モデルの置き場所", str(pvt_bridge.models_dir())),
    ]
    for r in model_fetch.model_info():
        items.append((f"モデルファイル {r['file']}", f"{r['bytes']:,} バイト / SHA-256 {r['sha256']}"))
    items += [
        ("モデルのライセンス（要検証を含む）",
         "SenseVoiceSmall・fsmn-vad・CAM++: FunASR / ModelScope の各モデルのライセンス（SenseVoice の重みは FunASR 独自で、出典表記の条件つきで商用可とされる。PLAN・research.md の方針どおり採用前に条文を確認）。\n"
         "内蔵 built の onnx（seg_model / nemo_en_titanet_small / 3dspeaker ERes2Net large）: 配布元は ModelScope himyworld/videotrans（pyVideoTrans 作者の再配布）。"
         "元モデルは pyannote segmentation-3.0（MIT）、NVIDIA NeMo TitaNet（CC-BY-4.0 または NVIDIA の規約。要検証）、3D-Speaker ERes2Net（Apache-2.0。要検証）。再配布元の同一性・条文は未確認 (要検証)。\n"
         "silero VAD: MIT。pyVideoTrans 本体: GPL-3.0（社外へ配布する場合は改造部分も含めソース公開が必要）。"),
        ("ライブラリのバージョン", pkg_versions()),
        ("Python / OS / CPU", f"Python {platform.python_version()} / {platform.platform()} / {cpu_model()} / {os.cpu_count()} コア（GPU なし）"),
        ("計測時の注意", "同じ機械で TTS の生成など別のジョブが動いており、CPU を取り合った状態で測った（load average 約 8〜11 / 4 コア）。処理時間・RTF は参考値で、KPI の基準機（Mac）とは別。"
                    "話者分離の呼び出しは本体どおり CPU 1 スレッド。文字起こしは torch のスレッド数 1。"),
    ]
    for c in conds:
        if c.cfg == "truth":
            continue
        t = c.times
        if "total" in t:
            items.append((f"処理時間 {c.name}", f"区間 {t.get('vad', 0):.0f} 秒 + 文字起こし {t.get('asr', 0):.0f} 秒 + 話者分離 {t.get('diar', 0):.0f} 秒 = {t['total']:.0f} 秒（RTF {t['rtf']:.3f}）"))
    for n in extra_notes or []:
        items.append(("備考", n))
    return items


def load_manifest(lay: Layout) -> dict:
    p = lay.root / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
