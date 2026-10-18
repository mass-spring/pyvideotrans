"""`python -m voicelab init` が書き出すひな形（設定ファイルと文の CSV）。"""
from __future__ import annotations

from pathlib import Path

from .texts import write_targets

EXP_YAML = """\
# voicelab 実験の設定（exp.yaml）。# から後ろはコメントです。パスは、このファイルのあるフォルダからの相対パスで書けます。
name: exp01

# 話者。profile: は voiceprofile ツールが作ったフォルダ（profile.json がある）。
# folder: は wav だけを入れたフォルダ（transcripts.csv か、wav と同名の .txt に書き起こし）。どちらか一方を書く。
speakers:
  - name: A
    profile: ../profiles/A
  # - name: B
  #   folder: ../data/enroll/B

targets: targets.csv          # 合成する文（init が作ったひな形を直す）
target_langs: [ja, en, fr, zh]

# 使うエンジン（いらないものは消す）。cosyvoice3 は :zero_shot / :cross_lingual / :instruct2 を選べる。
engines: [voxcpm2, qwen3tts, "cosyvoice3:zero_shot", "cosyvoice3:cross_lingual", chatterbox]

# 条件。line_ref_16k=C0（現状）, profile_ref=C1, stretch=C2（C1 を早回し）, emotion=C3（VoxCPM2 の感情）
conditions: [line_ref_16k, profile_ref, stretch, emotion]
repeats: 2                    # 同じ条件の繰り返し（VoxCPM は実行ごとにぶれる）
seed: 1234
stretch_factors: [1.3, 2.0]
emotion_engines: [voxcpm2]    # 感情の条件に使うエンジン（cosyvoice3:instruct2 / chatterbox も可）

# お試し（数分で終わらせたいとき）：言語ごとの文の数を減らす
# max_texts_per_lang: 1

# エンジンごとの細かい設定（必要なときだけ）
engine_options:
  voxcpm2: {cfg_value: 2.0, inference_timesteps: 10}
  # cosyvoice3: {repo_dir: /content/CosyVoice}
  # chatterbox: {exaggeration: 0.5, cfg_weight: 0.5}

# 測定（voicecheck）
embedder: ecapa               # ecapa / resemblyzer / all / none
# calib: ../outputs/calib.xlsx  # voicecheck calibrate が作った Excel（合格ラインを使うとき）
"""


def write_templates(out_dir: str | Path) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "exp.yaml").write_text(EXP_YAML, encoding="utf-8")
    t = write_targets(d / "targets.csv")
    return [d / "exp.yaml", t]
