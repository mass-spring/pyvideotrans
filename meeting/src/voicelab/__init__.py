"""voicelab: 声の似かた（本人らしさ）の比較実験を回す道具。

TTS エンジン（VoxCPM2・Qwen3-TTS・CosyVoice3・Chatterbox）を同じ条件で動かし、
合成音声を voicecheck（声の分析カルテ）で測って、1つの Excel にまとめる。

- 条件は 1 回に 1 つだけ変える（design.md 6章）。
- GPU が要るのは「生成」だけ。「測定（score）」は CPU で動く。
- 重いライブラリ（torch・各 TTS）は、使う時まで import しない。
"""

__version__ = "0.1.0"
