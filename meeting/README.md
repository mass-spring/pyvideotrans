# meeting/ — 会議翻訳アプリ（pyVideoTrans 改造）

会議の録音から「誰が何を話したか」を書き起こし、日・英・仏・中のあいだで直接翻訳し、
**本人の声で・元の感情のトーンで**読み上げるための改造と独自ツールを置くフォルダです。

| 場所 | 内容 |
|---|---|
| `docs/PLAN.md` | 計画・進捗・決定事項（最初に読む） |
| `docs/design.md` | 設計案（声の再現＋感情モジュール） |
| `docs/research.md` | 調査結果と出典 |
| `src/voicecheck/` | 声の分析カルテ：本人の声と合成音声の類似度・声の特徴を測って Excel に出す（使い方は `src/voicecheck/README.md`） |
| `src/voiceprofile/` | 声の登録：約3分の朗読録音から手本（10〜15秒×3〜4本）を自動で選び、pyVideoTrans が使う形で保存 |
| `src/voicelab/` | 声の似かたの比較実験（条件を1つずつ変えて合成→採点→Excel）。Colab 用 |
| `src/asrcheck/` | 文字起こし・話者分離の評価（誤字の率・話者の取り違え・DER を Excel に） |
| `src/meetingbench/` | 公開データ（JVNV）で作る疑似会議で、pyVideoTrans の話者分離（built / ali_CAM）と文字起こしの最初の数字（DER・話者正解率・CER）を測って Excel に（PLAN P1-2。手順は `src/meetingbench/README.md`） |
| `src/emocheck/` | 感情の聞き分け（SenseVoice の感情タグの正解率）と日本語 CER（SenseVoice・Whisper）の現状値を、ラベル付き音声フォルダ（公開の JVNV など）で測って Excel に（手順は `src/emocheck/README.md`） |
| `enroll/script_ja.md` | 登録用の朗読台本（同意文・日本語18文・任意の英仏中）と録音のやり方 |
| `notebooks/voice_baseline_colab.ipynb` | Google Colab で比較実験を動かすノートブック |

個人の声・会議音声・出力（`profiles/`、`data/`、`outputs/`）は Git に入れません。

## 声の分析カルテのすぐ使える手順

```bash
cd meeting
uv sync                     # 初回のみ（必要なものをまとめて入れる）
uv run voicecheck calibrate --enroll-dir 登録音声のフォルダ --out calib.xlsx
uv run voicecheck compare --pairs pairs.csv --calib calib.xlsx --out report.xlsx
```

## 次にやること（ユーザー向け）

1. **声を登録する（約3分）**：`enroll/script_ja.md` のとおりに、静かな部屋で 48kHz・WAV で録音する（ノイズ除去はオフ）。
2. **Colab で現状値を測る**：`notebooks/voice_baseline_colab.ipynb` を Google Colab で開き（GPU を選ぶ）、上から順に実行する。
   録音を登録 → 改造前の条件（C0）と改造後の条件（C1〜C3）で合成 → 声の似かたを Excel で比べる。
   **Colab に上げてよいのは、自分の声・同意を得た模擬会議・公開データだけ。**
3. **模擬会議を録る（優先1用）**：同意を得た3〜4人で10〜15分。正解の書き起こしを作り、`src/asrcheck` で文字起こしの精度を測る。
