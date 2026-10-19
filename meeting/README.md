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

## 今わかっていること（2026-10-19、公開データで測定。詳しくは docs/research.md 21〜25章）

| 項目 | 今の pyVideoTrans | 改造後 | 目標 |
|---|---|---|---|
| 日本語の文字起こし（文字の誤り） | 2.5〜4.6%（SenseVoice） | — | 8% 以下 |
| 誰が話したか（話者正解率） | 50% | **約94%**（登録した声と照合、`speaker_type=enrolled`） | 95% 以上 |
| 感情の読み取り（4分類） | 9%（SenseVoice） | **77%**（emotion2vec+、`emotion_backend=emotion2vec`） | 70% 以上 |
| 本人らしさ（SIM、本人どうし 0.75） | 0.46 | **0.61〜0.63**（登録の手本） | 測ってから決める |

結果の Excel と聞き比べの音声は `outputs/`（Git には入れない）。

## 次にやること（ユーザー向け）

ここまでの測定・改造は、この開発環境で全部動かしました。**ユーザーにしかできないこと**は次の4つです。

1. **声を登録する（約3分）**：`enroll/script_ja.md` を、静かな部屋で**落ち着いた普通の声**で読み、48kHz・WAV で録音する（ノイズ除去はオフ）。手本の声の調子が合成した声に移るため、普通の声で読むのが大事。
2. **模擬会議を録る**：同意を得た3〜4人で10〜15分（全員、1. の登録もする）。話者の照合と感情の精度を、実際の録音で測り直すため。
3. **社内の確認**：私物の Mac で会社の会議音声を処理してよいか（Q4）、FunASR モデルライセンス（SenseVoice・emotion2vec+）の「参考と学習の目的のみで提供」の解釈（Q5）。
4. （任意）GPU を使いたいとき：Colab（`notebooks/voice_baseline_colab.ipynb`）か、AWS の鍵の登録（`cloud/aws/README.md`、Q13）。CPU でも動くが遅い。
