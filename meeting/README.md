# meeting/ — 会議翻訳アプリ（pyVideoTrans 改造）

会議の録音から「誰が何を話したか」を書き起こし、日・英・仏・中のあいだで直接翻訳し、
**本人の声で・元の感情のトーンで**読み上げるための改造と独自ツールを置くフォルダです。

| 場所 | 内容 |
|---|---|
| `docs/PLAN.md` | 計画・進捗・決定事項（最初に読む） |
| `docs/design.md` | 設計案（声の再現＋感情モジュール） |
| `docs/research.md` | 調査結果と出典 |
| `src/voicecheck/` | 声の分析カルテ：本人の声と合成音声の類似度・声の特徴を測って Excel に出す（使い方は `src/voicecheck/README.md`） |

個人の声・会議音声・出力（`profiles/`、`data/`、`outputs/`）は Git に入れません。

## 声の分析カルテのすぐ使える手順

```bash
cd meeting
uv sync                     # 初回のみ（必要なものをまとめて入れる）
uv run voicecheck calibrate --enroll-dir 登録音声のフォルダ --out calib.xlsx
uv run voicecheck compare --pairs pairs.csv --calib calib.xlsx --out report.xlsx
```
