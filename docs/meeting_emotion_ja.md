# 感情モジュール（第1部）：発言ごとの感情を翻訳の指示文に渡す

pyVideoTrans の改造メモ（本家の動作は、設定を OFF にしたままなら変わりません）。

## 追加したもの

| 内容 | 場所 |
|---|---|
| SenseVoice の感情タグ（`<\|HAPPY\|>` など）を、捨てる前に控える | `videotrans/process/stt_funasr.py`、`videotrans/process/_stt_utils.py` |
| 感情データ（`emotion.json`）の形・読み書き・時刻での対応付け | `videotrans/util/emotion.py` |
| 翻訳の直前に `emotion.json` を用意する工程（`EmotionMixin`） | `videotrans/task/_stage_emotion.py` |
| 翻訳の指示文に「行ごとの感情の表」を差し込む | `videotrans/prompts/emotion/{srt,text}.txt`、`videotrans/translator/_base.py` |
| 全ての指示文テンプレートに `{emotion_info}` を追加（36ファイル） | `videotrans/prompts/{srt,text}/*.txt` |

## 使い方

1. pyVideoTrans を一度終了し、`videotrans/cfg.json` の次の2行を変える（画面のスイッチはまだありません。一度起動していれば `"enable_emotion": false` の行が既にあります。無ければ足します）。

```json
"enable_emotion": true,
"emotion_backend": "sensevoice"
```

2. 文字起こしのエンジンに **SenseVoiceSmall**、翻訳に **AI 翻訳**（ChatGPT 互換・Gemini・百炼の qwen-mt 以外・ローカル LLM など）を選んで実行する。

| 設定 | 意味 |
|---|---|
| `enable_emotion` | `false`（既定）なら何も変わらない。`true` で感情を使う |
| `emotion_backend` | `sensevoice`（既定）／`none`（使わない）／`emotion2vec`（**未実装**、選ぶと感情なしで続行） |

## できるファイル

- `emotion.json`：作業用フォルダと出力フォルダ（`speaker.json` と同じ場所）に作られる。最終の原文字幕1行につき1件。
  `{"start_time": 12340, "end_time": 15880, "label": "angry", "intensity": 0.5, "confidence": 0.6, "source": "sensevoice"}`
- 感情は7種：`neutral, happy, angry, sad, fearful, disgusted, surprised`。対応が取れない行は `neutral`（信頼度0、`source: none`）。
- 字幕を編集しても、行番号ではなく**開始時刻（±300ms）**で対応付けるのでずれません。
- もう一度翻訳しても同じ結果です（作り直しは、字幕の行が変わったときだけ）。

## 指示文での見え方

「信頼度0.5以上で、平常でない行」だけを表にして、`{context_info}` の直後に入れます。
SRT で送る設定のときは SRT の番号、行ごとのテキストで送る設定のときは**そのまとまりの中での位置（1から）**が番号です。

```
# EMOTION HINTS (OPTIONAL, READ-ONLY)
...
| Line | Emotion | Intensity |
| --- | --- | --- |
| 2 | angry | 0.5 |
| 4 | happy | 0.5 |

### RULES FOR USING THESE HINTS:
1. Use the hints ONLY to choose wording, register (tone of voice) and interjections ...
2. Do NOT add emotions that are not present in the source text ...
3. Do NOT output the table, the emotion labels, ...
4. Do NOT change the number of blocks, the Index Numbers or the timestamps ...
```

ヒントが1行も無いとき・OFF のときは空文字になり、指示文は従来と同じです。翻訳キャッシュのキーにはヒントも含めるので、感情の有無で結果が混ざりません。

## 限界（今の段階）

- **強さと信頼度は暫定の固定値**：SenseVoice は確率を出さないため、平常以外は強さ0.5・信頼度0.6、平常は強さ0・信頼度0.6。音の大きさや高さから求める本当の強さは次の段階。
- SenseVoice の感情は、会議のような平らな話し方ではほとんど `neutral` になる見込みです（要検証。模擬会議で正解率を測ってから判断）。
- 感情が取れるのは SenseVoiceSmall で文字起こししたときだけ。他のエンジンや、外から読み込んだ字幕では `emotion.json` は作られず、感情なしで翻訳されます。
- 感情を使うのは翻訳（指示文）だけ。声のトーン（TTS）への反映は次の段階です。
- 指示文は英語のテンプレートに合わせて英語です。ヒントを付けても LLM が従う保証はなく、訳が変わらないことも多いです。
- タグの出力形式（`<|ja|><|HAPPY|><|Speech|>…`）は FunASR の公式説明に基づくもので、**この環境では SenseVoice の実機で未確認**です（模擬文字列でのテストのみ）。
- 音声ファイルの感情分析モデル（emotion2vec+）は、ライセンスの法務確認が済むまで入れません。

## テスト

```
pytest tests/test_emotion.py tests/test_emotion_translate.py tests/test_emotion_stage.py
```
