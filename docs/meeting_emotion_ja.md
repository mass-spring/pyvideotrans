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
| 音声から感情を調べる **emotion2vec+** の検出元（`emotion_backend: "emotion2vec"`） | `videotrans/process/_emotion2vec.py`、`videotrans/task/_stage_emotion.py` |

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
| `emotion_backend` | `sensevoice`（既定）／`emotion2vec`（音声から調べる。下の章）／`none`（使わない） |
| `emotion2vec_model` | `large`（既定。正解率が高い）／`base`（軽い）。`emotion_backend` が `emotion2vec` のときだけ使う |
| `emotion2vec_label_mode` | `core4`（既定。平常・喜び・怒り・悲しみの4つから選ぶ）／`all`（9分類すべてから選ぶ） |
| `emotion2vec_threads` | 計算に使う CPU スレッド数。`0`（既定）= 自動（CPU の数、最大4）。他の重い処理と同時に動かすなら `2` など |

## emotion2vec+ に切り替える（音声から感情を調べる）

SenseVoice のタグは、会議のような平らな話し方ではほとんど `EMO_UNKNOWN`（= 感情なし）になります。**音声そのものから感情を調べる emotion2vec+** に切り替えると、行ごとに必ず結果が出ます。文字起こしのエンジンは何でもかまいません（音声を直接調べるので SenseVoice でなくても動きます）。

**切り替え方**：pyVideoTrans を一度終了し、`videotrans/cfg.json` を次のようにして起動し直します（`emotion2vec_model` と `emotion2vec_label_mode` は書かなければ既定の `large` と `core4` です）。

```json
"enable_emotion": true,
"emotion_backend": "emotion2vec",
"emotion2vec_model": "large",
"emotion2vec_label_mode": "core4"
```

- 初回は、翻訳の直前にモデルを ModelScope から自動で取得します（`models/emotion2vec_plus_large`、約 1.9 GB。`base` は `models/emotion2vec_plus_base`、約 1.1 GB）。SenseVoice などと同じ取得方法で、通信の証明書は検証します。通信できない環境では、`https://modelscope.cn/models/iic/emotion2vec_plus_large/files` の中身をそのフォルダに置いてください。
- 計算には `funasr` と `torch` が要ります（SenseVoice で文字起こしできる環境には入っています）。
- 字幕の1行ごとに、元の音声（作業用フォルダの 16kHz の wav）からその行の部分だけを切り出して調べ、`emotion.json` に書きます。**会議の音声は外部に送りません**（モデルの取得以外は通信しません）。

**large か base か**

| | large（既定） | base |
|---|---|---|
| 4分類の正解率（下の測定） | **76.8%** | 71.9% |
| 速さ（CPU 2スレッド、実時間に対する計算時間） | 0.15〜0.17 | 0.11〜0.13 |
| メモリ（読み込み中のピーク。CPU で測った値） | 約 4.8 GB（読み込み後は約 1.5 GB） | 約 3.0 GB |
| ディスク | 約 1.9 GB | 約 1.1 GB |

1時間の会議で、large は 2スレッドで約 10 分が目安です（話している行だけを調べるので、実際はもう少し短い。要検証）。メモリが足りない PC では `base` にしてください。

**`core4` とは**：emotion2vec+ は9つの分類（怒り・嫌悪・恐れ・喜び・平常・その他・悲しみ・驚き・不明）のスコアを出します。`core4` は、そのうち **平常・喜び・怒り・悲しみの4つ（計画の4分類）だけから、スコアが一番高いものを選びます**（同点なら平常が先）。恐れ・嫌悪・驚きは出ません。9つすべてから選ぶ `all` だと、4分類にない答え（驚きなど）が混ざるため、4分類の正解率は large で 55.9% に下がりました（下の測定）。`all` にすると恐れ・嫌悪・驚きも出ますが、「その他」「不明」が一番になった行は感情なし（平常・信頼度0）にします。

**`emotion.json` の内容**（数字は例。SenseVoice のときと同じ形に、2つの項目を足しています。足した項目は SenseVoice の行には出ないので、従来のファイルはそのまま読めます）

```json
{"start_time": 12340, "end_time": 15880, "label": "angry", "intensity": 0.93, "confidence": 0.93, "source": "emotion2vec",
 "scores": {"angry": 0.93, "disgusted": 0.01, "fearful": 0.0, "happy": 0.0, "neutral": 0.05, "other": 0.0, "sad": 0.01, "surprised": 0.0, "unknown": 0.0},
 "info": "emotion2vec_plus_large core4"}
```

- `confidence` は選んだラベルの**元のスコア**（4つに割り直しません。「恐れ 0.6・悲しみ 0.3」の行は悲しみ 0.3 で、信頼度が低いまま）。`intensity` は同じ値（平常は 0）。
- 翻訳の指示文（`{emotion_info}`）と TTS の感情は、SenseVoice のときと**同じ規則**（平常でなく、信頼度 0.5 以上の行だけ）で使います。
- 0.5 秒より短い行、音声の範囲外の行、モデルが失敗した行は、平常・信頼度0・`source: "none"`（感情ヒントなし）。
- 4つのスコアがすべてほぼ 0 の行（「驚き」などが圧倒的で、4分類のどれでもない）は、同点になって `neutral`・信頼度 0 になります（指示文には載りません）。
- `info` は作り方の控えです。`emotion2vec_model` や `emotion2vec_label_mode` を変えて同じ動画をもう一度翻訳すると、`emotion.json` を作り直します。字幕の統合・分割・時刻修正をしたときも、音声から調べ直します（行が多いと時間がかかります）。

**うまくいかないとき**：モデルが取得できない・`funasr` が無い・音声ファイルが無い（字幕だけを読み込んだ場合）・設定値が違う、のときは、ログに日本語で理由を警告して、**感情ヒントなしで翻訳を続けます**（翻訳は止まりません）。

**測った正解率（4分類）と注意点**（`meeting/docs/research.md` 25章、CPU・公開データ）

| 使い方 | 4分類の正解率（目標 70%） |
|---|---|
| SenseVoice（今の使い方） | 9.0% |
| emotion2vec+ base、4つから最大（`core4`） | 71.9% |
| **emotion2vec+ large、4つから最大（`core4`）** | **76.8%** |
| emotion2vec+ large、9分類から最大（`all`、そのまま） | 55.9% |

- 対象は 1052 件：公開の感情音声 JVNV の喜び・怒り・悲しみ 786 件と、公開の読み上げ音声 FLEURS 日本語の平常 266 件。**会議の音声ではまだ測っていません**（模擬会議で測り直す。要検証）。
- JVNV は演技の感情、FLEURS は別の録音なので、録音の違いを感情と取り違えている分が混ざる可能性があります（FLEURS の女性 30 件を large が全部「悲しみ」と判定した例があり、要検証）。会議の発言はもっと平坦なので、これより低い可能性があります。
- 平常の取りこぼしが多い（平常の再現率は large 48.5%、base 57.5%）。喜びを驚きと取り違える例も多いです。
- 「4つから最大を選ぶ」使い方は、同じデータで他の使い方と比べたあとに選んだので、数字がやや楽観的かもしれません（要検証）。
- この実装の確認：JVNV の 20 件をつなげた音声と字幕で動かし、測定用の道具（emocheck）が同じ 20 件に出した結果と比べました。large・base とも、9分類から最大のラベルは 20/20 件が一致、`core4` は 19/20 件が一致（残る1件は、4つのスコアがすべてほぼ 0 で、どれを選んでも意味がない行）。スコアの数字は、音の取り込み方（サンプリング周波数の変換など）の違いで最大 0.3〜0.4 ずれる行があります。

**ライセンスの注意（未解決）**：emotion2vec+ の重みは、SenseVoiceSmall と同じ **FunASR のモデルライセンス**（[MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)）です。使用・複製・改変・共有は認められ、出典（作者・モデル名）の表示が必要です。商用利用を禁じる条文はありませんが、第3章に「参考と学習の目的のみで提供」という一文があり、**商用利用してよいかは法務確認が済んでいません**（`docs/PLAN.md` の Q5）。そのため `emotion_backend` の既定は `sensevoice` のままです。

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
- 感情を使うのは翻訳（指示文）と、TTS が VoxCPM2・ChatterBox のときの声のトーン（`docs/meeting_emotion_tts_ja.md`）です。
- 指示文は英語のテンプレートに合わせて英語です。ヒントを付けても LLM が従う保証はなく、訳が変わらないことも多いです。
- タグの出力形式（`<|ja|><|HAPPY|><|Speech|>…`）は FunASR の公式説明に基づくもので、**この環境では SenseVoice の実機で未確認**です（模擬文字列でのテストのみ）。
- emotion2vec+ は、法務確認（Q5）が済むまで既定にしません（上の章）。ノイズ除去を使うと、除去後の音声から調べます。話者が重なった行は、混ざった音声で1つの感情になります。

## テスト

```
pytest tests/test_emotion.py tests/test_emotion_translate.py tests/test_emotion_stage.py tests/test_emotion2vec.py
```
