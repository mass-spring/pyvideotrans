# emocheck（感情の聞き分けと文字起こしの「現状値」を測る道具）

「ラベル（正解の感情と文章）が付いた音声フォルダ」を入れると、次の2つを測って **1つの Excel** にまとめます。

| 測るもの | 中身 | PLAN での位置づけ |
|---|---|---|
| **感情の正解率** | SenseVoiceSmall が出す感情タグ（HAPPY / SAD / ANGRY / NEUTRAL / FEARFUL / DISGUSTED / SURPRISED / EMO_UNKNOWN）が、正解の感情と合っているか。pyVideoTrans の今の使い方（`sensevoice`）と、EMO_UNKNOWN を禁止した場合（`sensevoice-nounk`）の2通り | §5.3「感情の正解率（平常・喜び・怒り・悲しみの4分類）70%以上」の**現状値** |
| **文字誤り率（CER）** | SenseVoiceSmall と faster-whisper `large-v3-turbo` の日本語の文字起こしが、正解の文章とどれだけ違うか。感情ごとにも出す（感情が強い声で文字起こしが悪くなるか） | 優先1（文字起こし）の目標 CER 8〜15% との比較 |

- 公開の感情音声コーパス **JVNV**（CC BY-SA 4.0、話者4人 × 感情6種）を使う手順を、下に書いてあります。**会社の会議音声は使いません**（公開データ・自分の声・同意を得た模擬会議だけ）。
- 音声は**パソコンの中だけ**で処理します（モデルの初回ダウンロード以外は通信しません）。GPU は要りません。
- 目標値は「今の値を測ってから決める」ための道具です。Excel の数字を見て、PLAN の目標を決めてください。

---

## 1. 準備（最初に1回だけ）

`meeting` フォルダ（`pyproject.toml` があるフォルダ）で実行します。uv の入れ方は `src/asrcheck/README.md` の §1-1 と同じです。

```
uv sync
uv pip install funasr faster-whisper
```

- `uv pip install ...` は、プロジェクトの環境に2つ足すだけです（`uv.lock` は変わりません。torch は入っているものを使うので入れ直しません）。
- 以降は **`uv run --no-sync python -m emocheck ...`** で実行します（`--no-sync` を付けないと、足した2つが消されることがあります）。
- Linux で CPU だけの PC のときは、`uv sync` が軽い CPU 版の torch を入れます（pyproject の設定）。
- ディスクの空きが約 **3 GB** 要ります（SenseVoiceSmall の重み 約 0.9 GB、Whisper large-v3-turbo 約 1.6 GB、ライブラリ）。
- 確認の範囲：`uv pip install --dry-run` で、足すものが衝突しないこと（42 個、torch はそのまま）を確認しました。実際の実行は、同じ構成（Python 3.10・torch 2.14 CPU・funasr 1.4.16・faster-whisper 1.2.1）の別の環境で行いました。

## 2. JVNV を用意する

### 2-1. ダウンロード（約 2.3 GB、公開データ）

```
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('litagin/jvnv_corpus_v1_no_nv', repo_type='dataset', local_dir='data/testdata/jvnv')"
```

`data/testdata/jvnv/` に `F1/ F2/ M1/ M2/`（wav）と `transcription_no_nv.csv` ができます。`data/` は Git に入りません。

### 2-2. ラベル CSV を作る

```
uv run --no-sync python -m emocheck prepare-jvnv --jvnv-dir data/testdata/jvnv --out data/testdata/jvnv_labels.csv
```

感情・話者・読み方（regular / free）は**ファイル名**から、文章は `transcription_no_nv.csv` から取ります。

## 3. 測る

```
uv run --no-sync python -m emocheck run --audio-dir data/testdata/jvnv --labels data/testdata/jvnv_labels.csv --dataset jvnv --models sensevoice sensevoice-nounk whisper --out outputs/jvnv_emotion_asr.xlsx
```

- 結果は `outputs/jvnv_emotion_asr.xlsx` に出ます（Git に入りません）。
- **SenseVoice は全件（1615 件）、Whisper は『話者×感情』ごとに 10 件ずつ（240 件）** を処理します（Whisper は遅いため。`--whisper-subset 0` で全件）。
- `sensevoice-nounk` は、SenseVoice の推論オプション `ban_emo_unk=True`（EMO_UNKNOWN を出させず、ほかの感情タグから選ばせる）で測る設定です。文章は `sensevoice` と同じで、感情タグだけが変わります（CER の表には重ねて載せません）。
- Whisper の重み（large-v3-turbo は約 1.6 GB）が重いときは `--whisper-model small`（約 0.5 GB）に替えられます。その場合は Excel に「large-v3-turbo ではない」と注意が出ます。
- **途中で止めても大丈夫**です。1件ごとに `outputs/emocheck_cache/` へ保存するので、同じコマンドをもう一度実行すると続きから再開します（モデルや設定を変えたときは使い回しません）。
- まず SenseVoice だけ測るなら `--models sensevoice`。あとから `--models sensevoice whisper` で Whisper を足せます（SenseVoice の結果は再利用）。
- 試しに少しだけ動かすなら `--sv-subset 2 --whisper-subset 2`（『話者×感情』ごとに2件 = 48 件）。

主なオプション

| オプション | 意味 |
|---|---|
| `--models sensevoice sensevoice-nounk whisper` | 動かすモデル（既定は sensevoice だけ） |
| `--sv-subset N` / `--whisper-subset N` | 『話者×感情』ごとに N 件だけ処理（0 = 全件。Whisper の既定は 10） |
| `--whisper-model small` | 小さい Whisper に替える（large-v3-turbo は重みが約 1.6 GB） |
| `--sv-language auto` | SenseVoice に渡す言語。既定 `auto`（pyVideoTrans の SenseVoiceSmall は言語を指定しない）。`ja` も試せる |
| `--threads 2` | CPU のスレッド数。ほかの重い処理と同時に動かすときは `1` |
| `--seed 0` | 件を選ぶ乱数の種（同じなら毎回同じ件） |
| `--cache-dir` / `--no-cache` | 途中結果の場所 / 使わない |
| `--data-note "..."` | JVNV 以外のデータを使うときの出典・ライセンスの書き込み |

## 4. 自分のデータで測る（JVNV 以外）

音声フォルダと、次の形のラベル CSV（Excel で作って「CSV UTF-8」で保存）があれば、同じ道具で測れます。**平常（neutral）の音声を入れると、4分類の正解率も測れます。**

```
file,emotion,text,speaker,style
a/001.wav,喜び,今日はいい天気ですね,Aさん,模擬会議1
a/002.wav,平常,資料を共有します,Bさん,模擬会議1
```

- `file`：音声ファイル（`--audio-dir` からの相対パス）。wav / flac / mp3 など
- `emotion`：正解の感情。`neutral happy angry sad fearful disgusted surprised`、または `平常 喜び 怒り 悲しみ 恐れ 嫌悪 驚き`（`anger / fear / disgust / surprise` でも可）
- `text`：正解の文章（空でもよい。空の行は CER に数えません）
- `speaker` / `style`：任意。あると話者ごと・区分ごとの表が出ます

```
uv run --no-sync python -m emocheck run --audio-dir 音声フォルダ --labels labels.csv --models sensevoice --data-note "自分で録音した模擬会議（同意あり）" --out outputs/my_check.xlsx
```

## 5. Excel の読み方

| シート | 何が分かるか |
|---|---|
| **まとめ** | 感情の正解率（全体・3感情）、平常／不明と判定された割合、CER、処理速度。数字の下に「読むときの注意」 |
| **感情の混同表** | 行＝正解の感情、列＝SenseVoice の予測。どの感情をどれと間違えるか（件数と割合） |
| **感情ごとの正解率** | 感情ごとの再現率（95%信頼区間つき）、話者×感情、読み方の区分ごと |
| **文字誤り率** | CER の全体と、感情ごと・話者ごと・区分ごと。置換・脱落・挿入の内訳 |
| **1件ごとの結果** | 1音声ごとの正解・予測・認識結果・CER（フィルターで絞れる） |
| **説明** | 用語（再現率・信頼区間・RTF など）の説明 |
| **実行情報** | モデル名とバージョン、ライセンス、データの出典（CC BY-SA の表示）、実行日時、処理時間、コマンド |

### 数字を読むときの大事な点

- **「不明（EMO_UNKNOWN）」**：SenseVoice は自信がないとき `EMO_UNKNOWN` を出します。pyVideoTrans の `map_sensevoice_tag` は、不明・タグなしを**平常**として扱います。この道具は不明・タグなしを別の列に残し、「平常に読み替えた場合」の割合もまとめに併記します。
- **再現率だけで判断しない**：`sensevoice-nounk` は怒りの再現率が約 100% ですが、これは『怒り』を答えすぎているためです（予測の 6 割が怒り）。**適合率**（その感情と答えた音声のうち、本当にその感情だった割合）も一緒に見ます。
- **JVNV には平常の音声がありません。** だから PLAN の「4分類」の正解率は測れません。測れるのは「平常以外の感情を当てられるか」（まとめの『3感情』は喜び・怒り・悲しみだけを見た値）です。平常の当てやすさ、平常を感情ありと間違える率は、平常の音声を含むデータで別に測ります。
- **JVNV は演技の朗読**です。会議の自然な発言より感情がはっきりしているので、会議での正解率はこれより低くなる可能性があります。
- **95%信頼区間**：件数が少ないほど幅が広くなります。区間が重なる差は「差がある」と言い切れません。
- **処理速度（RTF）** は、ほかの重い処理と同時に動かした CPU の値です。本番 PC の速さの判断には使わないでください。

## 6. 開発者向けメモ

- ソース：`src/emocheck/`、テスト：`src/tests/test_emocheck_*.py`（`uv run pytest -q src/tests/test_emocheck_*.py`）。モデルやネットは使わず、偽のモデルで通ります。
- 構成：`labels.py`（感情の名前・SenseVoice のタグ）、`dataset.py`（ラベル CSV・JVNV・件の抽出・出典）、`runners.py`（SenseVoice / Whisper / 途中保存）、`analysis.py`（正解率・混同表・信頼区間・CER）、`report.py`（Excel）、`cli.py`。
- 文字の整え方（CER）と編集距離は `asrcheck`（`textnorm.py` / `editdist.py`）をそのまま使います。Whisper の呼び出しも `asrcheck.runners.WhisperEngine`（beam_size=5、VAD なし、CPU は int8）です。
- 感情タグの対応表は `videotrans/util/emotion.py` と同じです（`test_emocheck_labels.py` で突き合わせ）。違いは「不明・タグなしを平常にしない」点だけです。
- **funasr の落とし穴**：`AutoModel(..., ncpu=...)` を渡さないと、funasr は torch のスレッド数を 4 に上書きします。ほかの重い処理と同時に動かすと、4 スレッドの待ち合わせで 1 本 30 秒以上かかりました（`ncpu=1` なら約 1 秒）。pyVideoTrans 側の SenseVoice も同じ状況なら遅くなる可能性があります（要検証）。
- SenseVoice は、あらかじめ切った 1 本の wav を VAD なしで渡します（pyVideoTrans の SenseVoiceSmall と同じ使い方）。
- ライセンス：このツールは asrcheck と同じく numpy・scipy・openpyxl・soundfile だけを使います。SenseVoiceSmall の重みは FunASR のモデルライセンス（商用可・出典表記が必要）、Whisper / faster-whisper は MIT。JVNV は CC BY-SA 4.0（Excel の『1件ごとの結果』に JVNV の文章が入るので、社外に出すときは表示と同条件での共有に注意）。
