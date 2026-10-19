# meetingbench（疑似会議で、話者分離と文字起こしの最初の数字を測る）

PLAN の優先1「P1-2 話者分離だけ試す」用の道具です。**公開データ JVNV（日本語の感情音声、CC BY-SA 4.0）から、4人が順番に話す「疑似会議」を作り、
pyVideoTrans が持っている話者分離（`built` と `ali_CAM`）と文字起こし（SenseVoiceSmall）を動かして、`asrcheck` で採点し、Excel 1枚にまとめます。**
pyVideoTrans 本体のコードは書き換えません（本体の関数をそのまま呼ぶだけ）。

- 会社の会議音声は使いません。使うのは公開データだけです（クラウドに上げてもよい種類）。
- 結果は `outputs/pseudo_meeting_asr.xlsx`。シートは「まとめ」「全指標」「区間の診断」「話者混同」「発言_…」「説明」「実行情報」。
- **注意**: 疑似会議はスタジオ録音の朗読で、雑音・反響・相づち・言い淀みがありません。実際の会議より「やさしい」ので、出た数字は実会議の精度の**上限に近い参考値**です。
  KPI を確定するときは、同意を得た模擬会議の録音（PLAN P1-1）で測り直してください。
- pyannote は Hugging Face の利用同意とトークンが要るため、ここでは測っていません（トークンがあれば pyVideoTrans の設定に入れて別途）。

## 何を測るか

| 条件の軸 | 選択肢 |
|---|---|
| 録音 | A：発言の間があく（重ならない）／B：発言の約12.5%が直前の発言に0.3〜1.0秒かぶさる |
| 字幕の区間の切り方 | 正解の区間（理想）／pyVideoTrans 既定の区切り（VAD=silero、最長5秒）／fsmn-vad（長め） |
| 話者分離 | `built`（言語=ja：英語用の埋め込み）／`built`（言語=zh：中国語用の埋め込み）／`ali_CAM`（CAM++） |
| 人数の指定 | 4人と指定／自動 |

出す指標：DER（collar 0 と 0.25）、話者正解率（発言単位）、CER、cpCER、処理時間・RTF。KPI は PLAN §4.4 の初期案（`asrcheck` と同じ色分け）。
各条件を「字幕単位」（アプリが実際に出す speaker.json と同じ見方）と「生出力」（話者分離モデルが出した話者ごとの区間そのもの）の2通りで採点します。

## 1. 準備（最初に1回だけ）

### 1-1. 疑似会議を作るだけなら（軽い）

`meeting` フォルダで（`uv` の入れ方は `src/asrcheck/README.md` §1-1）：

```
cd meeting
uv sync
```

JVNV は `data/testdata/jvnv/`（`F1/ F2/ M1/ M2/` と `transcription_no_nv.csv`）に置きます。取得：

```
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='litagin/jvnv_corpus_v1_no_nv', repo_type='dataset', local_dir='data/testdata/jvnv')"
```

### 1-2. 測る環境（pyVideoTrans の部品が入った、別の Python 環境。約2GB）

pyVideoTrans 本体と同じ Python 3.10 で、測定専用の環境を別に作ります（本体の環境や `meeting` の環境は壊れません）。

Mac / Linux：

```
uv venv --python 3.10 ~/.venvs/pvt-diar
uv pip install --python ~/.venvs/pvt-diar/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0
uv pip install --python ~/.venvs/pvt-diar/bin/python funasr==1.3.1 modelscope==1.34.0 "sherpa-onnx>=1.13.0" librosa==0.11.0 soundfile==0.13.1 numpy scipy openpyxl requests tqdm huggingface_hub pyyaml addict simplejson sortedcontainers "setuptools<81" datasets==3.0.0 pyarrow==20.0.0 pandas==2.3.0 pillow "ten-vad>=1.0.6.8" faster-whisper pydub
```

（Mac は最初の `--index-url ...` を付けずに `torch==2.6.0 torchaudio==2.6.0` だけでも入ります。Windows は `~/.venvs/pvt-diar/bin/python` を `%USERPROFILE%\.venvs\pvt-diar\Scripts\python.exe` に読み替えます。）

### 1-3. モデルを取得（初回だけ。合計 約1.4GB。ネットが要る）

```
cd meeting
PYTHONPATH=src ~/.venvs/pvt-diar/bin/python -m meetingbench models
```

pyVideoTrans と同じ場所（リポジトリ直下の `models/`）に置きます。pyVideoTrans の画面で一度動かしてあれば、取得済みのものは飛ばします。

## 2. 疑似会議を作る（数十秒）

```
cd meeting
uv run python -m meetingbench build
```

`data/testdata/pseudo_meeting/` に次ができます（乱数の種が同じなら、何度作っても同じ録音になります）。

| ファイル | 中身 |
|---|---|
| `meetingA.wav` / `meetingB.wav` | 16kHz モノラル（A：重ならない、B：約12.5%が重なる）。約32分、4人、発言275件 |
| `ref_meetingA.csv` / `ref_meetingB.csv` | 正解の書き起こし（`start,end,speaker,text` ＋ 感情・元の wav 名）。`asrcheck` がそのまま読める |
| `manifest.json` | 種・設定・各発言の元ファイルと時刻・SHA-256 |

- 1人あたり6〜8分（人ごとに乱数で決める）。同じ文は会議の中で1回しか使いません。感情（怒り・嫌悪・恐れ・喜び・悲しみ・驚き）は話者ごとに均等に混ぜます。
- 48kHz 版も欲しいときは `--keep-48k`（1本約180MB）。種を変えるときは `--seed 数字`、重ねる割合は `--overlap-ratio 0.125`。

## 3. 測る（時間がかかる：CPU 4コアで合計 1〜2時間。途中で止まっても続きから再開）

以降は測定用の環境の Python を使います（`PY` として覚えておくと楽です）。

```
cd meeting
export PYTHONPATH=src
PY=~/.venvs/pvt-diar/bin/python
nice -n 10 $PY -m meetingbench run --stage seg                       # 区間（正解／pyVideoTrans の VAD／fsmn-vad）
nice -n 10 $PY -m meetingbench run --stage asr --threads 2           # 文字起こし（SenseVoiceSmall、CPU）
nice -n 10 $PY -m meetingbench run --stage diar                      # 話者分離（本体の built_speakers / cam_speakers を呼ぶ。条件×録音で約12回）
$PY -m meetingbench run --stage assign                               # 評価用のフォルダに整える
```

`--stage all` で全部まとめて実行もできます。絞りたいとき：`--variant A`、`--backend built-ja cam`、`--num-speakers n4`、`--seg oracle silero`。
結果は `data/testdata/pseudo_meeting/runs/<A|B>/` の下に保存されます（`cond/` の中が、pyVideoTrans の出力と同じ形の `.srt` ＋ `speaker.json`）。

話者分離のモデルが出した「話者ごとの区間」は、本体の関数が最後に捨ててしまうので、呼び出しのあいだだけ `_assign_speakers` を包んで横取りして `raw_diar.json` に保存しています
（本体のファイルは書き換えません）。字幕の区間（ASR で切った区間）への割り当ては、本体の `_assign_speakers` をそのまま使って後から計算するので、モデルを回し直しません
（正解の区間では、本体の関数を実際に呼んだ結果と一致することを `cond/*/meta.json` の `replay_equals_real_call` で確認しています）。

## 4. Excel を作る

```
cd meeting
PYTHONPATH=src ~/.venvs/pvt-diar/bin/python -m meetingbench report --out outputs/pseudo_meeting_asr.xlsx
```

- 「まとめ」：条件ごとの CER・cpCER・DER・話者正解率と KPI の色（緑＝目標達成／黄＝最低ライン達成／赤＝未達）
- 「区間の診断」：字幕に複数の話者が混ざった割合など。話者正解率が上がらない理由の切り分けに
- 「実行情報」：モデルとバージョン、ライセンス、JVNV の出典（CC BY-SA 4.0 の表記）、処理時間、本体を変更していない確認

## 5. 開発者向けメモ

- テスト：`cd meeting && uv run pytest -q src/tests/test_meetingbench_*.py`（モデルのダウンロードなし。合成音声と偽の認識結果で、本体の `_assign_speakers` だけ実物を使う）。
- 構成：`jvnv.py`（索引）、`build.py`（疑似会議）、`pvt_bridge.py`（本体の関数を import。PySide6 が無い環境では空の部品に差し替え）、`model_fetch.py`、`segments.py`（区間・SenseVoice）、`diarize.py`（本体の話者分離を呼ぶ）、`pipeline.py`（段階）、`scoring.py`（asrcheck で採点）、`excel.py`、`reportinfo.py`。
- 本体の `down_file_from_hf` は TLS の検証を切って通信するため使わず、`model_fetch.py` は検証ありで同じファイルを取得します。
- 本体の `built` は、検出言語が `zh` 以外だと英語用の埋め込み（NeMo titanet small）を使います。日本語の会議でも `language='ja'` が渡るので英語用になります。
  `built-zh` 条件は、同じ関数に `language='zh'` を渡して中国語用（3D-Speaker）を試すもので、本体の設定を変える提案ではありません。
