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

## 5. つまずきやすい点（この環境で実際に起きたこと）

- **文字起こしが極端に遅い**：funasr（SenseVoice）はモデルを作るたびに torch のスレッド数を 4 に戻します。他の処理と CPU を取り合う PC では 4 本が空回りして、
  8 発言（68 秒）に約9分かかりました。`run --stage asr` は `--threads`（既定 2）で固定してあり、同じ量が約20秒で終わります（本体のファイルは変えていません）。
- **ModelScope から取得できない（証明書エラー）**：本体は `import` するときに証明書の環境変数（SSL_CERT_FILE など）を certifi のものに書き換えます。
  検査つきプロキシのある環境では通信が失敗するので、`pvt_bridge.py` が import の前後でこれらを元に戻しています。
- **ali_CAM が終わらない**：CPU では（pyVideoTrans は CPU 1 スレッド固定）150 秒の抜粋でも 20 CPU 分以上かかり、約32分の録音は実用になりませんでした。
  GPU または Mac で測ってください。動かすには `hdbscan`・`kaldiio` が要ります。
- **built の言語=zh は遅い**：ERes2Net large を使うため、同じ録音で `ja` の約4〜7倍かかります（この環境で1回40〜60分）。`--backend built-ja` だけにすると約10分です。
- **ディスク**：モデル約1.4GB＋環境約2GB＋疑似会議約120MB。`pytest` の全体実行は一時ファイルを約800MB作るので、空きが少ないときは
  `uv run pytest -q -o tmp_path_retention_policy=failed -o tmp_path_retention_count=0` で実行すると、成功したテストの一時ファイルをすぐ消します。

## 6. 登録した声との照合を試す（PLAN §4.1 ②。`enroll-embed` と `enroll-report`）

built（教師なしクラスタリング）は誰が話しているかを当てられない（§3 の結果）ので、**参加者が事前に登録した声（約3分、D14）に、字幕の区間をいちばん近い人として割り当てる**方式を測ります。
同じ疑似会議・同じ字幕の区間（正解の区間／silero）・同じ文字起こしを使い、built と並べて Excel `outputs/enroll_assign.xlsx` にします（`pseudo_meeting_asr.xlsx` の built の行はそのまま使います）。

- 登録に使う発話は、**会議で使っていない JVNV の発話だけ**（`manifest.json` と照合。会議に使った発話が混ざるとエラー）。登録の長さ（10・30・60・180秒/人）、感情（6種を混ぜる／落ち着いた喜び・悲しみだけ）、
  照合の方式（平均との類似度／登録発話との類似度の最大）、登録発話の選び方5通り（乱数）を変えて、話者正解率・DER・cpCER を測ります。
- 埋め込み（声の特徴のベクトル）は4種。ECAPA（`voicecheck` のものを再利用）、pyVideoTrans が持っている onnx の TitaNet small（英語用）と ERes2Net large（中国語用）、CAM++（任意。onnx 約27MB）。
- 不明（しきい値は登録音声だけから決める）、単純な平滑化、区間の中を1.5秒の窓で判定して話者の交代で切り直す見積もり（`--windows`）も測ります。結果は Excel の『不明と平滑化』シート。
- **注意**: 登録も会議も同じスタジオ録音を同じ処理に通したもので、実会議より有利です。雑音・反響・マイク違いで測り直すまでは上限に近い参考値です (要検証)。

### 手順（コピペで動く形。CPU 4コアで合計 1〜1.5 時間。途中で止まっても続きから再開）

```
cd meeting
export PYTHONPATH=src
PY=~/.venvs/pvt-diar/bin/python
# 1) 埋め込みを作る（モデルを動かす部分。CPU 1 スレッド固定。onnx の2種は測定用環境、ECAPA は meeting の uv 環境で実行）
nice -n 10 $PY -m meetingbench enroll-embed --embedder titanet --windows        # 軽い（約5分）。--windows は区間の切り直しの見積もり用
nice -n 10 $PY -m meetingbench enroll-embed --embedder eres2net                 # 重い（RTF 約0.13、約20分）。--windows は付けない（さらに約20分）
nice -n 10 uv run python -m meetingbench enroll-embed --embedder ecapa --windows  # 約25分
# 任意: CAM++（onnx を取得。TLS は検証あり）
mkdir -p data/models && curl -fL -o data/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
nice -n 10 $PY -m meetingbench enroll-embed --embedder campp --windows
# 2) 採点して Excel を作る（1 条件 約1秒、初回は 1520 条件で約30分。結果は runs/enroll/results.jsonl に保存され、2回目以降は約1.5分）
nice -n 10 $PY -m meetingbench enroll-report --out outputs/enroll_assign.xlsx
```

- 埋め込みは `data/testdata/pseudo_meeting/runs/enroll/emb_<名前>.npz`（Git に入れない）。条件を変えて採点し直すだけなら 2) だけで、数秒〜数分です。
- `enroll-report` の主なオプション：`--embedder titanet`（絞る）、`--secs 30 180`、`--seeds 3`、`--force`（採点し直し）。
- 出力：『まとめ』（built と並べた表と KPI の色）、『全指標』（条件×seed ごとの asrcheck の全列）、『登録時間ごとの比較』（表とグラフ）、『話者混同』、『不明と平滑化』、『説明』、『実行情報』（モデル・バージョン・ライセンス・JVNV の出典・処理時間と RTF）。
- 採点の近道：CER は話者に依らないので区間ごとに1回だけ計算し、cpCER・DER・話者正解率を条件ごとに計算します（asrcheck の標準と同じ値になることをテストで確認）。
  asrcheck は話者名の対応づけを自動で最適化するので、名前をそのまま比べた話者正解率（『全指標』の列）も並べて、名前の取り違えで有利になっていないことを確認します。

## 7. 開発者向けメモ

- テスト：`cd meeting && uv run pytest -q src/tests/test_meetingbench_*.py`（モデルのダウンロードなし。合成音声と偽の認識結果で、本体の `_assign_speakers` だけ実物を使う）。登録音声との照合は `test_meetingbench_enroll.py`（偽の埋め込み器で、埋め込み→採点→Excel まで通す）。
- 構成：`jvnv.py`（索引）、`build.py`（疑似会議）、`pvt_bridge.py`（本体の関数を import。PySide6 が無い環境では空の部品に差し替え）、`model_fetch.py`、`segments.py`（区間・SenseVoice）、`diarize.py`（本体の話者分離を呼ぶ）、`pipeline.py`（段階）、`scoring.py`（asrcheck で採点）、`excel.py`、`reportinfo.py`。登録音声との照合：`enroll_assign.py`（選び方・照合・しきい値・平滑化。numpy だけ）、`enroll_embed.py`（埋め込み器とその実行）、`enroll_eval.py`（asrcheck で採点）、`enroll_excel.py` / `enroll_report.py`（Excel）。
- 本体の `down_file_from_hf` は TLS の検証を切って通信するため使わず、`model_fetch.py` は検証ありで同じファイルを取得します。
- 本体の `built` は、検出言語が `zh` 以外だと英語用の埋め込み（NeMo titanet small）を使います。日本語の会議でも `language='ja'` が渡るので英語用になります。
  `built-zh` 条件は、同じ関数に `language='zh'` を渡して中国語用（3D-Speaker）を試すもので、本体の設定を変える提案ではありません。
