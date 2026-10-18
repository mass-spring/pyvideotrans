# asrcheck（文字起こしの精度を測る道具）

会議の録音から作った「話者付き文字起こし」が、**どれだけ正確か**を数字にして Excel にまとめる道具です（PLAN の優先1：ローカル高精度 日本語文字起こし）。
人が作った「正解の書き起こし」と、機械の出力を突き合わせて、次を測ります。

| 指標 | 何がわかるか | PLAN §4.4 の目標（初期案） |
|---|---|---|
| **CER**（文字誤り率） | 文字起こしの文字がどれだけ合っているか（日本語） | 最低ライン 15% 以下 / 目標 8% 以下 |
| WER（単語誤り率） | 同じことを英語・フランス語で | （目標は未設定） |
| **cpCER** | 「誰が何を言ったか」を同時に評価した誤り率 | — |
| **DER**（話者分離誤り率） | 「誰がいつ話したか」の誤り率（重なり発話も含む） | 最低ライン 20% 以下 / 目標 15% 以下 |
| **話者正解率**（発言単位） | 発言ごとに、話者が合っていた割合 | 最低ライン 90% 以上 / 目標 95% 以上 |
| 処理時間（1時間の会議） | 速さ | 最低ライン 60 分以内 / 目標 30 分以内（Mac） |
| 完全オフラインで動作 | — | 必須（プログラムでは判定できないので、ネットを切って手で確認） |

- **評価（`eval`）は通信しません。** 会議音声も文字も、パソコンの外には出ません。パソコンだけ（GPU なし）で、Mac と Windows で動きます。
- 目標値はまだ「初期案」です。**まず今の値（ベースライン）を測ってから**、PLAN で確定させます（`--kpi` で目標を差し替えられます）。
- 文字起こしの下書きを機械に作らせる `run` コマンドもあります（任意）。こちらは**初回だけ**モデルのダウンロードでネットが要ります。

---

## 1. 準備（最初に1回だけ）

### 1-1. uv を入れる（voicecheck と同じ）

- **Mac**: ターミナルで `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Windows**: PowerShell で `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

入れたら、ターミナル（PowerShell）を一度閉じて開き直します。

### 1-2. 部品を入れて、動作を確認する

プロジェクトの `meeting` フォルダ（`pyproject.toml` があるフォルダ）に移動して実行します。

```
uv sync
uv run python -m asrcheck --help
```

`eval`（評価）と `run`（下書き作り）の説明が出れば成功です。

> 以降のコマンドは、すべて `meeting` フォルダで、先頭に `uv run python -m asrcheck` を付けて実行します（Mac も Windows も同じ）。
> `pyproject.toml` に `asrcheck` の起動名を足すと `uv run asrcheck ...` と短く書けますが、足さなくても動きます（足し方は §8）。

---

## 2. 正解の書き起こしを作る（いちばん大事な作業）

**正解がこの評価のものさしです。** ここが雑だと、全部の数字が信用できなくなります。同意を得た模擬会議の録音（PLAN の P1-1：3〜4人 × 10〜15分 × 2〜3本）で作ります。

### 2-1. 書き方

Excel に、次の4列の表を作ります（1行＝1人が続けて話した区間。相づちも1行にします）。

| start | end | speaker | text |
|---|---|---|---|
| 0.5 | 4.2 | Aさん | 今日は2025年の予算会議を始めます。 |
| 4.6 | 7.0 | Bさん | よろしくお願いします。 |
| 6.5 | 8.0 | Cさん | えー、お願いします。 |

- 日本語の見出し（`開始` `終了` `話者` `発言`）でも読めます。複数の録音を1つの表にするときは、`file`（`録音`）列を足して録音の名前（例 `mock1`）を書きます。
- **時刻**は、秒（`12.5`）でも、`分:秒`（`1:02.5`）でも、`時:分:秒`（`0:01:02.5`）でも書けます。Excel が自動で時刻の形式に変えても読めます。
- **重なって話した部分**は、2行に分けて、時刻が重なってよい形で書きます（上の例の Bさん 6.5〜 と Cさん）。DER はこの重なりも採点します。
- 句読点は自由に書いてかまいません（比べる前に取り除きます）。数字は「2025」でも「二千二十五」でもかまいません（そろえます）。
- 「えー」「あのー」などの**フィラーを書くか書かないか**は、方針を決めて全部そろえます。書いた場合でも、評価のときに `--fillers drop` を付けると、フィラーを両側から外して測れます（Whisper のように、フィラーを書かないエンジンが多いので、フィラーを書いた正解だと CER が不利に出ます）。
- 話者の名前は、全体で同じ表記にします（`Aさん` と `A` を混ぜない）。

### 2-2. 保存の仕方

**名前を付けて保存 → 「CSV UTF-8（コンマ区切り）」**で保存します（普通の「CSV（コンマ区切り）」でも、Excel のまま `.xlsx` でも読めます。文字コードは UTF-8 / cp932 / UTF-16 を自動で判別します）。

### 2-3. 作る手順（2通り）

**A. 台本があるとき**（模擬会議は台本があるはず）
1. 台本を、話者と発言ごとに分けて Excel の `speaker` と `text` に貼る。
2. 録音を聞きながら、`start` と `end` を入れる（Audacity の「ラベル」機能や、再生ソフトの時刻表示が便利）。
3. **実際に話した言葉に、必ず直す。** 人は台本どおりには話しません（言い直し・言い間違い・アドリブ）。

**B. 機械に下書きを作らせて、人が直すとき**
1. 下書きを作る（§3-2 の `run`。話者も付けるなら `--diarizer pyannote --num-speakers 人数`）。
   ```
   uv run python -m asrcheck run --engine whisper --diarizer pyannote --num-speakers 3 --audio data/mock1.wav --out data/draft_mock1.csv
   ```
2. できた CSV を Excel で開き、**録音を頭から最後まで聞いて**、文字・話者・時刻をすべて直す。
3. `ref_mock1.csv` という名前で保存する。これが正解になる。

> **注意（下書きの偏り）**：下書きを作ったエンジンの間違い方が正解に残りやすく、そのエンジンが有利に評価されます。
> 必ず全部を聞いて直すこと、可能なら下書きを作ったエンジン以外のものも同じ正解で比べること、下書きに使ったエンジンの結果は「参考」と割り引いて見ることをおすすめします。
> 正解作りの所要時間は、実際に1本やって記録してください（録音の長さの何倍かかったかは、この道具の外の話なので、数字を決めていません）。

---

## 3. 認識結果を用意する

評価するのは、次のどれかの形の出力です（複数を同時に比べられます）。

### 3-1. pyVideoTrans の出力（字幕 .srt と speaker.json）

pyVideoTrans を画面なし（CLI）で動かして、話者付きで文字起こしさせます。`pyvideotrans` フォルダで：

```
uv run cli.py --task stt --name "D:/meeting/mock1.wav" --recogn_type 0 --model_name large-v3 --detect_language ja --diarize --num-speakers 3 --speaker-type pyannote
```

- `--diarize`：話者分離を有効にする（従来の `--enable_diariz` と同じ）。`--num-speakers 3`：参加人数（実際の人数。指定すると自動で `--diarize` も有効）。
  `--speaker-type`：`built`（既定）/ `ali_CAM` / `pyannote`。この実行だけに効き、設定ファイルは書き換えません。
- `pyannote` を選ぶには Hugging Face のトークン（pyVideoTrans の「高度なオプション」の `hf_token`、または `models/hf_token.txt`）が要ります。無いと、エラーで止まります（以前は黙って話者分離を飛ばしていました）。
- `--task vtv`（翻訳まで）でも同じ指定で `speaker.json` が出ますが、**吹き替え（`--voice_role`）を指定したときだけ**話者分離が動きます。文字起こしと話者だけが欲しいときは `--task stt` を使ってください。
- 出力は `pyvideotrans/output/mock1-wav/`（ファイル名の「.」は「-」になります。`--output-dir` で変更可）に、字幕（`stt` は `mock1.srt`、`vtv` は `ja.srt` など）と `speaker.json` ができます。

評価のときは、このフォルダを `--hyp` に渡します（`.srt` を直接渡してもかまいません。同じフォルダの `speaker.json` を自動で使います）。

> pyVideoTrans の `speaker.json` は「字幕1行に1人」のラベルです。DER と話者正解率は、その字幕の時間帯で計算します。実際の話者の交代より粗い評価になります（話者の区間そのものを評価したいときは、§3-2 の `.diar.csv` や RTTM を `--diar` に渡します）。

### 3-2. この道具で作る（`run`、任意）

```
uv run python -m asrcheck run --engine whisper --diarizer none --audio data/mock1.wav --lang ja --model large-v3 --out out/mock1.whisper.csv
```

| オプション | 意味 |
|---|---|
| `--engine` | `whisper`（faster-whisper）/ `cohere` / `qwen3asr` / `parakeet` / `sensevoice` |
| `--diarizer` | `pyannote`（community-1。話者付きになる）/ `none`（話者なし） |
| `--num-speakers 3` | 参加人数。分かっているときは**必ず**指定する（分けやすくなる）。人数が不明なら `--min-speakers` `--max-speakers` |
| `--hotwords 用語.txt` | 用語・人名リスト（1行1語）。対応は `whisper` と `qwen3asr` だけ |
| `--model` | モデル名。既定：whisper=`large-v3`、cohere=`CohereLabs/cohere-transcribe-03-2026`、qwen3asr=`Qwen/Qwen3-ASR-1.7B`、parakeet=`nvidia/parakeet-tdt_ctc-0.6b-ja`、sensevoice=`iic/SenseVoiceSmall` |
| `--device` | `auto`（GPU があれば使う）/ `cpu` / `cuda` |
| `--audio` | 録音。複数指定すると、出力の CSV に `file` 列が付く |

できるもの：`out/mock1.whisper.csv`（`start,end,speaker,text`）、`.meta.json`（処理時間など。`eval` が「1時間あたりの処理時間」に換算します）、`--diarizer pyannote` のときは `.diar.csv`（話者の区間そのもの）。

**エンジンごとの入れ方**（必要なものは初めて使うときだけ。`uv run --with ...` を先頭に付けて、その場で一時的に入れます。プロジェクトの設定は変わりません）

| エンジン | 実行例（先頭に付けるもの） | 注意 |
|---|---|---|
| whisper | `uv run --with faster-whisper python -m asrcheck run ...` | CPU は int8 で動く。Mac は CPU（GPU 対応は未確認）。小さいモデルは言葉を取りこぼしやすいので、本番の比較は `large-v3` で |
| cohere | Mac: `uv run --with 'transformers>=5.4' --with torch --with accelerate --with sentencepiece --with librosa python -m asrcheck run ...`（Windows は `'` の代わりに `"`） | 利用前に Hugging Face で同意（gated）。タイムスタンプなし・言語は手動指定。用語リスト非対応 |
| qwen3asr | `uv run --with qwen-asr python -m asrcheck run ...` | 用語は `context` に渡す。中国系モデルの社内可否に注意（PLAN P1-0） |
| parakeet | `pip install 'nemo_toolkit[asr]'` で NeMo を入れる | 公式は Linux / NVIDIA GPU 向け。Mac での動作は公式に記載なし（未確認）。日本語専用 |
| sensevoice | `uv run --with funasr --with torch --with torchaudio python -m asrcheck run ...` | 重みは初回に ModelScope から取得。感情タグが出れば `emotion` 列に残る |
| pyannote（話者分離） | `uv run --with pyannote.audio python -m asrcheck run ...` | 事前に Hugging Face で `pyannote/speaker-diarization-community-1` に同意し、トークンを環境変数に入れる（下） |

pyannote のトークン（Mac：`export HF_TOKEN=hf_xxxx`、Windows PowerShell：`$env:HF_TOKEN="hf_xxxx"`）。一度ダウンロードできたら、`HF_HUB_OFFLINE=1`（Windows：`$env:HF_HUB_OFFLINE="1"`）を付けるとネット無しで動きます。**会社の会議音声は、外部のサービスには送りません**（この道具の処理はすべてパソコンの中です。クラウドで試すのは自分の声・同意を得た模擬会議・公開データだけ）。

> API は各公式ページで確認して書いています（出典 URL と、確認できなかった部分は `runners.py` の冒頭に書いてあります）。

### 3-3. 他のツールの出力

次の形なら、そのまま読めます。

- **CSV / Excel**：正解と同じ列（`start,end,speaker,text`）。話者や時刻は無くてもよい（その分、使えない指標が出ます）。
- **JSON**：`[{"start": 0.5, "end": 2.0, "speaker": "A", "text": "..."}]`、または `{"segments": [...]}`（秒）。`start_time` / `end_time` はミリ秒として読みます。
- **RTTM**（話者の区間だけ）：pyannote の `write_rttm` などの出力。

---

## 4. 評価する（`eval`）

```
uv run python -m asrcheck eval --ref data/ref_mock1.csv --hyp pvt=C:\work\pyvideotrans\output\mock1-wav --hyp whisper=out/mock1.whisper.csv --audio data/mock1.wav --out outputs/asr_report.xlsx
```

（Windows のパスは `C:\...` のまま書けます。スペースがあるときは `"` で囲みます。）

- `--hyp 名前=パス`：評価したい認識結果。**何個でも**並べて比べられます（名前は Excel に出ます）。
- `--audio`：録音。長さを調べて、処理時間を「1時間あたり・分」に換算するために使います（`run` の `.meta.json` があれば省略できます）。pyVideoTrans の実行時間などを自分で測ったときは `--time pvt=1800`（秒）。
- 録音が複数のとき：正解に `file` 列を付け、`--hyp whisper@mock1=out/mock1.csv --hyp whisper@mock2=out/mock2.csv` のように録音名を `@` で付けます（同じ名前のものは合算されます）。
- `--diar 名前=パス`：話者の区間だけの結果（`.diar.csv` や RTTM）。付けると、その名前の DER と話者正解率はこちらで計算します（文字起こしは `--hyp` のものを使います）。
- `--out`：出力の Excel。`--csv` を付けると、サマリーを CSV（UTF-8 BOM 付き）でも出します。

評価の設定（比べ方を変えるとき）：

| オプション | 意味 |
|---|---|
| `--lang ja`（既定）/ `en` / `fr` | ja は CER、en・fr は WER（単語単位） |
| `--collar 0`（既定）/ `0.25` | DER の許容幅（秒）。KPI 判定に使う値。サマリーには 0 と 0.25 の**両方**が載ります |
| `--numbers arabic`（既定）/ `kanji` / `ignore` / `keep` | 数字の書き方の扱い（§6） |
| `--fillers drop` | 「えーと」「あのー」を両側から消して測る |
| `--long-vowel drop` | 長音「ー」を消す（コンピューター/コンピュータ の違いを許す） |
| `--kana-fold` | カタカナをひらがなにそろえる |
| `--kpi cer=0.15/0.08,der=0.2/0.15,spk_acc=0.9/0.95,time=60/30` | 目標値の上書き（最低ライン/目標。cer・der・spk_acc は 0〜1 の割合、time は分） |

---

## 5. Excel の読み方

どの Excel にも「**説明**」シートがあり、全部の列の意味と見方が日本語で書いてあります。迷ったらそこを見てください。「実行情報」シートには、使った設定と入力ファイル、注意が残ります。

1. **「サマリー」**：エンジンごとに1行。上に KPI の表があり、結果のセルが色分けされます。
   緑＝目標達成 / 黄＝最低ライン達成 / 赤＝未達。最後の列「KPI判定」が全体のまとめ（測れた項目だけで判定）。
   「処理時間」が空欄なのは未測定です（`run` の記録か `--time` で入ります）。
2. **「録音ごと」**：録音ごとの CER・DER など。場面や人数による違いを見ます。
3. **「発言_エンジン名」**：正解の発言ごとの比較。**間違えた所に色が付きます**（赤＝置換、赤＋下線＝脱落、青＝挿入）。「要確認」（発言の CER が 30% 以上、または話者が ×）でフィルターすると、人が聞いて確かめる優先順位が付きます。
4. **「話者混同」**：誰を誰と取り違えたかの表（行＝正解の話者、列＝認識側の話者）。対角線に数字が集まっていれば良好。

**数字の読み方の目安**

| 見えたこと | 疑うこと | 次の一手 |
|---|---|---|
| CER が高いが「CER（正規化なし）」との差が大きい | 数字や長音などの書き方の違い | `--numbers` や `--long-vowel` を変えて確認 |
| 脱落が多い | 小声・遠い声の取りこぼし、VAD が音声を切っている | マイクの位置、モデルを大きく |
| 挿入が多い | 無い言葉の作り出し（ハルシネーション）、重なり発話の拾いすぎ | 無音区間の扱い、モデルの選び直し |
| cpCER が CER よりずっと大きい | 文字は合っているが、話者を取り違えている | 人数の指定、声の事前登録 |
| DER の「話者混同」が大きい | 似た声の人を混同 | 参加人数を指定、`--diarizer pyannote` |
| DER の「見逃し」が大きい | 短い相づちを拾えていない | 正解で相づちを別行にしているか確認、collar 0.25 の値も見る |
| 「誤検出」が大きい | 雑音や咳を発言にしている（字幕の区間が実際の発話より長い場合も） | 区間の切り方 |
| 話者正解率（発言単位）が DER より低い | 短い発言を取り違えやすい | 「話者混同」シートで確認 |

---

## 6. 指標の定義と、文字の整え方

**CER** = (置換 + 脱落 + 挿入) ÷ 正解の文字数。正解と認識結果を、それぞれ「開始時刻順に全発言をつないだ1本の文字列」にして、全体で編集距離（レーベンシュタイン距離）を測ります。話者は区別しません（話者込みで見るのが cpCER・話者正解率・DER）。1時間の会議（約2万文字）でも省メモリで計算できる方式（Hirschberg 法）です。

**文字の整え方（正規化）**：正解と認識結果の**両方に同じ処理**をかけます。

1. NFKC（全角英数→半角、半角カナ→全角カナ、互換文字の統一）と小文字化
2. 「%」→「パーセント」
3. 数字：既定は**算用数字にそろえる**（`--numbers arabic`）。「二千二十五年」「２０２５年」→「2025年」、「三万五千円」「3万5千円」→「35000円」、「3,000」→「3000」、「二〇二五」→「2025」（単位の無い並びは桁をそのまま）。
   `kanji`＝漢数字にそろえる / `ignore`＝数字を両側から取り除く（数字の誤りは数えない） / `keep`＝何もしない。
   「一緒」「千代田」のような普通の語も同じ規則で変換されますが、両側に同じ規則をかけるので、一致・不一致の判定は変わりません（文字数だけ少し変わります）。
4. 句読点・括弧・記号・空白・改行をすべて取り除く
5. 長音記号（`ー` `―` `−` `～` など）を「ー」1つにそろえる（かなの直後のみ、連続は1つに）
6. 任意：フィラー除去（`--fillers drop`）、長音除去（`--long-vowel drop`）、カタカナ→ひらがな（`--kana-fold`）

英語・フランス語（WER）：NFKC → 小文字 → 記号を空白に（単語内の `'` は残す、ハイフンは空白）→ 空白で分割。アクセント記号は区別します。

サマリーには「**CER（正規化なし）**」（NFKC と記号除去だけ）も併記するので、正規化でどれだけ変わったかを確認できます。

**cpCER**：話者ごとに発言をつないだ文字列を作り、正解の話者と認識側の話者を、誤りが最小になる1対1に対応づけて、全話者の誤りの合計 ÷ 正解の文字数。話者の数が合わないときは、余った認識側の話者の文字は全部「挿入」、対応のない正解側の話者の文字は全部「脱落」。

**DER** = (見逃し + 誤検出 + 話者混同) ÷ 正解の話している時間。NIST md-eval と同じ数え方で、重なり発話もそのまま数えます（除外しません）。話者名の対応は、一致時間が最大になる1対1（Hungarian 法）。
**collar**：正解の話者区間の始まりと終わりの**前後それぞれ** collar 秒を採点から外します（NIST の `-c` と同じ）。**pyannote.metrics の collar は「前後あわせた幅」なので、pyannote の `collar=0.5` が、ここでの `--collar 0.25` に相当します。** 公開されている pyannote の DER（AMI で 17%）は collar 0 の値です。

**話者正解率（発言単位）**：正解の発言ごとに、その時間でいちばん長く重なった認識側の話者（多数決）を選び、名前の対応を「正解になる発言が最大」になる1対1で決めて（同数なら長い発言を優先）、一致した割合。重なりの無い発言は不正解。サマリーには、発言の長さで重みづけした値も出ます。

---

## 7. 知っておくこと（制限）

- **正解の質が上限です。** 正解の誤字・時刻のずれは、そのまま誤りになります（時刻のずれは collar 0.25 の値で緩和）。
- **下書きの偏り**（§2-3）：機械の出力を直して作った正解は、そのエンジンに有利です。
- **pyVideoTrans の話者は字幕1行に1人**：実際の話者の交代より粗いので、DER は悪めに出ます。
- 発言ごとの差分は、全体を1本に整列した結果を各発言に割り当てたものです。重なり発話の順番が正解と違うと、その周辺がまとめて誤りになります。
- **DER は時間の割合、話者正解率は発言数の割合**です。短い相づちを取り違えても DER はほとんど動きませんが、話者正解率は下がります。
- CER は正解が少ない（千文字未満）と安定しません。10〜15分の録音を2〜3本合算するのがおすすめです。
- 処理時間の KPI は **Mac での値**です。別の機械で測った値は参考にとどめてください。
- `run` の簡易区間分け（`--diarizer none` のとき、whisper 以外）は音量だけで話している区間を探す粗い方式です。静かな部屋の録音向けで、本番の比較には `--diarizer pyannote` の区間を使うことをおすすめします。
- WER・CER の KPI 判定は日本語（CER）だけです。英語・フランス語には目標を決めていません。

---

## 8. 開発者向けメモ

- ソース：`src/asrcheck/`、テスト：`src/tests/test_asrcheck_*.py`（`uv run pytest -q src/tests/test_asrcheck_*.py`）。モデルやネットは使わずに通ります（エンジンの API の呼び方は偽のモジュールで確認）。
- 構成：`textnorm.py`（正規化）、`editdist.py`（編集距離と整列）、`metrics.py`（CER/WER/cpCER）、`diar.py`（DER・話者正解率）、`ioformats.py`（読み込み）、`evaluate.py`（突き合わせ）、`report.py`（Excel）、`kpi.py`（KPI）、`runners.py`（`run`）、`cli.py`。
- 指標の検証：DER と cpCER は、小さな例の手計算に加えて、**総当たり（全順列）の別実装**と乱数で突き合わせています（`test_asrcheck_metrics.py`）。
- `uv run asrcheck ...` と短く書けるようにするには、`pyproject.toml` に次を足します（今は他の作業と衝突しないよう足していません）。
  ```
  [project.scripts]
  asrcheck = "asrcheck.cli:main"

  [tool.hatch.build.targets.wheel]
  packages = ["src/voicecheck", "src/asrcheck"]
  ```
- 想定外のエラーで詳細（Traceback）を見たいとき：環境変数 `ASRCHECK_DEBUG=1` を付けて実行。
- エンジンを足すには、`runners.py` の `Engine` を継承して `transcribe_batch` を書き、`ENGINE_CLASSES` と `DEFAULT_MODELS` に1行ずつ足します。
- 動作確認の例（Linux・CPU 4コア）：公開データ FLEURS 日本語（CC-BY-4.0）の読み上げ文から作った約59秒の疑似会議（2人・6発言）を `faster-whisper small`（int8）で文字起こしし、`eval` まで通した。処理 22.9 秒（モデルの読み込み 13 秒を含む。RTF 0.39）。CER 36%（small は 6発言のうち2つを取りこぼした。本番の比較は large-v3 で）。
- ライセンス：このツールは numpy・scipy・openpyxl だけを使います（GPL 互換）。`run` が使う各モデルのライセンスは、PLAN と research.md を確認してください（Cohere・Qwen3-ASR・Whisper は商用可、parakeet-tdt_ctc-0.6b-ja は CC-BY-4.0、SenseVoice の重みは FunASR 独自、pyannote community-1 は CC-BY-4.0）。
