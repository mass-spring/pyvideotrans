# voicelab（声の似かた 比較実験）

翻訳した声が、**本人の声にどれだけ似ているか**を測り、改善策を**1つずつ**試して効果を比べるための道具です。

- 音声合成（TTS）のエンジンを同じ条件で動かし、できた音声を [voicecheck](../voicecheck/README.md)（声の分析カルテ）で測って、**1つの Excel** にまとめます。
- 音声を作る部分（生成）だけ GPU が要ります。測る部分（測定）は CPU で動きます。
- 設計の根拠は `docs/design.md` 6〜7章（改善の実験の順番）と `docs/PLAN.md` 5.1b 節（V2 現状値の測定・V3 改善策の比較）です。

> **データの規則（CLAUDE.md）**：クラウド（Colab）に上げてよい音声は、**自分の声／同意を得た模擬会議／公開データ**だけです。会社の会議音声は上げません。

---

## 1. 条件の意味（なぜこの順番か）

「1回に1つだけ変える」ので、隣どうしを比べると**その1つの効果**が分かります。

| 条件 | 何をするか | 何が分かるか |
|---|---|---|
| **C0** `line_ref_16k` | 手本＝本人の実声から取った **2〜4秒** の1本を **16 kHz** にしたもの。上流 pyVideoTrans が「その場の字幕1行」を手本にするのを再現 | **現状値（V2）**。今の方式でどれだけ似るか |
| **C1** `profile_ref` | 手本＝登録した **良い手本**（10〜15秒・元のサンプルレート・書き起こし付き）。他は C0 と同じ | 手本を替えるだけでどれだけ良くなるか（改善策①）。C1 − C0 |
| **C2** `stretch_x1.3` / `stretch_x2.0` | C1 で作った音声を、声の高さを変えずに 1.3倍・2倍の速さにする（rubberband があれば rubberband、無ければ librosa） | 長さ合わせの早回しで声がどれだけ変わるか。stretch − C1 |
| **C3** `emotion_*`（VoxCPM2） | 感情のある文で次の3つを比べる：`emotion_similarity`＝手本＋書き起こしで複製（似かた優先）／`emotion_refonly`＝手本だけ（対照）／`emotion_instruct`＝手本＋感情の指示 | **D13** の判断材料。書き起こしを使う効果（similarity − refonly）と、指示を入れる代償（instruct − refonly） |
| `real` | 本人の実声どうし（1本を取り出し、残りと比べる） | これ以上は期待できない**上限の目安** |

どの条件でも、同じ エンジン・同じ文・同じ繰り返し番号には**同じ乱数の種**を使います（対で比べられる）。
各条件 × エンジン × 言語（ja / en / fr / zh）× 文 を、**繰り返し N 回**（既定 2。VoxCPM は実行ごとにぶれるため）作ります。

**評価の基準（ref）**：その話者の「評価用の実声」のうち、**TTS の手本に使っていないもの**。C0 の手本の元も、登録した手本も、基準には絶対に入れません（プログラムが毎回チェックして、混ざっていたら止まります）。

---

## 2. 比べるエンジン（商用利用できるものだけ）

| 名前 | モデル | ライセンス | 手本の書き起こし | 感情の指示 |
|---|---|---|---|---|
| `voxcpm2` | openbmb/VoxCPM2 | Apache-2.0 | 使う（Ultimate Cloning）。無ければ手本だけで複製 | 文頭の `(指示)`。**指示を使うと書き起こしは使えない（D13）** |
| `qwen3tts` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | Apache-2.0 | 使う。無ければ話者埋め込みだけ | 無し（無視） |
| `cosyvoice3:zero_shot` | Fun-CosyVoice3-0.5B-2512 | Apache-2.0 | 使う。無ければ cross_lingual に切替 | 無し |
| `cosyvoice3:cross_lingual` | 同上 | 同上 | 使わない | 無し |
| `cosyvoice3:instruct2` | 同上 | 同上 | 使わない | 公式の固定の指示（怒・悲・喜・速い・遅い…）。感情の条件だけで使う |
| `chatterbox` | Chatterbox Multilingual（0.1.7） | MIT | 使わない | 強さ（exaggeration）だけ |

使わないもの：F5-TTS・OmniVoice・Higgs Audio・Vevo（重みが非商用）、IndexTTS（大企業は別契約）。設定に書くとエラーで止まります。

公式 API の確認結果（どこまで確かめたか）は各アダプタ（`engines/*.py`）の先頭コメントにあります。**GPU が無い環境では、実際の音声合成は動かしていません**（呼び方は公式の README・ソースと、偽の部品を使ったテストで確認）。

---

## 3. Colab で動かす手順（初級者向け）

1. Google Drive に何も置かなくても、最初の動作確認はできます（公開データを使う）。
2. ブラウザで Colab を開き、ノートブック `meeting/notebooks/voice_baseline_colab.ipynb` を開きます
   （GitHub の `mass-spring/pyvideotrans`、ブランチ `meeting-translator` から。Colab の「ファイル → ノートブックを開く → GitHub」）。
3. メニュー「ランタイム → ランタイムのタイプを変更」で **T4 GPU** を選びます。
4. 上から順にセルを実行します（セルの左の ▶）。**最初は設定を変えず**（`MODE = "quick"`、`SPEAKER_SOURCE = "public_test"`）、全部実行して流れを確認します。
   - 手順 0：注意を読んでチェックを入れる（自分の声／同意済みの模擬会議／公開データだけ）
   - 手順 2：Google ドライブをつなぐ（結果は `マイドライブ/voicelab/実験名/` に保存。Colab が切れても残ります）
   - 手順 5：エンジンごとに別の環境を作る（初回は数分〜10分。**エンジンのライブラリ同士が衝突するため別環境**です）
   - 手順 8-2：ダミーでの確認（GPU・モデル不要）。ここで設定の間違いを先に見つけます
   - 手順 9：生成（GPU）。途中で切れても、同じセルをもう一度実行すれば続きから作ります
   - 手順 10：測定して Excel を作る
   - 手順 11〜12：表を見る・Excel をダウンロード
5. 自分の声で測るとき：`SPEAKER_SOURCE` を変えます。
   - `drive_profile`：voiceprofile ツールで作ったフォルダを `マイドライブ/voicelab/profiles/名前/` に置く
   - `drive_folder`：wav と書き起こしを `マイドライブ/voicelab/enroll/名前/` に置く（下の 4 章）
   - `upload`：セルが出す画面から wav を選ぶ
6. 慣れたら `MODE = "full"`（繰り返し2回・全部の文）にします。

### 時間と GPU の目安（すべて**推定・未実測**。最初の実測で書き換えてください）

| 項目 | 目安 |
|---|---|
| 環境づくり（初回のみ） | 全エンジンで 20〜40 分。CosyVoice3 が最も長く、最も壊れやすい |
| モデルのダウンロード（初回のみ） | エンジンごとに数GB。最初の1件まで数分 |
| 生成（T4）：1件あたり | 5〜20 秒（VoxCPM2 は 4090 で RTF 約0.3＝公式。T4 は数倍遅い見込み） |
| `quick`（4言語×会議文1つ＋感情文1つ） | 約 20〜50 件／エンジン → 生成 10〜20 分 |
| `full`（繰り返し2回・全部の文）、1人 | 約 310 件（5エンジン）→ 生成 1〜2 時間 |
| 測定（CPU） | 音声 1 秒あたり約 0.5 秒（voicecheck の実測：特徴量 0.4＋ecapa 0.08）。`full` の1人で 30 分〜1 時間。`quick` は数分 |
| GPU メモリ | 1つのエンジンだけ読み込み、終わったら解放。VoxCPM2 は約 8GB（公式）。他は T4 16GB に収まる見込み (要検証) |

`python -m voicelab plan` で、作る件数が先に分かります。

---

## 4. 入力データの形

### 4-1. 話者（設定の `speakers:`）

どちらか一方を書きます。

**A. `profile:`** … voiceprofile ツールが作ったフォルダ

```
profiles/Aさん/
  profile.json         {"refs": [{"wav": "refs/ref_ja_01.wav", "text": "...", "lang": "ja", "sec": 14.2, "snr_db": 32.1, "score": 88}, ...],
                        "eval": {"index": "eval/eval.json", "n": 5}}
  refs/ref_ja_01.wav   手本（登録音声）
  eval/eval.json       {"clips": [{"wav": "eval/eval_ja_01.wav", "text": "...", "lang": "ja"}, ...]}
  eval/eval_ja_01.wav  評価用に取り置いた実声（手本に使っていないもの）
```

- 手本のうち、書き起こしがあり、10〜15秒の範囲で、`score`（無ければ `snr_db`）が高い1本が C1 の手本になります（`profile_ref_select: each` にすると全部を1本ずつ試します）。
- 評価用の実声が **4本以上**必要です（C0 の手本の元に1本、基準に2本以上、上限の計算に使う）。

**B. `folder:`** … wav を入れただけのフォルダ

```
enroll/B/
  u0.wav  u1.wav  ...            3〜15秒くらいの本人の発話
  transcripts.csv                列：file,text,lang（任意で role = ref / eval）。無ければ wav と同名の .txt
```

`role` を書かなければ、「書き起こしがあって6秒以上の wav のうち、10〜15秒に近い3本」が手本、残りが評価用になります。

### 4-2. 合成する文（`targets:` の CSV）

`python -m voicelab init --out 作業フォルダ` で、`targets.csv` と `exp.yaml` のひな形ができます。

| 列 | 意味 |
|---|---|
| `id` | 文の名前（重複なし） |
| `lang` | `ja` / `en` / `fr` / `zh` など |
| `kind` | `meeting`（会議風）か `emotional`（感情のある文） |
| `emotion` | `emotional` のとき、TTS に渡す感情の指示（英語。例 `angry, tense, slightly faster`） |
| `text` | 読み上げる文 |

既定には、会議風の文を4言語で3つずつ、感情の文（怒り・喜び・悲しみ）を4言語で1つずつ入れてあります。Excel で直して「CSV UTF-8」で保存できます。

### 4-3. 設定ファイル（`exp.yaml` または `exp.json`）

主な項目（全項目は `python -m voicelab init` のひな形を見てください）。書き間違い（知らない項目名）はエラーで教えます。

| 項目 | 意味 | 既定 |
|---|---|---|
| `speakers` | 話者の一覧（`name` と `profile:` か `folder:`） | 必須 |
| `targets` | 文の CSV | 必須 |
| `target_langs` | 作る言語 | `[ja, en, fr, zh]` |
| `engines` | 使うエンジン | 上の表の5つ |
| `conditions` | `line_ref_16k` / `profile_ref` / `stretch` / `emotion` | 全部 |
| `repeats` | 繰り返し回数 | 2 |
| `stretch_factors` | 早回しの倍率 | `[1.3, 2.0]` |
| `emotion_engines` | 感情の条件に使うエンジン | `[voxcpm2]` |
| `max_texts_per_lang` | お試し用：言語ごとの文の数 | なし |
| `ref_lang` | 手本の言語を決め打ち（例 `ja`） | 空（自動） |
| `max_score_refs` | 基準にする実声の最大本数（多いと測定が遅い） | 8 |
| `embedder` / `calib` / `threshold` | voicecheck の埋め込み・合格ライン | `ecapa` |
| `engine_options` | エンジンごとの細かい設定（`voxcpm2: {cfg_value: 2.0}` など） | なし |

YAML は pyyaml が入っているときだけ読めます（Colab には入っています）。入っていなければ、同じ内容を `.json` で書いてください。

---

## 5. コマンド（パソコンで動かすとき）

`meeting` フォルダで、先頭に `uv run` を付けます（コピペで動く形）。

```
# ひな形を作る
uv run python -m voicelab init --out my_exp

# 何件作るか確認（何も作りません）
uv run python -m voicelab plan --config my_exp/exp.yaml --out outputs/exp01

# モデルなし・GPUなしで、流れだけ確認（ダミー音声。数字に意味はありません）
uv run python -m voicelab run --config my_exp/exp.yaml --out outputs/exp01 --dry-run --embedder none

# 本番（GPU のある環境で）：生成 → 測定 → Excel
uv run python -m voicelab run --config my_exp/exp.yaml --out outputs/exp01

# 生成だけ／測定だけ（Colab ではエンジンごとに別の環境から generate を実行します）
uv run python -m voicelab generate --config my_exp/exp.json --out outputs/exp01 --engines voxcpm2
uv run python -m voicelab score    --config my_exp/exp.json --out outputs/exp01
```

TTS のライブラリ（torch・各エンジン）は `pyproject.toml` に入れていません。実際に生成するときは、Colab のノートブックの手順 5 で入れます。

出力フォルダの中身：

| ファイル | 内容 |
|---|---|
| `voicelab_result.xlsx` | **結果の Excel**（下の 6 章） |
| `pairs.csv` | voicecheck に渡した表（`speaker, ref, test, label=エンジン\|条件\|言語`） |
| `manifest.csv` | 1件ごとの記録（成功・失敗の理由、実際に使った方式 `mode`、手本、乱数の種、生成時間） |
| `audio/` | 生成した音声 |
| `refs/` | 手本のコピー（C0 は 16 kHz、C1 は元のサンプルレート） |
| `engines_info.json` | 使ったライブラリの版 |

`--dry-run` の結果は、エンジン名に `dry:` が付き、Excel の各シートに「ダミー」と明記されます。

---

## 6. Excel の読み方（どの数字が現状値か）

シートの順番：**現状値** → **条件比較** → 特徴量の差 → 実験の説明 → （voicecheck の）行ごと／話者ごと／説明／実行情報 → 実験設定

1. **現状値シート**：`line_ref_16k` の行が**現状値**（V2）です。エンジン × 言語ごとの SIM の平均。`real` の行は本人の実声どうし（上限の目安）。
   `profile_ref` の行は、手本を替えたときの参考値。
2. **条件比較シート**（埋め込みごとに4つの表）
   - **SIM の平均**：緑が高い（似ている）、赤が低い。
   - **SIM のばらつき**：同じ条件のなかの揺れ（標準偏差）。
   - **条件を変えたときの SIM の差**：プラスなら、その条件のほうが似ている。比べる相手は固定です
     （`profile_ref` ← `line_ref_16k`、`stretch_x*` ← `profile_ref`、`emotion_instruct` / `emotion_refonly` ← `emotion_similarity`）。
   - **合格率**：`--calib` / `--threshold` を渡したときだけ。
3. **差がばらつきより小さい**ときは、偶然の揺れかもしれません。繰り返し（`repeats`）や文を増やして確かめます。
4. **SIM の水準は埋め込みごとに違います**（ecapa と resemblyzer で別物）。合格ラインは、登録データで `voicecheck calibrate` を実行して決めます（voicecheck の README 4章）。**日本語の手本 → 英語の合成のように言語をまたぐと、本人の実声どうしでも SIM は下がります**。上限の `real` の行と見比べてください。
5. **特徴量の差**シート：声の高さの差（半音）・声の高さの幅の比・話速の比・音量の差。早回し（stretch）では話速の比が倍率に近づくはずです。
6. **実験設定**シート：設定の一覧、成功・失敗の件数、**実際に使った方式**（例：書き起こしが無くて cross_lingual に切り替わった、など）、失敗の理由。
   → 「条件の取り違え」が起きていないかを、必ずここで確認してください。

目標値は「現状値を測ってから決める」流れです。まず C0 の SIM（と主観評価）を記録し、その上で改善後の目標を置きます。

---

## 7. 困ったとき

| 症状 | 対処 |
|---|---|
| `評価用の実声が足りません` | 手本に使っていない本人の発話を4本以上用意（voiceprofile の `eval/` か、`folder:` の wav） |
| `profile.json が見つかりません` | `profile:` に voiceprofile が作ったフォルダを指定（`profile.json` がある場所） |
| `知らない項目があります` | 設定のタイポ。メッセージに使える項目名が出ます |
| 生成が `failed` | `manifest.csv` の `error` 列を見る。1件の失敗では止まりません |
| エンジンが全件 failed | 読み込みの失敗（ライブラリ未インストール・メモリ不足）。ノートブックの手順 5 を確認 |
| SIM の列が無い | 埋め込み（ecapa）を読み込めませんでした。voicecheck の「実行情報」シートに理由があります |
| 測定が遅い | `max_score_refs` を減らす／`MODE=quick`／pyworld を入れる（入らないと声の高さを librosa.pyin で測るため遅い） |
| Excel が書けない | Excel で開いたままです。閉じて再実行 |
| 中断した | 同じコマンドをもう一度。できている音声は飛ばして続きから作ります（作り直すなら `--force`） |
| 詳しいエラーを見たい | 環境変数 `VOICELAB_DEBUG=1` を付けて実行 |

---

## 8. 開発者向け

- 構成：`config.py`（設定）／`texts.py`（文）／`speakers.py`（手本と評価用の分担）／`conditions.py`（条件 → Job）／`runner.py`（生成）／`scoring.py`（pairs.csv・voicecheck・Excel）／`engines/`（アダプタ）／`cli.py`
- エンジンの共通の形：`synthesize(text, lang, ref_wav, ref_text, emotion=None, *, ref_lang=None, seed=None) -> SynthResult(wav, sr, mode, note)`。重いライブラリは `load()` の中で import します。
- エンジンを足すには：`engines/` にアダプタを書き、`engines/__init__.py` の `_REGISTRY` に1行足します（**商用利用できるものだけ**）。
- テスト：`cd meeting && uv run pytest -q src/tests/test_voicelab_*.py`（CPU・モデル不要。アダプタは公式 API と同じ形の偽の部品で検証）
- 実 GPU でしか確認できないこと：各エンジンの実際の合成・メモリ・速度、Colab での環境づくり（特に CosyVoice3）、日本語のカタカナ化（CosyVoice3）の効果。
