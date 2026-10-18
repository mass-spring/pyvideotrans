# 感情モジュール（第2部）：発言の感情を声のトーンに反映する

pyVideoTrans の改造メモ（本家の動作は、設定を OFF にしたままなら変わりません）。
第1部（`docs/meeting_emotion_ja.md`）で作った行ごとの感情（`emotion.json`）を、**音声合成（TTS）にも渡します**。

## 1. 何ができるか

| 配音エンジン | 感情の扱い | 状態 |
|---|---|---|
| **VoxCPM（v2 / hf）** | 感情の種類と強さを、英語の短い指示（例：`very angry, tense`）にして渡す | 対応 |
| **ChatterBox** | 感情の**種類は指定できない**。強さ（`exaggeration`）だけ行ごとに変える | 対応（強さのみ） |
| CosyVoice | 今の画面（Gradio webui）では「声の複製＋感情の指示」ができない | **非対応**（6章に調査結果と提案） |
| VoxCPM v1 | v1 の呼び出しには指示の欄がない | 非対応 |
| IndexTTS | 感情を別入力にできるが、ライセンス（大企業は別契約）の確認待ち | 未対応（使わない） |
| GPT-SoVITS、Qwen3-TTS（Base）ほか | 手本の音声だけで、感情の指示口がない（調査済みなのは下の注） | 未対応 |
| F5-TTS・OmniVoice・Higgs Audio | 重みが非商用ライセンス | 使わない |
| Edge-TTS・各クラウド API | この改造では何も渡さない（クラウドへは会社の音声を送らない方針） | 未対応 |

注：GPT-SoVITS と Qwen3-TTS は `docs/research.md` 11.3 の調査結果（手本のみ／複製と指示の併用は不可）。それ以外のエンジンの感情指定の有無は未調査（要検証）。
対応していないエンジンでは、感情があっても**何も変わりません**（配音の使い回し＝キャッシュも壊しません）。

## 2. 使い方

1. pyVideoTrans を終了し、`videotrans/cfg.json` に次の行があることを確認する（無ければ足す。一度起動すれば `emotion_tts` などは自動で入る）。

```json
"enable_emotion": true,
"emotion_backend": "sensevoice",
"emotion_tts": true,
"chatterbox_crosslingual_cfg0": true
```

2. 文字起こしは **SenseVoiceSmall**（第1部のとおり。これで `emotion.json` ができる）。
3. 配音は **VoxCPM**（バージョンは v2）か **ChatterBox**、声は **clone** を選ぶ。
4. 実行して、`logs/` の最新のログで次の行を探す。

```
emotion for TTS: 3/12 lines (tts_type=15, engine uses it: True)
```

左の数字が「感情が付いた行数」（平常・信頼度0.5未満・対応が取れない行は数えない）、`engine uses it: True` なら VoxCPM か ChatterBox です。

| 設定（`cfg.json`） | 既定 | 意味 |
|---|---|---|
| `enable_emotion` | `false` | 第1部と共通。`false` なら感情は何にも使われない |
| `emotion_tts` | `true` | `enable_emotion` が `true` のとき、感情を声にも反映する。`false` なら翻訳の指示文だけに使う |
| `chatterbox_crosslingual_cfg0` | `true` | ChatterBox で、手本の言語と訳す言語が違うとき `cfg_weight` を 0 にする（`enable_emotion` と `emotion_tts` が `true` のときだけ有効） |

ChatterBox の元の設定（`chatterbox_exaggeration`、`chatterbox_cfg_weight`）は、これまでどおり画面で変えます。それが「基準値」になります。

## 3. どう動くか（データの流れ）

1. 配音の直前（`task/_stage_dubbing.py`）に、各行の**原文字幕の開始時刻**で `emotion.json` を引く（±300ミリ秒。行番号だけでは引かない）。
2. 平常でなく、信頼度が0.5以上の行だけ、配音の1行分のデータに `emotion` / `emotion_intensity` / `emotion_confidence` を足す。それ以外は何も足さない（本家と同じ）。
3. 配音の使い回し（キャッシュ）のキーに、VoxCPM・ChatterBox のときだけ「感情＋強さ」を足す。感情の付いた行だけキーが変わり、他の行や他のエンジンのキーは変わらない。
4. エンジンが行ごとのデータを読んで、指示や強さに変える（`util/emotion_tts.py` が変換の本体）。

`emotion.json` は作業用フォルダに無ければ、出力フォルダのコピーを使います（配音だけやり直すとき用）。

## 4. VoxCPM：指示の渡し方

### 4.1 公式の仕様（確認したこと）

- VoxCPM2 は、**テキストの先頭に丸かっこで英語（または中国語）の指示**を付けて、声の複製と一緒に使えます。公式の例：
  `text="(slightly faster, cheerful tone)This is a cloned voice with style control.", reference_wav_path="speaker.wav"`
  出典：[VoxCPM2 モデルカード](https://huggingface.co/openbmb/VoxCPM2)、[VoxCPM README](https://github.com/OpenBMB/VoxCPM)
- pyVideoTrans がつなぐ Gradio の画面（公式 `app.py`、API 名 `/generate`）には、**`control_instruction` という独立した入力欄**があり、画面側で `(指示)本文` に組み立てます（指示の中の丸かっこは取り除く）。出典：[app.py](https://github.com/OpenBMB/VoxCPM/blob/main/app.py)。`hf` を選んだときの入力名は Hugging Face の [VoxCPM デモ](https://huggingface.co/spaces/openbmb/VoxCPM-Demo) と同じで、そちらも同じ規則（`use_prompt_text` が `True` なら指示を使わない）。**`hf` は外部のデモ画面を指すことがあるので、会社の音声には使わない**
- **重要**：画面の「Ultimate Cloning（書き起こしを使う複製）」を ON にすると、**指示は捨てられます**（`app.py` が `use_prompt_text` のとき指示を空にする。画面の説明にも "Control Instruction will be disabled when active" とある）。
- 指示は英語・中国語に対応（[Cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html)）。語彙の決まった一覧は無く、「複雑な指示を足しすぎない」ほうがよいとされる。
- 公式の注意：Voice Design / Style Control は**実行のたびに結果が変わる**（1〜3回作り直すとよい）。

### 4.2 この改造でやること（`tts/_voxcpm.py`、`util/emotion_tts.py`）

感情の付いた行だけ、次のリクエストになります（v2 の例）。

| 項目 | 感情なし（本家と同じ） | 感情あり |
|---|---|---|
| `control_instruction` | `''` | 例：`very angry, tense` |
| `use_prompt_text` | 手本の書き起こしがあれば `True` | **`False`**（指示を有効にするため） |
| `text` | 訳文そのまま | 訳文そのまま（かっこは画面側が付ける） |

感情→英語の短い指示：

| 感情 | 指示 | 感情 | 指示 |
|---|---|---|---|
| angry | `angry, tense` | fearful | `anxious` |
| happy | `cheerful` | surprised | `surprised` |
| sad | `sad, subdued` | disgusted | `disgusted, displeased` |

強さ→言葉：**0.4未満 `slightly`（例：`slightly cheerful`）／0.4〜0.7 なし／0.7より大きい `very`（例：`very angry, tense`）**。
第1部の暫定値（強さ0.5）のままだと言葉は付かず、`angry, tense` のようになります。

### 4.3 トレードオフ（必ず知っておく）

感情の付いた行は「手本の書き起こしを使う複製」ができないため、**手本の音声だけで複製**（Controllable Cloning）になります。声が本人に似る度合いが、その行だけ少し下がる可能性があります（数値は未測定。E6 の測定器で「感情あり／なし」を比べて確認する）。平常の行は従来どおりです。

## 5. ChatterBox：強さの渡し方

出典：[Chatterbox README の Tips](https://github.com/resemble-ai/chatterbox)（`exaggeration` と `cfg_weight` の説明、言語が違うときは `cfg_weight=0`）。

- `exaggeration`（表現の強さ）：感情なしの行は画面の値のまま。感情ありの行は `min(0.9, 基準値 + 0.4 × 強さ)`（基準値が0.9を超えているときは基準値のまま）。例：基準0.5・強さ0.8 → 0.82。
- `cfg_weight`：手本の言語と訳す言語が**違う**ときは 0 にする（手本のなまりが移るのを抑える公式の助言）。同じ言語なら画面の値のまま。`chatterbox_crosslingual_cfg0` を `false` にすると常に画面の値。
- 手本の言語は、行のデータの `ref_language`（話者プロファイルを使うときはプロファイルの `lang`）。**言語が分からない行、手本のない行（内蔵の声）は変えません。**
- **感情の種類（怒り・喜び…）は ChatterBox では指定できません。** 強さが上がるだけなので、悲しみでも怒りでも「強めの抑揚」になります。公式は「`exaggeration` を上げると話が速くなりやすい」とも書いています（長さ合わせで補正される）。

## 6. CosyVoice：調査結果と提案

### 6.1 結論：今の画面（Gradio webui）では「複製＋感情の指示」はできない

調べた場所は CosyVoice 公式（[GitHub](https://github.com/FunAudioLLM/CosyVoice)、2026-10-18 時点の main）のコードです。

| 確認したこと | 結果 |
|---|---|
| pyVideoTrans が使っている `webui.py` のモード | 「预训练音色」「3s极速复刻」「跨语种复刻」「自然语言控制」の4つ。**`inference_instruct2` を呼ぶモードは無い** |
| 「自然語言控制」の中身 | `inference_instruct(テキスト, 預訓練の話者, instruct)`。手本の音声は**無視**される（画面にもそう出る）。しかもコード内に `assert ... == 'CosyVoice'`（初代のみ）とあり、CosyVoice2/3 では使えない |
| CosyVoice3 に感情の指示はあるか | ある。`inference_instruct2(tts_text, instruct_text, prompt_wav, ...)` が**手本音声の複製と指示を同時に**使う。公式の例の指示は `'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>'` の形（`example.py`） |
| 公式の一覧にある感情の指示 | `cosyvoice/utils/common.py` の一覧に、**開心（喜）・伤心（悲）・生气（怒）**、大声・小声、速い・遅い、方言、キャラ口調。恐れ・驚き・嫌悪は無い。すべて中国語の決まった文で、「非常（とても）」付き＝強さの段階は無い |
| 公式の FastAPI サーバー（`runtime/python/fastapi/server.py`） | `/inference_instruct2` がある（`tts_text`・`instruct_text`・`prompt_wav` をフォームで送る）。ただし手本を**テンソル**で `inference_instruct2` に渡しており、現在の `frontend.py` は**ファイルのパス**を想定している（`torchaudio.load` に渡す）ため、コードを読んだ限り**そのままでは動かない見込み（要検証）**。出力は WAV ヘッダの無い生の16bit PCM |
| 日本語など他言語への指示の効きかた | モデルカードは「言語・方言・感情・速さ・音量などの指示に対応」とだけ書き、**指示に使える言語や感情の一覧は無い**。中国語の指示で日本語・英語の音声に感情が乗るかは**未検証** |

したがって `tts/_cosyvoice.py` は変更していません（無理な書き換えはしない方針）。

### 6.2 提案：小さな FastAPI を1本足す（実装は次の段階）

CosyVoice の隣で動かす小さなサーバーを作り、感情の付いた行だけそちらへ送る案です。

```
pyVideoTrans (_cosyvoice.py) ──感情なしの行──> 既存の Gradio webui（これまでどおり）
                             └─感情ありの行──> 新しい FastAPI（/instruct2）──> CosyVoice3.inference_instruct2
```

サーバーの骨格（**まだ動かしていない案**。置き場所の案：`meeting/tools/cosyvoice_instruct2_server.py`）：

```python
import io, sys, tempfile
sys.path.append('third_party/Matcha-TTS')
import torch, torchaudio
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

@app.post('/instruct2')
def instruct2(tts_text: str = Form(), instruct_text: str = Form(), speed: float = Form(1.0), prompt_wav: UploadFile = File()):
    if '<|endofprompt|>' not in instruct_text:
        instruct_text = f'You are a helpful assistant. {instruct_text}<|endofprompt|>'
    with tempfile.NamedTemporaryFile(suffix='.wav') as f:       # frontend はファイルのパスを受け取る
        f.write(prompt_wav.file.read()); f.flush()
        chunks = [o['tts_speech'] for o in model.inference_instruct2(tts_text, instruct_text, f.name, stream=False, speed=speed)]
    buf = io.BytesIO()
    torchaudio.save(buf, torch.cat(chunks, dim=1), model.sample_rate, format='wav')
    return Response(buf.getvalue(), media_type='audio/wav')
# 起動: uvicorn cosyvoice_instruct2_server:app --port 8001
```

pyVideoTrans 側の変更案：

1. `util/emotion_tts.py` に `cosyvoice_instruction(label, intensity)` を足す。公式の一覧にある3つだけ：angry→`请非常生气地说一句话。`、happy→`请非常开心地说一句话。`、sad→`请非常伤心地说一句话。`（恐れ・驚き・嫌悪は指示なし＝従来の複製）。
2. 設定 `cosyvoice_instruct2_url`（既定 `''`＝無効＝今の動作）を足す。
3. `tts/_cosyvoice.py`：URL があり、行に上記の感情があるときだけ `requests.post` で `/instruct2` を呼び、返った WAV を保存する。それ以外は今の Gradio 経路のまま。
4. テスト：`requests` をモックして、感情あり／なしでどちらの経路になるかを確認（今回の `tests/test_emotion_tts.py` と同じ作り）。
5. 配音のキャッシュキーの対象に CosyVoice を加える（`emotion_engine_types`）。

先に決めておく点と注意：
- **声の似かたが下がる可能性**：`inference_instruct2` は手本の「書き起こし」を使わず、手本の音声をトークン化した LLM 用の手がかりも捨てる（`frontend_instruct2` が `llm_prompt_speech_token` を削除している）。3秒複製より本人に似にくい可能性があるので、E6 の測定器で類似度を比べてから採用を決める。
- **日本語・英語の音声で、中国語の指示が効くかは未検証**。実機で数文を作って、感情判定（SenseVoice）で一致率を測る。
- 感情の段階（弱い・強い）は公式の指示にはない。
- 2つのサーバー（webui とこの FastAPI）が同じモデルを読むとメモリを2倍使う。どちらかに寄せるか、感情用だけ別マシンに置く。

## 7. 限界（今の段階）

- **VoxCPM の結果は実行のたびに変わる**（公式の注意）。同じ指示でも、気に入らなければ配音のやり直しが必要。
- **言語をまたぐ感情の再現は未検証**。手本は日本語で、訳した英語の声に感情が乗るかは、実機で確認していない（このクラウド環境にはモデルが入っていない）。
- 感情の指示は短い英語の固定表で、LLM が決める「TTS 用スタイル」（`docs/research.md` 11.6 の案）はまだ使っていない。
- 強さ・信頼度は第1部の暫定値（強さ0.5・信頼度0.6）のまま。音響から求める強さは次の段階。したがって今は `slightly`／`very` はほぼ出ず、ChatterBox の `exaggeration` も基準値＋0.2になる。
- 感情の付いた VoxCPM の行は「書き起こしを使う複製」を使えない（4.3）。
- 手本が別の言語（日本語）で、訳す言語が英語のとき ChatterBox は `cfg_weight=0` になる。`cfg_weight=0` は言語の違いによるなまりを抑える助言で、感情表現を上げる設定ではない（公式は感情を強くしたいとき `cfg_weight` を 0.3 前後に下げるとも書いている。両立の調整は未検討）。
- 画面の VoxCPM「バージョン」の選択（v2／v1／hf）は本家の不具合で効いていない可能性がある。設定の保存名が `voxcpm_version` なのに、`tts/_voxcpm.py` は `voxcpmtts_version` を読んでいるため、常に v2 として動く（本家のコードを読んだ限りで、実機未確認）。この改造では直していない。
- 動作確認は、モックを使った自動テストのみ（実際のモデルでの音は未確認）。

## 8. テスト

```
pytest tests/test_emotion_tts.py
```

見ているもの：感情と強さ→指示の変換、時刻（±300ミリ秒）での対応付け、キャッシュキー（感情の付いた行だけが変わる）、VoxCPM に送るリクエスト、ChatterBox に渡す `exaggeration`／`cfg_weight`、設定の初期値。
