# 会議翻訳アプリ 調査まとめ（2026年10月時点）

- 作成日: 2026-10-18
- 対象: 「話者付き文字起こし → 翻訳 → 本人の声で読み上げ（感情つき）」を作るための技術調査
- 計画・進捗は `docs/PLAN.md` を参照。本書は**調査結果（事実と出典）**の置き場所
- 記号: **(要検証)** = 出典が弱い／自分たちの音声で確認が必要な項目
- 2026-10-18 追記：サブエージェントで公式ページを確認し、TranslateGemma・Chatterbox・emotion2vec+・pyannote・VibeVoice-ASR・Qwen3-ASR・IndexTTS・Colab・GCP の記載を修正

---

## 0. 結論サマリ

| 機能 | 実現性 | 一言 |
|---|---|---|
| 日本語の話者付き文字起こし（ローカル） | ◎ | 技術は成熟。精度を決める最大要因はマイク環境 |
| 翻訳（英・仏・中） | ◎ | ローカルLLM（8〜12B）でも実用レベル。クラウドならさらに上 |
| 感情の読み取り | △ | 研究でも未解決領域。「ヒント」として使う設計にする |
| 本人の声で他言語読み上げ | ○ | 3秒程度の声から多言語で複製できるオープンモデルあり |
| 同時通訳並みのリアルタイム | △ | ノートPCでは困難。1文ごとに数秒遅れなら狙える |

**方式の結論**：1つのモデルで音声→音声を直接訳す方式（End-to-End）ではなく、
**工程ごとにモデルをつなぐ方式（カスケード）**を採用する。理由は 5章。

---

## 1. 日本語の文字起こし（ASR）

### 1.1 日本語ASR 11モデル比較（HEROZ, 2026-08）

CER = 文字誤り率（小さいほど良い）。RTF = 処理時間 ÷ 音声長（小さいほど速い）。

| モデル | 種別 | FLEURS | Common Voice | JSUT | RTF | VRAM |
|---|---|---|---|---|---|---|
| cohere-transcribe-03-2026 | OSS | **2.89%** | **20.22%** | 8.59% | 0.04〜0.09 | 4.3GB |
| parakeet-tdt_ctc-0.6b-ja (NVIDIA) | OSS | 5.56% | 21.53% | **6.60%** | **0.01〜0.03** | 5.3GB |
| whisper-large-v3 | OSS | 4.24% | 28.61% | 7.01% | 0.15〜0.23 | 3.6GB |
| whisper-large-v3-turbo | OSS | 3.88% | 75.55%※ | 7.22% | 0.04〜0.09 | 1.7GB |
| kotoba-whisper-v2 | OSS | 5.88% | 26.62% | 7.36% | 0.03〜0.08 | 1.6GB |
| qwen3-asr-1.7b | OSS | 5.28% | 26.33% | 8.42% | 0.10〜0.17 | 3.9GB |
| qwen3-asr-0.6b | OSS | 8.44% | 34.12% | 11.90% | 0.10〜0.17 | 1.5GB |
| reazon-zipformer | OSS | 12.31% | 28.32% | 9.54% | 0.01〜0.02 | 0.8GB |
| gpt-4o-transcribe | API | 2.26% | 26.25% | 5.14% | — | — |

※ turbo は同じフレーズを延々と繰り返す出力（ハルシネーション）が1件あり、平均が跳ね上がった。

**読み取り方**
- 各データは約10分と小さい。1ポイント未満の差は「傾向」程度に見る。
- Common Voice（素人が様々なマイクで録った音声）は全モデルで20%台。**条件の悪い音声では一気に精度が落ちる**。会議音声はこちら寄りになりやすい。
- 記事の結論も「順位は評価音声で変わる。**実際の業務に近い音声で評価せよ**」。

### 1.2 その他の比較
- neosophie（IT用語中心のベンチ, 2026-04）: Qwen3-ASR-1.7B が CER 0.082、Whisper 0.157。ただし著者は Qwen3-ASR の日本語微調整版（`neosophie/Qwen3-ASR-1.7B-JA`）を配布する当事者。微調整版は固有名詞F1で改善、CERは元モデルと同等。
- 結論：**上位候補は Cohere Transcribe / Qwen3-ASR-1.7B / Whisper large-v3 / Parakeet-ja**。自分たちの会議音声で比べて決める。

### 1.3 候補モデルの特徴

| モデル | ライセンス | 長所 | 注意点 |
|---|---|---|---|
| Cohere Transcribe (約2B) | Apache-2.0 | 精度上位。mlx-audio（Mac）対応 | **タイムスタンプ非対応**、言語は手動指定、利用前に同意（gated） |
| Qwen3-ASR-1.7B | Apache-2.0 | 30言語＋中国語方言22種（日本語を含む）。**文脈・用語を指示できる**（公式 `qwen-asr` の `transcribe(audio, context=..., language=...)`。モデルカードには記載なし）。MLX対応。M5 ProでRTF 0.027 | Alibaba製（社内承認に注意）。用語の並び順が効く可能性 |
| Qwen3-ForcedAligner-0.6B | Apache-2.0 | 日本語を含む11言語で**単語単位の時刻**を付与（最大5分） | 文字起こし済みテキストが必要 |
| Whisper large-v3 | MIT | 安定・情報が多い | turbo は繰り返しハルシネーションに注意 |
| Parakeet-ja (0.6B) | (要検証) | 最速クラス、JSUTで最良 | FLEURSでは中位 |

---

## 2. 話者分離（誰が話したか）

| モデル | ライセンス | 特徴 | 注意点 |
|---|---|---|---|
| **pyannote community-1**（pyannote.audio 4.x） | CC-BY-4.0 | オープンの定番。初回DL後は**オフラインで動作**。CPU既定、CUDA可。参加人数を `num_speakers`（または `min_speakers`/`max_speakers`）で指定可。**`exclusive_speaker_diarization` 出力**（同時に1人だけ割り当て）で文字起こしの時刻と突き合わせやすい | 利用前にHFで同意＋トークン。Apple MPSの記載なし |
| VibeVoice-ASR（Microsoft, 約9B） | MIT | **文字起こし＋話者分離＋時刻を1回で**出力（最大60分）。用語（hotword）指定可。言語一覧に日本語（ja）あり、公式結果表に日本語行（MLC-Challenge WER 14.69） | 大きい（必要VRAMの公式記載なし。第三者は20〜24GBと記載）。2026-07 に CPU専用の量子化版 **VibeVoice-ASR-BitNet（約1.6GB）** あり → ローカル候補（精度は要検証） |
| NVIDIA Sortformer（offline/streaming） | (要検証) | ストリーミング版あり。言語によってはpyannoteより良い結果 | 最大4話者前提。NVIDIA GPU向け |
| DiariZen（BUT） | (要検証) | 第三者評価でオープン最良級（DER 13.3%） | 派生モデルに非商用ライセンスの例あり。本体の重みライセンスは要確認 |

**pyannote community-1 の誤り率（DER, 重なり発話も含む厳しい条件）**
AMI(IHM) 17.0% / AMI(SDM) 19.9% / AISHELL-4 11.7% / AliMeeting 20.3% / VoxConverse 11.2% / DIHARD3 20.2%

**ベンチマークの注意**：言語によって順位が入れ替わる（soniqo の8言語ベンチ）。日本語会議での公開比較は見当たらない → 自前評価が必須。

**録音側の工夫**：Zoom のローカル録画には「参加者ごとに個別の音声ファイルを録音」する設定がある（最大80人、電話参加は1ファイルに統合）。オンライン会議ならこれで話者分離をほぼ不要にできる。

---

## 3. 感情認識

| モデル | 内容 | 注意点 |
|---|---|---|
| emotion2vec+ large | 感情表現の汎用モデル＋9分類（怒り・嫌悪・恐れ・喜び・平常・その他・悲しみ・驚き・不明） | 自然な会話での正解率は研究でも約50〜80%。**重みのライセンスは FunASR 独自（code は MIT）→ 商用可否を要確認** |
| SenseVoiceSmall | 日本語を含む5言語のASR＋**感情タグ**（HAPPY/SAD/ANGRY/NEUTRAL/FEARFUL/DISGUSTED/SURPRISED）＋音響イベント | **コードはMIT、重みは FunASR 独自ライセンス**（メンテナ見解では商用可・出典表記が必要）。50言語版は研究用で未公開 |
| Qwen3-Omni-30B-A3B-Captioner | 音声を聞いて**感情・意図・話者の様子を文章で記述**（Apache-2.0） | 30B級で重い。1入力・テキスト指示不可・30秒以内推奨 |
| 日本語専用 | wav2vec2-xlsr-japanese-SER（JTESで学習）など小規模なもの | 日本語の会話データで学習したものは少ない |

**設計方針**：音声からの判定（emotion2vec+ 等）と、文章からの判定（LLM）を組み合わせる。
日本語は「ちょっと厳しいですね（＝実質No）」のような**言外の意味**が誤訳の原因になりやすく、これは文脈を渡せるLLM側で扱う。

関連データ：JVNV（日本語感情音声コーパス, CC BY-SA 4.0）、JNV（日本語非言語発声）

---

## 4. 翻訳

| 候補 | 特徴 |
|---|---|
| TranslateGemma（Google, 2026-01） | 翻訳特化、55言語、**4B / 12B / 27B**（Gemma 3ベース）。2026年の評価で 27B 版が英→日・中・露でローカル最良、32B級の汎用モデルを上回る。ライセンスは Gemma 利用規約（同意が必要、OSIのオープンソースではない） |
| gpt-oss-20b | 英→日で32B級と同等以上の評価 |
| Qwen3 系（8B〜） | 日英中のローカル翻訳でよく推奨される（評価は主にブログ） |
| 商用API（Gemini / Claude 等） | 最高品質。ただし**データが外部に出る** |

**感情・文脈を反映させるには**：翻訳特化モデルより、指示に従いやすい汎用LLMに
「前後の発言」「話者の感情ラベル」「用語集」を渡す方が扱いやすい。
評価ベンチ：JP-TL-Bench（日英双方向、60モデル）。

---

## 5. 本人の声での読み上げ（音声合成・声の複製）

| モデル | 言語 | 声の複製 | 感情制御 | ライセンス | 注意点 |
|---|---|---|---|---|---|
| **Qwen3-TTS**（0.6B / 1.7B） | 日英仏中を含む10言語 | 3秒の声で可、**言語をまたいだ複製可** | 1.7B の VoiceDesign / CustomVoice 版のみ文章指示可。**複製用 Base 版は指示不可** | Apache-2.0 | → 発言ごとの元音声を手本にする方式で感情を乗せる |
| **CosyVoice3-0.5B** | 日英仏中を含む9言語 | 可（言語またぎ可） | `inference_instruct2(文, 指示, 手本音声)` で複製＋指示を併用可。公式例の感情は「開心・傷心・生気」の3つ＋速度・音量 | Apache-2.0 | **言語またぎ・指示モードでは手本の抑揚は移らない**（コードで確認：手本の音声トークンを削除し声色だけ使う）→感情は指示で明示が必要 |
| **VoxCPM2**（OpenBMB, 2B） | **30言語**（日英仏中を含む） | 可（短い手本、または手本＋正確な書き起こしで高精度） | **文頭に括弧で指示**：`(slightly faster, cheerful tone)本文` で感情・速さ・表現を制御しつつ声色を保つ | **Apache-2.0（商用可）** | 結果がぶれるため1〜3回生成を推奨。言語をまたぐ複製の記載なし (要検証) |
| **IndexTTS2** / IndexTTS-2.5 | IndexTTS2 は**英・中のみ**。**IndexTTS-2.5 は中・英・日・西・アラビア語**（仏語なし） | 可 | **声色と感情を別々に指定**できる（感情は別音声・8次元ベクトル・文章で、強さ0〜1）。時間長も制御可 | bilibili 独自。**自社・関連会社の売上10億元超 または 月間利用者1億超は別途書面許諾が必要** → 大企業は実質契約が必要 | 日本語を使うなら 2.5。仏語は不可。契約確認まで不採用 |
| Chatterbox Multilingual（Resemble AI, 0.5B） | **23言語**（日・仏・中を含む） | 可 | 感情の強さを調整可 | MIT | — |
| XTTS-v2 | 17言語 | 可 | 弱い | — | 開発元が閉鎖、更新はコミュニティ頼み |

**感情を乗せる2つの方法**
1. 発言ごとに「その発言の元音声」を声の手本にする → 声色と一緒にその時のトーンも引き継がれやすい（一番簡単）
2. 登録済みの声＋感情の指示（CosyVoice3 / IndexTTS2）

---

## 6. 音声→音声の直接翻訳（End-to-End）の現状

| システム | 状況 |
|---|---|
| Hibiki（Kyutai） | 声を保ったまま同時翻訳。**仏→英のみ**。重みは CC-BY 4.0、Mac（MLX）で動作 |
| SeamlessExpressive（Meta） | 声色・抑揚を保持。**非商用ライセンス**で会社では使えない |
| UniSS（ICLR 2026） | 声・感情・長さを保持。**中⇔英のみ**、コード未確認 |
| PACT（arXiv 2610.18247, 2026-10） | 「既存システムは内容や声色は保てるが、**感情・口調・態度の保持は苦手**」と指摘。西⇔英で評価 |
| PLaST（AAAI） | パラ言語（口調・感情）を考慮した音声翻訳 |
| StressTransfer（2025-10） | 強調箇所をLLMで訳文側に写し、制御可能なTTSで再現 |

→ 日本語を含み感情まで保てる一体型の公開モデルは**現時点でない**。カスケード方式が現実解。
（フランス語話者がいる場合の仏→英だけは Hibiki を使う選択肢あり）

---

## 7. 実行環境

### 7.1 ローカルPC
| 項目 | Surface（Laptop 5 想定、型番未確定） | MacBook Air M5 24GB |
|---|---|---|
| 演算 | CPUのみ | 統合GPU＋メモリ帯域 153GB/s |
| 話者付き文字起こし | 会議後のまとめ処理なら可（遅い） | 高速 |
| 声の複製 | 実用は厳しい | 可能 |
| メモリ見積り | — | 全部載せて約15〜18GB → 24GBで収まる |
| 注意 | Snapdragon機ならPythonライブラリ非対応が多い | ファンレスで長時間は熱で減速の可能性 |

### 7.2 クラウド
| 選択肢 | 内容 | 注意点 |
|---|---|---|
| **Google Colab 無料** | 公式FAQ：1セッション**最大12時間**、GPUの種類は時期で変わる（通常はT4 16GBと第三者情報） | 割り当て保証なし。放置で切断 |
| Colab Pro / Pro+ | Pro: $9.99/月で100コンピュートユニット（T4 約1.76CU/時 → 約57時間、A100 約15CU/時）。Pro+: $49.99/月 | 価格は国で異なる。上位GPUも保証なし |
| **Google Cloud 無料トライアル（$300 / 90日）** | 公式ドキュメント：**無料トライアル中はVMにGPUを追加できない**。有料アカウントに切り替えると残りクレジットは引き継がれる | 切替後もGPUの割り当て申請が必要。個人アカウントは通りにくいという報告多数 |
| Google Cloud Speech-to-Text（Chirp 3） | 日本語（ja-JP）の話者分離に対応（BatchRecognize） | 比較用の基準（ベースライン）として有用。データは外部送信 |

**クラウド利用時の大原則**：会社の会議音声を個人のクラウドに上げない。
クラウドでは「自分の声」「同意を得た模擬会議」「公開データ」だけを使う。

---

## 8. 評価指標（用語）

| 指標 | 意味 |
|---|---|
| CER（文字誤り率） | 文字起こしの誤り率。日本語では表記ゆれ（数字・句読点・かな漢字）を正規化してから計算する |
| DER（話者分離誤り率） | 「誰が」の誤り率 |
| cpCER / cpWER | 話者ごとにまとめて比べる誤り率。「誰が何を」を同時に評価できる |
| 話者正解率（発言単位） | 実務的に分かりやすい指標 |
| 主観評価（MOS 5段階） | 翻訳品質・本人らしさ・自然さ |

---

## 出典

**文字起こし**
- [日本語ASRモデル11種比較 (HEROZ, 2026-08)](https://techblog.heroz.jp/entry/2026/08/18/120000)
- [Japanese ASR Benchmark 2026 (neosophie)](https://neosophie.com/en/blog/20260226-japanese-asr-benchmark) / [IT-Domain ASR Benchmark](https://neosophie.com/en/blog/20260414-it-asr-benchmark) / [Qwen3-ASR-1.7B-JA 公開記事](https://neosophie.com/en/blog/20260427-qwen-finetuned-model)
- [Cohere Transcribe (HF)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [Qwen3-ASR-1.7B (HF)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) / [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR 論文](https://hyper.ai/en/papers/Qwen3-ASR) / [文脈バイアス手法の比較 (arXiv 2608.05759)](https://arxiv.org/pdf/2608.05759) / [Qwen 音声認識ガイド](https://docs.platform.qianwenai.com/developer-guides/accuracy-tuning/speech-recognition)
- [mlx-audio](https://cdn.jsdelivr.net/gh/blaizzy/mlx-audio@main/README.md) / [soniqo Apple Silicon ベンチ](https://soniqo.audio/benchmarks) / [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr/)
- [Best Open Source STT 2026](https://www.sevenlabs.site/blogs/best-open-source-speech-to-text-models-2026)

**話者分離**
- [pyannote community-1 (HF)](https://huggingface.co/pyannote/speaker-diarization-community-1) / [pyannote vs Sortformer](https://www.pyannote.ai/blog/pyannote-vs.-nvidia-nemo-diarization-accuracy-setup-and-production-readiness) / [Top 7 Diarization 2026](https://www.pyannote.ai/blog/top-7-speaker-diarization-apis-and-models-in-2026)
- [Diarization Benchmarks (soniqo)](https://soniqo.audio/benchmarks/diarization) / [Benchmarking Diarization Models](https://www.researchgate.net/publication/396048562_Benchmarking_Diarization_Models) / [State of Diarization 2026 (Picovoice)](https://picovoice.ai/blog/state-of-speaker-diarization/)
- [VibeVoice-ASR (HF)](https://huggingface.co/microsoft/VibeVoice-ASR) / [VibeVoice-ASR 技術報告](https://www.arxiv.org/pdf/2601.18184) / [リリース記事](https://comfyui-wiki.com/en/news/2026-01-21-microsoft-vibevoice-asr-release)
- [SUPlime (DiariZen派生)](https://huggingface.co/rewayai/suplime-large)
- [VibeVoice-ASR ドキュメント (GitHub)](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md)
- [Zoom：参加者別の音声録音](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0076922)

**感情認識**
- [emotion2vec+ large (HF)](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
- [SenseVoice](https://github.com/QwenAudio/SenseVoice) / [emotion2vec 論文](https://aclanthology.org/2024.findings-acl.931.pdf) / [FunAudioLLM](https://arxiv.org/html/2407.04051v1)
- [Qwen3-Omni Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) / [Qwen3-Omni 技術報告](https://arxiv.org/abs/2509.17765)
- [SER 臨床ベンチ 2026](https://arxiv.org/pdf/2609.26054) / [JVNV](https://arxiv.org/pdf/2310.06072) / [JNV](https://arxiv.org/pdf/2305.12445) / [wav2vec2 日本語SER](https://mixpeek.com/model/Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition)

**翻訳**
- [TranslateGemma (Google Blog)](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) / [translategemma-4b-it (HF)](https://huggingface.co/google/translategemma-4b-it)
- [ローカルLLM翻訳ベンチ (arXiv 2605.31452)](https://arxiv.org/html/2605.31452v1) / [JP-TL-Bench](https://benchmarklist.com/benchmarks/jp_tl_bench_anchored_pairwise_llm_evaluation_for_bidirectional_japanese_english_translation/)
- [ローカルLLM 2026夏 (Classmethod)](https://dev.classmethod.jp/en/articles/local-llm-guide-2026-summer/) / [日英翻訳LLM比較 (note)](https://note.com/masajiro999/n/n93fce42d73b2?hl=en)

**音声合成**
- [Qwen3-TTS モデル一覧](https://www.mintlify.com/QwenLM/Qwen3-TTS/concepts/models) / [Qwen3-TTS-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) / [Qwen3-TTS ガイド](https://dev.to/czmilo/qwen3-tts-the-complete-2026-guide-to-open-source-voice-cloning-and-ai-speech-generation-1in6)
- [Fun-CosyVoice3-0.5B](https://huggingface.co/zhchhe/Fun-CosyVoice3-0.5B-2512) / [CosyVoice](https://github.com/qwenaudio/cosyvoice)
- [IndexTTS-2 (HF)](https://huggingface.co/IndexTeam/IndexTTS-2) / [IndexTTS-2.5 (HF)](https://huggingface.co/IndexTeam/IndexTTS-2.5) / [Chatterbox (HF)](https://huggingface.co/ResembleAI/chatterbox)
- [IndexTTS2 論文](https://arxiv.org/abs/2506.21619v2) / [Best Open-Source TTS 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models) / [free-voice-clone 一覧](https://github.com/0xSojalSec/free-voice-clone)

**音声→音声翻訳の研究**
- [Hibiki](https://github.com/kyutai-labs/hibiki) / [Seamless](https://arxiv.org/pdf/2312.05187) / [UniSS](https://mlanthology.org/iclr/2026/cheng2026iclr-uniss/) / [PACT](https://arxiv.org/abs/2610.18247) / [PLaST](https://ojs.aaai.org/index.php/AAAI/article/view/40449) / [StressTransfer](https://arxiv.org/abs/2510.13194v1)

**実行環境**
- [MacBook Air 仕様 (Apple)](https://www.apple.com/my/macbook-air/specs/) / [MacBook Air M5 (Apple Wiki)](https://theapplewiki.com/wiki/MacBook_Air_%2813-inch%2C_M5%29)
- [Colab FAQ (公式)](https://research.google.com/colaboratory/faq.html) / [Google Cloud 無料枠 (公式)](https://docs.cloud.google.com/free/docs/free-cloud-features)
- [Colab GPU ガイド](https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations) / [Colab 価格](https://aicoolies.com/pricing/google-colab) / [Colab 代替比較](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/)
- [GCP 無料トライアルとGPU](https://discuss.google.dev/t/300-free-trial-is-useless-without-gpu-quota-my-trial-period-is-wasting-away/290091) / [個人アカウントのGPU割当](https://discuss.google.dev/t/stuck-requesting-l4-gpu-quota-verified-personal-account-directed-to-sales-sales-refuses/186968)
- [Chirp 3 (Google Cloud)](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3)

---

## 9. 既存の製品・OSS調査（2026-10-18 追記）

サブエージェント（sonnet）3体で並列調査。**太字の主張は公式ページで再確認済み**。それ以外はベンダー記事・第三者情報を含む。

### 9.1 結論
- **「本人の声＋日本語＋仏/中＋リアルタイム」は Microsoft Teams Interpreter が公式に提供済み**（Copilot ライセンス必要、クラウド処理）。
- 「完全ローカル＋本人の声＋感情＋日本語＋会議」を全部満たすものは**見つからなかった**。
- **感情をそのまま移す機能は、どの製品・OSSにもない**（Teams は「感情を誇張せず」口調・高さ・話し方を保持、と明記）。
- ローカルの日本語話者付き文字起こし（優先1）は既存ツールが多数ある。ただし**日本語会議での話者分離精度の独立した比較はない**。

### 9.2 大手プラットフォーム

| 製品/機能 | 本人の声 | 感情 | 日本語 | 仏/中 | 処理場所 | 条件 | 状態 |
|---|---|---|---|---|---|---|---|
| **Teams Interpreter** | **○（Simulate my voice、声サンプルは保存しない）** | **△ 口調・高さ・話し方は保持、感情は誇張しない** | **○** | **○/○（他に英独伊韓葡西、計9言語）** | **クラウド（Azure）。音声認識→英語テキスト→翻訳→音声合成（英語経由）** | **M365 Copilot ライセンス（1人月20時間込み）。会議室は Teams Rooms Pro（Windows版のみ）** | 提供中。Teams events・Rooms on Android・Teams無料版は非対応 |
| **Google Meet 音声翻訳** | **○（トーンと話す速さを模倣）** | 明言なし | **×** | **仏○ / 中×（英⇔西仏独葡伊のみ）** | クラウド | **Business Standard 以上 等** | **2026-01-27 GA** |
| Zoom Voice translator | ×（合成音声。声の複製は開発中） | 記載なし | ○ | ○/○ | (要検証) | アドオン（従量制） | 提供中 (要検証) |
| Apple Live Translation（iOS 26+） | × | 記載なし | ○ | ○/○ | 端末内 | 対応 iPhone＋AirPods | 提供中 |
| Gemini 3.5 Live Translate | ○（抑揚・ピッチ保持） | 明言なし | (要検証) | (要検証) | (要検証) | Meet は限定プレビュー | 2026-06 発表 (要検証) |
| Webex / DeepL Voice / Samsung | (要検証) | (要検証) | ○ | (要検証) | 主にクラウド | — | 一部提供中 |

### 9.3 オープンソース（吹き替え・翻訳パイプライン）

| プロジェクト | 話者分離 | 話者ごとの声複製 | 感情 | 完全ローカル | 形態 | ライセンス | 活発さ |
|---|---|---|---|---|---|---|---|
| **pyVideoTrans** | **○（WhisperX 等）** | **○（多角色配音：F5-TTS/CosyVoice/GPT-SoVITS、話者ごとに手動割当）** | × | **○（faster-whisper＋Ollama/M2M100＋ローカルTTS）** | **ファイル処理のみ** | **GPL-3.0** | 約19k★、頻繁に更新 |
| SoniTranslate | ○ pyannote | ○（XTTS/OpenVoice/RVC） | × | ○（Linux+CUDA前提） | ファイル | Apache-2.0 | 2026-08 更新 |
| Linly-Dubbing | ○ | ○（CosyVoice/GPT-SoVITS/XTTS） | × | ○ | ファイル | Apache-2.0 | 2025-03 で停滞 |
| VideoLingo | × | ×（単一の声） | × | 一部（翻訳にAPI） | ファイル | Apache-2.0 | 活発 |
| open-dubbing | ○ | ×（性別で合成音声） | × | ○ | ファイル | Apache-2.0 | 2025-07 |
| ViDubb | ○ | (要検証) | 4分類のみ | 一部 | ファイル | Apache-2.0 | (要検証) |
| Hibiki-Zero（Kyutai） | 不要（1話者） | 声転写あり | 韻律のみ | ○（NVIDIA GPU） | リアルタイム | **重み CC BY-NC-SA（非商用）** | 2026-02。仏西葡独→英のみ |

その他（未確認が多く参考扱い）：VoiceStudio（旧 OmniVoice Studio、★数が不自然 (要検証)）、OpenCreator（旧 KrillinAI、ライセンス表記が食い違い (要検証)）、Qwen LiveTranslate・Seed LiveInterpret（APIのみ・クラウド）。

### 9.4 ローカルの日本語文字起こし＋話者分離（優先1の既製品候補）

| ツール | オフライン | 話者分離 | OS | 価格 | エンジン | メモ |
|---|---|---|---|---|---|---|
| AmiVoice ScribeAssist（VoXT One） | ○（クラウド要約のみ要ネット） | ○ 声を学習させると名前を自動付与 | Windows のみ | 見積制、14日トライアル | AmiVoice | 日本語ネイティブ、語彙登録あり |
| RecText AI 2 | ○ | ○（2026-03 新機能） | Windows | 9,980円 買い切り | オートメモ系 | 精度92.1%は自社測定・1人条件 |
| FFTrans Neu | ○ | ○（最大8人） | Mac（M1以降、macOS 26以降） | 7,500円、14日トライアル | Core ML（詳細不明） | 販売元記事のみ (要検証) |
| Whisper Notes | ○ | ○ pyannote community-1 | Mac/iPhone | $14 | Parakeet/Whisper/SenseVoice | 重なり発話は1人扱い |
| MacWhisper Pro | ○ | ○（最大8人） | Mac | 約€64 | Whisper/Parakeet | 日本語の記載未確認 |
| noScribe | ○ | ○ pyannote | Mac/Win/Linux | 無料 GPL-3.0 | Whisper系 | 研究インタビュー向け |
| WhisperX＋kotoba-whisper | ○ | ○ pyannote | Python | 無料 | Whisper系 | 要 Python |
| Vibe / Buzz / Meetily / Scriberr | ○ | ○〜△ | 各種 | 無料〜 | Whisper/Parakeet 等 | 日本語の話者分離は (要検証) |

- 小規模な検証（Shisa AI, 2026-08）では、日本語で pyannote community-1 が Sortformer に劣ったとの報告あり (要検証)。
- 「音声は外に出ない」はどれも自己申告。**ネット遮断状態で動くこと・通信が出ないことを自分で確認**する。

### 9.5 商用クラウド（本人の声＋感情、ファイル型中心）

| サービス | 形態 | 本人の声 | 感情 | 日本語 |
|---|---|---|---|---|
| ElevenLabs Dubbing v2（2026-05） | ファイル（リアルタイム不可と公式記載） | ○（最大32話者） | ○（原音から生成と主張） | ○ |
| Palabra.ai | リアルタイム（Zoom/Meet/Teams 連携） | ○ 自動複製 | 「近日対応」と記載 | ○ |
| Camb.ai / Perso AI / HeyGen / Rask.ai | 主にファイル | ○ | 主張のみ | ○〜(要検証) |

### 9.6 自作する意義（既存品との差）
1. 話者別の発言と翻訳を**Excel/CSVで記録**として残せる（Teams Interpreter の翻訳はライブのみ、録画には元音声だけ）
2. **社外に出さない処理**（既存の本人の声翻訳はすべてクラウド）
3. **対面会議**への対応（Teams Rooms は Windows 版のみ、同室の複数人の扱いは不明）
4. **感情の反映**（どの製品も未対応）
5. 英語を経由しない言語対（日→仏・日→中を直接）

### 9.7 出典
- [Teams Interpreter（管理者向け, Microsoft Learn）](https://learn.microsoft.com/en-us/microsoftteams/interpreter-agent-teams) / [Teams Interpreter（利用者向け）](https://support.microsoft.com/en-us/topic/c7efe2bb-535d-42ab-a5c4-d2d91619b46d)
- [Google Meet 音声翻訳 GA](https://workspaceupdates.googleblog.com/2026/02/speech-translation-meet-ga.html) / [Meet ヘルプ](https://support.google.com/meet/answer/16221730)
- [Zoom Voice translator](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0084896) / [Zoom ブログ](https://www.zoom.com/en/blog/voice-translation-zoom/)
- [Apple 機能提供状況](https://www.apple.com/ios/feature-availability/) / [Gemini Live Translate](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/) / [DeepL Voice](https://www.deepl.com/en/products/voice) / [Webex](https://help.webex.com/nqzpeei/)
- [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) / [SoniTranslate](https://github.com/R3gm/SoniTranslate) / [Linly-Dubbing](https://github.com/Kedreamix/Linly-Dubbing) / [VideoLingo](https://github.com/Huanshere/VideoLingo) / [open-dubbing](https://github.com/Softcatala/open-dubbing) / [ViDubb](https://github.com/medahmedkrichen/ViDubb) / [Hibiki-Zero](https://github.com/kyutai-labs/hibiki-zero)
- [AmiVoice ScribeAssist](https://voxt-one.advanced-media.co.jp/service/scribeassist) / [RecText AI 2（ASCII）](https://ascii.jp/elem/000/004/385/4385620/) / [FFTrans Neu](https://symphonics.biz/2026/10/13/mac-offline-minutes/) / [Whisper Notes](https://whispernotes.app) / [MacWhisper](https://goodsnooze.gumroad.com/l/macwhisper) / [noScribe](https://github.com/kaixxx/noScribe) / [WhisperX](https://github.com/m-bain/whisperX) / [kotoba-whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) / [Vibe](https://github.com/thewh1teagle/vibe) / [Meetily](https://meetily.ai) / [Scriberr](https://github.com/rishikanthc/Scriberr)
- [ElevenLabs Dubbing](https://elevenlabs.io/docs/overview/capabilities/dubbing) / [Palabra.ai](https://palabra.ai/voice-translation-api) / [Camb.ai](https://camb.ai/solutions/news) / [Perso AI](https://perso.ai/v3-ai-dubbing-model) / [HeyGen](https://heygen.com/translate)

---

## 10. pyVideoTrans のコード調査（2026-10-18 追記）

対象：`vendor/pyvideotrans`（v4.16、commit f9f35d1、2026-10-15）。サブエージェント（sonnet）が読み取り専用で調査し、**太字はコードで直接確認済み**。

### 10.1 処理の流れ
- 入口：GUI `sp.py`、CLI `cli.py`（`--task vtv` で全工程を画面なしで実行可。ただし **CLI では話者分離が動かない**）、Web UI `webui.py`
- タスク本体：`videotrans/task/trans_create.py:TransCreate`（各工程は `task/_stage_*.py` の Mixin）
- 工程：準備 → 文字起こし → 話者分離（GUIの単一動画のみ）→ 翻訳 → 配音（TTS）→ 長さ合わせ → 合成
- 工程間のデータ受け渡しは**ディスク上のSRTファイル**（本文と時刻しか保存されない）。追加情報は**別ファイル（例：`speaker.json`、行番号を揃えたJSON）**で持つ方式

### 10.2 声の複製（「似ない」問題に直結）
- **複製の手本音声 = その行自身の元音声**（`task/_stage_dubbing.py`：`ref_wav = clone-{i}.wav`、`task/_stage_recogn.py:_create_ref_from_vocal` が行の開始〜終了時刻で切り出す）
- 本人ごとの固定の手本・品質の選別・長さの確保は**していない**。GPT-SoVITS は手本を3〜10秒に制限
- 手本の書き起こし `ref_text` = 原文字幕の同じ行
- → 長所：その発言の抑揚・感情が手本に入る。短所：**短い・雑音入り・重なり発話の行は声色が崩れ、行ごとに声がばらつく**
- 対応エンジン（SUPPORT_CLONE）：CosyVoice、F5、Higgs、IndexTTS、VoxCPM、Spark、ChatterBox、GPT-SoVITS、Qwen3（ローカル）、OmniVoice、MOSS、Confucius、FireRed3、ZipVoice ほか

### 10.3 感情を差し込める場所
| 用途 | 場所 | 内容 |
|---|---|---|
| 感情の検出 | 新規 `task/_stage_emotion.py`（Mixin）＋ `process/_audio_emotion.py`（別プロセス） | 元音声と確定済み字幕の時刻から、行ごとの感情を `emotion.json` に保存（`speaker.json` と同じ方式） |
| 翻訳 (b) | `translator/_base.py`、`translator/_openaicompat.py` ほか、`prompts/{srt,text}/*.txt` に `{emotion_info}` を追加 | 本文には埋め込まず、「行番号→感情」の別表として指示文に渡す（行ずれ防止）。翻訳キャッシュのキーにも感情を含める |
| 声の複製 (a) | `task/_stage_dubbing.py` の行ごとの辞書（`tmp_dict`）に emotion 等を追加 | 配音キャッシュのキーにも感情を含める |
| エンジン別の感情指定 | ChatterBox（exaggeration）、IndexTTS2（感情参照音声）、CosyVoice（instruct_text、複製と併用可否は要検証）、VoxCPM、MiniMax/OpenAI（クラウド） | 既存の引数を行ごとに変える |
| 共通の後処理 | 配音直後（`_stage_dubbing.py` の run_tts の直後）に新規 `util/_emotion_prosody.py` | 音の高さ・音量などの補正。全エンジン共通で効く |

- **SenseVoiceSmall は既に文字起こしで使われており、感情タグ（`<|HAPPY|>` 等）を `process/_stt_utils.py` で捨てている** → 捨てずに拾えば追加モデルなしで感情が取れる可能性
- funasr・modelscope が依存に入っているので emotion2vec+ も載せられる見込み (要検証)

### 10.4 注意点
- 長さ合わせ：`task/_rate.py:SpeedRate`（pyrubberband または ffmpeg atempo で速度変更）→ 仮説6（早回しで声が変わる）の確認対象
- 速度・音量・音の高さは実質「全行共通の値」。行ごとに変えるには改造が必要
- Python 3.10 固定、uv 管理。macOS は CPU/MPS（多くの別プロセスは Mac で CPU 扱い）
- テストは pytest のモック中心（音声のテストはない）。`SrtItem` の項目数や Mixin 構成を固定するテストがあるので、改造時に更新が必要
- 本体は GPL-3.0

---

## 11. 感情を声のトーンに反映させる方法（2026-10-18 追記）

サブエージェント（sonnet）が公式README・モデルカード・LICENSE本文・論文を確認。**太字は自分でも公式ページ／コードで再確認済み**。

### 11.1 結論
- 声の複製と感情の指示を**同時に**使える候補：**VoxCPM2**（Apache-2.0、30言語）、**CosyVoice3 の instruct2**（Apache-2.0、9言語）。
- 感情制御が最も細かいのは IndexTTS-2.5（声色と感情を別入力、8種の感情ベクトル＋強さ）だが、**仏語不可・大企業は別契約が必要**。
- Qwen3-TTS は 2026-10 時点でも「複製（Base）＋指示」は不可。
- **pyVideoTrans はすでに「その行の元音声を手本」にしているが、言語をまたぐと手本の抑揚は弱くしか移らない** → 感情はエンジンへの指示で明示する必要がある。

### 11.2 ライセンスの重要な訂正
| モデル | 重みのライセンス | 判断 |
|---|---|---|
| **F5-TTS** | **CC-BY-NC（非商用）**（コードは MIT） | pyVideoTrans の推奨エンジンだが**使わない** |
| OmniVoice / EmoVoice | 重み CC-BY-NC | 使わない |
| Higgs Audio v3 | 研究・非商用（製品組込は別契約） | 使わない |
| IndexTTS-2 / 2.5 | bilibili 独自（売上10億元超等は別途許諾） | 契約確認まで使わない |
| emotion2vec+ | FunASR MODEL_LICENSE（商用禁止の文言なし、ただし「参考・学習目的」の記載あり） | **法務確認が必要** |
| SenseVoiceSmall | FunASR 独自（メンテナ見解では商用可・出典表記必要） | 条件付きで使用可 |
| audeering wav2vec2（感情の連続値） | CC BY-NC-SA | 使わない |
| VoxCPM2 / CosyVoice3 / Chatterbox / GPT-SoVITS | Apache-2.0 / Apache-2.0 / MIT（出力に透かし） / MIT | 使用可 |

### 11.3 エンジン別の感情の入れ方
| エンジン | 方法 | pyVideoTrans 側の対応 |
|---|---|---|
| VoxCPM2 | 文頭に英語で `(angry, tense, slightly faster)` 等。強さは slightly / very で3段階 | `tts/_voxcpm.py` の `control_instruction=''`（空）に入れる |
| CosyVoice3 | `inference_instruct2`（喜・悲・怒の3種＋速度）。公式 webui は instruct2 を呼ばない | `tts/_cosyvoice.py` は「3秒複製」固定 → Python API を直接呼ぶ改造が必要 |
| Chatterbox | 感情の種類は指定不可。強さ `exaggeration`（0.3〜0.9）、`cfg_weight`（言語またぎは0推奨：手本の訛りが移るため） | 全体固定 → 行ごとに可変へ改造 |
| IndexTTS-2.5 | 声色＝中立な長めの手本、感情＝元の発言（`emo_vector`＋`emo_alpha`） | `tts/_index.py` は声色と感情に同じ区間を渡している → 分離が必要（契約確認後） |
| GPT-SoVITS | 手本音声のみ（同じ話者の感情別クリップを選ぶ） | — |

### 11.4 抑揚の後処理（どのエンジンでも使える補助）
- 話者ごとの基準値で正規化した**相対値だけ**を移す：①声の高さの平均シフトと幅の倍率、②区間ごとの音量、③全体の話速比（±15%程度）
- 波形の細かい高さの動き（輪郭）はコピーしない（日本語はピッチアクセント、中国語は声調で意味が変わるため）
- 英⇔独・西・仏の吹替データでも、元と訳の高さの動きの相関は 0.23〜0.25 と低い（arXiv 2608.27848）
- 感情は声質（息漏れ・張り）にも出るので、後処理は強さの微調整にとどめる
- 道具：pyworld（MIT）、librosa（ISC）、parselmouth/Praat（GPL-3.0）

### 11.5 感情の判定（案）
- 主：emotion2vec+（法務確認後）、副：SenseVoice（pyVideoTrans が既に使っている）。両者が食い違う、確率0.5未満、other/unknown の場合は「指定なし」
- 強さ＝0.5×（平常以外の確率）＋0.5×（話者の普段からのずれ：声の高さの幅・音量・話速）、上限0.7
- 会議の発言はほぼ平常のはず → 平常のときはトーン調整をしない

### 11.6 翻訳の指示文に渡す項目（案）
話者 / 原文 / 前2発言・後1発言 / 感情（種類・強さ・信頼度）/ 抑揚（話速・音量・高さの幅）/ 敬語の度合い / 用語集 / 最大文字数。
指示：「感情は語の選び方・文体・間投詞で表す。原文にない感情語は足さない。信頼度0.5未満は無視」。
出力は `{訳文, TTS用スタイル（英語8語以内）}` の形にし、スタイルを許可リストで確認してから TTS に渡す。

### 11.7 出典
- [CosyVoice（GitHub）](https://github.com/FunAudioLLM/CosyVoice) / [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) / [VoxCPM（GitHub）](https://github.com/OpenBMB/VoxCPM)
- [IndexTTS（GitHub, LICENSE）](https://github.com/index-tts/index-tts) / [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) / [IndexTTS-2.5 論文](https://arxiv.org/abs/2601.03888)
- [Chatterbox](https://github.com/resemble-ai/chatterbox) / [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / [F5-TTS](https://github.com/SWivid/F5-TTS) / [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) / [Higgs Audio v3](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) / [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) / [MOSS-TTS v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5)
- [emotion2vec+ large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) / [FunASR MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) / [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
- [audeering 次元感情モデル](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) / [吹替の抑揚相関（arXiv 2608.27848）](https://arxiv.org/abs/2608.27848)
- [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) / [parselmouth](https://pypi.org/project/praat-parselmouth) / [librosa](https://github.com/librosa/librosa)

---

## 12. 声の分解・声質変換・追加学習・類似度（2026-10-18 追記）

サブエージェント（sonnet）の調査。**太字はコードまたは公式ページで再確認済み**。SIM＝話者埋め込みのコサイン類似度（測定器・テスト集合が論文ごとに違うので数値の横比較は不可）。

### 12.1 結論
- 電話の「分解」の発想は有効。ただし最新のTTS/声質変換（VC）は内部で内容・声色・抑揚をすでに分けている。分解専用の符号化（FACodec 等）を単体で使うと類似度は低い。
- 実用的な使い方は「分解を道具にする」：①声の分析カルテ（測定器）、②**TTS（内容＋抑揚）→ VC（声色）の二段構成**。
- **前回似なかった最大の要因は「手本音声の取り方」の可能性が高い**（下記 12.2）。モデルを替えるより先に直す。
- 言語をまたぐと類似度は下がる（本人の実声どうしでも下がる）。「本人らしさ」と「なまりの少なさ」は綱引きの関係。
- 本人ごとのTTS追加学習は必ずしも効かない（Qwen3-TTS を1.5時間分で追加学習してもゼロショットと有意差なしの報告）。**良い手本＋VC側の本人ごとの学習**の方が費用対効果が高い見込み（推定）。

### 12.2 pyVideoTrans の手本音声の問題（コードで確認）
| 問題 | 場所 | 影響 |
|---|---|---|
| **手本＝字幕1行ごとの元音声** | `task/_stage_dubbing.py`、`task/_stage_recogn.py:_create_ref_from_vocal` | 1〜数秒・重なり発話・話者の割り当て違いがそのまま手本になる |
| **切り出しが 16kHz 固定** | `util/_ffmpeg_audio.py:cut_from_audio`（`-ar 16000`） | 24kHz 以上で動くTTSには情報（高音域）が足りない |
| 手本の元がノイズ除去・人声分離後の音声 | `_stage_recogn.py`、`_stage_prepare.py` | 処理のくせが類似度を下げうる |
| **音声の加速を既定で最大50倍まで許可** | `task/_rate.py`（`max_audio_speed_rate=50`）、`configure/_app_settings.py` | 大きく早回しすると声が変わる |

### 12.3 分解の方式
| 方式 | 分けるもの | ライセンス | 評価 |
|---|---|---|---|
| LPC/CELP（電話） | 声道フィルタ（響き）＋励振（声の高さ・残差） | 規格 | 圧縮が目的で、話者らしさが両方に散る。複製には使えない |
| WORLD | 声の高さ＋スペクトル包絡＋非周期性 | 修正BSD | **分析と「声の高さの統計合わせ」の後処理に有用** |
| FACodec（NaturalSpeech 3） | 内容／抑揚／声色／細部 | Apache-2.0 | VCとしての類似度は低い。英語のみで学習 |
| Vevo / Vevo2 | 内容・スタイル／声色 | **非商用（CC-BY-NC / NC-ND）** | 使わない |
| SpeechTokenizer / Mimi | 意味／音響 | Apache-2.0 / CC-BY-4.0 | 声色分離が目的ではない |
| Kanade（2026-02） | 内容＋話者埋め込み | (要検証) | 有望だが英語のみ |

### 12.4 合成後の声質変換（VC）
| モデル | 方式 | 言語またぎ | ライセンス | 所見 |
|---|---|---|---|---|
| **Seed-VC** | ゼロショット（手本1〜30秒）＋追加学習（最小1発話・T4で約2分） | 英中で SIM 0.52/0.42（同言語 0.56/0.73） | **GPL-3.0、2025-11 にアーカイブ（更新停止）** | 性能は良いが保守なし |
| RVC | 本人ごとの学習（10分以上の低雑音音声を推奨） | (要検証) | MIT | 元音声の声の高さを引き継ぐので高さの補正が必要 |
| OpenVoice v2 | ゼロショット | 日仏中英を含む6言語と主張 | MIT | 軽いが類似度は低め |
| kNN-VC | 学習不要、本人の音声を5分ほど集めて使う | (要検証) | (要検証) | 出力が16kHz。事前登録データを使う本件に合う |
| X-VC（2026-04） | ゼロショット | 英中のみ検証 | (要検証) | 高速。日本語は (要検証) |

### 12.5 本人ごとの追加学習（TTS）
| モデル | 言語 | 追加学習 | ライセンス・注意 |
|---|---|---|---|
| GPT-SoVITS | 日英中韓粤（**仏なし**） | 公式は「1分」から。数分で20〜40分の学習（12〜14GB GPU） | MIT |
| CosyVoice3-0.5B | 9言語 | 学習スクリプトあり。本人ごとの手順書なし | Apache-2.0。言語またぎの類似度 66〜68 vs 同言語 76〜79 |
| Qwen3-TTS-Base | 10言語 | **公式の単一話者向け追加学習あり**（音声・書き起こし・手本の組で学習） | Apache-2.0。同言語 SIM：日 0.788 / 仏 0.714 / 中 0.799 / 英 0.775 |
| Chatterbox Multilingual | 23言語 | 公式手順なし | MIT（透かし入り）。`cfg_weight=0` でなまりの継承を抑える |

### 12.6 似ない原因と対策（まとめ）
| 原因 | 対策 |
|---|---|
| 手本が短い／長すぎる | 話者ごとに10〜20秒のきれいな手本を数本登録し、比べて選ぶ |
| 雑音・反響・電話音質 | 登録は静かな部屋・口元マイク・48kHz。**手本にノイズ除去・人声分離をかけない** |
| サンプリングレート（16kHz固定） | 24kHz以上のまま切り出す |
| 手本の書き起こしの誤り | 登録手本には人が確認した原稿を付ける |
| 言語またぎ・なまり | 目標値は「同じ人の別言語の実声」で較正する |
| 話者分離の誤り | 手本は字幕区間からではなく登録音声から取る。区間は登録声との照合で外れを除く |
| 加速しすぎ | 上限1.2〜1.3倍。はみ出す分は訳を短くする |
| 音量のばらつき | 手本・出力とも話者ごとに音量をそろえる (要検証) |

### 12.7 類似度の測り方
- 測定器：WavLM-large＋ECAPA（SIM-o、TTS研究の標準）、ERes2NetV2（3D-Speaker、Apache-2.0）、Resemblyzer（Apache-2.0、値が高めに出る）
- 参考値（WavLM系）：本人の実声どうし（別の発言）英 0.73／中 0.75。TTS は同言語 0.64〜0.83、言語またぎ 0.45〜0.75
- 「同一人物」の絶対的な基準値は文献にない → **登録データの「同じ人どうし」と「別の人どうし」の類似度分布を測り、境目（EER点）で合格ラインを決める**
- 経験則：0.7前後なら実声並み、0.5未満は別人に聞こえる (要検証)
- 声の分析カルテの項目：声の高さ（中央値・四分位）、話速、長時間平均スペクトルとその傾き、フォルマント F1〜F3（母音別。言語またぎでは長時間平均スペクトルを優先）、CPPS（声質の明瞭さ）。ジッター・シマー・HNR はマイクや反響で変わるので比較には不向き。MCD は同じ文どうしでないと使えない

### 12.8 出典
- 分解：[FACodec](https://huggingface.co/amphion/naturalspeech3_facodec) / [Amphion](https://github.com/open-mmlab/Amphion) / [Vevo](https://huggingface.co/amphion/Vevo) / [Vevo2](https://huggingface.co/RMSnow/Vevo2) / [Kanade](https://arxiv.org/abs/2602.00594) / [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer) / [Mimi](https://huggingface.co/kyutai/mimi) / [WORLD](https://github.com/mmorise/World) / [CELP](https://en.wikipedia.org/wiki/Code-excited_linear_prediction)
- VC：[Seed-VC](https://github.com/Plachtaa/seed-vc) / [Seed-VC 評価](https://github.com/Plachtaa/seed-vc/blob/main/EVAL.md) / [X-VC](https://arxiv.org/abs/2604.12456) / [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) / [OpenVoice](https://github.com/myshell-ai/OpenVoice) / [kNN-VC](https://github.com/bshall/knn-vc)
- TTS：[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) / [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) / [Qwen3-TTS 論文](https://arxiv.org/abs/2601.15621) / [Qwen3-TTS 追加学習例（Baseten）](https://www.baseten.co/blog/fine-tuning-qwen3-tts-for-high-quality-voice-cloning/) / [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) / [Chatterbox](https://github.com/resemble-ai/chatterbox)
- 言語またぎ・手本：[IWSLT 2026](https://arxiv.org/abs/2604.26136) / [AAG](https://arxiv.org/abs/2609.29123) / [KIT](https://arxiv.org/abs/2606.07240) / [arXiv 2505.13830](https://arxiv.org/abs/2505.13830) / [arXiv 2602.05770](https://arxiv.org/abs/2602.05770)
- 評価：[Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval) / [3D-Speaker](https://github.com/modelscope/3D-Speaker) / [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) / [Parselmouth](https://github.com/YannickJadoul/Parselmouth)

---

## 13. 登録音声はどれだけ必要か（2026-10-18 追記）

サブエージェント（sonnet）の調査。**太字は公式ページで再確認済み**。

### 13.1 結論
- **10〜30分は不要**。まずは「同意文＋日本語の朗読で合計2〜3分（きれいな10〜15秒の手本を3〜4本）」で始め、測って足りなければ追加する。
- Sora の「数字を読む」録画は**主に本人確認（なりすまし防止）**のためで、声の複製に必要な量の根拠にはならない（報道・第三者解説ベース (要検証)）。
- **OpenAI 自身の API（Custom voices）でも、手本は30秒以下・推奨は「10〜30秒で完全な文を数文」・実際の発話5秒以上**。同意文の録音が必須（日本語の固定文あり）、利用は審査済み顧客のみ。

### 13.2 商用サービスの目安
| サービス | 即席の複製 | 本格的な複製 | 同意 |
|---|---|---|---|
| **OpenAI Custom voices** | **10〜30秒（30秒以下）** | — | **固定文の読み上げ必須** |
| ElevenLabs | 1〜2分（3分超は避ける） | 最低30分、理想2〜3時間 | — |
| Azure Personal Voice | 5〜90秒 | — | 本人の口頭同意が必須 |
| Google Chirp 3 Instant Custom Voice | 最大10秒 | — | 固定文の同意、許可制 |
| Apple Personal Voice（iOS 26） | 10フレーズ・約1分（日本語は非対応） | — | — |
| Resemble AI | 10秒〜1分 | 10分以上 | 同意文が必要 |

### 13.3 手本の長さと類似度の研究
- Mega-TTS 2：類似度 3秒 0.898 → 10秒 0.905 → 60秒 0.926 → 300秒 0.932（300秒で追加学習とほぼ同じ）。**長いほど上がるが、すぐに頭打ち**
- kNN-VC：5分で8分と同等。30秒未満は急に悪化
- 5.5秒→7.7秒で類似度が大きく改善（F5-TTS 0.24→0.35）。**ノイズ除去した手本は音質は上がるが類似度は下がる**（0.35→0.28）
- 言語またぎ（英語手本→日仏中等）：VoxCPM2 0.70、CosyVoice3 0.69、Qwen3-TTS 0.50（IWSLT 2026、小規模 (要検証)）

### 13.4 推奨する登録の手順（段階式）
**段階A：約3分の収録**
1. 同意文を固定文で読む（約10秒）。例：「私は○○です。○○が、会議の翻訳と音声読み上げのために、私の声から合成音声を作成・利用することに同意します。保存は○○まで、同意はいつでも撤回できます。」
2. 日本語を約90秒：音素のかたよりが少ない文を、会議の話し方で自然に読む（文の出典候補：JSUT basic5000。ATR503 は商用に許諾が必要）。数字だけの朗読は音素が足りず不向き
3. 外国語（英仏中）は最初は必須にしない（話せる人だけ各15秒、任意）
4. 収録条件：静かな部屋、口元10〜20cm、48kHz WAV・モノラル。**ノイズ除去・音声分離はしない**。感情の演技はまだ入れない
5. 保存：10〜15秒に切った手本3〜4本＋人が確認した書き起こし

**段階B：測る** — 手本に使わなかった本人の実発話との類似度を、各モデル・各言語で測る。手本は1本ずつ比べて最良を選ぶ。

**段階C：足りなければ追加**
- 毎回ぶれる → 手本を合計3〜5分に増やし、感情別・速さ別を足して最良を選ぶ
- 声色が系統的にずれる → 追加録音より本人ごとの追加学習（GPT-SoVITS 1分、Seed-VC 1発話）
- 同意済みの会議録音から、登録声と照合してきれいな区間を拾って足す
- 10〜30分は、追加学習で有意な差が出た主要な話者だけ

**長さごとに得られるもの**：30秒＝1本の複製は動くが選別・測定ができない／1分＝3〜4本から最良を選べる／5分＝本命（複数手本・感情別・評価用の取り分けが可能）／30分＝利得は小さく負担が大きい

### 13.5 出典
- [OpenAI Custom voices](https://developers.openai.com/api/docs/guides/custom-voices) / [Sora 2 発表（NBC）](https://www.nbcnews.com/tech/tech-news/openai-announces-sora-2-ai-video-audio-app-rcna234753) / [Sora 2 解説](https://www.theneuron.ai/explainer-articles/your-complete-guide-to-ai-video-memes-and-mayhem-with-openais-sora-2) / [Voice Engine](https://decrypt.co/224165/open-ai-voice-engine-clone-speech)
- [Google Chirp 3 Instant Custom Voice](https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice)
- [ElevenLabs IVC](https://elevenlabs.io/docs/creative-platform/voices/voice-cloning/instant-voice-cloning) / [ElevenLabs PVC](https://elevenlabs.io/docs/creative-platform/voices/voice-cloning/professional-voice-cloning)
- [Azure Personal Voice](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-overview) / [同意の作成](https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-create-consent) / [Apple Personal Voice（9to5Mac）](https://9to5mac.com/2026/03/19/ios-26-made-one-of-iphones-wildest-most-unique-features-a-lot-better/) / [Resemble](https://docs.app.resemble.ai/docs/resource_voice/create)
- [Mega-TTS 2](https://arxiv.org/html/2307.07218v4) / [kNN-VC](https://arxiv.org/pdf/2305.18975) / [Voicebox](https://arxiv.org/pdf/2306.15687) / [手本の長さ・ノイズ除去の影響](https://arxiv.org/html/2602.05770v1) / [IWSLT 2026 言語またぎ比較](https://aclanthology.org/2026.iwslt-1.12.pdf) / [JSUT](https://arxiv.org/pdf/1711.00354) / [ATR503](https://research.nii.ac.jp/src/en/ATR503.html)

---

## 14. 感情を TTS に渡す実装で分かったこと（2026-10-18 追記）

E4 の実装時の調査（サブエージェント sonnet）。**太字は公式コードで再確認済み**。

- **VoxCPM2**：指示はテキスト先頭の丸かっこ（例 `(slightly faster, cheerful tone)本文`）。Gradio アプリ（`app.py`）には別の `control_instruction` 欄があり、アプリ側で組み立てる。**書き起こし付きの複製（use_prompt_text＝Ultimate Cloning）のときは指示が捨てられる**（`app.py`：`actual_control = "" if use_prompt_text else control_instruction`）。→ 感情の指示を使う行は書き起こしなしの複製になり、**声の似かたが下がる可能性**がある。設定 `voxcpm_emotion_priority`（"emotion"／"similarity"）で切り替え、声の分析カルテで比べて決める。結果は実行ごとにぶれる（公式）。
- **CosyVoice3**：公式 webui には複製＋指示を同時に使う instruct2 のモードがない（4モード：预训练音色／3s极速复刻／跨语种复刻／自然语言控制）。「自然语言控制」は手本音声を無視し、初代 CosyVoice 専用。公式の感情指示は「开心・伤心・生气」の3種のみ（中国語の固定文、強さの段階なし）。公式 FastAPI の `/inference_instruct2` は引数の形が現行版と合わない見込み (要検証)。→ instruct2 を呼ぶ小さな API を別に立てる案（pyVideoTrans 側は `docs/meeting_emotion_tts_ja.md` 6章に骨格）。instruct2 は手本の音声トークンを捨てるので、声の似かたが下がる可能性。
- **ChatterBox**：感情の種類は指定できず、強さ（exaggeration）のみ。手本と出力の言語が違うときは cfg_weight=0（公式 README：手本のなまりの移りを防ぐ）。
- **pyVideoTrans 本家の不具合（未修正）**：VoxCPM の版の設定名が保存側 `voxcpm_version`、読み込み側 `voxcpmtts_version` で食い違い、常に v2 になる（コード上の確認のみ）。

出典：[VoxCPM2（HF）](https://huggingface.co/openbmb/VoxCPM2) / [VoxCPM（GitHub, app.py）](https://github.com/OpenBMB/VoxCPM/blob/main/app.py) / [VoxCPM cookbook](https://voxcpm.readthedocs.io/en/latest/cookbook.html) / [VoxCPM デモ](https://huggingface.co/spaces/openbmb/VoxCPM-Demo) / [CosyVoice（webui.py・common.py・frontend.py・runtime/python/fastapi/server.py）](https://github.com/FunAudioLLM/CosyVoice) / [Chatterbox](https://github.com/resemble-ai/chatterbox)

---

## 15. 声の分析カルテ（voicecheck）の実測（2026-10-18 追記）

`meeting/src/voicecheck` を作成（サブエージェント sonnet、opus がレビュー・再実行）。

### 15.1 公開音声での較正（LibriSpeech test-clean の40人・335発話、重複を除外）
| 埋め込み | 同じ人の平均 | 別の人の平均 | EER | おすすめしきい値（EER点） | 厳しめ（別人の通過1%） |
|---|---|---|---|---|---|
| ECAPA（speechbrain） | 0.692 | 0.085 | 1.13% | 0.323 | 0.329 |
| Resemblyzer | 0.850 | 0.548 | 4.32% | 0.691 | 0.742 |

- ECAPA は文献の「本人の実声どうし約0.7」と一致。
- **Resemblyzer は別人どうしでも平均0.55と高い** → 「0.5未満は別人」という経験則は当てはまらない。**しきい値は測定器ごとに違うので、必ず自分たちの登録データで較正する**。
- LibriSpeech の同じ人のペアは同じ録音セッション由来なので、EER はやや楽観的。
- 実音声48行の比較テスト：別の発話は100%合格、別人は0%、声を4半音上げたものは0%、1.3倍速は ECAPA 100%・Resemblyzer 33%（→ Resemblyzer は速度変化に弱い）。
- opus の再実行（4人×3発話）：ECAPA 同じ人 0.752／別の人 0.162、EER 0%、しきい値 0.507。少人数で分布が完全に分かれると「厳しめ」が「おすすめ」を下回る不具合を発見・修正（厳しめは EER 点以上にする）。

### 15.2 環境まわりの注意
- pyworld 0.3.5 は `pkg_resources` を使うため、setuptools 81 以降では読み込みに失敗し、黙って librosa.pyin に切り替わっていた → `setuptools<81` に固定。使った方式は Excel の「実行情報」に出る。
- pyworld の配布済みパッケージ（wheel）は Windows 用のみ。Mac・Linux はビルドが必要（Mac は `xcode-select --install`）。入らない場合は `uv sync --no-install-package pyworld` で pyin に代える。
- ECAPA のモデルは初回だけダウンロード（約85MB）、以後はオフラインで動く。
- 未確認：Mac・Windows での実行、Excel 本体での表示、日本語・仏語・中国語での話速の近似、言語をまたぐ類似度。

出典：[speechbrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) / [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) / [LibriSpeech 話者照合データ（HF）](https://huggingface.co/datasets/DynamicSuperb/SpeakerVerification_LibriSpeech-TestClean) / [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)
