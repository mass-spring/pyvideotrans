# 設計案：声の再現 ＋ 感情モジュール（pyVideoTrans 改造）

- 版: 案 v0.1（**承認待ち**）
- 作成日: 2026-10-18
- 根拠: `docs/research.md` 10〜12章（コード調査・感情とトーン制御・声の類似度）
- 計画との対応: `docs/PLAN.md` 5章（E0〜E6、V0〜V5）

---

## 0. 方針（3つ）

1. **測定器を先に作る**：「声の分析カルテ」で本人らしさ・感情・声の特徴を数値にし、現状値を測ってから改善する。
2. **声色の手本を替える**：今の「字幕1行ごとの元音声（16kHz・数秒・雑音入り）」から、**事前に登録した本人のきれいな手本**に替える。調査上、改善効果が一番大きい見込み。
3. **感情は「指示」で明示する**：言語をまたぐと手本の抑揚はほとんど移らないため、感情を TTS への指示（VoxCPM2・CosyVoice3）と翻訳の指示文の両方に渡す。

## 1. 声の分解と組み立て直し

| 要素 | 出どころ | 実現方法 |
|---|---|---|
| 内容 | 翻訳結果 | LLM 翻訳（感情を考慮） |
| **声色** | **本人の登録手本** | TTS の手本音声＋（任意で）合成後の声質変換（VC） |
| 抑揚 | 元の発言 | 感情指示＋軽い補正（声の高さの中央値・幅、音量を話者の統計に合わせる） |
| 感情 | 感情モジュール | TTS への指示（例：`(angry, tense, slightly faster)`） |

## 2. 処理の流れ

```
[事前] 話者の登録（1人10〜30分、静かな部屋・口元マイク・48kHz）
   → 話者プロファイル：手本クリップ（10〜20秒×3〜5本、書き起こし付き）
                       話者埋め込み、声の統計（高さ・話速・スペクトル）

[会議の録音ファイル]
 ① 文字起こし＋話者分離 ………………… pyVideoTrans（既存）
 ② ★話者照合 ……………………………… spk0, spk1… を登録者名に対応付け。合わない区間は「要確認」
 ③ ★感情検出 ……………………………… 行ごとに {感情, 強さ, 信頼度} → emotion.json
 ④ 翻訳 …………………………………… ★「行番号→感情」の表を指示文に追加。出力＝訳文＋TTS用スタイル
 ⑤ 音声合成 ……………………………… ★声色の手本＝登録手本、★感情＝指示
 ⑥（任意）★声質変換（VC） ………… 声色を本人に寄せる
 ⑦ ★抑揚の軽い補正 ……………………… 声の高さ・音量を話者の統計に合わせる
 ⑧ 長さ合わせ ……………………………… ★加速の上限を1.3倍に。超える行は訳を短くして作り直す
 ⑨ ★測定 …………………………………… 行ごと・話者ごとの類似度・感情一致・声の統計 → Excel
```
★＝新規または改造

## 3. 新しく作る部品（本体とは別ファイル）

| 部品 | 役割 | 置き場所（案） |
|---|---|---|
| **voicecheck（測定器）** | 類似度・声の統計・感情一致を測って Excel に出す。pyVideoTrans なしでも単体で動く | `src/voicecheck/` |
| voiceprofile（話者の登録） | 登録音声から手本クリップを選び、埋め込み・統計を作る | `src/voiceprofile/` |
| 感情検出 | 行ごとの感情を出す | `videotrans/task/_stage_emotion.py`、`videotrans/process/_audio_emotion.py` |
| 抑揚の補正 | 声の高さ・音量の補正 | `videotrans/util/_emotion_prosody.py` |

## 4. pyVideoTrans 本体の改造箇所（最小限）

| # | ファイル | 変更内容 |
|---|---|---|
| 1 | `util/_ffmpeg_audio.py:cut_from_audio` | サンプリングレートを引数にする（手本用は 44.1/48kHz） |
| 2 | `task/_stage_dubbing.py`（行ごとの辞書を作る部分） | 手本音声・手本の書き起こしを**話者プロファイルの手本に差し替え**（設定で切替）。感情の項目を追加。配音キャッシュのキーに感情を含める |
| 3 | `task/_stage_translate.py`、`translator/_base.py`、`translator/_openaicompat.py`、`prompts/{srt,text}/*.txt` | 指示文に `{emotion_info}` を追加（本文には埋め込まない）。翻訳キャッシュのキーに感情を含める |
| 4 | `process/_stt_utils.py` | SenseVoice の感情タグを捨てる前に保存する |
| 5 | `tts/_voxcpm.py`、`tts/_cosyvoice.py` | 行ごとに感情の指示を渡す（CosyVoice は instruct2 を使う改造） |
| 6 | `cli.py` | 画面なしの実行でも話者分離を有効にできるようにする（今は動かない） |
| 7 | `task/taskcfg.py`、設定ファイル | `enable_emotion`、話者プロファイルの場所、加速の上限（`max_audio_speed_rate=1.3`）などの設定 |

注意：既存テスト（`SrtItem` の項目数、TransCreate の構成、CLI の引数）を合わせて更新する。

## 5. データの形

**話者プロファイル** `profiles/<名前>/profile.json`
```json
{
  "name": "Aさん",
  "refs": [{"wav": "refs/a_01.wav", "text": "（人が確認した書き起こし）", "lang": "ja", "sec": 14.2, "snr_db": 32.1}],
  "embedding": {"model": "ecapa", "file": "emb.npy"},
  "stats": {"f0_median_hz": 128.0, "f0_iqr_semitone": 4.1, "rate_mora_per_sec": 7.2, "ltas_tilt_db": -11.3}
}
```

実装（`meeting/src/voiceprofile`）では、上の項目に加えて refs ごとに score・text_status・consistency、プロファイル全体に stats・embedding・eval（評価用の取り置き）・enrollment を持つ。pyVideoTrans 側は必須項目だけを読む（research.md 19章）。

**感情** `emotion.json`（行の開始時刻で対応付け。行番号だけに頼らない）
```json
[{"start_time": 12340, "end_time": 15880, "label": "angry", "intensity": 0.6, "confidence": 0.72}]
```

**声の分析カルテ** `voice_report.xlsx`
- シート1「行ごと」：No／話者／原文／訳文／感情（元）／感情（合成）／類似度（登録手本との）／声の高さの差／話速の比／加速率／要確認
- シート2「話者ごと」：平均類似度、ばらつき、合格率
- シート3「合格ラインの較正」：同じ人どうし・別の人どうしの類似度分布と、その境目

## 6. 改善の実験の順番（1回に1つだけ変えて効果を測る）

| # | 変えること | 期待効果（調査からの見込み） |
|---|---|---|
| 0 | 何も変えない（現状） | 現状値 |
| 1 | 声色の手本を登録手本（48kHz・10〜20秒）に替える | **大** |
| 2 | 加速の上限を1.3倍にする | 中 (要検証) |
| 3 | TTS エンジンを比べる：VoxCPM2／CosyVoice3／Qwen3-TTS／Chatterbox | 不明 → 測る |
| 4 | 合成後の声質変換を足す：RVC／OpenVoice v2／kNN-VC | 中 |
| 5 | 感情の指示を ON | トーン（感情一致率） |
| 6 | 抑揚の補正を ON | 小 |

良かったものを組み合わせて、最後にもう一度全体で測る。

## 7. 評価と合格ラインの決め方

- **客観**：話者埋め込みのコサイン類似度を2種類の測定器で測る（例：ECAPA と WavLM 系）。
- **合格ライン**：登録データで「同じ人の実声どうし」と「別の人どうし」の類似度分布を測り、その境目（EER点）を基準にする。参考：実声どうしは約0.7、0.5未満は別人に聞こえやすい (要検証)。
- **主観**：本人と同僚が、どの方式か伏せた状態で5段階評価する。
- **感情**：元の発言と合成音声に同じ感情判定をかけ、一致率を見る（PLAN 5.3）。

## 8. 使うモデルとライセンス（採用候補のみ）

| 用途 | モデル | ライセンス | 注意 |
|---|---|---|---|
| TTS | VoxCPM2 | Apache-2.0 | — |
| TTS | CosyVoice3 | Apache-2.0 | Alibaba製（Q5） |
| TTS | Qwen3-TTS-Base | Apache-2.0 | Alibaba製（Q5）。指示との併用は不可 |
| TTS | Chatterbox Multilingual | MIT | 出力に透かし |
| TTS（追加学習） | GPT-SoVITS | MIT | 仏語なし |
| 声質変換 | RVC / OpenVoice v2 | MIT / MIT | — |
| 声質変換 | Seed-VC | GPL-3.0 | 2025-11 に更新停止 → 後回し |
| 感情 | SenseVoiceSmall | FunASR 独自 | 出典表記で商用可（メンテナ見解）。Alibaba製 |
| 感情 | emotion2vec+ | FunASR 独自 | **法務確認後に使う** |
| 分析 | pyworld / librosa / parselmouth | MIT / ISC / GPL-3.0 | — |
| 類似度 | Resemblyzer / 3D-Speaker | Apache-2.0 / Apache-2.0 | — |

**使わない**：F5-TTS・OmniVoice・Higgs Audio・Vevo（非商用）、IndexTTS（大企業は別契約）、audeering の感情モデル（非商用）。

## 9. 最初に作るもの：声の分析カルテ（測定器）

- 置き場所：`src/voicecheck/`（pyVideoTrans なしで単体実行）
- 入力：登録音声のフォルダと、比べたい音声のフォルダ（または pyVideoTrans の出力）
- 出力：`voice_report.xlsx`
- CPU だけで動く（Mac・Surface・このクラウド環境で開発とテストができる）
- 動作確認には公開音声（例：Common Voice＝CC0）を使い、会社の音声は使わない

## 10. 承認してほしい点

1. 方針（測定器を先に作る・声色の手本を差し替える・感情は指示で明示する）
2. 最初に「声の分析カルテ」を作ること
3. 実験の順番（6章）
