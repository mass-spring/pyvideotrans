"""測る音声の一覧 (ラベル CSV) の読み込みと、JVNV コーパスからのラベル CSV づくり。

ラベル CSV の形 (1行 = 1つの音声ファイル。UTF-8 / cp932 / UTF-16 を自動判別):

    file,emotion,text,speaker,style
    F1/F1_anger_regular_01.wav,anger,どうしてこんなに荒々しい態度をとるんだ？,F1,regular

  - file     音声ファイル。--audio-dir からの相対パス (または絶対パス)。見出しは 録音・ファイル でも可
  - emotion  正解の感情。anger / happy / sad / fear / disgust / surprise / neutral、喜び・怒り・悲しみ… でも可 (labels.parse_label)
  - text     正解の書き起こし。見出しは 発言・テキスト でも可。空でもよい (その行は CER に数えない)
  - speaker  (任意) 話者。『話者ごとの正解率』を出すのに使う
  - style    (任意) 読み方の種類など、好きな区分。『区分ごとの正解率』を出すのに使う
"""
from __future__ import annotations

import csv
import io
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from asrcheck.ioformats import ALIASES as _ASR_ALIASES
from asrcheck.ioformats import InputError, read_text

from .labels import EMOTIONS, JA, parse_label

HEADER_ALIASES = {
    "file": _ASR_ALIASES["file"],
    "emotion": ["emotion", "label", "emo", "感情", "正解の感情", "感情ラベル"],
    "text": _ASR_ALIASES["text"],
    "speaker": _ASR_ALIASES["speaker"],
    "style": ["style", "group", "kind", "種類", "区分", "スタイル", "グループ"],
    "dataset": ["dataset", "corpus", "データ", "データセット", "コーパス"],
}
_LOOKUP = {a.casefold(): k for k, v in HEADER_ALIASES.items() for a in v}


@dataclass
class Clip:
    file: str          # CSV に書かれた名前 (結果の表にもこの名前で出す)
    path: Path         # 実際の場所
    emotion: str       # 正解の感情 (labels.EMOTIONS のどれか)
    text: str = ""     # 正解の書き起こし
    speaker: str = ""
    style: str = ""
    dataset: str = ""  # どのデータ (jvnv / fleurs など) の音声か。ラベル CSV の dataset 列、または read_labels(dataset=...) で決まる


def read_labels(labels_csv: str | Path, audio_dir: str | Path, check_files: bool = True, dataset: str = "") -> list[Clip]:
    """ラベル CSV を読んで Clip のリストにする。音声が見つからないものがあれば、まとめてエラーにする。

    dataset を渡すと、全部の Clip にその名前を付ける (CSV に dataset 列があっても、渡した名前を使う)。
    """
    p = Path(labels_csv)
    if not p.is_file():
        raise InputError(f"ラベル CSV が見つかりません: {p}\n  JVNV なら先に: python -m emocheck prepare-jvnv --jvnv-dir ... --out ...")
    raw, _ = read_text(p)
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise InputError(f"ラベル CSV が空です: {p}")
    head = [_LOOKUP.get(h.strip().casefold(), h.strip()) for h in rows[0]]
    for need in ("file", "emotion"):
        if need not in head:
            raise InputError(
                f"ラベル CSV に '{need}' の列がありません (見出し: {rows[0]})。\n  1行目に file,emotion,text を書いてください (text は空でも可)"
            )
    base = Path(audio_dir)
    clips: list[Clip] = []
    missing: list[str] = []
    seen: set[str] = set()
    for n, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue
        d = {k: (row[i].strip() if i < len(row) else "") for i, k in enumerate(head)}
        name = d.get("file", "")
        if not name:
            raise InputError(f"{p.name} の {n} 行目: file が空です")
        if name in seen:
            raise InputError(f"{p.name} の {n} 行目: 同じ file が2回あります: {name}")
        seen.add(name)
        try:
            emo = parse_label(d.get("emotion", ""))
        except ValueError as e:
            raise InputError(f"{p.name} の {n} 行目: {e}") from None
        path = Path(name) if Path(name).is_absolute() else base / name
        if check_files and not path.is_file():
            missing.append(name)
        clips.append(Clip(name, path, emo, d.get("text", ""), d.get("speaker", ""), d.get("style", ""), dataset or d.get("dataset", "")))
    if not clips:
        raise InputError(f"ラベル CSV に音声の行がありません: {p}")
    if missing:
        shown = "\n  ".join(missing[:5])
        raise InputError(
            f"音声ファイルが {len(missing)} 件見つかりません (--audio-dir = {base})。例:\n  {shown}"
            + ("\n  ..." if len(missing) > 5 else "")
        )
    return clips


def read_sets(sets: list[tuple[str, str | Path, str | Path]]) -> list[Clip]:
    """複数のデータ [(名前, 音声フォルダ, ラベル CSV), ...] をまとめて読む。各 Clip の dataset に名前が入る。

    途中結果 (キャッシュ) は file の名前で引くので、別のデータで同じ file 名があると混ざる。見つけたらエラーにする。
    """
    names = [n for n, _, _ in sets]
    if len(set(names)) != len(names):
        raise InputError(f"データの名前が重なっています: {names}")
    clips: list[Clip] = []
    owner: dict[str, str] = {}
    for name, audio_dir, labels_csv in sets:
        for c in read_labels(labels_csv, audio_dir, dataset=name):
            if c.file in owner:
                raise InputError(f"同じ file 名が 2 つのデータにあります: {c.file}（{owner[c.file]} と {name}）。途中結果が混ざるので、どちらかの file 名を変えてください")
            owner[c.file] = name
            clips.append(c)
    return clips


def stratified_subset(clips: list[Clip], per_group: int, seed: int = 0) -> list[Clip]:
    """『話者 × 感情』の組み合わせごとに per_group 件ずつ選ぶ (話者が無ければ感情ごと)。

    乱数の種 (seed) が同じなら毎回同じ選び方。足りないグループは全部入れる。元の並び順で返す。
    per_group が 0 以下なら全件。
    """
    if per_group <= 0:
        return list(clips)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, c in enumerate(clips):
        groups.setdefault((c.speaker, c.emotion), []).append(i)
    rng = random.Random(seed)
    chosen: set[int] = set()
    for key in sorted(groups):
        idx = groups[key]
        chosen.update(idx if len(idx) <= per_group else rng.sample(idx, per_group))
    return [c for i, c in enumerate(clips) if i in chosen]


def describe_counts(clips: list[Clip]) -> str:
    """『喜び 100・怒り 80…』のような件数の説明。"""
    cnt = Counter(c.emotion for c in clips)
    return "・".join(f"{JA[e]} {cnt[e]}" for e in EMOTIONS if cnt[e])


# ---------------------------------------------------------------------------
# JVNV
# ---------------------------------------------------------------------------
JVNV_EMOTIONS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
_JVNV_FILE = re.compile(r"^(?P<spk>[FM]\d+)_(?P<emo>[a-z]+)_(?P<style>[a-z]+)_(?P<no>\d+)\.wav$")
JVNV_TRANSCRIPT = "transcription_no_nv.csv"


def read_jvnv_transcripts(path: Path) -> dict[str, str]:
    """transcription_no_nv.csv ('anger_regular_01|本文' の形、見出しなし、'|' 区切り) → {'anger_regular_01': '本文'}。"""
    raw, _ = read_text(path)
    out: dict[str, str] = {}
    for n, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        key, sep, text = line.partition("|")
        if not sep:
            raise InputError(f"{path.name} の {n} 行目が 'ID|本文' の形ではありません: {line[:40]}")
        out[key.strip()] = text.strip()
    return out


def build_jvnv_labels(jvnv_dir: str | Path, out_csv: str | Path) -> dict[str, int]:
    """JVNV (litagin/jvnv_corpus_v1_no_nv の形: F1/ F2/ M1/ M2/ と transcription_no_nv.csv) から ラベル CSV を作る。

    感情・話者・読み方 (regular/free) は ファイル名 'F1_anger_regular_01.wav' から、本文は transcription_no_nv.csv から取る。
    戻り値: 件数の集計 (wavs, written, no_text, unused_text)
    """
    root = Path(jvnv_dir)
    tr = root / JVNV_TRANSCRIPT
    if not tr.is_file():
        raise InputError(f"{tr} が見つかりません。--jvnv-dir には F1/ F2/ M1/ M2/ と {JVNV_TRANSCRIPT} があるフォルダを指定してください")
    texts = read_jvnv_transcripts(tr)
    wavs = sorted(root.glob("[FM][0-9]*/*.wav"))
    if not wavs:
        raise InputError(f"{root} の下に F1/*.wav などが見つかりません")
    rows: list[list[str]] = []
    used: set[str] = set()
    no_text = 0
    for w in wavs:
        m = _JVNV_FILE.match(w.name)
        if not m or m["emo"] not in JVNV_EMOTIONS:
            raise InputError(f"JVNV のファイル名として読めません: {w.relative_to(root)}  (例: F1_anger_regular_01.wav)")
        key = f"{m['emo']}_{m['style']}_{m['no']}"
        text = texts.get(key)
        if text is None:
            no_text += 1
            text = ""
        else:
            used.add(key)
        rows.append([w.relative_to(root).as_posix(), m["emo"], text, m["spk"], m["style"]])
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:   # BOM 付き UTF-8 (Excel でそのまま開ける)
        w_ = csv.writer(f)
        w_.writerow(["file", "emotion", "text", "speaker", "style"])
        w_.writerows(rows)
    return {"wavs": len(wavs), "written": len(rows), "no_text": no_text, "unused_text": len(set(texts) - used)}


# ---------------------------------------------------------------------------
# データの出典・ライセンス (Excel の『実行情報』と『まとめ』に載せる)
# ---------------------------------------------------------------------------
DATASET_PRESETS: dict[str, dict] = {
    "jvnv": {
        "name": "JVNV（日本語の感情音声コーパス）",
        "info": [
            ("データの出典", "JVNV コーパス（言語音声と非言語音声を持つ日本語感情音声コーパス）  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus"),
            ("使った版", "非言語音声の区間を削除した版 litagin/jvnv_corpus_v1_no_nv（Hugging Face のデータセット。https://huggingface.co/datasets/litagin/jvnv_corpus_v1_no_nv ）"),
            ("データのライセンス", ("CC BY-SA 4.0（原コーパスのライセンスを継承）。表示（出典・変更の明記）と、同じ条件での共有が必要。"
                                  "この版は、原コーパスから非言語音声の区間を単純に切り取ったもので、書き起こしにも誤りがありうると配布元に注記がある")),
            ("データの内容", ("話者4人（F1・F2・M1・M2）× 感情6種（anger 怒り・disgust 嫌悪・fear 恐れ・happy 喜び・sad 悲しみ・surprise 驚き）。平常（neutral）は無い。"
                            "ファイル名の regular / free は読み方の区分。感情の正解はファイル名から取った")),
            ("利用の範囲", "公開データなので、クラウドで処理してもよいデータ（会社の会議音声ではない）"),
        ],
        "notes": [
            ("JVNV は、話者4人が決まった文章を感情を込めて読み上げた収録音声（演技）です。会議での自然な発言とは違い、感情が誇張されやすいので、"
             "会議の音声での正解率は、ここより低くなる可能性があります（要検証。模擬会議の音声で別に測る）。"),
            "4人は同じ文章を読んでいます。文章の内容（『許せない』『嬉しい』など）が感情の手がかりになっている可能性があり、声の調子だけで当てているとは限りません（要検証）。",
        ],
    },
}
