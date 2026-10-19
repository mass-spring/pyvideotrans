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
import shutil
import tarfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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
# FLEURS (日本語。平常の読み上げ音声として使う)
# ---------------------------------------------------------------------------
FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"      # 2026-10-19 に確認した main の版 (取り直しても同じ音声になるように固定)
FLEURS_CONFIG = "ja_jp"
FLEURS_SPLITS = ("dev", "test", "train")
FLEURS_GENDER_JA = {"MALE": "FLEURS-男性", "FEMALE": "FLEURS-女性"}
_FLEURS_WAV = re.compile(r"^[0-9]+\.wav$")
_MAX_WAV_BYTES = 50 * 1024 * 1024       # 1 本の wav がこれより大きければ取り出さない (想定外のファイルを避ける)


@dataclass
class FleursRow:
    id: str
    file: str            # '11946010384058816161.wav'
    raw: str             # raw_transcription (句読点つきの元の文)
    text: str            # transcription (句読点なし、単語区切りの空白つき)
    n_samples: int
    gender: str          # MALE / FEMALE


def read_fleurs_tsv(path: str | Path) -> list[FleursRow]:
    """FLEURS の {split}.tsv (見出しなし、タブ区切り 7 列: id, file_name, raw_transcription, transcription, 字ごとの表記, num_samples, gender)。"""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"FLEURS の tsv が見つかりません: {p}")
    rows: list[FleursRow] = []
    seen: set[str] = set()
    with open(p, encoding="utf-8", newline="") as f:
        for n, r in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), start=1):
            if not r or not any(x.strip() for x in r):
                continue
            if len(r) != 7:
                raise InputError(f"{p.name} の {n} 行目が 7 列ではありません（{len(r)} 列）。FLEURS の tsv ではないかもしれません")
            file = r[1].strip()
            if not _FLEURS_WAV.match(file):
                raise InputError(f"{p.name} の {n} 行目: 音声のファイル名が想定外です: {file[:40]}")
            if file in seen:
                raise InputError(f"{p.name} の {n} 行目: 同じ音声が 2 回あります: {file}")
            seen.add(file)
            try:
                ns = int(r[5])
            except ValueError:
                ns = 0
            rows.append(FleursRow(r[0].strip(), file, r[2].strip(), r[3].strip(), ns, r[6].strip().upper()))
    if not rows:
        raise InputError(f"{p.name} が空です")
    return rows


def pick_fleurs(rows: list[FleursRow], max_clips: int, seed: int = 0) -> list[FleursRow]:
    """max_clips 件を、乱数の種 seed で選ぶ (同じ種なら毎回同じ件)。max_clips が 0 以下、または全件以下なら全件。元の並び順で返す。"""
    if max_clips <= 0 or len(rows) <= max_clips:
        return list(rows)
    chosen = set(random.Random(seed).sample(range(len(rows)), max_clips))
    return [r for i, r in enumerate(rows) if i in chosen]


def extract_wavs(tar_path: str | Path, wanted: set[str], out_dir: str | Path) -> set[str]:
    """tar.gz から、wanted に入っている wav だけを out_dir に取り出す。戻り値: 取り出せたファイル名。

    ダウンロードしたものは信用しない。tar の中のパスは使わず (展開先は out_dir の直下だけ)、'数字.wav' の通常ファイルで、
    wanted にあるものだけを書く。tar を丸ごと展開 (extractall) はしない。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    try:
        with tarfile.open(tar_path, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                base = PurePosixPath(m.name).name
                if base not in wanted or base in found or not _FLEURS_WAV.match(base):
                    continue
                if m.size > _MAX_WAV_BYTES:
                    raise InputError(f"{base} が大きすぎます（{m.size} バイト）。想定外のファイルなので取り出しません")
                src = tf.extractfile(m)
                if src is None:
                    continue
                (out / base).write_bytes(src.read())
                found.add(base)
                if found == wanted:
                    break
    except tarfile.TarError as e:
        raise InputError(f"tar.gz を読めませんでした: {tar_path}（{e}）") from None
    return found


def _hf_download(filename: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise InputError("huggingface_hub が入っていません。`uv pip install huggingface_hub` を実行してください（または --tsv と --tarball で手元のファイルを指定）") from None
    return Path(hf_hub_download(FLEURS_REPO, filename, repo_type="dataset", revision=FLEURS_REVISION, local_dir=str(local_dir)))


def build_fleurs(
    split: str, out_dir: str | Path, labels_csv: str | Path, max_clips: int = 300, seed: int = 0,
    tsv: str | Path | None = None, tarball: str | Path | None = None,
    download: Callable[[str, Path], Path] = _hf_download,
) -> dict[str, int]:
    """FLEURS 日本語の split から max_clips 件を選び、音声を out_dir に、ラベル CSV を labels_csv に作る (全部『平常』)。

    tsv / tarball を渡せば、それを使う (ダウンロードしない。渡したファイルは消さない)。渡さなければ Hugging Face から取り、
    音声の tar.gz は取り出した後に消す (この関数が作った一時フォルダごと)。
    戻り値: 件数の集計 (rows 全件数, picked, extracted, male, female)
    """
    if split not in FLEURS_SPLITS:
        raise InputError(f"--split は {', '.join(FLEURS_SPLITS)} のどれかにしてください: {split}")
    out = Path(out_dir)
    work: Path | None = None
    try:
        if tsv is None or tarball is None:
            work = out / ".download"
            work.mkdir(parents=True, exist_ok=True)
        tsv_path = Path(tsv) if tsv else download(f"data/{FLEURS_CONFIG}/{split}.tsv", work)       # type: ignore[arg-type]
        rows = read_fleurs_tsv(tsv_path)
        picked = pick_fleurs(rows, max_clips, seed)
        tar_path = Path(tarball) if tarball else download(f"data/{FLEURS_CONFIG}/audio/{split}.tar.gz", work)       # type: ignore[arg-type]
        got = extract_wavs(tar_path, {r.file for r in picked}, out)
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)       # この関数が作った一時フォルダだけを消す (ダウンロードした tar.gz もここ)
    missing = [r.file for r in picked if r.file not in got]
    if missing:
        raise InputError(f"tar.gz の中に見つからない音声が {len(missing)} 件あります。例: {', '.join(missing[:3])}")
    lab = Path(labels_csv)
    lab.parent.mkdir(parents=True, exist_ok=True)
    with open(lab, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "emotion", "text", "speaker", "style", "dataset"])
        for r in picked:
            w.writerow([r.file, "neutral", r.raw, FLEURS_GENDER_JA.get(r.gender, "FLEURS-" + (r.gender or "不明")), split, "fleurs"])
    return {"rows": len(rows), "picked": len(picked), "extracted": len(got),
            "male": sum(r.gender == "MALE" for r in picked), "female": sum(r.gender == "FEMALE" for r in picked)}


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
    "fleurs": {
        "name": "FLEURS 日本語（平常の読み上げ音声）",
        "info": [
            ("データの出典", ("FLEURS（Few-shot Learning Evaluation of Universal Representations of Speech）  Google。config は ja_jp（日本語）。"
                              "https://huggingface.co/datasets/google/fleurs  論文: Conneau et al., 2022, arXiv:2205.12446")),
            ("使った版", f"Hugging Face の {FLEURS_REPO}（リビジョン {FLEURS_REVISION}）の data/{FLEURS_CONFIG}/ の tsv と音声。どの分割（dev / test / train）かはラベル CSV の style 列"),
            ("データのライセンス", "CC BY 4.0（Hugging Face のデータセットカードの license: cc-by-4.0）。表示（出典・変更の明記）が必要。改変・商用利用も可"),
            ("データの内容", ("日本語の文章（FLoRes-101 の文。Wikinews・Wikijunior・Wikivoyage 由来（要検証））を、話者が読み上げた録音。感情の指定はなく、"
                            "この道具では全部『平常（neutral）』の正解として使う（人手で感情を確認したわけではない）。"
                            "1 つの文を複数の話者が読んでいる。FLEURS には話者 ID が無く、性別（MALE / FEMALE）だけが付いている")),
            ("利用の範囲", "公開データなので、クラウドで処理してもよいデータ（会社の会議音声ではない）"),
        ],
        "notes": [
            ("FLEURS の録音（話者・マイク・部屋・1 本の長さ（平均 約13秒））は JVNV（スタジオ録音、約数秒）と違います。"
             "モデルが『感情』ではなく『録音条件の違い』で平常と感情ありを見分けている可能性があり、平常の判定の良し悪しは、会議の音声にそのまま当てはまりません（要検証）。"),
            "FLEURS の日本語は男性の話者が大半です（dev・test とも約89%）。女性の平常の読み上げは少ないので、性別による違いは測れません。",
        ],
    },
}
