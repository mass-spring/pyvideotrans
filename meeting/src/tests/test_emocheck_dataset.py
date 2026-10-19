"""emocheck のラベル CSV の読み込み・件の抽出・JVNV からのラベル CSV づくり"""
from __future__ import annotations

from collections import Counter

import pytest
import soundfile as sf
from asrcheck.ioformats import InputError

from emocheck import dataset
from synth import SR


def make_jvnv(tmp_path, speakers=("F1", "M1"), styles=("regular",), n=3, drop_text=None):
    """小さな JVNV もどき: F1/F1_anger_regular_01.wav ... と transcription_no_nv.csv"""
    root = tmp_path / "jvnv"
    lines = []
    for s in speakers:
        (root / s).mkdir(parents=True, exist_ok=True)
        for emo in dataset.JVNV_EMOTIONS:
            for st in styles:
                for k in range(1, n + 1):
                    sf.write(root / s / f"{s}_{emo}_{st}_{k:02d}.wav", [0.0] * 160, SR)
    for emo in dataset.JVNV_EMOTIONS:
        for st in styles:
            for k in range(1, n + 1):
                key = f"{emo}_{st}_{k:02d}"
                if key != drop_text:
                    lines.append(f"{key}|{emo}の文章{k}。")
    (root / "transcription_no_nv.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_build_jvnv_labels_and_read_back(tmp_path):
    root = make_jvnv(tmp_path, styles=("regular", "free"), n=2, drop_text="sad_free_02")
    out = tmp_path / "labels.csv"
    stats = dataset.build_jvnv_labels(root, out)
    assert stats["wavs"] == 2 * 6 * 2 * 2 and stats["no_text"] == 2   # sad_free_02 は F1・M1 の2人分
    clips = dataset.read_labels(out, root)
    assert len(clips) == stats["wavs"]
    c = next(x for x in clips if x.file == "F1/F1_anger_regular_01.wav")
    assert (c.emotion, c.speaker, c.style, c.text) == ("angry", "F1", "regular", "angerの文章1。")
    assert c.path == root / "F1" / "F1_anger_regular_01.wav"
    assert Counter(x.emotion for x in clips)["fearful"] == 8     # JVNV の fear → fearful
    assert not any(x.emotion == "neutral" for x in clips)
    # BOM 付き UTF-8 (Excel でそのまま開ける)
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


def test_build_jvnv_labels_errors(tmp_path):
    with pytest.raises(InputError, match="transcription_no_nv.csv"):
        dataset.build_jvnv_labels(tmp_path, tmp_path / "x.csv")
    root = make_jvnv(tmp_path)
    (root / "F1" / "oops.wav").write_bytes(b"")
    with pytest.raises(InputError, match="読めません"):
        dataset.build_jvnv_labels(root, tmp_path / "x.csv")


def test_read_jvnv_transcripts_bad_line(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("anger_regular_01|あ\nおかしい行\n", encoding="utf-8")
    with pytest.raises(InputError, match="2 行目"):
        dataset.read_jvnv_transcripts(p)


def test_read_labels_japanese_headers_and_encodings(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    p = tmp_path / "l.csv"
    p.write_text("ファイル,感情,発言,話者\na.wav,喜び,こんにちは,A\n", encoding="cp932")
    (clip,) = dataset.read_labels(p, tmp_path)
    assert (clip.emotion, clip.text, clip.speaker) == ("happy", "こんにちは", "A")


def test_read_labels_errors(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"")
    p = tmp_path / "l.csv"
    with pytest.raises(InputError, match="見つかりません"):
        dataset.read_labels(p, tmp_path)
    p.write_text("file,text\na.wav,x\n", encoding="utf-8")
    with pytest.raises(InputError, match="emotion"):
        dataset.read_labels(p, tmp_path)
    p.write_text("file,emotion\na.wav,ぷんぷん\n", encoding="utf-8")
    with pytest.raises(InputError, match="2 行目"):
        dataset.read_labels(p, tmp_path)
    p.write_text("file,emotion\na.wav,happy\na.wav,sad\n", encoding="utf-8")
    with pytest.raises(InputError, match="2回"):
        dataset.read_labels(p, tmp_path)
    p.write_text("file,emotion\nnone1.wav,happy\nnone2.wav,sad\n", encoding="utf-8")
    with pytest.raises(InputError, match="2 件見つかりません"):
        dataset.read_labels(p, tmp_path)
    assert len(dataset.read_labels(p, tmp_path, check_files=False)) == 2   # 音声の存在確認を省ける
    p.write_text("file,emotion\n", encoding="utf-8")
    with pytest.raises(InputError, match="音声の行がありません"):
        dataset.read_labels(p, tmp_path)


def _clips(spk_emo_counts):
    out = []
    for (spk, emo), n in spk_emo_counts.items():
        for k in range(n):
            out.append(dataset.Clip(f"{spk}_{emo}_{k}.wav", None, emo, "", spk))
    return out


def test_stratified_subset_is_per_group_deterministic_and_keeps_order():
    clips = _clips({("F1", "happy"): 12, ("F1", "sad"): 3, ("M1", "happy"): 20, ("M1", "sad"): 10})
    a = dataset.stratified_subset(clips, 5, seed=1)
    b = dataset.stratified_subset(clips, 5, seed=1)
    assert [c.file for c in a] == [c.file for c in b]
    assert Counter((c.speaker, c.emotion) for c in a) == {("F1", "happy"): 5, ("F1", "sad"): 3, ("M1", "happy"): 5, ("M1", "sad"): 5}
    order = [c.file for c in clips]
    assert [c.file for c in a] == sorted((c.file for c in a), key=order.index)    # 元の並び順
    assert [c.file for c in dataset.stratified_subset(clips, 5, seed=2)] != [c.file for c in a]
    assert len(dataset.stratified_subset(clips, 0)) == len(clips)
    assert len(dataset.stratified_subset(clips, 100)) == len(clips)


def test_describe_counts():
    clips = _clips({("A", "happy"): 2, ("A", "sad"): 1})
    assert dataset.describe_counts(clips) == "喜び 2・悲しみ 1"


def test_jvnv_preset_has_attribution():
    info = dict(dataset.DATASET_PRESETS["jvnv"]["info"])
    assert "CC BY-SA 4.0" in info["データのライセンス"] and "litagin/jvnv_corpus_v1_no_nv" in info["使った版"]
    assert "平常" in info["データの内容"]


def test_read_labels_dataset_column_and_override(tmp_path):
    for n in ("x.wav", "y.wav"):
        sf.write(tmp_path / n, [0.0] * 160, SR)
    (tmp_path / "l.csv").write_text("file,emotion,text,dataset\nx.wav,喜び,あ,corpusA\ny.wav,sad,い,\n", encoding="utf-8")
    clips = dataset.read_labels(tmp_path / "l.csv", tmp_path)
    assert [c.dataset for c in clips] == ["corpusA", ""]
    assert [c.dataset for c in dataset.read_labels(tmp_path / "l.csv", tmp_path, dataset="jvnv")] == ["jvnv", "jvnv"]     # 渡した名前が優先
    (tmp_path / "m.csv").write_text("file,emotion\nx.wav,happy\n", encoding="utf-8")
    assert dataset.read_labels(tmp_path / "m.csv", tmp_path)[0].dataset == ""                                           # 列が無くても読める


def test_read_sets_combines_datasets_and_rejects_clashes(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    sf.write(tmp_path / "a" / "x.wav", [0.0] * 160, SR)
    sf.write(tmp_path / "b" / "y.wav", [0.0] * 160, SR)
    sf.write(tmp_path / "b" / "x.wav", [0.0] * 160, SR)
    (tmp_path / "a.csv").write_text("file,emotion\nx.wav,happy\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("file,emotion\ny.wav,neutral\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("file,emotion\nx.wav,neutral\n", encoding="utf-8")
    clips = dataset.read_sets([("jvnv", tmp_path / "a", tmp_path / "a.csv"), ("fleurs", tmp_path / "b", tmp_path / "b.csv")])
    assert [(c.file, c.dataset, c.emotion) for c in clips] == [("x.wav", "jvnv", "happy"), ("y.wav", "fleurs", "neutral")]
    assert clips[1].path == tmp_path / "b" / "y.wav"
    with pytest.raises(InputError, match="同じ file 名が 2 つのデータにあります: x.wav"):
        dataset.read_sets([("jvnv", tmp_path / "a", tmp_path / "a.csv"), ("other", tmp_path / "b", tmp_path / "c.csv")])
    with pytest.raises(InputError, match="名前が重なっています"):
        dataset.read_sets([("a", tmp_path / "a", tmp_path / "a.csv"), ("a", tmp_path / "b", tmp_path / "b.csv")])


def test_fleurs_preset_has_license_and_domain_shift_note():
    p = dataset.DATASET_PRESETS["fleurs"]
    info = dict(p["info"])
    assert "CC BY 4.0" in info["データのライセンス"] and "google/fleurs" in info["データの出典"] and dataset.FLEURS_REVISION in info["使った版"]
    assert "平常" in info["データの内容"] and "公開データ" in info["利用の範囲"]
    assert any("録音" in n and "JVNV" in n for n in p["notes"]) and any("男性" in n for n in p["notes"])
