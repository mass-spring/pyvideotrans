"""pairs.csv の読み込み。"""
from __future__ import annotations

import pytest

from voicecheck.pairs import PairsError, emotion_match, read_pairs, split_refs


def write(tmp_path, text, encoding="utf-8"):
    p = tmp_path / "pairs.csv"
    p.write_bytes(text.encode(encoding))
    return p


def test_multiple_refs_and_optional_columns(tmp_path):
    p = write(
        tmp_path,
        "speaker,ref,test,source,label,emotion_src,emotion_test\n"
        'A,ref/a1.wav;ref/a2.wav; ref/a3.wav ,out/t1.wav,src/s1.wav,baseline,angry,Angry\n'
        "B,ref/b1.wav,out/t2.wav,,profile_ref,,\n"
        "\n",
    )
    rows = read_pairs(p)
    assert len(rows) == 2                      # 空行は無視
    a, b = rows
    assert a.speaker == "A" and a.label == "baseline" and a.line == 2
    assert [r.rsplit("/", 1)[-1] for r in a.refs] == ["a1.wav", "a2.wav", "a3.wav"]
    assert a.test.endswith("out/t1.wav") and a.source.endswith("src/s1.wav")
    assert a.refs[0].startswith(str(tmp_path))  # 相対パスは pairs.csv のあるフォルダ基準
    assert (a.emotion_src, a.emotion_test) == ("angry", "Angry")
    assert len(b.refs) == 1 and b.source == "" and b.label == "profile_ref"
    assert not a.problems and not b.problems


def test_required_columns_only(tmp_path):
    rows = read_pairs(write(tmp_path, "speaker,ref,test\nA,r.wav,t.wav\n"))
    assert rows[0].label == "" and rows[0].source == "" and rows[0].emotion_src == ""


def test_bom_cp932_and_japanese_headers(tmp_path):
    text = "話者,手本,評価,ラベル\nAさん,r1.wav;r2.wav,t.wav,条件1\n"
    for enc in ("utf-8-sig", "cp932"):
        rows = read_pairs(write(tmp_path, text, enc))
        assert rows[0].speaker == "Aさん" and rows[0].label == "条件1" and len(rows[0].refs) == 2


def test_absolute_paths_are_kept(tmp_path):
    rows = read_pairs(write(tmp_path, f"speaker,ref,test\nA,{tmp_path}/r.wav,{tmp_path}/t.wav\n"))
    assert rows[0].refs == [str(tmp_path / "r.wav")] and rows[0].test == str(tmp_path / "t.wav")


def test_missing_required_column_gives_clear_error(tmp_path):
    with pytest.raises(PairsError) as e:
        read_pairs(write(tmp_path, "speaker,ref\nA,r.wav\n"))
    assert "test" in str(e.value)


def test_missing_file_and_empty_file(tmp_path):
    with pytest.raises(PairsError):
        read_pairs(tmp_path / "nope.csv")
    with pytest.raises(PairsError):
        read_pairs(write(tmp_path, ""))


def test_empty_cells_become_problems_not_crashes(tmp_path):
    rows = read_pairs(write(tmp_path, "speaker,ref,test\nA,,t.wav\n,r.wav,\n"))
    assert any("ref" in p for p in rows[0].problems)
    assert any("speaker" in p for p in rows[1].problems) and any("test" in p for p in rows[1].problems)


def test_split_refs_and_emotion_match():
    assert split_refs("a.wav; b.wav；c.wav;") == ["a.wav", "b.wav", "c.wav"]
    assert emotion_match("Angry ", "angry") is True
    assert emotion_match("angry", "joy") is False
    assert emotion_match("", "joy") is None and emotion_match("angry", "") is None
