"""asrcheck.ioformats: 正解・認識結果ファイルの読み込み (文字コード・見出し・時刻・SRT・pyVideoTrans・JSON・RTTM)"""
from __future__ import annotations

import datetime as dt
import json

import pytest
from openpyxl import Workbook

from asrcheck.cli import parse_spec
from asrcheck.ioformats import (
    InputError,
    parse_time,
    read_hypothesis,
    read_json_utts,
    read_pvt,
    read_rttm,
    read_srt,
    read_utts_table,
)

CSV = "start,end,speaker,text\n0.0,2.5,Aさん,こんにちは、世界\n3,5.25,Bさん,\"はい, よろしく\"\n"


def write(path, text, enc="utf-8"):
    path.write_bytes(text.encode(enc))
    return path


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "cp932", "utf-16"])
def test_reference_csv_encodings(tmp_path, enc):
    p = write(tmp_path / "ref.csv", CSV, enc)
    utts, notes, got = read_utts_table(p)
    assert [(u.start, u.end, u.speaker, u.text) for u in utts] == [(0.0, 2.5, "Aさん", "こんにちは、世界"), (3.0, 5.25, "Bさん", "はい, よろしく")]
    assert notes == []


def test_reference_tab_separated_unicode_text_from_excel(tmp_path):
    # Excel の「Unicode テキスト」で保存した形 (UTF-16 + タブ区切り)
    p = write(tmp_path / "ref.txt", "start\tend\tspeaker\ttext\n0\t2\tAさん\tこんにちは、世界\n2\t4\tBさん\tはい\n", "utf-16")
    utts, _, _ = read_utts_table(p)
    assert [(u.speaker, u.text) for u in utts] == [("Aさん", "こんにちは、世界"), ("Bさん", "はい")]


def test_japanese_headers_and_file_column(tmp_path):
    p = write(tmp_path / "ref.csv", "録音,開始,終了,話者,発言\nm1,0,1,Aさん,あ\nm2,1,2,Bさん,い\n", "utf-8-sig")
    utts, _, _ = read_utts_table(p)
    assert [(u.rec, u.speaker, u.text) for u in utts] == [("m1", "Aさん", "あ"), ("m2", "Bさん", "い")]


def test_reference_missing_columns_message(tmp_path):
    p = write(tmp_path / "ref.csv", "start,end,text\n0,1,あ\n")
    with pytest.raises(InputError) as e:
        read_utts_table(p)
    assert "speaker" in str(e.value)


def test_reference_bad_time_message(tmp_path):
    p = write(tmp_path / "ref.csv", "start,end,speaker,text\nabc,1,A,あ\n")
    with pytest.raises(InputError) as e:
        read_utts_table(p)
    assert "2 行目" in str(e.value)


def test_reference_notes_for_empty_speaker_and_bad_range(tmp_path):
    p = write(tmp_path / "ref.csv", "start,end,speaker,text\n0,1,,あ\n5,2,A,い\n")
    _, notes, _ = read_utts_table(p)
    assert any("speaker が空" in n for n in notes) and any("end は start より後" in n for n in notes)


def test_blank_rows_are_skipped_and_header_can_be_below_title(tmp_path):
    p = write(tmp_path / "ref.csv", "正解データ,,,\nstart,end,speaker,text\n\n0,1,A,あ\n,,,\n")
    utts, _, _ = read_utts_table(p)
    assert len(utts) == 1


def test_excel_reference_with_time_cells_and_sheet_name(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "メモ"
    ws.append(["関係ないシート"])
    ws2 = wb.create_sheet("参照")
    ws2.append(["開始", "終了", "話者", "発言"])
    ws2.append([dt.time(0, 0, 1, 500000), dt.timedelta(seconds=3), "Aさん", "こんにちは"])
    ws2.append([4, 5.5, "Bさん", 123])        # 数字だけのセル (発言が数値) も文字にする
    ws2.append(["0:00:06", "00:00:07,250", "Aさん", "あ"])
    p = tmp_path / "ref.xlsx"
    wb.save(p)
    utts, _, kind = read_utts_table(p)
    assert "参照" in kind
    assert [(u.start, u.end, u.text) for u in utts] == [(1.5, 3.0, "こんにちは"), (4.0, 5.5, "123"), (6.0, 7.25, "あ")]


def test_parse_time_variants():
    assert parse_time(3) == 3.0 and parse_time("12.5") == 12.5 and parse_time("12,5") == 12.5
    assert parse_time("1:02.5") == 62.5 and parse_time("00:01:02,345") == pytest.approx(62.345)
    assert parse_time(dt.time(1, 2, 3)) == 3723.0 and parse_time(dt.timedelta(minutes=2)) == 120.0
    assert parse_time("") is None and parse_time(None) is None
    with pytest.raises(InputError):
        parse_time("あ")


def test_ms_columns(tmp_path):
    p = write(tmp_path / "h.csv", "start_ms,end_ms,speaker,text\n1500,4000,x,あ\n")
    utts, _, _ = read_utts_table(p, role="hypothesis")
    assert (utts[0].start, utts[0].end) == (1.5, 4.0)


SRT = """1
00:00:00,000 --> 00:00:02,500
[spk0]こんにちは
世界

2
00:00:03,000 --> 00:00:05,250
はい
"""


def test_read_srt_joins_wrapped_lines(tmp_path):
    p = write(tmp_path / "a.srt", SRT.replace("[spk0]", ""))
    utts = read_srt(p)
    assert [(u.start, u.end, u.text) for u in utts] == [(0.0, 2.5, "こんにちは 世界"), (3.0, 5.25, "はい")]


def test_pyvideotrans_folder_with_speaker_json(tmp_path):
    (tmp_path / "ja.srt").write_text(SRT.replace("[spk0]", ""), encoding="utf-8")
    (tmp_path / "en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    (tmp_path / "speaker.json").write_text(json.dumps(["spk0", "spk1"]), encoding="utf-8")
    utts, notes = read_pvt(tmp_path, "ja")
    assert [(u.speaker, u.text) for u in utts] == [("spk0", "こんにちは 世界"), ("spk1", "はい")] and notes == []
    # srt を直接指定しても同じフォルダの speaker.json を使う
    utts2, _ = read_pvt(tmp_path / "ja.srt")
    assert [u.speaker for u in utts2] == ["spk0", "spk1"]


def test_pyvideotrans_folder_ambiguous_srt_message(tmp_path):
    for n in ("a.srt", "b.srt"):
        (tmp_path / n).write_text(SRT, encoding="utf-8")
    with pytest.raises(InputError) as e:
        read_pvt(tmp_path, "ja")
    assert "a.srt" in str(e.value) and "直接指定" in str(e.value)


def test_pyvideotrans_speaker_prefix_in_srt_text(tmp_path):
    p = write(tmp_path / "meeting.srt", SRT.replace("はい", "[spk1]はい"))
    utts, notes = read_pvt(p)
    assert [(u.speaker, u.text) for u in utts] == [("spk0", "こんにちは 世界"), ("spk1", "はい")] and notes == []


def test_pyvideotrans_speaker_json_length_mismatch_and_missing(tmp_path):
    (tmp_path / "ja.srt").write_text(SRT.replace("[spk0]", ""), encoding="utf-8")
    (tmp_path / "speaker.json").write_text(json.dumps(["spk0"]), encoding="utf-8")
    utts, notes = read_pvt(tmp_path)
    assert [u.speaker for u in utts] == ["spk0", ""] and any("行数" in n for n in notes)
    (tmp_path / "speaker.json").unlink()
    utts, notes = read_pvt(tmp_path)
    assert all(u.speaker == "" for u in utts) and any("話者情報がありません" in n and "--diarize" in n for n in notes)


def test_json_variants(tmp_path):
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps({"segments": [{"start": 0.5, "end": 2, "text": "あ", "speaker": "S1"}]}), encoding="utf-8")
    u = read_json_utts(p1)[0]
    assert (u.start, u.end, u.text, u.speaker) == (0.5, 2.0, "あ", "S1")
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps([{"start_time": 1500, "end_time": 4000, "text": "い"}]), encoding="utf-8")   # pyVideoTrans の字幕項目 (ミリ秒)
    u = read_json_utts(p2)[0]
    assert (u.start, u.end, u.speaker) == (1.5, 4.0, "")
    p3 = tmp_path / "c.json"
    p3.write_text("{}", encoding="utf-8")
    with pytest.raises(InputError):
        read_json_utts(p3)


def test_rttm(tmp_path):
    p = write(tmp_path / "a.rttm", "SPEAKER mtg 1 0.50 2.00 <NA> <NA> spk0 <NA> <NA>\nSPEAKER mtg 1 3.00 1.00 <NA> <NA> spk1 <NA> <NA>\n")
    utts = read_rttm(p)
    assert [(u.start, u.end, u.speaker, u.rec) for u in utts] == [(0.5, 2.5, "spk0", "mtg"), (3.0, 4.0, "spk1", "mtg")]
    utts, _, kind = read_hypothesis(p)
    assert "RTTM" in kind and utts[0].text == ""


def test_hypothesis_table_needs_text_or_speaker(tmp_path):
    p = write(tmp_path / "h.csv", "start,end\n0,1\n")
    with pytest.raises(InputError):
        read_hypothesis(p)
    p2 = write(tmp_path / "h2.csv", "text\nこんにちは\n")     # 時刻・話者がなくても読める
    utts, _, _ = read_hypothesis(p2)
    assert utts[0].start is None and utts[0].text == "こんにちは"


def test_missing_file_message(tmp_path):
    with pytest.raises(InputError) as e:
        read_utts_table(tmp_path / "nothing.csv")
    assert "見つかりません" in str(e.value)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("whisper=out/w.csv", ("whisper", "", "out/w.csv")),
        ("whisper@mock1=out/w.csv", ("whisper", "mock1", "out/w.csv")),
        ("out/w.csv", ("w", "", "out/w.csv")),
        ("output/demo/", ("demo", "", "output/demo/")),
        ("C:\\Users\\me\\ja.srt", ("ja", "", "C:\\Users\\me\\ja.srt")),
    ],
)
def test_parse_spec(spec, expected):
    assert parse_spec(spec) == expected
