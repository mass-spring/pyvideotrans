"""登録音声との照合の測定をまとめて実行して Excel にする（enroll-report コマンドの中身）。

流れ: 埋め込み（enroll-embed で保存済み）→ 条件ごとに照合・採点（結果は results.jsonl に追記。続きから再開）→ 集計 → Excel。
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import enroll_assign as ea
from . import enroll_eval as ee
from . import enroll_excel as ex
from . import reportinfo
from .enroll_embed import EMBEDDERS, meta_path, store_path
from .pipeline import Layout

REJECT_PCTS = (1.0, 5.0, 10.0, 20.0)
UNK_SEC, UNK_MODE = 180, "mixed"       # 不明・平滑化は『登録180秒・感情を混ぜる・平均』で見る


def available_embedders(lay: Layout, wanted: list[str] | None = None) -> list[str]:
    names = wanted or list(EMBEDDERS)
    return [n for n in names if store_path(lay, n).is_file()]


def extra_specs(embedders: list[str], seeds, variants=("A", "B")) -> list[ee.Spec]:
    out = []
    for e in embedders:
        for v in variants:
            for sd in seeds:
                for p in REJECT_PCTS:
                    out.append(ee.Spec(e, UNK_SEC, UNK_MODE, "mean", sd, v, "silero", unk_pct=p))
                for sm in ("groups", "short"):
                    out.append(ee.Spec(e, UNK_SEC, UNK_MODE, "mean", sd, v, "silero", smooth=sm))
    return out


def reject_rows(bench: ee.Bench, agg: dict, embedders: list[str], seeds, variants=("A", "B")) -> list[dict]:
    rows = []
    for e in embedders:
        for v in variants:
            for p in REJECT_PCTS:
                cur = []
                for sd in seeds:
                    s = ee.Spec(e, UNK_SEC, UNK_MODE, "mean", sd, v, "silero")
                    labels, best, _m, _t = bench.assign(s)
                    truth = bench.truth[(v, "silero")]
                    correct = np.array([a == b for a, b in zip(labels, truth)])
                    thr = ea.threshold_from_enrollment(bench.enrollment(e, UNK_MODE, UNK_SEC, sd), p, "mean")
                    cur.append(ea.reject_curve(best, correct, [thr])[0])
                a_plain = agg.get(ee_key(e, v, None, ""))
                a_unk = agg.get(ee_key(e, v, p, ""))

                def mean(k):
                    xs = [c[k] for c in cur if c[k] is not None]
                    return float(np.mean(xs)) if xs else None
                rows.append({"embedder": e, "variant": v, "pct": p, "rejected_pct": mean("rejected_pct"), "accuracy_accepted": mean("accuracy_accepted"),
                             "wrong_among_rejected": mean("wrong_among_rejected"), "errors_caught_pct": mean("errors_caught_pct"),
                             "acc_plain": a_plain["spk_acc"] if a_plain else None, "acc_unk": a_unk["spk_acc"] if a_unk else None,
                             "der_plain": a_plain["der_0"] if a_plain else None, "der_unk": a_unk["der_0"] if a_unk else None})
    return rows


def ee_key(e: str, v: str, unk, smooth: str) -> tuple:
    return (e, UNK_SEC, UNK_MODE, "mean", v, "silero", unk, smooth)


def smooth_rows(bench: ee.Bench, agg: dict, embedders: list[str], seeds, variants=("A", "B")) -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    names = {"": "なし（基準）", "groups": "隙間0〜0.15秒でつながる区間（最大3つ）をまとめて判定", "short": "1秒未満の短い区間は、つながる前後の区間の名前を引き継ぐ"}
    for e in embedders:
        for v in variants:
            items = bench.items[(v, "silero")]
            dur = np.array([(it["end_ms"] - it["start_ms"]) / 1000 for it in items])
            for sm in ("", "groups", "short"):
                a = agg.get(ee_key(e, v, None, sm))
                changed = []
                for sd in seeds:
                    base = bench.assign(ee.Spec(e, UNK_SEC, UNK_MODE, "mean", sd, v, "silero"))[0]
                    cur = bench.assign(ee.Spec(e, UNK_SEC, UNK_MODE, "mean", sd, v, "silero", smooth=sm))[0] if sm else base
                    changed.append(sum(1 for x, y in zip(base, cur) if x != y))
                if a:
                    rows.append({"embedder": e, "variant": v, "name": names[sm], "spk_acc": a["spk_acc"], "der_0": a["der_0"], "cpcer": a["cpcer"],
                                 "note": f"基準から名前が変わった区間 平均 {np.mean(changed):.1f} 個" if sm else f"区間 {len(items)} 個"})
            if e == embedders[0]:
                notes.append(f"録音{v}・silero の区間の長さ: 1秒未満 {int((dur < 1).sum())} 個、1.5秒未満 {int((dur < 1.5).sum())} 個、2秒未満 {int((dur < 2).sum())} 個"
                             f"（全 {len(dur)} 個）。")
    for v in variants:
        items, truth = bench.items[(v, "silero")], bench.truth[(v, "silero")]
        pairs = [(i, i + 1) for i in range(len(items) - 1) if 0 <= items[i + 1]["start_ms"] - items[i]["end_ms"] <= 150]
        same = sum(1 for i, j in pairs if truth[i] == truth[j])
        notes.append(f"録音{v}・silero で、隙間が0〜0.15秒でつながる区間の組は {len(pairs)} 組、そのうち正解の話者が同じだったのは {same} 組"
                     + (f"（{same / len(pairs) * 100:.0f}%）。" if pairs else "。"))
    return rows, notes


def window_rows(bench: ee.Bench, agg: dict, embedders: list[str], seeds, variants=("A", "B")) -> list[dict]:
    """区間の中を窓で判定して切り直した場合（窓の埋め込みがある埋め込み器だけ）。字幕の区間ごとに1人の名前を付ける基準と並べる。"""
    rows = []
    for e in embedders:
        for v in variants:
            rs = [bench.evaluate_windows(e, v, UNK_SEC, UNK_MODE, sd) for sd in seeds]
            if any(r is None for r in rs):
                continue
            base = agg.get(ee_key(e, v, None, ""))
            if not base:
                continue

            def mean(k):
                xs = [r[k] for r in rs if r.get(k) is not None]
                return float(np.mean(xs)) if xs else None
            rows.append({"embedder": e, "variant": v, "acc_base": base["spk_acc"], "acc_win": mean("spk_acc"), "der0_base": base["der_0"], "der0_win": mean("der_0"),
                         "der25_base": base["der_0.25"], "der25_win": mean("der_0.25"), "segments": rs[0]["segments"], "pieces": mean("pieces"),
                         "split_segments": mean("split_segments")})
    return rows


def build_info(bench: ee.Bench, lay: Layout, command: str, embedders: list[str], specs_n: int, seeds, secs, wall: dict) -> list[tuple[str, object]]:
    from datetime import datetime

    manifest = reportinfo.load_manifest(lay)
    repo = reportinfo.pvt_bridge.ensure_pvt_importable()
    commit = reportinfo.git(["rev-parse", "--short", "HEAD"], repo)
    changed = reportinfo.git(["status", "--porcelain", "--", "videotrans"], repo)
    used = ea.used_in_meeting(manifest)
    items: list[tuple[str, object]] = [
        ("ツール", "meetingbench enroll-embed / enroll-report（採点は asrcheck）"),
        ("作成日時", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("コマンド", command),
        ("目的", "PLAN §4.1 ②『事前登録した声と照合して名前を付ける』（D14: 参加者ごとに約3分の登録）を、教師なしの話者分離 built と同じ条件で比べる"),
        ("pyVideoTrans 本体", f"リポジトリ {repo}（commit {commit}）。本体 videotrans/ の未コミット変更: {'なし' if not changed else changed}。この測定は本体の関数を呼ばない"
                           "（内蔵 onnx の埋め込みモデルのファイルを sherpa-onnx で読むだけ）。built の比較値は meetingbench の既存の測定（runs/<A|B>/cond）"),
        ("疑似会議", "meetingbench build で作った A（重ならない 32.5分）/ B（約12.5%が0.3〜1.0秒重なる 31.8分）。4人・275発言。区間・文字起こしは既存の測定と同じ"
                  "（正解の区間 oracle / pyVideoTrans 既定の silero、SenseVoiceSmall）"),
        ("登録の作り方", f"会議で使っていない JVNV の発話だけを使う（manifest.json の 275 発言ぶんを除外。除外数 {len(used)}）。話者×感情ごとに乱数（種 {ea.POOL_SEED}）で並べ替え、"
                      f"元の長さの合計が {ea.POOL_SEC_PER_EMOTION:.0f} 秒以上になるまで取った {bench.meta[embedders[0]].get('pool_clips')} 本を登録の候補にした。"
                      "候補から、目標の長さ（" + "・".join(str(s) for s in secs) + "秒/人）になるまで、感情を順番に回しながら1本ずつ足す。"
                      "『感情を混ぜる』は6感情、『落ち着いた感情だけ』は喜び・悲しみ。選び方を変えた" + f"{len(list(seeds))}回（seed {', '.join(map(str, seeds))}）の平均を出す。"
                      "登録発話にも、会議を作ったときと同じ処理（前後の無音カット・RMS -24dBFS・16kHz）をかけた。登録の文と会議の文は別だが、4人が同じ文を読むコーパスなので、"
                      "ある人の登録に使った文を別の人が会議で読んでいる場合がある（本人の声の漏れではない）。"),
        ("照合のしかた", "区間の埋め込みと、各登録者の（長さ1に正規化した）埋め込みの平均とのコサイン類似度が最大の人の名前を付ける（平均との類似度）。"
                    "もう一つは、登録した各発話との類似度の最大（最大）。不明のしきい値は、登録発話を1本ずつ抜いた本人への点数の下位X%（会議の正解は使わない）。"
                    "区間の埋め込みは、前後の無音を落として（librosa.effects.trim, 40dB）から計算。0.5秒未満は繰り返して0.5秒にする。"),
        ("採点", "asrcheck（CER は NFKC・句読点除去・数字を算用数字にそろえる等の正規化後。DER の collar は 0 と 0.25。話者の対応は Hungarian 法で自動）。"
               "出力の話者名は F1/F2/M1/M2（登録名）。名前をそのまま比べた話者正解率と、対応づけが名前どおりかを全指標シートに載せた。"
               "CER は話者に依らないので区間ごとに1回だけ計算し、cpCER・DER・話者正解率を条件ごとに計算（近道が asrcheck.evaluate_recording と同じ値になることはテストで確認）"),
        ("条件の数", f"{specs_n} 条件（埋め込み器 {len(embedders)} × 登録の長さ {len(list(secs))} × 感情 2 × 方式 2 × seed {len(list(seeds))} × 録音 2 × 区間 2、ほか不明・平滑化）"),
        ("JVNV の出典・ライセンス・表記", reportinfo.JVNV_ATTRIBUTION + "\n登録に使った発話も同じ JVNV（CC BY-SA 4.0）。登録用の音声・埋め込みは meeting/data/ にあり Git に入れない。"),
    ]
    for e in embedders:
        m = bench.meta[e]
        info = EMBEDDERS[e]
        v = ", ".join(f"{k} {x}" for k, x in (m.get("versions") or {}).items())
        sha = f" / SHA-256 {m['model_sha256']} / {m['model_bytes']:,} バイト" if m.get("model_sha256") else ""
        items.append((f"埋め込み器 {e}", f"{info.label}。{m.get('dim')} 次元。モデル: {m.get('model_file')}{sha}。{v}（Python {m.get('python')}、CPU {m.get('threads')} スレッド）。"
                                          f"ライセンス: {info.license}。入手元: {info.origin}"))
        en = (m.get("enroll") or {}).get("all")
        if en:
            items.append((f"処理時間 {e}: 登録候補", f"{en['n']} 本・音声 {en['audio_sec']:.0f} 秒 を {en['elapsed_sec']:.0f} 秒で埋め込み（RTF {en['elapsed_sec'] / en['audio_sec']:.3f}）。"
                                                    f"1人が登録する約180秒ぶんなら約 {en['elapsed_sec'] / en['audio_sec'] * 180:.0f} 秒"))
        for k, g in sorted((m.get("segments") or {}).items()):
            v_, seg = k.split(":")
            aud = bench.audio_s[v_]
            items.append((f"処理時間 {e}: 録音{v_} 区間={seg}", f"{g['n']} 区間（音声 {g['audio_sec']:.0f} 秒）の埋め込みに {g['elapsed_sec']:.0f} 秒"
                                                             f"（区間の音声に対する RTF {g['elapsed_sec'] / g['audio_sec']:.3f}、録音全体 {aud:.0f} 秒に対する RTF {g['elapsed_sec'] / aud:.3f}、"
                                                             f"1時間の会議に換算 {g['elapsed_sec'] / aud * 60:.1f} 分）。照合（類似度と最大値）は 1 録音あたり 1 ミリ秒未満"))
    for e in embedders:
        for k, g in sorted(((bench.meta[e].get("windows")) or {}).items()):
            v_, seg = k.split(":")
            items.append((f"処理時間 {e}: 録音{v_} 窓", f"{g['n']} 個の窓（{ea.WIN_MS}ms、{ea.HOP_MS}ms ずつ）の埋め込みに {g['elapsed_sec']:.0f} 秒（窓の音声に対する RTF {g['elapsed_sec'] / g['audio_sec']:.3f}）"))
    items += [
        ("built との差", "built は『区間分け + 文字起こし + 話者分離』、ここは『区間分け + 文字起こし + 話者埋め込み』。区間分けと文字起こしの時間は既存の測定の値をそのまま使った"),
        ("ライブラリのバージョン（この集計を実行した環境）", reportinfo.pkg_versions()),
        ("Python / OS / CPU", f"Python {platform.python_version()} / {platform.platform()} / {reportinfo.cpu_model()} / {os.cpu_count()} コア（GPU なし）"),
        ("計測時の注意", "同じ機械で TTS の生成など別のジョブが動いており CPU を取り合った状態で測った。埋め込みは CPU 1 スレッドに固定し nice で優先度を下げて実行。"
                    "処理時間・RTF は参考値で、KPI の基準機（Mac）とは別。"),
        ("モデルのライセンス（要検証を含む）",
         "ECAPA: speechbrain/spkrec-ecapa-voxceleb は Hugging Face のモデルカードの license 欄が apache-2.0（学習データは VoxCeleb1+2。データの利用条件は (要検証)）。\n"
         "TitaNet small: NGC のモデルカードに『NeMo ツールキットのライセンスに従う』とある。sherpa-onnx 用 onnx（pyVideoTrans が持つもの。配布元 ModelScope himyworld/videotrans）の再配布条件は (要検証)。\n"
         "ERes2Net large / CAM++: 3D-Speaker のコードは Apache-2.0（リポジトリで確認）。事前学習済み重みのライセンスはリポジトリに記載がなく (要検証)。\n"
         "いずれも商用利用できるかを、採用前に条文で確認する（CLAUDE.md の方針）。"),
    ]
    return items


def run_report(lay: Layout, out: Path | str, embedders: list[str] | None = None, secs=(10, 30, 60, 180), seeds=range(5),
               say=print, force: bool = False, command: str = "") -> Path:
    import time

    t0 = time.perf_counter()
    embs = available_embedders(lay, embedders)
    if not embs:
        raise FileNotFoundError("埋め込みがありません。先に enroll-embed --embedder ... を実行してください")
    seeds = list(seeds)
    bench = ee.Bench(lay, embs)
    main = ee.make_grid(embs, secs=secs, seeds=seeds)
    extra = extra_specs(embs, seeds)
    allspecs = main + extra
    done = ee.run_grid(bench, allspecs, say=say, force=force)
    wanted = {s.key for s in allspecs}
    results = [done[k] for k in done if k in wanted]
    agg = ee.aggregate(results)
    say("比べる相手（built）を読み込みます ...")
    base = ee.load_baselines(lay)
    # 話者混同の表に載せる条件: built(zh) と、各埋め込み器の（180秒・感情を混ぜる・平均・seed 0）
    engines = []
    for v in ("A", "B"):
        c = base.conds.get((v, "silero__built-zh__n4"))
        if c:
            engines.append(c.sub)
        for e in embs:
            engines.append(bench.evaluate(ee.Spec(e, 180, "mixed", "mean", seeds[0], v, "silero"), keep_engine=True)["engine"])
    extras = {"reject": reject_rows(bench, agg, embs, seeds), "window": window_rows(bench, agg, embs, seeds)}
    sr, sn = smooth_rows(bench, agg, embs, seeds)
    extras["smooth"], extras["smooth_notes"] = sr, sn
    d = ex.Data(agg, base, bench, results, extras)
    manifest = reportinfo.load_manifest(lay)
    recs = {v: {"minutes": manifest["variants"][v]["audio_sec"] / 60, "turns": manifest["variants"][v]["turns"],
                "overlapped": manifest["variants"][v]["overlapped_turns"]} for v in ("A", "B") if v in manifest.get("variants", {})}
    info = build_info(bench, lay, command or "python -m meetingbench enroll-report", embs, len(allspecs), seeds, secs, {})
    info.append(("この集計にかかった時間", f"{time.perf_counter() - t0:.0f} 秒（採点を含む。結果が results.jsonl にあれば飛ばす）"))
    p = ex.write_enroll_workbook(out, d, info, recs, engines)
    say(f"完了: {Path(p).resolve()}")
    return Path(p)
