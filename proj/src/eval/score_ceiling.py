"""测量"教师上限"：对同一批抽样句，比较教师与本项目的各系统。

必须同集合比较：教师只翻了 n 句，不能拿它的分数和全量 test 的分数对比。

用法（comet venv）：
  python src/eval/score_ceiling.py --domain mtnt --n 300
  python src/eval/score_ceiling.py --domain emea --n 150
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
CEIL = ROOT / "reports" / "ceiling"
RUNS = ROOT / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="mtnt")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dom, n = args.domain, args.n

    teacher_path = CEIL / f"{dom}_{n}_teacher.jsonl"
    ref_path = CEIL / f"{dom}_{n}_ref.jsonl"
    if not teacher_path.exists():
        raise SystemExit(f"缺教师译文：{teacher_path}")

    refs = {json.loads(l)["id"]: json.loads(l)
            for l in ref_path.open(encoding="utf-8") if l.strip()}
    teacher = {json.loads(l)["id"]: (json.loads(l).get("pred") or "").strip()
               for l in teacher_path.open(encoding="utf-8") if l.strip()}

    out_dir = RUNS / f"_ceiling_teacher_{dom}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pass1.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for rid, r in refs.items():
            fh.write(json.dumps({"id": rid, "domain": dom, "src": r["src"], "tgt": r["tgt"],
                                 "pred": teacher.get(rid, ""), "prompt_tokens": 0,
                                 "gen_tokens": 0, "batch_ms_per_sentence": 0},
                                ensure_ascii=False) + "\n")
    empty = sum(1 for v in teacher.values() if not v)
    print(f"[prep] {len(refs)} 句，空译文 {empty}", flush=True)

    from comet import download_model, load_from_checkpoint
    da = load_from_checkpoint(str(download_model("Unbabel/wmt22-comet-da")))
    data = [{"src": refs[i]["src"], "mt": teacher.get(i, ""), "ref": refs[i]["tgt"]}
            for i in refs]
    try:
        res = da.predict(data, batch_size=32, gpus=1, progress_bar=False)
    except TypeError:
        res = da.predict(data, batch_size=32, accelerator="gpu", devices=1, progress_bar=False)
    scores = list(res[0] if isinstance(res, tuple)
                  else (getattr(res, "scores", None) or res["scores"]))
    seg = dict(zip(refs.keys(), [float(s) for s in scores]))
    teacher_mean = statistics.mean(seg.values())
    print(f"\n[TEACHER:{dom}] COMET-DA = {teacher_mean:.4f}  (n={len(seg)})", flush=True)

    (CEIL / f"teacher_scores_{dom}.json").write_text(json.dumps(
        {"domain": dom, "teacher_da": round(teacher_mean, 4), "n": len(seg),
         "per_id": {k: round(v, 5) for k, v in seg.items()}}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    print("\n同一批句子上与本项目各系统的对比：", flush=True)
    for run in ("g1_zs", "g2_rag", "g3_sft", "g4_sft_rag"):
        path = RUNS / run / "scores.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        sel = [r["comet_da"] for r in rows if r["id"] in seg and r["domain"] == dom]
        if sel:
            m = statistics.mean(sel)
            print(f"  {run:14s} {m:.4f}   （距教师 {teacher_mean - m:+.4f}）", flush=True)


if __name__ == "__main__":
    main()
