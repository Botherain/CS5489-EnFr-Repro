"""构造"教师上限"探测集：MTNT test 分层抽 300 句，只给源句（不带参考）。

目的：回答"还能提升多少"。用强模型翻译这 300 句并打 COMET-DA，
得到的分数就是当前管线在 MTNT 上的现实天花板参照。

输出：
  reports/ceiling/mtnt_300_input.jsonl   {id, src}
  reports/ceiling/mtnt_300_ref.jsonl     {id, src, tgt}   （只用于打分，不给模型看）
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "ceiling"

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="mtnt")
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
DOM, N = a.domain, a.n

rows = [json.loads(l) for l in (DATA / f"{DOM}.test.jsonl").open(encoding="utf-8") if l.strip()]
rng = random.Random(42)
# 按源句长度分层：短/中/长各 100
rows.sort(key=lambda r: len(r["src"].split()))
per = N // 3
buckets = [rows[: len(rows) // 3], rows[len(rows) // 3: 2 * len(rows) // 3], rows[2 * len(rows) // 3:]]
picked = []
for b in buckets:
    picked += rng.sample(b, min(per, len(b)))
rng.shuffle(picked)

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / f"{DOM}_{N}_input.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for r in picked:
        fh.write(json.dumps({"id": r["id"], "src": r["src"]}, ensure_ascii=False) + "\n")
with (OUT / f"{DOM}_{N}_ref.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for r in picked:
        fh.write(json.dumps({"id": r["id"], "src": r["src"], "tgt": r["tgt"]},
                            ensure_ascii=False) + "\n")

print(f"[ok] {len(picked)} 句 -> {OUT}")
print("长度分布:", [len(r["src"].split()) for r in picked[:5]], "…")
