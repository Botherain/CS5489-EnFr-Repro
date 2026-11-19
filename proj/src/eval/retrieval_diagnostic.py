"""诊断检索池质量：跨域误检率、相似度水平、检索结果与查询的长度匹配。

问题：seen 池是把 MTNT/EMEA/clean 三个域混在一起建的，没有按域门控。
      EMEA 的查询可能检索到 MTNT 的口语片段，反之亦然——这类"域错配"
      会直接污染 prompt，是 RAG 质量的系统性问题，值得先量化。
"""
from __future__ import annotations

import io
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    # id -> domain
    dom = {}
    for f in ("mtnt.train.jsonl", "emea.train.jsonl", "clean.train.jsonl"):
        for r in load(f):
            dom[r["id"]] = r["domain"]

    cache = ROOT / "data" / "index" / "cache" / "seen.top12.jsonl"
    if not cache.exists():
        raise SystemExit(f"缺检索缓存：{cache}")
    picks: dict[str, list[str]] = {}
    with cache.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                picks[d["id"]] = [e["id"] for e in d["examples"]]

    # 查询侧：用 test 的 id -> domain 映射（缓存里覆盖了 test+dev）
    qdom = {}
    for f in ("mtnt.test.jsonl", "emea.test.jsonl", "clean.test.jsonl",
              "mtnt.dev.jsonl", "emea.dev.jsonl", "clean.dev.jsonl"):
        p = DATA / f
        if p.exists():
            for r in load(f):
                qdom[r["id"]] = r["domain"]

    cross = Counter()
    total = Counter()
    per_query: dict[str, Counter] = defaultdict(Counter)
    for qid, ex_ids in picks.items():
        qd = qdom.get(qid)
        if not qd:
            continue
        for eid in ex_ids:
            ed = dom.get(eid)
            if not ed:
                continue
            total[qd] += 1
            per_query[qd][ed] += 1
            if ed != qd:
                cross[qd] += 1

    print("检索结果的域来源分布（查询域 → 检索到的域）：")
    for qd in ("mtnt", "emea", "clean"):
        if not total[qd]:
            continue
        dist = per_query[qd]
        s = "  ".join(f"{k}:{100*v/total[qd]:.0f}%" for k, v in dist.most_common())
        print(f"  {qd:6s} 共 {total[qd]} 条 →  {s}   跨域误检率 {100*cross[qd]/total[qd]:.1f}%")

    # 长度失配：检索到的示例比查询长多少
    q_len, e_len = {}, {}
    for f in ("mtnt.test.jsonl", "emea.test.jsonl", "clean.test.jsonl", "emea.train.jsonl",
              "mtnt.train.jsonl", "clean.train.jsonl"):
        p = DATA / f
        if not p.exists():
            continue
        for r in load(f):
            (q_len if r["id"] in qdom else e_len)[r["id"]] = len(r["src"].split())

    ratios = []
    for qid, ex_ids in picks.items():
        if qid not in q_len:
            continue
        ls = [e_len[i] for i in ex_ids if i in e_len]
        if ls and q_len[qid]:
            ratios.append(statistics.mean(ls) / q_len[qid])
    if ratios:
        ratios.sort()
        print(f"\n检索示例长度 / 查询长度: 中位 {statistics.median(ratios):.2f}，"
              f"p10 {ratios[int(0.1*len(ratios))]:.2f}，p90 {ratios[int(0.9*len(ratios))]:.2f}")
        print("  >3 倍的组数:", sum(1 for r in ratios if r > 3),
              f"/{len(ratios)}（这些等于往 prompt 里塞了远长于原句的内容）")


if __name__ == "__main__":
    main()
