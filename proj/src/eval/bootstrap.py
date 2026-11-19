"""S9.2 配对 bootstrap + Holm 校正。

口径（EXPERIMENT_PLAN §6）：
  - COMET：句级分数可分解 -> 直接对句级差值重采样
  - BLEU/chrF++：不可句级分解 -> **重采样句对后重算语料级指标**再取差
  - MTNT 同 comment_id 的句子相关 -> 按 comment_id 分块重采样（聚类 bootstrap）
  - 双侧 p 值 + 比较族内 Holm 校正

用法：
  python src/eval/bootstrap.py --a g4_sft_rag --b g3_sft
  python src/eval/bootstrap.py --family main
  python src/eval/bootstrap.py --a X --b Y --metric bleu --iterations 1000
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CONCLUSION = ("mtnt", "emea")

FAMILIES = {
    "main": [("g3_sft", "g1_zs"), ("g4_sft_rag", "g3_sft"),
             ("g5_full", "g4_sft_rag"), ("g5_full", "g1_zs")],
    "ablation": [("g5_full", "g5_no_dpo"), ("g5_full", "g5_no_rag"),
                 ("g5_full", "g5_no_con"), ("g5_full", "g5_no_agent")],
    # 本次跑完矩阵后实际需要的比较族：G5 整体 vs 轻通道，以及四个模块的留一
    "g5": [("g5_full", "g3_sft"), ("g5_full", "g5_no_dpo"), ("g5_full", "g5_no_rag"),
           ("g5_full", "g5_no_rag_len"), ("g5_full", "g5_no_con"),
           ("g5_full", "g5_no_agent")],
}


def load_scores(run: str) -> dict[str, dict]:
    path = RUNS / run / "scores.jsonl"
    with path.open(encoding="utf-8") as fh:
        return {json.loads(l)["id"]: json.loads(l) for l in fh if l.strip()}


def comment_ids(domain: str) -> dict[str, str]:
    """MTNT 的 comment_id 分组键；EMEA 用句子自身（iid）。"""
    path = DATA / f"{domain}.test.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for l in fh:
            if l.strip():
                d = json.loads(l)
                out[d["id"]] = d.get("comment_id") or d["id"]
    return out


def paired(a: dict, b: dict, metric: str, iterations: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for dom in CONCLUSION:
        ids = [i for i in a if i in b and a[i]["domain"] == dom]
        if not ids:
            continue
        cid = comment_ids(dom)
        key = (lambda i: cid.get(i, i)) if dom == "mtnt" else (lambda i: i)
        for i in ids:
            groups.setdefault(f"{dom}|{key(i)}", []).append(i)
    keys = list(groups)

    point = statistics.mean(a[i][metric] - b[i][metric]
                           for g in groups.values() for i in g)

    diffs = []
    for _ in range(iterations):
        sample = [g for _ in keys for g in groups[rng.choice(keys)]]
        if not sample:
            continue
        diffs.append(statistics.mean(a[i][metric] - b[i][metric] for i in sample))
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    p = 2 * min(sum(1 for d in diffs if d <= 0) / len(diffs),
                sum(1 for d in diffs if d >= 0) / len(diffs))
    return {"metric": metric, "delta": round(point, 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "p_value": round(min(1.0, p), 4),
            "clusters": len(keys), "iterations": len(diffs)}


def corpus_bleu_delta(a: dict, b: dict, metric: str, iterations: int, seed: int = 42) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    rng = random.Random(seed)
    ids = [i for i in a if i in b]
    by_dom = {d: [i for i in ids if a[i]["domain"] == d] for d in CONCLUSION}
    ids = [i for d in CONCLUSION for i in by_dom[d]]

    def make():
        return BLEU(tokenize="13a") if metric == "bleu" else CHRF(char_order=6, word_order=2)

    def score(sub):
        vals_a, vals_b = [], []
        for d in CONCLUSION:
            sel = [i for i in sub if a[i]["domain"] == d]
            if not sel:
                continue
            refs = [[a[i]["tgt"] for i in sel]]
            vals_a.append(make().corpus_score([a[i]["pred"] for i in sel], refs).score)
            vals_b.append(make().corpus_score([b[i]["pred"] for i in sel], refs).score)
        return statistics.mean(vals_a) - statistics.mean(vals_b)

    point = score(ids)
    diffs = sorted(score([ids[rng.randrange(len(ids))] for _ in range(len(ids))])
                   for _ in range(iterations))
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    p = 2 * min(sum(1 for d in diffs if d <= 0) / len(diffs),
                sum(1 for d in diffs if d >= 0) / len(diffs))
    return {"metric": metric, "delta": round(point, 2), "ci_lo": round(lo, 2),
            "ci_hi": round(hi, 2), "p_value": round(min(1.0, p), 4),
            "rows": len(ids), "iterations": len(diffs)}


def holm(results: list[dict], alpha: float = 0.05) -> None:
    ordered = sorted(results, key=lambda r: r["p_value"])
    m = len(ordered)
    for i, r in enumerate(ordered):
        r["holm_threshold"] = round(alpha / (m - i), 4)
        r["significant"] = r["p_value"] <= alpha / (m - i)
    for r in results:
        r.setdefault("holm_threshold", None)
        r.setdefault("significant", False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--family", default=None, choices=list(FAMILIES))
    ap.add_argument("--metric", default="comet_da")
    ap.add_argument("--iterations", type=int, default=10000)
    args = ap.parse_args()

    pairs = FAMILIES[args.family] if args.family else [(args.a, args.b)]
    out = []
    for a_id, b_id in pairs:
        if not (RUNS / a_id / "scores.jsonl").exists() or not (RUNS / b_id / "scores.jsonl").exists():
            print(f"[skip] {a_id} vs {b_id}：缺 scores.jsonl", flush=True)
            continue
        a, b = load_scores(a_id), load_scores(b_id)
        if args.metric in ("bleu", "chrfpp"):
            m = "bleu" if args.metric == "bleu" else "chrf"
            res = corpus_bleu_delta(a, b, m, max(200, args.iterations // 10))
        else:
            res = paired(a, b, args.metric, args.iterations)
        res.update({"a": a_id, "b": b_id})
        out.append(res)
        print(f"{a_id:16s} - {b_id:16s} Δ={res['delta']:+.4f} "
              f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] p={res['p_value']:.4f}", flush=True)

    if not out:
        return
    holm(out)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "bootstrap.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    prev = [r for r in prev if (r["a"], r["b"]) not in {(x["a"], x["b"]) for x in out}]
    path.write_text(json.dumps(prev + out, ensure_ascii=False, indent=2), encoding="utf-8")
    sig = sum(1 for r in out if r["significant"])
    print(f"\n{sig}/{len(out)} significant after Holm -> "
          f"{path.relative_to(ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    main()
