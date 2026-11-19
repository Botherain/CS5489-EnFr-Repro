"""G5 Pass3：把 τ 当离线过滤器，扫出 COMET–延迟 Pareto 并选点。

输入（同一 run 目录下）：
  scores.jsonl        heavy/final 的 DA+Kiwi（对照：always-heavy）
  scores_draft.jsonl  draft 的 DA+Kiwi（对照：always-light）
  meta.json           各阶段逐句平均耗时

选点规则**预注册**（EXPERIMENT_PLAN §5）：
  τ* = argmax_τ macro_COMET(τ)  s.t. latency(τ) ≤ 1.15 × latency(G4)
不可行时放宽到 1.30 并在输出里标注；完整 8 点全部落盘。

用法：
  python src/eval/route.py --run g5_full --g4 g4_sft_rag
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
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
CONCLUSION = ("mtnt", "emea")
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--g4", default="g4_sft_rag")
    args = ap.parse_args()

    run_dir = RUNS / args.run
    heavy = {r["id"]: r for r in load_jsonl(run_dir / "scores.jsonl")}
    light = {r["id"]: r for r in load_jsonl(run_dir / "scores_draft.jsonl")}
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    ms = meta["mean_ms"]

    ids = [i for i in heavy if i in light and light[i].get("kiwi") is not None]
    print(f"[route] {len(ids)} 句可用（heavy={len(heavy)} light={len(light)}）", flush=True)

    def macro(sel) -> dict:
        per = {}
        for dom in CONCLUSION:
            vals = [sel(i) for i in ids if heavy[i]["domain"] == dom]
            if vals:
                per[dom] = statistics.mean(vals)
        return {"macro": round(statistics.mean(per.values()), 4) if per else None,
                "by_domain": {k: round(v, 4) for k, v in per.items()}}

    t_light = ms.get("t_draft", 0) + ms.get("t_qe", 0)
    t_heavy_extra = (ms.get("t_heavy", 0) + ms.get("t_critic", 0) +
                     ms.get("t_refiner", 0) + ms.get("t_retr", 0))

    g4_meta_path = RUNS / args.g4 / "meta.json"
    g4_lat = None
    if g4_meta_path.exists():
        g4_lat = json.loads(g4_meta_path.read_text(encoding="utf-8")).get("latency_batch1", {}).get("p50_ms")
    budget = (g4_lat or 600) * 1.15

    points = []
    # always-light / always-heavy 两个端点
    pts = {
        "always_light": (macro(lambda i: light[i]["comet_da"]), t_light, 0.0),
        "always_heavy": (macro(lambda i: heavy[i]["comet_da"]), t_light + t_heavy_extra, 1.0),
    }
    for k, (m, lat, hr) in pts.items():
        points.append({"tau": None, "label": k, **m, "latency_ms": round(lat, 1),
                       "heavy_ratio": hr})

    for tau in TAUS:
        sel = lambda i, tau=tau: (light[i]["comet_da"] if light[i]["kiwi"] >= tau
                                  else heavy[i]["comet_da"])
        m = macro(sel)
        ratio = sum(1 for i in ids if light[i]["kiwi"] < tau) / len(ids)
        points.append({"tau": tau, **m,
                       "latency_ms": round(t_light + ratio * t_heavy_extra, 1),
                       "heavy_ratio": round(ratio, 3)})

    light_macro = pts["always_light"][0]["macro"]
    heavy_macro = pts["always_heavy"][0]["macro"]
    gain = heavy_macro - light_macro

    feasible = [p for p in points if p["tau"] is not None and p["latency_ms"] <= budget]
    relaxed = False
    if not feasible:
        relaxed = True
        feasible = [p for p in points if p["tau"] is not None and p["latency_ms"] <= (g4_lat or 600) * 1.30]
    chosen = max(feasible, key=lambda p: p["macro"]) if feasible else None

    out = {"run_id": args.run, "n_sentences": len(ids),
           "reference": {"always_light": light_macro, "always_heavy": heavy_macro,
                         "heavy_gain": round(gain, 4)},
           "latency_model": {"t_light_ms": round(t_light, 1),
                             "t_heavy_extra_ms": round(t_heavy_extra, 1),
                             "g4_p50_ms": g4_lat, "budget_ms": round(budget, 1),
                             "relaxed_to_1_30": relaxed},
           "points": points, "chosen_tau": chosen}
    (run_dir / "routing.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"[route] always-light {light_macro:.4f}  always-heavy {heavy_macro:.4f}  "
          f"增益 {gain:+.4f}", flush=True)
    print(f"[route] 预算 {budget:.0f}ms（G4 p50={g4_lat}）", flush=True)
    for p in points:
        tag = f"τ={p['tau']}" if p["tau"] is not None else p["label"]
        print(f"  {tag:>13s}  macro={p['macro']:.4f}  lat={p['latency_ms']:6.0f}ms  "
              f"heavy={p['heavy_ratio']:.0%}", flush=True)
    if chosen:
        if gain > 1e-6:
            share = f"（heavy 增益的 {100*(chosen['macro']-light_macro)/gain:.0f}%）"
        else:
            share = "（重路径无正增益：路由应全部走轻通道）"
        print(f"[route] τ* = {chosen['tau']}  macro={chosen['macro']:.4f} "
              f"lat={chosen['latency_ms']:.0f}ms {share}", flush=True)
    else:
        print("[route] 无可行 τ，需放宽预算或检查 Kiwi 分布", flush=True)


if __name__ == "__main__":
    main()
