"""S10.3 / S11：把 JSON 产物汇总成报告用的 Markdown 表。

输入（缺哪个就跳过哪节）：
  reports/metrics.json        各 run 分域指标（metrics.py 产出）
  runs/*/routing.json         τ 扫描与选点（route.py 产出）
  reports/bootstrap.json      配对 bootstrap（bootstrap.py 产出）
  reports/shapley.json        精确 Shapley（若已跑）
输出：
  reports/tables/main.md  ablation.md  routing.md  attribution.md
  reports/tables/all.md    （拼接版，直接贴进报告）
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
DOMAINS = ("mtnt", "emea")

ORDER = ["g1_zs", "g2_rag", "g3_sft", "g3_sft_aug", "g4_sft_rag", "g4_sft_rag_inline",
         "g4_sft_rag_gated", "g5_dpo_direct", "g5_full", "g5_no_dpo", "g5_no_rag",
         "g5_no_rag_len", "g5_no_con", "g5_no_agent"]
LABEL = {
    "g1_zs": "G1 零样本",
    "g2_rag": "G2 +RAG",
    "g3_sft": "G3 SFT",
    "g3_sft_aug": "G3+ 数据增强 SFT",
    "g4_sft_rag": "G4 SFT+RAG（few-shot 轮次）",
    "g4_sft_rag_inline": "G4 SFT+RAG（inline）",
    "g4_sft_rag_gated": "G4 SFT+RAG（域门控）",
    "g5_dpo_direct": "G5* SFT+DPO（无上下文）",
    "g5_full": "G5 完整管线（heavy）",
    "g5_no_dpo": "−DPO",
    "g5_no_rag": "−RAG",
    "g5_no_rag_len": "−RAG（长度匹配随机）",
    "g5_no_con": "−术语",
    "g5_no_agent": "−Agent",
}


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main_table(metrics: dict) -> str:
    head = ("| 系统 | MTNT | EMEA | 两域宏平均 | BLEU(EMEA) | chrF++(EMEA) | "
            "术语命中(EMEA) | p50 延迟 | 峰值显存 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    rows = []
    for rid in ORDER:
        m = metrics.get(rid)
        if not m:
            continue
        bd = m["by_domain"]
        mtnt = bd.get("mtnt", {}).get("comet_da")
        emea = bd.get("emea", {}).get("comet_da")
        em = bd.get("emea", {})
        th = em.get("term_hit_rate")
        rows.append(f"| {LABEL.get(rid, rid)} | {fmt(mtnt)} | {fmt(emea)} | "
                    f"**{fmt(m.get('macro_comet_da'))}** | {fmt(em.get('bleu'), 1)} | "
                    f"{fmt(em.get('chrfpp'), 1)} | "
                    f"{'—' if th is None else f'{100*th:.1f}%'} | "
                    f"{'—' if not m.get('latency_p50_ms') else str(int(m['latency_p50_ms']))+' ms'} | "
                    f"{fmt(m.get('peak_vram_gb'), 2)} GB |")
    return head + "\n".join(rows) + "\n"


def ablation_table(metrics: dict) -> str:
    base = metrics.get("g5_full")
    if not base:
        return "_（g5_full 未完成，暂无消融表）_\n"
    head = ("| 变体 | MTNT | EMEA | 宏平均 | Δ 宏平均 | Δ 延迟 |\n"
            "| --- | --- | --- | --- | --- | --- |\n")
    rows = []
    for rid in ["g5_full", "g5_no_dpo", "g5_no_rag", "g5_no_rag_len", "g5_no_con", "g5_no_agent"]:
        m = metrics.get(rid)
        if not m:
            continue
        d = None if rid == "g5_full" else (
            None if m.get("macro_comet_da") is None or base.get("macro_comet_da") is None
            else m["macro_comet_da"] - base["macro_comet_da"])
        lat = m.get("latency_p50_ms")
        dlat = None if (lat is None or not base.get("latency_p50_ms")) else lat - base["latency_p50_ms"]
        rows.append(f"| {LABEL.get(rid, rid)} | "
                    f"{fmt(m['by_domain'].get('mtnt', {}).get('comet_da'))} | "
                    f"{fmt(m['by_domain'].get('emea', {}).get('comet_da'))} | "
                    f"**{fmt(m.get('macro_comet_da'))}** | "
                    f"{'—' if d is None else f'{d:+.4f}'} | "
                    f"{'—' if dlat is None else f'{dlat:+.0f} ms'} |")
    return head + "\n".join(rows) + "\n"


def routing_table() -> str:
    rt = load(ROOT / "runs" / "g5_full" / "routing.json")
    if not rt:
        return "_（routing.json 未生成）_\n"
    ref = rt["reference"]
    head = ("| 条件 | 宏平均 COMET | 延迟 | 重路径占比 |\n| --- | --- | --- | --- |\n")
    rows = [f"| always-light（=SFT 直出） | {ref['always_light']:.4f} | — | 0% |",
            f"| always-heavy | {ref['always_heavy']:.4f} | — | 100% |"]
    for p in rt["points"]:
        if p["tau"] is None:
            continue
        star = " ⭐" if rt.get("chosen_tau") and p["tau"] == rt["chosen_tau"]["tau"] else ""
        rows.append(f"| τ = {p['tau']:.2f}{star} | {p['macro']:.4f} | "
                    f"{p['latency_ms']:.0f} ms | {p['heavy_ratio']*100:.0f}% |")
    body = "\n".join(rows)
    note = (f"\n延迟模型：t_light={rt['latency_model']['t_light_ms']:.0f}ms，"
            f"重路径增量={rt['latency_model']['t_heavy_extra_ms']:.0f}ms，"
            f"预算={rt['latency_model']['budget_ms']:.0f}ms"
            f"（G4 p50 {rt['latency_model']['g4_p50_ms']}×1.15"
            f"{'，已放宽到 1.30' if rt['latency_model']['relaxed_to_1_30'] else ''}）。\n")
    return head + body + "\n" + note


def attribution_table(metrics: dict) -> str:
    v = {}
    for rid in ("g1_zs", "g2_rag", "g3_sft", "g4_sft_rag", "g5_full"):
        m = metrics.get(rid)
        if m and m.get("macro_comet_da") is not None:
            v[rid] = m["macro_comet_da"]
    if len(v) < 4:
        return "_（主实验未跑全，暂无归因表）_\n"
    d_sft = v["g3_sft"] - v["g1_zs"]
    d_rag = v["g2_rag"] - v["g1_zs"]
    d_syn = v["g4_sft_rag"] - v["g3_sft"] - d_rag
    d_adv = v.get("g5_full", float("nan")) - v["g4_sft_rag"]
    total = v.get("g5_full", float("nan")) - v["g1_zs"]

    def pct(x):
        return "—" if total == 0 or total != total else f"{100*x/total:+.0f}%"

    head = ("| 因子 | Δ 宏平均 | 占总提升 |\n| --- | --- | --- |\n")
    rows = [f"| Δ_SFT = G3 − G1 | {d_sft:+.4f} | {pct(d_sft)} |",
            f"| Δ_RAG = G2 − G1 | {d_rag:+.4f} | {pct(d_rag)} |",
            f"| Δ_Synergy = G4 − G3 − Δ_RAG | {d_syn:+.4f} | {pct(d_syn)} |",
            f"| Δ_Advanced = G5 − G4 | {d_adv:+.4f} | {pct(d_adv)} |",
            f"| **合计 = G5 − G1** | **{total:+.4f}** | 100% |"]
    body = head + "\n".join(rows) + "\n"
    body += ("\n> 阶梯分解有顺序依赖，仅用于叙述增长曲线；结论以 Shapley 为准"
             "（见 `reports/shapley.json`）。\n")
    return body


def bootstrap_table() -> str:
    rows = load(REPORTS / "bootstrap.json", [])
    if not rows:
        return "_（bootstrap.json 未生成）_\n"
    head = ("| 比较 | Δ | 95% CI | p | Holm 阈值 | 显著 |\n| --- | --- | --- | --- | --- | --- |\n")
    out = []
    for r in rows:
        out.append(f"| {r['a']} − {r['b']} | {r['delta']:+.4f} | "
                   f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_value']:.4f} | "
                   f"{r.get('holm_threshold', 0):.4f} | "
                   f"{'是' if r.get('significant') else '否'} |")
    return head + "\n".join(out) + "\n"


def main() -> None:
    metrics = load(REPORTS / "metrics.json", {})
    TABLES.mkdir(parents=True, exist_ok=True)
    parts = {
        "main": "# 主表\n\n" + main_table(metrics),
        "ablation": "# 消融\n\n" + ablation_table(metrics),
        "routing": "# 动态路由 Pareto\n\n" + routing_table(),
        "attribution": "# 因子归因（阶梯分解）\n\n" + attribution_table(metrics),
        "bootstrap": "# 显著性检验\n\n" + bootstrap_table(),
    }
    for name, text in parts.items():
        (TABLES / f"{name}.md").write_text(text, encoding="utf-8")
    (TABLES / "all.md").write_text(
        "\n\n".join(parts[k] for k in ("main", "ablation", "routing", "attribution", "bootstrap")),
        encoding="utf-8")
    print(f"[ok] {len(parts)} tables -> {TABLES.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
