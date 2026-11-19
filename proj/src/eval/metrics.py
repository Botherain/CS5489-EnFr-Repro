"""S9.1 全量指标计算：把 runs/<id>/scores.jsonl 汇总成一张可比较的表。

口径（与 EXPERIMENT_PLAN §6 一致）：
  - 头条 = MTNT/EMEA 两域等权宏平均 COMET-DA（clean 只作 sanity）
  - BLEU/chrF++ 用 sacrebleu 语料级口径（不做句级均值，句级均值有偏）
  - 术语命中率分母 = 源句含术语的句子（用 glossary 的 accepted 变体做词序无关匹配）
  - 退化率/重复率对全系统统一口径

用法：
  python src/eval/metrics.py --runs g1_zs,g2_rag
  python src/eval/metrics.py --all
"""
from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CONCLUSION_DOMAINS = ("mtnt", "emea")
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ0-9][A-Za-zÀ-ÖØ-öø-ÿŒœÆæ0-9'\-]*")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_glossary() -> dict[str, list[dict]]:
    """两个域各一张表：EMEA 术语表 + MTNT 俚语表。

    MTNT 表里 mode=omit 的条目没有固定对应（整句改写），不参与命中率，
    否则会把"本来就没有固定译法"的条目算成 miss，压低指标。
    """
    out: dict[str, list[dict]] = {"emea": [], "mtnt": []}

    emea = DATA / "glossary.emea.json"
    if emea.exists():
        terms = json.loads(emea.read_text(encoding="utf-8"))
        for t in terms:
            t["_src_pat"] = re.compile(r"\b" + re.escape(t["src"].lower()) + r"\b")
            acc = [a.lower() for a in t.get("accepted", []) if a]
            t["_acc_pats"] = [re.compile(r"\b" + re.escape(a) + r"\b") for a in acc]
        out["emea"] = terms

    mtnt = DATA / "glossary.mtnt.json"
    if mtnt.exists():
        terms = [t for t in json.loads(mtnt.read_text(encoding="utf-8"))
                 if t.get("mode") != "omit"]
        for t in terms:
            t["_src_pat"] = re.compile(r"\b" + re.escape(t["src"].lower()) + r"\b")
            acc = [a.lower() for a in t.get("accepted", []) if a]
            t["_acc_pats"] = [re.compile(r"\b" + re.escape(a) + r"\b") for a in acc]
        out["mtnt"] = terms
    return out


def ngram_repeat_ratio(text: str, n: int = 4) -> float:
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def domain_metrics(rows: list[dict], terms: list[dict]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    preds = [r["pred"] for r in rows]
    refs = [r["tgt"] for r in rows]
    out = {"rows": len(rows),
           "comet_da": round(statistics.mean(r["comet_da"] for r in rows), 4),
           "bleu": round(BLEU(tokenize="13a").corpus_score(preds, [refs]).score, 2),
           "chrfpp": round(CHRF(char_order=6, word_order=2).corpus_score(preds, [refs]).score, 2)}

    kiwi = [r["kiwi"] for r in rows if r.get("kiwi") is not None]
    out["kiwi"] = round(statistics.mean(kiwi), 4) if kiwi else None

    # 重复退化
    out["repetition_rate"] = round(
        sum(1 for p in preds if ngram_repeat_ratio(p) > 0.30) / max(1, len(preds)), 4)
    bad = 0
    for r in rows:
        p = r["pred"].strip()
        if not p:
            bad += 1
            continue
        ratio = len(p.split()) / max(1, len(r["src"].split()))
        if not 0.3 <= ratio <= 3.0 or p.lower() == r["src"].strip().lower():
            bad += 1
    out["degeneration_rate"] = round(bad / max(1, len(rows)), 4)
    out["length_ratio"] = round(
        statistics.mean(len(r["pred"].split()) / max(1, len(r["src"].split())) for r in rows), 3)

    # 术语命中（只在源句含术语的子集上算）
    if terms:
        opp = hit = 0
        for r in rows:
            src = r["src"].lower()
            for t in terms:
                if t["_src_pat"].search(src):
                    opp += 1
                    if any(p.search(r["pred"].lower()) for p in t["_acc_pats"]):
                        hit += 1
        out["term_opportunities"] = opp
        out["term_hit_rate"] = round(hit / opp, 4) if opp else None
    return out


def run_metrics(run_id: str, terms: list[dict]) -> dict | None:
    scores = RUNS / run_id / "scores.jsonl"
    if not scores.exists():
        return None
    rows = load_jsonl(scores)
    by_dom: dict[str, list[dict]] = {}
    for r in rows:
        by_dom.setdefault(r["domain"], []).append(r)

    res = {"run_id": run_id,
           "by_domain": {d: domain_metrics(v, terms.get(d, [])) for d, v in by_dom.items()}}
    conclude = [res["by_domain"][d]["comet_da"] for d in CONCLUSION_DOMAINS if d in res["by_domain"]]
    res["macro_comet_da"] = round(statistics.mean(conclude), 4) if conclude else None

    meta = RUNS / run_id / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        res["latency_p50_ms"] = m.get("latency_batch1", {}).get("p50_ms")
        res["latency_p95_ms"] = m.get("latency_batch1", {}).get("p95_ms")
        res["peak_vram_gb"] = m.get("peak_vram_gb")
        res["sent_per_sec"] = m.get("sent_per_sec")
        res["decode"] = m.get("decode")
        res["k"] = m.get("k")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    terms = load_glossary()  # {domain: [terms]}
    if args.all or not args.runs:
        ids = [p.name for p in RUNS.iterdir()
               if p.is_dir() and (p / "scores.jsonl").exists() and not p.name.startswith("_")]
    else:
        ids = [r.strip() for r in args.runs.split(",") if r.strip()]

    out = {}
    for rid in ids:
        res = run_metrics(rid, terms)
        if res is None:
            print(f"[skip] {rid} 无 scores.jsonl", flush=True)
            continue
        out[rid] = res
        line = (f"{rid:22s} macro={res['macro_comet_da']}  "
                + "  ".join(f"{d}={v['comet_da']:.4f}" for d, v in res["by_domain"].items()))
        print(line, flush=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    prev_path = REPORTS / "metrics.json"
    merged = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}
    merged.update(out)
    prev_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {prev_path.relative_to(ROOT).as_posix()}  ({len(merged)} runs)")


if __name__ == "__main__":
    main()
