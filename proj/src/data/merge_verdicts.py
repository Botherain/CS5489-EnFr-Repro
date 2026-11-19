"""合并外部模型的裁决结果 -> reviewed.json（finalize 的输入）+ 抽查包。

用法：
  python src/data/merge_verdicts.py
  python src/data/merge_verdicts.py --spotcheck 30

输入：reports/adjudication/verdicts/*.verdicts.jsonl
输出：reports/reviewed.json            直接喂给 finalize_glossary.py
      reports/adjudication/summary.json 决策/类别/置信度分布 + 低置信条目
      reports/adjudication/spotcheck.json 分层抽样 30 条，供第二模型复核
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
ADJ = ROOT / "reports" / "adjudication"
REPORTS = ROOT / "reports"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotcheck", type=int, default=30)
    args = ap.parse_args()

    packs = {"emea": read_jsonl(ADJ / "emea.jsonl"), "mtnt": read_jsonl(ADJ / "mtnt.jsonl")}
    verdicts: dict[str, dict[int, dict]] = {"emea": {}, "mtnt": {}}

    files = sorted((ADJ / "verdicts").glob("*.verdicts.jsonl"))
    if not files:
        raise SystemExit("还没有任何 verdicts 文件")
    for f in files:
        dom = "emea" if f.name.startswith("emea") else "mtnt"
        for v in read_jsonl(f):
            verdicts[dom][int(v["i"])] = v
    print(f"[load] {len(files)} 个批次文件，emea {len(verdicts['emea'])}/{len(packs['emea'])}，"
          f"mtnt {len(verdicts['mtnt'])}/{len(packs['mtnt'])}")

    missing = {
        "emea": [i for i in range(len(packs["emea"])) if i not in verdicts["emea"]],
        "mtnt": [i for i in range(len(packs["mtnt"])) if i not in verdicts["mtnt"]],
    }
    for dom, ids in missing.items():
        if ids:
            print(f"[warn] {dom} 缺 {len(ids)} 条：{ids[:12]}{'…' if len(ids) > 12 else ''}")

    reviewed = {"meta": {"source": "external-llm adjudication", "packs": {d: len(p) for d, p in packs.items()}},
                "emea": [], "mtnt": []}

    stats: dict[str, Counter] = {"emea": Counter(), "mtnt": Counter()}
    low_conf: list[dict] = []

    for i, p in enumerate(packs["emea"]):
        v = verdicts["emea"].get(i)
        if not v:
            continue
        dec = {"keep": 1, "fix": 2, "reject": 0}.get(v.get("decision"), 0)
        fr_final = (v.get("fr") or "").strip() if dec == 2 else p["fr_current"]
        if dec == 2 and not fr_final:
            dec = 1                       # fix 但没给译法，降级为 keep
        reviewed["emea"].append({
            "en": p["en"], "fr_original": p["fr_current"], "fr_final": fr_final,
            "decision": dec, "score": p.get("cooc") and 1.0, "cooc": p.get("cooc"),
            "test_hits": p.get("test_hits", 0),
            "confidence": v.get("confidence"), "category": v.get("category"),
            "reason": v.get("reason"),
        })
        stats["emea"][v.get("decision")] += 1
        stats["emea"][f"cat:{v.get('category')}"] += 1
        stats["emea"][f"conf:{v.get('confidence')}"] += 1
        if v.get("confidence") == "low":
            low_conf.append({"domain": "emea", "i": i, "en": p["en"],
                             "fr": p["fr_current"], "decision": v.get("decision"),
                             "reason": v.get("reason")})

    for i, p in enumerate(packs["mtnt"]):
        v = verdicts["mtnt"].get(i)
        if not v:
            continue
        dec = 1 if v.get("decision") == "keep" else 0
        mode = v.get("mode") or ("translate" if v.get("fr") else "keep")
        fr = (v.get("fr") or "").strip()
        # 俚语的"对译"若超过 3 个词，通常是把整句意译当成了词条（gt → Fil de discussion du match），
        # 这类没有固定对应，降级为 omit，避免污染命中率
        if mode == "translate" and len(fr.split()) > 3:
            mode = "omit"
            fr = ""
        reviewed["mtnt"].append({
            "token": p["token"], "fr": fr, "decision": dec,
            "mode": mode, "mtnt_freq": p.get("mtnt_freq"), "test_hits": 0,
            "confidence": v.get("confidence"), "reason": v.get("reason"),
        })
        stats["mtnt"][v.get("decision")] += 1
        stats["mtnt"][f"mode:{mode}"] += 1
        stats["mtnt"][f"conf:{v.get('confidence')}"] += 1
        if v.get("confidence") == "low":
            low_conf.append({"domain": "mtnt", "i": i, "token": p["token"],
                             "decision": v.get("decision"), "reason": v.get("reason")})

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "reviewed.json").write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {d: dict(c) for d, c in stats.items()}
    summary["missing"] = {d: len(v) for d, v in missing.items()}
    summary["low_confidence"] = low_conf
    (ADJ / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    # 分层抽样交给第二个模型复核：keep / fix / reject 各取 1/3
    rng = random.Random(42)
    picked = []
    for want in (1, 2, 0):
        pool = [r for r in reviewed["emea"] if r["decision"] == want]
        picked += rng.sample(pool, min(len(pool), args.spotcheck // 3))
    idx = {p["en"]: p for p in packs["emea"]}
    spot = []
    for r in picked:
        src = idx.get(r["en"], {})
        spot.append({"en": r["en"], "fr_claimed": r["fr_final"],
                     "claimed_decision": {1: "keep", 2: "fix", 0: "reject"}[r["decision"]],
                     "model_reason": r.get("reason"),
                     "examples": src.get("examples", [])})
    (ADJ / "spotcheck.json").write_text(json.dumps(spot, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    print(json.dumps({d: dict(c) for d, c in stats.items()}, ensure_ascii=False, indent=2))
    print(f"[low] 低置信 {len(low_conf)} 条 -> summary.json")
    print(f"[spot] 抽查 {len(spot)} 条 -> spotcheck.json")
    print(f"[out] {REPORTS / 'reviewed.json'}")


if __name__ == "__main__":
    main()
