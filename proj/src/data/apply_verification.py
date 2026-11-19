"""把第二模型复核的结论落进产物：

1) 采纳复核提出的 2 条改译（concentrate for solution / Cardboard box）
2) 把复核统计（同意率、问题分布）追加到 reports/glossary_summary.json

在 finalize_glossary.py 之后运行（它会被 finalize 覆盖，所以必须后跑）。
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
ADJ = REPORTS / "adjudication"

# 复核建议 -> 采纳（problem=wrong_decision 且有 suggested）
ACCEPT = {
    "concentrate for solution": "solution à diluer",
    "Cardboard box": "boîte en carton",
}

reviewed_path = REPORTS / "reviewed.json"
data = json.loads(reviewed_path.read_text(encoding="utf-8"))
applied = []
for r in data["emea"]:
    if r["en"] in ACCEPT:
        r["decision"] = 2
        r["fr_final"] = ACCEPT[r["en"]]
        r["reason"] = "复核采纳：" + ACCEPT[r["en"]]
        applied.append((r["en"], ACCEPT[r["en"]]))
reviewed_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"采纳改译 {len(applied)} 条：{applied}")

import sys as _sys
if "--patch-only" in _sys.argv:
    raise SystemExit(0)

verify = [json.loads(l) for l in (ADJ / "verdicts" / "verify_claude.jsonl").open(encoding="utf-8")]
agree = sum(1 for v in verify if v.get("agree"))
prob = Counter(v.get("problem") for v in verify)
block = {
    "model": "claude-haiku-5-5",
    "sample": len(verify),
    "agree": agree,
    "agreement_pct": round(100 * agree / max(1, len(verify)), 1),
    "problems": dict(prob),
    "accepted_suggestions": [{"en": e, "fr": f} for e, f in applied],
    "notes": "pulmonary oedema 被判 no_evidence，但 œdème pulmonaire 是标准法语，保持 keep",
}

summary_path = REPORTS / "glossary_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["verification"] = block
summary["emea_terms"] = len([r for r in data["emea"] if r["decision"] in (1, 2)])
summary["mtnt_terms"] = len([r for r in data["mtnt"] if r["decision"] == 1])
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(block, ensure_ascii=False, indent=2))
print(f"-> {summary_path}")
