"""把 HTML 裁决结果（reviewed.json）落成最终术语表。

输入：reports/reviewed.json（页面导出）
输出：
  data/processed/glossary.emea.json   含 fr_ids / accepted / variants
  data/processed/glossary.mtnt.json   含 mode（keep/translate/omit）/ accepted
  reports/glossary_summary.json       条数、类别覆盖、test 覆盖率

命中率指标要用 accepted（同义/形态变体都算命中），软约束只用 fr_ids（首 token 偏置）。

用法：
  python src/data/finalize_glossary.py --reviewed reports/reviewed.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def load_reviewed(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


EN_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")
FR_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ][A-Za-zÀ-ÖØ-öø-ÿŒœÆæ\-']*")
FR_STOP = set("""de du des la le les l d un une et ou pour dans sur avec par au aux en qu que qui
quoi dont est sont être ont cette ce ces son sa ses leur leurs il elle ils elles on nous vous je tu
plus moins très tout tous toute toutes même aussi comme lors chez sans sous entre selon après avant
pendant depuis vers contre y ne pas se c n m t j s""".split())
EN_STOP = set("""a an the of in on at to for with and or by from as is are was were be been
this that these those it its their his her our your my we you they he she i not no than then
which who whom whose what when where why how all any both each few more most other some such""".split())
UNIT_TOK = set("""mg mcg µg ug ng g kg ml cl dl l iu ui miu mmol mol µm um nm cm mm m % cfu kcal
kj da kda ppm meq eq bq mbq gb kb mb""".split())


def sig_of(text: str, side: str = "en") -> list[str]:
    """内容词签名（排序去重）。命中率匹配用它，词序变化不影响判定。"""
    toks = (EN_WORD if side == "en" else FR_WORD).findall(text)
    out = [t.lower().strip("'-") for t in toks]
    out = [t for t in out if t and t not in UNIT_TOK and not t.isdigit()]
    if side == "fr":
        out = [t for t in out if t not in FR_STOP]
    else:
        out = [t for t in out if t not in EN_STOP]
    return sorted(set(out))


def variants_of(fr: str) -> list[str]:
    """保守的形态变体：单复数与首字母大小写。宁可少造，不要错配。"""
    out = {fr, fr[0].upper() + fr[1:]}
    if fr.endswith("s"):
        out.add(fr[:-1])
    else:
        out.add(fr + "s")
    if fr.endswith("e"):
        out.add(fr + "s")
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed", default=str(REPORTS / "reviewed.json"))
    ap.add_argument("--min-decision", type=int, default=1,
                    help="1=只要保留/改译；传 0 会把剔除的也写进去（不建议）")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    rev = load_reviewed(Path(args.reviewed))
    emea_rows, mtnt_rows = [], []

    for d in rev.get("emea", []):
        if d.get("decision") not in (1, 2):
            continue
        fr = (d.get("fr_final") or d.get("fr_original") or "").strip()
        if not fr:
            continue
        ids = tok(" " + fr, add_special_tokens=False)["input_ids"]
        variants = variants_of(fr)
        accepted = sorted(set(variants) | set(d.get("accepted") or []))
        emea_rows.append({
            "id": f"emea_{len(emea_rows):04d}",
            "domain": "emea",
            "src": d["en"].strip(),
            "fr": fr,
            "fr_ids": ids,
            "fr_first_id": ids[0] if ids else None,
            "src_sig": sig_of(d["en"]),
            "fr_sig": sig_of(fr, "fr"),
            "single_token": len(ids) == 1,
            "accepted": accepted,
            "variants": variants,
            "decision": d.get("decision"),
            "score": d.get("score"),
            "cooc": d.get("cooc"),
            "test_hits": d.get("test_hits", 0),
            "provenance": "align+manual",
        })

    for d in rev.get("mtnt", []):
        if d.get("decision") != 1:
            continue
        tok_ = d["token"].strip()
        mode = d.get("mode") or "translate"
        fr = (d.get("fr") or "").strip()
        if mode == "keep":
            fr = tok_
        accepted = sorted({tok_, tok_.upper(), fr, fr[:1].upper() + fr[1:] if fr else fr})
        mtnt_rows.append({
            "id": f"mtnt_{len(mtnt_rows):04d}",
            "domain": "mtnt",
            "src": tok_,
            "fr": fr,
            "mode": mode,
            "accepted": [a for a in accepted if a],
            "mtnt_freq": d.get("mtnt_freq"),
            "test_hits": d.get("test_hits", 0),
            "provenance": "attested+manual",
        })

    # 同一法语术语被多条英文命中时只留影响最大的（防止人工合并后重复）
    by_fr: dict[tuple, dict] = {}
    for r in sorted(emea_rows, key=lambda x: -x.get("test_hits", 0)):
        by_fr.setdefault(tuple(r["fr_sig"]), r)
    emea_rows = sorted(by_fr.values(), key=lambda x: -x.get("test_hits", 0))
    for i, r in enumerate(emea_rows):
        r["id"] = f"emea_{i:04d}"

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "glossary.emea.json").write_text(
        json.dumps(emea_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "glossary.mtnt.json").write_text(
        json.dumps(mtnt_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # 覆盖率：有多少 test 句能被术语表"碰到"（决定命中率指标的分母）
    summary = {"emea_terms": len(emea_rows), "mtnt_terms": len(mtnt_rows),
               "emea_single_token": sum(1 for r in emea_rows if r["single_token"]),
               "mtnt_modes": {m: sum(1 for r in mtnt_rows if r["mode"] == m)
                              for m in ("keep", "translate", "omit")}}
    for domain, terms in (("emea", emea_rows), ("mtnt", mtnt_rows)):
        path = DATA / f"{domain}.test.jsonl"
        if not path.exists() or not terms:
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        if domain == "emea":
            pats = [(t["id"], re.compile(re.escape(t["src"].lower()))) for t in terms]
        else:
            pats = [(t["id"], re.compile(r"(^|\W)" + re.escape(t["src"].lower()) + r"(\W|$)"))
                    for t in terms]
        covered, total_hits = 0, 0
        per_term = {tid: 0 for tid, _ in pats}
        for r in rows:
            src = r["src"].lower()
            hit = False
            for tid, p in pats:
                if p.search(src):
                    per_term[tid] += 1
                    hit = True
            covered += int(hit)
        total_hits = sum(per_term.values())
        summary[f"{domain}_test"] = {"rows": len(rows), "covered": covered,
                                     "coverage_pct": round(100 * covered / max(1, len(rows)), 1),
                                     "term_hits": total_hits}
        # 把 test 命中数回写到术语表，便于报告里按影响排序
        if domain == "emea":
            for t in emea_rows:
                t["test_hits"] = per_term.get(t["id"], 0)
        else:
            for t in mtnt_rows:
                t["test_hits"] = per_term.get(t["id"], 0)

    (DATA / "glossary.emea.json").write_text(
        json.dumps(emea_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "glossary.mtnt.json").write_text(
        json.dumps(mtnt_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORTS / "glossary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {DATA/'glossary.emea.json'}")
    print(f"-> {DATA/'glossary.mtnt.json'}")
    print(f"-> {REPORTS/'glossary_summary.json'}")


if __name__ == "__main__":
    main()
