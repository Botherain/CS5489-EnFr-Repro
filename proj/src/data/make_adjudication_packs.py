"""把候选表打成"裁决任务包"，交给外部大模型逐条判定。

每条任务包含判定所需的最小充分证据：
  - 英文术语 / 现有法语译法
  - 2 条语料实证句对（真实 EMEA 文本，法语侧是母语者译法）
  - 同一条英文在语料里出现过的其他法语写法（对齐阶段收集的 variants）
  - dev/test 命中句数（决定它对本项目有没有用）

输出：
  reports/adjudication/emea.jsonl   待判 EMEA 条目
  reports/adjudication/mtnt.jsonl   待判 MTNT 俚语
  reports/adjudication/PROMPT.md    统一的裁决说明（各批次复用，保证口径一致）
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "adjudication"


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def examples_for(en: str, rows: list[dict], n: int = 2) -> list[list[str]]:
    out, seen = [], set()
    key = en.lower()
    for r in rows:
        if key in r["src"].lower() and r["tgt"][:60] not in seen:
            seen.add(r["tgt"][:60])
            out.append([r["src"][:260], r["tgt"][:260]])
            if len(out) >= n:
                break
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    emea = load("glossary.emea.candidates.jsonl")
    slang = load("glossary.mtnt.candidates.jsonl")
    train = load("emea.train.jsonl")
    dev = [r["src"].lower() for r in load("emea.dev.jsonl")]
    test = [r["src"].lower() for r in load("emea.test.jsonl")]
    mtnt_train = load("mtnt.train.jsonl")

    with (OUT / "emea.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for i, c in enumerate(emea):
            en = c["en"]
            fh.write(json.dumps({
                "i": i, "en": en, "fr_current": c["fr"],
                "fr_variants": c.get("fr_variants", [])[:4],
                "en_variants": c.get("en_variants", [])[:3],
                "dev_hits": sum(1 for s in dev if en.lower() in s),
                "test_hits": sum(1 for s in test if en.lower() in s),
                "cooc": c["cooc"], "en_freq": c["en_freq"],
                "examples": examples_for(en, train, 2),
            }, ensure_ascii=False) + "\n")

    with (OUT / "mtnt.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for i, c in enumerate(slang):
            tok_ = c["token"]
            pat = re.compile(r"(^|\W)" + re.escape(tok_) + r"(\W|$)", re.I)
            ex = [["" + r["src"][:200], r["tgt"][:200]] for r in mtnt_train
                  if pat.search(r["src"])][:3]
            fh.write(json.dumps({
                "i": i, "token": tok_, "mtnt_freq": c["mtnt_freq"],
                "emea_freq": c["emea_freq"], "clean_freq": c["clean_freq"],
                "examples": ex,
            }, ensure_ascii=False) + "\n")

    (OUT / "PROMPT.md").write_text(PROMPT, encoding="utf-8")

    n_emea = sum(1 for _ in (OUT / "emea.jsonl").open(encoding="utf-8"))
    n_mtnt = sum(1 for _ in (OUT / "mtnt.jsonl").open(encoding="utf-8"))
    print(f"[ok] emea {n_emea} 条，mtnt {n_mtnt} 条 -> {OUT}")
    first = json.loads((OUT / "emea.jsonl").open(encoding="utf-8").readline())
    print("样例:", json.dumps(first, ensure_ascii=False)[:300])


PROMPT = """# 术语表裁决任务说明

你是英法双语医药文本的术语审校。任务：对一批"英文→法语"候选术语逐条裁决。

## 判定标准（按优先级）

1. **是不是术语**：指疾病、药物成分、剂型、给药途径、解剖、药理过程、法规程序等专业表达。
   通用词（body weight / powder / response / moderate）不算术语，但如果是固定搭配的剂型或
   给药表达（solution for injection / film-coated tablet）算。
2. **法语译法是否正确**：以 EMEA 语料里的实证句对为准。EMEA 是欧盟药监局官方文档，
   法语侧是机构的规范译法；`examples` 给的就是这类句对。若 `fr_current` 与例句里的写法
   不符，以例句为准。
3. **改动要保守**：只在确有必要时改译。`fr_variants` 是语料里同一英文出现过的其他法语写法，
   可以从中挑最规范的一个，也可以自己给，但必须是与 EMEA 语域一致的正式表达。
4. **品牌名、公司名、页码碎片、掩码占位（x / n）、跨栏拼接、说明书写法**（Take / Store /
   Select / 每片含…）一律 reject。

## 输出格式

只输出 JSON Lines，每行一条，顺序与输入一致，不要任何解释性前后文：

{"i": 12, "decision": "keep", "fr": "", "confidence": "high", "category": "disease", "reason": "简短中文理由"}
{"i": 13, "decision": "fix",  "fr": "filtration glomérulaire", "confidence": "high", "category": "anatomy", "reason": "..."}
{"i": 14, "decision": "reject", "fr": "", "confidence": "high", "category": "brand", "reason": "药品品牌名"}

字段要求：
- decision：keep（原译法正确）/ fix（是术语但译法要改，必须给 fr）/ reject（不是术语或无法修正）
- fr：decision=fix 时必填，其余可留空
- confidence：high / medium / low。把握不足就用 low，不要硬猜
- category：disease | drug | dosage_form | route | anatomy | procedure | regulatory | abbreviation | general | brand | fragment
- reason：一句中文，说明依据（引用例句里的写法最好）

## 纪律

- 不确定的一律 low confidence + keep 或 reject，不要编造法语写法。
- 不要改英文侧；英文侧有问题就 reject 并说明。
- 逐条独立判断，不要参考批次里其他条目的结论。

## MTNT 俚语批次（若输入是 token 而非 en/fr）

输入是 Reddit 俚语/缩写候选，附 3 条带官方译文的训练对。裁决规则：

- 在译文里被**意译**的（如 tbh → pour être honnête）：decision=keep 或 fix，
  在 fr 里写训练对中最常见的那个法语表达，category=abbreviation。
- 在译文里**原样保留**的（PCC / MVP / WTF）：decision=keep，fr 填空字符串，
  并在 reason 里写"译文保留原样"。
- 在译文里被**整句改写掉**的：decision=keep，category=general，
  reason 写"无固定对应，属语气词"。
- 训练对里完全没有该词（说明它只是普通英文词）：decision=reject。
- 输出多一个字段 mode："translate" | "keep" | "omit"。
"""


if __name__ == "__main__":
    main()
