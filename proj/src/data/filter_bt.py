"""检查并过滤回译数据里的"法语泄漏"。

Opus-MT fr→en 偶尔会在英文输出里留下未翻译的法语片段（如 "go to the nego d'é"）。
这种样本的源侧是"英法混合"，会让模型学到错误的源语言分布，必须过滤。

判据：源侧出现法语特有字符（à â ç é è ê ë î ï ô û ù ü ÿ œ）或常见法语词。
"""
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INTERIM = Path("data/interim")
p = INTERIM / "mtnt_bt.jsonl"
rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

FR_CHARS = re.compile(r"[àâçéèêëîïôûùüÿœæ]", re.I)
FR_WORDS = {"le", "la", "les", "des", "une", "est", "pas", "que", "qui", "pour", "dans",
            "avec", "sur", "mais", "nous", "vous", "ils", "elle", "c'est", "j'ai"}

kept, dropped = [], 0
for r in rows:
    src = r["src"]
    if FR_CHARS.search(src):
        dropped += 1
        continue
    toks = re.findall(r"[A-Za-zÀ-ÿ']+", src.lower())
    if toks and sum(1 for t in toks if t in FR_WORDS) / len(toks) > 0.30:
        dropped += 1
        continue
    kept.append(r)

print(f"[bt-filter] {len(rows)} -> {len(kept)}（剔除 {dropped}，{100*dropped/len(rows):.1f}%）")
with p.open("w", encoding="utf-8", newline="\n") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

for r in kept[:3]:
    print("  例 EN:", r["src"][:80])
