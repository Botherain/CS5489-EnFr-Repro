"""诊断 EMEA 的分数是否被 PDF 格式噪声压住。

做法：对同一批句子，分别用原始文本和"格式归一化后"的文本打 COMET-DA。
归一化只动排版（项目符号、编号、多余空格、断行、大小写首字母），不动词。
若归一化后教师与本地系统都明显上升，说明剩余差距主要来自格式而非翻译质量，
那么"后处理归一化"就是一条便宜且合法的增益路径。
"""
from __future__ import annotations

import io
import json
import re
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
CEIL = ROOT / "reports" / "ceiling"
RUNS = ROOT / "runs"

BULLET = re.compile(r"^[\s•·\-–—*]+")
NUMBERING = re.compile(r"^\s*\d+(\.\d+)*\s*[.)]?\s*")
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    s = text.replace("\u00a0", " ")
    s = BULLET.sub("", s)
    s = NUMBERING.sub("", s)
    s = SPACES.sub(" ", s).strip()
    return s


def main() -> None:
    dom, n = "emea", 150
    refs = {json.loads(l)["id"]: json.loads(l)
            for l in (CEIL / f"{dom}_{n}_ref.jsonl").open(encoding="utf-8") if l.strip()}
    teacher = {json.loads(l)["id"]: json.loads(l).get("pred", "")
               for l in (CEIL / f"{dom}_{n}_teacher.jsonl").open(encoding="utf-8") if l.strip()}
    g3 = {json.loads(l)["id"]: json.loads(l)
          for l in (RUNS / "g3_sft" / "scores.jsonl").open(encoding="utf-8") if l.strip()}

    from comet import download_model, load_from_checkpoint
    da = load_from_checkpoint(str(download_model("Unbabel/wmt22-comet-da")))

    def score(pairs):
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in pairs]
        try:
            res = da.predict(data, batch_size=32, gpus=1, progress_bar=False)
        except TypeError:
            res = da.predict(data, batch_size=32, accelerator="gpu", devices=1,
                             progress_bar=False)
        sc = list(res[0] if isinstance(res, tuple)
                  else (getattr(res, "scores", None) or res["scores"]))
        return statistics.mean(float(x) for x in sc)

    ids = [i for i in refs if i in teacher and i in g3]
    raw_t = score([(refs[i]["src"], teacher[i], refs[i]["tgt"]) for i in ids])
    norm_t = score([(normalize(refs[i]["src"]), normalize(teacher[i]), normalize(refs[i]["tgt"]))
                    for i in ids])
    raw_g = score([(refs[i]["src"], g3[i]["pred"], refs[i]["tgt"]) for i in ids])
    norm_g = score([(normalize(refs[i]["src"]), normalize(g3[i]["pred"]), normalize(refs[i]["tgt"]))
                    for i in ids])

    # 有多大比例的参考带格式噪声
    noisy = sum(1 for i in ids
                if normalize(refs[i]["tgt"]) != SPACES.sub(" ", refs[i]["tgt"]).strip())

    print(f"[n={len(ids)}] 参考侧带格式噪声: {noisy}/{len(ids)}（{100*noisy/len(ids):.0f}%）")
    print(f"\n{'':10s} {'原始':>8s} {'归一化后':>10s} {'Δ':>8s}")
    print(f"{'教师':10s} {raw_t:8.4f} {norm_t:10.4f} {norm_t-raw_t:+8.4f}")
    print(f"{'G3 SFT':10s} {raw_g:8.4f} {norm_g:10.4f} {norm_g-raw_g:+8.4f}")
    print(f"\n教师-学生差距：原始 {raw_t-raw_g:+.4f}，归一化后 {norm_t-norm_g:+.4f}")

    (CEIL / "format_diagnostic.json").write_text(json.dumps({
        "domain": dom, "n": len(ids), "noisy_refs": noisy,
        "teacher_raw": round(raw_t, 4), "teacher_norm": round(norm_t, 4),
        "g3_raw": round(raw_g, 4), "g3_norm": round(norm_g, 4),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
