"""给 DPO 候选打分（必须在 comet venv 里跑）。

输入：data/interim/candidates.jsonl   每行 {id, domain, src, tgt, cands:[K 个译文]}
输出：data/interim/candidates.scored.jsonl   cands 变成 [{text, da, kiwi}]

DA = COMET-22（有参考，主排序）；Kiwi = CometKiwi（免参考，兼作路由信号可靠性分析）。

用法：
  & "D:\\HKStudy\\.venv-comet\\Scripts\\python.exe" src\\train\\score_candidates.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
QE_LOCAL = str(ROOT / "runs" / "qe_local" / "wmt22-cometkiwi-da-mirror" / "checkpoints" / "model.ckpt")


def load_model(repo: str):
    from comet import download_model, load_from_checkpoint
    path = repo if Path(repo).exists() else download_model(repo)
    return load_from_checkpoint(path)


def predict(model, data: list[dict], batch_size: int):
    try:
        out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=False)
    except TypeError:
        out = model.predict(data, batch_size=batch_size, accelerator="gpu", devices=1,
                            progress_bar=False)
    if isinstance(out, tuple):
        return list(out[0])
    return list(getattr(out, "scores", None) or out["scores"])


def main() -> None:
    src = INTERIM / "candidates.jsonl"
    dst = INTERIM / "candidates.scored.jsonl"
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    n_cand = sum(len(r["cands"]) for r in rows)
    print(f"[score] {len(rows)} 个源句，{n_cand} 个候选", flush=True)

    da = load_model("Unbabel/wmt22-comet-da")
    qe = load_model(QE_LOCAL)
    print("[score] 两个打分模型已加载", flush=True)

    flat = [(i, j, c) for i, r in enumerate(rows) for j, c in enumerate(r["cands"]) if c.strip()]
    print(f"[score] 有效候选 {len(flat)}", flush=True)

    t0 = time.perf_counter()
    da_scores = predict(da, [{"src": rows[i]["src"], "mt": c, "ref": rows[i]["tgt"]}
                             for i, _, c in flat], batch_size=64)
    print(f"[score] DA 完成 {(time.perf_counter()-t0)/60:.1f} 分钟", flush=True)
    t0 = time.perf_counter()
    qe_scores = predict(qe, [{"src": rows[i]["src"], "mt": c} for i, _, c in flat],
                        batch_size=64)
    print(f"[score] Kiwi 完成 {(time.perf_counter()-t0)/60:.1f} 分钟", flush=True)

    packed: dict[int, list[dict]] = {}
    for (i, _, c), d, q in zip(flat, da_scores, qe_scores):
        packed.setdefault(i, []).append({"text": c, "da": float(d), "kiwi": float(q)})

    with dst.open("w", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(rows):
            cands = packed.get(i, [])
            if not cands:
                continue
            fh.write(json.dumps({**{k: r[k] for k in ("id", "domain", "src", "tgt")},
                                 "cands": cands}, ensure_ascii=False) + "\n")

    # 顺带算候选之间的 DA 离散度与 Kiwi/DA 秩相关（决定偏好对是否够"硬"）
    import statistics
    spreads = [max(c["da"] for c in v) - min(c["da"] for c in v) for v in packed.values()]
    print(f"[score] 组内 DA 极差: 中位 {statistics.median(spreads):.3f}，"
          f">0.10 的组 {sum(1 for s in spreads if s > 0.10)}/{len(spreads)}", flush=True)
    print(f"[score] -> {dst}", flush=True)


if __name__ == "__main__":
    main()
