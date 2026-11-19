"""S1.4 检索索引构建：bge-m3 dense（FAISS）+ BM25，分 seen / unseen 两个池。

seen   = SFT 训练用过的句对（mtnt.train + emea.train + clean.train）
unseen = 从未进 SFT 的句对（mtnt.memory_pool_unseen + emea.memory_pool）
两池在对照实验里都抽样到 6,000 条，规模匹配后才能把差异归因于"是否被 SFT 见过"。

用法：
  python src/data/build_index.py --pool seen
  python src/data/build_index.py --pool unseen --limit 6000
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

import os
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
INDEX = ROOT / "data" / "index"
ENCODER = "BAAI/bge-m3"

POOLS = {
    "seen": ["mtnt.train.jsonl", "emea.train.jsonl", "clean.train.jsonl"],
    "unseen": ["mtnt.memory_pool_unseen.jsonl", "emea.memory_pool.jsonl"],
}


def load_pool(names: list[str], limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        path = DATA / name
        if not path.exists():
            print(f"  [skip] {name} 不存在", flush=True)
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rows), size=min(limit, len(rows)), replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def encode(texts: list[str], batch: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    print(f"  encoded {len(texts)} 句，用时 {time.perf_counter()-t0:.1f}s，"
          f"峰值显存 {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del model
    torch.cuda.empty_cache()
    return emb.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=list(POOLS), default="seen")
    ap.add_argument("--limit", type=int, default=None, help="每域抽样上限（对照实验用 6000）")
    args = ap.parse_args()

    import faiss
    from rank_bm25 import BM25Okapi

    INDEX.mkdir(parents=True, exist_ok=True)
    rows = load_pool(POOLS[args.pool], args.limit)
    print(f"[{args.pool}] 句对 {len(rows)}", flush=True)

    srcs = [r["src"] for r in rows]
    emb = encode(srcs)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss_path = INDEX / f"{args.pool}.faiss"
    faiss.write_index(index, str(faiss_path))

    tok = [s.lower().split() for s in srcs]
    bm25 = BM25Okapi(tok)
    bm25_path = INDEX / f"{args.pool}.bm25.pkl"
    with bm25_path.open("wb") as fh:
        pickle.dump(bm25, fh)

    meta = INDEX / f"{args.pool}.meta.jsonl"
    with meta.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "domain": r["domain"], "src": r["src"],
                                 "tgt": r["tgt"]}, ensure_ascii=False) + "\n")

    print(f"  -> {faiss_path.name} ({faiss_path.stat().st_size/1e6:.1f}MB), "
          f"{bm25_path.name} ({bm25_path.stat().st_size/1e6:.1f}MB), {meta.name}", flush=True)

    # 自检：用 dev 的前 5 句查 top3，人工可判断相关性
    dev = DATA / "mtnt.dev.jsonl"
    if dev.exists() and args.pool == "seen":
        with dev.open(encoding="utf-8") as fh:
            probes = [json.loads(l) for l in fh.readlines()[:5]]
        q = encode([p["src"] for p in probes])
        D, I = index.search(q, 3)
        print("\n[自检] MTNT dev 前 5 句的 dense top-3：", flush=True)
        for p, scores, ids in zip(probes, D, I):
            print(f"  Q: {p['src'][:70]}", flush=True)
            for s, i in zip(scores, ids):
                print(f"     {s:.3f}  {rows[i]['src'][:60]} || {rows[i]['tgt'][:60]}", flush=True)


if __name__ == "__main__":
    main()
