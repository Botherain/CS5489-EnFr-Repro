"""统一批量推理：G1(零样本) / G2(+RAG) / G3(SFT) / G4(SFT+RAG) 共用一条代码路径。

设计原则：
  1) 各系统之间只允许"模型/adapter"与"是否检索"两处不同，prompt 骨架完全一致，
     否则 RAG 因子的对比会被 prompt 差异污染。
  2) 只负责生成与落盘，不打分（COMET 在独立 venv 的 Pass2 里做）。
  3) 逐句与批量两套延迟口径分开测：--latency-sample N 用 batch=1 重测 N 句。

用法：
  python src/infer/generate.py --run-id g1_zs --mode zeroshot --set test
  python src/infer/generate.py --run-id g2_rag --mode rag --set test --k 3
  python src/infer/generate.py --run-id g3_sft --mode sft --adapter runs/g3_sft_r16_s42/adapter --set test
  python src/infer/generate.py --run-id g4_sft_rag --mode sft_rag --adapter ... --set test --k 3
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.85")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
INDEX = ROOT / "data" / "index"
RUNS = ROOT / "runs"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = ("You are a professional English-to-French translator. Translate the user's sentence "
          "into natural, accurate French. Output only the translation.")
SET_FILES = {
    "test": {"mtnt": "mtnt.test.jsonl", "emea": "emea.test.jsonl", "clean": "clean.test.jsonl"},
    "dev": {"mtnt": "mtnt.dev.jsonl", "emea": "emea.dev.jsonl", "clean": "clean.dev.jsonl"},
}
GB = 1e9


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=15).stdout.strip() or "n/a"
    except Exception:  # noqa: BLE001
        return "n/a"


def load_rows(names: list[str], limit: int | None) -> list[dict]:
    rows = []
    for n in names:
        with (DATA / n).open(encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows[:limit] if limit else rows


class Retriever:
    """Dense(bge-m3 + FAISS) + BM25，RRF 融合。pool=seen|unseen。"""

    def __init__(self, pool: str = "seen", rrf_k: int = 60):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.rrf_k = rrf_k
        self.index = faiss.read_index(str(INDEX / f"{pool}.faiss"))
        with (INDEX / f"{pool}.bm25.pkl").open("rb") as fh:
            self.bm25 = pickle.load(fh)
        self.meta = [json.loads(l) for l in (INDEX / f"{pool}.meta.jsonl").open(encoding="utf-8")]
        self.enc = SentenceTransformer("BAAI/bge-m3", device="cuda",
                                       model_kwargs={"torch_dtype": torch.float16})
        self.pool = pool

    def search(self, queries: list[str], k: int = 3) -> list[list[dict]]:
        # 过取 4k 再按源句去重：EMEA/MTNT 里同一源句常有多条参考译文，
        # 同一源句常有多条参考译文，不去重会让 3-shot 出现重复示例。
        fetch = max(k * 4, 12)
        q = self.enc.encode(queries, batch_size=64, normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")
        _, dense_ids = self.index.search(q, fetch)
        out = []
        for qi, query in enumerate(queries):
            scores = self.bm25.get_scores(query.lower().split())
            bm25_ids = scores.argsort()[::-1][:fetch]
            rrf: dict[int, float] = {}
            for rank, i in enumerate(dense_ids[qi]):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.rrf_k + rank + 1)
            for rank, i in enumerate(bm25_ids):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (self.rrf_k + rank + 1)
            picked, seen_src = [], set()
            for i, _ in sorted(rrf.items(), key=lambda x: -x[1]):
                key = self.meta[i]["src"].strip().lower()
                if key in seen_src or key == query.strip().lower():
                    continue
                seen_src.add(key)
                picked.append(self.meta[i])
                if len(picked) == k:
                    break
            out.append(picked)
        return out


def load_examples(rows: list[dict], pool: str, k: int, same_domain: bool = False) -> list[list[dict]]:
    """带磁盘缓存的检索：同一 (pool, id) 只算一次，所有系统与所有 k 共用。

    缓存固定存 top-12，取用时再切前 k 条——这样 k=1/3/5 的扫描只付一次 BM25 成本
    （61,393 文档上是纯 Python 打分，实测 ~0.21s/句）。
    """
    cache_dir = INDEX / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_k = max(k, 12)
    cache_path = cache_dir / f"{pool}.top{cache_k}.jsonl"

    cached: dict[str, list[dict]] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    cached[d["id"]] = d["examples"]

    missing = [r for r in rows if r["id"] not in cached]
    if missing:
        print(f"[retrieval] 缓存命中 {len(rows) - len(missing)}，需计算 {len(missing)}", flush=True)
        retr = Retriever(pool)
        t0 = time.perf_counter()
        with cache_path.open("a", encoding="utf-8", newline="\n") as fh:
            for start in range(0, len(missing), 64):
                chunk = missing[start:start + 64]
                exs = retr.search([r["src"] for r in chunk], k=cache_k)
                for r, ex in zip(chunk, exs):
                    cached[r["id"]] = ex
                    fh.write(json.dumps({"id": r["id"], "examples": ex}, ensure_ascii=False) + "\n")
        print(f"[retrieval] 完成 {len(missing)} 句，用时 {time.perf_counter() - t0:.1f}s", flush=True)
        del retr
        torch.cuda.empty_cache()
    else:
        print(f"[retrieval] 全部命中缓存（{len(rows)} 句）", flush=True)
    if same_domain:
        # 域门控：MTNT 查询约 12% 会检索到 EMEA 样板句。
        # 缓存里存的是 top-12，足以在同域内重排出前 k 条。
        out = []
        for r in rows:
            ex = cached[r["id"]]
            same = [e for e in ex if e.get("domain") == r["domain"]]
            other = [e for e in ex if e.get("domain") != r["domain"]]
            out.append((same + other)[:k])
        return out
    return [cached[r["id"]][:k] for r in rows]


def build_messages(src: str, examples: list[dict] | None, fmt: str = "turns") -> list[dict]:
    """两种注入格式。

    turns  : 每个示例占一对 user/assistant 轮次（G2 用的原格式）
    inline : 把示例塞进同一个 user 消息，消息结构仍是 system + 单条 user，
             与 SFT 训练时完全一致（SFT 只见过单轮，多轮示例会造成训练/推理格式错配）
    """
    if fmt == "inline" and examples:
        lines = ["Reference translations of similar sentences:"]
        for ex in examples:
            lines.append(f"EN: {ex['src']}\nFR: {ex['tgt']}")
        lines.append(f"Translate this sentence:\nEN: {src}\nFR:")
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "\n\n".join(lines)}]

    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in examples or []:
        msgs.append({"role": "user", "content": ex["src"]})
        msgs.append({"role": "assistant", "content": ex["tgt"]})
    msgs.append({"role": "user", "content": src})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--mode", choices=["zeroshot", "rag", "sft", "sft_rag"], required=True)
    ap.add_argument("--set", dest="split", choices=["test", "dev"], default="test")
    ap.add_argument("--domains", default="mtnt,emea,clean")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-adapter", dest="base_adapter",
                    default=str(ROOT / "runs" / "g3_sft_r16_s42" / "adapter"),
                    help="DPO adapter 依赖的 SFT adapter，会先合并再叠 DPO")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rag-format", dest="rag_format", choices=["turns", "inline"], default="turns")
    ap.add_argument("--domain-gated", dest="domain_gated", action="store_true",
                    help="只用与查询同域的检索结果")
    ap.add_argument("--pool", default="seen")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--decode", choices=["greedy", "beam4"], default="greedy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--latency-sample", type=int, default=200, help="batch=1 重测的句数")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(0.95)
    out_dir = RUNS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [SET_FILES[args.split][d] for d in args.domains.split(",") if d in SET_FILES[args.split]]
    rows = load_rows(files, args.limit)
    print(f"[data] {args.split} {len(rows)} 句 from {files}", flush=True)

    # 先检索再加载 LLM：bge-m3 用完即释放，避免与 Qwen 叠加占显存
    examples_all = (load_examples(rows, args.pool, args.k, args.domain_gated)
                    if args.mode in ("rag", "sft_rag") else None)
    retrieved_log = ([{"id": r["id"], "examples": [e["id"] for e in ex]}
                      for r, ex in zip(rows, examples_all)] if examples_all else [])

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                attn_implementation="sdpa").cuda().eval()
    if args.adapter:
        from peft import PeftModel
        # DPO adapter 训练时叠在已合并 SFT 的权重上，adapter_config 里
        # 却只记了 base=Qwen。直接挂到原始 Qwen 上等于丢掉 SFT 的全部增益
        # 直接挂到原始 Qwen 会退回零样本水平。先合并 SFT 再叠 DPO。
        if args.base_adapter and Path(args.adapter).resolve() != Path(args.base_adapter).resolve():
            model = PeftModel.from_pretrained(model, args.base_adapter).merge_and_unload()
            print(f"[load] 先合并 base adapter: {args.base_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        print(f"[load] adapter merged: {args.adapter}", flush=True)

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id)
    if args.decode == "beam4":
        gen_kw.update(num_beams=4, num_return_sequences=1)

    t_start = time.perf_counter()
    results, total_gen = [], 0
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        examples_per_row = examples_all[start:start + len(chunk)] if examples_all else None
        texts = [tok.apply_chat_template(build_messages(r["src"], exs if exs else None,
                                                          args.rag_format),
                                         tokenize=False, add_generation_prompt=True)
                 for r, exs in zip(chunk, examples_per_row or [None] * len(chunk))]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[:, enc["input_ids"].shape[1]:]
        # 批量生成按本批最长 padding，逐句长度要数非 pad token
        gen_len = (gen != tok.pad_token_id).sum(dim=1)
        total_gen += int(gen_len.sum())
        preds = tok.batch_decode(gen, skip_special_tokens=True)
        for r, p, gl in zip(chunk, preds, gen_len):
            results.append({"id": r["id"], "domain": r["domain"], "src": r["src"],
                            "tgt": r.get("tgt", ""), "pred": p.strip(),
                            "prompt_tokens": int(enc["input_ids"].shape[1]),
                            "gen_tokens": int(gl),
                            "batch_ms_per_sentence": round(1000 * dt / len(chunk), 2)})
        if start % (args.batch * 20) == 0:
            print(f"  {start + len(chunk)}/{len(rows)}  {dt:.2f}s/batch", flush=True)

    wall = time.perf_counter() - t_start
    with (out_dir / "pass1.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if retrieved_log:
        with (out_dir / "retrieved.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in retrieved_log:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # batch=1 逐句延迟（真实交互口径）
    lat = []
    sample = rows[:args.latency_sample] if args.latency_sample else []
    if sample:
        for r in sample:
            text = tok.apply_chat_template(build_messages(r["src"], None), tokenize=False,
                                           add_generation_prompt=True)
            enc = tok([text], return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model.generate(**enc, **gen_kw)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
    lat_stats = {"n": len(lat),
                 "p50_ms": round(lat[len(lat) // 2], 1) if lat else None,
                 "p95_ms": round(lat[int(len(lat) * 0.95)], 1) if lat else None,
                 "mean_ms": round(sum(lat) / len(lat), 1) if lat else None}

    meta = {"run_id": args.run_id, "mode": args.mode, "split": args.split,
            "domains": args.domains, "adapter": args.adapter,
            "k": args.k if examples_all else None,
            "rag_format": args.rag_format,
            "pool": args.pool if examples_all else None, "decode": args.decode,
            "batch": args.batch, "max_new_tokens": args.max_new_tokens,
            "rows": len(results), "wallclock_min": round(wall / 60, 2),
            "sent_per_sec": round(len(results) / wall, 2),
            "gen_tokens_total": total_gen,
            "tok_per_sec": round(total_gen / wall, 1),
            "latency_batch1": lat_stats,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "git_commit": git_commit(), "seed": 42,
            "model": MODEL, "model_revision": os.environ.get("MODEL_REV", "main")}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[meta] " + json.dumps(meta, ensure_ascii=False), flush=True)
    print(f"saved -> {out_dir/'pass1.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
