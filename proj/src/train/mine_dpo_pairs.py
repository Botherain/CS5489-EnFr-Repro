"""DPO 偏好数据构造：SFT 模型采样候选 -> 外部打分 -> 成对。

三个阶段（打分必须在 comet venv 里跑，见 score_candidates.py）：
  --stage generate   采样 K=4 候选，落 data/interim/candidates.jsonl
  --stage pair       读回打分，构造成对，落 data/processed/dpo.pairs.{train,dev}.jsonl

采样域配比与评测域同向（MTNT test 1,020 : EMEA test 2,000 ≈ 1:2），所以默认 6,000:12,000。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.85")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
RUNS = ROOT / "runs"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = ("You are a professional English-to-French translator. Translate the user's sentence "
          "into natural, accurate French. Output only the translation.")
GB = 1e9


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def stage_generate(args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(0.95)
    rng = random.Random(args.seed)

    mtnt = load("mtnt.train.jsonl")
    emea = load("emea.train.jsonl")
    rng.shuffle(mtnt)
    rng.shuffle(emea)
    prompts = ([{**r, "domain": "mtnt"} for r in mtnt[: args.n_mtnt]] +
               [{**r, "domain": "emea"} for r in emea[: args.n_emea]])
    rng.shuffle(prompts)
    print(f"[generate] {len(prompts)} 个源句 × K={args.k} 候选", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().cuda().eval()

    INTERIM.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM / "candidates.jsonl"
    t0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for start in range(0, len(prompts), args.batch):
            chunk = prompts[start:start + args.batch]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": r["src"]}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=True, temperature=0.9, top_p=0.95,
                                     num_return_sequences=args.k, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            preds = tok.batch_decode(gen, skip_special_tokens=True)
            for i, r in enumerate(chunk):
                cands = [p.strip() for p in preds[i * args.k:(i + 1) * args.k]]
                fh.write(json.dumps({"id": r["id"], "domain": r["domain"], "src": r["src"],
                                     "tgt": r["tgt"], "cands": cands},
                                    ensure_ascii=False) + "\n")
            if start % (args.batch * 20) == 0:
                done = start + len(chunk)
                rate = done / (time.perf_counter() - t0)
                print(f"  {done}/{len(prompts)}  {rate:.1f} 句/s  "
                      f"峰值 {torch.cuda.max_memory_allocated()/GB:.2f}GB", flush=True)
    print(f"[generate] -> {out_path} 用时 {(time.perf_counter()-t0)/60:.1f} 分钟", flush=True)


def stage_pair(args) -> None:
    scored = [json.loads(l) for l in
              (INTERIM / "candidates.scored.jsonl").open(encoding="utf-8") if l.strip()]
    print(f"[pair] 读入 {len(scored)} 条已打分样本", flush=True)

    pairs, stats = [], Counter()
    for r in scored:
        cands = r["cands"]
        if len(cands) < 2:
            stats["候选不足"] += 1
            continue
        ranked = sorted(cands, key=lambda c: -c["da"])
        best, worst = ranked[0], ranked[-1]
        if best["text"].strip().lower() == worst["text"].strip().lower():
            stats["首尾相同"] += 1
            continue
        if best["da"] - worst["da"] < args.min_delta:
            stats["分差过小"] += 1
            continue
        lb, lw = len(best["text"].split()), len(worst["text"].split())
        if lw == 0 or not (args.ratio_lo <= lb / lw <= args.ratio_hi):
            stats["长度比越界"] += 1
            continue
        pairs.append({"prompt": r["src"], "chosen": best["text"], "rejected": worst["text"],
                      "domain": r["domain"], "delta_da": round(best["da"] - worst["da"], 4),
                      "chosen_da": round(best["da"], 4), "rejected_da": round(worst["da"], 4)})
        stats["保留"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_dev = min(2000, len(pairs) // 10)
    dev, train = pairs[:n_dev], pairs[n_dev:]
    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("dev", dev)):
        with (DATA / f"dpo.pairs.{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[pair] 保留 {len(pairs)}（train {len(train)} / dev {len(dev)}）", flush=True)
    print("[pair] 丢弃原因: " + json.dumps(dict(stats), ensure_ascii=False), flush=True)
    deltas = sorted(p["delta_da"] for p in pairs)
    if deltas:
        q = lambda v, p: v[min(len(v) - 1, int(len(v) * p / 100))]
        print(f"[pair] Δ_DA 分位: p10={q(deltas,10):.3f} p50={q(deltas,50):.3f} "
              f"p90={q(deltas,90):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "pair"], required=True)
    ap.add_argument("--adapter", default=str(RUNS / "g3_sft_r16_s42" / "adapter"))
    ap.add_argument("--n-mtnt", type=int, default=6000, dest="n_mtnt")
    ap.add_argument("--n-emea", type=int, default=12000, dest="n_emea")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160, dest="max_new_tokens")
    ap.add_argument("--min-delta", type=float, default=0.10, dest="min_delta")
    ap.add_argument("--ratio-lo", type=float, default=0.6, dest="ratio_lo")
    ap.add_argument("--ratio-hi", type=float, default=1.6, dest="ratio_hi")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"generate": stage_generate, "pair": stage_pair}[args.stage](args)


if __name__ == "__main__":
    main()
