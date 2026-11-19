"""G5 管线 Pass1：对每句同时产出 light(draft) 与 heavy(rag→critic→refiner) 两路结果。

为什么两路都产：τ 是离线过滤器（见 EXPERIMENT_PLAN §5），一次生成就能覆盖 8 个 τ 点，
τ 扫描的 GPU 成本为 0，也避开了"COMET 在独立 venv"与"路由需要 Kiwi 常驻生成进程"的冲突。

输出：
  runs/<id>/pass1.jsonl   pred = 最终（guarded）译文，附 draft / heavy / issues 等中间态
  runs/<id>/draft.jsonl   draft 单独一份，供 Pass2 打 Kiwi/DA
  runs/<id>/meta.json     阶段耗时、显存、各 fallback 计数

消融由开关控制：--no-dpo / --no-rag / --no-con / --no-agent
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.85")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GB = 1e9

SYSTEM = ("You are a professional English-to-French translator. Translate the user's sentence "
          "into natural, accurate French. Output only the translation.")

CRITIC_SYS = ("You are a bilingual reviewer of English-to-French medical and social-media "
              "translations. List concrete problems, or reply with {\"issues\": []} if the "
              "translation is faithful and idiomatic. Output JSON only: "
              '{"issues":[{"type":"term|idiom|grammar|omission|register","fix":"..."}],'
              '"severity":1-5}')

REFINER_SYS = ("You are a professional English-to-French translator. Given the source, a draft "
               "and reviewer notes, produce the final translation. Apply the fixes, keep "
               "everything that was already correct. Output the translation only.")

SET_FILES = {"test": {"mtnt": "mtnt.test.jsonl", "emea": "emea.test.jsonl",
                      "clean": "clean.test.jsonl"},
             "dev": {"mtnt": "mtnt.dev.jsonl", "emea": "emea.dev.jsonl",
                     "clean": "clean.dev.jsonl"}}


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_glossary() -> list[dict]:
    path = DATA / "glossary.emea.json"
    if not path.exists():
        return []
    terms = json.loads(path.read_text(encoding="utf-8"))
    for t in terms:
        t["_pat"] = re.compile(r"\b" + re.escape(t["src"].lower()) + r"\b")
    return terms


def load_retrieval(pool: str, k: int) -> dict[str, list[dict]]:
    # 索引在 data/index，不在 data/processed/index 下
    path = ROOT / "data" / "index" / "cache" / f"{pool}.top{max(k, 12)}.jsonl"
    out: dict[str, list[dict]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    out[d["id"]] = d["examples"][:k]
    return out


def load_pool_rows() -> list[dict]:
    rows = []
    for name in ("mtnt.train.jsonl", "emea.train.jsonl"):
        rows += load(name)
    return rows


def random_examples(pool: list[dict], k: int, target_tokens: int,
                    rng) -> list[dict]:
    """长度匹配的随机示例：用来做 `--rag-control random` 对照组。

    检索组与随机组放进 prompt 的 token 数接近，才能把"检索内容有效"和
    "prompt 变长本身有效"分开——否则 RAG 的增益说不清是哪来的。
    """
    lo, hi = int(target_tokens * 0.75), int(target_tokens * 1.25)
    for _ in range(200):
        picked = rng.sample(pool, k)
        total = sum(len((e["src"] + " " + e["tgt"]).split()) for e in picked)
        if lo <= total <= hi:
            return [{"src": e["src"], "tgt": e["tgt"]} for e in picked]
    return [{"src": e["src"], "tgt": e["tgt"]} for e in rng.sample(pool, k)]


def chat(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def repeat_ratio(text: str, n: int = 4) -> float:
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def guard(draft: str, final: str) -> tuple[str, str | None]:
    """任一条不满足就回退 draft，并返回回退原因。"""
    if not final.strip():
        return draft, "empty"
    if not 0.5 <= len(final.split()) / max(1, len(draft.split())) <= 2.0:
        return draft, "length"
    if repeat_ratio(final) > 0.30:
        return draft, "repeat"
    if re.search(r"[A-Za-z]{4,}\s+[A-Za-z]{4,}\s*$", final) and final.strip().endswith((" the", " and", " of")):
        return draft, "dangling"
    return final, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="g5_full")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--domains", default="mtnt,emea,clean")
    ap.add_argument("--sft-adapter", dest="sft_adapter",
                    default=str(RUNS / "g3_sft_r16_s42" / "adapter"))
    ap.add_argument("--dpo-adapter", dest="dpo_adapter",
                    default=str(RUNS / "g5_dpo_r16_s42" / "adapter"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", default="seen")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=160)
    ap.add_argument("--no-dpo", dest="use_dpo", action="store_false")
    ap.add_argument("--no-rag", dest="use_rag", action="store_false")
    ap.add_argument("--no-con", dest="use_con", action="store_false")
    ap.add_argument("--no-agent", dest="use_agent", action="store_false")
    ap.add_argument("--rag-control", dest="rag_control", choices=["none", "random"],
                    default="none", help="random = 用长度匹配的随机示例替代检索")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(0.95)
    out_dir = RUNS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    sft = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload().cuda().eval()
    print(f"[load] SFT adapter 合并完成", flush=True)

    dpo = None
    if args.use_dpo and Path(args.dpo_adapter).exists():
        # DPO adapter 挂在已合并 SFT 的权重上：adapter_config 只记 base=Qwen，
        # 直接挂到原始 Qwen 会丢掉 SFT 的增益。deepcopy 是为了不污染轻通道的权重。
        import copy
        dpo = copy.deepcopy(sft)
        dpo = PeftModel.from_pretrained(dpo, args.dpo_adapter).merge_and_unload().eval()
        print(f"[load] DPO adapter 叠在 SFT 之上合并完成", flush=True)

    terms = load_glossary() if args.use_con else []
    retr = load_retrieval(args.pool, args.k) if args.use_rag else {}
    pool_rows = load_pool_rows() if args.rag_control == "random" else []
    rng_ctrl = __import__("random").Random(42)
    print(f"[load] 术语 {len(terms)} 条，检索缓存 {len(retr)} 条", flush=True)

    files = [SET_FILES[args.split][d] for d in args.domains.split(",") if d in SET_FILES[args.split]]
    rows = [r for f in files for r in load(f)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {args.split} {len(rows)} 句", flush=True)

    def gen(model, prompts: list[str], max_new: int) -> tuple[list[str], float]:
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in texts], dt / len(prompts)

    stats = {"rows": len(rows), "fallback": 0, "fallback_reasons": {}, "glossary_hits": 0,
             "t_draft": 0.0, "t_heavy": 0.0, "t_critic": 0.0, "t_refiner": 0.0}
    results, drafts = [], []
    t_all = time.perf_counter()

    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]

        # 1) light path：SFT 直出
        draft_prompts = [chat(tok, SYSTEM, r["src"]) for r in chunk]
        drafts_txt, t_d = gen(sft, draft_prompts, args.max_new_tokens)
        stats["t_draft"] += t_d * len(chunk)

        # 2) heavy path 的上下文：检索示例（user/assistant 轮次）+ 术语提示（最后一条 user）
        #    单条 user 消息塞示例（inline）比 user/assistant 轮次低约 0.09 COMET，
        #    显式轮次结构（turns）明显更好，所以这里按 turns 组 prompt。
        heavy_prompts, hits = [], []
        for r, d in zip(chunk, drafts_txt):
            exs = retr.get(r["id"], []) if retr else []
            if args.rag_control == "random":
                tgt_tokens = sum(len((e["src"] + " " + e["tgt"]).split()) for e in exs) or 60
                exs = random_examples(pool_rows, args.k, tgt_tokens, rng_ctrl)
            gl = [t for t in terms if t["_pat"].search(r["src"].lower())] if terms else []
            msgs = [{"role": "system", "content": SYSTEM}]
            for e in exs:
                msgs.append({"role": "user", "content": e["src"]})
                msgs.append({"role": "assistant", "content": e["tgt"]})
            tail = []
            if gl:
                tail.append("Required terminology: " +
                            "; ".join(f"{t['src']} -> {t['fr']}" for t in gl))
            tail.append(r["src"])
            msgs.append({"role": "user", "content": "\n\n".join(tail)})
            heavy_prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                         add_generation_prompt=True))
            hits.append(len(gl))

        heavy_model = dpo or sft
        heavy_txt, t_h = gen(heavy_model, heavy_prompts, args.max_new_tokens)
        stats["t_heavy"] += t_h * len(chunk)
        stats["glossary_hits"] += sum(hits)

        # 3) Critic
        final_txt = list(heavy_txt)
        if args.use_agent:
            critic_prompts = [chat(tok, CRITIC_SYS,
                                   f"Source: {r['src']}\nDraft: {h}\nReview JSON:")
                              for r, h in zip(chunk, heavy_txt)]
            critiques, t_c = gen(sft, critic_prompts, 160)
            stats["t_critic"] += t_c * len(chunk)
            refiner_prompts = [chat(tok, REFINER_SYS,
                                    f"Source: {r['src']}\nDraft: {h}\nReviewer notes: {c}\n"
                                    f"Final French translation:")
                               for r, h, c in zip(chunk, heavy_txt, critiques)]
            refined, t_r = gen(heavy_model, refiner_prompts, args.max_new_tokens)
            stats["t_refiner"] += t_r * len(chunk)
            final_txt = []
            for r, h, f in zip(chunk, heavy_txt, refined):
                kept, reason = guard(h, f)
                if reason:
                    stats["fallback"] += 1
                    stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + 1
                final_txt.append(kept)
        else:
            critiques = [""] * len(chunk)

        for r, d, h, c, f in zip(chunk, drafts_txt, heavy_txt, critiques, final_txt):
            results.append({"id": r["id"], "domain": r["domain"], "src": r["src"],
                            "tgt": r.get("tgt", ""), "pred": f, "draft": d,
                            "heavy": h, "critic": c[:400],
                            "glossary_hits": len([t for t in terms
                                                  if t["_pat"].search(r["src"].lower())])})
            drafts.append({"id": r["id"], "domain": r["domain"], "src": r["src"],
                           "tgt": r.get("tgt", ""), "pred": d})
        if start % (args.batch * 10) == 0:
            done = start + len(chunk)
            print(f"  {done}/{len(rows)}  {time.perf_counter()-t_all:.0f}s  "
                  f"fallback {stats['fallback']}", flush=True)

    wall = time.perf_counter() - t_all
    for name, data in (("pass1", results), ("draft", drafts)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"run_id": args.run_id, "split": args.split, "rows": len(rows),
            "flags": {"dpo": args.use_dpo and dpo is not None, "rag": args.use_rag,
                      "rag_control": args.rag_control,
                      "con": args.use_con, "agent": args.use_agent},
            "k": args.k, "pool": args.pool,
            "wallclock_min": round(wall / 60, 2),
            "mean_ms": {k: round(1000 * v / len(rows), 1)
                        for k, v in stats.items() if k.startswith("t_")},
            "fallback": stats["fallback"], "fallback_reasons": stats["fallback_reasons"],
            "glossary_hits": stats["glossary_hits"],
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / GB, 3)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("[meta] " + json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
