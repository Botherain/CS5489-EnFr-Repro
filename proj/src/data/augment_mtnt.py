"""S3+ 数据增强：为 MTNT 语域扩训练量。

三条来源：
  chat     WMT24 Chat en-fr（人工译文，客服对话，15,027 对）
  forward  MTNT 英文单语 81,650 句 -> 用我们的 SFT 模型前向翻译（源侧是真噪声）
  back     MTNT 法语单语 26,497 句 -> 用 Opus-MT fr-en 回译（目标侧是真法语）

合成数据必须与真实数据混采，且比例受控（默认合成:真实 ≈ 2:1），
否则模型会把自己的错误模式固化（自蒸馏退化）。

用法：
  python src/data/augment_mtnt.py --stage chat
  python src/data/augment_mtnt.py --stage back
  python src/data/augment_mtnt.py --stage forward --limit 45000 --batch 32
  python src/data/augment_mtnt.py --stage mix
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.85")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROC = DATA / "processed"
INTERIM = DATA / "interim"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER = ROOT / "runs" / "g3_sft_r16_s42" / "adapter"
SYSTEM = ("You are a professional English-to-French translator. Translate the user's sentence "
          "into natural, accurate French. Output only the translation.")
GB = 1e9


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  {len(rows)} 行", flush=True)


def stage_chat(_) -> None:
    src_dir = DATA / "raw" / "wmt24_chat"
    rows, dropped = [], Counter()
    for split, out_name in (("train", "wmt24chat.train.jsonl"),
                            ("valid", "wmt24chat.dev.jsonl")):
        path = src_dir / split / "en-fr.csv"
        if not path.exists():
            print(f"[chat] 缺 {path}", flush=True)
            continue
        seen = set()
        for i, r in enumerate(csv.DictReader(path.open(encoding="utf-8", newline=""))):
            # 该 CSV 同时包含 en→fr 与 fr→en 两个方向，必须按方向过滤
            if (r.get("source_language"), r.get("target_language")) != ("en", "fr"):
                dropped["wrong_direction"] += 1
                continue
            s, t = (r.get("source") or "").strip(), (r.get("reference") or "").strip()
            if not s or not t:
                dropped["empty"] += 1
                continue
            if (s.lower(), t.lower()) in seen:
                dropped["dup"] += 1
                continue
            seen.add((s.lower(), t.lower()))
            rows.append({"id": f"chat_{split}_{i:06d}", "domain": "chat", "split": split,
                         "src": s, "tgt": t})
        print(f"[chat] {split}: 累计 {len(rows)}", flush=True)
    write_jsonl(PROC / "wmt24chat.train.jsonl",
                [r for r in rows if r["split"] == "train"])
    write_jsonl(PROC / "wmt24chat.dev.jsonl",
                [r for r in rows if r["split"] == "valid"])
    print(f"[chat] 丢弃 {dict(dropped)}", flush=True)


def stage_back(args) -> None:
    """法语单语 -> 英语（Opus-MT fr-en），目标侧是真法语。"""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-fr-en", dtype=torch.float16).cuda().eval()
    print(f"[back] Opus-MT fr-en 加载完成，峰值 {torch.cuda.max_memory_allocated()/GB:.2f}GB",
          flush=True)

    lines = [l.strip() for l in (INTERIM / "mtnt_mono" / "train.fr").open(encoding="utf-8")
             if l.strip()]
    if args.limit:
        lines = lines[: args.limit]
    print(f"[back] {len(lines)} 句法语单语", flush=True)

    out, t0 = [], time.perf_counter()
    for start in range(0, len(lines), args.batch):
        chunk = lines[start:start + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
        preds = tok.batch_decode(gen, skip_special_tokens=True)
        for j, (fr, en) in enumerate(zip(chunk, preds)):
            out.append({"id": f"bt_{start + j:06d}", "domain": "mtnt", "split": "train",
                        "src": en.strip(), "tgt": fr})
        if start % (args.batch * 20) == 0:
            rate = (start + len(chunk)) / (time.perf_counter() - t0)
            print(f"  {start + len(chunk)}/{len(lines)}  {rate:.1f} 句/s", flush=True)
    write_jsonl(INTERIM / "mtnt_bt.jsonl", out)
    print(f"[back] 用时 {(time.perf_counter()-t0)/60:.1f} 分钟", flush=True)


def stage_forward(args) -> None:
    """英文单语 -> 法语（用我们的 SFT 模型），源侧是真噪声。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER)).merge_and_unload().cuda().eval()
    print("[forward] SFT 模型加载完成", flush=True)

    lines = [l.strip() for l in (INTERIM / "mtnt_mono" / "train.en").open(encoding="utf-8")
             if l.strip()]
    lines = lines[: args.limit] if args.limit else lines
    print(f"[forward] {len(lines)} 句英文单语 -> 法语", flush=True)

    out, t0, kept = [], time.perf_counter(), 0
    for start in range(0, len(lines), args.batch):
        chunk = lines[start:start + args.batch]
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": s}],
            tokenize=False, add_generation_prompt=True) for s in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        preds = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, (src, fr) in enumerate(zip(chunk, preds)):
            fr = fr.strip()
            # 质量闸：空、过短、长度比异常、重复度过高的一律不进训练集
            if not fr or len(fr.split()) < 2:
                continue
            ratio = len(fr.split()) / max(1, len(src.split()))
            if not 0.4 <= ratio <= 2.5:
                continue
            toks = fr.split()
            if len(toks) > 8 and len(set(toks)) / len(toks) < 0.35:
                continue
            out.append({"id": f"fwd_{start + j:06d}", "domain": "mtnt", "split": "train",
                        "src": src, "tgt": fr})
            kept += 1
        if start % (args.batch * 20) == 0:
            rate = (start + len(chunk)) / (time.perf_counter() - t0)
            print(f"  {start + len(chunk)}/{len(lines)}  保留 {kept}  {rate:.1f} 句/s  "
                  f"峰值 {torch.cuda.max_memory_allocated()/GB:.2f}GB", flush=True)
    write_jsonl(INTERIM / "mtnt_forward.jsonl", out)
    print(f"[forward] 用时 {(time.perf_counter()-t0)/60:.1f} 分钟", flush=True)


def stage_mix(args) -> None:
    """真实 + Chat + 合成，合成占比受控。"""
    def load(name):
        p = PROC / name
        with p.open(encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    real = load("mtnt.train.jsonl")
    chat = load("wmt24chat.train.jsonl") if (PROC / "wmt24chat.train.jsonl").exists() else []
    emea = load("emea.train.jsonl")
    clean = load("clean.train.jsonl")

    def load_interim(name):
        p = INTERIM / name
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    fwd = load_interim("mtnt_forward.jsonl")
    bt = load_interim("mtnt_bt.jsonl")
    import random
    rng = random.Random(42)
    for pool in (fwd, bt, chat):
        rng.shuffle(pool)

    fwd = fwd[: args.n_forward]
    bt = bt[: args.n_back]
    chat = chat[: args.n_chat]
    synth = fwd + bt
    real_noisy = real + chat

    mix = synth + real_noisy + emea + clean
    rng.shuffle(mix)
    print(f"[mix] 真实噪声 {len(real_noisy)}（MTNT {len(real)} + Chat {len(chat)}）", flush=True)
    print(f"[mix] 合成 {len(synth)}（前向 {len(fwd)} + 回译 {len(bt)}）", flush=True)
    print(f"[mix] EMEA {len(emea)}，clean {len(clean)}", flush=True)
    print(f"[mix] 合计 {len(mix)}，合成:真实 = {len(synth)/max(1,len(real_noisy)):.1f}:1",
          flush=True)
    write_jsonl(PROC / "sft.train.augmented.jsonl", mix)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["chat", "back", "forward", "mix"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-forward", type=int, default=45000, dest="n_forward")
    ap.add_argument("--n-back", type=int, default=25000, dest="n_back")
    ap.add_argument("--n-chat", type=int, default=15000, dest="n_chat")
    args = ap.parse_args()
    {"chat": stage_chat, "back": stage_back,
     "forward": stage_forward, "mix": stage_mix}[args.stage](args)


if __name__ == "__main__":
    main()
