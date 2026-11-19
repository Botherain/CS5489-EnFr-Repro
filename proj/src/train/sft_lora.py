"""G3：LoRA SFT（TRL 1.18 / transformers 5.17 实测可用的参数组合）。

关键点（都来自本机实测，见 logs/smoke_all.log、logs/smoke_env.log）：
  - max_length=384：评测集零截断，MTNT train 截断 1.73%
  - packing=False：数据极短（p50≈40 token），打包反而把吞吐从 17.6 降到 6.6 样本/s
  - assistant_only_loss=True：取代已删除的 DataCollatorForCompletionOnlyLM
  - batch=16：b16→b32 吞吐不再增长（15.3 → 15.0 样本/s），显存却从 8.8GB 涨到 9.5GB
  - warmup_steps：本版 TRL 没有 warmup_ratio
  - optim=adamw_torch：不用 bitsandbytes（sm_120 内核风险，LoRA 优化器状态仅 0.22GB）
  - dataloader_num_workers=0：Windows 下避免 spawn 卡死

用法：
  python src/train/sft_lora.py --epochs 2 --batch 16
  python src/train/sft_lora.py --limit 2000 --epochs 1 --run-id _smoke_sft_full
"""
from __future__ import annotations

import argparse
import io
import json
import os
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
RUNS = ROOT / "runs"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = ("You are a professional English-to-French translator. Translate the user's sentence "
          "into natural, accurate French. Output only the translation.")
GB = 1e9


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def to_messages(rows: list[dict]) -> list[dict]:
    return [{"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["src"]},
        {"role": "assistant", "content": r["tgt"]},
    ]} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="g3_sft_r16_s42")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", dest="max_length", type=int, default=384)
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="调试用，限制每个域样本数")
    ap.add_argument("--train-file", dest="train_file", default=None,
                    help="用指定 jsonl 覆盖默认训练集（如数据增强后的 sft.train.augmented.jsonl）")
    ap.add_argument("--eval-limit", dest="eval_limit", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.cuda.set_per_process_memory_fraction(0.95)

    if args.train_file:
        path = Path(args.train_file)
        if not path.is_absolute():
            path = DATA / args.train_file
        with path.open(encoding="utf-8") as fh:
            train_rows = [json.loads(l) for l in fh if l.strip()]
        print(f"[data] 使用 {path.name}：{len(train_rows)} 条", flush=True)
    else:
        train_rows = (load("mtnt.train.jsonl") + load("emea.train.jsonl")
                      + load("clean.train.jsonl"))
    if args.limit:
        train_rows = (train_rows[: args.limit // 3] +
                      load("emea.train.jsonl")[: args.limit // 3] +
                      load("clean.train.jsonl")[: args.limit // 3])
    eval_rows = load("mtnt.dev.jsonl")[: args.eval_limit // 2] + \
                load("emea.dev.jsonl")[: args.eval_limit // 2]
    print(f"[data] train {len(train_rows)}，eval {len(eval_rows)}", flush=True)

    out_dir = RUNS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    steps_per_epoch = max(1, len(train_rows) // (args.batch * args.accum))
    cfg = SFTConfig(
        output_dir=str(out_dir),
        max_length=args.max_length,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, int(0.03 * steps_per_epoch * args.epochs)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=False,
        assistant_only_loss=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_empty_cache_steps=100,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=400,
        per_device_eval_batch_size=args.batch,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        report_to=[],
        run_name=args.run_id,
        seed=args.seed,
        use_liger_kernel=False,
        torch_compile=False,
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=Dataset.from_list(to_messages(train_rows)),
        eval_dataset=Dataset.from_list(to_messages(eval_rows)),
        peft_config=lora, processing_class=tok,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[lora] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M = "
          f"{100*trainable/total:.3f}%", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    wall = (time.perf_counter() - t0) / 3600

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    meta = {"run_id": args.run_id, "train_rows": len(train_rows), "eval_rows": len(eval_rows),
            "epochs": args.epochs, "batch": args.batch, "accum": args.accum,
            "max_length": args.max_length, "lr": args.lr, "lora_r": args.lora_r,
            "trainable_M": round(trainable / 1e6, 2),
            "trainable_pct": round(100 * trainable / total, 3),
            "hours": round(wall, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "final_train_loss": trainer.state.log_history[-1].get("loss"),
            "steps_per_epoch": steps_per_epoch,
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip() or "n/a"}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("[meta] " + json.dumps(meta, ensure_ascii=False), flush=True)
    print(f"[out] {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
