"""G5 前置：DPO 偏好对齐（DPOConfig 字段已按 TRL 1.18 实测校验）。

关键：precompute_ref_log_probs=True。
PEFT 下 ref_model=None 时 TRL 用同一份基座权重关掉 adapter 算 ref logprob，
权重本来就一份，省不下模型显存；真正的收益是把 ref 前向从训练步里拿掉
（预先算完并缓存，实测 36 对/s，18k 对约 8 分钟）。

用法：
  python src/train/dpo_lora.py
  python src/train/dpo_lora.py --stage-random   # 对照臂：打乱 chosen/rejected
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
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


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="g5_dpo_r16_s42")
    ap.add_argument("--init-adapter", dest="init_adapter",
                    default=str(RUNS / "g3_sft_r16_s42" / "adapter"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-length", dest="max_length", type=int, default=384)
    ap.add_argument("--stage-random", dest="stage_random", action="store_true",
                    help="对照臂：随机交换 chosen/rejected，检验增益是否真来自偏好数据")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    torch.cuda.set_per_process_memory_fraction(0.95)

    train = load("dpo.pairs.train.jsonl")
    dev = load("dpo.pairs.dev.jsonl")
    if args.stage_random:
        rng = random.Random(args.seed)
        for r in train:
            if rng.random() < 0.5:
                r["chosen"], r["rejected"] = r["rejected"], r["chosen"]
        print(f"[random] 已随机交换一半样本的 chosen/rejected（{len(train)} 对）", flush=True)

    print(f"[data] train {len(train)} 对，dev {len(dev)} 对", flush=True)

    out_dir = RUNS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.init_adapter and Path(args.init_adapter).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
        print(f"[init] 已合并 SFT adapter: {args.init_adapter}", flush=True)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    steps = max(1, len(train) // (args.batch * args.accum))
    cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=args.beta,
        loss_type="sigmoid",
        label_smoothing=0.0,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(10, int(0.1 * steps)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=8,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_empty_cache_steps=100,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        run_name=args.run_id,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None, args=cfg,
        train_dataset=Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"],
                                          "rejected": r["rejected"]} for r in train]),
        eval_dataset=Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"],
                                         "rejected": r["rejected"]} for r in dev]),
        peft_config=lora, processing_class=tok,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    wall = (time.perf_counter() - t0) / 3600

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    hist = [h for h in trainer.state.log_history if "rewards/margins" in h]
    meta = {"run_id": args.run_id, "pairs": len(train), "beta": args.beta, "lr": args.lr,
            "hours": round(wall, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "final_margin": hist[-1]["rewards/margins"] if hist else None,
            "final_accuracy": hist[-1]["rewards/accuracies"] if hist else None,
            "stage_random": args.stage_random,
            "git_commit": os.popen("git rev-parse --short HEAD").read().strip() or "n/a"}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("[meta] " + json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
