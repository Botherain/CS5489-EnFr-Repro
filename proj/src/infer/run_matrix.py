"""按矩阵批量跑后续实验（生成 -> 打分 -> 路由），一条命令跑完一组。

设计：
  - 每个 (run_id, split) 一个产物目录，产物存在则跳过（--force 覆盖）
  - 主 venv 与 comet venv 分开调度
  - 日志统一落 logs/matrix_<group>.log，失败即停并打印最后 20 行

组：
  g4       G4 的 inline 变体（修 few-shot 轮次与 SFT 单轮格式冲突的对照）
  g5       G5 全管线：Pass1（含 draft）-> Pass2（final+draft 打分）-> Pass3（τ 扫描）
  abl      G5 四个留一（test）
  all      以上全部

用法：
  python src/infer/run_matrix.py --group g4
  python src/infer/run_matrix.py --group all
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
PY = sys.executable
CMT = os.environ.get("COMET_PY", "")
SFT = "runs/g3_sft_r16_s42/adapter"
DPO = "runs/g5_dpo_r16_s42/adapter"


def has(run_id: str, *files: str) -> bool:
    d = RUNS / run_id
    return all((d / f).exists() for f in files)


def sh(cmd: list[str], log) -> None:
    """跑一步，子进程输出边到边打印（--group all 时能实时看到批次进度）。"""
    if not cmd[0]:
        raise SystemExit("COMET 解释器未设置：先 export COMET_PY 指向 .venv-comet 的解释器")
    print("$ " + " ".join(cmd[:3] + ["…"]) if len(cmd) > 3 else "$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    log.write(f"\n### {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
        log.write(line)
    proc.wait()
    dt = time.perf_counter() - t0
    log.flush()
    if proc.returncode != 0:
        raise SystemExit(f"失败（{dt:.0f}s）：{cmd[1:3]}")
    print(f"  ok {dt:.0f}s", flush=True)


def gen(run_id: str, mode: str, split: str, adapter: str | None, extra: list[str]) -> list[str]:
    cmd = [PY, "src/infer/generate.py", "--run-id", run_id, "--mode", mode, "--set", split]
    if adapter:
        cmd += ["--adapter", adapter]
    return cmd + extra


def score(run_id: str, variant: str = "pass1", qe: bool = True) -> list[str]:
    cmd = [CMT, "src/eval/score_comet.py", "--runs", run_id, "--batch-size", "48"]
    if variant != "pass1":
        cmd += ["--variant", variant]
    if not qe:
        cmd += ["--skip-qe"]
    return cmd


def plan(group: str, force: bool) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []

    def add(tag, run_id, cmd, files=("pass1.jsonl",)):
        if not force and has(run_id, *files):
            print(f"[skip] {run_id} 已有 {files}", flush=True)
            return
        steps.append((tag, cmd))

    if group in ("g4", "all"):
        add("g4 inline test", "g4_sft_rag_inline",
            gen("g4_sft_rag_inline", "sft_rag", "test", SFT,
                ["--k", "3", "--rag-format", "inline", "--latency-sample", "100"]))
        steps.append(("score g4 inline", score("g4_sft_rag_inline")))
        add("g4 inline dev", "g4_sft_rag_inline_dev",
            gen("g4_sft_rag_inline_dev", "sft_rag", "dev", SFT,
                ["--k", "3", "--rag-format", "inline", "--latency-sample", "0"]))
        steps.append(("score g4 inline dev", score("g4_sft_rag_inline_dev", qe=False)))

    if group in ("g5", "all"):
        add("g5 pass1 test", "g5_full",
            [PY, "src/infer/pipeline_g5.py", "--run-id", "g5_full", "--split", "test"],
            files=("pass1.jsonl", "draft.jsonl"))
        steps.append(("score g5 final", score("g5_full", "pass1")))
        steps.append(("score g5 draft", score("g5_full", "draft")))
        steps.append(("route", [PY, "src/eval/route.py", "--run", "g5_full",
                                "--g4", "g4_sft_rag_inline"]))

        add("g5 pass1 dev", "g5_full_dev",
            [PY, "src/infer/pipeline_g5.py", "--run-id", "g5_full_dev", "--split", "dev"],
            files=("pass1.jsonl", "draft.jsonl"))
        steps.append(("score g5 dev final", score("g5_full_dev", "pass1")))
        steps.append(("score g5 dev draft", score("g5_full_dev", "draft")))
        steps.append(("route dev", [PY, "src/eval/route.py", "--run", "g5_full_dev",
                                    "--g4", "g4_sft_rag_inline_dev"]))

    if group in ("abl", "all"):
        arms = {
            "g5_no_dpo": ["--no-dpo"],
            "g5_no_rag": ["--no-rag"],
            "g5_no_rag_len": ["--rag-control", "random"],
            "g5_no_con": ["--no-con"],
            "g5_no_agent": ["--no-agent"],
        }
        for run_id, flags in arms.items():
            add(run_id, run_id,
                [PY, "src/infer/pipeline_g5.py", "--run-id", run_id, "--split", "test"] + flags,
                files=("pass1.jsonl",))
            steps.append((f"score {run_id}", score(run_id)))
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["g4", "g5", "abl", "all"], required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    steps = plan(args.group, args.force)
    if not steps:
        print("[done] 该组产物齐全，无需执行", flush=True)
        return
    if not CMT and any(not cmd[0] for _tag, cmd in steps):
        raise SystemExit("这批步骤要在 comet 环境里打分：先 export COMET_PY 指向 .venv-comet 的解释器")

    log_path = LOGS / f"matrix_{args.group}.log"
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"group={args.group} steps={len(steps)}\n")
        for i, (tag, cmd) in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {tag}", flush=True)
            sh(cmd, log)
    print(f"\n[全部完成] {len(steps)} 步，{(time.perf_counter()-t0)/60:.1f} 分钟 -> {log_path}",
          flush=True)


if __name__ == "__main__":
    main()
