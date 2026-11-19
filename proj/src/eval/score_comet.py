"""Pass2 打分：在独立 comet venv 里对 pass1.jsonl 批量算 COMET-DA（有参考）与 CometKiwi（免参考）。

必须在 comet venv 下运行：
    & "D:\\HKStudy\\.venv-comet\\Scripts\\python.exe" src\\eval\\score_comet.py --runs g1_zs,g2_rag

设计：
  1) 句级分数全部落盘，供 bootstrap / 长度分桶 / Shapley 复用，绝不重算
  2) Kiwi 是 gated 仓库，缺失时自动跳过并记录，不阻塞主流程（见 FEASIBILITY_REVIEW §4）
  3) 与生成进程完全隔离：本脚本运行时显存里只有 COMET
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

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

DA_MODEL = "Unbabel/wmt22-comet-da"
QE_LOCAL = str(ROOT / "runs" / "qe_local" / "wmt22-cometkiwi-da-mirror" / "checkpoints" / "model.ckpt")
QE_CANDIDATES = [QE_LOCAL, "Unbabel/wmt22-cometkiwi-da"]


def load_model(repo: str):
    """repo 可以是 HF 仓库名，也可以是本地 ckpt 路径（镜像转换产物）。"""
    from comet import download_model, load_from_checkpoint
    path = repo if Path(repo).exists() else download_model(repo)
    return load_from_checkpoint(path)


def predict(model, data: list[dict], batch_size: int):
    """兼容 comet 2.x 的两种 predict 签名与两种返回类型。"""
    try:
        out = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=False)
    except TypeError:
        out = model.predict(data, batch_size=batch_size, accelerator="gpu", devices=1,
                            progress_bar=False)
    if isinstance(out, tuple):          # (scores, system_score)
        return list(out[0]), float(out[1])
    scores = getattr(out, "scores", None) or out["scores"]
    system = getattr(out, "system_score", None)
    if system is None:
        system = out.get("system_score")
    return list(scores), float(system)


def read_pass1(run_id: str, variant: str = "pass1") -> list[dict]:
    path = RUNS / run_id / f"{variant}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="逗号分隔的 run_id")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--variant", default="pass1", help="pass1 / draft，决定读哪个 jsonl")
    ap.add_argument("--skip-qe", action="store_true")
    ap.add_argument("--qe-model", default=None, help="覆盖 QE 仓库名")
    args = ap.parse_args()

    import torch
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    da = load_model(DA_MODEL)
    print(f"[load] {DA_MODEL} ok", flush=True)

    qe = None
    qe_used = None
    if not args.skip_qe:
        for repo in ([args.qe_model] if args.qe_model else QE_CANDIDATES):
            try:
                qe = load_model(repo)
                qe_used = repo
                print(f"[load] QE = {repo} ok", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] QE {repo} 不可用：{type(exc).__name__} {str(exc)[:90]}", flush=True)
        if qe is None:
            print("[warn] 无可用 QE 模型，Kiwi 分数留空（走 FEASIBILITY_REVIEW §4 降级）", flush=True)

    for run_id in [r.strip() for r in args.runs.split(",") if r.strip()]:
        rows = read_pass1(run_id, args.variant)
        if not rows:
            print(f"[skip] {run_id} 无数据", flush=True)
            continue
        da_data = [{"src": r["src"], "mt": r["pred"], "ref": r["tgt"]} for r in rows]
        t0 = time.perf_counter()
        da_seg, da_sys = predict(da, da_data, args.batch_size)
        da_time = time.perf_counter() - t0

        qe_seg = [None] * len(rows)
        qe_time = 0.0
        if qe is not None:
            qe_data = [{"src": r["src"], "mt": r["pred"]} for r in rows]
            t1 = time.perf_counter()
            qe_seg, _ = predict(qe, qe_data, args.batch_size)
            qe_time = time.perf_counter() - t1

        suffix = "" if args.variant == "pass1" else f"_{args.variant}"
        out = RUNS / run_id / f"scores{suffix}.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for r, d, q in zip(rows, da_seg, qe_seg):
                rec = {"id": r["id"], "domain": r["domain"], "src": r["src"], "tgt": r["tgt"],
                       "pred": r["pred"], "comet_da": round(float(d), 5),
                       "kiwi": (round(float(q), 5) if q is not None else None),
                       "src_tokens": r.get("prompt_tokens"), "gen_tokens": r.get("gen_tokens")}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        by_domain: dict[str, list[float]] = {}
        for r, d in zip(rows, da_seg):
            by_domain.setdefault(r["domain"], []).append(float(d))
        summary = {"run_id": run_id, "rows": len(rows),
                   "comet_da_system": round(da_sys, 5),
                   "comet_da_by_domain": {k: round(sum(v) / len(v), 5) for k, v in by_domain.items()},
                   "macro_comet_da": round(sum(sum(v) / len(v) for v in by_domain.values())
                                           / len(by_domain), 5),
                   "da_seconds": round(da_time, 1), "qe_seconds": round(qe_time, 1),
                   "sent_per_sec": round(len(rows) / max(da_time, 1e-6), 1),
                   "qe_model": qe_used}
        (RUNS / run_id / f"scores_summary{suffix}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{run_id}] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
