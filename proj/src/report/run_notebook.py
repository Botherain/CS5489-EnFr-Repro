"""在终端里执行报告 notebook，并把每个单元的输出实时打出来。

`jupyter nbconvert --execute` 会把输出攒进 ipynb 再一次性写盘，跑十小时的实验时看不到
进度；这里直接跟 kernel 说话，拿到 stream 消息就打印，同时把输出写回 notebook。

用法：
    python src/report/run_notebook.py                 # 只重算指标与表格
    python src/report/run_notebook.py --all           # 全部阶段（约 10 GPU 小时）
    python src/report/run_notebook.py --only data,sft # 只跑指定组
    python src/report/run_notebook.py --setup         # 先装环境、权重与数据（Cell S0.1–S0.3）
    python src/report/run_notebook.py --setup=venv,data
    python src/report/run_notebook.py --force         # 忽略已有产物重跑
    python src/report/run_notebook.py --kernel my_local_env
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "reports" / "report" / "CS5489_Project_Report.ipynb"

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def pick_kernel(requested: str | None) -> str:
    from jupyter_client.kernelspec import KernelSpecManager
    specs = KernelSpecManager().find_kernel_specs()
    name = requested or os.environ.get("NB_KERNEL") or ""
    if name and name in specs:
        return name
    if not name:
        for cand in ("my_local_env", "python3"):
            if cand in specs:
                return cand
    raise SystemExit(
        f"kernel {name!r} 不可用；现有：{sorted(specs)}。"
        "先建一个指向本项目 venv 的内核：\n"
        "  python -m ipykernel install --user --name my_local_env --display-name my_local_env")


def outputs_of(msg: dict):
    """把 iopub 消息转成 nbformat 输出对象；不可见的类型返回 None。"""
    from nbformat.v4 import output_from_msg
    try:
        return output_from_msg(msg)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="打开全部阶段（NB_RUN=all）")
    ap.add_argument("--only", default=None, help="只打开这些组，逗号分隔（NB_RUN=...）")
    ap.add_argument("--setup", nargs="?", const="all", default=None,
                    help="跑安装单元：--setup 全部，--setup=venv,data 只跑指定项（NB_SETUP=...）")
    ap.add_argument("--force", action="store_true", help="忽略已有产物重跑（NB_FORCE=1）")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--nb", default=str(NOTEBOOK))
    args = ap.parse_args()

    if args.all:
        os.environ["NB_RUN"] = "all"
    elif args.only:
        os.environ["NB_RUN"] = args.only
    if args.setup:
        os.environ["NB_SETUP"] = args.setup
    if args.force:
        os.environ["NB_FORCE"] = "1"

    import nbformat
    from jupyter_client import KernelManager

    nb_path = Path(args.nb).resolve()
    nb = nbformat.read(nb_path, as_version=4)
    kernel = pick_kernel(args.kernel)
    print(f"notebook {nb_path.name} | kernel {kernel} | NB_RUN={os.environ.get('NB_RUN', '-')} "
          f"| NB_SETUP={os.environ.get('NB_SETUP', '-')} | NB_FORCE={os.environ.get('NB_FORCE', '-')}",
          flush=True)

    km = KernelManager(kernel_name=kernel)
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    # kernel 自己的 stderr（ipykernel 的告警）落到日志，别混进进度输出
    klog = (log_dir / "kernel.log").open("w", encoding="utf-8")
    km.start_kernel(cwd=str(ROOT), stdout=klog.fileno(), stderr=klog.fileno())
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=180)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    t_run = time.time()
    failed: dict | None = None
    interrupted = False

    for idx, cell in enumerate(code_cells, 1):
        head = next((l for l in cell.source.splitlines() if l.strip()), "")[:78]
        print(f"\n[{time.strftime('%H:%M:%S')}] {idx}/{len(code_cells)}  {head}", flush=True)
        cell.outputs = []
        t0 = time.time()
        msg_id = kc.execute(cell.source, store_history=False, allow_stdin=False,
                            stop_on_error=False)
        try:
            while True:
                msg = kc.get_iopub_msg(timeout=None)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind = msg["msg_type"]
                if kind == "stream":
                    text = msg["content"]["text"]
                    sys.stdout.write(text)
                    sys.stdout.flush()
                elif kind == "error":
                    print("\n".join(msg["content"]["traceback"]), flush=True)
                    failed = {"cell": idx, "ename": msg["content"]["ename"]}
                elif kind == "display_data":
                    mimetypes = list(msg["content"].get("data", {}))
                    print(f"    [display {','.join(mimetypes)}]", flush=True)
                elif kind == "status" and msg["content"]["execution_state"] == "idle":
                    break
                out = outputs_of(msg) if kind != "status" else None
                if out is not None:
                    cell.outputs.append(out)
        except KeyboardInterrupt:
            interrupted = True
            print("\n^C 中断，正在停止 kernel …", flush=True)
            kc.interrupt_kernel()
            break
        print(f"    ({time.time() - t0:.1f}s)", flush=True)
        if failed:
            break

    kc.stop_channels()
    km.shutdown_kernel(now=True)
    klog.close()
    nbformat.write(nb, nb_path)
    print(f"\n已写回 {nb_path.relative_to(ROOT)} | 总耗时 {(time.time() - t_run) / 60:.1f} min",
          flush=True)

    if interrupted:
        return 130
    if failed:
        print(f"单元 {failed['cell']} 报错：{failed['ename']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
