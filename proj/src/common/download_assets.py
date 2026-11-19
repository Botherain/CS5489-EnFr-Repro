"""Stage-0 资产下载：模型权重与评测模型统一落到 D 盘 HF 缓存。

用法：
    $env:HF_HOME="./hf"
    python src/common/download_assets.py [--only qwen]
"""
import argparse
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ASSETS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "bge": "BAAI/bge-m3",
    "comet_da": "Unbabel/wmt22-comet-da",
    "kiwi": "Unbabel/wmt22-cometkiwi-da",
}


def _label(p: str) -> str:
    """仓库内的路径按相对路径回显，仓库外只留目录名，避免把机器路径写进日志。"""
    path = Path(p)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的 key 列表")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    keys = args.only.split(",") if args.only else list(ASSETS)
    print(f"HF_HOME = {_label(os.environ['HF_HOME'])}", flush=True)
    for k in keys:
        repo = ASSETS[k]
        print(f"[download] {k} -> {repo}", flush=True)
        try:
            path = snapshot_download(repo_id=repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.bin", "*.ckpt", "*.yaml", "*.yml"])
            print(f"[ok] {k}  snapshot {Path(path).name[:8]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {k}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    main()
