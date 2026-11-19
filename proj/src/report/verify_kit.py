"""校验复现包完整性：按 MANIFEST.txt 逐个比对 sha256。

在解压目录根执行（脚本自己找到包根，不依赖当前目录）：
    python proj/src/report/verify_kit.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = ROOT / "MANIFEST.txt"
    if not manifest.exists():
        print(f"{manifest} 不存在：请在复现包根目录执行", flush=True)
        return 2

    checked = missing = bad = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, size, rel = line.split(None, 2)
        path = ROOT / rel
        if not path.exists():
            print(f"[missing] {rel}", flush=True)
            missing += 1
            continue
        if path.stat().st_size != int(size) or sha256(path) != digest:
            print(f"[mismatch] {rel}", flush=True)
            bad += 1
        checked += 1

    print(f"checked {checked}  mismatch {bad}  missing {missing}", flush=True)
    return 0 if bad == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
