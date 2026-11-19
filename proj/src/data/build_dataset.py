"""D0 数据流水线：三个原始资产 -> 统一 schema 的 jsonl + manifest。

输入（只读，不解压）：
    Dataset/archive.zip        通用干净集（实测唯一句对 2,251）
    Dataset/MTNT.1.1.tar.gz    MTNT en-fr（train 36,058 / valid 852 / test 1,020）
    Dataset/en-fr.txt.zip      OPUS EMEA v3（1,092,568 行）
    （也可用 Dataset/MTNT/{train,valid,test}/*.en-fr.tsv，见 notebook 的下载单元）

输出：
    data/processed/<domain>.<split>.jsonl
    data/manifests/<domain>.<split>.json   (行数 + sha256 + 过滤计数)
    reports/data_stats.json

设计要点：
    1. 全程确定性：按 sha1(src) 排序或按固定 seed 洗牌，重跑结果逐字节一致。
    2. EMEA 用"连续 100 行成块"的文档块切分，防止 PDF 相邻行跨集泄漏。
    3. MTNT 保留原始噪声，只做编码修复与长度过滤，不做拼写/大小写归一。
    4. clean 集去重后只作在域 sanity，test 251 条。
    5. memory_pool 专供 G2/G4 的 seen-vs-unseen 检索对照，SFT 不得触碰。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[2]          # proj/
DATASET = ROOT.parent / "Dataset"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"
REPORTS = ROOT / "reports"

SEED = 42
BLOCK = 100                                          # EMEA 文档块大小

EMEA_PLAN = {"test": 20, "dev": 10, "memory_pool": 60, "train": 270}   # 块数
CLEAN_PLAN = {"train": 1800, "dev": 200, "test": 251}
MTNT_UNSEEN_POOL = 3000                              # 从 MTNT train 留出，不进 SFT

PUNCT_ONLY = re.compile(r"^[\W\d_\.]*$")
WS = re.compile(r"\s+")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    """只做空白与不可见字符修复，保留大小写与拼写噪声。"""
    return WS.sub(" ", text.replace("\u00a0", " ").replace("\t", " ")).strip()


def wc(text: str) -> int:
    return len(text.split())


NON_ALNUM = re.compile(r"[^a-z0-9à-ÿ' ]+")


def canonical(text: str) -> list[str]:
    """小写 + 去标点，返回词序列，用于近重复检测（不影响落盘原文）。"""
    return NON_ALNUM.sub(" ", text.lower()).split()


class GramIndex:
    """8-gram 指纹索引（int64 摘要），用于近重复过滤与跨集泄漏检测。

    281k 句 x ~15 gram 约 4M 个 int，占用 300-500MB 内存，32GB 机器可承受。
    """

    def __init__(self, n: int = 8):
        self.n = n
        self.seen: set[int] = set()

    def grams(self, text: str) -> set[int]:
        toks = canonical(text)
        if len(toks) < self.n:
            toks = toks + ["<pad>"] * (self.n - len(toks))
        return {int.from_bytes(hashlib.blake2b(" ".join(toks[i:i + self.n]).encode(),
                                               digest_size=8).digest(), "big")
                for i in range(len(toks) - self.n + 1)}

    def is_dup(self, text: str) -> bool:
        g = self.grams(text)
        return bool(g & self.seen)

    def add(self, text: str) -> None:
        self.seen |= self.grams(text)

    def add_many(self, texts) -> None:
        for t in texts:
            self.seen |= self.grams(t)


# clean 集（archive.zip）

def build_clean(stats: dict) -> None:
    zf = zipfile.ZipFile(DATASET / "archive.zip")
    seen: dict[str, dict] = {}
    dropped_dup = 0
    total = 0
    for name in ("train.csv", "validation.csv", "test.csv"):
        with zf.open(name) as fh:
            for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")):
                total += 1
                src, tgt = norm(row["english"]), norm(row["french"])
                if not src or not tgt:
                    continue
                key = (src.lower(), tgt.lower())
                if key in seen:
                    dropped_dup += 1
                    continue
                seen[key] = {
                    "src": src, "tgt": tgt, "domain": "clean",
                    "category": row.get("category", ""), "difficulty": row.get("difficulty", ""),
                }
    stats["clean"] = {"rows_read": total, "unique_pairs": len(seen), "dropped_duplicates": dropped_dup}

    items = sorted(seen.values(), key=lambda d: sha1(d["src"]))
    rng = random.Random(SEED)
    rng.shuffle(items)
    cursor = 0
    for split, n in CLEAN_PLAN.items():
        chunk = items[cursor:cursor + n]
        cursor += n
        for i, d in enumerate(chunk):
            d["id"] = f"clean_{split}_{i:05d}"
            d["split"] = split
            d["src_words"] = wc(d["src"])
        write_jsonl("clean", split, chunk)
    stats["clean"]["splits"] = CLEAN_PLAN


# MTNT

def read_mtnt(member: str) -> list[dict]:
    tar = DATASET / "MTNT.1.1.tar.gz"
    if tar.exists():
        with tarfile.open(tar, "r:gz") as tf:
            raw = tf.extractfile(member).read().decode("utf-8", errors="replace")
    else:
        # Notebook 的下载单元把官方 tsv 直接放到 Dataset/MTNT/<split>/ 下
        raw = (DATASET / member).read_text(encoding="utf-8", errors="replace")
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        out.append({"comment_id": parts[0], "src": norm(parts[1]), "tgt": norm(parts[2])})
    return out


def mtnt_ok(d: dict) -> bool:
    n = wc(d["src"])
    return bool(d["src"]) and bool(d["tgt"]) and 1 <= n <= 400


def build_mtnt(stats: dict) -> None:
    mapping = {"train": "MTNT/train/train.en-fr.tsv",
               "dev": "MTNT/valid/valid.en-fr.tsv",
               "test": "MTNT/test/test.en-fr.tsv"}
    stats["mtnt"] = {}
    pools: dict[str, list[dict]] = {}
    for split, member in mapping.items():
        rows = read_mtnt(member)
        kept = [d for d in rows if mtnt_ok(d)]
        for i, d in enumerate(kept):
            d.update(id=f"mtnt_{split}_{i:05d}", domain="mtnt", split=split, src_words=wc(d["src"]))
        pools[split] = kept
        stats["mtnt"][split] = {"raw": len(rows), "kept": len(kept), "dropped": len(rows) - len(kept)}

    # train 侧去污染：任何与 dev/test 源句共享 8-gram 的训练样本都剔除。
    # dev/test 保持官方切分不动，便于与已发表结果对照。
    eval_grams = GramIndex(8)
    eval_grams.add_many([d["src"] for d in pools["dev"]] + [d["src"] for d in pools["test"]])
    before = len(pools["train"])
    pools["train"] = [d for d in pools["train"] if not eval_grams.is_dup(d["src"])]
    stats["mtnt"]["train_dropped_for_eval_overlap"] = before - len(pools["train"])
    stats["mtnt"]["train_after_decontam"] = len(pools["train"])
    print(f"  MTNT train 去污染: {before} -> {len(pools['train'])}", flush=True)

    # train 里留出 unseen 检索池：按 comment_id 分块抽，避免同帖跨池
    train = pools["train"]
    by_comment: dict[str, list[dict]] = {}
    for d in train:
        by_comment.setdefault(d["comment_id"], []).append(d)
    cids = sorted(by_comment)
    rng = random.Random(SEED)
    rng.shuffle(cids)
    unseen, sft_train = [], []
    for cid in cids:
        if len(unseen) < MTNT_UNSEEN_POOL:
            unseen.extend(by_comment[cid])
        else:
            sft_train.extend(by_comment[cid])
    stats["mtnt"]["memory_pool_unseen"] = len(unseen)
    stats["mtnt"]["train_for_sft"] = len(sft_train)

    write_jsonl("mtnt", "train", sft_train)
    write_jsonl("mtnt", "memory_pool_unseen", unseen)
    write_jsonl("mtnt", "dev", pools["dev"])
    write_jsonl("mtnt", "test", pools["test"])


# EMEA

def emea_keep(src: str, tgt: str) -> bool:
    if not src or not tgt or src == tgt:
        return False
    ls, lt = wc(src), wc(tgt)
    if not (4 <= ls <= 60) or lt < 3:
        return False
    if not 0.5 <= (lt / ls) <= 2.2:
        return False
    if PUNCT_ONLY.match(src) or PUNCT_ONLY.match(tgt):
        return False
    letters = sum(c.isalpha() for c in src)
    return letters >= max(8, int(0.5 * len(src)))


def build_emea(stats: dict) -> None:
    zf = zipfile.ZipFile(DATASET / "en-fr.txt.zip")
    en = zf.open("EMEA.en-fr.en").read().decode("utf-8", errors="replace").splitlines()
    fr = zf.open("EMEA.en-fr.fr").read().decode("utf-8", errors="replace").splitlines()
    assert len(en) == len(fr), "EMEA 行不对齐"

    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    dropped = Counter()
    for a, b in zip(en, fr):
        src, tgt = norm(a), norm(b)
        if not src or not tgt:
            dropped["empty"] += 1
            continue
        if not emea_keep(src, tgt):
            dropped["filter"] += 1
            continue
        key = (src.lower(), tgt.lower())
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"src": src, "tgt": tgt})

    stats["emea"] = {"lines": len(en), "kept_unique": len(kept), "dropped": dict(dropped)}

    # 近重复过滤：EMEA 来自 PDF，跨文档的样板句（免责声明、章节标题）大量重复。
    # 只做精确去重时，test 与 train 的 8-gram 重叠率高达 26.6%（实测），
    # 因此这里对全量候选做一次 8-gram 指纹过滤，再切分。
    idx = GramIndex(8)
    dedup: list[dict] = []
    near_dup = 0
    for d in kept:
        if idx.is_dup(d["src"]):
            near_dup += 1
            continue
        idx.add(d["src"])
        dedup.append(d)
    stats["emea"]["near_dup_dropped"] = near_dup
    stats["emea"]["after_near_dup"] = len(dedup)
    print(f"  EMEA 近重复过滤: {len(kept)} -> {len(dedup)} (drop {near_dup})", flush=True)
    kept = dedup

    rng = random.Random(SEED + 1)
    n_blocks = len(kept) // BLOCK
    block_ids = list(range(n_blocks))
    rng.shuffle(block_ids)
    cursor = 0
    for split, n_blocks_split in EMEA_PLAN.items():
        take = block_ids[cursor:cursor + n_blocks_split]
        cursor += n_blocks_split
        rows = []
        for bi in sorted(take):
            rows.extend(kept[bi * BLOCK:(bi + 1) * BLOCK])
        for i, d in enumerate(rows):
            d.update(id=f"emea_{split}_{i:05d}", domain="emea", split=split, src_words=wc(d["src"]))
        write_jsonl("emea", split, rows)
        stats["emea"][split] = len(rows)



def write_jsonl(domain: str, split: str, rows: list[dict]) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{domain}.{split}.jsonl"
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for d in rows:
            line = json.dumps(d, ensure_ascii=False, sort_keys=True)
            fh.write(line + "\n")
            h.update(line.encode("utf-8"))
    manifest = {"file": str(path.relative_to(ROOT)), "rows": len(rows), "sha256": h.hexdigest(),
                "domain": domain, "split": split, "seed": SEED}
    (MANIFESTS / f"{domain}.{split}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {path.name}: {len(rows)} rows  sha256={h.hexdigest()[:12]}", flush=True)
    return path


def load_jsonl(domain: str, split: str) -> list[dict]:
    path = PROCESSED / f"{domain}.{split}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def ngrams(text: str, n: int = 8) -> set[str]:
    toks = text.lower().split()
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def leakage_check(stats: dict) -> None:
    """每个域的 test/dev 与 train+memory_pool 做句级 hash 与 8-gram 重叠检查。"""
    leaks = {}
    for domain in ("clean", "mtnt", "emea"):
        train_src = set()
        gram = set()
        for split in ("train", "memory_pool", "memory_pool_unseen"):
            path = PROCESSED / f"{domain}.{split}.jsonl"
            if not path.exists():
                continue
            for d in load_jsonl(domain, split):
                train_src.add(d["src"].strip().lower())
                gram |= ngrams(d["src"])
        for split in ("dev", "test"):
            path = PROCESSED / f"{domain}.{split}.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(domain, split)
            exact = sum(1 for d in rows if d["src"].strip().lower() in train_src)
            g = sum(1 for d in rows if ngrams(d["src"]) & gram)
            leaks[f"{domain}.{split}"] = {
                "rows": len(rows), "exact_src_overlap": exact, "8gram_overlap": g,
                "8gram_rate": round(g / max(1, len(rows)), 4),
            }
            print(f"  leakage {domain}.{split}: exact={exact} 8gram={g}", flush=True)
    stats["leakage"] = leaks


def length_stats(stats: dict) -> None:
    out = {}
    # 只统计本脚本产出的三个域；processed 目录里还有后续步骤生成的文件（chat、DPO 对、增强集）
    produced = [p for p in sorted(PROCESSED.glob("*.jsonl"))
                if p.name.split(".")[0] in {"clean", "mtnt", "emea"}]
    for path in produced:
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        ws = sorted(d["src_words"] for d in rows)
        wt = sorted(wc(d["tgt"]) for d in rows)
        if not ws:
            continue
        q = lambda v, p: v[min(len(v) - 1, int(len(v) * p / 100))]
        out[path.stem] = {
            "rows": len(rows),
            "src_words": {"p50": q(ws, 50), "p90": q(ws, 90), "p95": q(ws, 95),
                          "p99": q(ws, 99), "max": ws[-1]},
            "tgt_words": {"p50": q(wt, 50), "p95": q(wt, 95), "max": wt[-1]},
            "est_src_tokens_p99": int(q(ws, 99) * 1.35),
        }
    stats["length"] = out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-emea", action="store_true")
    args = ap.parse_args()

    stats: dict = {"seed": SEED, "emea_block": BLOCK}
    print("[1/5] clean (archive.zip)", flush=True)
    build_clean(stats)
    print("[2/5] MTNT", flush=True)
    build_mtnt(stats)
    if not args.skip_emea:
        print("[3/5] EMEA (1.09M 行，约 2-3 分钟)", flush=True)
        build_emea(stats)
    print("[4/5] 泄漏检查", flush=True)
    leakage_check(stats)
    print("[5/5] 长度统计", flush=True)
    length_stats(stats)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDONE -> reports/data_stats.json", flush=True)


if __name__ == "__main__":
    main()
