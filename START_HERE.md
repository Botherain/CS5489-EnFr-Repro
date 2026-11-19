# CS5489 复现包

En→Fr 翻译项目（G1–G5 阶梯、CometKiwi 路由、因子归因）。这个包里是全部实验代码、
存档的逐句分数、报告和文档。跑通分两条路：

- **只核对结果**（约 2 分钟，不需要 GPU）：直接从存档分数重算指标与全部表格。
- **从头复现**（约 10 GPU 小时）：重建语料、索引、术语表、SFT、DPO、推理、G5、COMET 打分。

## 1. 解压与环境

解压后 `proj/` 与 `Dataset/` 是同级兄弟，代码按这个相对位置找文件。先验完整性（逐文件核 sha256）：

```bash
python proj/src/report/verify_kit.py
```

两个环境都建在包根目录（脚本按 `<包根>/.venv` 找它们）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r proj/requirements-main.txt

python -m venv .venv-comet      # 用 Python 3.13 建
.venv-comet/Scripts/python -m pip install -r proj/requirements-comet.txt
```

Linux 把 `.venv/Scripts/python` 换成 `.venv/bin/python`。

打分要单独解释器，因为 `unbabel-comet 2.2.7` 依赖 `functools._HashedSeq`，Python 3.14 已删除。
没有 3.13 就先只跑第 2 节，到 E1 打分单元再补。

## 2. 先核对结果（2 分钟）

```bash
cd proj
export COMET_PY=$(cd .. && pwd)/.venv-comet/Scripts/python.exe
../.venv/Scripts/python src/report/run_notebook.py     # Linux: ../.venv/bin/python
```

PowerShell 里 `COMET_PY` 写成绝对路径即可；解释器同样用 `..\.venv\Scripts\python.exe`。

终端逐单元打印，末尾给阶段状态表与表 3 的宏平均，断言的数字见第 5 节。
`run_notebook.py` 直接跟 kernel 通信，输出边跑边显示；`jupyter nbconvert --execute` 也能跑，
但输出攒到结束才落盘。

三个语料（`archive.zip`、`MTNT.1.1.tar.gz`、`en-fr.txt.zip`）随包附带，S1 检查应该全绿；
Cell S0.3 在它们存在时只做 sha256 复核，不会重新下载。

## 3. 再从头复现（约 10 GPU 小时）

```bash
python src/report/run_notebook.py --setup --all
```

`--setup` 做三件事：建两个环境并装依赖（S0.1）、下载约 8 GB 模型权重（S0.2）、
下载并校 sha256 的语料（S0.3）。之后 `--all` 按依赖顺序跑完全部阶段：

```
D1 语料 -> D2 索引 -> D3/D4 术语表 -> T1 SFT -> T2 偏好对 -> T3 候选打分
-> T4 DPO -> R1 全矩阵（G4/G5/五个留一）-> I1/I2 推理 -> E1 COMET -> E2–E5 指标/统计/表格
```

每个阶段打印横幅、每分钟心跳和耗时。产物已存在就跳过，中断后重跑同一命令即可续跑；
`--force` 才会重跑。

## 4. 包里有什么、没有什么

有：`proj/src` 里跑实验的代码（`data/` 语料、索引、术语表、增强；`train/` SFT、偏好对、
候选打分、DPO；`infer/` 生成、G5、矩阵；`eval/` COMET 打分、指标、bootstrap、路由、表格、上限；
`common/download_assets.py`），外加 `report/run_notebook.py` 这个终端驱动；
报告 notebook 与 PDF/tex、`proj/reports` 的结果 JSON 与五张表、
`proj/runs` 的 `pass1.jsonl` / `scores.jsonl` / `scores_draft.jsonl` / `meta.json`、
`proj/data/processed`（逐字节等于训练输入）与 manifests、`Dataset/archive.zip`（课程模板集）、
三份项目文档、`MANIFEST.txt`（每个文件的 sha256）。

`data/processed` 在包里，所以 T1 以后可以直接跑，不必先跑 D1 重建语料。

包里不含我这边的工作代码：报告排版脚本（`build_report.py`、`build_notebook_inline.py`、
`collect_numbers.py`、`make_requirements.py`）、打包工具（`make_kit.py`）、冒烟测试
（`smoke_train.py`）与硬件探针（`common/probe_*.py`）。实验本身不依赖它们。
报告附录 A 里 "Tables for this report" 指向的 `build_report.py` 同属排版脚本，表格由
Cell E5（`make_tables.py`）从 `runs/*/scores.jsonl` 重新生成。

没有，以及怎么补：

| 缺的东西 | 大小 | 怎么来 |
| --- | --- | --- |
| 模型权重 Qwen2.5-1.5B / bge-m3 / COMET-DA | ~8 GB | `--setup=assets` 或 `python src/common/download_assets.py` |
| `data/index`（FAISS + BM25） | 321 MB | Cell D2 重建，约 0.7 分钟 |
| `runs/*/adapter`、`checkpoint-*` | 1.5 GB | Cell T1/T4 重训 |
| 日志 `logs/*.log` | 2 MB | 跑的时候自然生成 |

`wmt22-cometkiwi-da` 是 gated 仓库：申请许可后设 `HF_TOKEN` 再跑 S0.2，或者沿用包里的
本地 QE checkpoint（`runs/qe_local` 没在包里，路由需要时用 `src/eval/prepare_qe.py` 从镜像重建）。

## 5. 核对数字

test 3,271 句（MTNT 1,020 + EMEA 2,000 + clean 251），两域等权宏平均 COMET-DA：

| 系统 | 宏平均 |
| --- | --- |
| G1 零样本 | 0.7471 |
| G2 零样本 + RAG | 0.7633 |
| G3 LoRA SFT | 0.7981 |
| G5* SFT + DPO | 0.8015 |
| G4 SFT + RAG | 0.7827 |
| G5 四模块管线 | 0.7552 |

`S1` 环境检查单元会打印解释器、GPU、三个语料文件与缓存大小；这些和结果一起判定环境是否一致。

## 6. 常见问题

- **`comet` 装不上**：主环境必须是 3.14 之外的版本才装 3.13 的 comet；内核里没有 comet 时，
  设 `COMET_PY` 指到 `.venv-comet`，打分单元会流式调用它。
- **CUDA 报 "no kernel image is available"**：torch 版本低于 cu128，`sm_120` 跑不了。
- **没有 GPU**：只跑第 2 节那条命令，它不建 CUDA 上下文。
- **内核找不到**：`python -m ipykernel install --user --name my_local_env`，或 `--kernel python3`。
