# CS5489-Project-Think-v2 工程目录

En→Fr 机器翻译：G1–G5 阶梯对照 + CometKiwi 动态路由 + 因子归因，全部在单张 RTX 5080（16GB）上跑完。

## 文档

| 文件 | 内容 |
| --- | --- |
| `../EXPERIMENT_PLAN.md` | 设计方案 plan-v2：分组、配置、归因方法、闸门、排期 |
| `../FEASIBILITY_REVIEW.md` | 可行性与完善性检验：实测数据、审阅判定、被推翻的参数 |
| `../EXECUTION_CHECKLIST.md` | 执行清单：每步命令、资源、耗时、验收标准、失败处置、bug 附录 |
| `reports/report/CS5489_Project_Report.ipynb` | 项目报告：正文 + **全部实验代码内联**（15 个阶段 + 4 个驱动单元，79 个单元），可直接执行复现 |
| `reports/report/CS5489_Project_Report.pdf` | 同一份报告的 PDF（XeLaTeX，CityU 版式，14 页） |
| `reports/tables/all.md` | 从 JSON 产物生成的五张结果表 |

## 一条命令复现（recommended）

报告 notebook 自带阶段调度器：每个阶段打印开始横幅、每分钟心跳和耗时，产物已存在就跳过。
安装也做成了单元（S0.1 建两个环境，S0.2 拉模型权重，S0.3 下数据集），默认关闭，按需打开。

```bash
# 只重算指标与全部表格（约 2 分钟，不占 GPU）
python src/report/run_notebook.py

# 先装环境、权重与数据（Cell S0.1–S0.3），再跑完整实验（约 10 GPU 小时）
python src/report/run_notebook.py --setup
python src/report/run_notebook.py --all

# 也可以只装数据；--only 选阶段，--force 忽略已有产物重跑
python src/report/run_notebook.py --setup=data
python src/report/run_notebook.py --only data,sft
python src/report/run_notebook.py --only index --force
```

- S0.1 建 `.venv` / `.venv-comet` 并装 `requirements-*.txt`：`PY_MAIN`、`PY_COMET` 指定基础解释器（comet 要 3.13）；内核本身已满足依赖时直接跳过。
- S0.2 把 Qwen2.5-1.5B、bge-m3、COMET-DA 拉进 `HF_HOME`（默认 `<repo>/hf`）。`wmt22-cometkiwi-da` 是 gated 仓库，需要 `HF_TOKEN` 或本地重建。
- S0.3 下载 MTNT 三个官方 tsv（HF 镜像，逐字节同官方包）与 OPUS EMEA v3，两者都校验 sha256；`archive.zip` 由课程提供，只做校验。
- 打分阶段需要 `unbabel-comet`：内核里没有就设 `COMET_PY` 指向 `.venv-comet` 的解释器，单元会流式调用它。
- 终端驱动的内核名默认 `my_local_env`，可用 `--kernel` 覆盖；缺内核时按提示 `python -m ipykernel install --user --name my_local_env`。
- `jupyter nbconvert --execute` 也能跑，但输出攒到结束才落盘，看不到实时进度。
- 用 Jupyter / VS Code 打开时，把 §0 的 `RUN[...]`、`SETUP[...]` 打开即可逐段执行，语义与上面一致。

## 两个 Python 环境

本机路径（换机器按 `proj/requirements-*.txt` 重建，见上）：

```
D:\HKStudy\.venv         Python 3.14.5   torch 2.11.0+cu128 / transformers 5.17.0 / trl 1.18.0
                         训练、推理、检索、BLEU/chrF
D:\HKStudy\.venv-comet   Python 3.13.13  torch 2.11.0+cu128 / unbabel-comet 2.2.7
                         COMET-DA 与 CometKiwi 打分（comet 不兼容 Python 3.14）
```

`HF_HOME` 默认 `<repo>/hf`（脚本与 notebook 都按文件位置推断），要复用已有缓存再自己设。
可移植依赖清单是 `requirements-main.txt` / `requirements-comet.txt`，
由 `src/report/make_requirements.py` 从 `logs/requirements-*.lock` 生成。

## 目录

- `configs/` — base + g3_sft / g5_dpo / g5_pipeline / eval / ablations
- `data/` — processed（构建好的 jsonl + 术语表）、manifests（sha256）、index（FAISS + BM25 + 检索缓存）
- `src/` — common（探针）、data（构建与增强）、train（SFT/DPO/采样）、infer（生成与管线）、eval（指标/统计/路由）
- `runs/` — 每个 run 一个目录：pass1.jsonl / scores.jsonl / meta.json / routing.json
- `reports/` — tables、ceiling、adjudication、图表

## 常用命令

```powershell
$env:HF_HOME='D:\HKStudy\hf'
$py  = 'D:\HKStudy\.venv\Scripts\python.exe'
$cmt = 'D:\HKStudy\.venv-comet\Scripts\python.exe'

# 数据与索引
& $py src\data\build_dataset.py
& $py src\data\build_index.py --pool seen
& $py src\data\build_index.py --pool unseen

# 训练
& $py src\train\sft_lora.py --run-id g3_sft_r16_s42 --epochs 2 --batch 16
& $py src\train\mine_dpo_pairs.py --stage generate
& $cmt src\train\score_candidates.py
& $py src\train\mine_dpo_pairs.py --stage pair
& $py src\train\dpo_lora.py --epochs 2 --batch 8 --accum 2

# 推理与打分
& $py src\infer\generate.py --run-id g3_sft --mode sft --adapter runs\g3_sft_r16_s42\adapter --set test
& $cmt src\eval\score_comet.py --runs g3_sft --batch-size 48

# 全矩阵（G4 / G5 / 消融，已完成的自动跳过）
& $py src\infer\run_matrix.py --group all

# 统计与表格
& $py src\eval\metrics.py --all
& $py src\eval\bootstrap.py --family g5
& $py src\eval\make_tables.py
```

## 打包给队员

```bash
python src/report/make_kit.py          # -> ../dist/CS5489-Repro-Kit.zip（约 52 MB）
python src/report/make_kit.py --keep   # 同时保留解压后的目录
```

包里只放实验代码：`src` 下的 data/train/infer/eval、`common/download_assets.py`、`report/run_notebook.py`
与 `report/verify_kit.py`；再加 `reports` 的产物、`runs` 的 jsonl 与 meta、`data/processed`、
`Dataset/archive.zip`、三份文档、`START_HERE.md` 与逐文件 sha256 的 `MANIFEST.txt`。
报告排版脚本（`build_report.py`、`build_notebook_inline.py`、`collect_numbers.py`、
`make_requirements.py`）、打包工具 `make_kit.py`、冒烟测试 `smoke_train.py` 与硬件探针 `probe_*.py`
不进包。解压后用 `python proj/src/report/verify_kit.py` 校完整性。

## 术语表流程

```powershell
& $py src\data\build_glossary.py --mode align --top 600
& $py src\data\build_glossary.py --mode slang --top 200
& $py src\data\make_review_page.py                 # reports\glossary_review.html，可离线裁决
& $py src\data\make_adjudication_packs.py          # 交给外部模型裁决的任务包
& $py src\data\merge_verdicts.py                   # 合并裁决 -> reports\reviewed.json
& $py src\data\finalize_glossary.py --reviewed reports\reviewed.json
```

`finalize_glossary.py` 补上 `fr_ids`、`accepted` 变体集与 `src_sig`/`fr_sig`，并算 test 覆盖率。

## 结果快照

test 3,271 句（MTNT 1,020 + EMEA 2,000 + clean 251），两域等权宏平均 COMET-DA：

| 系统 | 宏平均 |
| --- | --- |
| G1 零样本 | 0.7471 |
| G2 零样本 + RAG | 0.7633 |
| G3 LoRA SFT | 0.7981 |
| G5* SFT + DPO | 0.8015 |
| G5 四模块管线 | 0.7553 |

天花板参照：强模型零样本在 MTNT 子集 0.8066、EMEA 子集 0.8223（G3 分别为 0.7720 / 0.8127）。
DPO adapter 必须先合并 SFT 再加载，否则会退到零样本水平，详见 `../EXECUTION_CHECKLIST.md` 的 bug 附录 B2。
