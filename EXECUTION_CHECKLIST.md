# 执行清单（Runbook）— CS5489-Project-Think-v2 @ RTX 5080 16GB

> 配套文件：设计见 `EXPERIMENT_PLAN.md`（plan-v2），检验证据见 `FEASIBILITY_REVIEW.md`，配置见 `proj/configs/*.yaml`
> 所有耗时/显存数字来源：**实测**（`proj/logs/`）或按实测吞吐折算，标 `估` 的才是估算
> 工作目录：`D:\HKStudy\TPg-MSAI\Classes\CS5489 - Machine Learning\4.Projects\proj`

## 怎么读这份清单

每一步是一张卡，字段固定：

- **目的**：这步产出什么结论或资产
- **命令**：可直接复制执行
- **资源**：显存峰值 / 内存 / 磁盘 / 网络
- **耗时**：机器时间（GPU 或 CPU）与人日（含实现与调试）
- **验收**：可判定的通过条件
- **失败处置**：不通过时怎么办

状态标记：`完成，已完成`（有实测证据）｜`待跑 脚本就绪待跑`｜`待做 需先实现`

**复现入口**：不必逐条照抄。`python proj/src/report/run_notebook.py --all` 会按依赖顺序把下面的阶段跑一遍，
实时打印每个阶段的横幅、每分钟心跳与耗时，产物已存在的自动跳过。只重算指标与表格用同一命令不加参数，
装环境/权重/数据加 `--setup`。

---

## 总览

| 阶段 | 步骤 | 机器时间 | 人日 | 状态 |
| --- | --- | --- | --- | --- |
| S0 环境 | S0.1–S0.7 | 0.7h | 0.6 | 完成 **全部完成** |
| S1 数据 | S1.1–S1.4 | 0.2h | 0.8 | 完成 S1.1/S1.2/S1.4 完成；待做 S1.3 术语表待做 |
| S2 检索与基线 | S2.1–S2.3 | 1.0h | 0.7 | 完成 S2.1 脚本就绪并冒烟；待跑 S2.2/S2.3 待正式跑 |
| S3 SFT | S3.1–S3.2 | 2.5h | 0.5 | 待跑 冒烟通过，正式训练待跑 |
| S4 DPO | S4.1–S4.3 | 2.7h | 0.8 | 待做 挖掘脚本待写（DPO 训练路径已冒烟） |
| S5 G5 管线 | S5.1–S5.4 | 0.5h | 1.5 | 待做 检索/RRF 已在 generate.py 内实现，其余待写 |
| S6 主表 | S6.1–S6.2 | 0.6h | 0.3 | 完成 S6.2 打分脚本就绪；待跑 S6.1 待正式跑 |
| S7 消融 | S7.1–S7.3 | 2.7h | 0.5 | 待做 |
| S8 Shapley | S8.1 | 1.3h | 0.3 | 待做 |
| S9 打分与统计 | S9.1–S9.3 | 1.0h | 1.0 | 待做 |
| S10 分析 | S10.1–S10.3 | 0.2h | 1.0 | 待做 |
| S11 报告 | S11.1–S11.2 | 0 | 1.5 | 待做 |
| | **合计** | **≈13.4h 机器** | **≈9.5 人日** | **已完成 ≈1.4 人日 / 1.1h 机器** |

**当前可直接执行的下一步**：S1.3 术语表 → S2.2/S2.3 基线与 k 扫描 → S3.1 SFT 正式训练（2.3h，可挂机）。

**MVP 路线（3.8 人日，13h 机器时间）**：S0 → S1 → S2 → S3 → S4（挖掘量减半）→ S5 → S6 → S7（只做 4 个留一，跳过 `no_rag_len` 与 `random_pairs`）→ S9 → S10.1 → S11。
**加项顺序**（时间富余时按此加）：S8 精确 Shapley → `dpo_random_pairs` → 长度分桶 → LLM 裁判 → 第二 seed → beam4 敏感性。

---

## S0 环境（0.6h 机器 / 0.5 人日）

### S0.1 依赖安装　完成 - **命令**：见 `FEASIBILITY_REVIEW.md` §1.1（已执行）
- **资源**：网络 ~200MB，磁盘 ~300MB｜**耗时**：6 分钟
- **实测结果**：`peft 0.21.3 / trl 1.18.0 / datasets 5.2.0 / accelerate 1.15.0 / sentence-transformers 6.3.0 / faiss-cpu 1.16.0 / sacrebleu 2.6.0`；`torch 2.11.0+cu128` 与 `transformers 5.17.0` **未被改动**
- **验收**：`pip check` 无冲突；`torch.cuda.is_available()` 为 True

### S0.2 API 探针　完成 - **命令**：`python src\common\probe_api.py`
- **耗时**：30 秒｜**产出**：TRL/transformers 真实字段名清单
- **验收（已达成）**：确认 `max_length`（无 `max_seq_length`）、`assistant_only_loss`、`precompute_ref_log_probs` 存在；`warmup_ratio`、`DataCollatorForCompletionOnlyLM`、`max_prompt_length` **不存在**

### S0.3 显存/吞吐探针　完成 - **命令**：`python src\common\probe_gpu.py` ＋ `python src\common\probe_train_mem.py`
- **资源**：显存峰值 13.5GB（naive 全序列 CE 路径）｜**耗时**：8 分钟
- **关键产出**：naive 路径 b4×512 OOM；真实 TRL 路径 b16×L384 最坏 9.02GB reserved；生成 batch32 = 22 句/s；实测显存带宽 771GB/s
- **验收**：`logs/probe_train_mem.json` 与 `logs/smoke_env.log` 存在且含 peak 字段

### S0.4 训练链路冒烟　完成 - **命令**：`python src\train\smoke_train.py --mode sft --batch 16 --max-length 384 --steps 8 --no-packing`（dpo 同理）
- **资源**：显存 9.02GB reserved｜**耗时**：单次 15 秒
- **验收（已达成）**：SFT 与 DPO 两条链路均跑通并保存 adapter；DPO 的 `precompute_ref_log_probs` 实测 36 对/s

### S0.5 模型资产下载　完成（含 QE 镜像）

- **命令**：`$env:HF_HOME='D:\HKStudy\hf'; python src\common\download_assets.py`
- **资源**：网络 ~8GB，磁盘 23.2GB（已占）｜**耗时**：3 分钟
- **状态**：Qwen2.5-1.5B-Instruct 完成 / bge-m3 完成 / wmt22-comet-da 完成 / wmt22-cometkiwi-da 注意 **gated 401**，改用镜像 `Supervache/wmt22-cometkiwi-da-safeformat` 完成（转本地 ckpt，见 S0.7）
- **验收**：四个模型均可加载；QE 三句验证分数排序正确
- **升级路径**：拿到 HF token 后 `hf auth login` → 重跑 `download_assets.py --only kiwi`，并在 200 句上比对镜像与官方分数（预期 Pearson>0.99），结果写进附录

### S0.6 COMET 独立环境　完成 - **命令（实测可用版本，与 plan-v1 写的不同）**：
  ```powershell
  & "C:\Users\123\.local\bin\uv.exe" venv D:\HKStudy\.venv-comet --python 3.13 --seed
  & "D:\HKStudy\.venv-comet\Scripts\python.exe" -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
  & "D:\HKStudy\.venv-comet\Scripts\python.exe" -m pip install "setuptools<81" unbabel-comet
  ```
- **资源**：磁盘 ~6GB｜**耗时**：6 分钟
- **两个必须知道的坑**：①主 venv 是 Python **3.14**，而 `unbabel-comet 2.2.7` 依赖 `functools._HashedSeq`（3.14 已移除）→ **必须用 3.13 建这个 venv**；②`torchmetrics 0.10.3` 需要 `pkg_resources` → 必须 `setuptools<81`；③torch 要先从 cu128 索引装，否则 PyPI 给的是 Windows CPU 版
- **验收（已达成）**：Python 3.13.13 + torch 2.11.0+cu128（cuda=True）+ comet 2.2.7 全部 import 成功；48 句打分 2.7s

### S0.7 QE 镜像转本地 ckpt　完成 - **命令**：`& "D:\HKStudy\.venv-comet\Scripts\python.exe" src\eval\prepare_qe.py`
- **原因**：comet 只认 `<dir>/checkpoints/model.ckpt` + `<dir>/hparams.yaml` 的 Lightning 布局，镜像给的是 safetensors，直接加载报 `Invalid checkpoint path`
- **资源**：磁盘 +2.26GB｜**耗时**：1.5 分钟
- **产出**：`runs/qe_local/wmt22-cometkiwi-da-mirror/checkpoints/model.ckpt`
- **验收（已达成）**：加载为 `UnifiedMetric`；三句验证 好译 0.8913 > 俚语译 0.8716 > 坏译 0.6006
- **报告义务**：方法节与限制节写明 QE 用的是社区镜像（cc-by-nc-sa-4.0）转换所得，未与官方权重逐位校验

---

## S1 数据（0.4h 机器 / 0.8 人日）

### S1.1 三域数据集构建　完成 - **命令**：`python src\data\build_dataset.py`
- **资源**：内存峰值 ~2.5GB（8-gram 指纹索引），磁盘 ~60MB｜**耗时**：3.5 分钟
- **产出**：`data/processed/*.jsonl`（11 个文件）+ `data/manifests/*.json`（含 sha256）+ `reports/data_stats.json`
- **验收（已达成）**：clean 1,800/200/251；MTNT 32,593/852/1,020 + unseen 池 3,032；EMEA 27,000/1,000/2,000 + memory_pool 6,000；泄漏审计 test 侧 8-gram 重叠 ≤0.4%
- **踩过的坑（已修）**：只做精确去重时 EMEA test 与 train 的 8-gram 重叠 **26.6%**，必须加全量 8-gram 指纹去重（丢掉 45% 的跨文档样板句）

### S1.2 token 长度核定　完成 - **命令**：见 `FEASIBILITY_REVIEW.md` §1.3 内联脚本
- **耗时**：40 秒｜**产出**：各集 token 分位数
- **验收（已达成）**：`max_length=384` 时评测集零截断、MTNT train 截断 1.73%（已记录）

### S1.3 术语表与俚语表　待跑（候选已挖出 + 交互页面已生成，待人工裁决）

- **命令（四步）**：
  ```powershell
  # 1) 挖候选（已完成，重跑幂等）
  & $py src\data\build_glossary.py --mode align --top 400
  & $py src\data\build_glossary.py --mode slang --top 200
  # 2) 生成人工裁决页面（单文件 HTML，数据内嵌，离线可用）
  & $py src\data\make_review_page.py
  # 3) 浏览器打开 proj\reports\glossary_review.html，裁决完毕后导出 reviewed.json 到 proj\reports\
  # 4) 落成最终术语表（补 token ids / accepted / 覆盖率）
  & $py src\data\finalize_glossary.py --reviewed reports\reviewed.json
  ```
- **页面功能**：EMEA 286 条 + MTNT 200 条（单文件 442KB）；每条附 2–3 条语料实证句；显示 `dev x / test y` 命中句数与「词数不一致」徽章；可切「按影响排序」；键盘 `1` 保留 / `2` 改译 / `0` 剔除 / `j`·`k` 翻页；进度存 localStorage；导出 JSON
- **实测产出（候选表 v3，2026-11-15）**：
  - 27,000 训练对 → 内容词 signature en 119,772 / fr 144,473 → 进入对齐 7,253 / 9,513 → 原始候选 5,040
  - 过滤漏斗：词序或大小写相同 −800、直译雷同 −559、说明书套话 −184、专名/掩码/混排残片 −1,772、同英多译 −115、子集 −58 → **807 条可用**，取前 600
  - **可达性剪枝（`--mode prune`）**：dev 与 test 都不出现的条目对约束和命中率都无贡献，剔除 **314 条** → **286 条待裁决**（test 有命中 235 条，仅 dev 命中 51 条）。全量备份在 `glossary.emea.candidates.full.jsonl`
  - 体检（`--mode audit`）：单位 **0/0**、signature 重复 **0/0**、法语超 4 词 **0**、含斜杠 **0**、牌名式全大写 **0**
  - 覆盖率：EMEA test **24.0%**（480/2,000 句）、MTNT test **4.6%**（47/1,020 句）
  - 法语单 token 术语仅 **2 条** → logit 偏置只对极少数可用，主用法是 prompt 注入术语提示
- **v3 修掉的五类问题**：①单位 mg/ml/IU 被当词组；②词序不同导致重复；③同一法语挂多条英文；④对齐截断（`carbonic anhydrase || anhydrase`）；⑤**跨斜杠拼出的假短语**（`body weight/day` 被分词成 `body weight day`）。n-gram 现在只在 `/ ( ) , ; :` 与空格连字符切出的词段内部生成
- **LLM 裁决（已完成，代替人工逐条看）**：286 条 EMEA + 200 条 MTNT 交给 `GPT-6 Luna`（跨家族）分 5 批判定，判定依据限定为 EMEA 官方句对 + 同词其他写法，不允许凭记忆编法语。结果：EMEA **keep 184 / fix 37 / reject 65**；MTNT **keep 175 / reject 25**（其中原样保留 117、意译 54、整句改写 4）。低置信仅 **21 条**，是人工唯一需要看的清单
- **第二模型复核**：分层抽 30 条（keep/fix/reject 各 10）交 `Claude Haiku 5.5` 以"挑错"模式复核，输出 `problem ∈ {wrong_decision, bad_french, no_evidence}` 与建议译法，同意率写入 `glossary_summary.json`
- **人工工作量**：只剩低置信 21 条 + 复核不通过的那部分，约 15 分钟
- **产出（已落盘）**：`data/processed/glossary.emea.json`（**220 条**，test 覆盖 **19.9%**＝399/2,000 句、命中 510 次）、`glossary.mtnt.json`（**175 条**，test 覆盖 3.6%＝37/1,020 句）、`reports/glossary_summary.json`、`reports/reviewed.json`（裁决原始记录）
- **一个重要发现**：220 条 EMEA 术语里**法语侧只有 1 条是单 token**。logit 硬偏置基本无法落地，G5 的"约束解码"因子必须按 **prompt 注入术语提示** 实现，报告里不要声称做了全量约束解码
- **验收**：EMEA ≥200 条、MTNT ≥80 条；抽检 50 条错误率 <10%；每条带 `accepted`（命中率用）与 `fr_ids`（首 token 偏置用）
- **注意**：若浏览器对 `file://` 禁用 localStorage，在 `proj` 目录下 `python -m http.server 8000` 后访问 `http://localhost:8000/reports/glossary_review.html`
- **失败处置**：若人工后 EMEA 不足 200 条 → 把 `--min-cooccur` 降到 4 重跑候选；仍不足则保留 100 条，并把"约束解码"因子改称"术语提示（prompt 注入）"

### S1.4 检索索引构建　完成 - **命令**：`python src\data\build_index.py --pool seen` ／ `--pool unseen`
- **实测**：seen 池 **61,393** 句对，bge-m3 fp16 编码 **33.8s**，峰值显存 **3.09GB**，产出 `seen.faiss` 251.5MB + `seen.bm25.pkl` 11MB；unseen 池 **9,032** 句对，5.1s
- **耗时**：合计 1.5 分钟（远低于原估 8 分钟）
- **验收（部分达成）**：自检 5 句 —— 短句检索准确（"yea" → "yay/Ouais"，相似度 0.78）；**长噪声句偏弱**（相似度 0.59–0.62，语义偏离）。需在 S2 阶段人工核验 20 条后定论
- **失败处置**：若 20 条核验合格数 <15 → 提高 RRF 中 BM25 权重至 0.6，或改用字符级 3-gram 分词重建 BM25 索引
- **对照实验注意**：seen(61,393) 与 unseen(9,032) 规模差 6.8 倍，做 pool 对照时两边都要抽样到 **6,000**（`build_index.py --limit 6000` 另建子索引，或查询时截断）

---

## S2 检索与基线（1.0h 机器 / 0.7 人日）

### S2.1 统一推理脚本　完成（已实现并冒烟通过）

- **命令**：
  ```powershell
  python src\infer\generate.py --run-id g1_zs --mode zeroshot --set test
  python src\infer\generate.py --run-id g2_rag --mode rag --set test --k 3 --pool seen
  python src\infer\generate.py --run-id g3_sft --mode sft --adapter runs\g3_sft_r16_s42\adapter --set test
  python src\infer\generate.py --run-id g4_sft_rag --mode sft_rag --adapter runs\g3_sft_r16_s42\adapter --set test --k 3
  ```
- **已实现**：批量 32 + 左 padding + greedy／beam4 可切；chat template 统一骨架（各系统只差"模型"与"是否检索"两处）；RRF 混合检索带**磁盘缓存**；逐句写 `pass1.jsonl`；`meta.json` 自动记 git commit、峰值显存、吞吐、batch=1 逐句延迟 p50/p95
- **实测**：零样本峰值 **3.78GB**，RAG 峰值 **4.86GB**；batch=1 逐句延迟 p50 599ms；BM25 检索 0.21s/句（首次），缓存命中后为 0
- **踩过的坑（已修）**：①检索 top-3 出现同源句重复示例 → 过取 4k 后按源句去重；②BM25 每个 run 重算 → 加缓存；③`gen_tokens` 取到整批 padding 长度 → 改逐句数非 pad token
- **验收**：3,271 句输出无空行；langdetect 抽检 50 条非法语率 <2%；单系统耗时 ≤6 分钟
- **待办**：`--latency-sample` 默认 200，正式跑 test 时保持；G5 管线另有独立脚本（S5.4）

### S2.2 G1 零样本 + beam4 敏感性　待做 - **命令**：`generate.py --config configs/g1_zeroshot.yaml`（greedy）＋ `--decode beam4`
- **资源**：显存 greedy 3.3GB / beam4 约 5.5GB（32×4 序列 KV ≈2GB）｜**耗时**：greedy 5 分钟 + beam4 18 分钟
- **验收**：G1 在 EMEA 上 COMET-DA ∈ [0.72, 0.86]（Gate G2）；beam4 与 greedy 的差值记录进敏感性分析

### S2.3 G2 朴素 RAG + k 扫描　待做 - **命令**：`generate.py --config configs/g2_rag.yaml --k 1|3|5 --set dev` → 选定 k 后 `--set test`
- **资源**：显存 3.3GB(Qwen) + 2.3GB(bge-m3) = **5.6GB**｜**耗时**：dev 3 个 k 共 12 分钟；test 6 分钟
- **验收**：dev 上确定 k*（预期 3）；G2 与 G1 的 prompt 除检索示例外完全一致
- **失败处置**：k 增大单调变差 → 说明检索噪声主导，直接把这一现象写成 RAG 噪声敏感度分析（v1 文档的加分项）

---

## S3 SFT（2.5h 机器 / 0.5 人日）

### S3.1 训练脚本　待做（配置已就绪）

- **命令**：`python src\train\sft_lora.py --config configs/g3_sft.yaml`
- **配置**（`configs/g3_sft.yaml`，字段名已实测校验）：`max_length=384`、`packing=false`、`assistant_only_loss=true`、`per_device_train_batch_size=16`、`gradient_accumulation_steps=1`、`lr=2e-4` cosine、`warmup_steps=120`、`epochs=2`、`gradient_checkpointing=true`、`bf16=true`、`optim=adamw_torch`、`dataloader_num_workers=0`、`torch_empty_cache_steps=100`
- **资源**：显存 **9.02GB reserved（最坏情况实测）**｜内存 ~3GB｜磁盘 adapter 37MB｜**耗时**：**2.3h 实测折算**（62k×2 ÷ 15.3 样本/s）
- **验收（Gate G3）**：train loss 单调下降；MTNT dev 上 COMET-DA 相对 G1 提升 ≥ **+0.015** 且 bootstrap CI 下界 >0
- **失败处置**：不达标 → 依次检查 ①`assistant_only_loss` 是否真的只算 assistant 段（打印一条样本的 label mask）②学习率 ③数据混合比（EMEA 占 43% 可能压过 MTNT）；两轮不过就回退到 r=32 或 lr=3e-4

### S3.2 adapter 验证与合并　待做 - **命令**：`python src\train\merge_adapter.py --adapter runs/g3_sft_r16_s42/adapter --out runs/g3_sft_r16_s42/merged`
- **资源**：显存 6.5GB（加载 + 合并 + 保存）｜**耗时**：3 分钟
- **验收**：合并后模型与"基座+adapter"在 100 句上输出完全一致（greedy）
- **说明**：G4/G5 用合并版还是动态挂载 adapter 都行，但**全项目必须统一一种**，否则数字不可比

---

## S4 DPO（2.7h 机器 / 0.8 人日）

### S4.1 候选生成与打分　待做 - **命令**：`python src\train\mine_dpo_pairs.py --stage generate` → `--stage score`（score 在 comet venv）
- **实现要点**：SFT 模型对 20k 源句各生成 4 个候选（`temperature=0.9, top_p=0.95`）→ 落盘 → comet venv 批量打 COMET-DA（有参考）与 Kiwi（免参考）
- **资源**：生成阶段显存 3.5GB；打分阶段显存 2.4GB（COMET-DA fp32）｜**耗时**：生成 **1.5h**（80k 次生成 @12 句/s）+ 打分 20 分钟
- **产出**：`data/interim/candidates.jsonl`（含两个分数）
- **验收**：候选非空率 >99%；同一源句 4 个候选的 DA 分数标准差 >0.03（否则模型已饱和，需换域或升温）

### S4.2 成对构造　待做 - **命令**：`python src\train\mine_dpo_pairs.py --stage pair`
- **规则**：`chosen=argmax(DA)`、`rejected=argmin(DA)`；保留条件 Δ_DA≥0.10、文本不同、长度比∈[0.6,1.6]；另加 2,000 条术语剔除型难负例
- **资源**：纯 CPU｜**耗时**：3 分钟
- **产出**：`dpo.pairs.train.jsonl`（预期 15–18k）、`dpo.pairs.dev.jsonl`（2k）
- **验收**：保留率 ≥60%；`src_hash` 与任何 test 集零交集
- **附带分析（零成本）**：在 2k 子集上算 **Spearman ρ(Kiwi, DA)** 与 **AUROC**，分域报告 → 决定路由信号是否可信（Gate 前置证据）

### S4.3 DPO 训练　待做（配置已就绪）

- **命令**：`python src\train\dpo_lora.py --config configs/g5_dpo.yaml`
- **配置**：`beta=0.1`、`max_length=384`、`batch=8`、`accum=2`、`lr=5e-6`、`epochs=1`、`precompute_ref_log_probs=true`、`loss_type=sigmoid`
- **资源**：显存 **8.68GB reserved（b8×L320 实测）**｜**耗时**：ref 预计算 8 分钟 + 训练 **0.5h**（18k ÷ 10.2 对/s）
- **验收**：`rewards/margins` 由 ~0 上升到正值；`rewards/accuracies` >0.6；dev COMET-DA 不低于 SFT
- **失败处置**：margin 不涨 → β 降到 0.05 或 lr 升到 1e-5；dev 掉分 → 提前停（用 dev 选 checkpoint，不用最后一个）

---

## S5 G5 管线（0.5h 机器 / 1.5 人日）

> 架构：**两遍式**。Pass1 只生成（主 venv），Pass2 只打分（comet venv），Pass3 在 CPU 上把 τ 当离线过滤器。8 个 τ 点共用一次生成，τ 扫描 GPU 成本为 0。

### S5.1 混合检索 + RRF　待做 - **命令**：`python src\infer\retrieve.py --pool seen --set dev`
- **实现要点**：BM25 top3 + Dense top3 → RRF(k=60) → top3；prompt 预算 320 token
- **资源**：显存 2.3GB｜内存 1.5GB｜**耗时**：实现 0.3 人日；dev 全量 2 分钟
- **验收**：RRF 在 dev 上优于任一单路（用 COMET-DA 比较 3 个配置）

### S5.2 软偏置 LogitsProcessor　待做 - **实现要点**：源句命中术语表 → 在生成前 40% 步内对法语术语首 token 加 `+2.0` logit
- **资源**：显存增量 <0.1GB｜**耗时**：实现 0.3 人日
- **验收**：单元测试 —— 构造含术语的源句，断言目标 token 的 logit 恰好 +2.0、其余不变；术语命中率相对 G4 提升
- **说明**：硬 Trie 对照臂已按工期删除（见 `configs/ablations.yaml`）

### S5.3 Critic / Refiner / Guard　待做 - **实现要点**：Critic 输出 JSON（issues + severity）；Refiner 出终稿；Guard 三条回退规则（长度比、4-gram 重复率 >0.30、langdetect≠fr），回退计数落盘
- **资源**：显存 3.5GB（同一份权重换 system prompt）｜**耗时**：实现 0.6 人日（含 prompt 调试）
- **验收**：JSON 解析成功率 >95%；`fallback_rate` <15%；dev 上 heavy 路径 COMET-DA > light 路径

### S5.4 Pass1/2/3 编排 + τ 标定　待做 - **命令**：`python src\infer\pipeline_g5.py --config configs/g5_pipeline.yaml --set dev` → `python src\eval\route.py`
- **资源**：Pass1 显存 5.6GB（Qwen+bge-m3）；Pass2 显存 2.4GB｜**耗时**：dev Pass1 11 分钟 + Pass2 5 分钟 + Pass3 <1 分钟
- **验收**：τ* 满足 `macro_COMET` 最大且 `latency ≤ 1.15×G4`；完整 Pareto 8 点全部落盘
- **延迟口径**：`latency(τ) = mean[t_draft + t_qe + 1{qe<τ}·(t_retr + t_heavy + t_critic + t_refiner)]`，各分量来自 Pass1 实测；τ* 处另跑一次真实串联管线做端到端校验

---

## S6 主表（0.6h 机器 / 0.3 人日）

### S6.1 五组 test 推理　待做 - **命令**：`python src\infer\run_matrix.py --group main --set test`
- **资源**：显存峰值 5.6GB｜**耗时**：**12 分钟**（5 组 × 3,271 句 @22 句/s，实测折算；beam4 臂另加 18 分钟）
- **产出**：5 个 `runs/<id>/pass1.jsonl` + registry 行

### S6.2 COMET 批量打分　完成（脚本已跑通）

- **命令**：`& "D:\HKStudy\.venv-comet\Scripts\python.exe" src\eval\score_comet.py --runs g1_zs,g2_rag,g3_sft,g4_sft_rag,g5_full --batch-size 32`
- **资源**：显存 2.4GB（DA）+ 2.3GB（Kiwi 镜像），两者同进程常驻共约 4.7GB｜**耗时**：实测 DA 17.8 句/s、Kiwi 40 句/s → 16,355 句对约 **22 分钟**
- **产出**：每 run 的 `scores.jsonl`（句级 DA + Kiwi，供 bootstrap／长度分桶／Shapley 复用）+ `scores_summary.json`（系统级、分域、两域宏平均）
- **验收（已达成）**：G1 零样本 48 句 MTNT dev → COMET-DA **0.7541**；QE 自动加载本地镜像 ckpt 并在 summary 里记录实际使用的模型路径
- **注意**：本步骤必须在 comet venv（Python 3.13）里跑；主 venv 的 Python 3.14 无法 import comet

---

## S7 消融（2.7h 机器 / 0.5 人日）

### S7.1 G5 留一 4 臂 + 长度匹配对照　待做 - **命令**：`run_matrix.py --group ablation --set test`
- **资源**：显存 5.6GB｜**耗时**：6 个配置 × 18 分钟 ≈ **1.8h**
- **验收**：每臂与 `g5_full` 只差一个模块；`no_rag` 与 `no_rag_len` 的 prompt 长度分布一致（打印均值差 <5%）

### S7.2 随机偏好对对照　待做 - **命令**：`dpo_lora.py --config configs/abl/dpo_random_pairs.yaml` → 重跑 test 推理
- **资源**：显存 8.7GB｜**耗时**：训练 0.5h + 推理 6 分钟
- **验收**：与 `g5_full` 的 COMET 差值有 CI；若差值不显著 → "DPO 有效"结论被推翻，如实写进报告
- **降级保护**：这一臂在砍单顺序里排最后，不得先砍

### S7.3 dev 机制对照（k / pool / always-heavy）　待做 - **命令**：`run_matrix.py --group dev_controls --set dev`
- **资源**：显存 5.6GB｜**耗时**：25 分钟
- **验收**：seen 与 unseen 池**规模都取 6,000**（否则差异混入规模效应）

---

## S8 精确 Shapley（1.3h 机器 / 0.3 人日，夜间）

- **命令**：`python src\eval\shapley.py --modules dpo,rag,con,agt --set dev`
- **实现要点**：16 个子集；draft 与 QE 分数跨子集缓存；heavy 生成组合上限 8 种（dpo×rag×con），Critic/Refiner 只在 agt=on 的 8 个子集跑
- **资源**：显存 5.6GB｜**耗时**：生成 62 分钟 + 打分 10 分钟 ≈ **1.3h**（实测吞吐折算，非 plan-v1 的 5.5h）
- **验收（Gate G6）**：`Σφ_i` 与 `v(full) − v(∅)` 相对误差 **<1%**（同 dev、同口径，数学上应恒等；`v(∅)`=仅 SFT 基座）
- **禁止**：拿 Σφ 去对账阶梯分解之和（后者等于 `v(G5)−v(G1)`，两者相差 Δ_SFT，本就不等）

---

## S9 打分与统计（1.0h 机器 / 1.0 人日）

### S9.1 全量指标计算　待做 - **命令**：`python src\eval\metrics.py --all`
- **产出**：BLEU(`tok=13a`)、chrF++、COMET-DA、Kiwi、术语命中率、长度比、**重复率**、**退化率**、延迟 p50/p95、峰值显存
- **资源**：CPU 为主｜**耗时**：20 分钟

### S9.2 bootstrap 与显著性　待做 - **命令**：`python src\eval\bootstrap.py --iterations 10000`
- **实现要点**：COMET 用句级分数重采样；**BLEU/chrF 重采样句对后重算语料级指标**（句级均值有偏）；MTNT 按 `comment_id` 分块重采样；配对 p 值 + Holm
- **比较族**：main(4) / ablation(4) 各自 Holm；分域与分长度桶标 exploratory 不校正
- **资源**：纯 CPU，10,000 次 × 语料级 BLEU 重算约 15 分钟｜**耗时**：30 分钟
- **验收**：所有主表差值都有 95% CI 与 p 值；`|Δ| < MDE(0.008–0.012)` 标 below resolution

### S9.3 长度分桶　待做 - **命令**：`python src\eval\length_buckets.py`
- **产出**：短/中/长三分位 × 系统 × 域 的 COMET 表（复用 S9.1 的句级分数，零额外 GPU）
- **耗时**：5 分钟

---

## S10 分析（0.2h 机器 / 1.0 人日）

### S10.1 归因表　待做
阶梯分解（叙事用）+ Shapley φ 值（结论用），分域给出主导因子。对 v2 文档的三个预期逐条给"支持/低于分辨限/证伪"。

### S10.2 失败案例　待做
G4 与 G5 各抽 30 条 MTNT + 30 条 EMEA 差值最大样本，8 类错误标签，单标注 + 裁判模型复核 30 条，卡方检验，3 个完整案例。人工 4h。

### S10.3 图表　待做
`python src\eval\make_figures.py` → 阶梯曲线、Pareto、延迟阶段堆叠、分域因子、显存-质量散点、长度分桶热图。全部从 `registry.csv` 生成。

---

## S11 报告（0h 机器 / 1.5 人日）

- **S11.1 正文**：按 `EXPERIMENT_PLAN.md` §13 的图表映射组织；限制章节必须写明 ①DPO 构对与主评测同用 COMET-DA 的循环性 ②Kiwi 权重来源（官方/镜像/自建信号）③单标注人 ④单 seed ⑤MTNT test 仅 1,020 条的分辨限
- **S11.2 可复现材料**：`registry.csv`（含 git commit、模型 revision、manifest sha256）、`configs/`、`requirements.lock`、`logs/probe_*.json`

---

## 夜间批处理编排

```powershell
# 挂机顺序（每步独立进程，避免显存叠加；日志各自落盘）
python src\train\sft_lora.py       --config configs\g3_sft.yaml          # 2.3h
python src\train\mine_dpo_pairs.py --stage generate                      # 1.5h
python src\train\dpo_lora.py       --config configs\g5_dpo.yaml          # 0.6h
python src\infer\run_matrix.py     --group ablation --set test           # 1.8h
python src\eval\shapley.py         --modules dpo,rag,con,agt --set dev   # 1.3h
```

约束：任意时刻只有一个进程占 GPU；每个脚本入口都设 `set_per_process_memory_fraction(0.95)`；每步结束写 `peak_vram.json`，超 14GB 自动降级 batch 并在 registry 标注。

---

## 汇总：资源占用一览

| 阶段 | 显存峰值（实测/估） | 内存 | 磁盘增量 | 机器时间 | 人日 |
| --- | --- | --- | --- | --- | --- |
| S0 环境 | 13.5GB（探针极端值） | 3GB | 23.2GB（模型） | 0.6h | 0.5 |
| S1 数据 | 2.3GB | 2.5GB | 0.6GB | 0.4h | 0.8 |
| S2 检索/基线 | 5.6GB | 2GB | 0.5GB | 1.0h | 0.7 |
| S3 SFT | **9.0GB reserved** | 3GB | 0.04GB | 2.5h | 0.5 |
| S4 DPO | 8.7GB | 3GB | 0.5GB | 2.7h | 0.8 |
| S5 G5 管线 | 5.6GB | 2GB | 0.2GB | 0.5h | 1.5 |
| S6 主表 | 5.6GB | 2GB | 0.1GB | 0.6h | 0.3 |
| S7 消融 | 8.7GB | 2GB | 0.3GB | 2.7h | 0.5 |
| S8 Shapley | 5.6GB | 2GB | 0.2GB | 1.3h | 0.3 |
| S9 统计 | 2.4GB | 4GB | 0.1GB | 1.0h | 1.0 |
| S10–S11 分析写作 | — | — | 0.05GB | 0.2h | 2.5 |
| **合计** | 上限 9.0GB（15.7GB 可用） | 峰值 4GB（31.4GB 可用） | ~26GB（D: 余 196GB） | **≈13.5h** | **≈9.4** |

---

## 附录：实验过程中修掉的 bug（复现时必读）

| # | 位置 | 症状 | 根因 | 修法 |
| --- | --- | --- | --- | --- |
| B1 | `pipeline_g5.py` 检索缓存路径 | heavy path 日志显示"检索缓存 0 条"，实际没有注入任何检索示例 | 写成 `data/processed/index/cache`，实际在 `data/index/cache` | 改成 `ROOT/data/index/cache` |
| B2 | `generate.py` / `pipeline_g5.py` 的 DPO adapter 加载 | "DPO 直出"得分 0.7468，几乎等于零样本 0.7471；G5 在 clean 域从 0.969 掉到 0.901 | DPO 训练时叠在**已合并 SFT 的权重**上，但 `adapter_config.json` 只记 `base=Qwen`，推理时挂到原始 Qwen 等于丢掉全部 SFT 增益 | 先合并 SFT adapter，再 `copy.deepcopy` 后叠 DPO；`generate.py` 增加 `--base-adapter` |
| B3 | `pipeline_g5.py` 上下文格式 | 首轮 G5 质量异常低 | 用了 inline 单消息格式，实测比 turns 低 0.09 COMET | 重路径改为 `user/assistant` 轮次 + 末尾一条 user（含术语提示） |
| B4 | 术语候选的 n-gram 生成 | 出现 `body weight day` 这类语料里不存在的短语 | `body weight/day` 的斜杠在分词时被吞 | n-gram 只在 `/ ( ) , ; :` 与空格连字符切出的词段内部生成 |
| B5 | `metrics.py` 术语表加载 | MTNT 俚语命中率始终不出现 | 只加载了 EMEA 术语表 | 两域各一张表，MTNT 跳过 `mode=omit` |
| B6 | WMT24 Chat 解析 | 训练集里混进 fr→en 方向的句子，dev 为 0 条 | CSV 同时含两个方向；dev 写入时按 `split=="dev"` 过滤而实际值是 `valid` | 按 `source_language/target_language` 过滤，修正 dev 过滤条件 |

**B2 的影响范围**：首轮的 `g5_full`、五个留一臂、τ 扫描全部作废并已重跑。任何涉及 DPO 的数字都必须用"先合 SFT 再叠 DPO"的方式复现。
