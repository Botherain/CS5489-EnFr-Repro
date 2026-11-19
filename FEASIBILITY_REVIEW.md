# 可行性与完善性检验报告（EXPERIMENT_PLAN.md plan-v1）

> 检验日期：2026-11-14　执行环境：RTX 5080 单卡 / Windows / `D:\HKStudy\.venv`
> 检验方式：三条腿并行 —— ①本机实测探针（6 个脚本，全部落盘 `proj/logs/`）②独立模型审阅（reviewer 子代理）③纸面复算
> 结论摘要：**方案主体可行，但 5 处关键参数被实测推翻，1 处归因闸门存在数学错误，1 处依赖被 gated 挡住。全部已修订，plan 升级到 v2。**

---

## 1. 实测结论汇总

所有数字来自 `proj/logs/` 下的探针输出，可复跑。

### 1.1 环境与依赖（`logs/probe_api.py` 输出）

| 项 | 实测 | 影响 |
| --- | --- | --- |
| GPU / capability | RTX 5080，`(12,0)`，总 17.07GB，空闲 15.71GB | 与原分析一致 |
| torch | `2.11.0+cu128`，bf16 正常，SDPA 正常 | 达标 |
| 显存带宽 | device-to-device 2GB 拷贝实测 **771 GB/s**（厂标 960） | 报告里用实测值，别抄规格书 |
| transformers | `5.17.0` | 见下条 |
| **TRL / transformers API 代差** | `SFTConfig` 无 `warmup_ratio`（只有 `warmup_steps`）；`train_dataset` 必须是 `datasets.Dataset`（传 list 直接 `TypeError`）；`DataCollatorForCompletionOnlyLM` 已删除；`DPOConfig` 无 `max_prompt_length` | v2 文档与网上 4.x 教程的配置**全部跑不起来**，必须按实测字段名写 |
| 安装结果 | `peft 0.21.3 / trl 1.18.0 / datasets 5.2.0 / accelerate 1.15.0 / sentence-transformers 6.3.0 / faiss-cpu 1.16.0 / sacrebleu 2.6.0`；**torch 与 transformers 未被改动** | 主环境安全，无需回滚 |
| assistant-only loss | `assistant_only_loss=True` + `packing`/`padding_free` 字段均存在 | 取代已删除的 collator |
| **COMET 独立 venv** | Python **3.13.13**（uv 创建）+ `torch 2.11.0+cu128` + `unbabel-comet 2.2.7` + `setuptools 78.1.0`，GPU 可用，打分 15–18 句/s | 见下方说明 |

**Python 3.14 装不上 COMET（新发现的硬约束）**：主 venv 是 Python 3.14.5，而 `unbabel-comet 2.2.7` 依赖的 `comet/models/lru_cache.py` 会 `from functools import _HashedSeq`，该私有符号在 **Python 3.14 中已被移除**（3.13 仍在）。此外 `torchmetrics 0.10.3` 需要 `pkg_resources`，要求 `setuptools<81`。处置：用 `uv venv --python 3.13` 单独建 `.venv-comet`，先装 cu128 版 torch 再装 comet（否则 PyPI 默认给 Windows CPU 版 torch）。**主环境用 Python 3.14，打分环境用 Python 3.13，两个解释器缺一不可。**

依赖锁定文件：`proj/logs/requirements-main.lock`（260 项）、`proj/logs/requirements-comet.lock`（49 项）。

### 1.2 模型资产（`logs/download.log`、`logs/probe_qe.py`）

| 资产 | 状态 | 体积 |
| --- | --- | --- |
| `Qwen/Qwen2.5-1.5B-Instruct` | 完成，已下载到 `D:\HKStudy\hf` | 3.09GB safetensors，1543.71M 参数，vocab 151,936，28 层，KV heads 2，head_dim 128，tie embeddings |
| `BAAI/bge-m3` | 完成，已下载 | 2.27GB |
| `Unbabel/wmt22-comet-da` | 完成，已下载，**非 gated** | 2.32GB ckpt，backbone `xlm-roberta-large`（560M） |
| `Unbabel/wmt22-cometkiwi-da` | 注意 gated=auto，无 token 时 401 → **已用镜像转换打通**（见 §4） | 2.26GB（XLM-R large 级，`class_identifier=unified_metric`，`pretrained_model=microsoft/infoxlm-large`，license cc-by-nc-sa-4.0） |

**这是 plan-v1 没有预见的 P0 阻塞**：v2 设计稿的动态路由整条链路建立在 CometKiwi 上，而本机 `HF_TOKEN` 为空、C 盘也无 token 文件。三级降级路径（见 §4）。

非 gated 替代品实测清单：

| 仓库 | gated | 体积 | 可用性判断 |
| --- | --- | --- | --- |
| `Supervache/wmt22-cometkiwi-da-safeformat` | False | 2.26GB safetensors | 同权重换格式，可直接用；报告须披露镜像来源 |
| `solailabs/wmt22-cometkiwi-da-int8` | False | 1.13GB | int8 量化版，省显存，打分与官方有偏差 |
| `vince62s/wmt22-cometkiwi-da-roberta-large` | False | 2.28GB | 键名重映射版，加载路径非标准，风险高 |
| `vince62s/wmt23-cometkiwi-da-roberta-xl` | False | 6.99GB | XLM-R XL（3.5B），只能独占显存离线打分，不能与 Qwen 常驻 |

### 1.3 训练显存与吞吐（`logs/probe_train_mem.json`、`logs/smoke_all.log`、`logs/smoke_env.log`）

**关键发现：显存瓶颈不在模型，在 151,936 词表的 loss。** `transformers.loss.loss_utils.ForCausalLMLoss` 第一行就是 `logits = logits.float()`，且**不分块**（探针实测 `loss_chunked: False`）。全序列打标签时：

| 配置（naive 全序列 CE） | 峰值 |
| --- | --- |
| batch 1 × 512 | 7.33GB |
| batch 2 × 512 | 11.38GB |
| batch 4 × 512 | **OOM** |
| DPO pairs 1 × 640 | 13.52GB |
| DPO pairs 2 × 640 | **OOM** |

走真实 TRL 路径（`assistant_only_loss=True`，只对 assistant 段计损失）后，同样硬件的实测数字完全不同：

| 配置（TRL SFTTrainer 实跑） | peak_alloc | peak_reserved | 吞吐 |
| --- | --- | --- | --- |
| b8 × L320，packing=True（padding_free 自动开） | 6.20GB | 11.36GB | 6.6 样本/s |
| b8 × L320，packing=False | 6.19GB | 10.69GB | **17.6 样本/s** |
| b4 × L512，packing=True | 5.62GB | 11.67GB | 5.7 样本/s |
| DPO b2 × L320，`precompute_ref_log_probs=True` | **3.99GB** | 4.45GB | 6.9 样本/s，0.31 s/step |

最坏情况包络（`logs/smoke_env.log`，`--longest` 取全库最长 512 条样本，构造满长度 batch）：

| 配置 | peak_alloc | peak_reserved | s/step | 样本/s |
| --- | --- | --- | --- | --- |
| SFT b8 × L384（最长样本） | 6.76GB | 7.36GB | 0.974 | 8.2 |
| SFT **b16 × L384（最长样本）** | 8.12GB | **9.02GB** | 1.679 | 9.5 |
| SFT b16 × L320（常规混合长度） | 7.78GB | 8.75GB | 1.045 | **15.3** |
| SFT b32 × L320（常规混合长度） | 8.53GB | 9.53GB | 2.126 | 15.0 |
| DPO b4 × L384（最长样本） | 6.76GB | 7.30GB | 0.830 | 4.8 对/s |
| DPO b8 × L320（常规） | 8.01GB | 8.68GB | 0.786 | 10.2 对/s |

`b16→b32` 吞吐不增（15.3 → 15.0 样本/s）说明已进入计算受限区，**batch 16 是甜点**，再大只吃显存不提速。最坏情况 reserved 9.02GB，距 15.7GB 上限还有 6.7GB。

下面三条推翻了 plan-v1。

1. **packing 要关掉。** 本项目数据极短（clean p50=30 token、EMEA p50=43、MTNT p50=41），打包到 320 会让每步处理 8×320=2,560 token，而不打包时动态 padding 只处理约 400 token。实测不打包快 **2.7 倍**。plan-v1 写的 `packing: true` 是负优化。
2. **`max_length=640` 是浪费。** 实测 token 长度（含 prompt）：MTNT test 最大 268、EMEA test 最大 321、clean 最大 47；只有 MTNT **train** 的长尾超 384（1.73%）与 512（0.92%）。取 `max_length=384`，评测集零截断，训练集截断 1.7%（记录在案）。
3. **DPO 显存比 plan-v1 估的低 36%**（3.99GB vs 6.3GB），因为 `precompute_ref_log_probs=True` 生效且序列短。

`peak_reserved` 与 `peak_alloc` 的差（6.2 vs 10.7GB）说明碎片化占 4GB 量级。处置：`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.85` + `SFTConfig.torch_empty_cache_steps=100`，并且**训练与推理不共进程**。

### 1.4 生成吞吐与延迟（`logs/probe_train_mem.json` §5，独占 CPU/GPU 实测）

| batch | new_tokens | peak | 吞吐 | 每句 |
| --- | --- | --- | --- | --- |
| 1 | 48 | 3.1GB* | 47 tok/s | **1,014 ms** |
| 1 | 128 | 3.1GB* | 47 tok/s | 2,726 ms |
| 8 | 64 | 3.2GB* | 365 tok/s | 175 ms |
| 16 | 64 | 3.2GB* | 723 tok/s | 89 ms |
| 32 | 64 | 3.3GB* | **1,422 tok/s** | 45 ms |
| 32 | 111 | 3.3GB* | 1,422 tok/s | 78 ms |

\* 该探针进程内残留了上一份模型实例，peak 读数偏高约 3.1GB；干净进程下的生成峰值见 `logs/probe_gpu.json`（b32 = 3.30GB）。

两个结论必须写进方案。

- **batch=1 的每 token 延迟 21ms，是 batch=32（3.1ms/token 等效）的 6.8 倍**，说明小 batch 下 Python/驱动开销主导，不是显存带宽主导。原分析"960GB/s 带宽红利"只在批处理下成立。延迟指标必须区分两个口径：**逐句延迟（batch=1，反映真实交互）** 与 **吞吐（batch=32，反映批量离线）**，Pareto 曲线用前者，成本核算用后者。
- 评测总耗时远低于 plan-v1 估算：3,271 句 × 5 个系统，按 22 句/s 只需 **12 分钟**（不含重路径）。plan-v1 的 S6 预算 1.5h 是过估，而 §10.1 表内推理成本合计 2.7h 也是过估。

### 1.5 数据流水线实跑结果（`logs/build_dataset.log`、`reports/data_stats.json`）

| 域 | train | dev | test | memory_pool |
| --- | --- | --- | --- | --- |
| clean（去重后） | 1,800 | 200 | 251 | — |
| MTNT（官方切分 + 去污染） | 32,593 | 852 | 1,020 | 3,032（unseen） |
| EMEA（过滤 + 近重复去除 + 块切分） | 27,000 | 1,000 | 2,000 | 6,000 |

- EMEA 原始 1,092,568 行 → 长度/质量过滤 259,778 → **8-gram 近重复过滤后 143,056**（丢掉 116,722 条跨文档样板句，占 45%）。
- 泄漏审计（test/dev vs train+memory_pool，句级精确 + 8-gram）：clean 全 0；MTNT test 全 0、dev 1 条 8-gram；EMEA test 8 条 8-gram（0.4%）、dev 0。
- **plan-v1 的 EMEA 切分方案实测不合格**：只做精确去重 + 块切分时，EMEA test 与 train 的 8-gram 重叠率是 **26.6%**（532/2000）。加了全量 8-gram 指纹去重后降到 0.4%，才过 G1 闸门（<0.5%）。这是审阅没抓到、实跑才暴露的问题。
- MTNT 采取"清洗 train、保留官方 test"策略：剔除与 dev/test 有 8-gram 重叠的训练句 260 条，test 保持 1,020 条原样以便与文献对照。

### 1.6 端到端链路验证（Pass1 生成 → Pass2 打分）

不是纸面推演，四段链路都已实跑通并落盘：

| 环节 | 脚本 | 实测结果 |
| --- | --- | --- |
| 检索索引 | `src/data/build_index.py` | seen 池 **61,393** 句对，bge-m3 编码 33.8s，峰值显存 3.09GB，FAISS 251.5MB + BM25 11MB；unseen 池 9,032 句，5.1s |
| 批量生成 | `src/infer/generate.py` | 零样本 48 句跑通，峰值 3.78GB；RAG 模式峰值 4.86GB；batch=1 逐句延迟 p50 599ms；meta 自动记 git commit / 峰值显存 / 吞吐 |
| COMET 打分 | `src/eval/score_comet.py`（comet venv） | DA + Kiwi 双模型同时打分，15–18 句/s @batch32；G1 零样本在 48 句 MTNT dev 上 **COMET-DA = 0.7541** |
| QE 信号 | 同上 | ρ(Kiwi, DA) = 0.753（§4） |

跑通过程中修掉的三个真实缺陷（都已写回代码）：

1. **检索结果重复**：同一源句在 EMEA/MTNT 里常有多条参考译文，直接取 top-3 会出现两条几乎相同的示例（实测已出现）。修法：过取 4k 后按源句归一化去重，并排除与查询同源的条目。
2. **BM25 是纯 Python 打分**，61,393 文档下约 0.21s/句，3,271 句 test 要 12 分钟；G2/G4/G5/消融/Shapley 每个 RAG run 都重付一遍就是数小时。修法：加磁盘检索缓存 `data/index/cache/{pool}.k{k}.jsonl`，同一 (pool, k, id) 只算一次，跨 run 复用（实测第二次全命中）。
3. **`gen_tokens` 统计错误**：批量生成会把整批 padding 到最长序列，`gen.shape[1]` 恒等于 `max_new_tokens`。修法：逐句数非 pad token（`(gen != pad_id).sum(1)`）。这个字段是吞吐与成本核算的分母，错了会让所有延迟结论失真。

---

## 2. 独立审阅结果与采纳判定

审阅由 `reviewer` 子代理执行。**模型记录：`bailian-token-plan-personal/qwen3.8-max`，与主代理同家族 → 按约定标记 `same-family / provisional`**，其结论不作为独立第三方证据，只作为交叉检查；凡与本机实测冲突处，以实测为准。

审阅总体判断（引）：事实基础扎实、对照设计超出课程平均水平，但存在一处归因方法学硬伤、一处架构矛盾、多处预算自相矛盾；GPU 小时低估 15–25%，真正瓶颈是未预算的工程墙钟；"一周单人单卡完成"打 **4/10**。

逐条采纳判定：

| # | 审阅意见 | 判定 | 依据 / 处置 |
| --- | --- | --- | --- |
| P0 | Shapley 与阶梯分解不可能对账：Σφ = v(full)−v(∅)，v(∅)≈SFT 基座；阶梯和 = v(G5)−v(G1)，两者相差 Δ_SFT | **采纳** | 数学上确实恒不相等，G6 闸门必然误报。plan-v2 改为：闸门只校验 Shapley 效率公理自身（同 dev 同口径，误差应 <1%），阶梯分解标注"仅叙事用" |
| P1 | 两 venv 隔离 vs Kiwi 需在生成循环内打分，架构矛盾 | **采纳（并强化）** | 改为两遍式：Pass1 只生成（Qwen+检索器常驻）→ Pass2 在 comet venv 批量打分 → Pass3 在 CPU 上把 τ 当离线过滤器。**副产品：8 个 τ 点复用同一次生成，τ 扫描 GPU 成本为 0**（plan-v1 估 1.5h） |
| P1 | bs=8 的安全性未经自家 bs=4 表格支撑；151,936 词表 fp32 logits 可能吃 6–9GB | **采纳，已实测** | 见 §1.3：真实路径 b8×L320 = 6.20GB alloc / 10.69GB reserved；naive 路径 b4×L512 直接 OOM。审阅的担忧方向正确，量级被高估（因为它假设全序列 CE） |
| P1 | S6 预算 1.5h 与自家推理成本 2.7h 矛盾；S7 3h 装不下全部对照 | **采纳，但两个数都过估** | 实测 22 句/s → 主表 5 组共 12 分钟生成；S7 全部对照约 1.5h。真实瓶颈是工程墙钟，plan-v2 单列人日列 |
| P1 | "三域宏平均"与"clean 不参与结论"互斥 | **采纳** | plan-v2 统一为 **两域（MTNT/EMEA）等权宏平均**，clean 只出现在 sanity 节 |
| P1 | 只预算 GPU 小时，无工程墙钟 | **采纳** | plan-v2 §11 增加"人日"列，并给出 MVP / 完整两条路线 |
| P2 | BLEU/chrF 不能句级分解，句级均值有偏 | **采纳** | bootstrap 改为"重采样句对 → 重算语料级指标"；COMET 保留句级分解 |
| P2 | MTNT 同 comment_id 相关，iid bootstrap 低估方差 | **采纳** | 按 comment_id 分块重采样（字段已在数据里保留） |
| P2 | 未声明功效/MDE | **采纳** | 预注册 MDE：n=1,020 配对 COMET 差值约 **0.008–0.012**；`|Δ|<MDE` 记为 below resolution，与"证伪"分开 |
| P2 | Holm 只覆盖 4 个比较，族未定义 | **采纳** | 显式划分三族：main(4)、ablation(4)、exploratory(不校正，标注) |
| P2 | 泄漏审计要扩到 memory_pool；seen/unseen 池规模要匹配 | **采纳** | 审计已含 memory_pool（§1.5）；对照臂两边都抽 6,000 |
| P2 | DPO 用 comet-da 构对、又用 comet-da 评测，循环性未声明 | **采纳** | 限制章节写明；chrF++ 与裁判分列为共主证据；Kiwi（若可用）作独立第三方 |
| P2 | CoT 组件被删但未登记 | **采纳** | 附录 A 增行，理由：1.5B 上 CoT 收益不稳 + 工期 |
| P2 | 双人标注（κ/α）单人不可行 | **采纳** | 改单标注 + 裁判模型复核 30 条，如实披露 |
| P2 | 小口径错误（τ 网格 8 点写成 7 点、6.0 vs 6.5GB、G3 闸门无单位、DPO 采样比与评测域比反向、beam4/RAG-KV 未入账） | **采纳** | 全部订正；DPO 挖掘比改为 MTNT:EMEA = 1,020:2,000 同向（即 6k:12k） |
| P2 | 风险表缺 Kiwi 进程架构、第二标注人、10GB 下载 | **采纳** | 补三行；Kiwi gated 已升级为 P0 风险 |
| P2 | 降级顺序把 `dpo_random_pairs` 排在 Shapley 之前砍 | **采纳** | 顺序改为：裁判 → 硬 Trie（已删）→ 第二 seed → Shapley 降留一；`dpo_random_pairs` 保底不砍 |
| 加 | 长度分桶分析、重复退化率两个零成本分析 | **采纳** | 已写入 `configs/eval.yaml` |
| 删 | 硬 Trie 对照臂 | **采纳** | 已删，理由记录在 `configs/ablations.yaml` |
| 删 | 双人信度协议 | **采纳** | 已删 |
| 保留 | 16 子集精确 Shapley 不砍，加缓存复用 | **采纳** | 缓存复用后实测口径约 1.5–2h，不是审阅估的 5.5h，也不是 plan-v1 的 5.5h |
| 判定 | Shapley 5.5h 是"低估" | **驳回** | 审阅按"每句 3–4 次生成"估，但忽略了 draft 与 QE 分数可跨子集缓存、且 heavy 生成的组合上限是 8 种（dpo×rag×con）而非 16 种。实测 22 句/s 下 1,852×24 次生成 ≈ 34 分钟，加 Critic/Refiner 长输出翻倍 ≈ 70 分钟 |
| 判定 | 4/10 的一周可行性 | **部分采纳** | GPU 侧 4/10 过于悲观（实测总量约 12–14 GPU 小时）；墙钟侧判断正确。plan-v2 给出 MVP 路线（3.5 人日）与完整路线（7–8 人日）两档 |

---

## 3. 被实测推翻的 plan-v1 参数（一览）

| 项 | plan-v1 | 实测/修订 | 变化幅度 |
| --- | --- | --- | --- |
| SFT max_length | 640 | **384** | −40% 序列长度 |
| SFT packing | true | **false**（数据太短，打包是负优化） | 吞吐 +2.7× |
| SFT batch | 8（未验证） | 8 已验证 6.2GB；最坏情况 batch 见 §1.3 | 已验证 |
| SFT 峰值 | 4.8GB（估算） | **6.2GB alloc / 10.7GB reserved** | +30% alloc，reserved 是真实约束 |
| SFT 时长 | 1.7–2.5h | 62k×2 epoch ÷ 15.3 样本/s（b16×L320 实测）≈ **2.3h** | 落在区间内 |
| DPO 峰值 | 6.3GB（估算） | **3.99GB**（b2×L320）；b8×L320 = 8.01GB | −37% ~ +27% |
| DPO 时长 | 1.0–1.3h | 18k 对 ÷ 10.2 对/s ≈ **0.5h** + ref 预计算 8 分钟（实测 36 对/s） | −50% |
| τ 扫描 | 1.5h GPU | **0h**（离线过滤器，复用 Pass1） | −100% |
| 主表推理 | 2.7h | **12 分钟**（5 组 × 3,271 句 @22 句/s） | −93% |
| Shapley | 5.5h | **1.5–2h**（缓存复用 + 实测吞吐） | −65% |
| 显存带宽 | 960 GB/s（规格） | **771 GB/s**（实测 d2d） | −20% |
| LoRA 可训练比 | 1.20%（已修正 v2 文档的 0.2%） | 实测 **1.182%**（18.46M） | 一致 |
| Kiwi 可用性 | 默认可用 | **gated，401** | P0 阻塞 |
| EMEA 切分 | 精确去重 + 块切分 | 需再加 **8-gram 指纹去重**（否则 test 泄漏 26.6%） | 设计缺陷已修 |
| TRL 配置字段 | 按 4.x 教程 | `warmup_steps`（无 ratio）、`Dataset` 对象、`assistant_only_loss` | 全部改写 |

修订后 GPU 总时约 **12–14 小时**（plan-v1 写 21.5h，审阅估 25–28h）。

---

## 4. P0 阻塞已解除：CometKiwi 通过镜像转换打通

**问题**：`Unbabel/wmt22-cometkiwi-da` 为 `gated=auto`，本机 `HF_TOKEN` 为空 → 下载 401。v2 设计稿的整条动态路由都建立在这个模型上。

**已执行的解法（次选路径，实测通过）**：

1. 非 gated 镜像 `Supervache/wmt22-cometkiwi-da-safeformat` 下载成功（2.26GB safetensors）。其 `hparams.yaml` 为 `class_identifier=unified_metric`、`pretrained_model=microsoft/infoxlm-large`、`input_segments=[mt, src]`、license `cc-by-nc-sa-4.0`，与官方 CometKiwi 的规格一致（官方仓库 2.26GB 也是 XLM-R large 级，不是 XL 级）。
2. comet 2.2.7 的 `load_from_checkpoint` 只认 `<dir>/checkpoints/model.ckpt` + `<dir>/hparams.yaml` 的 Lightning 布局，直接加载 safetensors 报 `Invalid checkpoint path`。已写 `src/eval/prepare_qe.py` 做一次性格式转换：读 safetensors → 包成 `{"state_dict", "hyper_parameters", "epoch", "global_step", "pytorch-lightning_version"}` → 存为 `runs/qe_local/wmt22-cometkiwi-da-mirror/checkpoints/model.ckpt`。
3. 加载结果为 `UnifiedMetric`，三句验证分数 **好译 0.8913 / 坏译 0.6006 / 俚语译 0.8716**，排序正确（唯一告警是 `encoder.model.embeddings.position_ids` 这个 buffer 不在 state dict 里，属正常）。

**首个路由信号可靠性证据**（48 句 MTNT dev 的零样本 draft）：

| 指标 | 值 |
| --- | --- |
| COMET-DA 均值 / 值域 | 0.7541 / [0.242, 0.991] |
| Kiwi 均值 / 值域 | 0.7540 / **[0.485, 0.886]** |
| Spearman ρ(Kiwi, DA) | **0.753**（p<1e-4） |
| Pearson r | 0.735 |

ρ=0.753 远高于方案设定的 0.4 降级阈值 → 路由信号可用。但 **Kiwi 的值域被压缩到约 0.4 宽度**，两个后果必须写进方案：①τ 网格要按 Kiwi 自身分布标定，不能沿用 DA 的分位；②v2 设计稿默认的 τ=0.65 恰好落在 Kiwi 值域中部，说明原设定并非离谱，但仍需 dev 扫描确认。

**残留风险与升级路径**：镜像权重未与官方逐位校验。报告的方法节与限制节必须写明"QE 使用社区镜像（cc-by-nc-sa-4.0）转换所得"；若后续拿到 HF token，重跑 `download_assets.py --only kiwi` 并在 200 句上比对两套分数的一致性（预期 Pearson>0.99），把结果补进附录。

**兜底方案（保留但暂不启用）**：自建置信度信号 = draft 平均 token logprob（零成本）+ K=3 采样自一致性，用 COMET-DA 在 dev 上验 AUROC/Spearman 后选优或线性组合。仅在镜像分数与 DA 的相关性在完整 dev（1,852 句）上跌破 0.4 时启用。


---

## 5. 完善性检查：还没验证的项

| 项 | 状态 | 何时验证 |
| --- | --- | --- |
| comet venv 能否装上 | 完成，已验证（需 Python 3.13，见 §1.1） | 完成 |
| COMET-DA / Kiwi 打分链路 | 完成，已验证，15–18 句/s | 完成 |
| 最坏情况 batch 的显存（全长样本） | 完成，已测：SFT b16×L384 = 9.02GB reserved | `logs/smoke_env.log` |
| Pass1 生成链路（零样本 / RAG） | 完成，已冒烟通过（48 句 / 16 句） | 完成 |
| 检索索引与缓存 | 完成，已建（seen 61,393 / unseen 9,032） | 完成 |
| QE 信号可靠性 | 待跑 初步 ρ=0.753（48 句）；需在完整 dev 1,852 句上复算 + 补 AUROC | S5 阶段 |
| **检索质量（MTNT 噪声长句）** | 注意 **存疑**：自检 5 句里，短句（"yea"→"yay/Ouais" 0.78）命中好，长噪声句 top-3 相似度只有 0.59–0.62 且语义偏离 | S2 阶段人工看 20 条；若确实差，G2 增益会很小，需把这一现象本身写成"噪声域检索失效"分析 |
| 真实 SFT 收敛（loss 下降、dev COMET 提升） | 未验证，只跑通 8–20 步冒烟 | S3 阶段，前 400 步看 loss 曲线 |
| BM25 + Dense 的 RRF 是否优于单路 | 未验证 | S2 阶段 dev 对照 |
| 软偏置 LogitsProcessor 的正确性 | 未实现 | S5 阶段，先写单元测试 |
| Critic 输出 JSON 的解析鲁棒性 | 未实现 | S5 阶段，失败即回退并统计 fallback_rate |
| 术语表（EMEA 200 + MTNT 100） | 未做 | S1.3 |
| bootstrap / Holm / 长度分桶 harness | 未实现 | S9 |

---

## 6. 可行性判定（修订后）

- **显存**：可行，且余量充足。最重的一步是 SFT（b16×L384 最坏 8.12GB alloc / 9.02GB reserved），距 15.7GB 上限有 6.7GB 缓冲；碎片化用 `garbage_collection_threshold` + `torch_empty_cache_steps` 控制。禁止训练与推理共进程。
- **时间**：GPU 侧 12–14 小时，可在一周内夜间跑完。墙钟侧是真正约束，见 checklist 的人日列。
- **数据**：可行，泄漏已压到 0.4% 以下并通过审计；代价是 EMEA 可用量从 28 万降到 14.3 万（仍远超所需的 3.6 万）。
- **依赖**：主环境已装好且未破坏 torch；COMET 独立 venv **已建成并验证**（必须 Python 3.13，3.14 下 `functools._HashedSeq` 缺失导致 comet 无法 import）；Kiwi **已通过镜像转换打通**，并给出 ρ=0.753 的可用性证据，残留风险是镜像未与官方逐位校验（报告披露 + 拿到 token 后补一致性检查）。
- **方法学**：P0 归因闸门已修正，统计协议已补齐（语料级重采样、聚类 bootstrap、MDE、比较族）。
- **已完成进度（本次检验的副产品）**：S0 环境全部完成、S1 数据流水线全部完成（含泄漏审计）、检索索引已建、Pass1 生成与 Pass2 打分代码已冒烟通过。按 checklist 口径，**9.4 人日中已消耗约 1.3 人日**，且最容易踩坑的环境与数据两段已经落地。
- **新增风险（实测发现）**：bge-m3 对 MTNT 长噪声句的检索相关性偏弱（top-3 相似度仅 0.59–0.62，短句则达 0.78），G2/G4 在俚语域的增益可能低于预期。不阻塞流程，但会改变归因结论的形态；S2 阶段人工核验 20 条后决定是否加大 BM25 权重或改字符级 n-gram。
- **交付风险**：完整版约 7–8 人日，单人一周紧张。checklist 给出 MVP 路线（3.5 人日，含 G1–G5 主表 + 4 个留一 + 路由 Pareto + 归因），可先交 MVP 再按剩余时间加项。
