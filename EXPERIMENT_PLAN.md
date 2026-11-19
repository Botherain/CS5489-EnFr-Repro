# CS5489-Project-Think-v2 实验方案（plan-v2，单卡 RTX 5080 16GB）

> 版本：plan-v2（2026-11-14）　取代 plan-v1
> 修订依据：本机实测探针（`proj/logs/`）+ 独立审阅（`FEASIBILITY_REVIEW.md` §2）
> 操作手册：`EXECUTION_CHECKLIST.md`　机器可读配置：`proj/configs/*.yaml`
> 本版所有显存、吞吐、时长数字均为**实测或按实测折算**，plan-v1 的估算值已废弃

---

## 0. 相对 plan-v1 的九处实质修订

| # | 修订 | 依据 |
| --- | --- | --- |
| R1 | `max_length` 640 → **384**；`packing` true → **false** | 实测 token 长度：MTNT test 最大 268、EMEA test 最大 321；数据极短，打包反而使吞吐从 17.6 降到 6.6 样本/s |
| R2 | SFT batch 8 → **16**，实测最坏 9.02GB reserved | `logs/smoke_env.log`；b16→b32 吞吐不增，已计算受限 |
| R3 | DPO 峰值 6.3GB → 实测 **3.99GB**(b2) / **8.01GB**(b8) | `precompute_ref_log_probs=True` 生效 |
| R4 | 路由改**两遍式**：Pass1 生成 → Pass2 打分（comet venv）→ Pass3 把 τ 当离线过滤器 | 消除"两 venv 隔离"与"Kiwi 常驻生成进程"的架构矛盾；8 个 τ 点复用一次生成，τ 扫描 GPU 成本归零 |
| R5 | 头条指标改为 **MTNT/EMEA 两域等权宏平均**，clean 集只作 sanity | plan-v1 里"三域宏平均"与"clean 不参与结论"互斥 |
| R6 | 归因闸门修正：只校验 Shapley 效率公理 `Σφ = v(full) − v(∅)`，`v(∅)` = 仅 SFT 基座 | 阶梯分解之和 = `v(G5) − v(G1)`，与 Σφ 相差 Δ_SFT，恒不相等，plan-v1 的 <5% 对账闸门必然误报 |
| R7 | EMEA 切分增加**全量 8-gram 指纹去重** | 只做精确去重时 test 与 train 的 8-gram 重叠实测 **26.6%**，加指纹去重后降到 0.4% |
| R8 | 统计协议：BLEU/chrF 改**重采样句对后重算语料级**；MTNT 按 `comment_id` 分块 bootstrap；预注册 MDE 0.008–0.012；比较族显式划分 | 句级 BLEU 均值有偏；同帖句子相关；n=1,020 下需声明分辨限 |
| R9 | 删除硬 Trie 对照臂、双人信度协议；登记 CoT 组件删除 | 工期；单人无法履行第二标注人 |
| R10 | QE 模型改走**镜像转本地 ckpt**；τ 网格按 Kiwi 自身值域标定 | 官方仓库 gated（401）；实测 Kiwi 值域压缩到 [0.485, 0.886]，而 DA 是 [0.242, 0.991]，沿用 DA 分位会选错阈值 |
| R11 | COMET 打分环境必须用 **Python 3.13** 单独建 venv | `unbabel-comet 2.2.7` 依赖 `functools._HashedSeq`，该符号在 Python 3.14 已移除；主 venv 是 3.14.5 |

**外部依赖状态**：官方 `Unbabel/wmt22-cometkiwi-da` 为 gated，本机无 HF token。已用非 gated 镜像转换为本地 ckpt 并验证可用（`FEASIBILITY_REVIEW.md` §4），ρ(Kiwi, DA)=0.753。

---

## 1. 硬件与环境（实测）

| 项 | 实测值 |
| --- | --- |
| GPU | RTX 5080，capability `(12,0)`，总 17.07GB（16GiB），空闲 15.71GB |
| 显存带宽 | device-to-device 实测 **771 GB/s**（规格书 960，报告用实测值） |
| 软件栈 | torch `2.11.0+cu128`、transformers `5.17.0`、trl `1.18.0`、peft `0.21.3`、datasets `5.2.0`、faiss-cpu `1.16.0`、sacrebleu `2.6.0` |
| 基座 | `Qwen/Qwen2.5-1.5B-Instruct`：1,543.71M 参数、bf16 权重 **3.088GB**、vocab 151,936、28 层、KV heads 2、head_dim 128、tie embeddings |
| 环境隔离 | 主 venv `D:\HKStudy\.venv`（**Python 3.14.5**，训练/推理/检索）；COMET venv `D:\HKStudy\.venv-comet`（**Python 3.13.13**，只打分）。comet 2.2.7 依赖 `functools._HashedSeq`，3.14 已移除，故双版本是必需而非优化 |
| 模型缓存 | `HF_HOME=D:\HKStudy\hf`（C 盘仅余 27.9GB，必须放 D 盘） |

**API 代差（照 4.x 教程写必炸）**：`SFTConfig`/`DPOConfig` 只有 `warmup_steps`（无 `warmup_ratio`）；只有 `max_length`（无 `max_seq_length`、无 `max_prompt_length`）；`train_dataset` 必须是 `datasets.Dataset`；`DataCollatorForCompletionOnlyLM` 已删除，改用 `assistant_only_loss=True`；`compute_loss_func` 已标记 deprecated。

**显存的真实瓶颈是 loss，不是模型**：`transformers.loss.loss_utils.ForCausalLMLoss` 首行 `logits = logits.float()` 且不分块。全序列打标签时 b2×512 就吃到 11.38GB、b4×512 直接 OOM。走 `assistant_only_loss` 的真实路径后 b16×L384 最坏只有 9.02GB。**因此禁止用"labels=input_ids"的合成探针评估显存，必须走真实 trainer 路径。**

Windows/sm_120 约束：不装 flash-attn、不装 liger、不用 `torch.compile`（都依赖 triton）；`attn_implementation="sdpa"`；`dataloader_num_workers=0`；训练与推理不共进程。

---

## 2. 数据（实跑结果）

| 域 | train | dev | test | 检索池 | 备注 |
| --- | --- | --- | --- | --- | --- |
| clean | 1,800 | 200 | 251 | — | 原始 45,000 行去重后仅 **2,251 条唯一句对**；原 test 与 train 源句重叠 **2,490/2,500**，故降级为 sanity |
| MTNT | 32,593 | 852 | 1,020 | unseen 3,032 | 官方切分不动；从 train 剔除与 dev/test 有 8-gram 重叠的 260 条 |
| EMEA | 27,000 | 1,000 | 2,000 | memory_pool 6,000 | 1,092,568 行 → 质量过滤 259,778 → **8-gram 近重复过滤 143,056**（丢 45% 样板句）→ 连续 100 行整块切分 |

- 泄漏审计（test/dev vs train+全部检索池，句级精确 + 8-gram）：clean 0/0；MTNT test 0/0、dev 0/1；EMEA test 0/8（0.4%）、dev 0/0。**Gate G1 阈值 <0.5%，通过。**
- token 长度（Qwen tokenizer，含 prompt）：clean 最大 47；EMEA test 最大 321；MTNT test 最大 268；MTNT train p99=504，>384 占 1.73%。故 `max_length=384`：评测集零截断，训练集截断 1.73%（记录在案）。
- MTNT 保留原始噪声，只做编码修复与长度过滤，不做拼写/大小写归一。
- 全部产物带 sha256 manifest，run 配置引用 manifest 名。

---

## 3. 实验组与实测成本

| run | 配置 | 显存峰值 | 机器时间 |
| --- | --- | --- | --- |
| `g1_zs` | 零样本 greedy | 3.3GB | 5 min |
| `g1_zs_beam4` | 零样本 beam=4（解码敏感性） | ~5.5GB | 18 min |
| `g2_rag` | 零样本 + Dense RAG top-k | 5.6GB | 6 min |
| `g3_sft` | LoRA SFT（r=16 全线性，18.46M = **1.18%**） | **9.02GB reserved** | 训练 2.3h + 推理 5 min |
| `g4_sft_rag` | G3 + Dense RAG（prompt 与 G2 完全一致） | 5.6GB | 6 min |
| `g5_full` | DPO + 混合 RAG + 软偏置 + Agent + 路由 | 5.6GB(Pass1) / 2.4GB(Pass2) | Pass1 25 min + Pass2 10 min |
| 留一 4 臂 + 长度匹配对照 | `no_dpo` `no_rag` `no_rag_len` `no_con` `no_agent` | 5.6GB | 1.8h |
| `dpo_random_pairs` | 偏好对随机化重训（保护头条结论） | 8.7GB | 0.6h + 6 min |
| dev 机制对照 | k∈{1,3,5}、pool seen/unseen（**各 6,000，规模匹配**）、always-heavy | 5.6GB | 25 min |
| Shapley 16 子集 | dev 上精确 φ | 5.6GB | 1.3h（draft/QE 缓存复用） |

评测规模：test = MTNT 1,020 + EMEA 2,000（结论域）。clean 251 只做 sanity。dev = 852 + 1,000（选 k/τ/prompt，绝不进主表）。

生成吞吐实测（batch 32，greedy）：**22.2 句/s、1,422 tok/s**。batch 1 时 **21 ms/token**（框架开销主导，为 batch32 等效值的 6.8 倍）。因此延迟必须双口径报告：**逐句延迟（batch=1）** 用于 Pareto 与交互体验，**吞吐（batch=32）** 用于成本核算。

GPU/机器总时约 **13.5 小时**。人日约 **9.4**（完整）/ **3.8**（MVP）。

---

## 4. 训练配置（字段名已实测校验）

### 4.1 G3 SFT（`configs/g3_sft.yaml`）

```yaml
data: mtnt.train 32,593 + emea.train 27,000 + clean.train 1,800 = 61.4k，2 epoch
format: messages(system/user/assistant)
max_length: 384          truncation_mode: keep_end
packing: false           # 实测：短序列下打包使吞吐 17.6 → 6.6 样本/s
assistant_only_loss: true
per_device_train_batch_size: 16    gradient_accumulation_steps: 1
learning_rate: 2.0e-4    lr_scheduler_type: cosine    warmup_steps: 120
optim: adamw_torch       # 不用 bitsandbytes：sm_120 内核有风险，且 LoRA 优化器状态仅 0.22GB
bf16: true   gradient_checkpointing: true(use_reentrant=false)
dataloader_num_workers: 0   torch_empty_cache_steps: 100
eval_steps: 400   save_steps: 400   save_total_limit: 2   seed: 42
lora: r=16, alpha=32, dropout=0.05, target=all-linear
```

实测：15.3 样本/s（b16×L320 常规混合长度）、9.5 样本/s（b16×L384 全长最坏）→ 122.8k 样本 ≈ **2.3h**。
碎片化提示：`peak_alloc` 与 `peak_reserved` 差可达 4GB，必须设 `garbage_collection_threshold:0.85` + `torch_empty_cache_steps`。

### 4.2 G5 DPO（`configs/g5_dpo.yaml`）

```yaml
data: dpo.pairs.train（预期 15–18k），format: {prompt, chosen, rejected}
max_length: 384
beta: 0.1   loss_type: sigmoid   label_smoothing: 0.0
per_device_train_batch_size: 8   gradient_accumulation_steps: 2
learning_rate: 5.0e-6   cosine   warmup_steps: 100   epochs: 1
precompute_ref_log_probs: true   precompute_ref_batch_size: 8
bf16: true   gradient_checkpointing: true
```

关于 ref model 的准确机制：PEFT 下 `ref_model=None` 时 TRL 用**同一份基座权重关掉 adapter** 算 ref logprob，权重本来就只有一份，省不下 3.1GB。真正的收益来自 `precompute_ref_log_probs=True`：训练前一次性算完并缓存（实测 36 对/s，18k 对约 8 分钟），训练步内不再有 ref 前向。
实测：b8×L320 = 8.01GB alloc / 8.68GB reserved，10.2 对/s → 18k 对 ≈ **0.5h**。

### 4.3 偏好数据构造

20k 源句（MTNT 12k + EMEA 8k，与 test 域比例 **1,020:2,000 同向**）× K=4 候选（T=0.9, top_p=0.95）→ COMET-DA 排序（有参考）+ Kiwi 交叉验证（免参考）→ `chosen=argmax`、`rejected=argmin`，保留条件 Δ_DA≥0.10、文本不同、长度比∈[0.6,1.6]；另加 2,000 条术语剔除型难负例。
副产物分析：在 2k 子集上算 **分域 Spearman ρ(Kiwi, DA)** 与 **AUROC**（"draft 是否低于 DA 阈值"的二分类），作为路由信号可靠性的前置证据。
成本：生成 1.5h + 打分 20 min。

---

## 5. G5 管线（两遍式）

**Pass1 生成（主 venv，显存 5.6GB）**
- light：SFT adapter，greedy，`max_new_tokens=160`，batch 32
- heavy：DPO adapter + RRF 混合检索 top3（BM25 k3 + bge-m3 dense k3，RRF k=60，prompt 预算 320 token）+ 软偏置约束 + Critic → Refiner
- Guard：长度比 ∉[0.5,2.0] / 4-gram 重复率 >0.30 / langdetect≠fr → 回退 draft，计入 `fallback_rate`
- 逐句落盘：`draft, heavy, issues, refined, final_guarded` + 七个阶段的 CUDA event 耗时

**Pass2 打分（comet venv，显存 2.4GB）**：对 `draft` 与 `heavy_final` 同时打 Kiwi(QE) 与 COMET-DA，句级分数落盘。

**Pass3 路由合成（CPU）**：τ 作为离线过滤器在 8 点网格 `{0.50…0.85}` 上取值；
`latency(τ) = mean[t_draft + t_qe + 1{qe<τ}·(t_retr + t_heavy + t_critic + t_refiner)]`，分量全部来自 Pass1 实测；
选点规则**预注册**：`τ* = argmax_τ macro_COMET(τ) s.t. latency(τ) ≤ 1.15 × latency(G4)`，不可行则放宽至 1.30 并标注；完整 Pareto 8 点全报；τ* 处另跑一次真实串联管线做端到端校验。
τ 只在 dev（1,852 条）上选，test 只跑 τ*。

**约束解码**：软偏置为主 —— 源句命中术语时对法语术语首 token 加 `+2.0` logit，仅前 40% 生成步生效。硬 Trie 臂已删除（工期），报告中说明软偏置已覆盖"约束解码"这一因子。

---

## 6. 评测协议（`configs/eval.yaml`）

指标：COMET-DA（主）、CometKiwi（或替代信号）、BLEU(`tok=13a`)、chrF++、术语命中率（EMEA 200 + MTNT 100）、长度比、**重复率**、**退化率**（空输出/非法语/截断）、延迟 p50/p95、峰值显存。
新增两项零成本分析：**长度分桶**（源句 token 三分位 × 系统 × 域）与**重复退化率**，均复用句级分数。

统计：
- COMET 可句级分解 → 对句级分数做 10,000 次配对 bootstrap；MTNT 按 `comment_id` 分块重采样
- BLEU/chrF 不可句级分解 → **重采样句对后重算语料级指标**
- 配对 p 值 + Holm；比较族显式划分：main(4)、ablation(4) 各自校正，分域/分桶标 exploratory 不校正
- 预注册 **MDE = 0.008–0.012**（n=1,020 配对 COMET）；`|Δ| < MDE` 记 below resolution，与"证伪"分开表述

LLM 裁判（可选，工期紧张整块砍）：每域 300 条分层抽样、盲评 A/B、四维 1–5 分、30 条重复项测自一致性。后端优先 API，无 API 时用本机已缓存的 `unsloth/Qwen3.8-27B-GGUF`（llama.cpp，需披露量化等级）。**双人 κ/α 协议已删除**，改单标注 + 裁判模型复核 30 条并如实披露。

失败分析：G4/G5 差值最大样本各 30 条（MTNT）+ 30 条（EMEA），8 类错误标签，卡方检验 + 3 个完整案例。

---

## 7. 归因方法

**阶梯分解（叙事用）**
```
Δ_SFT      = v(G3) − v(G1)
Δ_RAG      = v(G2) − v(G1)
Δ_Synergy  = v(G4) − v(G3) − Δ_RAG
Δ_Advanced = v(G5) − v(G4)
恒等式：四项之和 = v(G5) − v(G1)   （已代数验证）
```
`v(·)` = **两域（MTNT/EMEA）等权宏平均 COMET-DA**。阶梯分解有顺序依赖，只用于讲增长曲线，**不与 Shapley 对账**。

**精确 Shapley（结论用）**
```
N = {DPO, RAG, CON, AGT}；v(∅) = 仅 SFT 基座
φ_i = Σ_{S⊆N\{i}} |S|!(|N|−|S|−1)!/|N|! · [v(S∪{i}) − v(S)]
效率公理：Σφ_i = v(full) − v(∅)      ← Gate G6 只校验这一条，误差应 <1%
```
16 个子集在 dev（1,852 条）上跑。draft 与 QE 分数跨子集缓存，heavy 生成组合上限 8 种（dpo×rag×con），Critic/Refiner 只在 agt=on 的 8 个子集跑 → 实测折算 **1.3h**。

**三个预期的可证伪判据**

| v2 文档预期 | 判据 | 若不成立 |
| --- | --- | --- |
| MTNT 上 DPO+Agent 主导 | φ_DPO + φ_AGT 在 MTNT 最大且 CI 下界 >0（或 >MDE） | 改写为检索主导，解释噪声域的检索效应；低于 MDE 则标 below resolution |
| EMEA 上 RAG+约束主导 | φ_RAG + φ_CON 在 EMEA 最大 | 讨论"术语已进参数"与检索池污染 |
| 路由以 ≤15% 延迟换 ≥90% 增益 | τ* 处 ΔCOMET_routed/ΔCOMET_heavy ≥0.9 且延迟增幅 ≤15% | 报完整 Pareto 与失败原因，作为负结果呈现 |

**循环性声明（限制章节必写）**：DPO 偏好对由 COMET-DA 构造，主评测指标又是 COMET-DA，存在"训练到指标"风险。对冲证据：chrF++、术语命中率、LLM 裁判分、Kiwi（若可用，与 DA 独立训练数据）。

---

## 8. 风险登记

### 8.1 QE 模型来源（已解决，留残留风险）

官方 `Unbabel/wmt22-cometkiwi-da` 是 `gated=auto`，无 token 时 401。当前采用**镜像 + 本地格式转换**：

- 源：`Supervache/wmt22-cometkiwi-da-safeformat`（非 gated，2.26GB safetensors，`class_identifier=unified_metric`、`pretrained_model=microsoft/infoxlm-large`、license `cc-by-nc-sa-4.0`，规格与官方一致）
- 转换：`src/eval/prepare_qe.py` 把 safetensors 包成 Lightning ckpt 布局（`<dir>/checkpoints/model.ckpt` + `<dir>/hparams.yaml`），产物在 `runs/qe_local/wmt22-cometkiwi-da-mirror/`
- 验证：加载为 `UnifiedMetric`；三句打分 好译 0.8913 > 俚语译 0.8716 > 坏译 0.6006；48 句 MTNT dev 上 **ρ(Kiwi, DA)=0.753（p<1e-4）**，远高于 0.4 的降级阈值

**残留风险与义务**：①镜像未与官方权重逐位校验 → 报告方法节与限制节必须写明来源与许可；②拿到 HF token 后重下官方权重，在 200 句上比对两套分数（预期 Pearson>0.99），结果进附录；③**Kiwi 值域压缩**（实测 [0.485, 0.886]，DA 为 [0.242, 0.991]），τ 网格必须按 Kiwi 自身分布标定，v2 文档默认的 τ=0.65 只能当起点。

**兜底（暂不启用）**：自建置信度信号 = draft 平均 token logprob + K=3 采样自一致性，用 COMET-DA 在完整 dev 上验 AUROC/Spearman。仅当完整 dev 上 ρ 跌破 0.4 时切换，届时延迟预算需按 draft×3 重算。

### 8.2 其他

| 风险 | 触发信号 | 处置 |
| --- | --- | --- |
| COMET venv 装不上 | pip 要求降级 torch | 锁 `unbabel-comet==2.2.x` + `--no-deps` 补依赖；再失败用 ONNX 版或 CPU 打分 |
| 显存碎片导致 reserved 触顶 | reserved 逼近 15GB | batch 16→8；`torch_empty_cache_steps=50`；仍不行则 `max_length` 384→320 |
| SFT 不涨分 | Gate G3 不过 | 查 label mask → 学习率 → 域混合比；两轮不过改 r=32 或 lr=3e-4 |
| G5 重路径不如 G4 | Δ<0 且 CI 跨 0 | 先修 Refiner prompt（最常见），两轮无改善则砍 Agent 模块，G5 重定义为 DPO+混合RAG+路由 |
| Kiwi 在 MTNT 不可信 | 分域 Spearman ρ<0.4 | 走 §8.1 第 3 条自建信号；报告如实披露 |
| MTNT test 太小 | CI 宽度 >0.02 | 主结论用两域宏平均 + 分域 CI；bootstrap 分布图代替单点 |
| 检索池污染对照 | seen/unseen 规模不等 | 两臂都取 6,000（已写入 `configs/ablations.yaml`） |
| 时间超支 | Day 5 未完 | 砍单顺序：LLM 裁判 → 第二 seed → Shapley 降为留一 →（`dpo_random_pairs` 保底不砍） |
| 单 seed 质疑 | 评审意见 | 时间富余则 G3/G5 各补 1 seed（+3.5h）；否则限制章节声明，用 bootstrap CI 覆盖 |
| 模型下载 ~8GB | 网络中断 | 已下 3/4；断点续传由 `snapshot_download` 处理 |

---

## 9. 阶段闸门

| Gate | 通过条件 | 不通过 |
| --- | --- | --- |
| G0 环境 | 探针全绿；torch/transformers 版本未被 pip 改动；SFT+DPO 冒烟跑通 | 修环境 |
| G1 数据 | test/dev 对 train+检索池的 8-gram 重叠 **<0.5%**（实测 EMEA test 0.4%）；截断率 <2% | 回炉清洗 |
| G2 基线 | G1 零样本在 EMEA 上 COMET-DA ∈ [0.72, 0.86] | 查 prompt 与语言方向 |
| G3 SFT | MTNT dev 上 COMET-DA 相对 G1 提升 **≥ +0.015** 且 bootstrap CI 下界 >0 | 禁止进入下游，先修训练 |
| G4 可复现 | 同配置重跑一次，COMET 差 <0.002 | 修非确定性来源（seed/padding/解码） |
| G5 路由 | τ* 满足"≤15% 延迟换 ≥90% 重路径增益" | 报 Pareto 与负结果 |
| G6 归因 | `Σφ_i` 与 `v(full) − v(∅)` 相对误差 **<1%** | 检查是否有子集用了不同 prompt/decode/数据版本 |

---

## 10. 排期（含人日）

| 日 | 白天（人日） | 夜间挂机（GPU） |
| --- | --- | --- |
| D1 | S0 环境 + S1 数据（0.9） | — |
| D2 | S2.1 推理脚本 + S1.3/S1.4 术语表与索引（1.0） | S3.1 SFT 2.3h |
| D3 | S4.1/S4.2 挖掘脚本（0.8） | S4.1 候选生成 1.5h → S4.3 DPO 0.6h |
| D4 | S5 G5 管线实现（1.5） | S5.4 dev Pass1/2/3 + τ 标定 0.5h |
| D5 | S6 主表 + S9 打分脚本（0.8） | S7 消融 2.7h |
| D6 | S9.2/S9.3 统计 + S10 分析（1.5） | S8 Shapley 1.3h |
| D7 | S11 报告（1.5） | 补跑 / 第二 seed |

**完整路线 ≈9.4 人日，单人一周偏紧**（审阅给 4/10，主要扣在墙钟而非 GPU）。**MVP 路线 3.8 人日**：S0→S1→S2→S3→S4（挖掘量减半至 10k）→S5→S6→S7 的 4 个留一→S9→S10.1→S11，已能交付主表 + 留一归因 + 路由 Pareto + 完整统计。加项按 `EXECUTION_CHECKLIST.md` 的顺序补。

---

## 11. 图表与报告映射

| 报告元素 | 来源 run | 说明 |
| --- | --- | --- |
| 表1 主结果 | S6 五组 × 两域 | BLEU/chrF++/COMET-DA/Kiwi + 95% CI |
| 图1 阶梯曲线 | 表1 | G1→G5 宏平均 COMET |
| 表2 阶梯分解 | §7 公式 | 标注"仅叙事用" |
| 表3 Shapley φ | S8 | 分域；含 Σφ 与 v(full)−v(∅) 的对账行 |
| 图2 分域主导因子 | 表3 | MTNT vs EMEA 堆叠柱 |
| 图3 Pareto | S5.4 τ 扫描 + G4 点 | 逐句延迟口径 |
| 图4 延迟阶段占比 | Pass1 阶段耗时 | 检索/初稿/QE/重路径/Critic/Refiner |
| 表4 消融 | S7 | 留一 + 长度匹配对照 + 随机偏好对 |
| 表5 术语/俚语命中 | S9.1 | G1/G3/G4/G5 |
| 表6 长度分桶 | S9.3 | 短/中/长 × 系统 × 域 |
| 表7 退化与重复率 | S9.1 | 全系统一列 |
| 图5 显存-质量 | 各 run `peak_vram.json` | 支撑"单卡可复现" |
| 表8 失败类型分布 | S10.2 | 卡方 + 3 案例 |
| 附录 | registry.csv / manifests / configs / logs | 含 git commit 与模型 revision |

---

## 12. 执行进度快照（2026-11-14）

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| S0 环境（依赖、探针、模型、COMET venv、QE 转换） | 完成 完成 | `proj/logs/probe_api.py` 输出、`probe_gpu.json`、`probe_train_mem.json`、`smoke_env.log`、`comet_venv.log`、`requirements-{main,comet}.lock` |
| S1 数据（三域构建、去重、泄漏审计、长度核定、检索索引） | 完成 完成（术语表 S1.3 待做） | `proj/data/processed/*.jsonl`、`data/manifests/*.json`、`reports/data_stats.json`、`logs/build_dataset.log`、`logs/build_index.log` |
| Pass1 生成链路（G1–G4 共用） | 完成 冒烟通过 | `proj/runs/_smoke_g1/`、`_smoke_g2/`（含 `meta.json`） |
| Pass2 打分链路（DA + Kiwi） | 完成 冒烟通过，G1 零样本 48 句 MTNT dev COMET-DA = **0.7541** | `proj/runs/_smoke_g1/scores.jsonl`、`scores_summary.json` |
| S3 之后（正式训练、G5 管线、消融、统计、报告） | 待做 待执行 | 见 `EXECUTION_CHECKLIST.md` |

已落地的代码：`src/common/{probe_api,probe_gpu,probe_train_mem,probe_hf,probe_qe,download_assets}.py`、`src/data/{build_dataset,build_index}.py`、`src/train/smoke_train.py`、`src/infer/generate.py`、`src/eval/{score_comet,prepare_qe}.py`。
待实现：`src/data/build_glossary.py`、`src/train/{sft_lora,dpo_lora,mine_dpo_pairs,merge_adapter}.py`、`src/infer/{pipeline_g5,router,constraints}.py`、`src/eval/{metrics,bootstrap,term_hit,length_buckets,shapley,make_figures}.py`。

---

## 附录 A：与 v2 设计稿的差异登记

| v2 设计稿 | 本方案 | 原因 |
| --- | --- | --- |
| 干净集作主评测 | 降级为 251 条 sanity | 唯一句对仅 2,251，原 test 泄漏 99.6% |
| CoT 显式规划 | **删除** | 1.5B 模型上 CoT 收益不稳，且与工期冲突（plan-v1 漏登记，本版补上） |
| 硬 Trie 约束解码 | 改软 logit 偏置，硬 Trie 臂删除 | 形态丰富语言上硬约束伤流畅性；工期 |
| CometKiwi 在线路由 | 两遍式离线 τ 过滤 | 消除 venv 隔离矛盾；τ 扫描零 GPU 成本 |
| 阶梯式归因为唯一方法 | 阶梯（叙事）+ 精确 Shapley（结论） | 阶梯差值有顺序依赖 |
| LoRA ~0.2% 参数 | 实测 1.18%（r=16 全线性，18.46M） | 与配置对齐；若要 0.2% 叙事需改 r=8+q/v/o 并另设对照臂 |
| 双人标注信度 | 单标注 + 裁判复核 30 条 | 单人无法履行第二标注人 |

## 附录 B：审阅可追溯性

独立审阅由 `reviewer` 子代理执行，**模型 `bailian-token-plan-personal/qwen3.8-max`，与主代理同家族 → 按项目约定标记 `same-family / provisional`**，不作为第三方独立证据。凡与本机实测冲突处一律以实测为准（冲突项与判定见 `FEASIBILITY_REVIEW.md` §2）。
