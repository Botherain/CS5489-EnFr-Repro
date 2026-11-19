# 术语裁决复核（第二意见）

输入：`spotcheck.json`，是一批已经由另一个模型判过的条目，字段：

- `en`：英文术语
- `fr_claimed`：该模型给出的最终法语译法（若它判 reject，则为原候选译法，仅供参考）
- `claimed_decision`：keep / fix / reject
- `model_reason`：它的中文理由
- `examples`：EMEA 实证句对（英文原文 / 官方法语译文）

## 任务

对每一条判断"该模型是否判对"，不是重新做一遍裁决，而是**挑错**：

1. 看 `examples` 里这个英文短语在官方译文中实际怎么写。
2. 判断 `claimed_decision` + `fr_claimed` 是否站得住：
   - keep：原译法确实是该领域规范表达，且与例句一致
   - fix：改后的译法确实比原译法更规范，且与例句一致
   - reject：确实不该进术语表（品牌名、片段、通用词、说明书写法）
3. 找出这三类问题：**判错**（把术语丢掉 / 把非术语留下）、**译法不规范**（法语不合 EMEA 语域）、**证据不足**（例句里根本没有该写法，模型却在编）。

## 输出

JSON Lines，每条一行，顺序与输入一致：

{"en": "...", "agree": true, "problem": "none|wrong_decision|bad_french|no_evidence", "suggested": "", "note": "一句中文"}

- `agree`：是否同意该模型的判断
- `problem`：不同意时的主要问题类型
- `suggested`：若你认为应该改成别的法语译法，填在这里；否则留空
- `note`：一句话说明依据，最好引用例句里的原文

最后在回复里给：同意率（agree 为 true 的比例）、四类 problem 各多少条、以及你认为最不可靠的 3 条的 `en`。
