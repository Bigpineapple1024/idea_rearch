# EXPERIMENT_RECORD：IDEA-0011 的确认（fill 对 CDPruner-64，新数据与 MME、POPE）

状态：计划 v1（2026-10-31 下午），在 EXP-0041 的任何结果之前写定（写时 EXP-0041 的 R01 正在运行，只看过运行号、状态与答案数）。只在 EXP-0041 的 Q1“有希望”，或“部分”且两集平均 ≥ +0.5 时运行；否则不运行，本记录注明。

- 实验 ID：EXP-0042。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)；开发实验 [EXP-0041](EXP-0041_record.md)；DEC-0080。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（新抽的数据）**：fill − CDPruner-64，在 VQAv2 test6 与 GQA val400d 上。
2. **Q2（推广，IDEA-0008 的教训）**：fill − CDPruner-64，在 MME 与 POPE 上（逐题准确率）。
3. **同时报告（描述）**：按题型（VQAv2 答案类型，GQA 结构与语义类型，被路由与不路由的题）；GQA 位置题；是非题按标准答案分开的答“是”比例（新数据与 POPE、MME）；MME 子任务分数；POPE 各分集的准确率、F1 与答“是”比例；不剪枝 − fill；填补的分配；每题秒数与峰值显存。

### 数据

- VQAv2 test6（1000 张 val2014 图、5057 题）与 GQA val400d（400 张 GQA val 图、5397 题）：make_confirm_sets3.py 在 2026-10-31 12:07 为 IDEA-0010 的确认抽出，在 EXP-0040 的任何结果之前，之后从未用过；脚本断言它们与已用的全部 VQAv2 子集（test、dev、test2、dev2、test3、test4、test5）、GQA 子集（val400、val400b、val400c、testdev）不重合；sha256 见 vqav2_val/SHA256SUMS_test6、gqa/SHA256SUMS_val400d 与 gqa/images_val400d_SHA256SUMS。
- MME（2374 题）、POPE（8910 题）：本流程的标准版本（EXP-0002 起）。

### 配置与运行（10 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `cdpruner64_fill_logits`、`cdpruner64` | vqav2_test6 |
| R03、R04 | 同上 | gqa_val400d |
| R05、R06 | 同上 | mme |
| R07、R08 | 同上 | pope |
| R09、R10 | `vanilla` | vqav2_test6、gqa_val400d |

- fill 与 EXP-0041 的完全相同（vfill.py，代码 33d3472）。复用：不剪枝 MME（EXP-0002-R05）、POPE（EXP-0002-R34）。失败的运行保留，用新编号（R11 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- Q2：MME、POPE 各自的 95% 区间整个在 0 以下 →“更差”，否则 →“不更差”。
- **结论**：Q1“有希望”且 Q2 两个都“不更差” →“确认”；Q1“部分”且 Q2 两个都“不更差” →“部分确认”；否则 →“未确认”。
- **之后**：“确认” → IDEA-0011 作为论文方法的候选，是否在别的底座（VisionZip、DivPrune）、别的模型上检验，以及只存 64 个代表、解码时现算的实现，交用户决定；“部分确认”或“未确认” → 记录，交用户决定。
- 区间：每个 benchmark 内按图像重抽 1000 次（种子 0；新数据按 analyze_0028 的图像顺序，与开发集的分析相同；MME、POPE 按 analyze_0036）。

### 实现核对（门槛 ≥ 99%，analyze_0042.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；新数据上重算的得分等于计分文件与登记的准确率；MME 每个子任务的分数（acc + acc+）等于 metrics.json；POPE 第 k 题是第 k 个标注，每个分集的准确率与答“是”比例等于 metrics.json；fill 与 CDPruner-64 留下的 patch 相同（≥ 99%）；fill 的布局日志每题一条（64 个图像 token、576 个槽、位置正确、32 层用了布局）。
- analyze_0042.py 已用替身空跑（开发集上的 EXP-0039、EXP-0040 运行与 EXP-0036 的 MME、POPE 运行当替身）：开发集上的区间与 EXP-0040 封存的逐位相同（−0.98 [−1.49, −0.45] / −0.81 [−1.41, −0.17]），MME、POPE 上复现 F94、F95 的 −1.52、−1.84（替身不是结果）。

### 代码（若运行：分支 code/token-pruning，在 33d3472 之上，`D:\6CJNM007\tmp\exp0042\code\`，由 make_code_0042.py 生成）

- exp0002.py（登记 vqav2_test6、gqa_val400d）、score_pope_mme.py（让它们能计分）、make_confirm_sets3.py（原样提交，已于 12:07 运行过）、analyze_0042.py（新）。CDPruner 在 MME、POPE 与这两个新集上是第一次在本流程中运行；开门前另做 GPU 冒烟测试，另写 MANIFEST 与排队脚本。

### 预算

- 10 次，约 3 GPU 小时；IDEA-0011 合计 18 次（EXP-0041 的 8 次加这 10 次），在 30 次上限之内。
