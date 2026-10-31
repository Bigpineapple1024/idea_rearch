# EXPERIMENT_RECORD：EXP-0043 的确认（条件性；新抽的 VQAv2 test7、GQA val400e 与 MME、POPE）

状态：计划 v1（2026-10-31 晚上 11 点后），在 EXP-0043 的任何结果之前写定（写时 EXP-0043 的 R01 正在运行，只看过运行号、状态与答案数）。只对 EXP-0043 里读作“有希望”、或“部分”且两集平均 ≥ +0.5 的设置运行；都不满足则不运行，本记录注明。

- 实验 ID：EXP-0045（EXP-0044 留给 VisionZip-64 加 fill 与省显存的实现，DEC-0083）。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)；开发实验 [EXP-0043](EXP-0043_record.md)；DEC-0083、DEC-0084。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题（每一部分只在 EXP-0043 的对应问题满足条件时运行）

1. **A：DivPrune-64**（EXP-0043 的 Q1）。**A1**：新抽的数据上 fill − DivPrune-64（VQAv2 test7、GQA val400e）。**A2**：MME、POPE 上 fill − DivPrune-64（逐题准确率）是否更差。
2. **B：CDPruner-128**（EXP-0043 的 Q2）：新抽的数据上 fill − CDPruner-128。
3. **C：CDPruner-192**（EXP-0043 的 Q3）：新抽的数据上 fill − CDPruner-192。
4. **同时报告（描述）**：不剪枝 − fill（新数据）；按题型、GQA 位置题；是非题按标准答案的答“是”比例；与不剪枝的回答相同的比例；MME 子任务分数；POPE 各分集的准确率、F1 与答“是”比例，POPE 上 fill − 不剪枝；填补的分配；每题秒数与峰值显存。

### 数据

- VQAv2 test7（1000 张 val2014 图、5229 题）与 GQA val400e（400 张 GQA val 图、5037 题）：make_confirm_sets4.py 在 2026-10-31 23:15 抽出（EXP-0043 的 R01 运行时、在它的任何结果之前），之后没有用过；脚本断言它们与已用的全部 VQAv2 子集（test、dev、test2、dev2、test3～test6）、GQA 子集（val400、val400b～val400d、testdev）不重合；sha256 见 vqav2_val/SHA256SUMS_test7、gqa/SHA256SUMS_val400e 与 gqa/images_val400e_SHA256SUMS。
- MME（2374 题）、POPE（8910 题）：本流程的标准版本。B、C 不在 MME、POPE 上跑（64 个时的推广已由 EXP-0042 检验；预算的问题看一般题目）。

### 配置与运行（按此顺序；只运行条件满足的部分；编号固定，不运行的部分编号空着）

| 运行 | 配置 | 数据 | 部分 |
| --- | --- | --- | --- |
| R01、R02 | `divprune64_fill_logits`、`divprune64` | vqav2_test7 | A |
| R03、R04 | 同上 | gqa_val400e | A |
| R05、R06 | `divprune64_fill_logits` | mme、pope（DivPrune-64 复用 EXP-0036-R02、R08） | A |
| R07、R08 | `cdpruner128_fill_logits`、`cdpruner128_logits` | vqav2_test7 | B |
| R09、R10 | 同上 | gqa_val400e | B |
| R11、R12 | `cdpruner192_fill_logits`、`cdpruner192_logits` | vqav2_test7 | C |
| R13、R14 | 同上 | gqa_val400e | C |
| R15、R16 | `vanilla` | vqav2_test7、gqa_val400e（任一部分运行时） | — |

- fill 与 EXP-0041～EXP-0043 的完全相同（vfill.py 不改）。复用：不剪枝 MME（EXP-0002-R05）、POPE（EXP-0002-R34）；DivPrune-64 MME、POPE（EXP-0036-R02、R08，之后 divprune.py 只多记一个属性，核对留下的 patch 相同）。失败的运行保留，用新编号（R17 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- A1、B、C：两个新集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- A2：MME、POPE 各自的 95% 区间整个在 0 以下 →“更差”，否则 →“不更差”（与 EXP-0042 相同）。另外预先写定一个次要读法（不进结论，因为 F112、P23）：POPE 上 fill − 不剪枝的区间整个在 0 以下 →“比不剪枝差”，否则 →“不比不剪枝差”。
- **结论**：A：A1“有希望”且 A2 两个都“不更差” →“确认”；A1“部分”且 A2 两个都“不更差” →“部分确认”；否则 →“未确认”。B、C：“有希望” →“在该预算上确认（一般题目）”；“部分” →“部分确认”；否则 →“未确认”。
- **之后**：记录，交用户决定（写成方法时的推广性证据）。
- 区间：每个 benchmark 内按图像重抽 1000 次（种子 0；新数据按 analyze_0028 的图像顺序，MME、POPE 按 analyze_0036）。

### 实现核对（门槛 ≥ 99%，analyze_0045.py，在看到 EXP-0045 的任何结果之前写好并用替身空跑）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；新数据上重算的得分等于计分文件与登记的准确率；MME 每个子任务的分数等于 metrics.json；POPE 第 k 题是第 k 个标注，每个分集的准确率与答“是”比例等于 metrics.json；fill 与底座留下的 patch 相同；fill 的布局日志每题一条（n 个图像 token、576 个槽、位置正确、32 层用了布局）。

### 代码（若运行：分支 code/token-pruning，在 EXP-0043 的代码之上）

- exp0002.py（登记 vqav2_test7、gqa_val400e）、score_pope_mme.py（让它们能计分）、make_confirm_sets4.py（原样提交，已于 23:15 运行过）、analyze_0045.py（新）。开门前另做 GPU 冒烟测试，另写 MANIFEST 与排队脚本。

### 预算

- 最多 16 次，约 5.5 GPU 小时；IDEA-0011 合计最多 28 + EXP-0044 + 16 次（上限 50 次、20 GPU 小时，DEC-0083）。超出时先停下复盘。
