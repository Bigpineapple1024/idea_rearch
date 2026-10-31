# EXPERIMENT_RECORD：IDEA-0011 的推广（虚拟填补换底座、换预算，开发集）

状态：计划 v1（2026-10-31 晚上），在 EXP-0043 的任何运行之前写定。写计划时已看过 EXP-0041、EXP-0042 与 SQZ-EXP-0041、SQZ-EXP-0042 的全部结果。

- 实验 ID：EXP-0043。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)（MODE_C-0002 回流 31、33 的 C40）；用户决定 DEC-0083。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（换底座）**：DivPrune-64 加 fill − DivPrune-64（EXP-0032 的 DivPrune-64，复用）。
2. **Q2（预算 128）**：CDPruner-128 加 fill − CDPruner-128。
3. **Q3（预算 192）**：CDPruner-192 加 fill − CDPruner-192。
4. **同时报告（描述）**：不剪枝 − fill（三种设置）；按题型（VQAv2 答案类型，GQA 结构与语义类型，被路由与不路由的题）；GQA 位置题；是非题按标准答案分开的答“是”比例；与不剪枝的回答相同的比例（答案文字，SQZ-EXP-0042 S1 的做法）；第一个答案 token 的对数几率（三个 fill 运行与 CDPruner-128、CDPruner-192 都记录）：是非题上 gt yes 对 gt no 的 AUC，fill 与 CDPruner-128、CDPruner-192 各自比较；填补的分配；FLOPs（按结构、逐题，同 analyze_0041）；图像 KV 缓存（按本流程的实现：576 / n 倍）；每题秒数与峰值显存；增益随预算的变化（64 个时用 EXP-0041 的 fill − CDPruner-64 作对照，只描述）。

### 数据

- VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图），开发集（P14），与 EXP-0041 相同。

### 配置与运行（10 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `divprune64_fill_logits`（--method divprune --divprune-keep 64 --layout fill；已登记，EXP-0041 起） | vqav2_dev2、gqa_val400 |
| R03、R04 | `cdpruner128_logits`（--method cdpruner --cdpruner-keep 128） | 同上 |
| R05、R06 | `cdpruner128_fill_logits`（同上加 --layout fill） | 同上 |
| R07、R08 | `cdpruner192_logits`（--cdpruner-keep 192） | 同上 |
| R09、R10 | `cdpruner192_fill_logits`（同上加 --layout fill） | 同上 |

- fill 与 EXP-0041、EXP-0042 的完全相同（vfill.py 不改）：选择不变；每个剪掉的 patch 的槽在每一层放投影后特征余弦最大的保留 token 的 K、V，按槽的位置旋转；保留 token 与文字按不剪枝的编号、按位置因果地看 576 个槽。CDPruner-128、-192 是同一移植（cdpruner.py）留 128、192 个；“_logits”只多记第一个答案 token 的对数概率，不改计算。
- 复用：DivPrune-64（EXP-0032-R01、R02，代码 e1daaab；之后 divprune.py 只多记一个属性，不改计算，本实验核对留下的 patch 相同）、不剪枝（EXP-0033-R03、R04）。失败的运行保留，用新编号（R11 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1、Q2、Q3 各自：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- **之后**：“有希望”，或“部分”且两集平均 ≥ +0.5 的设置 → 在新抽的数据上确认（新抽 VQAv2 与 GQA 各一组，加 MME、POPE；POPE 同时报告对不剪枝的差与答“是”的比例，F112）；确认计划在看到 EXP-0043 的任何结果之前写定、提交，读法不改。其余设置记录为“在该底座 / 预算上没有检出”。三个都不满足 → fill 的增益只在 CDPruner-64 上，记录，交用户决定。
- 区间：每个集内按图像重抽 1000 次（种子 0，analyze_0028 的读法与计分）。

### 可检验的预测（推断，只为事后对照，不是判据）

- Q1：DivPrune 也是连续编号、丢布局（F89、F100），fill 应同样补回：VQAv2 +0.5～+2，GQA +1.5～+4，GQA 位置题 > +8。
- Q2、Q3：留得越多，剪掉的布局越少，增益变小：128 时 VQAv2 0～+1、GQA +1～+2.5；192 时 VQAv2 −0.5～+0.8、GQA +0.5～+2。
- 三种设置上“否”题答“是”的比例 fill 都不高于底座；fill 与不剪枝的回答相同的比例都高于底座；AUC 方向不确定。

### 实现核对（门槛 ≥ 99%，analyze_0043.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；选择日志每题一条，留 64 / 128 / 192 个；fill 留下的 patch 与对应底座相同（DivPrune：EXP-0032 的 div.json；CDPruner-128、-192：R03、R04、R07、R08 的 cdp.json）；布局日志是 fill、每题一条：35 个 token 之后 n 个图像 token、576 个槽、位置是预期的、32 层都用了布局、保留的槽指向自己；fill 与底座的提示相同；记录对数几率的运行每题一条第一个答案 token 的记录。
- CPU：vfill.py、cdpruner.py、divprune.py 都不改；仓库里重跑 check_vfill.py、check_cdpruner.py、check_divprune.py（全部通过）。
- GPU 冒烟测试（tmp，不是 EXP 运行；VQAv2 dev2 的前 8 题，五个新配置）：都正常结束；fill 与底座留下的 patch 相同；DivPrune-64 加 fill 留下的 patch 与 EXP-0032-R01 的前 8 题相同；布局日志正确。答案不计分、不看。
- analyze_0043.py 先用替身空跑（已封存的运行当替身），不是结果。

### 代码（分支 code/token-pruning，在 d6c6fdf 之上，`D:\6CJNM007\tmp\exp0043\code\`）

- exp0002.py（登记 cdpruner128、cdpruner192 与它们加 fill 的配置）、analyze_0043.py（新）。sha256 记在 `D:\6CJNM007\tmp\exp0043\MANIFEST`；排队脚本 `D:\6CJNM007\tmp\chain_0043.sh`（读完改动部分、写门文件之后才继续）。

### 预算

- 10 次，约 3.7 GPU 小时；IDEA-0011 合计 28 次、约 9.4 GPU 小时（上限 50 次、20 GPU 小时，DEC-0083）。
