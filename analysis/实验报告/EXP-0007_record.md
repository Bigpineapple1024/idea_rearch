# EXPERIMENT_RECORD：把层分布与选择规则拆开（C11）

状态：计划 v1 已写定（2026-10-24，第一次运行之前），执行中。
实验 ID：EXP-0007 / IDEA ID：none（研究流程第 ④⑤ 步：MODE_C-0002 的 C11）/ 日期：2026-10-24 起 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-24，运行前写定）

- **来源**：MODE_C-0002 的 C11 与 E-MC2-2。EXP-0002 中，层平均都在 64 附近时：FastV（576×2、30×30）POPE 36.19，同分布的随机选择 60.14，SparseVLM（576×3、82×4、32×9、18×16，层平均 82.25）77.67，VisionZip（每层 64）77.03。SparseVLM 与 FastV 的差别同时来自选择规则和层分布，EXP-0002 无法拆开。
- **要回答的问题**：
  1. 在 SparseVLM 实测的逐层 token 数下，随机选择能否接近 SparseVLM 本身（即 SparseVLM 的优势有多少来自层分布、多少来自文本评分者与回收）。
  2. 在同一分布下，FastV 的打分（上一层最后一个 token 的注意力）是否仍不如随机。
  3. 层平均相同时，“前 3 层 576、之后逐段减少”的分布是否优于“前 2 层 576、之后平坦”的分布。
- **配置**（都是 LLaVA-1.5-7B、贪心解码、4 个 DataLoader worker；随机选择用种子 2021）：
  - SVP（SparseVLM 实测分布）64：在第 3、7、16 层前分别剪到 82、32、18 个，按 EXP-0002-R49 的实测值；`fastv_svp64`（FastV 打分）与 `random_svp64_s2021`（随机）。
  - 平坦分布、层平均相同：K=2、保留 49 个，层平均 81.9（SVP 为 82.25）；`fastv_flat64` 与 `random_flat64_s2021`。
  - 对照取自 EXP-0002：SparseVLM-64（R40 MME、R41 POPE）、FastV-64、随机-64。
- **运行**（run ID 按执行顺序）：
  - R01、R02：`random_svp64_s2021_layers`、`fastv_svp64_layers` × pope50，核对逐层 token 数确为 576×3、82×4、32×9、18×16。核对不通过就停止并排查，不跑后面的评测。
  - R03～R06：上述 4 个配置 × POPE；R07～R10：同 4 个配置 × MME。
- **比较**（同一批题目上的配对 bootstrap，1000 次，与 EXP-0002 相同）：
  - Q1：random_svp64 − SparseVLM-64；
  - Q2：fastv_svp64 − random_svp64；
  - Q3：random_svp64 − random_flat64，fastv_svp64 − fastv_flat64。
- **预先写下的预测与解释规则**（方向，不设门槛）：
  - 若 random_svp64 与 SparseVLM-64 的 POPE 差的区间跨 0：SparseVLM 在 POPE 上的优势主要来自层分布；若明显低于 SparseVLM：文本评分者或回收确有作用。
  - 若 fastv_svp64 仍明显低于 random_svp64：FastV 的打分问题不依赖层分布（支持 C9 的方向）。
  - 若 random_svp64 明显高于 random_flat64：前几层保留更多 token 本身有收益（支持 M4）。
- **预算与停止**：10 次运行，约 2 GPU 小时（POPE 约 20～25 分钟/次，MME 约 6～7 分钟/次）；失败用新 run ID 重跑。
- **留存**：`code/output/EXP-0007/<RUN>/`，备份到 `D:\6CJNM007\backup\EXP-0007\<RUN>\`。
- **代码**：`fastv_llava.py` 支持多阶段时间表（单阶段时的计算与随机抽取和以前完全相同）；`run_eval.py --fastv-schedule`；`exp0002.py` 的 `fastv_svp*`、`random_svp*`、`fastv_flat*`、`random_flat*` 配置。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
