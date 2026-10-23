# EXPERIMENT_RECORD：去掉位置项后 FastV 的选择是否恢复（1a-4b 文档的 3a）

状态：计划 v1 已写定（2026-10-24，第一次运行之前），执行中。
实验 ID：EXP-0008 / IDEA ID：none（研究流程第 ⑤ 步：[1a-4b_FastV不如随机选择](../1a-4b_FastV不如随机选择.md) 的 3a 最小干预）/ 日期：2026-10-24 起 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-24，运行前写定）

- **来源**：EXP-0006 显示 FastV（K=2）的保留集合由位置主导（F15）。1a-4b 文档中“RoPE 衰减 → 分数随位置单调上升”这条边还没有直接证据，“位置偏差 → 召回下降”也只是推断。
- **要回答的问题**：
  1. 与内容无关时（空白图像），FastV 的分数在各 patch 位置上是什么形状；它能解释多少真实图像上的分数？
  2. 把真实图像的分数除以这条位置曲线后再选 token，保留分布是否变均匀，子集上的 yes 比例是否回升？
  3. 同样的校正用在完整 POPE 上，FastV-64 的 F1 与召回是否明显上升（对照 EXP-0002 的 FastV-64 36.19、随机-64 60.14）？
- **数据**：
  - pope_img20（与 EXP-0006 相同，358 道题）；
  - pope_img20_blank：同样的 358 道题，图像换成一张 336×336 的纯灰图（像素值 128），由 `make_blank_questions.py` 生成，提示词与 token 位置不变；
  - 完整 POPE（8910 题）与 MME（2374 题），与 EXP-0002 相同。
- **运行**（run ID 按执行顺序）：
  - R01 `fastv64_scores` × pope_img20_blank：记录空白图像上每道题的 576 个分数（分数在剪枝前计算，与 keep 无关）。
  - R02 `fastv64_scores` × pope_img20：真实图像上的分数。
  - 之后在 CPU 上运行 `analyze_score_prior.py EXP-0008 EXP-0008-R01 EXP-0008-R02`，把 R01 的平均分数存为 `code/output/EXP-0008/analysis/position_prior.npy`。
  - R03 `fastv64_prior_kept` × pope_img20、R04 `fastv192_prior_kept` × pope_img20：校正后的保留位置（用 `analyze_kept_tokens.py` 与 EXP-0006 同样的统计）。
  - R05 `fastv64_prior` × POPE、R06 `fastv64_prior` × MME。
- **预先写下的预测与解释规则**：
  - 若空白曲线随行号上升、底部 4 行占比明显高于 16.7%，且与真实图像的平均分数高度相关：支持“分数的位置成分与内容无关”（RoPE 衰减或 sink），1a-4b 第一条边由假设变为观察。
  - 若校正后平均行号回到 11.5 附近、块覆盖率接近随机，且子集 yes 比例回升：支持“位置偏差导致覆盖不足、进而答 no”。
  - 若 R05 的 POPE F1 明显高于 36.19（配对 bootstrap 区间不含 0）：位置偏差是 F8 的主要原因之一；若接近或超过随机-64 的 60.14，说明去掉位置项后注意力打分至少不比随机差。
  - 若校正后保留分布变均匀但 POPE 不升：覆盖不是充分条件，回到 2a（例如第 1 层的内容信号本身太弱）。
- **比较**：R05、R06 与 EXP-0002 的 FastV-64（R11、R10）、随机-64（R29、R28）做同一批题目上的配对 bootstrap（1000 次）。
- **预算与停止**：6 次运行，约 0.6 GPU 小时；空白曲线若全是 NaN 或常数（分数记录失败），停止并排查。
- **留存**：`code/output/EXP-0008/<RUN>/`，备份到 `D:\6CJNM007\backup\EXP-0008\<RUN>\`。
- **代码**：`fastv_llava.py` 可选的 `score_prior`（只用于诊断；不设置时计算与之前完全相同）；`run_eval.py --score-log`、`--fastv-score-prior`；`exp0002.py` 的 `_scores` 后缀、`fastv*_prior` 配置与 pope_img20_blank 基准；`make_blank_questions.py`、`analyze_score_prior.py`。另把 `exp0002.py` 写入 config.json 的 experiment 字段改为由 run ID 取得（EXP-0006 的四次运行中误写为 EXP-0002，目录与 runs.tsv 正确）。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
