# EXPERIMENT_RECORD：FastV 与随机选择保留了图像的哪些位置（F8 的机制诊断）

状态：计划 v1 已写定（2026-10-24，第一次运行之前），执行中。
实验 ID：EXP-0006 / IDEA ID：none（研究流程第 ⑤ 步：深挖意外现象 F8）/ 日期：2026-10-24 起 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-24，运行前写定）

- **来源**：SQZ-EXP-0002 的 P-SQZ2-1 与 M-SQZ2-1，MODE_C-0001 的 M1（RoPE 远程衰减使注意力打分偏向靠后的位置）。EXP-0002 中，同一层平均预算下随机选择在 POPE 三档都优于 FastV（F8）。
- **要回答的问题**：在 EXP-0002 的设置下（LLaVA-1.5-7B，K=2，keep 30/166），FastV 保留的图像 patch 落在什么位置；它是否集中在少数固定位置；同一张图换一个问题时保留集合是否改变。随机选择作为参照。
- **预先写下的预测**（只描述方向，不设显著性门槛；结果按实际记录）：
  - H1（位置偏差，M1）：FastV 保留 patch 的平均行号大于 11.5，底部 1/4（第 18～23 行）占比高于 25%；随机选择接近 11.5 与 25%。
  - H2（空间集中，M-SQZ2-1）：同一 keep 下，FastV 覆盖的 4×4 块比随机选择少。
  - H3（固定位置）：少数 patch 在大多数题目中都被保留，FastV 的归一化熵明显低于随机选择。
  - H4（是否依赖问题）：只报告同图不同题与不同图之间的 Jaccard 重叠，不预设方向。
- **数据**：pope_img20 子集，取 llava_pope_test.jsonl 中按出现顺序排列的 500 张图里每隔 25 张的 20 张，包含这些图在三个子集中的全部问题（`code/token_pruning/make_pope_subset.py`）：共 358 道（adversarial 120、popular 120、random 118），其中正样本 180 道（每张图 3 道正样本题，在三个子集中文字相同）。题目文件 `llava_pope_test_img20.jsonl` 的 sha256 为 f3ae79a4…，标签文件 `llava_pope_test_img20_labels.jsonl` 为 0dedbc18…。不打分；子集上的准确率与 yes 比例只作描述。
- **运行**（run ID 按执行顺序）：R01 fastv64_kept、R02 random64_s2021_kept、R03 fastv192_kept、R04 random192_s2021_kept，都在 pope_img20 上；贪心解码、4 个 DataLoader worker，与 EXP-0002 相同。
- **代码**：`run_eval.py --kept-token-log` 在每次预填充后读取 `fastv_llava.py` 已有的 `fastv_kept_image_positions`（被保留 patch 的行优先索引，0～575）；`exp0002.py` 的 `_kept` 配置传入该参数，实验目录按 run ID 前缀放在 `code/output/EXP-0006/`；分析脚本 `analyze_kept_tokens.py`。
- **分析**：每次运行计算平均行列、底部 1/4 占比、4×4 块覆盖率、最近邻距离、各 patch 被保留的频率与归一化熵、同图不同题与不同图的 Jaccard 重叠，以及正样本答对与答错时的块覆盖率。
- **解释规则**：
  - H1 与 H3 成立：支持“FastV 的打分主要由位置决定”，下一步检查打分去掉位置项后的保留分布（仍是诊断，不是方法）。
  - H2 成立而 H1、H3 不成立：保留集中在内容区域，下一步用 COCO 物体框检查保留 patch 与题目物体的重叠。
  - 都不成立：F8 的解释另找（例如 K=2 太早），记录为负结果。
- **预算与停止**：4 次运行，约 0.2 GPU 小时；运行失败时记录原因并用新 run ID 重跑。
- **留存**：同 EXP-0002，`code/output/EXP-0006/<RUN>/`，备份到 `D:\6CJNM007\backup\EXP-0006\<RUN>\`。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
