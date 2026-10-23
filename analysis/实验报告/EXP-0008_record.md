# EXPERIMENT_RECORD：去掉位置项后 FastV 的选择是否恢复（1a-4b 文档的 3a）

状态：completed（2026-10-24）。6 次运行全部完成，计划内的比较已做完；结果按预先写下的规则判断，见“结果”和“观察与解释”。
实验 ID：EXP-0008 / IDEA ID：none（研究流程第 ⑤ 步：[1a-4b_FastV不如随机选择](../1a-4b_FastV不如随机选择.md) 的 3a 最小干预）/ 日期：2026-10-24 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

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
| EXP-0008-R01 | fastv64_scores × pope_img20_blank | 10-24 06:17 / 06:18 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --score-log D:/6CJNM007/idea_rearch/code/output/EXP-0008/EXP-0008-R01/scores.npy；主指标 358 | pope_img20_blank 358 题（纯灰图） | `code/output/EXP-0008/EXP-0008-R01/`，SHA256SUMS 的 sha256 aa6291c6…；备份同名目录 | 主指标一栏是记录了分数的预填充次数（不计分） |
| EXP-0008-R02 | fastv64_scores × pope_img20 | 10-24 06:18 / 06:20 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --score-log D:/6CJNM007/idea_rearch/code/output/EXP-0008/EXP-0008-R02/scores.npy；主指标 358 | pope_img20 358 题 | `code/output/EXP-0008/EXP-0008-R02/`，SHA256SUMS 的 sha256 defdddc3…；备份同名目录 | 同上 |
| EXP-0008-R03 | fastv64_prior_kept × pope_img20 | 10-24 06:20 / 06:21 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy --kept-token-log；主指标 358 | pope_img20 358 题 | `code/output/EXP-0008/EXP-0008-R03/`，SHA256SUMS 的 sha256 e55814df…；备份同名目录 | 主指标一栏是记录了保留位置的预填充次数 |
| EXP-0008-R04 | fastv192_prior_kept × pope_img20 | 10-24 06:21 / 06:23 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 166 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy --kept-token-log；主指标 358 | pope_img20 358 题 | `code/output/EXP-0008/EXP-0008-R04/`，SHA256SUMS 的 sha256 d78bc437…；备份同名目录 | 同上 |
| EXP-0008-R05 | fastv64_prior × pope | 10-24 06:23 / 06:44 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy；主指标 57.41283792328229 | POPE 8910 题 | `code/output/EXP-0008/EXP-0008-R05/`，SHA256SUMS 的 sha256 e14647d4…；备份同名目录 |  |
| EXP-0008-R06 | fastv64_prior × mme | 10-24 06:44 / 06:50 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy；主指标 1184.923369347739 | MME 2374 题 | `code/output/EXP-0008/EXP-0008-R06/`，SHA256SUMS 的 sha256 c2675305…；备份同名目录 |  |

- 六次运行的 config.json 的 experiment 字段均为 EXP-0008（字段修正生效）。
- 运行之间的 CPU 分析：`analyze_score_prior.py`（R02 之后，写 position_prior.npy、score_curve.md/json）、`analyze_kept_tokens.py EXP-0008 EXP-0008-R03 EXP-0008-R04`（R06 之后，写 kept_stats.md、kept_summary.json）；配对 bootstrap 由 `bootstrap_ci.py` 在 R06 之后计算（analysis/paired_bootstrap.txt，调用脚本 compare_0008_0007.sh）；MME 子任务表 mme_subtasks.md（mme_sub_0008.py）。analysis 目录已封存：SHA256SUMS 的 sha256 a3be9b26…，附上述脚本副本，备份到 `D:\6CJNM007\backup\EXP-0008\analysis\` 并核对。

## 数据与证据血缘

- 空白图像：`make_blank_questions.py` 生成 blank_gray.png（sha256 9863ae9e…）与题目文件（sha256 27a3f761…），题目与 pope_img20 逐条相同、只换图像。
- 位置先验：R01 的 358 次预填充、576 个位置的分数取平均，存为 position_prior.npy；R03～R06 都用这一个文件。校正为“分数 ÷ 先验”后取 top-k，只在 FastV 选 token 的那一步生效。
- 评测、计分与 EXP-0002 相同（POPE 三个 split 的 F1 平均；MME Perception 总分，官方计分工具）。

## 结果

| 结果 | 条件 | 对照 / 基线 | 差异 / 效应 | 不确定性 | 意义 | 证据位置 |
| --- | --- | --- | --- | --- | --- | --- |
| 空白图像上的分数形状 | R01，第 1 层、最后一个 token、head 平均 | 均匀分布 | 底部 4 行占图像注意力的 31.7%（均匀 16.7%）；按行相对均匀的倍数从顶部的 0.48～1.00 升到第 22 行的 2.56；最后一个 patch 占 0.70%（均匀 0.17%） | 358 次预填充同一张图、不同问题 | 分数有一条与内容无关的位置曲线 | analysis/score_curve.md |
| 真实图像分数与空白曲线的关系 | R02 | R01 | 真实图像的行剖面较平（0.64～1.70）；576 个位置上与空白曲线的 Pearson r：平均分数 0.329，逐题中位数 0.138 | — | 位置曲线存在，但只解释真实分数的一小部分 | analysis/score_curve.md |
| 校正后保留的位置（keep 30） | R03 | EXP-0006-R01（FastV）、R02（随机） | 平均行 10.83（FastV 15.92，随机 11.53）；底部 1/4 占比 19.0%（57.3%，24.8%）；4×4 块覆盖率 50.4%（45.4%，57.7%）；同图不同题 Jaccard 0.957（0.948，0.026）；子集准确率 / yes 比例 67.0% / 17.9%（56.4% / 6.7%，70.4% / 22.3%） | 单一子集 | 行分布回到中部，覆盖与 yes 比例部分恢复；选择仍与问题无关 | analysis/kept_stats.md |
| 校正后保留的位置（keep 166） | R04 | EXP-0006-R03、R04 | 平均行 10.49（16.56，11.50）；底部占比 11.9%（56.0%，24.8%）；块覆盖率 94.3%（85.4%，99.6%）；Jaccard 0.955；子集准确率 / yes 比例 88.3% / 40.2%（79.3% / 32.4%，83.2% / 37.4%） | 单一子集，358 题 | 子集上已高于随机 5.1 个百分点；需完整 POPE 确认 | analysis/kept_stats.md |
| POPE F1（64 档） | R05 | FastV-64（EXP-0002-R11，36.19） | **+21.22**（57.41） | 配对 bootstrap 95% 区间 [19.61, 22.81] | 位置偏差是 F8 的主要原因之一 | analysis/paired_bootstrap.txt |
| POPE F1（64 档） | R05 | 随机-64 种子 2021（R29，60.26） | −2.85 | [−4.40, −1.25] | 校正后仍略低于随机 | 同上 |
| POPE F1（64 档） | R05 | VisionZip-64（R17，77.03） | −19.61 | [−20.88, −18.38] | 与编码器侧方法的差距仍大 | 同上 |
| POPE 召回 / yes 比例 | R05 | FastV-64、随机-64 | 召回 40.7%（FastV 22.3%，随机 43.5%～44.7%）；yes 比例 20.6%～21.3%（11.4%～11.9%，23.1%～24.2%）；精确率 95.6%～98.7% | 三个 split 的正样本相同，TP 都是 610 | 召回回升是 F1 上升的来源 | EXP-0008-R05/metrics.json |
| MME-P（64 档） | R06 | FastV-64（R10，979.6） | **+205.3**（1184.9） | [141.4, 274.8] | — | analysis/paired_bootstrap.txt |
| MME-P（64 档） | R06 | 随机-64 种子 2021（R28，1168.2） | +16.8 | [−74.5, 103.9] | 分辨不出差异 | 同上 |
| MME-P（64 档） | R06 | VisionZip-64（R35，1364.2） | −179.3 | [−236.8, −120.0] | — | 同上 |
| MME 子任务 | R06 | R10、R28 | 10 个子任务全部高于 FastV（+2.0～+38.3）；与随机相比，existence −30.0、count −23.3、scene −13.8，celebrity +38.5、OCR +27.5，其余相差 ≤ 8.8 | OCR 40 题，其余每项 60～400 题；未做子任务区间 | F11 的交互仍在：注意力保住细节，随机保住存在与计数 | 由各运行 metrics.json 的 perception_tasks 计算 |
| 资源 | R05、R06 | FastV-64（R11、R10） | 每题 0.142 / 0.152 s，峰值显存 13.94 / 13.96 GiB | — | 校正不增加成本 | runs.tsv |

按预先写下的规则判断：

1. 空白曲线随行号上升、底部占比 31.7% 远高于 16.7%，但与真实图像分数的相关只有 0.33（逐题中位数 0.14），不算“高度相关”。**部分支持**：位置成分存在且与内容无关，第一条边从假设变为观察；但真实分数并不主要由这条曲线决定。
2. 校正后平均行回到 10.5～10.8、底部占比低于均匀，块覆盖率向随机靠近（50.4% vs 57.7%，94.3% vs 99.6%），子集 yes 比例回升（6.7% → 17.9%，32.4% → 40.2%）。**支持**“位置偏差导致覆盖不足、进而答 no”。
3. R05 比 FastV-64 高 21.2（区间不含 0），**支持**“位置偏差是 F8 的主要原因之一”：它补回了 FastV 与随机（种子 2021）之间 24.07 分差距的 88%。但 R05 仍比随机低 2.85（区间不含 0），“去掉位置项后注意力打分至少不比随机差”在 POPE 上**不成立**；在 MME 上分辨不出差异。
4. 不适用（POPE 上升了）。

## 观察与解释

| 类型 | 观察 | 当前解释 | 待复验 / 替代解释 | 证据位置 |
| --- | --- | --- | --- | --- |
| 主要观察 | 除以空白图像的位置曲线后，FastV-64 的 POPE 从 36.2 升到 57.4、MME-P 从 980 升到 1185，接近随机 | FastV 不如随机的大部分原因是打分中的位置成分 | 校正也可能只是让选择“更分散”，与位置无关的分散化（例如按块均匀抽取最高分）可能有同样效果 | analysis/paired_bootstrap.txt |
| 意外现象 | 校正后的选择仍与问题无关：同图不同题的 Jaccard 0.957，最常被保留的是图像中部的固定 patch（(12, 11) 在 keep 166 时 100% 被保留） | 第 1 层最后一个 token 的注意力去掉位置项后，剩下的主要是图像本身的显著性，而不是题目相关性（推断） | 更深的层（K 更大）或以全部文本 token 为 query 时，是否开始看问题 | analysis/kept_stats.md |
| 意外现象 | keep 166 时，校正后的 FastV 在子集上比随机高 5.1 个百分点（88.3% vs 83.2%），而 keep 30 时仍低 3.4 | 名额多时，显著性打分能保住细节又不丢覆盖；名额少时覆盖不足的损失更大（推断） | 只是 358 题的子集；需要完整 POPE/MME 上的 fastv192_prior、fastv128_prior | analysis/kept_stats.md |
| 意外现象 | 空白曲线与真实分数的相关只有 0.33，真实图像的行剖面比空白平 | 真实图像上内容信号与位置信号叠加，位置成分被稀释；或者位置成分本身随内容变化（例如与 sink 的强弱有关） | 用更多空白图像（不同灰度、噪声图）检查曲线是否稳定 | analysis/score_curve.md |
| 技术 / 质量 | 六次运行 commit 相同、无未提交改动；分数记录无 NaN；校正只改 top-k 的输入，资源不变 | — | — | 各运行 git.txt、status.json |

## 偏差与限制

- 计划与实际：无偏差。规则 1 的“高度相关”没有事先给出数值门槛，这里按 0.33 判为“不高”。
- 位置先验来自一张纯灰图、358 道题；只有一种灰度，没有检查先验对图像选择的敏感性。
- 随机对照只用种子 2021 做配对（EXP-0002 的三个随机种子 POPE 为 60.26、60.38、59.77，标准差 0.32），结论不因种子改变。
- 子集结论（R03、R04）只有 358 题，置信区间宽；keep 166 的完整评测没有做。
- MME 子任务差异没有区间。

## 结论边界与下一步

- 当前证据支持：FastV（K=2）的打分里有一条与内容无关的位置曲线；把它除掉，FastV-64 在 POPE 上补回与随机差距的约 88%，在 MME 上与随机持平。位置偏差是 F8 的主要原因之一。
- 当前证据不能支持：校正后的注意力打分优于随机（POPE 上仍低 2.85）；位置曲线来自 RoPE（本实验没有区分 RoPE 与 sink）；校正后的选择“看问题”（Jaccard 0.957 说明仍不看）。
- 下一项最低成本检查：完整 POPE、MME 上的 fastv192_prior 与 fastv128_prior（4 次运行，约 55 分钟），看名额多时校正后的 FastV 是否超过随机。
- 后续路由：squeeze-experiment（SQZ-EXP-0008）→ deep-read Mode C 回流（MODE_C-0002 回流 2）→ 第 ⑦ 步排序。
- 下一 skill：squeeze-experiment
