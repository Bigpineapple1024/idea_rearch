# EXPERIMENT_RECORD：位置校正的收益来自“去位置”还是“更分散”（对 FEATHER 两个部件的独立复核）

状态：completed（2026-10-24）。6 次运行全部完成，计划内的比较已做完；结果按预先写下的规则判断，见“结果”和“观察与解释”。
实验 ID：EXP-0010 / IDEA ID：none（研究流程第 ⑤ 步：[1a-4b_FastV不如随机选择](../1a-4b_FastV不如随机选择.md) 的 3b；SQZ-EXP-0008 的 P-SQZ8-3）/ 日期：2026-10-24 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-24，运行前写定）

- **来源**：EXP-0008 中，把 FastV（K=2）的分数除以空白图像的位置曲线后，FastV-64 的 POPE 从 36.19 升到 57.41，仍比随机-64 低 2.85。收益来自“去掉位置成分”还是“保留集合更分散”没有分开。
- **已有工作（检索摘要级，SC）**：FEATHER（Endo 等，ICCV 2025，arXiv 2412.13180）报告 FastV 式剪枝丢掉图像上部的 token，提出去掉底部偏置、加均匀采样保证覆盖、两阶段剪枝，消融显示去偏置与均匀采样都改进选择。所以本实验不是新方法，而是在本项目的设置（K=2、层平均 64、完整 POPE/MME、配对区间）下独立复核这两个部件，并把它们分开。
- **要回答的问题**：
  1. 不去位置、只强制覆盖（把排序分到 6×6 个 4×4 patch 的块里轮流取，`--fastv-stratify 6`），FastV-64 能恢复多少？
  2. 去位置 + 强制覆盖，FastV-64 能否超过随机-64？
- **运行**（run ID 按执行顺序；FastV 均为 K=2、keep 30）：
  - R01 `fastv64_strat_kept` × pope_img20、R02 `fastv64_prior_strat_kept` × pope_img20：保留位置统计（`analyze_kept_tokens.py`，与 EXP-0006、EXP-0008 同口径）。
  - R03 `fastv64_strat` × POPE、R04 × MME；R05 `fastv64_prior_strat` × POPE、R06 × MME。
- **比较**（同一批题目上的配对 bootstrap，1000 次）：R03、R05 与 FastV-64（EXP-0002-R11）、随机-64 种子 2021（EXP-0002-R29）、只去位置的 FastV-64（EXP-0008-R05）比较；R04、R06 与 EXP-0002-R10、R28、EXP-0008-R06 比较。
- **预先写下的解释规则**：
  - 若 R03（只覆盖）≥ EXP-0008-R05（只去位置）− 1 分（POPE）：收益主要来自“更分散”，位置校正本身不是必要的。
  - 若 R03 明显低于 EXP-0008-R05（区间不含 0）：去掉位置成分有超出覆盖的作用。
  - 若 R05（去位置 + 覆盖）高于随机-64 且区间不含 0：在 K=2 时，第 1 层注意力去掉位置后有可用的内容信号，与 FEATHER 的结论方向一致；否则说明在这么浅的层上，注意力打分加上覆盖也不比随机好（FEATHER 也报告浅层的注意力打分较弱）。
- **预算与停止**：6 次运行（2 次子集诊断，4 次完整评测），约 0.9 GPU 小时。R01 的块覆盖率若不等于 30/36（keep 30 最多覆盖 30 块），说明分块实现有误，停止排查。
- **未做**：192、128 档的位置校正完整评测（SQZ-EXP-0008 的 P-SQZ8-2）。在 FEATHER 已报告同类修正的情况下，这一项对第 ⑦ 步的判断影响小，暂不运行。
- **留存**：`code/output/EXP-0010/<RUN>/`，备份到 `D:\6CJNM007\backup\EXP-0010\<RUN>\`。
- **代码**：`fastv_llava.py` 的 `stratify` 选项（只用于诊断；不设置时计算与之前完全相同），`run_eval.py --fastv-stratify`，`exp0002.py` 的 `fastv{b}_strat`、`fastv{b}_prior_strat`。CPU 单元测试（合成分数、随行号上升）：keep 30/98/166 时分别覆盖 30/36/36 个块，每块 0～1 / 2～3 / 4～5 个，块内取的都是该块分数最高的 patch。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0010-R01 | fastv64_strat_kept × pope_img20 | 10-24 08:42 / 08:44 | completed（评测退出码 0） | commit 399b2d0，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-stratify 6 --kept-token-log；主指标 358 | pope_img20 358 题 | `code/output/EXP-0010/EXP-0010-R01/`，SHA256SUMS 的 sha256 52c9e936…；备份同名目录 | 主指标一栏是记录了保留位置的预填充次数；块覆盖率核对通过 |
| EXP-0010-R02 | fastv64_prior_strat_kept × pope_img20 | 10-24 08:44 / 08:45 | completed（评测退出码 0） | commit 399b2d0，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy --fastv-stratify 6 --kept-token-log；主指标 358 | pope_img20 358 题 | `code/output/EXP-0010/EXP-0010-R02/`，SHA256SUMS 的 sha256 577875fe…；备份同名目录 | 同上 |
| EXP-0010-R03 | fastv64_strat × pope | 10-24 08:45 / 09:07 | completed（评测退出码 0） | commit 399b2d0，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-stratify 6；主指标 61.47374426684411 | POPE 8910 题 | `code/output/EXP-0010/EXP-0010-R03/`，SHA256SUMS 的 sha256 7f1ad59b…；备份同名目录 |  |
| EXP-0010-R04 | fastv64_strat × mme | 10-24 09:07 / 09:13 | completed（评测退出码 0） | commit 399b2d0，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-stratify 6；主指标 1280.7306922769108 | MME 2374 题 | `code/output/EXP-0010/EXP-0010-R04/`，SHA256SUMS 的 sha256 83d1e896…；备份同名目录 |  |
| EXP-0010-R05 | fastv64_prior_strat × pope | 10-24 09:13 / 09:33 | completed（评测退出码 0） | commit 399b2d0，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy --fastv-stratify 6；主指标 64.66371522585975 | POPE 8910 题 | `code/output/EXP-0010/EXP-0010-R05/`，SHA256SUMS 的 sha256 5568f47c…；备份同名目录 |  |
| EXP-0010-R06 | fastv64_prior_strat × mme | 10-24 09:34 / 09:40 | completed（评测退出码 0） | commit 399b2d0，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 30 --fastv-score-prior D:/6CJNM007/idea_rearch/code/output/EXP-0008/analysis/position_prior.npy --fastv-stratify 6；主指标 1253.3211284513804 | MME 2374 题 | `code/output/EXP-0010/EXP-0010-R06/`，SHA256SUMS 的 sha256 5e53e37c…；备份同名目录 |  |

- 六次运行的 config.json 的 experiment 字段都是 EXP-0010。
- 停止条件：R01、R02 之后，批处理脚本（run_vision_0010.sh）从 kept_summary.json 核对 keep 与块覆盖率，两次都是 keep 30、覆盖率 0.8333（= 30/36），之后才开始 R03。
- 运行后的 CPU 分析：
  - 配对 bootstrap：`bootstrap_ci.py`，由 compare_0010.sh 调用，结果在 analysis/paired_bootstrap.txt。
  - POPE 分 split 统计、漏答正样本、MME 子任务：`analyze_0010_details.py`，输出 details.md、details.json。
  - MME 的 yes 比例并入 EXP-0002/deepdive2（`analyze_mme_yes_ratio.py`，第 ⑦ 步前的普遍性检查，F23）。
  - analysis 目录已封存：SHA256SUMS 的 sha256 6acc04cd…，附上述脚本、analyze_kept_tokens.py 和批处理脚本的副本，备份到 `D:\6CJNM007\backup\EXP-0010\analysis\` 并核对。
  - 封存时工作区已切到 code/ts-forecasting，token-pruning 的三个脚本从 commit a9c4f1e 导出后复制，内容与 399b2d0 相同。

## 数据与证据血缘

- 分块：24×24 的 patch 网格分成 6×6 个块，每块 4×4 个 patch。先按分数从高到低给每个块里的 patch 排名（块内第几名），再按“块内名次、分数”排序取前 keep 个。keep 30 时，分数最高的 30 个块各取块内最高的 1 个。
- 位置先验：与 EXP-0008 相同的 position_prior.npy（一张纯灰图、358 道题的平均分数），R02、R05、R06 先除以它再分块。
- 评测与计分与 EXP-0002 相同。FastV 是确定性的，同一道题在三个 split 的回答相同。

## 结果

| 结果 | 条件 | 对照 / 基线 | 差异 / 效应 | 不确定性 | 意义 | 证据位置 |
| --- | --- | --- | --- | --- | --- | --- |
| 块覆盖率（停止条件） | R01、R02 | 30/36 | 每个保留集合都正好覆盖 30 个块 | 358 次预填充 | 分块实现与设计一致 | analysis/kept_summary.json |
| 保留位置 | R01（只覆盖）、R02（去位置 + 覆盖） | EXP-0006 的 FastV / 随机，EXP-0008-R03（只去位置） | 平均行 12.71 / 11.00（FastV 15.92，随机 11.53，只去位置 10.83）；底部 1/4 占比 31.8% / 20.1%（57.3%，24.8%，19.0%）；最近邻距离 2.94 / 2.68（2.01，2.45，2.11）；同图不同题 Jaccard 0.945 / 0.934（0.948，0.026，0.957）；子集准确率 / yes 比例 69.3% / 20.7%，70.4% / 22.9%（56.4% / 6.7%，70.4% / 22.3%，67.0% / 17.9%） | 单一子集 | 分块后保留集合比随机更分散，但仍与问题无关 | analysis/kept_stats.md |
| POPE F1，规则 1 | R03 只覆盖（61.47） | 只去位置（EXP-0008-R05，57.41） | **+4.06** | 配对 bootstrap 95% 区间 [2.83, 5.27] | 只覆盖已经高于只去位置 | analysis/paired_bootstrap.txt |
| MME-P，规则 1 | R04（1280.7） | EXP-0008-R06（1184.9） | +95.8 | [29.5, 157.0] | 方向相同 | 同上 |
| POPE F1，规则 3 | R05 去位置 + 覆盖（64.66） | 随机-64 种子 2021（R29，60.26） | **+4.40** | [2.81, 5.83] | 超过随机 | 同上 |
| MME-P，规则 3 | R06（1253.3） | 随机-64（R28，1168.2） | +85.2 | [5.3, 164.7] | 超过随机 | 同上 |
| POPE F1，其他 | R03、R05 | FastV-64（R11，36.19）、随机-64、R03 | R03 − FastV-64 +25.28 [23.82, 26.87]；R03 − 随机 +1.21 [−0.30, 2.80]；R05 − FastV-64 +28.47 [27.01, 30.03]；R05 − 只去位置 +7.25 [6.08, 8.39]；R05 − R03 +3.19 [2.20, 4.21] | 配对 bootstrap | 只覆盖补回 FastV 与随机之间的全部差距；去位置再加 3.2 | 同上 |
| MME-P，其他 | R04、R06 | FastV-64（R10，979.6）、随机-64、R04 | R04 − FastV-64 +301.2 [231.2, 370.1]；R04 − 随机 +112.6 [36.1, 191.0]；R06 − FastV-64 +273.7 [211.3, 337.4]；R06 − 只去位置 +68.4 [4.7, 128.2]；R06 − R04 −27.4 [−84.3, 27.5] | 配对 bootstrap | MME 上只覆盖已明显高于随机；去位置不再增加 | 同上 |
| 与编码器侧 / 文本引导方法（探索性，未做配对区间） | R03～R06 | VisionZip-64（R17 77.03、R35 1364.2）、SparseVLM-64（R41 77.67、R40 1279.6） | POPE：R05 比 VisionZip-64 低 12.4、比 SparseVLM-64 低 13.0；MME-P：R04 比 VisionZip-64 低 83.5，与 SparseVLM-64 相同（+1.1；SparseVLM 的层平均是 82.25，这里是 64） | — | MME 上的差距已基本消失，POPE 上仍大 | 各运行 metrics.json |
| POPE 召回 / yes 比例 | R03、R05 | FastV-64、只去位置、随机-64、VisionZip-64 | 召回 44.9%、48.5%（22.3%，40.7%，44.2%，64.9%）；yes 比例 22.7%～23.7%、24.6%～25.7%；精确率 94.3%～99.4% | — | 上升仍全部来自召回 | analysis/details.md |
| 漏答的正样本 | R05 | 随机-64、VisionZip-64 | 逐正样本漏答比例：R05 − 随机 −4.3 个百分点（vanilla 答对处 −5.7，vanilla 漏掉处 +1.3）；R05 − VisionZip +16.5（vanilla 答对处 +19.4） | 1500 个正样本；未做区间 | 与 VisionZip 的差距在 vanilla 能答对的正样本上 | 同上 |
| MME 子任务 | R04、R06 | FastV-64、随机-64、VisionZip-64 | R04 − FastV-64：existence +65.0，count +55.0，position +48.3，color +38.3，landmark +35.2，scene +29.0，celebrity −2.1；R06 − 随机：celebrity +34.1，landmark +18.5，existence +15.0，count −18.3，其余在 ±10.2 以内 | OCR 40 题，其余每项 60～400 题；未做子任务区间 | 覆盖修好了 FastV 在存在、计数、位置、颜色上的损失，又保住了注意力在人脸等细节上的优势 | 同上 |
| MME 的 yes 比例 | R04、R06 | FastV-64（47.9%）、随机-64（58.5%） | 53.2%、54.4%；yes 题召回 76.1%、77.7%，no 题特异度 69.6%、68.8% | — | 介于两者之间 | EXP-0002/deepdive2/mme_yes_ratio.md |
| 资源 | R03～R06 | FastV-64 | 每题 0.139～0.148 s；峰值显存 13.94～13.96 GiB | — | 分块不增加可测的开销 | runs.tsv |

按预先写下的规则判断：

1. R03（只覆盖）比 EXP-0008-R05（只去位置）高 4.06（[2.83, 5.27]），满足“R03 ≥ R05 − 1”。**支持**“收益主要来自更分散，位置校正本身不是必要的”：只强制覆盖就补回了 FastV-64 与随机之间的全部 POPE 差距（R03 − 随机 +1.21，区间跨 0），MME 上还高于随机 112.6。
2. 不适用（R03 没有低于 EXP-0008-R05）。
3. R05（去位置 + 覆盖）比随机-64 高 4.40（[2.81, 5.83]），MME 高 85.2（[5.3, 164.7]），区间都不含 0。**支持**“K=2 时第 1 层注意力去掉位置后有可用的内容信号”，与 FEATHER 的方向一致。（2026-10-30 完整阅读补注：对照的随机是不分块的随机（i.i.d.），本实验没有“分块随机”的对照，R05 − 随机里含分块本身的作用；只覆盖（R03）比随机 +1.21，区间跨 0。所以只能写“支持，但分不开打分的内容信号与分块”）

## 观察与解释

| 类型 | 观察 | 当前解释 | 待复验 / 替代解释 | 证据位置 |
| --- | --- | --- | --- | --- |
| 主要观察 | 只强制覆盖，FastV-64 的 POPE +25.3、MME-P +301，追平随机（POPE）并超过随机（MME）；再去掉位置项，POPE 比随机高 4.4 | F8 的主要机制是保留集合在空间上太集中；位置偏差是造成集中的原因之一，但单独去掉它不如直接强制覆盖 | 分块的方式（6×6、按块内名次轮流取）只是一种实现；FEATHER 的均匀采样细节不同，数值不能直接比 | analysis/paired_bootstrap.txt |
| 意外现象 | 只覆盖的 FastV-64 在 MME-P 上与 SparseVLM-64 相同（1280.7 vs 1279.6），而层平均少 18 个 token；POPE 上仍低 16 | MME 上 FastV 与 SparseVLM 的差距几乎全是覆盖造成的；POPE 上剩下的差距来自别的因素（推断：题目相关的选择，C12） | 两者层分布不同（FastV 576×2 后 30 个，SparseVLM 576×3 后 82/32/18），MME 上的相等可能是巧合 | 各运行 metrics.json |
| 意外现象 | 分块后保留集合仍与问题无关（同图不同题 Jaccard 0.945、0.934）；只覆盖时最常保留的仍是第 23 行的 patch | 块内的选择仍由同一种位置/显著性打分决定，与题目无关 | 用题目相关的打分替换块内排序（C12 的一种检验） | analysis/kept_stats.md |
| 意外现象 | MME 子任务上，去位置 + 覆盖比随机在 celebrity 上高 34.1、landmark 高 18.5，count 低 18.3 | F11 的交互在覆盖之后仍部分存在：注意力打分保住细节，均匀随机在计数上仍略好 | 子任务题数少，未做区间 | analysis/details.md |
| 技术 / 质量 | 六次运行同一 commit、无未提交改动；块覆盖率核对通过；每题时间与 FastV-64 相同 | — | — | 各运行 git.txt、status.json |

## 偏差与限制

- 计划与实际：无偏差。与 VisionZip-64、SparseVLM-64 的比较和 details.md 的细分析是计划外的，标为探索性。
- 只有 64 档；分块只有 6×6 一种粒度；位置先验仍来自一张灰图。
- 随机对照只用种子 2021（EXP-0002 三个种子的 POPE 标准差 0.32）。
- MME 子任务差异、漏答分析和与 VisionZip、SparseVLM 的差距没有区间。
- FEATHER 的代码与正文没有读（只有检索摘要），本实验与它的实现不同，只比较方向。

## 结论边界与下一步

- 当前证据支持：在 K=2、keep 30（层平均 64）时，FastV 不如随机的主要原因是保留集合太集中；强制覆盖就追平随机（POPE）、超过随机（MME）；再去掉位置项，POPE、MME 都超过随机。这与 FEATHER 报告的两个部件方向一致，是独立复核。
- 当前证据不能支持：改进后的 FastV 接近编码器侧（VisionZip）或文本引导（SparseVLM）方法——POPE 上仍低 12～13；分块的具体方式最优；结论在 192、128 档同样成立。
- 下一项最低成本检查：剩下的 POPE 差距来自层分布还是选择规则。FastV-64 把层平均 64 中的 36 个花在前 2 层（576×2），第 2 层以后只剩 30 个；VisionZip 每层 64 个。用“第 0 层前就随机选 64 个”的对照（需要让随机选择支持 K=0）可以把 VisionZip 的优势拆成分布与规则两部分，与 EXP-0007 拆 SparseVLM 的方法相同。
- 后续路由：squeeze-experiment（SQZ-EXP-0010）→ deep-read Mode C 回流（MODE_C-0002 回流 3）→ 第 ⑦ 步排序。
- 下一 skill：squeeze-experiment
