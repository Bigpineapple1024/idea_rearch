# EXPERIMENT_RECORD：把层分布与选择规则拆开（C11）

状态：completed（2026-10-24）。10 次运行全部完成，计划内的比较已做完；结果按预先写下的规则判断，见“结果”和“观察与解释”。
实验 ID：EXP-0007 / IDEA ID：none（研究流程第 ④⑤ 步：MODE_C-0002 的 C11）/ 日期：2026-10-24 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

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
| EXP-0007-R01 | random_svp64_s2021_layers × pope50 | 10-24 06:51 / 06:51 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method random --fastv-schedule 3:82,7:32,16:18 --seed 2021 --layer-token-log；主指标 82.25 | POPE 50 题 | `code/output/EXP-0007/EXP-0007-R01/`，SHA256SUMS 的 sha256 bde58bf1…；备份同名目录 | 主指标一栏是 50 题上的层平均视觉 token 数；逐层数核对通过 |
| EXP-0007-R02 | fastv_svp64_layers × pope50 | 10-24 06:51 / 06:52 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-schedule 3:82,7:32,16:18 --layer-token-log；主指标 82.25 | POPE 50 题 | `code/output/EXP-0007/EXP-0007-R02/`，SHA256SUMS 的 sha256 f8e3d173…；备份同名目录 | 同上 |
| EXP-0007-R03 | random_svp64_s2021 × pope | 10-24 06:52 / 07:13 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method random --fastv-schedule 3:82,7:32,16:18 --seed 2021；主指标 62.18213077175731 | POPE 8910 题 | `code/output/EXP-0007/EXP-0007-R03/`，SHA256SUMS 的 sha256 534b8ddc…；备份同名目录 |  |
| EXP-0007-R04 | fastv_svp64 × pope | 10-24 07:13 / 07:35 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-schedule 3:82,7:32,16:18；主指标 55.025289766544795 | POPE 8910 题 | `code/output/EXP-0007/EXP-0007-R04/`，SHA256SUMS 的 sha256 e5779b6d…；备份同名目录 |  |
| EXP-0007-R05 | random_flat64_s2021 × pope | 10-24 07:35 / 07:55 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method random --fastv-k 2 --fastv-keep 49 --seed 2021；主指标 67.73499651449566 | POPE 8910 题 | `code/output/EXP-0007/EXP-0007-R05/`，SHA256SUMS 的 sha256 0e1d9390…；备份同名目录 |  |
| EXP-0007-R06 | fastv_flat64 × pope | 10-24 07:55 / 08:15 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 49；主指标 53.48888160270344 | POPE 8910 题 | `code/output/EXP-0007/EXP-0007-R06/`，SHA256SUMS 的 sha256 87791b6c…；备份同名目录 |  |
| EXP-0007-R07 | random_svp64_s2021 × mme | 10-24 08:15 / 08:21 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method random --fastv-schedule 3:82,7:32,16:18 --seed 2021；主指标 1179.7597038815527 | MME 2374 题 | `code/output/EXP-0007/EXP-0007-R07/`，SHA256SUMS 的 sha256 e71abc73…；备份同名目录 |  |
| EXP-0007-R08 | fastv_svp64 × mme | 10-24 08:21 / 08:27 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-schedule 3:82,7:32,16:18；主指标 1087.4643857543017 | MME 2374 题 | `code/output/EXP-0007/EXP-0007-R08/`，SHA256SUMS 的 sha256 e90996cd…；备份同名目录 |  |
| EXP-0007-R09 | random_flat64_s2021 × mme | 10-24 08:27 / 08:33 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method random --fastv-k 2 --fastv-keep 49 --seed 2021；主指标 1242.7374949979992 | MME 2374 题 | `code/output/EXP-0007/EXP-0007-R09/`，SHA256SUMS 的 sha256 b7ae893b…；备份同名目录 |  |
| EXP-0007-R10 | fastv_flat64 × mme | 10-24 08:33 / 08:39 | completed（评测退出码 0） | commit 0ea34bd，无未提交改动；`tp`；--method fastv --fastv-k 2 --fastv-keep 49；主指标 1174.889855942377 | MME 2374 题 | `code/output/EXP-0007/EXP-0007-R10/`，SHA256SUMS 的 sha256 4adef758…；备份同名目录 |  |

- 十次运行的 config.json 的 experiment 字段都是 EXP-0007。
- 逐层核对：`check_layer_profile.py` 对 R01、R02 都输出 profile ok（逐层 576×3、82×4、32×9、18×16，与 EXP-0002-R49 的 SparseVLM-64 实测相同），之后才开始 R03。两行输出从批处理日志摘到 analysis/profile_check.txt。
- 运行后的 CPU 分析：
  - 配对 bootstrap：`bootstrap_ci.py`，由 compare_0008_0007.sh 的 0007 部分调用，结果在 analysis/paired_bootstrap.txt；计划外的“FastV − 随机（平坦分布）”在 paired_bootstrap_extra.txt（compare_0007_extra.sh），标为探索性。
  - POPE 分 split 统计、漏答正样本、MME 子任务：`analyze_0007_details.py`，输出 details.md、details.json。
  - analysis 目录已封存：SHA256SUMS 的 sha256 a6310ab6…，附上述脚本和批处理脚本的副本，备份到 `D:\6CJNM007\backup\EXP-0007\analysis\` 并核对。

## 数据与证据血缘

- 层分布：SVP 的三段（第 3、7、16 层前剪到 82、32、18）取自 EXP-0002-R49（SparseVLM-64 × pope50 的 `--layer-token-log`）。每一段都在剩下的图像 token 里重新排序：FastV 用上一层（第 2、6、15 层）最后一个 token 的 head 平均注意力；随机选择用同一个生成器继续抽。因此随机配置第 16 层以后的 18 个 token 是 576 个中的一个均匀随机子集。
- 平坦分布：K=2、保留 49 个，层平均 81.94。
- 随机选择每道题重新抽取。POPE 三个 split 的 1500 道正样本题相同，所以对随机配置来说，同一道正样本题在三个 split 中是三次独立抽取；FastV 和 SparseVLM 对重复的题回答相同。
- 评测与计分与 EXP-0002 相同：POPE 取三个 split 的 F1 平均；MME 取 Perception 总分，用官方计分工具。

## 结果

| 结果 | 条件 | 对照 / 基线 | 差异 / 效应 | 不确定性 | 意义 | 证据位置 |
| --- | --- | --- | --- | --- | --- | --- |
| 逐层 token 数 | R01、R02（pope50） | EXP-0002-R49（SparseVLM-64 实测） | 两种打分的逐层数都是 576×3、82×4、32×9、18×16，层平均 82.25 | 50 题逐题相同 | 层分布按计划复制 | analysis/profile_check.txt，各运行 metrics.json |
| POPE F1，Q1 | R03 随机 svp（62.18） | SparseVLM-64（EXP-0002-R41，77.67） | **−15.48** | 配对 bootstrap 95% 区间 [−16.80, −14.20] | 同一层分布下，随机选择远低于 SparseVLM | analysis/paired_bootstrap.txt |
| MME-P，Q1 | R07（1179.8） | SparseVLM-64（R40，1279.6） | −99.8 | [−180.3, −22.3] | 方向相同 | 同上 |
| POPE F1，Q2 | R04 FastV svp（55.03） | R03 | **−7.16** | [−8.87, −5.62] | 同一分布下 FastV 仍不如随机 | 同上 |
| MME-P，Q2 | R08（1087.5） | R07 | −92.3 | [−166.4, −14.9] | 方向相同 | 同上 |
| POPE F1，Q3（随机） | R03 | R05 随机 flat（67.73） | **−5.55** | [−7.00, −4.23] | 与预测相反：前多后少的分布让随机选择更差 | 同上 |
| MME-P，Q3（随机） | R07 | R09（1242.7） | −63.0 | [−136.2, 12.9] | 分辨不出差异，方向同 POPE | 同上 |
| POPE F1，Q3（FastV） | R04 | R06 FastV flat（53.49） | +1.54 | [0.07, 2.99] | 小幅改善，区间刚好不含 0 | 同上 |
| MME-P，Q3（FastV） | R08 | R10（1174.9） | −87.4 | [−156.4, −15.0] | MME 上反而更差 | 同上 |
| 名额 30 → 49（平坦） | R05、R06；R09、R10 | 随机-64 种子 2021（R29 60.26、R28 1168.2）、FastV-64（R11 36.19、R10 979.6） | POPE：随机 +7.47 [6.58, 8.42]，FastV +17.30 [15.97, 18.75]；MME：随机 +74.6 [28.2, 123.6]，FastV +195.3 [136.9, 257.4] | 配对 bootstrap | 名额增加时，FastV 的上升约为随机的 2.3 倍（POPE）、2.6 倍（MME） | 同上 |
| FastV − 随机，平坦分布（探索性，计划外） | R06、R10 | R05、R09 | POPE −14.25 [−15.90, −12.64]；MME −67.8 [−154.4, 14.7] | 配对 bootstrap | SVP 分布下的 POPE 差距（−7.16）约为平坦分布下的一半 | analysis/paired_bootstrap_extra.txt |
| POPE 召回 / yes 比例 | R03～R06 | SparseVLM-64、随机-64、FastV-64、vanilla | 召回（三个 split 平均）：随机 svp 46.2%，随机 flat 52.7%，FastV svp 38.4%，FastV flat 36.8%，SparseVLM 65.7%，vanilla 79.1%；yes 比例 24.5%、28.1%、20.0%、19.0%、35.0%、42.6%；精确率都在 90%～99% | — | 各配置之间的差别几乎都在召回上：剪枝后更多答 no（F10） | analysis/details.md |
| 漏答的正样本 | R03～R06 | SparseVLM-64、vanilla（EXP-0002-R04，漏 313 个） | 逐个正样本的漏答比例（三个 split 平均），随机 svp − 随机 flat：+6.5 个百分点，其中 vanilla 答对处 +8.6、vanilla 漏掉处 −1.6，多出的漏答全部来自 vanilla 答对的正样本；随机 svp − SparseVLM：+19.6，93% 来自 vanilla 答对处 | 1500 个正样本；未做区间 | 分布和规则造成的损失都落在 vanilla 本来答对的正样本上 | 同上 |
| 随机抽取的逐题波动 | R03、R05 | 各正样本漏答概率相同时的期望 | 三次抽取回答相同的比例：58.5%（svp）、62.6%（flat）、56.1%（EXP-0002 随机-64）。R03 漏 0/1/2/3 次的正样本数为 386/298/325/491；若每题漏答概率都等于 53.8%，应约为 148/517/602/234 | — | 正样本难度很不均匀：约三分之一每次都漏，约四分之一每次都答对 | 同上 |
| MME 子任务 | R07～R10 | SparseVLM-64、随机-64、FastV-64、vanilla | 随机 svp − flat：existence −30.0，color −16.7，count −13.3，position −10.0，OCR +15.0，其余在 ±9.2 以内；FastV svp − flat：existence +20.0，count +11.7，position +8.3，OCR −52.5，celebrity −34.1，color −28.3；随机 svp − SparseVLM：existence −40.0，count −30.0，color −25.0，celebrity −24.7 | OCR 40 题，其余每项 60～400 题；未做子任务区间 | 层分布的作用随选择规则和任务类型而变 | 同上 |
| 资源 | R03～R10 | — | 每题时间：POPE 0.142～0.143 s（svp）、0.132～0.136 s（flat）；MME 0.145 s（svp）、0.139～0.141 s（flat）；峰值显存 13.95～13.99 GiB | — | 层平均相同时，SVP 每题多用 3%～8% 时间 | runs.tsv |

按预先写下的规则判断：

1. Q1：随机 svp 比 SparseVLM-64 低 15.48，区间 [−16.80, −14.20] 不含 0，MME 上也低 99.8（[−180.3, −22.3]）。**支持**“文本评分者或回收确有作用”：在 SparseVLM 自己的层分布下，换成随机选择会失去它相对 FastV-64 优势的大部分。（2026-10-29 完整阅读补注：“大部分”不成立。SparseVLM-64 的 POPE（77.67）比 FastV-64（36.19）高 41.47（SQZ-EXP-0007 深度分析第一行拆成 24.07 + 1.92 + 15.48）；随机 svp（62.18）仍比 FastV-64 高 25.99，只失去 15.48，约 37%。“大部分”只在与随机-64（种子 2021，60.26，EXP-0002-R29）比时成立：SparseVLM-64 比它高约 17.4，其中 15.48（约 89%）来自规则。Q1 的判断（规则确有作用）不变。）本实验没有把文本评分者与回收分开。
2. Q2：FastV svp 比随机 svp 低 7.16（[−8.87, −5.62]），MME 低 92.3（[−166.4, −14.9]）。**支持**“FastV 的打分问题不依赖层分布”（C9 的方向）。差距的大小随分布而变：平坦分布下是 −14.25（探索性）。
3. Q3：随机 svp 比随机 flat 低 5.55（[−7.00, −4.23]），规则的条件（“明显高于”）不成立，方向还相反。**不支持 M4**：层平均相同时，前 3 层保留全部、之后逐段减少的分布对随机选择更差。对 FastV，POPE +1.54（[0.07, 2.99]）而 MME −87.4（[−156.4, −15.0]），两个基准方向相反，不能算作收益。

## 观察与解释

| 类型 | 观察 | 当前解释 | 待复验 / 替代解释 | 证据位置 |
| --- | --- | --- | --- | --- |
| 主要观察 | 同一层分布下，随机选择比 SparseVLM 低 15.5（POPE）、99.8（MME-P）；多出的漏答 93% 在 vanilla 答对的正样本上 | SparseVLM 的优势来自选择规则（文本评分者打分加回收），不是层分布 | SparseVLM 的评分者取自题目文字，而 POPE 的题目直接写出物体名，题目相关的选择可能是 POPE 收益的主要部分（推断）；回收的作用需要关闭回收的对照 | analysis/paired_bootstrap.txt、details.md |
| 意外现象 | 层平均相同，SVP 分布让随机选择的 POPE 低 5.55，多出的漏答全部落在 vanilla 答对的正样本上；MME 的 existence −30、count −13 | 第 16～31 层只剩 18 个 token（平坦为 49），中后层仍在读取图像 token；前 7 层多保留的 token 补不回来（推断） | 也可能是第 3～15 层的 82/32 已不足以覆盖物体，与第 16 层后的数量无关。区分方法：随机选择、时间表 2:49,16:18（前段同平坦、后段同 SVP），1 次 POPE | details.md |
| 意外现象 | FastV 与随机的 POPE 差距在 SVP 分布下（−7.16）约为平坦分布下（−14.25）的一半；SVP 让 FastV 的 existence、count、position 上升，OCR、celebrity、color 下降 | 第 3、7、16 层用更深层的注意力重新打分，深层注意力的位置偏差可能较小（推断；EXP-0008 只测了第 1 层的位置曲线） | 记录 fastv_svp64 各段保留的位置（kept-token 日志需要支持多段）后检查 | details.md |
| 意外现象 | 名额从 30 增到 49（平坦）时，FastV 的 POPE 上升 17.3，随机只上升 7.5 | 名额越少，位置偏差造成的覆盖损失越大（与 EXP-0008 中 keep 166 时校正后的 FastV 在子集上已超过随机一致） | — | paired_bootstrap.txt |
| 意外现象 | 对随机选择，同一道正样本题的三次独立抽取只有 56%～63% 回答相同；漏答次数的分布比“每题漏答概率相同”分散得多 | 正样本难度很不均匀（物体大小、位置、场景杂乱程度；推断）；随机选择总分稳定（EXP-0002 三个种子标准差 0.32）是大量题目平均的结果 | 需要 COCO 物体标注（物体面积）来检验，需要下载许可 | details.md |
| 技术 / 质量 | 十次运行同一 commit、无未提交改动；逐层数核对通过；SparseVLM-64 的 1500 道正样本题中只有 1 道在三个 split 的回答不同 | — | — | 各运行 git.txt、details.md |

## 偏差与限制

- 计划与实际：无偏差。“FastV − 随机（平坦分布）”的配对比较和 details.md 中的细分析是计划外的，标为探索性。
- 随机选择只用了种子 2021。EXP-0002 中随机-64 三个种子的 POPE 标准差是 0.32，SVP 与平坦之间的 5.55 远大于它；但这两种分布本身的种子方差没有测。
- SVP 分布只有一种（SparseVLM-64 的实测），平坦分布只有 K=2 一种；“前多后少不利”只在这一对分布上观察到。
- 层平均相同不等于计算量相同：注意力计算量随 token 数平方增长，SVP 每题多用 3%～8% 时间。
- MME 子任务差异和漏答分析没有区间。
- 没有记录 SVP 下 FastV 各段保留的位置。

## 结论边界与下一步

- 当前证据支持：在 SparseVLM-64 实测的层分布下，SparseVLM 的选择规则比随机选择高 15.5（POPE）、约 100（MME-P）；FastV 的打分在两种分布下都不如随机；层平均相同时，SVP 分布不比平坦分布好，随机选择下 POPE 反而更差。
- 当前证据不能支持：SparseVLM 的收益来自文本评分者还是回收；“后层 token 数更重要”这一机制（目前只是推断）；前多后少的分布在其他选择规则（如 SparseVLM 自己的规则）下也不利。
- 已有工作（检索摘要级，SC）：PyramidDrop（arXiv 2410.17247）用注意力排序做单段剪枝的预研显示浅层剪枝损失大、第 16 层只留 10% 也无明显下降；HiDrop（ICLR 2026，arXiv 2602.23699）认为浅层对视觉 token 是被动的、融合从中层开始，提出晚注入和凹形金字塔剪枝（需要训练）。本实验在训练无关、随机选择、层平均相同的条件下看到“前段多保留无益”，与 HiDrop 的出发点方向一致，与 PyramidDrop 的金字塔分布直觉相反；两篇的正文都没有读。
- 下一项最低成本检查：(a) 随机选择、时间表 2:49,16:18 的 POPE（1 次，约 20 分钟），检查 SVP 的损失是否全部来自第 16 层之后的 18 个；(b) 查 SparseVLM 代码能否关闭回收，若能，跑关闭回收的 SparseVLM-64 POPE（1 次），把评分者与回收分开。
- 后续路由：squeeze-experiment（SQZ-EXP-0007）→ deep-read Mode C 回流（MODE_C-0002 回流 3）→ 第 ⑦ 步排序。
- 下一 skill：squeeze-experiment
