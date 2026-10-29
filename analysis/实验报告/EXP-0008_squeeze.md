# SQUEEZE_REPORT：去掉位置项后 FastV 的深挖（EXP-0008）

状态：完成（2026-10-24）。
报告 ID SQZ-EXP-0008 / 实验 ID EXP-0008 / 输入记录：analysis/实验报告/EXP-0008_record.md（2026-10-24 完成版）/ 日期 2026-10-24。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0008
squeeze_id: SQZ-EXP-0008
created_at: 2026-10-24
source_ids: [EXP-0008, EXP-0006, EXP-0002, SQZ-EXP-0006, MODE_C-0002, 1A4B-0001]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0008_record.md
- code/output/EXP-0008/analysis/（score_curve.md/json、position_prior.npy、kept_stats.md、kept_summary.json、paired_bootstrap.txt、mme_subtasks.md；SHA256SUMS 的 sha256 a3be9b26…，附 analyze_score_prior.py、analyze_kept_tokens.py、bootstrap_ci.py、compare_0008_0007.sh、mme_sub_0008.py 副本，已备份）
- code/output/EXP-0008/EXP-0008-R01、R02/scores.npy（每道题 576 个分数）；R03、R04/kept_tokens.json
missing_evidence:
- 位置先验只来自一张纯灰图；没有用其他灰度或噪声图检查先验是否稳定
- 没有区分 RoPE 衰减与 attention sink
- 校正后的完整评测只有 64 档；192、128 档只有子集上的保留统计（R04）
- 没有“与位置无关的分散化”对照（例如按块分层取最高分），不能排除校正只是让选择更分散
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0008]
claim_scope: LLaVA-1.5-7B，FastV（K=2，第 1 层最后一个 token 的 head 平均注意力）除以空白图像上的平均分数后取 top-k；64 档完整 POPE、MME，keep 30/166 的 pope_img20 子集
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“主要结果”
process_dynamics: 不适用（推理诊断）
mechanism_and_intermediate_findings: 见“深度分析”第 3 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 2、4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 5 行
new_P: [P-SQZ8-1, P-SQZ8-2, P-SQZ8-3]
new_F: [F16, F17]
new_M: [M1 状态更新：位置成分与内容无关已直接测到，并且是 F8 的主要原因之一；来源（RoPE 或 sink）未拆开]
evidence_index: 见 mode_C_input_bundle.E
open_questions: [名额多时校正后的 FastV 是否超过随机, 校正的效果来自“去位置”还是只来自“更分散”, 第 1 层去掉位置项后剩下的是图像显著性还是问题相关性]
next_routes: [完整 POPE/MME 上的 fastv192_prior、fastv128_prior（4 次运行）, 按块分层取最高分的对照（不校正位置）, deep-read Mode C 回流, 第 ⑦ 步排序]
mode_C_input_bundle:
  source_ids: [EXP-0008, SQZ-EXP-0008]
  P:
  - id: P-SQZ8-1
    text: "去掉位置项后，FastV 的选择仍与问题无关（同图不同题 Jaccard 0.957）：第 1 层最后一个 token 的注意力里是否有问题相关的信号；更深的 K 或以全部文本 token 为 query 时是否出现"
    evidence: [E-SQZ8-3]
    status: UN
  - id: P-SQZ8-2
    text: "名额多时（keep 166，层平均 192）校正后的 FastV 在子集上已比随机高 5.1 个百分点；完整评测上是否成立，128 档如何"
    evidence: [E-SQZ8-3]
    status: UN（子集证据）
  - id: P-SQZ8-3
    text: "校正的收益来自去掉位置成分，还是只来自让保留集合在空间上更分散"
    evidence: [E-SQZ8-3, E-SQZ8-4]
    status: UN
  F:
  - id: F16（新）
    text: "FastV（K=2）的分数有一条与内容无关的位置曲线：纯灰图上底部 4 行占图像注意力的 31.7%（均匀 16.7%），按行的相对倍数从顶部 0.48～1.00 升到第 22 行 2.56，最后一个 patch 占 0.70%（均匀 0.17%）；真实图像的行剖面较平（0.64～1.70），与空白曲线的相关只有 0.33（逐题中位数 0.14）"
    evidence: [E-SQZ8-1, E-SQZ8-2]
    status: DE
  - id: F17（新）
    text: "把 FastV 的分数除以空白图像的位置曲线后再选，FastV-64 的 POPE F1 +21.2 [19.6, 22.8]（36.2 → 57.4，召回 22.3% → 40.7%），MME-P +205 [141, 275]，补回与随机差距的 88%；仍比随机低 2.85 [−4.40, −1.25]（POPE），MME 上分辨不出差异（+16.8 [−74.5, 103.9]）。MME 子任务上仍是注意力保住细节（celebrity +38.5、OCR +27.5）、随机保住存在与计数（existence −30、count −23）"
    evidence: [E-SQZ8-4, E-SQZ8-5]
    status: DE
  M:
  - id: M1（状态更新）
    text: "注意力打分里的位置成分与内容无关（空白图像上直接测到），去掉它后 FastV 在 64 档接近随机：位置偏差是 F8 的主要原因之一。来源（RoPE 远程衰减或 attention sink）仍未拆开；剩下 2.85 分的差距可能来自覆盖仍不足（块覆盖率 50.4% vs 57.7%）"
    evidence: [E-SQZ8-1, E-SQZ8-3, E-SQZ8-4]
    status: 结果 DE；来源 AI
  E:
  - id: E-SQZ8-1
    text: "空白图像（R01）与真实图像（R02）的分数：图像 token 得到的注意力总和 0.328 / 0.334；底部 4 行占比 31.7% / 25.2%；最后一个 patch 0.70% / 0.45%；逐行相对倍数（24 行）"
    location: code/output/EXP-0008/analysis/score_curve.md、score_curve.json
  - id: E-SQZ8-2
    text: "576 个位置上真实分数与空白曲线的 Pearson r：平均分数 0.329，逐题的中位数 0.138"
    location: score_curve.json
  - id: E-SQZ8-3
    text: "校正后的保留统计：keep 30 平均行 10.83、底部 1/4 占 19.0%、块覆盖率 50.4%、Jaccard 0.957、子集准确率 67.0%、yes 比例 17.9%；keep 166 为 10.49、11.9%、94.3%、0.955、88.3%、40.2%（EXP-0006 的 FastV：15.92/16.56、57.3%/56.0%、45.4%/85.4%、0.948/0.957、56.4%/79.3%、6.7%/32.4%；随机：11.53/11.50、24.8%、57.7%/99.6%、0.026/0.169、70.4%/83.2%、22.3%/37.4%）"
    location: code/output/EXP-0008/analysis/kept_stats.md；code/output/EXP-0006/analysis/kept_stats.md
  - id: E-SQZ8-4
    text: "配对 bootstrap（1000 次）：POPE R05 − FastV-64 +21.22 [19.61, 22.81]，− 随机-64 −2.85 [−4.40, −1.25]，− VisionZip-64 −19.61 [−20.88, −18.38]；MME R06 − FastV-64 +205.3 [141.4, 274.8]，− 随机-64 +16.8 [−74.5, 103.9]，− VisionZip-64 −179.3 [−236.8, −120.0]"
    location: code/output/EXP-0008/analysis/paired_bootstrap.txt
  - id: E-SQZ8-5
    text: "R05 召回 40.7%（三个 split 的 TP 都是 610），yes 比例 20.6%～21.3%，精确率 95.6%～98.7%；R06 的 10 个感知子任务全部高于 FastV-64（+2.0～+38.3）"
    location: EXP-0008-R05/metrics.json、EXP-0008-R06/metrics.json
  evidence_locations: [code/output/EXP-0008/analysis/SHA256SUMS]
```

## 证据审计

- **完整性**：R01～R04 各记录到 358 次预填充；R05 计分 8910 题、R06 计分 2374 题，与 EXP-0002 相同。分数无 NaN、不是常数（停止条件未触发）。
- **对照的有效性**：配对比较用的 EXP-0002 运行与本实验题目、计分完全相同；随机对照只用种子 2021，三个种子的 POPE 标准差 0.32，不影响结论。
- **血缘**：代码 commit 0ea34bd，六次运行都无未提交改动；位置先验由 R01 生成后不再改动，R03～R06 共用同一个文件。
- **预先计划与事后提出**：四条解释规则在计划 v1 中写定；MME 子任务比较、召回与 yes 比例的拆分、“补回 88%”的算法是看到结果后补充的描述。规则 1 的“高度相关”没有事先给出数值门槛。

## 主要结果

| 结果 | 对照 | 差异 | 不确定性 | 实际意义 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 位置曲线（问题 1） | 均匀分布 | 底部 4 行 31.7% vs 16.7%；第 22 行 2.56 倍 | 358 次预填充、一张灰图 | 分数里有与内容无关的位置成分 | E-SQZ8-1 |
| 能解释多少真实分数 | 空白曲线 | 相关 0.33（逐题 0.14） | — | 只是一部分 | E-SQZ8-2 |
| 保留分布（问题 2） | EXP-0006 的 FastV 与随机 | 平均行回到 10.5～10.8；块覆盖率 50.4% / 94.3%（随机 57.7% / 99.6%）；yes 比例 6.7% → 17.9% | 子集 358 题 | 覆盖与 yes 比例部分恢复 | E-SQZ8-3 |
| POPE F1（问题 3） | FastV-64 / 随机-64 | +21.2 / −2.85 | 区间都不含 0 | 补回 88% 的差距，但不超过随机 | E-SQZ8-4 |
| MME-P | FastV-64 / 随机-64 | +205 / +17 | 后者区间跨 0 | 与随机持平 | E-SQZ8-4 |

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 校正后 POPE +21.2、MME-P +205，接近随机（E-SQZ8-4） | 位置偏差是 F8 的主要原因之一（M1 的结果层面） | 校正只是让选择更分散，与“去位置”无关（P-SQZ8-3） | 只有 64 档的完整评测 | 按块分层取最高分的对照 |
| 错误和失败模式 | F1 的上升全部来自召回（22.3% → 40.7%），精确率仍为 95.6%～98.7%；yes 比例 11.6% → 21%（E-SQZ8-5） | 被删的物体证据减少，模型“默认不存在”的情况减轻（与 F10 一致） | — | 没有物体框 | 补 COCO 物体框 |
| 机制、消融和交互 | 空白图像上分数随行号上升（E-SQZ8-1）；真实图像上剖面较平，相关 0.33（E-SQZ8-2）；校正后仍与问题无关，最常保留的是图像中部的固定 patch（E-SQZ8-3） | 第 1 层最后一个 token 的注意力 = 位置成分 + 图像显著性；几乎不含问题信息（推断） | 位置成分随图像内容变化（例如 sink 的强弱），一条平均曲线校正不干净 | 先验只来自一张灰图 | 不同灰度、噪声图的先验；更深的 K（P-SQZ8-1） |
| 子组、切片和边界 | keep 30：子集准确率仍比随机低 3.4；keep 166：比随机高 5.1（E-SQZ8-3）。MME 子任务：celebrity +38.5、OCR +27.5，existence −30、count −23（E-SQZ8-5） | 名额少时覆盖不足的损失占主导；名额多时显著性打分能兼顾覆盖与细节（推断） | 子集太小，keep 166 的优势可能是噪声 | 358 题；子任务无区间 | 完整评测上的 fastv192_prior、fastv128_prior（P-SQZ8-2） |
| 稳健性及替代解释 | 六次运行同一 commit；随机种子间差异小（0.32） | 结论不是运行或种子造成的 | 先验对灰度的敏感性未知 | 一张灰图 | 同上 |
| 局部收益、整体收益和代价 | 每题时间与峰值显存与 FastV 相同（0.142 s、13.94 GiB）（2026-10-29 完整阅读补注：只有峰值显存相同（13.94 / 13.96 GiB，FastV-64 为 13.93 / 13.96 GiB）。每题时间不同：校正后 POPE 0.1418 s、MME 0.1517 s（EXP-0008-R05、R06），FastV-64 为 0.1353 s、0.145 s（EXP-0002-R11、R10），慢约 5%。这个差来自校正本身还是运行间波动，没有单独计时，所以右边“校正没有额外开销”在时间上没有证实，只有“先验离线算一次”这一条理由。） | 校正没有额外开销（先验离线算一次） | — | — | — |

## 学到什么

- **假设**：规则 1 部分成立（位置曲线存在，但对真实分数的解释力有限），规则 2 成立，规则 3 的前半成立、后半不成立（不超过随机）。
- **已排除与仍无法区分的解释**：
  - 已排除：“FastV 不如随机与位置无关”——除掉位置曲线就补回了大部分差距。
  - 仍无法区分：收益来自“去位置”还是“更分散”；位置曲线来自 RoPE 还是 sink。
- **保留 / 复验 / 回退 / 停止**：保留 F16、F17；1a-4b 文档的第一条边（位置曲线）由假设改为观察。“位置校正的注意力打分”本身最多与随机持平，不能单独作为改进方法；名额多时是否占优待完整评测（P-SQZ8-2）。
- **最小、最能区分的下一次实验**：完整 POPE、MME 上的 fastv192_prior 与 fastv128_prior（4 次运行，约 55 分钟）；若 192 档超过随机，再做“按块分层、不校正位置”的对照，区分“去位置”与“更分散”。
- **回流 Deep Read**：F16、F17、P-SQZ8-1～3、M1 的状态更新与 E-SQZ8-1～5 交给 MODE_C-0002（回流 2，C9）。

## 同步清单

- [ ] experiment-results.md：登记 EXP-0008 的 6 次运行与本报告。
- [ ] 已发现的实验现象.md：新增 F16、F17，更新 F8、F15。
- [ ] 论文素材库.md：候选图（空白与真实图像的逐行分数曲线；校正前后保留位置的热力图）。
- [ ] MODE_C-0002：回流 2（C9、M1 状态更新，新增 F16、F17）。
- [ ] 1a-4b_FastV不如随机选择.md：3a 的结果。
- [ ] 最新实验分析报告.md、index.md。

（2026-10-29 完整阅读补注：上面六项实际都已完成，只是没有勾选：experiment-results 的 EXP-0008 行与 EXP-0008-R01～R06 六行；已发现的实验现象 F16、F17（F8、F15 已补 EXP-0008）；论文素材库 MAT-0008；MODE_C-0002 回流 2（C9、M1、F16、F17）；1a-4b_FastV不如随机选择 的“3a 结果（EXP-0008）”一行；最新实验分析报告“EXP-0008”一节与 index 中的两条登记。）
