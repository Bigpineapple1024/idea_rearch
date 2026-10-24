# SQUEEZE_REPORT：层分布与选择规则的深挖（EXP-0007）

状态：完成（2026-10-24）。
报告 ID SQZ-EXP-0007 / 实验 ID EXP-0007 / 输入记录：analysis/实验报告/EXP-0007_record.md（2026-10-24 完成版）/ 日期 2026-10-24。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0007
squeeze_id: SQZ-EXP-0007
created_at: 2026-10-24
source_ids: [EXP-0007, EXP-0002, EXP-0008, SQZ-EXP-0002, SQZ-EXP-0008, MODE_C-0002]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0007_record.md
- code/output/EXP-0007/analysis/（paired_bootstrap.txt、paired_bootstrap_extra.txt、details.md、details.json、profile_check.txt；SHA256SUMS 的 sha256 a6310ab6…，附 bootstrap_ci.py、check_layer_profile.py、analyze_missed_objects.py、analyze_0007_details.py、compare_0008_0007.sh、compare_0007_extra.sh、run_vision_0008_0007.sh 副本，已备份）
- code/output/EXP-0007/EXP-0007-R01～R10/（answers.jsonl、metrics.json；R01、R02 的逐层 token 数）
missing_evidence:
- 没有把 SparseVLM 的文本评分者与回收分开
- 没有只改第 16 层以后 token 数的对照，“后层数量更重要”只是推断
- 没有记录 SVP 下 FastV 各段保留的位置
- 随机选择只用种子 2021；两种分布各自的种子方差没有测
- 没有 COCO 物体框，不能把“总被漏掉的正样本”与物体大小对应
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0007]
claim_scope: LLaVA-1.5-7B，层平均约 82 时的两种层分布（SVP：576×3、82×4、32×9、18×16；平坦：576×2、49×30）× 两种选择规则（FastV 打分、随机种子 2021），完整 POPE、MME；对照为 EXP-0002 的 SparseVLM-64、随机-64、FastV-64、vanilla
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“主要结果”
process_dynamics: 不适用（推理诊断）
mechanism_and_intermediate_findings: 见“深度分析”第 3 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 2、4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 5 行
new_P: [P-SQZ7-1, P-SQZ7-2, P-SQZ7-3, P-SQZ7-4]
new_F: [F19, F20, F21, F22]
new_M: [M4 状态更新：在随机选择、层平均相同的条件下，前段多保留、后段少保留没有收益，反而更差]
evidence_index: 见 mode_C_input_bundle.E
open_questions: [SparseVLM 的 15.5 分来自文本评分者还是回收, SVP 对随机选择的损失是否全部来自第 16 层以后只剩 18 个, 多段重新打分为什么让 FastV 与随机的差距减半, 随机选择下总被漏掉的正样本有什么共同点]
next_routes: [随机选择、时间表 2:49,16:18 的 POPE（1 次）, 关闭回收的 SparseVLM-64（若代码支持）, deep-read Mode C 回流, 第 ⑦ 步排序]
mode_C_input_bundle:
  source_ids: [EXP-0007, SQZ-EXP-0007]
  P:
  - id: P-SQZ7-1
    text: "在同一层分布下，SparseVLM 的规则比随机高 15.5（POPE）：这部分来自文本评分者（用题目文字给视觉 token 打分，POPE 的题目直接写出物体名），还是来自回收（把删掉的 token 合并成少数几个）"
    evidence: [E-SQZ7-1, E-SQZ7-4]
    status: UN
  - id: P-SQZ7-2
    text: "层平均相同时，SVP 让随机选择低 5.55：损失是否全部来自第 16～31 层只剩 18 个（平坦为 49）；前段多保留的 token 是否完全没有作用"
    evidence: [E-SQZ7-2, E-SQZ7-4, E-SQZ7-5]
    status: UN
  - id: P-SQZ7-3
    text: "FastV 与随机的 POPE 差距，在 SVP 分布下（−7.16）约为平坦分布下（−14.25）的一半：在第 3、7、16 层用更深层的注意力重新打分，是否位置偏差更小、覆盖更好"
    evidence: [E-SQZ7-3, E-SQZ7-5]
    status: UN
  - id: P-SQZ7-4
    text: "随机选择下约三分之一的正样本三次抽取都被漏掉、约四分之一三次都答对：这些正样本的物体大小、位置或场景复杂度有什么共同点（与 F18 的“各方法漏掉同一批正样本”是否同一现象）"
    evidence: [E-SQZ7-6]
    status: UN（需要 COCO 物体框，需下载许可）
  F:
  - id: F19（新）
    text: "在 SparseVLM-64 实测的层分布（576×3、82×4、32×9、18×16）下，随机选择的 POPE F1 比 SparseVLM 低 15.48 [−16.80, −14.20]，MME-P 低 99.8 [−180.3, −22.3]；SparseVLM 的优势来自选择规则，不是层分布。多出的漏答 93% 落在 vanilla 答对的正样本上"
    evidence: [E-SQZ7-1, E-SQZ7-4]
    status: DE
  - id: F20（新）
    text: "层平均相同（82.25 vs 81.94）时，前多后少的 SVP 分布让随机选择的 POPE 比平坦分布低 5.55 [−7.00, −4.23]（召回 46.2% vs 52.7%），MME-P 低 63.0 [−136.2, 12.9]；多出的漏答全部来自 vanilla 答对的正样本，MME 上 existence −30、count −13。FastV 打分下 POPE +1.54 [0.07, 2.99]、MME −87.4 [−156.4, −15.0]，方向不一致"
    evidence: [E-SQZ7-2, E-SQZ7-4, E-SQZ7-5]
    status: DE
  - id: F21（新）
    text: "FastV 与随机的差距随层分布和名额变化：POPE 上，平坦 30 个 −24.07、平坦 49 个 −14.25（探索性）、SVP −7.16；名额从 30 增到 49 时，FastV +17.30 [15.97, 18.75]，随机 +7.47 [6.58, 8.42]（MME 为 +195.3、+74.6）"
    evidence: [E-SQZ7-2, E-SQZ7-3]
    status: DE
  - id: F22（新）
    text: "随机选择下，同一道正样本题的三次独立抽取只有 56%～63% 回答相同；漏答次数的分布比“每题漏答概率相同”分散得多（随机 svp：漏 0/1/2/3 次 386/298/325/491，等概率时约为 148/517/602/234）。正样本难度很不均匀，总分的稳定（种子标准差 0.32）来自大量题目的平均"
    evidence: [E-SQZ7-6]
    status: DE
  M:
  - id: M4（状态更新）
    text: "“浅层需要完整视觉信息、深层只需少量 token”：在随机选择、层平均相同的条件下，前 3 层保留全部、深层只剩 18 个的分布不如平坦 49 个（F20）；深层的 token 数仍然重要，至少在 POPE 和 MME 的 existence、count 上。FastV 打分下结果不一致。只在一对分布上观察到"
    evidence: [E-SQZ7-2, E-SQZ7-5]
    status: 结果 DE；“后层数量更重要”为 AI
  E:
  - id: E-SQZ7-1
    text: "配对 bootstrap（1000 次）：POPE 随机 svp − SparseVLM-64 −15.48 [−16.80, −14.20]；MME −99.8 [−180.3, −22.3]"
    location: code/output/EXP-0007/analysis/paired_bootstrap.txt
  - id: E-SQZ7-2
    text: "POPE 随机 svp − 随机 flat −5.55 [−7.00, −4.23]，FastV svp − FastV flat +1.54 [0.07, 2.99]，随机 flat − 随机-64 +7.47 [6.58, 8.42]，FastV flat − FastV-64 +17.30 [15.97, 18.75]；MME 对应为 −63.0 [−136.2, 12.9]、−87.4 [−156.4, −15.0]、+74.6 [28.2, 123.6]、+195.3 [136.9, 257.4]"
    location: 同上
  - id: E-SQZ7-3
    text: "POPE FastV svp − 随机 svp −7.16 [−8.87, −5.62]，MME −92.3 [−166.4, −14.9]；探索性：FastV flat − 随机 flat POPE −14.25 [−15.90, −12.64]，MME −67.8 [−154.4, 14.7]"
    location: paired_bootstrap.txt、paired_bootstrap_extra.txt
  - id: E-SQZ7-4
    text: "按 split 的召回 / yes 比例 / 精确率；逐个正样本的漏答比例之差：随机 svp − 随机 flat +6.5 个百分点（vanilla 答对处 +8.6，漏掉处 −1.6），随机 svp − SparseVLM +19.6（93% 来自 vanilla 答对处），FastV svp − 随机 svp +7.8（88%）"
    location: code/output/EXP-0007/analysis/details.md、details.json
  - id: E-SQZ7-5
    text: "MME 子任务：随机 svp − flat existence −30.0、color −16.7、count −13.3、position −10.0、OCR +15.0；FastV svp − flat existence +20.0、count +11.7、position +8.3、OCR −52.5、celebrity −34.1、color −28.3"
    location: details.md
  - id: E-SQZ7-6
    text: "随机配置三个 split 回答相同的比例 58.5%（svp）、62.6%（flat）、56.1%（随机-64）；FastV、VisionZip、vanilla 为 100%，SparseVLM 为 99.9%（1 道题不同）；漏答次数分布见 details.md"
    location: details.md
  evidence_locations: [code/output/EXP-0007/analysis/SHA256SUMS]
```

## 证据审计

- **完整性**：R03～R06 各计分 8910 题，R07～R10 各计分 2374 题，与 EXP-0002 相同；逐层核对（R01、R02）在评测之前通过。
- **对照的有效性**：EXP-0002 的对照运行与本实验题目、计分完全相同；SVP 分布与 SparseVLM-64 的实测逐层数一致（E 见记录）。随机对照只用种子 2021；EXP-0002 的三个种子 POPE 标准差 0.32，本实验中两种分布之间的差距（5.55）远大于此。
- **血缘**：十次运行都在 commit 0ea34bd、无未提交改动；analysis 目录已封存并备份。
- **预先计划与事后提出**：Q1～Q3 和三条解释规则在计划 v1 中写定。“FastV − 随机（平坦分布）”的比较、按 split 的召回、漏答正样本分析、MME 子任务表、“漏答次数分布”都是看到结果后补充的，标为探索性。

## 主要结果

| 结果 | 对照 | 差异 | 不确定性 | 实际意义 | 证据 |
| --- | --- | --- | --- | --- | --- |
| Q1：同一分布下的规则 | SparseVLM-64 | 随机 svp −15.48（POPE）、−99.8（MME） | 区间都不含 0 | SparseVLM 的优势来自规则 | E-SQZ7-1 |
| Q2：同一分布下 FastV 与随机 | 随机 svp | −7.16（POPE）、−92.3（MME） | 区间都不含 0 | FastV 的问题不依赖层分布 | E-SQZ7-3 |
| Q3：层平均相同时的分布 | 平坦 49 | 随机 −5.55、FastV +1.54（POPE）；随机 −63.0、FastV −87.4（MME） | 随机的 MME 区间跨 0；FastV 的 POPE 区间下限 0.07 | 前多后少没有收益 | E-SQZ7-2 |
| 名额 30 → 49 | 平坦 30 | 随机 +7.47、FastV +17.30（POPE） | 区间都不含 0 | 名额少时 FastV 的损失最大 | E-SQZ7-2 |

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 规则的作用（15.5）大于分布的作用（随机下 −5.55）；FastV 与 SparseVLM 在同一分布下差 22.6（E-SQZ7-1、E-SQZ7-3） | EXP-0002 中 SparseVLM-64 与 FastV-64 的 41.5 分差距可以拆成：FastV 规则的劣势 24.07（平坦 30 个）+ 名额与分布 1.92 + SparseVLM 规则的优势 15.48（加起来 41.47） | 拆分的顺序不同，各部分大小会变（另一种路径：名额 17.30 + 分布 1.54 + FastV→随机 7.16 + 随机→SparseVLM 15.48） | 只有一对分布、一个种子 | 关闭回收的 SparseVLM（P-SQZ7-1） |
| 错误和失败模式 | 各配置的差别几乎都在召回上（精确率 90%～99%）；随机 svp 比随机 flat 多出的漏答全部来自 vanilla 答对的正样本（E-SQZ7-4） | 损失的是“本来能看到的物体”，与 F10（剪枝后更多答 no）、F18（各方法漏掉同一批正样本）一致 | — | 没有物体框 | COCO 物体框（P-SQZ7-4） |
| 机制、消融和交互 | 随机：SVP 在 existence、count、color 上低于平坦（E-SQZ7-5）；FastV：SVP 在 existence、count、position 上高于平坦，OCR、celebrity、color 上低（E-SQZ7-5）；FastV 与随机的差距在 SVP 下减半（E-SQZ7-3） | 中后层仍在读取图像 token，深层只剩 18 个时物体更难被找到（M4 不成立的方向，推断）；FastV 在深层重新打分时位置偏差较小，覆盖改善、细节变差（推断） | SVP 的损失也可能来自第 3～15 层的 82/32 个，而不是第 16 层以后 | 没有只改后段的对照；没有 SVP 下 FastV 的保留位置 | 随机、时间表 2:49,16:18（P-SQZ7-2）；多段 kept-token 日志（P-SQZ7-3） |
| 子组、切片和边界 | 随机选择下同一题三次抽取的回答常不同，漏答次数两极化（E-SQZ7-6） | 正样本难度很不均匀：有一批正样本在任何 82 个左右的随机子集下都看不到 | 也可能是 POPE 题目本身的标注噪声（vanilla 漏掉的 313 个里，SparseVLM 和 VisionZip 也漏掉 86%～89%） | 没有物体框 | 同上 |
| 稳健性及替代解释 | 十次运行同一 commit；SVP 的逐层数与实测一致；随机种子间差异小（0.32） | 结论不是运行或种子造成的 | 一对分布的结论可能不能推广到其他分布 | 只有一对分布 | 若做后续，换一组层平均更低的分布复验 |
| 局部收益、整体收益和代价 | SVP 每题多用 3%～8% 时间（E 见记录） | 层平均相同时，SVP 在本设置下没有精度收益，计算还略多 | — | 只测了每题时间，没测 FLOPs | — |

## 学到什么

- **假设**：规则 1 成立（SparseVLM 的优势来自规则），规则 2 成立（FastV 的问题不依赖层分布），规则 3 的预测不成立且方向相反（随机选择下前多后少更差）。
- **已排除与仍无法区分的解释**：
  - 已排除：“SparseVLM-64 的优势主要来自它前 3 层保留全部 token 的层分布”。
  - 已排除：“只要换成 SparseVLM 的层分布，FastV 就能赶上随机”。
  - 仍无法区分：SparseVLM 的规则收益来自文本评分者还是回收；SVP 对随机的损失来自深层数量还是中层数量。
- **对 MODE_C-0002 的意义**：
  - C11（层分布与规则）回答完毕。
  - B4-b 的“层分布”方向在本设置下没有收益：前多后少不比平坦好，所以层分布本身不宜作为改进方向。
  - 规则的作用大，题目相关的选择（SparseVLM 的文本评分者）是候选机制。
  - 深层 token 数的作用（F20）与 PyramidDrop 的金字塔直觉相反，与 HiDrop 的出发点（浅层被动、融合从中层开始）方向一致；这两篇只看了摘要级材料。
- **保留 / 复验 / 回退 / 停止**：
  - 保留 F19～F22。
  - M4 的状态改为“在随机选择、层平均相同的条件下不成立”。
  - 停止把“层分布”作为单独的改进方向。
  - 规则方向（题目相关性）进入第 ⑦ 步候选。
- **最小、最能区分的下一次实验**：
  - 随机选择、时间表 2:49,16:18 的 POPE（1 次，约 20 分钟）：若接近随机 svp（62.2），SVP 的损失来自深层只剩 18 个；若接近随机 flat（67.7），损失来自中层。
  - 查 SparseVLM 代码能否关闭回收（P-SQZ7-1）。
- **回流 Deep Read**：F19～F22、P-SQZ7-1～4、M4 的状态更新与 E-SQZ7-1～6 交给 MODE_C-0002（回流 3，C11）。

## 同步清单

- [ ] experiment-results.md：登记 EXP-0007 的 10 次运行与本报告。
- [ ] 已发现的实验现象.md：新增 F19～F22，F18 一并登记；更新 M4 的状态。
- [ ] 论文素材库.md：候选表（同一分布下规则的作用；层平均相同时两种分布 × 两种规则的 2×2 表）。
- [ ] MODE_C-0002：回流 3（C11 回答，M4 状态更新，新增 F19～F22）。
- [ ] 第 ⑥ 步交叉分析：B4-b 更新。
- [ ] 最新实验分析报告.md、index.md。
