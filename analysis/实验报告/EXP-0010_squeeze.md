# SQUEEZE_REPORT：覆盖与去位置的深挖（EXP-0010）

状态：完成（2026-10-24）。
报告 ID SQZ-EXP-0010 / 实验 ID EXP-0010 / 输入记录：analysis/实验报告/EXP-0010_record.md（2026-10-24 完成版）/ 日期 2026-10-24。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0010
squeeze_id: SQZ-EXP-0010
created_at: 2026-10-24
source_ids: [EXP-0010, EXP-0008, EXP-0007, EXP-0006, EXP-0002, SQZ-EXP-0008, MODE_C-0002, 1A4B-0001]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0010_record.md
- code/output/EXP-0010/analysis/（kept_stats.md、kept_summary.json、paired_bootstrap.txt、details.md、details.json；SHA256SUMS 的 sha256 6acc04cd…，附 bootstrap_ci.py、analyze_kept_tokens.py、analyze_missed_objects.py、analyze_0010_details.py、compare_0010.sh、run_vision_0010.sh 副本，已备份）
- code/output/EXP-0002/deepdive2/（mme_yes_ratio.md、.json；SHA256SUMS 的 sha256 d433815a…，附 analyze_mme_yes_ratio.py，已备份）
- code/output/EXP-0010/EXP-0010-R01、R02/kept_tokens.json
missing_evidence:
- 只有 64 档、一种分块粒度（6×6）
- 没有“随机选择在第 0 层前剪到 64 个”的对照，不能把与 VisionZip 的差距拆成分布与规则
- FEATHER 的正文与代码没有读，只能比较方向
- MME 子任务与漏答分析没有区间
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0010]
claim_scope: LLaVA-1.5-7B，FastV K=2、keep 30（层平均 64），排序按 6×6 块轮流取，分数可先除以 EXP-0008 的位置先验；完整 POPE、MME，pope_img20 子集上的保留统计
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“主要结果”
process_dynamics: 不适用（推理诊断）
mechanism_and_intermediate_findings: 见“深度分析”第 3 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 2、4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 5 行
new_P: [P-SQZ10-1, P-SQZ10-2, P-SQZ10-3]
new_F: [F24]
new_M: [M9 状态更新：空间集中是 F8 的主要机制；M1 状态更新：位置偏差是集中的原因之一，单独去掉它不如直接强制覆盖]
evidence_index: 见 mode_C_input_bundle.E
open_questions: [与 VisionZip 的 POPE 差距（12.4）来自层分布还是选择规则, 块内换成题目相关的排序能否补上 POPE 差距, 结论在 128、192 档与其他分块粒度下是否成立]
next_routes: [随机选择在第 0 层前剪到 64 个的 POPE（需要让随机选择支持 K=0）, 块内用题目相关的排序（C12 的检验）, deep-read Mode C 回流, 第 ⑦ 步排序]
mode_C_input_bundle:
  source_ids: [EXP-0010, SQZ-EXP-0010]
  P:
  - id: P-SQZ10-1
    text: "去位置 + 覆盖的 FastV-64 在 POPE 上仍比 VisionZip-64 低 12.4：差距来自层分布（FastV 把层平均 64 中的 36 个花在前 2 层，第 2 层以后只剩 30 个；VisionZip 每层 64 个），还是来自选择规则（CLS 注意力 + 合并）"
    evidence: [E-SQZ10-3, E-SQZ7-2]
    status: UN
  - id: P-SQZ10-2
    text: "分块后选择仍与问题无关（Jaccard 0.945）：块内换成题目相关的排序，能否补上与 SparseVLM 的 POPE 差距（13.0）"
    evidence: [E-SQZ10-1, E-SQZ7-1]
    status: UN
  - id: P-SQZ10-3
    text: "覆盖的作用在 128、192 档和其他分块粒度下是否同样大"
    evidence: [E-SQZ10-2]
    status: UN
  F:
  - id: F24（新）
    text: "FastV-64（K=2、keep 30）的排序按 6×6 块轮流取（只强制覆盖），POPE +25.28 [23.82, 26.87]（36.19 → 61.47），追平随机（+1.21 [−0.30, 2.80]），并比只去位置高 4.06 [2.83, 5.27]；MME-P +301 [231, 370]，比随机高 112.6 [36.1, 191.0]，与 SparseVLM-64 相同（1280.7 vs 1279.6）。再去掉位置项：POPE 64.66，比随机高 4.40 [2.81, 5.83]，MME-P 比随机高 85.2 [5.3, 164.7]。选择仍与问题无关（Jaccard 0.945 / 0.934）；POPE 上仍比 VisionZip-64 低 12.4"
    evidence: [E-SQZ10-1, E-SQZ10-2, E-SQZ10-3]
    status: DE
  M:
  - id: M9（状态更新）
    text: "注意力高的 token 在空间上集中、覆盖不到题目所问的物体：强制覆盖一项就补回 FastV 与随机之间的全部 POPE 差距，是 F8 的主要机制"
    evidence: [E-SQZ10-2]
    status: 结果 DE
  - id: M1（状态更新）
    text: "位置偏差把选择集中到图像底部，是造成集中的原因之一；但单独去掉位置项（57.41）不如直接强制覆盖（61.47）；两者合用最好（64.66）"
    evidence: [E-SQZ10-2, E-SQZ8-4]
    status: 结果 DE
  E:
  - id: E-SQZ10-1
    text: "保留统计（keep 30，pope_img20）：只覆盖 / 去位置 + 覆盖的块覆盖率都是 83.3%；平均行 12.71 / 11.00；底部 1/4 占比 31.8% / 20.1%；同图不同题 Jaccard 0.945 / 0.934；子集准确率 69.3% / 70.4%"
    location: code/output/EXP-0010/analysis/kept_stats.md、kept_summary.json
  - id: E-SQZ10-2
    text: "配对 bootstrap（1000 次）：POPE R03 − EXP-0008-R05 +4.06 [2.83, 5.27]，R03 − FastV-64 +25.28 [23.82, 26.87]，R03 − 随机 +1.21 [−0.30, 2.80]，R05 − 随机 +4.40 [2.81, 5.83]，R05 − R03 +3.19 [2.20, 4.21]；MME R04 − EXP-0008-R06 +95.8 [29.5, 157.0]，R04 − FastV-64 +301.2 [231.2, 370.1]，R04 − 随机 +112.6 [36.1, 191.0]，R06 − 随机 +85.2 [5.3, 164.7]，R06 − R04 −27.4 [−84.3, 27.5]"
    location: code/output/EXP-0010/analysis/paired_bootstrap.txt
  - id: E-SQZ10-3
    text: "召回 44.9%（R03）、48.5%（R05），精确率 94.3%～99.4%；逐正样本漏答比例 R05 − 随机 −4.3 个百分点，R05 − VisionZip-64 +16.5（vanilla 答对处 +19.4）；MME 子任务 R04 − FastV-64 existence +65、count +55、position +48.3、color +38.3，R06 − 随机 celebrity +34.1、count −18.3"
    location: code/output/EXP-0010/analysis/details.md、details.json
  evidence_locations: [code/output/EXP-0010/analysis/SHA256SUMS, code/output/EXP-0002/deepdive2/SHA256SUMS]
```

## 证据审计

- **完整性**：R03、R05 各计分 8910 题，R04、R06 各计分 2374 题；R01、R02 各 358 次预填充，块覆盖率核对在完整评测之前通过。
- **对照的有效性**：配对比较用的 EXP-0002、EXP-0008 运行与本实验题目、计分完全相同；只去位置的对照（EXP-0008-R05、R06）与本实验用同一个位置先验文件。
- **血缘**：六次运行都在 commit 399b2d0、无未提交改动；analysis 目录已封存并备份。
- **预先计划与事后提出**：三条规则和比较对象在计划 v1 中写定。与 VisionZip-64、SparseVLM-64 的比较、召回拆分、漏答分析、MME 子任务表和 MME 的 yes 比例都是看到结果后补充的，标为探索性。

## 主要结果

| 结果 | 对照 | 差异 | 不确定性 | 实际意义 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 只覆盖（规则 1） | 只去位置 | POPE +4.06，MME +95.8 | 区间都不含 0 | 覆盖比去位置更重要 | E-SQZ10-2 |
| 只覆盖 | FastV-64 / 随机-64 | POPE +25.28 / +1.21；MME +301.2 / +112.6 | 与随机的 POPE 差区间跨 0 | 补回全部差距 | E-SQZ10-2 |
| 去位置 + 覆盖（规则 3） | 随机-64 | POPE +4.40，MME +85.2 | 区间都不含 0 | 超过随机 | E-SQZ10-2 |
| 与强方法的差距（探索性） | VisionZip-64 / SparseVLM-64 | POPE −12.4 / −13.0；MME −83.5 / +1.1 | 未做区间 | POPE 上差距仍大 | 记录“结果”表 |

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 只覆盖已追平随机并比只去位置高 4.1；两者合用比随机高 4.4（E-SQZ10-2） | F8 的主要机制是空间集中（M9）；位置偏差是集中的原因之一（M1） | 分块本身引入了“位置均匀”的先验，与“去位置”并非完全独立的两件事 | 只有一种分块粒度 | 换 4×4 或 8×8 分块（P-SQZ10-3） |
| 错误和失败模式 | 召回从 22.3% 升到 44.9%～48.5%，精确率不变；与 VisionZip 的差距在 vanilla 能答对的正样本上（E-SQZ10-3） | 覆盖修好后，剩下的漏答仍是“本来能看到的物体”，与 F18、F19 一致 | — | 没有物体框 | COCO 物体框（需下载许可） |
| 机制、消融和交互 | 分块后选择仍与问题无关，块内仍偏向第 23 行（E-SQZ10-1）；MME 子任务上覆盖修好了存在、计数、位置、颜色，celebrity 上仍比随机高 34（E-SQZ10-3） | 打分里的位置/显著性成分决定块内取哪个 patch；注意力在细节上的优势在覆盖之后保留下来 | MME 上与 SparseVLM 相等可能是巧合（层分布不同） | 子任务题数少 | 块内用题目相关的排序（P-SQZ10-2） |
| 子组、切片和边界 | 只有 64 档 | 名额少时覆盖的作用最大（与 F21 一致） | 名额多时覆盖本来就够，分块作用变小 | 只有一个预算 | 128、192 档（P-SQZ10-3） |
| 稳健性及替代解释 | 六次运行同一 commit；FastV 确定性；随机种子间差异小 | 结论不是运行或种子造成的 | — | — | — |
| 局部收益、整体收益和代价 | 每题时间与 FastV-64 相同 | 分块是排序上的改动，没有额外开销 | — | — | — |

## 学到什么

- **假设**：规则 1 成立（收益主要来自更分散），规则 3 成立（去位置 + 覆盖超过随机）。
- **已排除与仍无法区分的解释**：
  - 已排除：“FastV 不如随机主要是因为位置偏差本身”——只强制覆盖、不动位置，就补回了全部差距。
  - 已排除：“K=2 时第 1 层注意力没有可用的内容信号”——去位置 + 覆盖后超过随机。
  - 仍无法区分：与 VisionZip 的 POPE 差距来自层分布还是选择规则。
- **对第 ⑦ 步的意义**：
  - B4-a 的“去位置 + 覆盖”在本项目的 baseline 上有效且效应大（POPE +28.5，MME +274），但 FEATHER 已提出同样的两个部件，只能作为独立复核和强 baseline，不作为新方向。
  - 剩下的差距指向两个候选：层分布（FastV 在前 2 层花掉了层平均的一半以上）与题目相关的选择（C12）。
- **保留 / 复验 / 回退 / 停止**：
  - 保留 F24；M1、M9 状态更新。
  - C9 回答完毕。
  - “去位置 + 覆盖”的 FastV-64（fastv64_prior_strat）作为后续实验的强对照。
- **最小、最能区分的下一次实验**：随机选择在第 0 层前剪到 64 个（每层 64 个，与 VisionZip 相同的分布）的 POPE。
  - 若接近 VisionZip-64（77.0）：VisionZip 的优势主要来自层分布。
  - 若接近随机-64（60.3）：VisionZip 的优势主要来自规则。
- **回流 Deep Read**：F24、P-SQZ10-1～3、M1 与 M9 的状态更新和 E-SQZ10-1～3 交给 MODE_C-0002（回流 3，C9）。

## 同步清单

- [ ] experiment-results.md：登记 EXP-0010 的 6 次运行与本报告。
- [ ] 已发现的实验现象.md：新增 F24，更新 F8、F15、F17。
- [ ] 论文素材库.md：候选表（FastV-64、只去位置、只覆盖、两者合用、随机、VisionZip 的 POPE/MME，带配对区间）。
- [ ] MODE_C-0002：回流 3（C9 回答，M1、M9 状态更新，新增 F24）。
- [ ] 1a-4b_FastV不如随机选择.md：3b 的结果。
- [ ] 第 ⑥ 步交叉分析：B4-a 更新。
- [ ] 最新实验分析报告.md、index.md。
