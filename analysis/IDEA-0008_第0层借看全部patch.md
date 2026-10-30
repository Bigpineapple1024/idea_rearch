# IDEA-0008：第 0 层借看全部 patch，按 FLOPs 对齐多留（第 ⑨～⑫ 步第八轮）

- IDEA ID：IDEA-0008；包类型：IDEA_PACKAGE（research-pipeline-contract v1.0）
- 日期：2026-10-31（凌晨，EXP-0033 与 SQZ-EXP-0033 之后，MODE_C-0002 回流 25）
- 来源：MODE_C-0002 回流 25 的 C29（P17 × F78 × M21）。
- 用户的要求：“我目标是创造一个新算法或者方法或者模块 且新颖有效提升大”（2026-10-25）；“我要使用max 你一直跑吧直到都修复完 且我没跑完的研究跑完并有了成果”（2026-10-30，DEC-0058）。
- 选用模式：Pattern 3（假设挑战）加 Pattern 5（系统/架构审查）。被挑战的默认做法：同一层平均预算下，K = 1 的第 0 层要把全部 576 个 patch 完整算一遍，所以第 1 层起只能留 47 个。替换的只有一处：被剪的 patch 在第 0 层只提供 key / value（它们自己的输出本来就不用），省下的计算让第 1 层起留 61 个。
- 状态（2026-10-31）：设计完成；CPU 核对 check_peek.py 29/29（三种模式的预填充与解码都等于手算的参照），回归 check_divprune.py 26/26、check_k1cls.py 32/32、check_zoom.py 77/77；分析脚本 analyze_0034.py 用替身数据空跑通过（替身不是结果）。开发实验 [EXP-0034](实验报告/EXP-0034_record.md)（6 次）。
- 证据状态：partial。F78 是开发集上的分组结果；“第 0 层借看”的机制没有检验过。

## 为什么这样做

- **64 档的缺口在哪里**：不剪枝 77.91 / 70.08，DivPrune-64 74.13 / 63.83；GQA 上属性题、二选一题分别低 9.92、11.42（F77）。
- **K = 1 补回了这一块，却在别处吃亏**：同样的多样性选择放到 K = 1（第 0 层看全部、第 1 层起留 47 个），GQA 属性 +4.93 [+2.62, +7.07]、二选一 +6.62 [+2.77, +10.79]，VQAv2 计数 −2.88、颜色 −4.15（F78）。按 token 数计的层平均（63.5）里，第 0 层的 576 个占了 18 个名额。
- **这 18 个名额大多是白花的**：K = 1 时，被剪的 515 个 patch 在第 0 层的输出不再被使用，真正用到的只有它们在第 0 层的 key、value（让题目与留下的 token 看见它们）。只算 K、V 投影约是一个 token 一层计算量的 1/6（LLaMA-7B：2d² 对 4d² + 3dm）。按 FLOPs 对齐 64 个 × 32 层：32 × 61 + 515 × 0.166 = 2037 ≤ 2048，第 1 层起可以留 61 个（DivPrune 的前 61 个）。
- **要检验的是**：在约相等的 FLOPs 下，“第 0 层借看全部 + 留 61”能否高于 DivPrune-64；高出的部分是否来自“第 0 层看过被剪的 patch”（对照：同样的 61 个与位置，第 0 层看不到它们）。

## 方法

- 选择：DivPrune 的最大最小多样性（divprune.py，按论文的 Algorithm 1 自写），在投影层输出上取 61 个，在第 0 层之前选好。
- 第 0 层：题目、系统提示与留下的 61 个照常计算；被剪的 515 个只作 key / value（本实验的实现照常算它们并丢掉输出，结果与只算 K / V 相同，计时以后另做）。
- 第 1～31 层：只留系统提示、61 个 patch（原始位置编号）与题目。
- 解码：生成的 token 在任何一层都看不到被剪的 patch（第 0 层用遮罩），所以 KV 缓存除第 0 层外每层都只有 61 个图像 token；只算 K / V 的实现可以在预填充后把它们从缓存删掉。
- 两个对照（同样的 61 个、同样的位置）：**nopeek**：第 0 层谁也看不到被剪的 patch（等于把 DivPrune 的 61 个按原始位置交给 LLM）；**peektext**：只有题目看得到（分开“题目读到全部”与“留下的 token 吸收邻居”两条路）。

## 假设与机制（推断）

- **假设**：属性、比较类题需要的细节，题目 token 在第 0 层就能从全部 patch 里读到；之后的层主要靠题目 token 与留下的 token 推理，不必再看被剪的 patch（M21）。
- **替代解释与风险**：K = 1 的好处来自保留原始位置编号（F6），而不是看过全部（nopeek 对照同样保留原始位置，能分开）；开发集上的偶然（P14）；VQAv2 上 61 个仍比 64 个少 3 个；GQA 的属性、二选一题是看过 EXP-0032 的分组之后挑出来的。

## 可检验的预测（推断，用于事后对照，不是判据）

- peek − DivPrune-64：VQAv2 −0.5～+0.5，GQA +1.0～+2.5；增益集中在 GQA 的属性、二选一题。
- peek − nopeek：VQAv2 +0.5～+1.5，GQA +1.5～+3.0。
- nopeek − DivPrune-64：−0.5～+0.2（少 3 个 token、位置编号不同）。
- peektext 介于 nopeek 与 peek 之间，更接近 peek（题目读到全部是主路）。

## 最小验证

见 [EXP-0034](实验报告/EXP-0034_record.md) 的计划 v1：VQAv2 dev2、GQA val400，6 次（peek、nopeek、peektext 各 2 次）；复用 DivPrune-64、K = 1 留 47、VisionZip-64、不剪枝。Q1（主要）peek − DivPrune-64；Q2（机制）peek − nopeek。Q1 两集都在 0 以上，或一集在 0 以上且两集平均 ≥ +0.5 时，先抽新的确认集、写确认计划并提交，再运行（P14）。

## 相近工作与新颖性

- 检查范围：PAPER-0004～0006、0008～0020 的笔记（2026-10-31 只用这些；没有联网检索，用户睡觉期间不用可能弹授权的工具）。
- **不新的部分**：K = 1 的放置（DivPrune Table 3、ShaRP、本项目 CL-11）；按 FLOPs 或层平均比较剪枝层（VScan、VisionTrim、DivPrune）；多样性选择本身（DivPrune）。
- **可能新的部分**（UN，要检索）：被剪的 token 只在第一层作 key / value、用省下的计算多留 token，并用“同样的 token、看不到被剪的”对照证明好处来自借看。可能相近的方向：冻结或跳过视觉 token 层计算的工作（例如只让视觉 token 作 key / value 的交叉注意力式 MLLM、跳过视觉 token 的 FFN 的工作），需要在用户醒来后用 WebSearch 查（DEC-0064）。
- **结论**：组合与预算计法层面的增量；能成立的贡献（推断）是“同样的 FLOPs 下第 0 层借看全部 + 多留”的证据与机制对照。查新完成前不写“首次”。

## 风险

- 计时：本实验的实现在第 0 层完整计算全部 576 个，每题秒数会比 DivPrune-64 慢，不能代表只算 K / V 的实现；FLOPs 是按结构算的。
- KV 缓存：本实现第 0 层的缓存保留全部（解码时遮住）；只算 K / V 的实现可以删掉，口径要写清。
- 只在 LLaVA-1.5-7B、两个开发集上；题目在图像之后的模板才适用。

## 预算

- IDEA-0008 单独计：试验阶段 30 次运行或 20 GPU 小时。
- EXP-0034：6 次，约 1.6 GPU 小时（K = 1 每题约 0.17 秒）。
- 确认（只在 EXP-0034 满足条件时）：另写计划。

## IDEA_PACKAGE

```text
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
idea_id: IDEA-0008
source_matrix_id: MODE_C-0002（回流 25）
selected_C_id: C29
source_P_ids / source_F_ids / source_M_ids / source_E_ids: P16、P17 / F73、F77、F78 / M21 / EXP-0032 analysis（610b1536…，按题型的分组）、EXP-0033 analysis（d1059016…）、SQZ-EXP-0033（b77c21c3…、0d4f57e7…）
research_question: 同样的 FLOPs（被剪的 patch 在第 0 层只算 K、V），第 0 层借看全部 patch、第 1 层起留 DivPrune 的 61 个，能否高于 DivPrune-64；好处是否来自第 0 层看过被剪的 patch
hypothesis_and_mechanism: 题目 token 在第 0 层从全部 patch 读到属性、比较类题需要的细节，之后不必再看被剪的 patch；多留的 14 个（相对 K = 1 留 47）补回计数、颜色题（AI）
testable_predictions: peek − DivPrune-64 VQAv2 约 −0.5～+0.5、GQA 约 +1.0～+2.5；peek − nopeek 两集都为正，GQA 更大；peektext 接近 peek
design_and_controls: DivPrune-64（底座，FLOPs 约相等）；nopeek（同样的 61 个与位置，第 0 层看不到被剪的）；peektext（只有题目看得到）；K = 1 留 47（EXP-0032）；VisionZip-64、不剪枝作参照
sample_materials_or_experimental_units: VQAv2 dev2（5192 题、1000 张图）、GQA val400（4786 题、400 张图）；确认用新抽的数据
primary_outcome_and_estimand: 全部题准确率的配对差（peek − DivPrune-64），按图像重抽 1000 次（种子 0）；两集平均
secondary_outcomes: peek − nopeek（机制）；nopeek − DivPrune-64；peektext 对 nopeek、peek；按题型分组；FLOPs（按结构）、每题秒数、峰值显存
analysis_plan_and_uncertainty: analyze_0034.py（替身数据空跑通过）；95% 区间；分组为描述
resources_time_cost_and_risk: 6 次，约 1.6 GPU 小时；上限 30 次或 20 GPU 小时
ethics_safety_privacy_or_compliance: 只用公开 benchmark；不涉及个人数据
stopping_and_interpretation_rules: Q1 两集都在 0 以上，或一集在 0 以上且两集平均 ≥ +0.5 → 抽新的确认集、写确认计划、提交后运行；Q1 为正但区间都含 0 → 记录，交用户决定是否扩大开发集；Q1 ≤ 0 → 不确认，Q2 照常报告机制
status: formal-validation（开发集阶段）
next_skill: record-experiment
```
