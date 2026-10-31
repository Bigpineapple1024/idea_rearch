# SQUEEZE_REPORT：EXP-0041（IDEA-0011 的开发：虚拟填补，CDPruner 底座，开发集）

（2026-11-01 补写成单独的报告：深挖在 2026-10-31 傍晚完成，结果先写在 [EXP-0041 记录](EXP-0041_record.md)的“深挖”一节；按研究流程与 squeeze-experiment 的交接协议补成单独的报告与回流输入包。数字与记录里的相同，都抄自封存的输出，没有新算。）

## 0. 交接元数据

- 包类型：SQUEEZE_REPORT；包版本：research-pipeline-contract v1.0
- 深挖 ID / 包 ID：SQZ-EXP-0041
- 来源 IDs：EXP-0041（R01～R08，都 completed）；复用的 EXP-0039-R01、R02（CDPruner-64）、EXP-0033-R03、R04（不剪枝）
- 生成日期：深挖 2026-10-31 傍晚；本报告 2026-11-01
- 输入实验记录：[EXP-0041_record](EXP-0041_record.md)（结果一节，DEC-0081）
- 证据状态：verified（数字）/ partial（机制）。脚本 `squeeze_0041.py`（sha256 de283a74…）写于看到它算的任何量之前（2026-10-31 17:49），运行前的预期在 notes.md；输出封存在 `code/output/EXP-0041/squeeze/`（SHA256SUMS e6d5a241…），已备份；脚本另收入代码分支的 sealed_copies（MANIFEST 第 72、73 行）。核对 14 项都 100%（重算的准确率与 0041.json 的表逐一相同）。
- 缺失证据：没有测注意力；计数题的平均数被少数很大的数左右（脚本没有设上限），只用比例。

## 1. 事实与证据边界

- 本报告追问：fill 对 CDPruner-64 的增益（+1.02 / +2.63）从哪里来：①是否靠代表拟合得好；②一个 token 复制到许多槽是否有害；③GQA 位置题的补回在哪类细分题型；④复制是否让模型把一个物体数成几个；⑤逐题翻转的方向。
- 实验单位：VQAv2 dev2 5192 题、GQA val400 4786 题（开发集，已多次使用）；每个配置 1 次。事后、探索性的计算，不改 EXP-0041 的读法；区间按图像重抽 1000 次（种子 0）。

## 2. 主要结果（抄自已封存的 0041.md）

| 比较（百分点） | VQAv2 dev2 | GQA val400 |
| --- | --- | --- |
| Q1 fill − CDPruner-64 | +1.02 [+0.25, +1.75] | +2.63 [+1.53, +3.69] |
| Q2 fill − shuffle | −0.18 [−0.70, +0.31] | +2.17 [+1.40, +2.96] |
| Q3 fill − orig | +4.62 [+3.73, +5.49] | +5.39 [+4.09, +6.63] |

## 3. 深挖发现

| 维度 | 发现 | 证据 | 稳健性 | 新 / 已知 |
| --- | --- | --- | --- | --- |
| 代表拟合的好坏（S1） | 每题按代表与剪掉的 patch 的平均余弦三等分：GQA fill − CDPruner-64 +2.19 [+0.61, +3.95] / +2.76 [+0.99, +4.50] / +2.94 [+0.93, +4.87]（低 / 中 / 高）；fill − shuffle +1.19 / +3.20 / +2.13；VQAv2 +0.87 / +0.99 / +1.18（单个三分之一里都没有检出） | squeeze_0041.md S1 | 三等分是逐题的 | 新（F110） |
| 一个 token 复制到多少槽（S2） | 按一个保留 token 最多代表的槽数三等分：GQA +2.37 / +3.26 / +2.27，VQAv2 +1.28 / +1.16 / +0.62；没有一致的趋势 | S2 | — | 新（F110） |
| GQA 位置题的细分（S3） | positionChoose（173 题）fill − CDPruner-64 **+27.17 [+18.47, +35.54]**，fill 91.91 与不剪枝 90.17 相当，只换编号（orig）就有 +23.12 [+15.30, +31.29]；positionQuery（217 题）+2.76 [−3.74, +9.22] 没有检出，但内容放错的 shuffle 只有 68.20（fill − shuffle +12.90 [+7.17, +19.25]）；positionVerifyC（61 题）+14.75 [+2.96, +26.67] | S3 | 题数少（54～217） | 新（F109） |
| 计数（S4） | VQAv2 计数题里六个运行与人的答案都读得成整数的 595 题：答的数比人的答案大的比例 fill 0.247、CDPruner-64 0.272、不剪枝 0.190；复制没有让模型多数 | S4 | 平均数不可用，只用比例 | 新（F110） |
| 逐题翻转（S5） | VQAv2 变好 348、变差 277 题，GQA 339、213 题；“否”题上 CDPruner-64 答“是”而 fill 答“否”的 84 / 65 题，反向 29 / 26 题；“是”题上 fill 变差 40 / 54、变好 19 / 25 | S5 | — | 补充（F107） |

## 4. 解释与不确定性

- **最支持的解释**（AI）：增益主要来自“没有缺口”（消除偏“是”，F107）与“编号对”（二选一的位置题，原始编号就给出绝对位置），与复制的内容有多像关系不大；问“什么在哪里”的题要把内容与位置对上，内容放错比连续编号还差（F109）。
- **替代解释**：开发集上的偶然（P14）；GQA 的题型标注；三等分随题目变（CDPruner 的选择依赖题目）。
- **运行前的预期**：“GQA 上 fill − shuffle 在余弦低的三分之一更大”不符合；“剪枝的运行答的数比不剪枝小”不符合（CDPruner-64 多数的比例比不剪枝高）；其余符合。
- **当前证据不支持的强结论**：“复制让一个物体出现多次”（M26 的替代解释，没有观察到）。

## 5. 知识沉淀

- **新现象**：F109、F110。**修正**：M26 的替代解释不支持；C41 的①（按代表数把槽的 V 打折）理由变弱。
- **需要复验**：位置题的细分在新数据上（EXP-0042 的 S5 已并排：位置题 +13.07 → +13.93、二选一 +9.93 → +8.58）。
- **更新位置**：已发现的实验现象（F109、F110）；EXP-0041 记录；MODE_C-0002 回流 32。

## 6. 下一步决策

1. 确认 EXP-0042 照计划运行（已完成，DEC-0082）。
2. 不做：按事后分组改读法；把“复制有害 / 有益”写成机制。

## 7. 回流到 deep-read Mode C 的输入包

```text
package_type: SQUEEZE_REPORT
squeeze_id: SQZ-EXP-0041
input_experiment_record_ids: [EXP-0041]
claim_scope: LLaVA-1.5-7B；CDPruner-64 与四种布局；开发集 VQAv2 dev2、GQA val400；每个配置 1 次；事后分析
data_quality_and_provenance_gaps: 核对都 100%；脚本写于看到其结果之前；没有测注意力；计数的平均数不可用
primary_result_with_uncertainty: fill − CDPruner-64 +1.02 / +2.63；fill − shuffle −0.18 / +2.17；fill − orig +4.62 / +5.39
process_dynamics: none
mechanism_and_intermediate_findings: 增益不随代表拟合变差而消失；positionChoose 的补回靠原始编号（+23.12），positionQuery 要内容放对（fill − shuffle +12.90）
errors_heterogeneity_and_boundary_conditions: VQAv2 上三等分都没有检出；位置题细分题数少
robustness_sensitivity_and_alternative_explanations: 开发集多次使用；题型标注
new_P: []
new_F: [F109, F110]
new_M: [M26 更新：复制让物体出现多次未观察到]
new_C_candidate: []
evidence_index: code/output/EXP-0041/squeeze（e6d5a241…）、analysis（b6e124c5…）
open_questions: 位置题的细分在新数据上
next_routes: [deep-read Mode C 回流 32（已并入）]
next_skill: deep-read Mode C
```
