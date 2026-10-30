# SQUEEZE_REPORT：EXP-0032（本流程中的 DivPrune-64，以及一层全看对多样性选择的作用）

## 0. 交接元数据

- 包类型：SQUEEZE_REPORT；包版本：research-pipeline-contract v1.0
- 深挖 ID / 包 ID：SQZ-EXP-0032
- 来源 IDs：EXP-0032（R01～R04，都 completed）；复用的 EXP-0028-R01、R02（VisionZip-64）、R19、R20（每题覆盖 16），EXP-0024-R06、R07（K = 1 路由），EXP-0030-R01、R02（K = 1 CLS 排序；其 cls.json 记录了 VisionZip 的 53 个主导 patch）
- 生成日期：2026-10-30（20 点后，EXP-0033 运行中写；预注册的分析已封存）
- 输入实验记录：[EXP-0032_record](EXP-0032_record.md)（结果一节，DEC-0061）
- 证据状态：verified。预注册的分析 `code/output/EXP-0032/analysis/0032.md`（SHA256SUMS 610b1536…）；本报告的新计算 `D:\6CJNM007\tmp\exp0032\squeeze\squeeze_0032.py`，输出、脚本与日志封存在 `code/output/EXP-0032/squeeze/`（SHA256SUMS 72cadb31…，已备份）。32 项核对都 100%：12 项是所用运行的状态与配置，12 项是重算的整体配对差与已封存的分析逐项相同，8 项是选择记录的完整性（含 K = 1 多样性的 47 个等于 DivPrune 的前 47 个、CLS 排序的 47 个等于其记录）。
- 缺失证据：VisionZip-64 的运行没有记每题保留的 patch，用 EXP-0030 的 cls.json 里同一个 CLS 分数的前 53 个代替（EXP-0030 的核对：与 VisionZip 自己的主导 token 逐题相同）；没有随机选择、也没有“强制铺开的 CLS 排序”的运行，“铺开”与“多样性”分不开。

## 1. 事实与证据边界

- 研究问题：EXP-0032 的预注册问题见记录；本报告追问三件事：①DivPrune-64 与 VisionZip-64 选的 patch 有多不同、铺得多开；②DivPrune 不排除填充区，留下的填充 patch 与准确率有什么关系；③K = 1 时多样性为什么好于 CLS 排序，差别是否随两者分歧变大。
- 实验单位：VQAv2 dev2 5192 题、1000 张图；GQA val400 4786 题、400 张图（开发集，已多次使用）；每个配置 1 次，贪心解码。
- 本报告的计算都是事后的、探索性的，分组多，没有做多重比较校正，不改 EXP-0032 的读法。区间：每个集内按图像重抽 1000 次（种子 0）。
- 空间的度量：24 × 24 的 patch 网格分成 6 × 6 个块（每块 4 × 4 个 patch），数被选的 patch 落在几个块里；有图像内容的块平均 32.3 / 32.5 个（其余整块在 LLaVA-1.5 补成方形的填充里）。

## 2. 主要结果（抄自已封存的 0032.md）

| 结果 | 对照 | 差 [95% 区间]（VQAv2 dev2 / GQA val400） | 证据 |
| --- | --- | --- | --- |
| DivPrune-64 | VisionZip-64（Q1） | +2.20 [+1.38, +3.03] / +1.63 [+0.34, +2.92] | 0032.md“按计划写定的读法” |
| K = 1 多样性（前 47 个） | DivPrune-64（Q2） | −1.51 [−2.23, −0.73] / +0.88 [−0.22, +2.01] | 同上 |
| K = 1 多样性 | K = 1 CLS 排序（同时报告） | +2.02 [+1.14, +2.88] / +2.38 [+1.02, +3.75] | 0032.md 配对差 |

## 3. 深挖发现

| 维度 | 发现 | 证据 | 稳健性 | 新 / 已知 |
| --- | --- | --- | --- | --- |
| 机制 / 中间量（S1） | **DivPrune 与 VisionZip 选的是很不一样的 patch**：DivPrune 的前 53 个与 VisionZip 的 53 个主导 patch 平均只相同 13.1 / 13.2 个（Jaccard 0.142 / 0.143）；K = 1 时多样性的 47 个与 CLS 排序的 47 个平均相同 11.2 / 11.3 个（Jaccard 约 0.14） | squeeze_0032.md 的 S1 | 两个集几乎相同 | 新（F75） |
| 机制 / 中间量（S1） | **DivPrune 铺得开得多**：在有图像内容的约 32 个块里，DivPrune 的前 53 个覆盖 26.4 / 26.5 个，VisionZip 的 53 个 21.6 / 22.0 个；K = 1 时多样性的 47 个 25.2 个，CLS 排序的 47 个 18.4 / 18.7 个 | 同上 | 两个集几乎相同 | 新（F75） |
| 机制 / 中间量（S1） | DivPrune-64 平均留 5.06 / 4.77 个整块落在填充里的 patch，95.1% / 94.6% 的题至少留一个；VisionZip 的 53 个里 3.99 / 3.98 个（F63）；CLS 排序按设计不留 | 同上 | 同上 | 新（F75） |
| 错误 / 异质性（S2） | 按 DivPrune-64 留下的填充 patch 数分组，DivPrune-64 − VisionZip-64 在各组都为正（VQAv2：0 个 +2.42、1～4 个 +1.13、5～9 个 +2.67 [+1.55, +3.72]、10 个以上 +4.87；GQA 相近）；没有看到“留的填充越多越差”。K = 1 放置在 VQAv2 上的损失集中在 5～9 个那组（−1.88 [−2.81, −0.95]，3105 题）；GQA 上没有填充的方形图（259 题）K = 1 放置 +9.65 [+4.18, +16.11] | S2 | 事后分组；小组区间很宽；方形图那一组只有 259 题 | 补充（方形图那一条只作线索） |
| 逐题（S3） | DivPrune-64 与 VisionZip-64 得分不同的题 15.5% / 15.6%（变好 469、变差 338；412 对 334）；K = 1 多样性与 CLS 排序 16.7% / 15.5%（499 对 370；428 对 314）；K = 1 多样性与 DivPrune-64 11.4% / 11.6% | S3 | 计数 | 补充 |
| 剂量（S4） | 两种 K = 1 选法按分歧（Jaccard）三等分，多样性 − CLS 排序：VQAv2 分歧最大 +2.88 [+1.40, +4.27]、中间 +1.11、最小 +1.82；GQA 三组都在 +2.3～+2.5。没有清楚的“分歧越大、差得越多” | S4 | 三组的 Jaccard 只在 0.10～0.18 之间变化，两种选法都很不同 | 补充 |

## 4. 解释与不确定性

- **最支持的解释**（AI，未检验）：DivPrune 的优势主要来自把 token 铺满整幅图（覆盖约 26 / 32 个有内容的块），而不是挑到更“重要”的 patch：它与 VisionZip 的主导 patch 只重合约四分之一，增益又集中在要看整幅图的 yes/no、计数题上（F72）；这与 F24（FastV 只强制铺开就追平随机）、F17 的方向一致。K = 1 时多样性好于 CLS 排序（F74），同样可以是“铺开”好于“集中在显著区域”。
- **仍可成立的替代解释**：
  - 多样性在特征空间里去掉了相似的 patch（冗余），铺开只是它的副产物；要分开两者，需要“强制铺开的 CLS 排序”或“分块随机”这样的对照（S4 的剂量不清楚，也不支持“挑得不同就更好”）。
  - VisionZip 的 10 个上下文 token 与 CLS 的作用没有在这里拆开。
  - 开发集上的偶然（P14）：两个开发集已多次使用。
- **当前证据不支持的强结论**：“DivPrune 的增益来自铺开”（没有铺开的对照）；“留填充 patch 有害”（各组都为正，没有去掉填充的运行）；GQA 方形图上 K = 1 的 +9.65（259 题，事后分组）。

## 5. 知识沉淀

- **新现象**：F75（DivPrune 与 VisionZip 选的 patch 只重合约四分之一，DivPrune 覆盖的图像块多约 5 个；DivPrune 平均留约 5 个填充 patch）、F76（K = 1 时多样性好于 CLS 排序，但差别不随两种选法的分歧变大，探索性）。
- **新限制**：F72 的增益来源（铺开还是去冗余）还分不开；DivPrune-64 作为下一轮的底座，它约 8% 的名额落在填充上。
- **已回答问题**：DivPrune 与 VisionZip 不是“选同一批、顺序不同”，而是选的东西本身差别很大。
- **需要复验**：①排除填充区的 DivPrune（名额让给图像 patch，预算不变）与 DivPrune-64 比；②“强制铺开的 CLS 排序”（例如按块轮流取 CLS 分数最高的）与 DivPrune-64、VisionZip-64 比，分开“铺开”与“多样性”；③在新抽的数据上确认 F72 的大小。
- **更新位置**：已发现的实验现象（F75、F76）；EXP-0032 记录末尾；结果总表 EXP-0032 一行；index；MODE_C-0002 回流 25（与 PAPER-0010～0020 的笔记一起）。

## 6. 下一步决策

1. **不改正在进行的实验**：EXP-0033 的计划 v1、v1.1 在本报告之前写定。
2. **Mode C 候选（交回流 25，不在这里设计）**：①排除填充区的 DivPrune（P15 × F75）；②铺开与多样性的拆分对照（F72 × F75）；③以 DivPrune-64 为底座、在被路由的题上加题目信息（EXP-0033 的 X3 先检验）；④DivPrune-64 在新数据上的确认。
3. **不做**：按事后分组（方形图、填充数）改任何读法。

## 7. 回流到 deep-read Mode C 的输入包

```text
package_type: SQUEEZE_REPORT
squeeze_id: SQZ-EXP-0032
input_experiment_record_ids: [EXP-0032]
claim_scope: LLaVA-1.5-7B；层平均 64（DivPrune-64、VisionZip-64）/ 63.5（K = 1）；开发集 VQAv2 dev2（5192 题）、GQA val400（4786 题）；每个配置 1 次；事后分析
data_quality_and_provenance_gaps: 32 项核对都 100%；VisionZip 的保留用同一 CLS 分数的前 53 个代替（逐题相同已核对）；没有铺开或随机的对照；开发集多次使用
primary_result_with_uncertainty: DivPrune-64 − VisionZip-64 +2.20 [+1.38, +3.03] / +1.63 [+0.34, +2.92]；K = 1 多样性 − DivPrune-64 −1.51 [−2.23, −0.73] / +0.88 [−0.22, +2.01]
process_dynamics: none（单次推理）
mechanism_and_intermediate_findings: DivPrune 的前 53 个与 VisionZip 的 53 个只重合约 13 个；DivPrune 覆盖约 26.5 / 32 个有内容的块，VisionZip 约 21.8，K = 1 CLS 排序约 18.5；DivPrune 平均留约 5 个填充 patch
errors_heterogeneity_and_boundary_conditions: 按填充数分组 DivPrune − VisionZip 都为正；K = 1 放置在 VQAv2 的 5～9 个填充组上损失最大；GQA 方形图上 K = 1 放置 +9.65（259 题，事后）
robustness_sensitivity_and_alternative_explanations: 铺开与去冗余分不开；VisionZip 的上下文 token 没有拆开；开发集上的偶然
new_P: [P16 候选：DivPrune-64 约 8% 的名额落在整块填充上（F75，与 P15 同类）]
new_F: [F75, F76]
new_M: [M 候选（AI）：多样性选择的增益主要来自把 token 铺满整幅图（覆盖），对“有没有、有几个”的题最有用；与 F24、F17 同向]
new_C_candidate: [排除填充区的 DivPrune；铺开与多样性的拆分对照；DivPrune-64 底座上的题目路由（EXP-0033 X3）；DivPrune-64 在新数据上的确认]
evidence_index: code/output/EXP-0032/analysis（610b1536…）、squeeze（72cadb31…）
open_questions: 增益来自铺开还是去冗余；排除填充区是否有帮助；新数据上的大小
next_routes: [deep-read Mode C 回流 25（与 PAPER-0010～0020 的笔记一起）]
next_skill: deep-read Mode C
```
