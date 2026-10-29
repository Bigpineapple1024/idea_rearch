# SQUEEZE_REPORT：EXP-0030（一层全看，再按编码器排序的开发集运行，IDEA-0007）

## 0. 交接元数据

- 包类型：SQUEEZE_REPORT；包版本：research-pipeline-contract v1.0
- 深挖 ID / 包 ID：SQZ-EXP-0030
- 来源 IDs：EXP-0030（R01～R06，都 completed）；复用的 EXP-0028-R01、R02（VisionZip-64）、R13、R14（VisionZip-64 + 覆盖 16）
- 生成日期：2026-10-29（13 点后，EXP-0031 运行中写；没有看过 EXP-0031 的任何结果）
- 输入实验记录：[EXP-0030_record](EXP-0030_record.md)（结果一节，DEC-0057）
- 证据状态：verified。预注册的分析 `code/output/EXP-0030/analysis/0030.md`（SHA256SUMS c6a48c42…）；本报告的新计算 `D:\6CJNM007\tmp\exp0030\squeeze_0030.py`，输出与脚本副本封存在 `code/output/EXP-0030/squeeze/`（SHA256SUMS 5769d507…，已备份）。
- 缺失证据：VisionZip-64 的运行没有记每题的 53 个主导 patch，这里用 EXP-0030 的 cls.json 里同一个 CLS 分数的前 53 个代替（分析里 102 项核对之一：它与 VisionZip 自己的主导 token 逐题相同，用的是 EXP-0028-R13、R14 的覆盖记录）；没有“在 LLM 之前按 CLS 留 47 个”的运行，“第 0 层全看”与“少 17 个 token”分不开。

## 1. 事实与证据边界

- 研究问题：EXP-0030 的预注册问题见记录；本报告追问两件事：①一层全看 + CLS 排序（`k1cls`）为什么在 VQAv2 上比 VisionZip-64 低；②不路由的题上，所选（`k1qcls_c16`）为什么比拼接的那一半（VisionZip-64 + 覆盖 16）低。
- 实验单位：VQAv2 dev2 5192 题、1000 张图；GQA val400 4786 题、400 张图；每个配置 1 次，贪心解码。
- 本报告的计算都是事后的、探索性的，分组很多，没有做多重比较校正；区间按图像重抽 1000 次（种子 0），与 analyze_0030 相同。

## 2. 主要结果（抄自已封存的 0030.md）

| 结果 | 对照 | 差 [95% 区间] | 证据 |
| --- | --- | --- | --- |
| 所选 `k1qcls_c16` | VisionZip-64 | +1.18 [+0.30, +2.06] / +1.38 [+0.19, +2.62] | 0030.md“选择与判读” |
| `k1cls`（一层全看 + CLS 排序） | VisionZip-64 | −1.33 [−2.03, −0.67] / +0.13 [−0.99, +1.17] | 同上 |
| 所选，不路由的题 | VisionZip-64 + 覆盖 16 | −1.60 [−2.56, −0.65] / +0.38 [−0.94, +1.76] | 0030.md 分组配对差 |

## 3. 深挖发现

| 维度 | 发现 | 证据 | 稳健性 | 新 / 已知 |
| --- | --- | --- | --- | --- |
| 机制 / 中间量（S1） | `k1cls` 留下的 47 个，平均 46.4 个在 VisionZip 自己的 53 个主导 patch 里；47 个都在其中的题占 79.6% / 78.2%。两者用同一个 CLS 分数，差别只在 `k1cls` 不留整块落在填充里的 patch | squeeze_0030.md 的 A | 两个集相近 | 新 |
| 机制 / 中间量（S2） | VisionZip 的 53 个主导 patch 里，平均 3.99 / 3.98 个整块落在填充里（LLaVA-1.5 把图补成方形的那部分）；至少有一个的题占 90.1% / 90.3% | 同上 | 两个集相近 | 新（F63） |
| 错误 / 异质性（S3） | VQAv2 上 `k1cls` 的差距与选了哪些 patch 无关：47 个都在 53 个里的题 −1.47 [−2.30, −0.74]（4131 题），不都在的 −0.79 [−2.14, +0.74]；53 个里没有填充 patch 的题差距最大 −3.81 [−5.99, −1.62]（512 题），有的 −1.06 [−1.82, −0.35]（4680 题）。GQA 上各组都含 0 | squeeze_0030.md 的 B | 事后分组；没有填充 patch 的组只有 512 题 | 新（F64） |
| 错误 / 异质性（S4） | VQAv2 上的差距分散在各题型（is the −2.28、what is the −1.66、what color is the −1.23，区间都含 0），只有 what +1.70 为正；不路由的题上所选 − VisionZip-64 + 覆盖 16 也分散（what color is the −2.56、what is the −2.82） | 同上 | 单个题型的区间都含 0 | 补充 |
| 错误 / 异质性（S5） | GQA 上按语义类型方向不同：`k1cls` − VisionZip-64 在 attr 上 +3.20 [+1.01, +5.38]（1563 题），rel −1.53 [−2.92, −0.12]（2228 题），cat −3.15 [−6.47, +0.00]；不路由的题上所选 − VisionZip-64 + 覆盖 16：attr +3.52 [+0.38, +6.56]，cat −3.33 [−5.93, −0.84] | 同上 | 事后分组，多重比较 | 新（F65，探索性） |

## 4. 解释与不确定性

- **最支持的解释**（AI，未检验）：`k1cls` 与 VisionZip-64 留的几乎是同一批主导 patch，差在 VisionZip 多出的 6 个主导 patch、10 个合并其余 patch 的上下文 token 和 CLS token；第 0 层看全部 576 个补不回这些。53 个里没有填充 patch 时 VisionZip 的有效 token 最多（约 64 个都有用），`k1cls` 差得最多；有填充 patch 时 VisionZip 自己浪费约 4 个，差距变小。这与“token 数比第 0 层全看更要紧”一致。
- **仍可成立的替代解释**：没有填充 patch 的图是另一类图（接近方形、构图不同），差距大可能来自图的类型而不是 token 数；位置编号不同（FastV 一族保留原始位置，VisionZip 压缩，F6）；GQA 上没有这一差距，说明它依赖题目类型。
- **当前证据不支持的强结论**：“一层全看没有用”（没有在 LLM 之前留 47 个的对照）；“VisionZip 浪费的填充 patch 有害”（没有去掉填充 patch 的 VisionZip 运行）；GQA 语义类型上的差别（多重比较）。

## 5. 知识沉淀

- **新现象**：F63（VisionZip 的主导 patch 里约有 4 个整块落在填充里，约 90% 的题至少一个）、F64（`k1cls` 在 VQAv2 上的差距与选的 patch 无关、在 VisionZip 不浪费填充时最大）、F65（GQA 上按语义类型方向不同，探索性）。
- **新限制**：IDEA-0007 的“按编码器排序”在本流程里与 VisionZip 选的是同一批 patch，它的价值只能来自“第 0 层全看”与“不留填充”，而这两者在 VQAv2 上抵不过少 17 个 token。
- **已回答问题**：`k1cls` 的差距不是因为排序选错了 patch。
- **需要复验**：F63 的预算浪费是否有害——去掉填充 patch 的 VisionZip（名额让给下一个 CLS 分数最高的非填充 patch）与 VisionZip-64 比；“第 0 层全看”的作用——与在 LLM 之前按 CLS 留 47 个比。两者都是便宜的单次运行，先登记为 Mode C 的候选，不在本报告里设计方法。
- **更新位置**：已发现的实验现象（F63～F65）；EXP-0030 记录的结果一节末尾；结果总表 EXP-0030 一行的深挖一栏；index。MODE_C-0002 的回流与 EXP-0031 的结果一起做（回流 24）。

## 6. 下一步决策

1. **不改正在进行的确认**：EXP-0031 的计划（v1、v1.1）都在本报告之前写定，本报告不影响它的运行与判读。
2. **Mode C 候选（交回流 24，不在这里设计）**：①填充感知的 VisionZip（F63）；②K = 1 时“第 0 层全看”的对照与补上下文 token（F64）；③拼接在新数据上的结果（EXP-0031 计划 v1.1 的 S1、S2）。
3. **不做**：按事后分组挑子集写结论；在 EXP-0031 出结果之前改它的分析。

## 7. 回流到 deep-read Mode C 的输入包

```text
package_type: SQUEEZE_REPORT
squeeze_id: SQZ-EXP-0030
input_experiment_record_ids: [EXP-0030]
claim_scope: LLaVA-1.5-7B；K = 1、留 47 个（层平均 63.5）对 VisionZip-64（CLS + 53 个主导 + 10 个上下文）；开发集 VQAv2 dev2（5192 题）、GQA val400（4786 题），每个配置 1 次；只是开发集上的事后分析
data_quality_and_provenance_gaps: 102 项实现核对都 100%；VisionZip 的 53 个主导 patch 用 cls.json 的同一分数代替（逐题相同已核对）；事后分组多、未校正；没有在 LLM 之前留 47 个的对照，也没有去掉填充 patch 的 VisionZip 运行
primary_result_with_uncertainty: 所选 − VisionZip-64 +1.18 [+0.30, +2.06] / +1.38 [+0.19, +2.62]（判读“有希望”）；k1cls − VisionZip-64 −1.33 [−2.03, −0.67] / +0.13 [−0.99, +1.17]
process_dynamics: none（单次推理）
mechanism_and_intermediate_findings: k1cls 的 47 个平均 46.4 个在 VisionZip 的 53 个里（79.6% / 78.2% 的题全在）；VisionZip 的 53 个里平均约 4 个整块在填充里（约 90% 的题至少一个）
errors_heterogeneity_and_boundary_conditions: VQAv2 上 47 个全在 53 个里的题 −1.47 [−2.30, −0.74]；53 个里没有填充 patch 的题 −3.81 [−5.99, −1.62]（512 题），有的 −1.06；GQA 上 attr +3.20、rel −1.53、cat −3.15（探索性）
robustness_sensitivity_and_alternative_explanations: 没有填充 patch 的图是另一类图；位置编号不同（F6）；多重比较
new_P: [P15 候选：VisionZip-64 在 LLaVA-1.5 的方形填充图上，约 6% 的主导名额（53 个中约 4 个）落在整块填充上（F63）]
new_F: [F63, F64, F65（探索性）]
new_M: [M 候选（AI）：K = 1 时少 17 个 token（含 VisionZip 的上下文 token）的损失大于第 0 层全看的收益，VQAv2 上明显、GQA 上没有（F64）]
new_C_candidate: [填充感知的 VisionZip（P15 × F63）；K = 1 + 上下文 token 或预算对齐的对照（F62 × F64）]
evidence_index: code/output/EXP-0030/analysis（c6a48c42…）、squeeze（5769d507…）
open_questions: 去掉填充 patch 的 VisionZip 是否更好；第 0 层全看本身有没有作用；GQA 语义类型上的差别是否复现；拼接在新数据上的结果（EXP-0031 v1.1）
next_routes: [deep-read Mode C 回流 24（与 EXP-0031 的结果一起）]
next_skill: deep-read Mode C
```
