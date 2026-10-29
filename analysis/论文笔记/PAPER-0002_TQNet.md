# TQNet：研究分析笔记

> 源码核验与本项目实测见补充笔记 [PAPER-0002_TQNet_源码核验](PAPER-0002_TQNet_源码核验.md)（证据 ID 带 C 前缀）。

分析日期：2026-10-23。上游材料已取得；本笔记由 analyze-paper 工作流整理，不代表 Claude CLI 或实验复现已运行。

## 来源与阅读范围

- 标题：Temporal Query Network for Efficient Multivariate Time Series Forecasting。
- 作者/机构：Shengsheng Lin 等，华南理工大学、鹏城实验室。
- 版本：arXiv:2505.12917v2，2025-09-11；ICML 2025，阅读版本晚于会议。
- [原文](https://arxiv.org/pdf/2505.12917v2)；[固定源码](https://github.com/ACAT-SCUT/TQNet/tree/15e19cb23483ed52398566c4baa959168cfffa57)（分支 `master`）。
- PDF：`papers/TQNet-arxiv-2505.12917.pdf`，18 页；SHA-256：`3a406d4a89215be1ab70f6bf1c9f38c9cdde04cb021b8c4a4ef3ea6a896af2f0`。原文访问不等于获得再分发许可，全文未提交 Git。
- 代码许可：Apache-2.0（代码）。
- 实际阅读：研究动机、方法、主要表格、相关消融/附录和上文指定代码路径；未逐项复算所有数据或证明。提取工具为 pdftotext/PyMuPDF，双栏公式可能有解析误差，引用以原 PDF 页/节/表为准；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

能否用跨样本学习的周期查询减弱局部噪声对通道相关性估计的影响？

## 方法与贡献

学习 C×W 的周期查询参数，根据当前时刻 t mod W 循环取长度 L 的片段作为 Q；K/V 来自当前样本。单层 TQ-MHA 捕捉通道关系，再以浅 MLP 建模时间依赖，配残差、线性输出及可选实例归一化。Q 为全局学习先验，K/V 保留当前样本信息，注意力并不是固定的全局相关矩阵。

## 研究设计

计算模型基准研究。12 个数据集，固定历史长度 96；PEMS 预测 12/24/48/96，其余 96/192/336/720，评价 MSE/MAE。附录 A.2 报告 ETT/PEMS 为 6:2:2，其余 7:1:2；实际复现须以数据加载代码和固定长度切分为准。单 RTX 4090 24GB、Adam、L2、30 epochs、验证集早停。主设置 seed=2024；附录 Table 9 另列种子与学习率敏感性。

## 结果及不确定性

Table 2 把各数据集四个预测长度平均后，24 项 MSE/MAE 中有 22 项进入前二。Electricity MSE 0.164，对照 iTransformer 0.178；ETTh2 MSE 0.378，TimeXer 0.368，因此不能称为全任务第一。Table 3 比较 Q/K 都来自样本、TQ-query+样本-key、Q/K 都来自先验，支持混合全局与局部信息的设计（Traffic 的 MSE 除外，见下文候选 F 的 2026-10-29 补注）。主表并非所有格子都有不确定性估计；Table 9 不等于整张主表完成统计显著性检验。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-TQ-1 | DE | 周期切片 Q 与样本 K/V 的定义 | PDF p.3，§3.2，Eq.1–3 | 算法定义；机制解释仍须实验 |
| E-TQ-2 | DE | 固定窗口主结果与 Q/K 消融 | PDF p.6，Table 2–3 | 作者结果，不是本项目测量 |
| E-TQ-3 | SC | 周期、弱相关和长历史窗口的限制 | PDF p.9，§5 | 作者明确列出的边界 |
| E-TQ-4 | DE | 4090 24GB、seed=2024 和独立敏感性表 | PDF p.13，A.2；p.18，Table 9 | 不能外推为所有结果均多种子 |

前次筛选已检查固定 commit 的 `models/TQNet.py` 与 `scripts/TQNet/etth1.sh`，包含 `use_tq`、`channel_aggre` 开关。论文正文定义的周期 Q 与实现思路一致；脚本的默认 batch、seed、周期不应被误认为跨数据集最优值。未执行训练或逐项重现附录。

## 限制与替代解释

作者 §5 明确承认对周期 W 的依赖、弱通道相关时的副作用，以及较长历史窗口下的收益下降。[AI] 周期漂移和跨域时间对齐可能削弱先验；注意力仍在通道维度发生，不能把结构简单误认为任意高维都线性扩展。仅有 benchmark 优势不足以支持真实部署普遍有效。

## 与当前项目的关系

复现材料可得性中，独立复现未知。4090 24GB 的原文报告使单张 3090 试跑更有依据，但不保证同吞吐。适合从简单骨架研究先验与样本信息的取舍；迁移到视觉前必须找到对应的稳定结构，不能凭空假设图像具有同样周期。（2026-10-29 完整阅读补注：ETTh1 已复现：CPU 种子 2024 与 result.txt 的 MSE 差 ≤ 0.0025（EXP-0003）；GPU 三个种子的均值为 0.3728 / 0.4295 / 0.4796 / 0.5019（EXP-0005）；时序方向自 DEC-0020 暂停。）

## 交接 Deep Read

- 候选 P：单样本通道相关性受噪声扰动。
- 候选 F：混合先验查询和样本键优于论文测试中的两个极端设置；周期/相关性弱时存在边界。（2026-10-29 完整阅读补注：Table 3（PDF p.6）共 7 个数据集，Traffic 是例外：MSE 上 Q=Raw,K=Raw 为 0.426、Q=TQ,K=TQ 为 0.429，都低于默认 Q=TQ,K=Raw 的 0.445，只有 MAE 上默认最低（0.276）。“优于两个极端”在其余 6 个数据集的 MSE 和全部 7 个数据集的 MAE 上成立；p.5 正文说默认设置表现最好，没有提这一例外。）
- 机制候选见「方法与贡献」；其解释与适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功/失败结果。
- 综合比较：[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)；视觉三篇进入 [Mode C](../视觉Token压缩_ModeC.md)，时序三篇保留为旁支参考。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-TQNet-20261023
analysis_id: AP-TQNet-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0002
- https://github.com/ACAT-SCUT/TQNet/tree/15e19cb23483ed52398566c4baa959168cfffa57
evidence_status: partial
evidence_locations:
- 'E-TQ-1: PDF p.3，§3.2，Eq.1–3'
- 'E-TQ-2: PDF p.6，Table 2–3'
- 'E-TQ-3: PDF p.9，§5'
- 'E-TQ-4: PDF p.13，A.2；p.18，Table 9'
missing_evidence:
- 没有本项目独立运行、原始逐样本输出或3090实测（2026-10-29 完整阅读补注：ETTh1 已由 EXP-0003、EXP-0005 部分补上，见源码核验补充）
- 未完成全部基准数据许可/划分/污染与所有引用文献审计
- 引用量和2026年完整后续工作未核验
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」及 sources-20261023.json
research_question: 能否用跨样本学习的周期查询减弱局部噪声对通道相关性估计的影响？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: 单样本通道相关性受噪声扰动。
candidate_F: 混合先验查询和样本键优于论文测试中的两个极端设置；周期/相关性弱时存在边界。（Traffic 的 MSE 是例外，见「交接 Deep Read」候选 F 的 2026-10-29 补注）
evidence_index:
- E-TQ-1
- E-TQ-2
- E-TQ-3
- E-TQ-4
unresolved_questions:
- 3090显存/延迟与独立复现待验证
- 方法机制是否跨任务成立、对照协议是否可完全对齐
```
