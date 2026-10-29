# DUET：研究分析笔记

> 源码核验与本项目实测见补充笔记 [PAPER-0001_DUET_源码核验](PAPER-0001_DUET_源码核验.md)（证据 ID 带 C 前缀）。

分析日期：2026-10-23。上游材料已取得；本笔记由 analyze-paper 工作流整理，不代表 Claude CLI 或实验复现已运行。

## 来源与阅读范围

- 标题：DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting。
- 作者/机构：Xiangfei Qiu 等，华东师范大学、Aalborg University。
- 版本：arXiv:2412.10859v3，2025-01-10；KDD 2025。
- [原文](https://arxiv.org/pdf/2412.10859v3)；[固定源码](https://github.com/decisionintelligence/DUET/tree/dcc6e6780a9138731b64b9b5398a94a1d97033f0)（分支 `main`）。
- PDF：`papers/DUET-arxiv-2412.10859.pdf`，14 页；SHA-256：`a26e3deb2ec3821c56f39e12ce78c1097e5482a0620d50f2f7fa4c0d9bd76a19`。原文访问不等于获得再分发许可，全文未提交 Git。
- 代码许可：MIT（代码；不自动覆盖数据和第三方依赖）。
- 实际阅读：研究动机、方法、主要表格、相关消融/附录和上文指定代码路径；未逐项复算所有数据或证明。提取工具为 pdftotext/PyMuPDF，双栏公式可能有解析误差，引用以原 PDF 页/节/表为准；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

时序分布异质性和复杂通道关系能否通过两个互补聚类模块改善预测？

## 方法与贡献

输入归一化后走两路：TCM 用带噪声的分布路由器选择 top-k 线性提取器，每个提取器分解趋势/季节项；CCM 对 rFFT 幅值学习 Mahalanobis 距离，生成概率关系和可训练的掩码；FM 用掩码通道注意力融合，线性输出未来值。软聚类指允许重叠/部分联系，不是预先把变量硬分成互斥组。

## 研究设计

计算模型基准研究。作者称覆盖 25 个真实数据集、10 个领域，正文主表展示 10 个数据集；其余结果在代码材料。正文比较多个 look-back 并取各模型最好结果；一般预测长度为 96/192/336/720，ILI 应以 Table 3 的 12/24/36/48 为准（§5.1 的另一处列表不一致）。MSE/MAE 越低越好，训练用 L1/Adam；原实验 GPU 为 A800。未逐项审计所有数据切分及全部外置结果。

## 结果及不确定性

Table 2 的 Traffic，跨四个预测长度平均 MSE：完整 DUET 0.393、去掉 CCM 0.439、去掉 TCM 0.398。这个消融支持通道关系对该设置有用，不等于 CCM 对所有数据集都贡献最大。主表结果不是固定 look-back=96 协议，不与 TQNet 表直接排序。这里所引表格未提供逐次运行分布或置信区间；没有本项目复现。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-DUET-1 | SC | 双聚类分别应对时间分布与通道关系 | PDF p.4–6，§4.2–4.4 | 建模动机，不是因果证明 |
| E-DUET-2 | DE | Traffic 完整/无 CCM 的 MSE 为 0.393/0.439 | PDF p.7，Table 2 | 作者表格；未独立复验 |
| E-DUET-3 | DE | 主结果采用多个历史窗口的最好结果 | PDF p.6，§5.1；p.8，Table 3 | 限制跨论文直接比较 |

已在前次筛选检查固定 commit 的 `ts_benchmark/baselines/duet/models/duet_model.py` 和 `scripts/multivariate_forecast/DUET_unified_seq_len_96.sh`；统一窗口脚本存在不代表原文主表全部使用该协议。本次重点核对 §4、§5 和消融，尚未运行数据加载或模型。

## 限制与替代解释

[AI] 多个路由/掩码模块增加归因难度；rFFT 幅值忽略相位，频谱相近是否等于预测相关需验证。路由器的潜在分布解释是建模假设，注意力图不提供因果真值。多窗口择优与统一窗口的资源预算不同。原始数据许可及可能的应用偏差需按数据集另查。

## 与当前项目的关系

复现材料可得性评为中（静态审计），实证可复现性未知。ETTh1 小配置适合单卡起步，但 3090 显存和耗时未测。当前项目转向视觉 token 压缩后，本篇只作信息筛选/聚类的跨领域参考，不把时间频率轴机械替换为空间 token 序号。（2026-10-29 完整阅读补注：ETTh1-96 已复现：CPU 两个种子 0.3898 / 0.3790（EXP-0003），GPU 三个种子 0.3795 ± 0.0020（EXP-0005），报告值 0.377；时序方向自 DEC-0020 暂停。）

## 交接 Deep Read

- 候选 P：异质时间模式和无关通道干扰预测。
- 候选 F：Traffic 去掉 CCM 后 MSE 明显增加，而移除 TCM 的差异较小；贡献随数据结构变化。
- 机制候选见「方法与贡献」；其解释与适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功/失败结果。
- 综合比较：[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)；视觉三篇进入 [Mode C](../视觉Token压缩_ModeC.md)，时序三篇保留为旁支参考。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-DUET-20261023
analysis_id: AP-DUET-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0001
- https://github.com/decisionintelligence/DUET/tree/dcc6e6780a9138731b64b9b5398a94a1d97033f0
evidence_status: partial
evidence_locations:
- 'E-DUET-1: PDF p.4–6，§4.2–4.4'
- 'E-DUET-2: PDF p.7，Table 2'
- 'E-DUET-3: PDF p.6，§5.1；p.8，Table 3'
missing_evidence:
- 没有本项目独立运行、原始逐样本输出或3090实测（2026-10-29 完整阅读补注：ETTh1 已由 EXP-0003、EXP-0005 部分补上，见源码核验补充）
- 未完成全部基准数据许可/划分/污染与所有引用文献审计
- 引用量和2026年完整后续工作未核验
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」及 sources-20261023.json
research_question: 时序分布异质性和复杂通道关系能否通过两个互补聚类模块改善预测？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: 异质时间模式和无关通道干扰预测。
candidate_F: Traffic 去掉 CCM 后 MSE 明显增加，而移除 TCM 的差异较小；贡献随数据结构变化。
evidence_index:
- E-DUET-1
- E-DUET-2
- E-DUET-3
unresolved_questions:
- 3090显存/延迟与独立复现待验证
- 方法机制是否跨任务成立、对照协议是否可完全对齐
```
