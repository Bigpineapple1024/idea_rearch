# WPMixer：研究分析笔记

> 源码核验与本项目实测见补充笔记 [PAPER-0003_WPMixer_源码核验](PAPER-0003_WPMixer_源码核验.md)（证据 ID 带 C 前缀）。

分析日期：2026-10-23。上游材料已取得；本笔记由 analyze-paper 工作流整理，不代表 Claude CLI 或实验复现已运行。

## 来源与阅读范围

- 标题：WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting。
- 作者/机构：Md Mahmuddun Nabi Murad、Mehmet Aktukmak、Yasin Yilmaz，University of South Florida / Intel。
- 版本：arXiv:2412.17176v1，2024-12-22；AAAI 2025。
- [原文](https://arxiv.org/pdf/2412.17176v1)；[固定源码](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/tree/74104c9dddd54d279eb8323f48934b4fd75fcae7)（分支 `main`）。
- PDF：`papers/WPMixer-arxiv-2412.17176.pdf`，12 页；SHA-256：`f95ac458aef1fbfc5b1df187ac99605889efd9dd30025a7b4bd0f5d6f6da26e9`。原文访问不等于获得再分发许可，全文未提交 Git。
- 代码许可：MIT（代码）。
- 实际阅读：研究动机、方法、主要表格、相关消融/附录和上文指定代码路径；未逐项复算所有数据或证明。提取工具为 pdftotext/PyMuPDF，双栏公式可能有解析误差，引用以原 PDF 页/节/表为准；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

多分辨率小波表示与 patch mixing 能否更好保留时间局部变化和不同频段的预测信息？

## 方法与贡献

实例归一化→多层小波分解为近似/细节系数→各分辨率独立归一化、切重叠 patch、embedding→patch mixing 与 embedding mixing→预测各系数序列→逆小波重建→反归一化。这里的 mixing 主要在 patch 和 embedding 维度，不应误写成显式通道注意力模型。训练用 SmoothL1，测量用 MSE/MAE。

## 研究设计

7 个常用数据集，预测长度 96/192/336/720。两套协议：优化 look-back 及超参数；统一 TimeMixer 的窗口/batch/epochs 后仍优化其他超参数。使用 Optuna/TPE。统一设置报告三种随机种子的均值与标准差。ETT/Weather 使用单张 RTX 4090；Electricity/Traffic 使用两张 A100。

## 结果及不确定性

Table 4 统一设置：ETTh1 MSE 0.422±0.001，TimeMixer 0.447±0.002；Weather 为 0.243±0.001 对 0.240±0.010，Traffic 为 0.489±0.005 对 0.484±0.015。部分数据集更好，但不是所有 MSE 都胜出，均值小差异也不能直接视为统计显著。Table 5 在 look-back=512 下提供 14 种模块组合；每种组合又优化超参数，因此估计的是各变体经调优后的表现。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-WP-1 | DE | 分解—各尺度 mixing—重建的数据流 | PDF p.3–5，Methodology，Eq.1–10 | 方法定义 |
| E-WP-2 | DE | 三种种子、统一设置结果 | PDF p.7，Table 4；补充统一设置说明 | 作者均值/标准差，不是显著性结论 |
| E-WP-3 | DE | 14 种组合、各自调参的消融 | PDF p.7，Table 5 及紧邻说明 | 不是冻结全部其他因素的纯单变量比较 |

前次已查看 `models/wavelet_patch_mixer.py`、`models/model.py`、`run_LTF.py` 和 `scripts/Unified/ETTh1_Unified_setup.sh`。固定 commit 的环境声明与正文硬件报告分别保存，不能宣称已安装或运行。分辨率分支/逆变换结构与方法一致；默认大规模搜索不纳入单卡复现承诺。

## 限制与替代解释

[AI] 小波类型、分解级数、边界 padding、patch 尺寸和搜索预算都会影响结果；用不同调参预算比较容易误归因。SmoothL1 自身可能贡献鲁棒性。原文补充中的“PyTorch version 3.10.12”疑似把 Python 版本写成框架版本，环境应依据源码依赖实际核验。数据许可、所有窗口选择与真实部署成本未审计。

## 与当前项目的关系

材料可得性中，独立复现未知。适合研究频段/细节保真，但当前视觉主线只借鉴“把不同尺度信息分开分析”的原则，不直接搬入小波模块。时序序列的有序采样与视觉编码器聚合后的 token 语义不同。

## 交接 Deep Read

- 候选 P：单一尺度表示难以兼顾趋势与局部高频变化。
- 候选 F：统一设置下部分数据集误差下降，但 Weather/Traffic MSE 未胜过 TimeMixer；效果依赖数据与设置。
- 机制候选见「方法与贡献」；其解释与适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功/失败结果。
- 综合比较：[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)；视觉三篇进入 [Mode C](../视觉Token压缩_ModeC.md)，时序三篇保留为旁支参考。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-WPMixer-20261023
analysis_id: AP-WPMixer-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0003
- https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer/tree/74104c9dddd54d279eb8323f48934b4fd75fcae7
evidence_status: partial
evidence_locations:
- 'E-WP-1: PDF p.3–5，Methodology，Eq.1–10'
- 'E-WP-2: PDF p.7，Table 4；补充统一设置说明'
- 'E-WP-3: PDF p.7，Table 5 及紧邻说明'
missing_evidence:
- 没有本项目独立运行、原始逐样本输出或3090实测
- 未完成全部基准数据许可/划分/污染与所有引用文献审计
- 引用量和2026年完整后续工作未核验
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」及 sources-20261023.json
research_question: 多分辨率小波表示与 patch mixing 能否更好保留时间局部变化和不同频段的预测信息？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: 单一尺度表示难以兼顾趋势与局部高频变化。
candidate_F: 统一设置下部分数据集误差下降，但 Weather/Traffic MSE 未胜过 TimeMixer；效果依赖数据与设置。
evidence_index:
- E-WP-1
- E-WP-2
- E-WP-3
unresolved_questions:
- 3090显存/延迟与独立复现待验证
- 方法机制是否跨任务成立、对照协议是否可完全对齐
```
