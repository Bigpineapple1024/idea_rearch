# FastV：研究分析笔记

> 源码核验与本项目实测见补充笔记 [PAPER-0004_FastV_源码核验](PAPER-0004_FastV_源码核验.md)（证据 ID 带 C 前缀）。

分析日期：2026-10-23。上游材料已取得；本笔记由 analyze-paper 工作流整理，不代表 Claude CLI 或实验复现已运行。

## 来源与阅读范围

- 标题：An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models。
- 作者/机构：Liang Chen 等，北京大学、阿里巴巴。
- 版本：arXiv:2403.06764v3，2024-09-02；ECCV 2024。
- [原文](https://arxiv.org/pdf/2403.06764v3)；[固定源码](https://github.com/pkunlp-icler/FastV/tree/d1659729b5bf1be225e99ee15783deeea80f63b1)（分支 `main`）。
- PDF：`papers/FastV-arxiv-2403.06764.pdf`，18 页；SHA-256：`b4532650958db4f5d2955d76987386bb43318745e4a35d168ff9ce46dda6895b`。原文访问不等于获得再分发许可，全文未提交 Git。
- 代码许可：根目录未发现统一 LICENSE；子目录含上游许可证，不能据此推断全部新增代码许可。
- 实际阅读：研究动机、方法、主要表格、相关消融/附录和上文指定代码路径；未逐项复算所有数据或证明。提取工具为 pdftotext/PyMuPDF，双栏公式可能有解析误差，引用以原 PDF 页/节/表为准；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

完整视觉 token 是否需要在每个语言解码器层都继续参与计算？

## 方法与贡献

图像经视觉编码器/projector 后先完整进入 LLM；完成前 K 层后，按注意力重要性保留部分视觉 token，后续层使用更短序列。原文 §4.1 表述为平均接收注意力，核验官方实现则跨 head 平均后取最后一个 query 对视觉位置的分数。K=0 的原文特殊设置使用随机选择。基础方法无额外训练参数。

## 研究设计

LLaVA-1.5-7B/13B、QwenVL-Chat、Video-LLaVA 等；覆盖图像描述、VQA、推理、OCR 和视频 QA。主表四项为 Nocaps/Flickr30k 的 CIDEr 与 A-OKVQA/MMMU 的 accuracy，使用 greedy search。改变 K 和剪除比例 R，另有随机剪枝、文本剪枝、低 token 训练等消融。没有为此阅读下载全部评估样本或独立统计复验。

## 结果及不确定性

Table 1：LLaVA-1.5-13B，K=2、R=50%，Avg 73.6→73.6，理论 FLOPs 154.6B→84.6B（约减 45.3%）；7B 的 Avg 69.8→69.7。Table 4 的实测延迟是在单 A40、A-OKVQA、K=0/R=50% 下，7B 从 0.344s→0.230s，不能与 K=2 的质量表混成同一配置。未报告这些差异的置信区间；混合原始量纲的 Avg 不宜称统一准确率。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-FV-1 | DE | K=2/R=50% 质量与理论 FLOPs | PDF p.9，Table 1；p.8，Eq.5 | 只支持表中模型/任务与计算口径 |
| E-FV-2 | DE | A40 实测延迟采用 K=0 | PDF p.10，Table 4 | 不可和 K=2 数字混用 |
| E-FV-3 | DE | 跨头平均后 last-query 排名 | 固定 commit modeling_llama.py L738–746；fastv_kvcache.py L119–142 | 与论文平均所有 token 的文字表述不同 |
| E-FV-4 | SC | 深层视觉注意力低效及 anchor 聚合解释 | PDF p.6–8，§3.3–4.1 | 观察与解释分开；不能推出所有任务都可安全删除 |

固定 commit 的 `src/transformers/src/transformers/models/llama/modeling_llama.py` L721–790 有 inplace 与 mask 分支，L738–746 有 last-query 排名；`src/FastV/lmms-eval/fastv_kvcache.py` L119–142 也有最后 query 选择，但其 K=3、视觉区间 35:611 等硬编码必须核对。idx==K 进入下一层前删，对应已完成前 K 个零基循环层。旧 README 所述 mask demo 不可当真实序列压缩。静态检查不等于这些入口已经能在当前依赖运行。

## 限制与替代解释

[AI] 最后 query 和提示格式可能影响保留集合；attention 低不意味着信息绝对无用。剪得越早，越节省但信息丢失越难补偿；高压缩与已由 Q-Former 压缩的模型更需要重新选 K/R。早期层仍处理全 token。Fig.3 的 attention sink 现象是解释线索，不能据注意力直接证明因果。

## 与当前项目的关系

材料可得性中，独立复现未知。适合做简单对照，先统一 LLaVA-1.5-7B、图像范围、prompt 和 kernel，核查实际序列是否变短。当前任务不修改上游。相比 VisionZip，它更侵入 LLM/缓存路径；不能把 last-query 实现称严格文本无关，也不声称 FastV 是整个视觉压缩领域首创。

## 交接 Deep Read

- 候选 P：LLM 深层仍计算大量可能冗余的视觉 token。
- 候选 F：原文配置下，浅层后剪掉一半视觉 token 可基本维持主表表现，但极端压缩明显下降。
- 机制候选见「方法与贡献」；其解释与适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功/失败结果。
- 综合比较：[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)；视觉三篇进入 [Mode C](../视觉Token压缩_ModeC.md)，时序三篇保留为旁支参考。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-FastV-20261023
analysis_id: AP-FastV-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0004
- https://github.com/pkunlp-icler/FastV/tree/d1659729b5bf1be225e99ee15783deeea80f63b1
evidence_status: partial
evidence_locations:
- 'E-FV-1: PDF p.9，Table 1；p.8，Eq.5'
- 'E-FV-2: PDF p.10，Table 4'
- 'E-FV-3: 固定 commit modeling_llama.py L738–746；fastv_kvcache.py L119–142'
- 'E-FV-4: PDF p.6–8，§3.3–4.1'
missing_evidence:
- 没有本项目独立运行、原始逐样本输出或3090实测
- 未完成全部基准数据许可/划分/污染与所有引用文献审计
- 引用量和2026年完整后续工作未核验
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」及 sources-20261023.json
research_question: 完整视觉 token 是否需要在每个语言解码器层都继续参与计算？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: LLM 深层仍计算大量可能冗余的视觉 token。
candidate_F: 原文配置下，浅层后剪掉一半视觉 token 可基本维持主表表现，但极端压缩明显下降。
evidence_index:
- E-FV-1
- E-FV-2
- E-FV-3
- E-FV-4
unresolved_questions:
- 3090显存/延迟与独立复现待验证
- 方法机制是否跨任务成立、对照协议是否可完全对齐
```
