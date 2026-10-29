# SparseVLM：研究分析笔记

> 源码核验与本项目实测见补充笔记 [PAPER-0005_SparseVLM_源码核验](PAPER-0005_SparseVLM_源码核验.md)（证据 ID 带 C 前缀）。

分析日期：2026-10-23。上游材料已取得；本笔记由 analyze-paper 工作流整理，不代表 Claude CLI 或实验复现已运行。

## 来源与阅读范围

- 标题：SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference。
- 作者/机构：Yuan Zhang、Chun-Kai Fan、Junpeng Ma 等，北京大学/复旦/UC Berkeley 等。
- 版本：arXiv:2410.04417v4，2025-06-03；ICML 2025；不是 SparseVLM+。
- [原文](https://arxiv.org/pdf/2410.04417v4)；[固定源码](https://github.com/Gumpest/SparseVLMs/tree/734d9003879689713a132b83297dfad060a5c9af)（分支 `v1.5`）。
- PDF：`papers/SparseVLM-arxiv-2410.04417.pdf`，18 页；SHA-256：`ccb0ff92d74d782fdf6aead5580b9c50c224ade23e4f99ed3ab9541f19cb167b`。原文访问不等于获得再分发许可，全文未提交 Git。
- 代码许可：Apache-2.0（代码）。
- 实际阅读：研究动机、方法、主要表格、相关消融/附录和上文指定代码路径；未逐项复算所有数据或证明。提取工具为 pdftotext/PyMuPDF，双栏公式可能有解析误差，引用以原 PDF 页/节/表为准；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

能否利用当前问题中真正与视觉有关的文本，指导免训练视觉稀疏化并减少信息损失？

## 方法与贡献

先在 LLM 输入 embedding 空间计算图像与文本关联，选相关性超过均值的文本 token 作为 raters。取这些 query 到视觉 key 的注意力并平均，得到重要性。论文提出 N=λ(Lv−rank(P)) 决定删除数；再从待删池回收较高分 token，以密度峰聚类压成少量新 token。基础方法训练免费；附录讨论 FlashAttention 兼容，不能把导出注意力的成本预设为零。

## 研究设计

LLaVA-1.5、Mini-Gemini、Qwen2-VL 和 Video-LLaVA 的图像/视频基准。Table 1 为 LLaVA-1.5-7B 的八项基准，原始视觉 token 576，配置标为 192/128/64。比较 FastV/ToMe/PDrop，另消融文本 raters 和回收机制。原实验单 A100-80GB；延迟口径见附录 F，不含模型加载等外部时间。

## 结果及不确定性

Table 1：Retain192 归一化平均 99.1%，FLOPs 4.62T→2.14T，latency 57.82→36.50ms；Retain128 为 96.7%、1.72T、33.28ms；Retain64 为 89.3%。这些是任务分数相对各自 vanilla 的平均，不是原始准确率。Table 4 在 64 设置下，POPE 无/有回收为 72.8/77.5，GQA 为 52.2/53.8。紧邻正文部分百分比与表中数值不完全对应，引用以表及设置为准，不转述不清楚的增益口径。没有独立重复或统计区间。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-SV-1 | DE | raters、rank 策略和聚类回收公式 | PDF p.3–4，§3.2–3.3，Eq.2–10 | 论文算法；不是固定分支全部实现的证明 |
| E-SV-2 | DE | 99.1%/96.7%/89.3% 及 latency | PDF p.6，Table 1 | 八项基准归一化；硬件和预算受限 |
| E-SV-3 | DE | 回收消融 72.8→77.5、52.2→53.8 | PDF p.8，Table 4 | POPE/GQA 的作者结果 |
| E-SV-4 | DE | [2,6,15] 和固定保留列表 | v1.5 modelling_sparse_llama.py L69；score.py L5–14 | 与论文 rank 自适应表述需要区分 |

官方主分支已扩展到 SparseVLM+，本次固定原方法 v1.5。`llava/model/language_model/modelling_sparse_llama.py` L69 默认 `pruning_loc=[2,6,15]`（零基索引），L195–205 选择 raters，L263–310 筛选并回收；`score.py` L5–14 固定阶段数量 `[300,200,110]` / `[303,110,36]` / `[66,30,17]`。核验路径使用 top-k 计划，不是每次执行论文 rank 自适应公式。配置名 192/128/64 不等于每层或最后一层的实际 token 数，且回收会增加数量。当前代码/论文是否精确生成全部已发表表格尚未知。

## 限制与替代解释

[AI] 问题相关选择有利于当前问题，但其可复用性取决于未来问题和缓存策略；不能仅凭单轮分数判断多轮表现。rater 选择本身依赖跨模态 embedding，矩阵秩也受数值精度/阈值影响。聚类可能损失小对象细节且有运行开销。原文/代码版本与预算解释必须先统一。

## 与当前项目的关系

材料可得性中，独立复现未知。作为文本引导对照很有价值，但必须记录真实逐层长度、回收数和预填充/解码成本。3090 上仅建议先检查 7B 小批推理，A100 数字不能直接迁移。不要把主分支 SparseVLM+ 的结果或环境说明当作 v1.5 的结果。（2026-10-29 完整阅读补注：“独立复现未知”是 2026-10-23 写作时的状态。EXP-0002 已在 RTX 3090、batch 1 上复现 v1.5（SDPA 替代 FlashAttention-2）：POPE F1 85.29 / 85.07 / 77.67，与 v4 Table 1 的 85.3 / 85.0 / 77.5 相差不超过 0.2；64 档实测每层 576×3、82×4、32×9、18×16，评测显存 18.1 GiB，见源码核验补充 E-SV-C06、E-SV-C13。）

## 交接 Deep Read

- 候选 P：不区分问题语义的剪枝可能删掉任务需要的视觉信息。
- 候选 F：选相关文本与回收被删 token 在指定消融中改善质量；更低预算下仍会损失。
- 机制候选见「方法与贡献」；其解释与适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功/失败结果。
- 综合比较：[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)；视觉三篇进入 [Mode C](../视觉Token压缩_ModeC.md)，时序三篇保留为旁支参考。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-SparseVLM-20261023
analysis_id: AP-SparseVLM-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0005
- https://github.com/Gumpest/SparseVLMs/tree/734d9003879689713a132b83297dfad060a5c9af
evidence_status: partial
evidence_locations:
- 'E-SV-1: PDF p.3–4，§3.2–3.3，Eq.2–10'
- 'E-SV-2: PDF p.6，Table 1'
- 'E-SV-3: PDF p.8，Table 4'
- 'E-SV-4: v1.5 modelling_sparse_llama.py L69；score.py L5–14'
missing_evidence:
- 没有本项目独立运行、原始逐样本输出或3090实测（2026-10-29 完整阅读补注：已由 EXP-0002 部分补上，见源码核验补充）
- 未完成全部基准数据许可/划分/污染与所有引用文献审计
- 引用量和2026年完整后续工作未核验
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」及 sources-20261023.json
research_question: 能否利用当前问题中真正与视觉有关的文本，指导免训练视觉稀疏化并减少信息损失？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: 不区分问题语义的剪枝可能删掉任务需要的视觉信息。
candidate_F: 选相关文本与回收被删 token 在指定消融中改善质量；更低预算下仍会损失。
evidence_index:
- E-SV-1
- E-SV-2
- E-SV-3
- E-SV-4
unresolved_questions:
- 3090显存/延迟与独立复现待验证
- 方法机制是否跨任务成立、对照协议是否可完全对齐
```
