# VisionZip：研究分析笔记

分析日期：2026-10-23。上游材料已取得；本笔记由 analyze-paper 工作流整理，不代表 Claude CLI 或实验复现已运行。

## 来源与阅读范围

- 标题：VisionZip: Longer is Better but Not Necessary in Vision Language Models。
- 作者/机构：Senqiao Yang 等，CUHK / HKUST / HITSZ。
- 版本：CVPR 2025 正式 PDF + 官方 supplement；不采用 arXiv v2（2026-03-15）新增内容作会议版证据。
- [原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.pdf)；[固定源码](https://github.com/JIA-Lab-research/VisionZip/tree/8f86b55c6f000eb033e6912538af2dd7dcb30502)（分支 `main`）。
- PDF：`papers/VisionZip-cvpr.pdf`，11 页；SHA-256：`d4e22f2136ba1b7e95a971d99f008a53141b9dbe1b2dfae5bdfbf8c06778e13d`。原文访问不等于获得再分发许可，全文未提交 Git。
- 代码许可：Apache-2.0（代码；模型权重另按其许可）。
- 实际阅读：研究动机、方法、主要表格、相关消融/附录和上文指定代码路径；未逐项复算所有数据或证明。提取工具为 pdftotext/PyMuPDF，双栏公式可能有解析误差，引用以原 PDF 页/节/表为准；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

能否在进入语言模型前利用视觉编码器的信息聚合特点，构造更短、可复用的视觉表示？

## 方法与贡献

在选定视觉层（CLIP 通常倒数第二层）取特征和注意力；用 CLS 对 patch 的跨头注意力选 dominant tokens，无 CLS 时用平均接收注意力。对其余 token 的 key 特征归一化，均匀选目标、按相似度分配并合并成 contextual tokens；合并后进入 projector/LLM。免训练版为主，另有仅微调 projector 的 ‡ 版。64 配置为 54 dominant+10 contextual，不能写为纯剪掉低分特征。

## 研究设计

LLaVA-1.5、LLaVA-NeXT、Mini-Gemini 与 Video-LLaVA；LLaVA-1.5 主表含 11 个任务，576→192/128/64；NeXT 2880→640/320/160。免训练与微调行分开。效率评估 POPE、单 A800-80GB；少量 projector tuning 报告 8×A800，补充完整训练用 8×3090。multi-turn 证据主要为案例，不是大规模多轮统计。

## 结果及不确定性

Table 1 免训练归一化平均：192 为 98.5%，128 为 97.6%，64 为 94.0%；64 的 projector tuning 为 95.2%。Table 4，NeXT-7B/160/POPE：prefill 218→27.8ms，总时间 2293→756s。Table 5，TextVQA：SparseVLM 576→64 得 51.1；先屏蔽视觉前50高分 token 后 526→64 得46.4；先给 VisionZip 的128 token再选64得52.5。作者数据未独立复验，无完整置信区间。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-VZ-1 | DE | 64=54+10，选择与合并机制 | 主文 PDF p.3–4，§2.3；supp PDF p.2，Table 6 | 实际编码器/CLS 条件要匹配 |
| E-VZ-2 | DE | 94.0% 免训练 vs 95.2% projector tuning | 主文 PDF p.5，Table 1 | 归一化平均，不是原始 accuracy；对手是2024.10版 |
| E-VZ-3 | DE | 7.8倍prefill和3.0倍总时长改善 | 主文 PDF p.7，Table 4 | NeXT7B/160/POPE/A800 条件 |
| E-VZ-4 | DE | 预屏蔽与预选后 SparseVLM 的结果 | 主文 PDF p.7–8，Table 5、§4.2 | 已有组合实验，非普遍因果证明 |
| E-VZ-5 | SC | text-agnostic 更适合多轮缓存复用 | 主文 PDF p.8，§4.3、Fig.7 | 案例证据，尚缺大规模多轮对照 |

固定 commit 的 `visionzip/clip_encoder.py` L41–83 先执行视觉塔，再取 `attentions[-2]`/`hidden_states[-2]` 并选择合并，实际主要节省后续 LLM 开销。`main.py` L5–7 有 dominant-1 的 CLS 处理；合并为 target hidden + 被分配 hidden 的平均，不能笼统称所有 token 的统一均值。当前 main 包含后续模型适配，未证明全部与 CVPR 发布代码完全一致；本次静态检查 LLaVA/CLIP 路径，未运行。

## 限制与替代解释

比较版本：Table 1 对手明确为 SparseVLM(2024.10)，不能判定胜过 ICML2025 最终版。信息聚合/空间错位解释有局部干预支持，但遮挡也会改变输入分布，尚未证明通用因果机制。§4.1 把 softmax 梯度解释成随 z 一直指数增加不严谨：p(1−p)≤1/4，p→1 时反而趋零；不能用该段当严格机制证明。低注意力细节能否被 contextual merging 保住未知，text-agnostic 也不等于全任务无损。

## 与当前项目的关系

材料可得性中，独立复现未知。当前用户将其指定为开发 baseline，7B 免训练版本是单张3090值得验证的入口；仍需实测 batch1 显存和端到端延迟。Qwen2.5-VL 等后续适配不等于原论文已评估；“VisionZip后再SparseVLM”已有 Table5，不能当组合创新。适合围绕细节丢失、上下文合并和跨问题复用继续诊断。

## 交接 Deep Read

- 候选 P：LLM 内压缩仍支付早期全 token 成本，且文本挑选的位置可能不是信息真正承载的位置。
- 候选 F：进入LLM前选dominant并合并context可维持较高平均表现；视觉预选后再文本筛选在一个TextVQA设置改善结果。
- 机制候选见「方法与贡献」；其解释与适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功/失败结果。
- 综合比较：[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)；视觉三篇进入 [Mode C](../视觉Token压缩_ModeC.md)，时序三篇保留为旁支参考。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-VisionZip-20261023
analysis_id: AP-VisionZip-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0006
- https://github.com/JIA-Lab-research/VisionZip/tree/8f86b55c6f000eb033e6912538af2dd7dcb30502
evidence_status: partial
evidence_locations:
- 'E-VZ-1: 主文 PDF p.3–4，§2.3；supp PDF p.2，Table 6'
- 'E-VZ-2: 主文 PDF p.5，Table 1'
- 'E-VZ-3: 主文 PDF p.7，Table 4'
- 'E-VZ-4: 主文 PDF p.7–8，Table 5、§4.2'
- 'E-VZ-5: 主文 PDF p.8，§4.3、Fig.7'
missing_evidence:
- 没有本项目独立运行、原始逐样本输出或3090实测
- 未完成全部基准数据许可/划分/污染与所有引用文献审计
- 引用量和2026年完整后续工作未核验
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」及 sources-20261023.json
research_question: 能否在进入语言模型前利用视觉编码器的信息聚合特点，构造更短、可复用的视觉表示？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: LLM 内压缩仍支付早期全 token 成本，且文本挑选的位置可能不是信息真正承载的位置。
candidate_F: 进入LLM前选dominant并合并context可维持较高平均表现；视觉预选后再文本筛选在一个TextVQA设置改善结果。
evidence_index:
- E-VZ-1
- E-VZ-2
- E-VZ-3
- E-VZ-4
- E-VZ-5
unresolved_questions:
- 3090显存/延迟与独立复现待验证
- 方法机制是否跨任务成立、对照协议是否可完全对齐
```
