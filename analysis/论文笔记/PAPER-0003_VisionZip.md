# ANALYSIS_PACKAGE：VisionZip: Longer is Better but Not Necessary in Vision Language Models

状态：partial。paper_id PAPER-0003 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 dvlab-research/VisionZip@8f86b55c6f000eb033e6912538af2dd7dcb30502、README、论文与补充材料检索摘要。
**论文全文未读**（arxiv.org、openaccess.thecvf.com 被网络策略拦截），SC 条目需回查原文。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0003
analysis_id: PAPER-0003
created_at: 2026-10-23
source_ids: [arXiv:2412.04467, github:dvlab-research/VisionZip@8f86b55c6f000eb033e6912538af2dd7dcb30502, EXP-0001]
evidence_status: partial
evidence_locations: [README.md, visionzip/clip_encoder.py, visionzip/utils.py, visionzip/llava_arch.py, setup.py, EXP-0001]
missing_evidence:
- 论文全文与主表未读取；64/128/192 的 (dominant, contextual) 设置来自补充材料检索摘要
- 未在真实权重上运行
next_skill: deep-read Mode C
source_identity_and_version: "Yang et al., CVPR 2025, arXiv 2412.04467；代码 Apache-2.0，8f86b55c (2025-07-21)"
research_question: 视觉 token 是否大量冗余，能否在进入 LLM 之前用文本无关的方式压缩到很少数量
study_type_and_context: 免训练推理（也提供高效微调和训练模式）；LLaVA-1.5、LLaVA-NeXT、Mini-Gemini、Qwen2.5-VL
design_units_samples_or_materials: LLaVA 系列评测集；视频任务
intervention_input_or_procedure: 在 CLIP 倒数第二层按 CLS 注意力选 dominant token，剩余 token 按 key 相似度合并成 contextual token
measurements_and_analysis: 准确率保留比例、prefill 时间、总推理时间
results_with_uncertainty: 见 E-VZ-01～02；未见方差
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 见“研究判断”
reproducibility_and_transferability: 代码短小，能插入官方 LLaVA；但 PyPI 包为空、写死 24 层 CLIP、需传入论文预算（默认值不是论文设置）
candidate_P: [P1 注意力作为重要性的可靠性, P4 被丢弃信息的处理, P5 压缩 token 与 LLM 训练分布的匹配]
candidate_F: [F3 文本无关优于文本引导（作者主张）, F6 位置被压缩, F7 合并 token 幅值翻倍与 CLS 注入]
evidence_index: E-VZ-01..E-VZ-12
unresolved_questions: [论文是否讨论 CLS 注入与合并 token 的幅值, 位置压缩的影响]
```

## 来源

- 论文：Senqiao Yang 等，CVPR 2025，arXiv 2412.04467；补充材料 Table 6 给出 LLaVA-1.5 的设置：64 → (54, 10)，128 → (108, 20)，192 → (162, 30)（检索摘要）。
- 代码：<https://github.com/dvlab-research/VisionZip>，Apache-2.0，固定 commit 8f86b55c（2025-07-21）。
- 实际阅读范围：README；`visionzip/` 下全部源码（main.py、clip_encoder.py、utils.py、llava_arch.py）；setup.py；PyPI 上的 visionzip-0.1.3 wheel。

## 研究事实

- **核心问题**：视觉 token 的注意力集中在少数 token 上，大量 token 冗余，能否在送入 LLM 前直接压缩。
- **方法（代码实际做法）**：
  1. **dominant**：取 CLIP 倒数第二层（第 22 层）CLS 对各 patch 的注意力，对 head 求和，保留前 dominant−1 个 patch，**连同 CLS 本身**一起保留（E-VZ-06、E-VZ-08）。
  2. **contextual**：其余 token 用同层 key（对 head 取平均）的余弦相似度，以等间隔选出的 token 为中心，其他 token 分配给最相似的中心；contextual token = 中心 token + 被分配 token 的**平均**（E-VZ-07）。
  3. dominant 按原始空间顺序排列，之后接 contextual，送入投影层和 LLM；LLM 的所有层都只看到这些 token（E-VZ-12）。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-VZ-01 | SC | 只保留 10% 视觉 token、免训练时约保留 95% 性能 | README Highlights | LLaVA 系列 | 平均口径未回查 |
| E-VZ-02 | SC | LLaVA-1.5、576→64 时保留 94.0%，FastV 75.6%，SparseVLM 85.8% | 论文主表（检索摘要） | 作者自己的对比 | 对比方法的预算口径需回查 |
| E-VZ-03 | SC | 注意力集中于少数 token，大量 token 注意力很低 | 论文（检索摘要） | — | 低注意力 ≠ 无信息 |
| E-VZ-04 | SC | 文本引导的方法常选到非主导 token，信息更少，性能更低；在 LLM 外剪枝更高效 | 补充材料（检索摘要） | — | 与 SparseVLM 的主张相反，需在相同预算下比较 |
| E-VZ-05 | SC | 文本无关，适合多轮对话，能与其他 LLM 加速方法叠加；显著降低 prefill 时间 | README | — | 未实测 |
| E-VZ-06 | DE | dominant 用 CLIP 第 22 层（0 起计）CLS→patch 注意力对 head 求和后 top-k；contextual 用同层 key 的 head 平均 | clip_encoder.py；utils.py apply_info；EXP-0001 probe_visionzip.py | — | — |
| E-VZ-07 | DE | contextual token = 中心 + 被分配 token 的平均；当所有 token 相同时，合并后的值是原来的 2 倍 | clip_encoder.py；EXP-0001 probe_visionzip.py 第 2 部分 | — | 是否有意为之、对 LLM 的影响未知 |
| E-VZ-08 | DE | 第 0 个视觉 token 是 CLIP 倒数第二层的 CLS 隐状态；而 LLaVA-1.5 训练时用 `select_feature='patch'` 去掉了 CLS | EXP-0001 probe_visionzip.py 第 1 部分；LLaVA clip_encoder.py L37-38 | LLaVA-1.5 路径 | 影响大小未知 |
| E-VZ-09 | DE | 压缩后的 token 在 LLM 中按连续位置排列，LLM 看不到原图中的空间间隔 | llava_arch.py prepare_inputs_labels_for_multimodal_visionzip（position_ids 为 None，由 generate 按长度生成） | LLaVA-1.5 | 与 FastV、SparseVLM+ 保留原始位置的做法相反 |
| E-VZ-10 | DE | 写死 24 层 CLIP 的调度（r 列表长度 24、第 22 层取 metric），anyres 路径写死 24×24 网格；换视觉编码器需要改代码 | utils.py apply_info；llava_arch.py restore_image_features_sorted | — | — |
| E-VZ-11 | DE | PyPI 上 visionzip 0.1.3 的 wheel 只有元数据、没有代码（setup.py 写成 `find_packages(where='visionzip')`）；`pip install visionzip` 后 import 失败，需要把仓库根目录加入 PYTHONPATH | wheel 文件清单；setup.py | — | — |
| E-VZ-12 | DE | 通过官方 LLaVA generate() 跑 (54,10) 时，LLM 每层都只有 64 个视觉 token；`visionzip()` 的默认值 (191,30) 共 221 个，不是论文的任何一档预算 | EXP-0001 smoke_test.py；main.py 函数签名 | — | 复现时必须显式传入论文设置 |

## 研究判断

- **优点**：
  - 代码很短（约 600 行），两行代码即可接入官方 LLaVA。
  - LLM 侧可以继续用 SDPA / FlashAttention，prefill 所有层都变短，多轮对话可复用。
  - 作者报告的低预算结果在三篇中最好。
- **缺点**：
  - 完全不看问题，问题相关的小目标或文字可能被合并掉（E-VZ-04 是作者的反向主张，尚未验证）。
  - 合并 token 幅值翻倍、注入了 LLaVA 训练时没见过的 CLS token、位置被压缩：这三点都是“压缩后的输入与 LLM 训练分布不一致”的潜在来源（AI），论文是否讨论需回查。
  - 写死 CLIP-L；PyPI 包为空。
- **与自身实验的关系**：作为对照方法和组合实验素材（DEC-0009 改为以 FastV 为主 baseline，见 [baseline 选择](../专题分析/所有论文在baseline骨架的改进内容.md)）。E-VZ-07～09 是本项目自己发现、有代码证据的问题，可在组合实验和后续分析中使用。
- **核心总结**：VisionZip 把剪枝前移到视觉编码器，简单且强；但压缩后的 token 在幅值、CLS 和位置上与 LLaVA 训练分布不一致，这是值得验证的问题。

## 交接 Deep Read

候选 P1 ← E-VZ-06；P4 ← E-VZ-07；P5 分布匹配 ← E-VZ-07、E-VZ-08、E-VZ-09。F3 ← E-VZ-04；F6 ← E-VZ-09；F7 ← E-VZ-07、E-VZ-08。
