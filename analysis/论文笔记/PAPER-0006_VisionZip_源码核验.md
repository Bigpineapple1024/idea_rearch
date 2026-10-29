# ANALYSIS_PACKAGE：VisionZip: Longer is Better but Not Necessary in Vision Language Models

> 本文件是源码核验补充；原文笔记见 [PAPER-0006_VisionZip](PAPER-0006_VisionZip.md)。两份冲突时以能追溯到原文页码或源码行号的一方为准。

状态：partial。paper_id PAPER-0006 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 dvlab-research/VisionZip@8f86b55c6f000eb033e6912538af2dd7dcb30502、README、论文与补充材料检索摘要。
**论文全文已回查（2026-10-24）**：初稿写作时 arxiv.org、openaccess.thecvf.com 被拦截，SC 条目来自检索摘要。2026-10-24 用 MinerU 解析的 CVPR 正式版与补充材料（literature/processed/PAPER-0006/）和 PDF 回查了 E-VZ-C01、C02、C05、C07～C09，新增 E-VZ-C13、C14；Fig.7 只有图片，未能读取。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0006
analysis_id: PAPER-0006
created_at: 2026-10-23
source_ids: [arXiv:2412.04467, github:dvlab-research/VisionZip@8f86b55c6f000eb033e6912538af2dd7dcb30502, EXP-0001]
evidence_status: partial
evidence_locations: [README.md, visionzip/clip_encoder.py, visionzip/utils.py, visionzip/llava_arch.py, setup.py, EXP-0001]
missing_evidence:
- 已回查正文与补充材料；64/128/192 的 (dominant, contextual) 设置见补充材料 Table 6
- 真实权重上的复现见 EXP-0002（POPE 三档与论文相差 ≤0.2，E-VZ-C14）
next_skill: deep-read Mode C
source_identity_and_version: "Yang et al., CVPR 2025, arXiv 2412.04467；代码 Apache-2.0，8f86b55c (2025-07-21)"
research_question: 视觉 token 是否大量冗余，能否在进入 LLM 之前用文本无关的方式压缩到很少数量
study_type_and_context: 免训练推理（也提供高效微调和训练模式）；LLaVA-1.5、LLaVA-NeXT、Mini-Gemini、Qwen2.5-VL
design_units_samples_or_materials: LLaVA 系列评测集；视频任务
intervention_input_or_procedure: 在 CLIP 倒数第二层按 CLS 注意力选 dominant token，剩余 token 按 key 相似度合并成 contextual token
measurements_and_analysis: 准确率保留比例、prefill 时间、总推理时间
results_with_uncertainty: 见 E-VZ-C01～02；未见方差
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 见“研究判断”
reproducibility_and_transferability: 代码短小，能插入官方 LLaVA；但 PyPI 包为空、写死 24 层 CLIP、需传入论文预算（默认值不是论文设置）
candidate_P: [P1 注意力作为重要性的可靠性, P4 被丢弃信息的处理, P5 压缩 token 与 LLM 训练分布的匹配]
candidate_F: [F3 文本无关优于文本引导（作者主张）, F6 位置被压缩, F7 合并 token 幅值翻倍与 CLS 注入]
evidence_index: E-VZ-C01..E-VZ-C14
unresolved_questions: [合并幅值、CLS 注入与位置压缩对输出的影响（论文三点都没有讨论，已回查；本项目未测）]
```

## 来源

- 论文：Senqiao Yang 等，CVPR 2025，arXiv 2412.04467；补充材料 Table 6 给出 LLaVA-1.5 的设置：64 → (54, 10)，128 → (108, 20)，192 → (162, 30)（已回查；VisionZip 自己的 N 是进入 LLM 的 token 数，正文 §3.3）。
- 代码：<https://github.com/dvlab-research/VisionZip>，Apache-2.0，固定 commit 8f86b55c（2025-07-21）。
- 实际阅读范围：README；`visionzip/` 下全部源码（main.py、clip_encoder.py、utils.py、llava_arch.py）；setup.py；PyPI 上的 visionzip-0.1.3 wheel。

## 研究事实

- **核心问题**：视觉 token 的注意力集中在少数 token 上，大量 token 冗余，能否在送入 LLM 前直接压缩。
- **方法（代码实际做法）**：
  1. **dominant**：取 CLIP 倒数第二层（第 22 层）CLS 对各 patch 的注意力，对 head 求和，保留前 dominant−1 个 patch，**连同 CLS 本身**一起保留（E-VZ-C06、E-VZ-C08）。
  2. **contextual**：其余 token 用同层 key（对 head 取平均）的余弦相似度，以等间隔选出的 token 为中心，其他 token 分配给最相似的中心；contextual token = 中心 token + 被分配 token 的**平均**（E-VZ-C07）。
  3. dominant 按原始空间顺序排列，之后接 contextual，送入投影层和 LLM；LLM 的所有层都只看到这些 token（E-VZ-C12）。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-VZ-C01 | SC（已回查，部分成立） | Fig.1 称 LLaVA-1.5 在 11 个基准上用约 10% 的 token 保留近 95% 性能。对应 Table 1：LLaVA-1.5-7B、免训练、576→64（11.1%），GQA、MMB、MME、POPE、SQA、VQAv2、TextVQA、MMMU、SEED、MMVet、LLaVA-Bench 11 项归一化分数的算术平均为 94.0%；只微调投影层的 VisionZip‡ 为 95.2%。13B 为 93.7%/94.8%（补充 Table 10，9 项平均，不含 LLaVA-Bench 与 MMVet） | 正文 p.1 Fig.1、p.5 Table 1；补充 §B.1.3、Table 10 | LLaVA-1.5 7B/13B | “约 95%”对应的是微调版或四舍五入 |
| E-VZ-C02 | SC（已回查） | Table 1 的 64 档 11 项归一化均值：VisionZip 94.0%，FastV 75.6%，SparseVLM（2024.10 版）85.8%，VisionZip‡ 95.2%；§3.1 写作领先 18.4 与 8.2 个点。论文没有说明对比方法的“64”怎么算：只写沿用 FastV、SparseVLM 的原设置，没有给出 FastV 的 K/R，也没有说明是进入 LLM 的数量、第 K 层之后的数量还是层平均 | 正文 p.4 §3.1、p.5 Table 1；补充材料 | 作者自己的对比 | 同为“FastV@64”，VisionZip 与 SparseVLM 两篇的表部分相同（GQA 46.1、TextVQA 47.8），部分不同（POPE 48.0 vs 38.2，SQA 51.1 vs 68.7） |
| E-VZ-C03 | SC | 注意力集中于少数 token，大量 token 注意力很低 | 论文（检索摘要） | — | 低注意力 ≠ 无信息 |
| E-VZ-C04 | SC | 文本引导的方法常选到非主导 token，信息更少，性能更低；在 LLM 外剪枝更高效 | 补充材料（检索摘要） | — | 与 SparseVLM 的主张相反，需在相同预算下比较 |
| E-VZ-C05 | SC（已回查） | 文本无关，适合多轮对话，能与其他 LLM 加速方法叠加；显著降低 prefill 时间。多轮对话的优势在论文中只有 §4.3 的论证和 Fig.7 一个示例：前提是首轮选出的视觉 token 存在 KV cache 中、被后续各轮复用；正文和补充材料都没有多轮评测协议、预算或数值 | README；正文 p.8 §4.3、Fig.7（图片，未能读出 token 数） | — | 多轮优势未经系统评测 |
| E-VZ-C06 | DE | dominant 用 CLIP 第 22 层（0 起计）CLS→patch 注意力对 head 求和后 top-k；contextual 用同层 key 的 head 平均 | clip_encoder.py；utils.py apply_info；EXP-0001 probe_visionzip.py | — | — |
| E-VZ-C07 | DE | contextual token = 中心 + 被分配 token 的平均；当所有 token 相同时，合并后的值是原来的 2 倍。论文 Algorithm 2 只写“按分配索引平均合并”（avg_merge，已回查 p.3），没有说明中心 token 是否计入平均；代码得到“中心 + 平均”，与“平均”的字面含义不一致。论文没有讨论合并 token 的幅值 | clip_encoder.py；EXP-0001 probe_visionzip.py 第 2 部分；正文 p.3 Algorithm 2 | — | 是否有意为之、对 LLM 的影响未知 |
| E-VZ-C08 | DE | 第 0 个视觉 token 是 CLIP 倒数第二层的 CLS 隐状态；而 LLaVA-1.5 训练时用 `select_feature='patch'` 去掉了 CLS。论文 Algorithm 1 明确把 CLS 并入 dominant token（设计如此），但没有讨论 LLaVA 训练时不含 CLS 带来的偏移（已回查 p.2） | EXP-0001 probe_visionzip.py 第 1 部分；LLaVA clip_encoder.py L37-38；正文 p.2 Algorithm 1 | LLaVA-1.5 路径 | 影响大小未知 |
| E-VZ-C09 | DE | 压缩后的 token 在 LLM 中按连续位置排列，LLM 看不到原图中的空间间隔；正文和补充材料都没有讨论压缩后的 position id（已回查） | llava_arch.py prepare_inputs_labels_for_multimodal_visionzip（position_ids 为 None，由 generate 按长度生成） | LLaVA-1.5 | 与 FastV、SparseVLM+ 保留原始位置的做法相反 |
| E-VZ-C10 | DE | 写死 24 层 CLIP 的调度（r 列表长度 24、第 22 层取 metric），anyres 路径写死 24×24 网格；换视觉编码器需要改代码 | utils.py apply_info；llava_arch.py restore_image_features_sorted | — | — |
| E-VZ-C11 | DE | PyPI 上 visionzip 0.1.3 的 wheel 只有元数据、没有代码（setup.py 写成 `find_packages(where='visionzip')`）；`pip install visionzip` 后 import 失败，需要把仓库根目录加入 PYTHONPATH | wheel 文件清单；setup.py | — | — |
| E-VZ-C12 | DE | 通过官方 LLaVA generate() 跑 (54,10) 时，LLM 每层都只有 64 个视觉 token；`visionzip()` 的默认值 (191,30) 共 221 个，不是论文的任何一档预算 | EXP-0001 smoke_test.py；main.py 函数签名 | — | 复现时必须显式传入论文设置 |
| E-VZ-C13 | SC（已回查） | 论文讨论的“不对齐”只针对 token 数骤减：在全部 token 上训练的模型可能不适应输入突然变少。对应的 VisionZip‡ 冻结其余部分，只用 1/10 的 LLaVA-1.5 指令数据微调投影层（7B 在 8×A800 上约 30 分钟，学习率 2e-5）：64 档 94.0%→95.2%，128 档 97.6%→98.4%，192 档 98.5%→99.1%；补充 Table 16 显示换用别的微调数据变化不到 0.5%，作者据此认为增益来自适应 token 减少。另有训练阶段使用的模式（补充 Table 12，64 个 token 为 97.1%） | 正文 p.2 §1、p.4 §2.4、p.5 Table 1；补充 p.2、Table 12、Table 16 | LLaVA-1.5 | 幅值、CLS、位置三个具体来源没有被单独处理 |
| E-VZ-C14 | DE | EXP-0002（LLaVA-1.5-7B，RTX 3090）：POPE 85.49/83.12/77.03，论文 85.3/83.2/77.0；MME P+C 1783.8/1759.2/1696.0，论文 1782.6/1761.7/1690。POPE 上同档预算每题最快（0.158/0.143/0.128 秒）。64 档 MME 的 OCR 不降（40 题），count 降 41.7、color 降 25 | EXP-0002 记录与 tables.md | 一个模型、两个 benchmark | — |

## 研究判断

- **优点**：
  - 代码很短（约 600 行），两行代码即可接入官方 LLaVA。
  - LLM 侧可以继续用 SDPA / FlashAttention，prefill 所有层都变短，多轮对话可复用。
  - 作者报告的低预算结果在三篇中最好。
- **缺点**：
  - 完全不看问题，问题相关的小目标或文字可能被合并掉（E-VZ-C04 是作者的反向主张）。EXP-0002 中 64 档 MME 的 OCR 没有下降（只有 40 题），计数类损失最大；按物体的召回与 SparseVLM 整体相同，但丢失的物体不同（探索性，SQZ-EXP-0002）。
  - 合并 token 幅值翻倍、注入了 LLaVA 训练时没见过的 CLS token、位置被压缩：这三点都是“压缩后的输入与 LLM 训练分布不一致”的潜在来源（AI）。已回查：论文只讨论了 token 数骤减造成的不对齐，并用只微调投影层的 VisionZip‡ 处理（E-VZ-C13）；这三个具体来源论文都没有讨论。
  - 写死 CLIP-L；PyPI 包为空。
- **与自身实验的关系**：作为对照方法和组合实验素材（DEC-0011 改为以 FastV 为主 baseline，见 [baseline 选择](../专题分析/所有论文在baseline骨架的改进内容.md)）。（2026-10-29 完整阅读补注：这是 DEC-0011～DEC-0014 时的定位。DEC-0044 起，VisionZip-64 被用作 IDEA-0006 的开发底座（IDEA-0006 已于 DEC-0054 结束）；DEC-0049 的 IDEA-0007 也用它的 CLS 分数排序；FastV 一族的结果保留。）E-VZ-C07～09 是本项目自己发现、有代码证据的问题，可在组合实验和后续分析中使用。
- **核心总结**：VisionZip 把剪枝前移到视觉编码器，简单且强；但压缩后的 token 在幅值、CLS 和位置上与 LLaVA 训练分布不一致，这是值得验证的问题。

## 交接 Deep Read

候选 P1 ← E-VZ-C06；P4 ← E-VZ-C07；P5 分布匹配 ← E-VZ-C07、E-VZ-C08、E-VZ-C09。F3 ← E-VZ-C04；F6 ← E-VZ-C09；F7 ← E-VZ-C07、E-VZ-C08。

（2026-10-29 完整阅读补注）P1、P4、P5 和 F3、F6、F7 只是本分析包写作时给的候选标签，不是 [MODE_C-0002](../MODE_C-0002_视觉token剪枝.md) 的编号；这里不改号。P1（注意力能否代表重要性）、F3、F6、F7 与矩阵一致。此处的 P5“压缩 token 与 LLM 训练分布的匹配”在矩阵中是 P2（证据 E-VZ-C07～C09）；矩阵的 P5 是“按首轮问题剪枝的表示能否用于后续问题”（证据 E-VZ-5、E-VZ-C05 等）；此处的 P4“被丢弃信息的处理”也不是矩阵的 P4（不同选择规则各自丢失哪类信息）。跨文件引用时写明所属文件。
