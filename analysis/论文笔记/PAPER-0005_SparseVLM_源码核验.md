# ANALYSIS_PACKAGE：SparseVLM（ICML 2025）与 SparseVLM+（TPAMI 2026）

状态：partial。paper_id PAPER-0002 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 Gumpest/SparseVLMs main@a9e71427220c4cacb6afb5a96d23cfb43142e0c3 与 v1.5@734d9003879689713a132b83297dfad060a5c9af、README、论文检索摘要。
**论文全文未读**（arxiv.org 被网络策略拦截），SC 条目需回查原文。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0002
analysis_id: PAPER-0002
created_at: 2026-10-23
source_ids: [arXiv:2410.04417, github:Gumpest/SparseVLMs@a9e71427220c4cacb6afb5a96d23cfb43142e0c3, github:Gumpest/SparseVLMs@734d9003879689713a132b83297dfad060a5c9af, EXP-0001]
evidence_status: partial
evidence_locations: [README.md, llava/model/language_model/score.py, llava/model/language_model/modelling_sparse_llama.py L186-322 L540-615 L651-668, llava/model/language_model/utils.py cluster_and_merge, EXP-0001]
missing_evidence:
- 论文全文未读取；各版本论文数字不同（v1 与 ICML 版），需确认引用的是哪个版本
- 论文描述的“按注意力矩阵的秩自适应确定稀疏率”在 main 与 v1.5 分支代码中均未找到
- 未在真实权重上运行
next_skill: deep-read Mode C
source_identity_and_version: "Zhang et al., ICML 2025 (arXiv 2410.04417)；SparseVLM+ TPAMI 2026；代码 Apache-2.0，main 分支用 USE_VERSION=2_0 切换到 SparseVLM+"
research_question: 视觉 token 应由问题文本引导、在 LLM 内部逐层稀疏，能否在保持准确率的同时降低计算
study_type_and_context: 免训练推理加速；LLaVA-1.5、Mini-Gemini、Video-LLaVA 等
design_units_samples_or_materials: LLaVA-1.5 评测集（GQA、MMB、MME、POPE、SQA、VQAv2、TextVQA 等，按 README 脚本）
intervention_input_or_procedure: 从文本中选“评分者”token，在第 2/6/15 层按其对视觉 token 的注意力保留固定数量，被剪 token 部分聚类回收（v1），v2 取消回收并做 RoPE 校正
measurements_and_analysis: FLOPs、CUDA 时间、准确率保留比例
results_with_uncertainty: 见 E-SV-02；不同论文版本数字不同，未见方差
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 见“研究判断”
reproducibility_and_transferability: 只支持 batch size 1；写死 .cuda() 和作者集群路径；需要 flash_attn 2.3.3 与 transformers 4.37.0
candidate_P: [P1 注意力作为重要性的可靠性, P2 位置偏差, P3 层间预算分配, P4 被丢弃信息的处理]
candidate_F: [F2 SparseVLM+ 的 RoPE 校正, F3 文本引导 vs 文本无关, F4 预算口径, F5 回收被取消, F6 位置处理不同]
evidence_index: E-SV-01..E-SV-12
unresolved_questions: [秩自适应策略在哪个版本的代码里, v2 取消回收的消融数据]
```

## 来源

- 论文：Yuan Zhang 等，“SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference”，ICML 2025，arXiv 2410.04417；扩展版 SparseVLM+，TPAMI 2026。
- 代码：<https://github.com/Gumpest/SparseVLMs>，Apache-2.0。README 写明 ICML 版本对应 v1.5 分支（734d9003），main（a9e71427，2026-07-30）默认运行 v1.0 逻辑，设置 `USE_VERSION=2_0` 后运行 SparseVLM+。
- 实际阅读范围：README；`score.py`；`utils.py`；`modelling_sparse_llama.py` 的稀疏化主循环（L186–322）和注意力（L540–668）；eval 脚本；v1.5 分支的 `score.py`。

## 研究事实

- **核心问题**：已有方法（如 FastV）不考虑问题文本；作者主张应按问题自适应地稀疏视觉 token。
- **方法（代码实际做法）**：
  1. **选评分者**：在 LLM **输入嵌入层**，计算视觉嵌入与文本嵌入的点积，对文本维 softmax 后按视觉 token 取平均，高于均值的文本 token 作为评分者，只算一次（E-SV-04）。
  2. **逐层剪枝**：在第 2、6、15 层，用评分者对视觉 token 的 softmax 注意力（head 平均；v2 只取 text→visual 注意力最大的 14 个 head）打分，保留固定数量（E-SV-05、E-SV-07）。
  3. **回收**（仅 v1）：被剪 token 中相关度前 30% 用 DPC-KNN 聚成约 1/10 个 token，插回序列（E-SV-10）。
  4. **SparseVLM+（v2）**：注意力除以“全 1 的 q/k 经 RoPE 后的点积”来校正位置衰减，保留原始 position id，并取消回收（E-SV-08、E-SV-09）。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-SV-01 | SC | 用相关文本 token 作评分者；用注意力矩阵的秩决定自适应稀疏率；回收被剪 token | 论文摘要（检索摘要） | — | 秩自适应在代码中未找到（E-SV-05） |
| E-SV-02 | SC | ICML 版：LLaVA FLOPs 降 54%、CUDA 时间降 37%、准确率保留 97%；arXiv v1：FLOPs 降 61～67%、保留 93% | 论文摘要（不同版本） | LLaVA-1.5 | 版本间不一致，引用时注明版本 |
| E-SV-03 | SC | 与 FastV 延迟相近时高 11.2～17.3% | arXiv v1 摘要 | LLaVA | 未回查表格 |
| E-SV-04 | DE | 评分者在第 0 层输入嵌入上选一次：`m_v_t = v_t @ t_t^T`，softmax(dim=文本)，按视觉取平均，大于均值者入选 | modelling_sparse_llama.py L199-204 | main 分支 | 输入嵌入层的相似度能否代表语义相关性未验证 |
| E-SV-05 | DE | 剪枝层固定为 [2, 6, 15]，每阶段保留数来自查表（如 64 → 66/30/17），只支持 64/96/128/192；main 与 v1.5 分支都没有基于矩阵秩的计算 | score.py；v1.5 分支 score.py；全仓库搜索 rank/svd/linalg 无结果 | 两个分支 | 论文版本与代码可能不一致 |
| E-SV-06 | DE | “保留 64”是 32 层平均值：前 2 层 576 个，最后 17 层 17（v1）或 20（v2）个；平均 61.7 / 64.4 | EXP-0001 probe_sparsevlm.py 第 2 部分 | — | 与 VisionZip 的 64 口径不同 |
| E-SV-07 | DE | 打分是 softmax 后的注意力，v1 对所有 head 取平均，v2 只取 text→visual 注意力和最大的 14 个 head | score.py；L253-259、L562-576 | — | — |
| E-SV-08 | DE | v2 把注意力除以“全 1 q/k 的 RoPE 点积”；在 LLaVA-1.5-7B 维度下该除数在 1.5～6.2 之间、距离越远越小，因此会放大远处视觉 token 的分数 | L241-251；EXP-0001 probe_sparsevlm.py 第 3 部分 | head_dim=128，base=10000 | 作者主张是“注意力引力校正”；实际效果需消融 |
| E-SV-09 | DE | v1 剪枝后把 position id 压成连续编号；v2 保留原始 position id | L311-314、L653-662 | — | — |
| E-SV-10 | DE | v1 回收：被剪 token 中相关度前 30% 用 DPC-KNN 聚成 n/10+1 个；v2 设 `s_flag=False` 取消回收 | L273-306；utils.py cluster_and_merge | — | 取消回收的理由与消融未见 |
| E-SV-11 | DE | batch=2 时第二个样本保留 0 个视觉 token（mask 只写到第 0 行）；写死 `.cuda()`；eval 脚本路径是作者集群的 /mnt/bn/... | EXP-0001 probe_sparsevlm.py 第 1 部分；L552、L655；scripts/v1_5/eval/*.sh | — | 必须用 batch size 1 评测 |
| E-SV-12 | DE | score.py 注释“2*576 4*300 10*200 16*110”与剪枝层 [2,6,15] 实际对应的 2/4/9/17 层不一致 | score.py 第 9 行 | — | 只是注释问题 |

## 研究判断

- **优点**：明确引入问题文本引导；多阶段逐层剪枝；后续版本（SparseVLM+）主动处理了 RoPE 带来的位置偏差，说明作者也认识到注意力打分有位置问题。
- **缺点**：
  - 论文描述的秩自适应在代码中找不到，预算是查表得来的。
  - 前 2 层仍处理全部 576 个 token；打分层需要显式注意力。
  - 只支持 batch size 1；工程上写死较多。
  - 回收机制在 v2 中被取消，说明它的价值不确定（推断）。
- **可能的混杂**：v1 / v1.5 / v2 三个版本的打分、位置和回收都不同，引用数字时必须对应代码版本；“64”是层平均口径，与 VisionZip 直接比较时口径不同。
- **与自身实验的关系**：作为“文本引导、LLM 内部剪枝”的代表，用作对照方法和组合实验素材（DEC-0009）；工程上依赖其 LLaVA 分叉、FlashAttention 编译、写死较多，组合时优先移植其打分规则，而不是整体搬用其代码。
- **核心总结**：SparseVLM 的贡献在于“文本引导 + 逐层剪枝”的框架；SparseVLM+ 的 RoPE 校正和保留原始位置，对本项目的“位置偏差”问题最有参考价值。

## 交接 Deep Read

候选 P1 ← E-SV-07；P2 位置偏差 ← E-SV-08、E-SV-09；P3 层间预算 ← E-SV-05、E-SV-06；P4 被丢弃信息 ← E-SV-10。F2 ← E-SV-08；F3 ← E-SV-01；F4 ← E-SV-06；F5 ← E-SV-10；F6 ← E-SV-09。
