# ANALYSIS_PACKAGE：An Image is Worth 1/2 Tokens After Layer 2（FastV）

> 本文件是源码核验补充；原文笔记见 [PAPER-0004_FastV](PAPER-0004_FastV.md)。两份冲突时以能追溯到原文页码或源码行号的一方为准。

状态：partial。paper_id PAPER-0004 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 pkunlp-icler/FastV@d1659729b5bf1be225e99ee15783deeea80f63b1、仓库 README、论文检索摘要。
**论文全文未读**：本环境网络策略拦截 arxiv.org 与 ecva.net，标 SC 的论文内容来自检索摘要和 README，需回查原文。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0004
analysis_id: PAPER-0004
created_at: 2026-10-23
source_ids: [arXiv:2403.06764, github:pkunlp-icler/FastV@d1659729b5bf1be225e99ee15783deeea80f63b1, EXP-0001]
evidence_status: partial
evidence_locations: [README.md, src/FastV/lmms-eval/fastv_kvcache.py L119-142, src/transformers/src/transformers/models/llama/modeling_llama.py L717-799, EXP-0001]
missing_evidence:
- 论文全文（方法节的排序公式、FLOPs 公式、各 benchmark 完整表格）未读取
- 未在真实 LLaVA-1.5-7B 权重上运行
next_skill: deep-read Mode C
source_identity_and_version: "Chen et al., ECCV 2024 (oral), arXiv 2403.06764; 代码 d1659729 (2025-01-04)，仓库根目录无 LICENSE 文件"
research_question: LVLM 深层对视觉 token 的注意力是否低效，能否在推理时直接丢弃一部分视觉 token 而不损失性能
study_type_and_context: 免训练推理加速；LLaVA-1.5 7B/13B 等，图像与视频任务
design_units_samples_or_materials: 见“研究事实”；样本为各 benchmark 的问题
intervention_input_or_procedure: 在第 K 层前，按第 K-1 层最后一个 token 对图像 token 的平均注意力保留前 R 比例
measurements_and_analysis: 理论 FLOPs、benchmark 分数、首 token 延迟、显存
results_with_uncertainty: 见 E-FV-C02～04；未报告重复或方差（README 表格为单次数字）
claim_evidence_implementation_map: 见 E 表，E-FV-C05～09 为代码核验
limitations_biases_and_ethics: 见“研究判断”
reproducibility_and_transferability: 可运行，但 KV-cache 版有写死常量与 KV 重复写入，需要 README 的 rotary 补丁（E-FV-C07、E-FV-C08）
candidate_P: [P1 注意力作为重要性的可靠性, P3 固定预算在层间的分配]
candidate_F: [F1 深层对图像注意力很低, F4 预算口径不同, F6 位置处理不同]
evidence_index: E-FV-C01..E-FV-C10
unresolved_questions: [论文是否对比随机选择, 最后一个 token 作为 query 的理由]
```

## 来源

- 标题、作者、年份、平台：Liang Chen 等，ECCV 2024 Oral，arXiv 2403.06764。
- 代码：<https://github.com/pkunlp-icler/FastV>，固定 commit d1659729（2025-01-04）。仓库根目录**没有 LICENSE 文件**，引用和改写前需确认许可。
- 实际阅读范围：README 全文；`fastv_kvcache.py` 全文；修改版 `modeling_llama.py` 的 FastV 分支（L717–799）。论文全文未取得。

## 研究事实

- **核心问题**：LVLM 深层对视觉 token 的注意力非常低效，能否丢弃一部分视觉 token 以加速推理。
- **方法**：在第 K 层之前，用第 K-1 层的注意力给图像 token 排序，只保留前 R 比例（论文默认 K=2、R=50%），其余丢弃。不需要训练。
- **两种实现**：①“token mask”版：只用 mask 屏蔽 token，**不加速**，README 说明它用于方便公平地比较性能；②inplace / KV-cache 版：真正删除 token，用于测速。
- **报告的结果**：13B 模型理论 FLOPs 降低 45% 而性能不降（SC）；lmms-eval 上 LLaVA-1.5-7B、K=3、R=50%（FLOPs 57%）时，GQA 62.0→60.4，SEED 60.5→59.9，MME-P 1510.8→1511.7（SC，README 表格）。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-FV-C01 | SC | 图像描述任务中，深层图像 token 平均注意力仅为系统提示的 0.21%，前两层约为 50% | 论文摘要/引言（检索摘要） | LLaVA-1.5 等，描述任务 | 未回查原文；注意力低不等于信息不重要 |
| E-FV-C02 | SC | LLaVA-1.5-13B 理论 FLOPs 降 45%，性能不降 | 论文摘要、README | 13B，论文所列任务 | 未回查每个 benchmark 的数字 |
| E-FV-C03 | SC | lmms-eval：7B、K=3、R=50% 时 GQA −1.6、SEED −0.6、MMMU −0.1，描述类基本不变 | README “Support LMMs-Eval” 表 | 单次数字，无方差 | K=3 与论文默认 K=2 不同 |
| E-FV-C04 | SC | A100 上 13B、K=2、R=75% 首 token 延迟 0.124s，vanilla 0.203s | README 延迟表 | A-OKVQA，A100 80G | 单 GPU 型号 |
| E-FV-C05 | DE | 排序依据：第 K-1 层**最后一个 token** 对图像 token 的注意力，对所有 head 取平均；保留下来的 token 使用**原始 position id** | modeling_llama.py L738-756；fastv_kvcache.py L124-131 | 两种实现一致 | 论文是否这样描述需回查 |
| E-FV-C06 | DE | token mask 版不加速；K=0 时随机选择 | README；modeling_llama.py L763-798 | — | — |
| E-FV-C07 | DE | KV-cache 版写死 K=3、R=0.5、系统提示长度 35、图像长度 576；换 prompt 模板会静默取错片段 | fastv_kvcache.py L119-126 | 只适用于 LLaVA-1.5 vicuna_v1 模板 | — |
| E-FV-C08 | DE | KV-cache 版用 eager 注意力会报错；用 SDPA 但不打 README 第 5 步的 rotary 补丁会 IndexError；打补丁后能跑，但第 K-1 层被执行两次，KV cache 长度翻倍（1270 vs 635）。输出与不重复写入的实现一致（相对误差 ≤3e-7） | EXP-0001 probe_fastv.py、check_fastv_equivalence.py | 随机权重小模型 | 只影响效率测量，不影响准确率 |
| E-FV-C09 | DE | KV-cache 版只在 prefill 剪一次，之后的解码 token 都看同一组图像 token；原始实现每次前向都重新剪 | README “Support KV Cache” | — | 两种版本的分数可能不同 |
| E-FV-C10 | AI | 按“32 层平均 64 个 token”换算，K=2 时第 2 层以后只剩 30 个图像 token；同 FLOPs 下深层的视觉信息远少于 VisionZip 的 64 | EXP-0001 flops_budget.py | 理论计算 | 可解释其他论文中 FastV 在低预算下掉点多，待实测 |

## 研究判断

- **优点**：思路简单，几十行代码就能实现；不需要训练；给出了“深层视觉注意力低”这个被后续工作广泛引用的观察。
- **缺点**：
  - 重要性只来自一个 query（最后一个 token）和一层注意力，后续工作指出注意力分数有位置偏差，在部分数据集上甚至不如随机丢弃（见 MODE_C S-04）。
  - 前 K 层仍处理全部 576 个 token，prefill 显存峰值不下降。
  - 打分那一层需要显式注意力矩阵，不能用 FlashAttention。
- **可能的混杂**：README 中 lmms-eval 的数字是单次运行，K 与论文默认值不同，而且 token mask 版和真正删除 token 的版本可能给出不同分数。
- **与自身实验的关系**：**视觉 token 剪枝方向的主 baseline**（DEC-0011，按研究流程第 ① 步：近两年高引用、有源码、衍生工作最多）。官方代码有无许可证、依赖修改版 transformers、写死常量等问题，因此本项目在自己的重新实现 `code/token_pruning/fastv_llava.py` 上开展实验；该实现已证明与官方实现输出一致。
- **核心总结**：FastV 提出了“LLM 内部按注意力剪视觉 token”的范式；它的观察（F1）比它的选择规则更有价值。

## 交接 Deep Read

候选 P1（注意力作为重要性是否可靠）← E-FV-C05、S-04；F1（深层视觉注意力低）← E-FV-C01；F4（预算口径）← E-FV-C10；F6（保留原始位置）← E-FV-C05。作者主张均为 SC，未经本项目在真实模型上验证。
