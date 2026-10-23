# EXPERIMENT_RECORD：三个 token 剪枝仓库的代码级核验（CPU，随机权重）

状态：completed（代码级核验）。**不是性能复现**：没有真实权重和 benchmark，不产生任何准确率或速度结论。
实验 ID EXP-0001 / IDEA ID none / 2026-10-23 / 执行者 Claude Code 云会话（无 GPU）。

## 结构化交接包

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
package_id: EXP-0001
experiment_id: EXP-0001
created_at: 2026-10-23
source_ids: [PAPER-0004, PAPER-0005, PAPER-0006]
evidence_status: verified
evidence_locations: [code/token_pruning/probes/, code/token_pruning/check_fastv_equivalence.py, code/token_pruning/smoke_test.py, 本文“原始输出”节]
missing_evidence:
- 真实 LLaVA-1.5-7B 权重下的行为（本环境无法访问 huggingface.co，且没有 GPU）
- SparseVLM 完整推理路径未在 CPU 上运行（代码写死 .cuda()），只测试了其打分函数与预算表
next_skill: squeeze-experiment
source_idea_id: none
plan_version: "无 IDEA 包；协议 = 论文笔记中列出的待核验代码问题"
execution_status: completed
research_question: 三个仓库的代码实际做了什么，是否与论文和 README 的描述一致，能否在官方 LLaVA 推理路径中跑通
hypothesis: 无预设方向的核验
research_question_and_hypothesis:
  research_question: 三个仓库的代码实际做了什么，是否与论文和 README 的描述一致，能否在官方 LLaVA 推理路径中跑通
  hypothesis: 无预设方向的核验
planned_vs_actual_design_and_controls: 见“计划与实际”
actual_configuration: 见“计划与实际”
samples_materials_or_data_provenance: 随机初始化的小模型，固定种子；无真实数据
all_runs_or_repetitions: 见“全部运行”
results_with_uncertainty: 确定性代码检查，重复运行两次结果一致；不涉及统计不确定性
observations_and_deviations: 见“分层记录”
known_limits_and_missing_evidence: 随机权重只能检验代码路径和数据流，不能代表真实模型的注意力分布
next_routes: [GPU 上执行 code/token_pruning/run_pope_mme.sh 复现 baseline, 结果回流 MODE_C-0001]
```

## 计划与实际

- **问题**：逐项核验论文笔记中的代码问题，包括选择规则、预算口径、位置处理、合并方式、KV cache 行为和工程可运行性。
- **代码**：FastV d1659729、SparseVLM main a9e71427（v1.5 分支 734d9003 只做了文件对比）、VisionZip 8f86b55c、LLaVA c121f043；本项目脚本见 code/token_pruning/（本次提交）。
- **环境**：Linux 云容器，4 CPU，15GB 内存，无 GPU；Python 3.11.x、torch 2.1.2+cu121（CPU 运行）、transformers 4.37.2、tokenizers 0.15.1、accelerate 0.21.0、timm 0.6.13、torchvision 0.16.2。
- **模型**：随机初始化，固定种子 0/1。小 Llama：hidden 128，6 层，4 头；小 LLaVA：LLM hidden 64、6 层，CLIP 保持 ViT-L/14-336 的几何（24 层、576 patch）但宽度 64。
- **与计划的偏差**：原本计划阅读论文全文，但 arxiv.org、openaccess.thecvf.com、ecva.net 被网络策略拦截；也无法下载真实权重。因此论文侧证据都标为 SC（来自检索摘要）。

## 全部运行

| 运行 ID | 脚本 | 状态 | 实际配置 | 原始输出 |
| --- | --- | --- | --- | --- |
| EXP-0001-R1 | probes/probe_fastv.py | completed（4 组中 3 组按预期报错） | FastV 官方 fastv_kvcache.py，eager/sdpa × 是否打 rotary 补丁 | 下文 R1 |
| EXP-0001-R2 | probes/probe_sparsevlm.py | completed | score.py v1.0/v2.0；RoPE 除数按 LLaVA-1.5-7B 维度计算 | 下文 R2 |
| EXP-0001-R3 | probes/probe_visionzip.py | completed | VisionZip clip_encoder/utils，随机 CLIP + 构造输入 | 下文 R3 |
| EXP-0001-R4 | probes/flops_budget.py | completed | LLaVA-1.5-7B LLM 维度，文本 20/60 token | 下文 R4 |
| EXP-0001-R5 | check_fastv_equivalence.py | completed | 官方实现 vs fastv_llava，K=3，keep=288，5 步解码 | 下文 R5 |
| EXP-0001-R6 | smoke_test.py（运行两次） | completed，两次输出完全一致 | 官方 LLaVA generate()，vanilla / FastV(2,288) / FastV(2,30) / VisionZip(54,10) | 下文 R6 |

## 原始输出

```text
### R1 probe_fastv.py
expected per-layer cache length without duplication: [635, 635, 635, 347, 347, 347]
attn=eager rotary_patch=False -> error: ValueError: Attention mask should be of size (1, 1, 635, 1270), but is torch.Size([1, 1, 635, 635])
attn=eager rotary_patch=True  -> error: ValueError: Attention mask should be of size (1, 1, 635, 1270), but is torch.Size([1, 1, 635, 635])
attn=sdpa  rotary_patch=False -> error: IndexError: index 515 is out of bounds for dimension 0 with size 347
attn=sdpa  rotary_patch=True  -> ok, prefill_cache_len_per_layer [635, 635, 1270, 347, 347, 347]

### R2 probe_sparsevlm.py
B=2: kept visual tokens per sample: [129, 0] (budget at layer 2 = 66)
v1.0 64: (66,30,17) mean 61.7 | 96: (238,48,26) 93.1 | 128: (303,110,36) 123.9 | 192: (300,200,110) 188.2
v2.0 64: (66,34,20) mean 64.4 | 96: (246,54,28) 96.8 | 128: (238,108,60) 128.0 | 192: (300,200,118) 192.4
first 2 layers = 576 in every setting
RoPE divisor, text query at 650 over 576 visual keys: min 1.509, max 6.179; <=0: 0
  distance 40: 5.759 | 100: 5.399 | 200: 3.458 | 300: 3.474 | 400: 3.150 | 500: 3.377 | 600: 2.349

### R3 probe_visionzip.py
output tokens: 64 (dominant incl. CLS = 54, contextual = 10)
token 0 equals CLS hidden state of layer -2: True
dominant patch tokens are passed in spatial order: True
metric layer: 22 (0-based)
identical-token input: dominant token value 1.0 | contextual token values [2.0]

### R4 flops_budget.py（文本 20 / 60 token）
vanilla 576                    4.19 / 4.46 TFLOPs 100%     KV 315.5 / 335.5 MB
VisionZip 54+10                0.77 / 1.04       18.5/23.2% KV 59.5 / 79.5 MB
FastV K=2 keep 30 (mean 64)    0.78 / 1.04       18.6/23.3% KV 59.6 / 79.6 MB
FastV K=2 keep 288             2.37 / 2.64       56.6/59.2% KV 180.5 / 200.5 MB
SparseVLM v1.0 '64'            0.76 / 1.03       18.2/23.0% KV 58.4 / 78.4 MB
SparseVLM+ v2.0 '64'           0.78 / 1.04       18.7/23.4% KV 59.7 / 79.7 MB

### R5 check_fastv_equivalence.py
KV cache length per layer  official: [640, 640, 1280, 352, 352, 352]  fastv_llava: [640, 640, 640, 352, 352, 352]
relative L2 difference: prefill 3.23e-07, decode 1-5: 2.44e-07 2.40e-07 2.77e-07 2.65e-07 2.28e-07

### R6 smoke_test.py（两次运行一致）
vanilla           visual tokens per layer [576]*6  ok
fastv_k2_keep288  [576, 576, 288, 288, 288, 288]  ok
fastv_k2_keep30   [576, 576, 30, 30, 30, 30]      ok
visionzip_54_10   [64]*6                          ok
```

## 汇总

- 纳入全部 6 个运行，无排除。R1 中 3 组报错是被检验的行为本身，不是运行失败。
- 本记录只说明代码行为，不说明任何方法在真实模型上的准确率或速度。

## 分层记录

- **观察事实（E，已登记到各论文笔记）**：
  - FastV KV 版：eager 下崩溃；不打 rotary 补丁会 IndexError；第 K-1 层 cache 重复写入，但输出与不重复的实现一致（E-FV-C08）。
  - SparseVLM：batch=2 出错（E-SV-C11）；“64”是层平均口径（E-SV-C06）；v2 的 RoPE 除数为正值、随距离减小（E-SV-C08）。
  - VisionZip：CLS 被送入 LLM（E-VZ-C08）；合并 token 在相同输入下是原值的 2 倍（E-VZ-C07）。
  - 在同一平均预算下，三者的理论 FLOPs 相同，但每层的视觉 token 分布不同（R4、R6）。
- **当前解释（推断）**：VisionZip 在低预算下表现更好，可能部分来自深层保留更多视觉 token，而不完全是选择规则更好。
- **待复验假设**：VisionZip 的 CLS 注入、合并幅值和位置压缩会影响真实模型的输出（见 MODE_C-0001 的 C 点）。
- **不能由本次数据得出的结论**：任何方法的准确率、速度或优劣；随机权重下的注意力分布不代表真实模型。

## 交接

原始输出已全文保存在本文件中，可用 code/token_pruning/ 的脚本按 README 重新生成。下一步：在 GPU 上执行 run_pope_mme.sh 完成真实复现（EXP-0002），并把本记录的事实交给 deep-read 回流 MODE_C-0001。
