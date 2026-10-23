# ANALYSIS_PACKAGE：DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting

> 本文件是源码核验补充；原文笔记见 [PAPER-0001_DUET](PAPER-0001_DUET.md)。两份冲突时以能追溯到原文页码或源码行号的一方为准。

状态：partial（代码已核验，ETTh1 已在 CPU 上真实运行；论文全文未读）。paper_id PAPER-0001 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 decisionintelligence/DUET@dcc6e6780a9138731b64b9b5398a94a1d97033f0、README、figures/DUET_unified_seq_len_96.png、论文检索摘要、EXP-0003。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0001
analysis_id: PAPER-0001
created_at: 2026-10-23
source_ids: [arXiv:2412.10859, github:decisionintelligence/DUET@dcc6e6780a9138731b64b9b5398a94a1d97033f0, EXP-0003]
evidence_status: partial
evidence_locations: [ts_benchmark/baselines/duet/models/duet_model.py, ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py L255-290, ts_benchmark/baselines/duet/layers/distributional_router_encoder.py, ts_benchmark/baselines/duet/utils/masked_attention.py L141-207, ts_benchmark/baselines/deep_forecasting_model_base.py L357-435 L600-700, ts_benchmark/evaluation/strategy/rolling_forecast.py L183-223 L342-440, config/rolling_forecast_config.json, EXP-0003]
missing_evidence:
- 论文全文未读；主结果“多个回看窗口中报告最好结果”的选择依据未知
- 本项目的 ETTh1 由 ETDataset 原始 CSV 转换而来，未与 TFB 官方预处理文件逐值比对
next_skill: deep-read Mode C
source_identity_and_version: "Qiu, Wu, Lin, Guo, Hu, Yang, KDD 2025, pp.1185-1196, arXiv 2412.10859；代码 MIT（Huawei 版权声明），基于 TFB 框架，dcc6e678 (2026-05-12)"
research_question: 如何同时处理时间维度上的分布异质性和变量之间的相关性
study_type_and_context: 多变量长期预测；25 个数据集、10 个领域；TFB 统一评测框架
design_units_samples_or_materials: TFB 数据；ETT 采用 0.6/0.2/0.2 划分（ETTh1 长度 14400 时等同 12/4/4 个月）
intervention_input_or_procedure: 时间聚类模块（按输入路由到若干线性专家的混合专家）+ 通道聚类模块（频域可学习距离 → 采样稀疏通道 mask → 通道间注意力）
measurements_and_analysis: TFB 的 mse_norm、mae_norm 等（在训练集标准化尺度上计算）
results_with_uncertainty: 单次运行；无方差
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 测试时通道 mask 随机；依赖栈重
reproducibility_and_transferability: 评测框架公平（训练集拟合 scaler、不丢测试窗口）；ETTh1-96 两个种子 MSE 0.3898、0.3790，报告值 0.377 在种子波动范围内（E-DU-C10）
candidate_P: [P-TS1 评测协议差异, P-TS5 推理时的随机性, P-TS6 路由依据]
candidate_F: [F-TS5 测试时 mask 随机采样, F-TS6 路由在未做实例归一化的输入上计算]
evidence_index: E-DU-C01..E-DU-C12
unresolved_questions: [更多种子下的均值与方差, 回看窗口的选择依据]
```

## 来源

- 论文：Xiangfei Qiu 等，“DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting”，KDD 2025，arXiv 2412.10859。作者团队同时维护 TFB 基准（VLDB 2024）。
- 代码：<https://github.com/decisionintelligence/DUET>，MIT（版权声明为 Huawei Technologies），固定 commit dcc6e678（2026-05-12）。仓库是 TFB 框架加上 DUET 与 20 多个对比方法。
- 实际阅读范围：README；DUET 模型、时间聚类、路由器、线性专家、Mahalanobis mask 全部源码；TFB 的训练基类、滚动评测策略、数据读取与划分；ETTh1 的两套脚本。

## 研究事实

- **方法（代码实际做法）**：
  1. **时间聚类模块（TCM）**：每个变量单独处理（CI=1）。路由器是两层 MLP，输入为原始窗口（只经过全局标准化、未做实例归一化）；用带噪声的 top-k 门控（Shazeer et al. 2017）在 num_experts 个专家中选 k 个。专家是 DLinear 式的“移动平均分解 + 两个线性层”，把长度 seq_len 映射到 d_model；只有专家的输入做 RevIN。门控的负载均衡损失加到训练损失上。
  2. **通道聚类模块（CCM）**：对各变量的 rFFT 幅值，用可学习矩阵 A 计算变量两两之间的距离，转成“相关概率”，再用 Gumbel-softmax（hard=True）采样出 0/1 通道 mask，作为通道间 Transformer 的注意力 mask。
  3. 线性头输出预测，再用时间聚类模块的 RevIN 反归一化。
- **两套结果**：README 主结果对 {96, 336, 512} 三个回看窗口“报告最好结果”；另外单独发布了回看固定为 96 的统一超参数结果。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-DU-C01 | SC | 时间聚类处理分布异质性，通道软聚类在频域用度量学习刻画变量关系并稀疏化以抑制噪声通道；25 个数据集上最好 | 论文摘要 | — | 第三方摘要称平均 MSE 降 7.1%，未回查 |
| E-DU-C02 | SC | 回看固定 96 的统一结果：ETTh1-96 MSE 0.377 / MAE 0.393，四个长度平均 0.443 / 0.436 | figures/DUET_unified_seq_len_96.png | 单次 | — |
| E-DU-C03 | SC | 主结果在 {96, 336, 512} 中“报告最好结果”；README 称遵循 TFB、不使用 drop last | README Results 节 | — | 若按测试集选回看窗口，会偏乐观；选择依据待回查 |
| E-DU-C04 | DE | 路由器在未做实例归一化的输入上计算门控，RevIN 只作用于专家输入 | linear_extractor_cluster.py L267-284；distributional_router_encoder.py | CI=1 | 可能是有意让路由利用窗口的水平和尺度；作者意图未知 |
| E-DU-C05 | DE | 专家为 DLinear 式线性层（输出维度 d_model）；噪声 top-k 门控 + 负载均衡损失 | linear_pattern_extractor.py；linear_extractor_cluster.py | — | — |
| E-DU-C06 | DE | 通道 mask 在 **eval 模式下也是随机采样**（`gumbel_softmax(hard=True)` 无训练/推理区分）。随机权重探针：同一输入两次前向输出最大差 0.186，196 个 mask 元素中 36 个不同；只有固定 torch 种子才能复现 | masked_attention.py L182-207；EXP-0003 probe_duet_mask.py | 随机权重 | 训练后对分数的影响见 E-DU-C10 |
| E-DU-C07 | DE | TFB 评测：标准化器只在训练段拟合；测试按步长 1 滚动、所有窗口计分（不丢样本）；指标在训练集标准化尺度上计算；ETTh1 按元信息长度 14400 划分为 8640/2880/2880 | deep_forecasting_model_base.py L400；rolling_forecast.py L183-223、L388-420；config/rolling_forecast_config.json | ETT | — |
| E-DU-C08 | DE | 数据需从 Google Drive / 百度网盘下载 TFB 预处理版；本项目用 ETDataset 原始 CSV 截取前 14400 行转换（code/ts_forecasting/to_tfb_format.py） | README；EXP-0003 | — | 与官方预处理文件是否逐值一致未核对 |
| E-DU-C09 | DE | 依赖栈重：requirements 含 darts、salesforce-merlion、ray、dash、lightgbm 等；官方只在 Python 3.8 下测试；ETTh1 按预测长度使用不同的回看窗口与超参数（ETTh1_script/DUET.sh 中回看为 512/336） | requirements.txt；scripts/multivariate_forecast/ETTh1_script/DUET.sh | — | — |
| E-DU-C10 | DE | 本项目 CPU 复现（统一设置 ETTh1-96）：种子 2021 得 mse_norm 0.3898 / mae_norm 0.3990（TFB 自身报告与本项目计算一致），种子 2022 得 0.3790 / 0.3960；报告值 0.377 / 0.393。两个种子均值 0.3844，相差 0.011，大于 TQNet 四个种子的极差 0.0068 | EXP-0003 R12、R14；TFB test_report | ETTh1-96，CPU fp32，torch 2.1.2 | 能在种子波动范围内复现；但单种子结果不稳定 |
| E-DU-C12 | DE | 训练好的模型在测试时只重新采样通道 mask（10 个种子）：MSE 0.3894～0.3905，标准差 0.0003；即测试随机性真实存在，但对分数的影响很小 | EXP-0003 R12 | ETTh1-96 | 其他数据集、通道更多时可能更大 |
| E-DU-C11 | SC | README 致谢中说明曾有用户发现代码 bug，已修复并重新测试所有结果 | README Acknowledgement | — | 修复前后差异未知 |

## 研究判断

- **优点**：
  - 评测框架（TFB）公平：训练集拟合 scaler、不丢测试窗口、统一的 rolling 评测，便于和 20 多个对比方法在同一协议下比较。
  - 模型由两个可以分开消融的模块组成，研究空间大。
  - KDD 2025，引用量高。
- **缺点**：
  - 测试时通道 mask 随机；在 ETTh1-96 上实测影响很小（MSE 标准差 0.0003），但报告时仍应注明。
  - 种子敏感：ETTh1-96 两个种子的 MSE 相差 0.011（0.3898 vs 0.3790），报告值 0.377 接近较好的那个种子。
  - 路由器用未做实例归一化的输入，“按分布聚类”可能部分是“按水平/尺度聚类”（推断）。
  - 依赖栈重、训练慢（CPU 上每个 epoch 数分钟）。
  - 主结果的回看窗口选择依据不明。
- **与自身实验的关系**：DEC-0012 起改为**强对照与组合实验素材**（主 baseline 改为 iTransformer）。此前曾是时序预测方向的主 baseline（DEC-0011，第 ② 步复现门槛已在 ETTh1-96 上初步通过），并用它的 TFB 框架做统一评测；详见 [时序三篇对比](../专题分析/时序预测_DUET_TQNet_WPMixer对比.md)。
- **核心总结**：DUET 的价值在“双聚类”的问题设定和 TFB 的公平评测。本项目发现的测试随机性经实测影响很小，不值得作为改进方向；路由输入（E-DU-C04）仍待回查论文。

## 交接 Deep Read

候选 P-TS1 ← E-DU-C03、E-DU-C07；P-TS5（推理随机性）← E-DU-C06；P-TS6（路由依据）← E-DU-C04。F-TS5 ← E-DU-C06、E-DU-C10；F-TS6 ← E-DU-C04。
