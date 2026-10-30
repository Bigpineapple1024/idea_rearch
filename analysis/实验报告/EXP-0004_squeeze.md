# SQUEEZE_REPORT：EXP-0004（iTransformer 在 ETT 上的 GPU 复现，阶段 A + B）

深挖 ID：SQZ-EXP-0004 / 输入实验记录：[EXP-0004](EXP-0004_record.md) / 日期：2026-10-24 / 执行者：Claude Code（实验室电脑，分支 code/ts-forecasting）。

## 0. 交接元数据

- 包类型：SQUEEZE_REPORT（research-pipeline-contract v1.0）。
- 来源 IDs：EXP-0004-R01～R56；SQZ-EXP-0009 共用其检查点。
- 证据状态：verified。56 次运行都有 SHA256SUMS 并已备份。分析目录已封存：
  - analysis：6a2d3d5b…；
  - deepdive：d80a47a8…。
- 缺失证据：
  - 论文作者的超参数搜索脚本与最终配置（仓库中没有）；
  - 论文的运行环境；
  - 本机以外的 GPU 上是否同样确定。

## 1. 事实与证据边界

- **研究问题**：官方 iTransformer（thuml/iTransformer@c2426e68）在实验室电脑上能否复现论文 Table 10 的 ETT 结果（回看 96）。
- **运行规模**：
  - 阶段 A：ETTh1，20 次，含直接运行 run.py 与包装脚本，commit b140cf9。
  - 阶段 B：ETTh2、ETTm1、ETTm2 × 4 个预测长度 × 3 个种子，36 次，commit e40bd90。
  - 所有运行都没有未提交的改动。
- **数据**：ETDataset@1d16c8f4 的 CSV，sha256 记录在每次运行的 config.json 里。
- **已知偏差**：复现判定偏离研究流程第 ② 步，由用户决定（DEC-0018）。

## 2. 主要结果

| 结果 | 对照 | 差异 | 不确定性 | 实际意义 | 证据位置 |
| --- | --- | --- | --- | --- | --- |
| 16 个“数据集 × 预测长度”中 6 个落在 ±2 SD 内 | 论文 Table 10 | 均值 − 论文：−0.0080～+0.0090，相对差不超过 2.8% | 种子标准差 0.0003～0.0047 | 偏差小于 2025 年方法与 iTransformer 之间的差距（约 0.01，论文报告值；2026-10-30 完整阅读补注：没有写出处；最大偏差 0.0090 与它同量级，“小于”只是勉强成立） | analysis/tables.md |
| 包装脚本（种子 2023）与两次直接运行逐位相同 | 直接运行 run.py | 0 | — | 这台 GPU 上训练是确定的，种子是唯一的随机来源 | 阶段 A 表 |

## 3. 深挖发现

| 维度 | 发现 | 证据 | 稳健性 | 新信息 / 已知 |
| --- | --- | --- | --- | --- |
| 数据质量 | 各运行的数据 sha256 一致；测试窗口数与 batch 1、不丢窗口的口径一致（2785 / 2689 / 2545 / 2161；ETTm 为 11425 / 11329 / 11185 / 10801） | config.json、metrics | 高 | 已知口径 |
| 过程 / 动态 | ETTh1 上与论文的差随预测长度增大（+0.0002 → +0.0090）；阶段 B 没有一致的趋势，12 个组合中 3 个低于论文（2026-10-29 完整阅读补注：ETTh1-96 的均值 0.3858 低于论文 0.386，差应为 −0.0002（原写 +0.0002），即 −0.0002 → +0.0090；算上它，16 个组合中有 4 个均值低于论文，不只是阶段 B 的 3 个；数据见 code/output/EXP-0004/analysis/tables.md） | tables.md | 中 | 新：偏差有正有负 |
| 机制 / 中间量 | 测试期水平部分占 MSE 的比例因数据集差别很大：ETTh1 16%～20%，ETTm1 22%～32%，ETTh2 31%～45%，ETTm2 48%～49%。预测水平变化对真实变化的斜率在 −0.01～0.29，相关在 −0.03～0.47，都随预测长度下降 | deepdive/level_shape_*.md；ETTh1 见 EXP-0005 deepdive | 高（3 个种子合并） | 新：水平部分在 ETTh1 以外占得更多 |
| 错误 / 异质性 | 测试期上“水平不变”（a=0）比模型原样好的，只有 ETTh1 的 192 / 336 / 720 档和 ETTm1-720 | EXP-0009 analysis/rules.md | 中 | 修正 F-TS10 的范围 |
| 边界 / 敏感性 | ±2 SD 判据只反映种子差异；不达标的 10 个组合，与论文的差是种子标准差的 2.1～14 倍 | tables.md | 高 | 新 |

## 4. 解释与不确定性

- **最支持的解释（AI）**：与论文的差主要来自配置与环境，而不是实现错误。
  - 官方脚本的 d_model = 128 不在论文网格 {256, 512} 内，学习率固定为 1e-4（E-IT-C05、C08），论文报告值可能来自脚本之外的配置。
  - 软件与硬件环境不同。
  - 偏差有正有负，也不支持“实现错误”。
- **仍可成立的替代解释**：
  - 论文使用的数据副本不同（例如 Autoformer 仓库的副本），本项目没有逐值比对。
  - 论文报告的是单种子 2023 的结果，本项目 2023 种子的结果同样与论文不同。
- **不支持的强结论**：“iTransformer 不可复现”；“本项目的环境更好或更差”。

## 5. 知识沉淀

- **新现象 F-TS12**：用官方脚本在本机复现 iTransformer，16 个组合与论文的差都在 0.0090 以内（2026-10-25 跑偏检查补注：原写“0.0081 以内”，只对阶段 B 的 12 个组合成立；ETTh1-720 为 +0.0090，code/output/EXP-0004/analysis/tables.md），有正有负。按“均值 ± 2 SD”的严格判据只有 6 个达标，原因是种子标准差远小于跨环境的差异。
- **新限制**：本项目所有时序结论都以本项目自己的 iTransformer 运行为基准（DEC-0018）。
- **已回答**：iTransformer 可以在本机稳定运行，结果确定；阶段 A、B 的判定见记录。
- **需要复验**：在另一台 GPU 上复跑一个组合，检查确定性是否跨机器成立（不急）。
- **更新位置**：已发现的实验现象（F-TS12，F-TS10 的范围）；MODE_C-0003 回流。

## 6. 下一步决策

1. 复现部分不再追加运行（DEC-0018）。
2. 水平部分在 ETTh1 以外占比更高，模型几乎预测不出水平变化（SQZ-EXP-0009 的 F-TS13）。这一点进入第 ⑦ 步的排序。

## 7. 回流到 deep-read Mode C 的输入包

```text
package_type: SQUEEZE_REPORT
squeeze_id: SQZ-EXP-0004
input_experiment_record_ids: [EXP-0004]
claim_scope: iTransformer（官方脚本，回看 96）在本机 ETT 四个数据集上的复现与测试期误差结构
data_quality_and_provenance_gaps: 论文作者的最终超参数未知；论文数据副本未逐值比对
primary_result_with_uncertainty: 16 个组合中 6 个落在 ±2 SD 内；与论文的差在 −0.0080～+0.0090 之间（相对差不超过 2.8%）；种子标准差 0.0003～0.0047
process_dynamics: ETTh1 的差随预测长度增大；其他数据集没有一致趋势
mechanism_and_intermediate_findings: 测试期水平占比 16%～49%，因数据集而异；预测水平变化的斜率不超过 0.29，随预测长度下降
errors_heterogeneity_and_boundary_conditions: 测试期“不变”优于模型只在 ETTh1 的 192 / 336 / 720 档和 ETTm1-720
robustness_sensitivity_and_alternative_explanations: 训练确定（包装脚本与直接运行逐位相同）；偏差来源未拆开（配置 vs 环境 vs 数据副本）
new_P: [P7：用官方脚本无法得到论文的配置，基线比较必须以本项目运行为准]
new_F: [F-TS12]
new_M: [官方脚本与论文配置不一致造成系统性偏差（AI）]
evidence_index: [code/output/EXP-0004/analysis/tables.md, code/output/EXP-0004/deepdive/level_shape_*.md, EXP-0004_record.md]
open_questions: [偏差中配置、环境、数据副本各占多少]
next_routes: [deep-read Mode C 回流, 第 ⑦ 步排序]
next_skill: deep-read Mode C
```
