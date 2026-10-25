# ANALYSIS_PACKAGE：iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

> 本文件是源码核验补充；原文笔记见 [PAPER-0007_iTransformer](PAPER-0007_iTransformer.md)。两份冲突时以能追溯到原文页码或源码行号的一方为准。

状态：partial（代码已核验；GPU 复现 EXP-0004 已完成：16 个组合中 6 个落在 ±2 SD 内，与论文的差在 −0.0080～+0.0090 之间（SQZ-EXP-0004），记为偏差、继续使用（DEC-0018）。复现结果见 [EXP-0004_record](../实验报告/EXP-0004_record.md) 与 [SQZ-EXP-0004](../实验报告/EXP-0004_squeeze.md)，没有另补 E-IT-C11 起的条目；下文结构化交接包与“交接 Deep Read”中“待 EXP-0004”的写法是当时的状态，保留不改（2026-10-25 跑偏检查补注）。原写：GPU 复现见 EXP-0004，完成后补入 E-IT-C11 起的条目）。paper_id PAPER-0007 / 2026-10-23 / 分析者 Claude Code 实验室会话 / 输入：代码 thuml/iTransformer@c2426e68ca13f74aaec08045c5c724d8ad328124、官方 ETT 脚本、arXiv:2310.06625v4。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0007
analysis_id: PAPER-0007
created_at: 2026-10-23
source_ids: [arXiv:2310.06625v4, github:thuml/iTransformer@c2426e68ca13f74aaec08045c5c724d8ad328124]
evidence_status: partial
evidence_locations: [model/iTransformer.py, layers/Embed.py L127-140, data_provider/data_loader.py L47-60/L135-148, data_provider/data_factory.py L18-34, run.py, utils/tools.py L13-14, utils/timefeatures.py L106, experiments/exp_long_term_forecasting.py, scripts/multivariate_forecasting/ETT/*.sh]
missing_evidence:
- 本项目 GPU 复现（EXP-0004）尚未完成
- 作者的超参数搜索脚本不在仓库中
next_skill: deep-read Mode C
source_identity_and_version: "Liu et al., ICLR 2024, arXiv 2310.06625v4；代码 MIT，c2426e68 (2025-07-17)"
research_question: 把每个变量的整段序列作为 token、在变量维做注意力，能否改进 Transformer 预测器
study_type_and_context: 多变量长期预测；回看 96；ETT 等 7 类数据集
design_units_samples_or_materials: ETTh1/ETTh2/ETTm1/ETTm2（12/4/4 个月划分）等
intervention_input_or_procedure: 逐变量 Linear 嵌入（另拼接日历特征 token）→ L 层注意力 + FFN → Linear 投影；实例归一化包在外层
measurements_and_analysis: 标准化尺度上的 MSE、MAE，全部测试窗口；种子写死为 2023
results_with_uncertainty: 论文 Table 5 有 5 个种子的标准差（不含 ETTh1）；本项目结果待 EXP-0004
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 日历特征 token 未在论文中描述；部分官方脚本的 D 不在论文所写的网格内
reproducibility_and_transferability: 待 EXP-0004；代码与数据完整，种子需包装
candidate_P: [P-IT-2 日历特征 token 的贡献]
candidate_F: [F-IT-C1 部分 ETT 脚本的超参数不在论文网格内]
evidence_index: E-IT-C01..E-IT-C10
unresolved_questions: [GPU 非确定性对同种子重复运行的影响, 去掉日历 token 后 ETT 上的变化]
```

## 来源

- 论文：Yong Liu 等，“iTransformer: Inverted Transformers Are Effective for Time Series Forecasting”，ICLR 2024，arXiv 2310.06625v4。
- 代码：<https://github.com/thuml/iTransformer>，MIT，固定 commit c2426e68（2025-07-17）。
- 实际阅读范围：`model/iTransformer.py` 全文；`layers/Embed.py` 的 `DataEmbedding_inverted`；`data_provider/` 的划分、归一化与 DataLoader；`run.py` 的参数与种子；`experiments/exp_long_term_forecasting.py` 的训练、早停与测试；`utils/tools.py`、`utils/timefeatures.py`；四个 ETT 官方脚本。

## 研究事实

- **代码实际做法**：
  1. 实例归一化：每个变量减去回看窗口均值、除以标准差，预测后还原。
  2. 嵌入：`Linear(seq_len→d_model)`；把日历特征（时、星期、日、年中日）当作额外 token 拼接在变量 token 之后。
  3. L 层编码器：变量 token 之间全注意力 + FFN。
  4. `Linear(d_model→pred_len)` 投影，丢掉日历 token 的输出。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-IT-C01 | DE | 实例归一化默认开启（`use_norm=True`）：均值与总体方差按回看窗口逐变量计算（+1e-5），预测后乘回标准差、加回均值 | model/iTransformer.py L20、L43-48、L66-69；run.py L86 | 与论文 Fig.4 的 x̃=(x−µ)/σ 一致 | — |
| E-IT-C02 | DE | **日历特征作为 token**：`DataEmbedding_inverted` 把 `x_mark`（freq='h' 时 4 个特征）转置后与变量拼接，同一个 Linear 嵌入；注意力因此在 N+4 个 token 上进行（ETT 为 7+4=11），投影后用 `[:, :, :N]` 丢掉日历 token | layers/Embed.py L136-140；model/iTransformer.py L57、L64；utils/timefeatures.py L106 | 论文正文、算法 1 与附录都没有提到这一点 | 对 ETT 这类低维数据，日历 token 占全部 token 的 36%，贡献未测 |
| E-IT-C03 | DE | 划分：ETTh 训练/验证/测试为 12/4/4 个月（8640/2880/2880 点，验证和测试起点各前移 seq_len）；ETTm 为其 4 倍；`StandardScaler` 只在训练段上拟合 | data_provider/data_loader.py L47-48、L60、L135-136、L148 | 无泄漏 | 论文 Table 4 的 (8545, 2881, 2881) 与代码的点数口径不同；测试窗口数按代码为 2880−pred_len+1（如 96 档 2785 个） |
| E-IT-C04 | DE | 测试 batch 1、shuffle=False、drop_last=True：batch 1 时不丢任何窗口；指标在标准化尺度上计算（`--inverse` 默认关闭） | data_provider/data_factory.py L20-23；run.py L81；experiments/exp_long_term_forecasting.py L233-236 | 与 TSLib 惯例一致 | — |
| E-IT-C05 | DE | 训练：MSELoss；默认 lr 1e-4、batch 32、10 个 epoch、早停 patience 3，早停只看验证损失；学习率每个 epoch 减半（type1）；每个 epoch 也计算并打印测试损失，但不用于选择 | exp_long_term_forecasting.py L36、L167-177；run.py L62-68；utils/tools.py L13-14 | 选择过程不看测试集 | 论文说 lr 取自网格，脚本都用默认 1e-4 |
| E-IT-C06 | DE | 种子写死：`fix_seed = 2023` 后设置 random/torch/numpy 种子；没有设置 cuDNN 确定性；DataLoader 默认 10 个 worker | run.py L9-12、L60 | 多种子复现需要包装脚本（code/ts_forecasting/itransformer_seed.py，在内存中只改这一行） | GPU 上同种子重复运行可能不完全一致 |
| E-IT-C07 | DE | ETT 官方脚本：e_layers=2；ETTh1 的 d_model=d_ff 在 96/192 档为 256、336/720 档为 512；其他参数用 run.py 默认值 | scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh | — | — |
| E-IT-C08 | DE | ETTh2、ETTm1、ETTm2 的脚本 d_model=d_ff=128，不在论文 A.2 所写的 D∈{256, 512} 内 | 同目录 iTransformer_ETTh2/ETTm1/ETTm2.sh；PDF p.14 A.2 | 以脚本为准复现 | 可能是论文描述的网格不完整 |
| E-IT-C09 | DE | ETTm 脚本没有设置 `--freq`，按默认 'h' 生成 4 个日历特征（不含分钟），尽管数据是 15 分钟粒度 | run.py L29；utils/timefeatures.py L106-113 | 与 ETTh 相同的日历 token | — |
| E-IT-C10 | AI | 论文 Table 5 的 5 个种子在公开代码中无法直接复现（种子写死），ETTh1 也不在 Table 5 中；因此 ETTh1 的复现判据只能用本项目 3 个种子的标准差 | E-IT-6、E-IT-C06 | — | — |

## 研究判断

- **优点**：
  - 结构简单，核心约 70 行，组件可以逐个替换或去掉（论文消融本身就是这样做的）。
  - 评测协议干净：标准划分，训练段拟合归一化，测试集全部窗口计分，早停只看验证集。
  - 依赖少（requirements 只有 6 个包），复现成本低。
- **缺点**：
  - 日历特征 token 是论文没有描述的输入通道（E-IT-C02）；对低维数据影响可能不小。
  - 部分脚本超参数与论文网格不符（E-IT-C08）；ETTm 的日历特征粒度与数据不匹配（E-IT-C09）。
  - 种子写死，GPU 上没有确定性设置。
- **与自身实验的关系**：时序方向的主 baseline（DEC-0012）；复现按 EXP-0004 进行。日历 token 与变量注意力在 ETT 上的作用是后续组件级诊断的自然切入点（P-IT-1、P-IT-2）。
- **核心总结**：iTransformer 的贡献是换了 token 的维度而不是换组件；代码干净、便于诊断，但在 ETT 这类低维数据上，它的核心机制（变量间注意力）发挥空间有限，代码里还多了一路论文没写的日历输入。

## 交接 Deep Read

候选 P-IT-2（日历 token 的贡献）← E-IT-C02、E-IT-C09。F-IT-C1（脚本超参数不在论文网格内）← E-IT-C08。复现结论待 EXP-0004 补入。
