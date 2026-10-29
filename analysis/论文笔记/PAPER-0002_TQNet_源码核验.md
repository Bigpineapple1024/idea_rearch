# ANALYSIS_PACKAGE：TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting

> 本文件是源码核验补充；原文笔记见 [PAPER-0002_TQNet](PAPER-0002_TQNet.md)。两份冲突时以能追溯到原文页码或源码行号的一方为准。

状态：partial（代码已核验，ETTh1 已在 CPU 上真实复现；论文全文未读）。paper_id PAPER-0002 / 2026-10-23 / 分析者 Claude Code 会话 / 输入：代码 ACAT-SCUT/TQNet@15e19cb23483ed52398566c4baa959168cfffa57、README、仓库内 result.txt 与 Figures/Table2.png、论文检索摘要、EXP-0003。
**论文已回查（2026-10-24）**：E-TQ-C01 已按论文 p.1、p.5～6、p.9、p.13 核对；状态中的“论文全文未读”和 missing_evidence 第 1 条的“论文全文未读（arxiv.org 被网络策略拦截）”都是初稿时的说法；超参数怎样选出，论文确实没有写（2026-10-29 完整阅读补注）。

## 结构化交接包

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: PAPER-0002
analysis_id: PAPER-0002
created_at: 2026-10-23
source_ids: [arXiv:2505.12917, github:ACAT-SCUT/TQNet@15e19cb23483ed52398566c4baa959168cfffa57, EXP-0003]
evidence_status: partial
evidence_locations: [models/TQNet.py, data_provider/data_factory.py L15-33, data_provider/data_loader.py L49-100, scripts/TQNet/etth1.sh, result.txt, Figures/Table2.png, EXP-0003]
missing_evidence:
- 论文全文未读（arxiv.org 被网络策略拦截），超参数如何选定未知
- 只复现了 ETTh1；其余 11 个数据集未运行
next_skill: deep-read Mode C
source_identity_and_version: "Lin et al., ICML 2025 (PMLR 267), arXiv 2505.12917；代码 Apache-2.0，15e19cb2 (2026-05-06)"
research_question: 如何在轻量模型中稳健地建模多变量之间的相关性
study_type_and_context: 多变量长期预测；12 个数据集；回看窗口统一为 96
design_units_samples_or_materials: ETT 四个、Electricity、Solar、Traffic、Weather、PEMS03/04/07/08
intervention_input_or_procedure: 用按周期移位的可学习向量作为注意力的 query，输入数据作为 key/value，做一层跨变量注意力，再接两层 MLP
measurements_and_analysis: MSE、MAE；单种子 2024
results_with_uncertainty: 论文与 result.txt 为单种子；本项目 4 个种子在 ETTh1-96 上 MSE 标准差约 0.003（E-TQ-C07）（CPU；GPU 三个种子在 96 档的标准差为 0.0002，见 E-TQ-C08 的 2026-10-29 补注）
claim_evidence_implementation_map: 见 E 表
limitations_biases_and_ethics: 需要人为指定周期长度 cycle；README 自述难以扩展到超长回看窗口
reproducibility_and_transferability: 高：ETTh1 四个预测长度与官方 result.txt 的 MSE 差 ≤0.0025，CPU 每次约 1 分钟（CPU、种子 2024；GPU 三种子均值与 result.txt 的差见「研究判断」中复现误差一条的 2026-10-29 补注）
candidate_P: [P-TS1 评测协议差异导致结论不可比, P-TS3 周期先验需要人工指定]
candidate_F: [F-TS1 TQNet 可在 CPU 上精确复现, F-TS4 种子波动与方法间差距同量级]
evidence_index: E-TQ-C01..E-TQ-C08
unresolved_questions: [超参数（lr、dropout、d_model）如何选定, cycle 选错时性能如何变化]
```

## 来源

- 论文：Shengsheng Lin 等，“Temporal Query Network for Efficient Multivariate Time Series Forecasting”，ICML 2025，arXiv 2505.12917。同一团队的 SparseTSF（ICML 2024）、CycleNet（NeurIPS 2024）的后续工作。
- 代码：<https://github.com/ACAT-SCUT/TQNet>，Apache-2.0，固定 commit 15e19cb2（2026-05-06）。
- 实际阅读范围：README；`models/TQNet.py` 全文；`data_provider/` 的划分与 DataLoader；`scripts/TQNet/etth1.sh`；`result.txt`；`Figures/Table2.png`。

## 研究事实

- **核心问题**：作者的前作 CycleNet 用可学习的周期向量建模单变量周期性，但不能建模变量之间的关系。
- **方法（代码实际做法）**：
  1. 实例归一化（不带可学习仿射参数的 RevIN）。
  2. **Temporal Query**：参数 `temporalQuery` 形状为（周期长度 × 变量数），按每个时间步在周期中的位置取出，得到每个变量一条长度为 seq_len 的 query。
  3. 一层 `nn.MultiheadAttention`：以变量为 token（每个 token 维度 = seq_len），query 来自 TQ，key/value 来自输入，输出加回输入。
  4. Linear(seq_len→d_model) → 两层 GELU MLP（残差）→ Dropout → Linear(d_model→pred_len) → 反归一化。

| E ID | 类型 | 内容 | 定位 | 支持范围 | 未知 / 替代解释 |
| --- | --- | --- | --- | --- | --- |
| E-TQ-C01 | SC（已回查，更正，2026-10-24） | 周期移位的可学习向量作 query，同时利用全局先验和样本局部信息。12 个数据集（ETT×4、Electricity、Solar、Traffic、Weather、PEMS03/04/07/08），回看 96，四个预测长度取平均（Table 2）。论文的说法是 24 个指标中 22 个进入前二（Count 行：TQNet 22、TimeXer 11、CycleNet 9、iTransformer 4、SCINet 2），不是“全部最好”；按加粗计 TQNet 最好 16/24，ETTm2 的两项都不在前二。效率证据只有 Fig.7（图片）：Electricity（H=720）上的训练时间（秒/epoch）与参数量，以及 Traffic 上训练时间随变量数的变化，作者称接近 DLinear；没有推理延迟或 FLOPs，实验用单张 RTX 4090。超参数是固定值（附录 A.2），没有说明怎样选出 | 论文 p.1 摘要、p.5 §4.1～4.2、p.6 Table 2、p.9 Fig.7、p.13 A.2 | 回看 96 | “效率接近线性模型”只有训练时间与参数量的证据 |
| E-TQ-C02 | SC | 论文 Table 2（回看 96）ETTh1 四个预测长度平均 MSE 0.441、MAE 0.434；result.txt 中 ETTh1-96 为 0.3712 / 0.3928 | Figures/Table2.png；result.txt | 单种子 2024 | — |
| E-TQ-C03 | DE | 结构如上：TQ 参数初始化为 0；注意力 embed_dim = seq_len、4 个头（因此 seq_len 必须能被 4 整除）、注意力 dropout 固定 0.5 | models/TQNet.py | — | — |
| E-TQ-C04 | DE | `cycle_index` 按全数据中的绝对行号对周期取模，训练/验证/测试相位一致；周期长度需要逐数据集手动指定（ETTh1 为 24） | data_loader.py L84、L97；scripts/TQNet/etth1.sh | — | 周期未知或多周期的数据需要额外处理 |
| E-TQ-C05 | DE | 测试集 `drop_last=False`（所有测试窗口都计分）；所有主实验脚本回看固定 96；ETT 采用 12/4/4 个月的标准划分 | data_factory.py L19-22；scripts/TQNet/*.sh；data_loader.py L49-50 | — | 验证集 drop_last=True，只影响早停 |
| E-TQ-C06 | DE | `run.py` 每次运行都会把结果追加写入仓库根目录的 result.txt | run.py | — | 复现时应先备份原始 result.txt |
| E-TQ-C07 | DE | 本项目在 CPU 上复现 ETTh1（种子 2024）：96/192/336/720 的 MSE 为 0.3703/0.4282/0.4754/0.4849，官方为 0.3712/0.4284/0.4757/0.4874；ETTh1-96 用 4 个种子（2021–2024）得到 0.3722/0.3718/0.3771/0.3703，均值 0.3729，标准差 0.0030 | EXP-0003 | ETTh1，CPU fp32 | 只测了一个数据集 |
| E-TQ-C08 | AI | 4 个种子的 MSE 极差 0.0068，与论文中相邻方法在 ETTh1 上的差距同一量级（例如 TQNet 0.441 vs TimeXer 0.437），单种子结论不可靠（2026-10-29 完整阅读补注：这是 CPU 结果。GPU 三种子复验（EXP-0005）中 96 档标准差只有 0.0002，720 档为 0.0206（主要来自种子 2022 的 0.5255）；种子敏感性随预测长度变化，见现象台账 F-TS4。） | E-TQ-C07、E-TQ-C02 | ETTh1 | 其他数据集的波动可能更小 |

## 研究判断

- **优点**：
  - 结构极简（单层注意力 + MLP），代码约 80 行，便于阅读和消融。
  - 评测协议干净：测试集不丢样本、回看窗口固定、划分标准。
  - 本项目复现误差小于 0.003。（2026-10-29 完整阅读补注：指 CPU、种子 2024 的结果；GPU 三种子均值为 0.3728 / 0.4295 / 0.4796 / 0.5019，result.txt 为 0.3712 / 0.4284 / 0.4757 / 0.4874，720 档差得最多，见 EXP-0005。）
  - CPU 上每次约 1 分钟，单卡 GPU 上可以负担多种子、多数据集的实验。
- **缺点**：
  - 周期长度需要人为指定，多周期或周期不明显的数据怎么办，代码中没有处理。
  - 注意力维度等于 seq_len，回看窗口变化时注意力规模随之变化；README 自述难以扩展到超长回看。
  - 论文和脚本只用一个种子。
- **与自身实验的关系**：作为对照方法和组合实验素材（DEC-0012 起主 baseline 为 iTransformer，DUET 也改为对照）；因为训练快、复现精确，也适合在主 baseline 上做正式实验前先做快速诊断（见 [时序三篇对比](../专题分析/时序预测_DUET_TQNet_WPMixer对比.md)）。
- **核心总结**：TQNet 的贡献是把“周期先验”从单变量（CycleNet）推广到变量间相关性；它可复现、协议干净、实验便宜，这三点对一年期的硕士课题最重要。

## 交接 Deep Read

候选 P-TS1（评测协议差异）← E-TQ-C05；P-TS3（周期先验需人工指定）← E-TQ-C04。F-TS1（可精确复现）← E-TQ-C07；F-TS4（种子波动与方法间差距同量级）← E-TQ-C08。

（2026-10-25 跑偏检查补注）P-TS1、P-TS3 和 F-TS1 只是本分析包当时给的候选标签，[MODE_C-0003](../MODE_C-0003_时序预测.md) 和[现象台账](../已发现的实验现象.md)都没有定义它们；这里不改号。MODE_C-0003 实际用的问题编号是 P1～P9，现象编号是 F-TS2、F-TS4、F-TS5、F-TS7～F-TS15 和 F-L1（F-TS7 在 2026-10-25 跑偏检查时从台账补登）。与本文件有关的是 F-TS4（TQNet 的种子波动，编号与这里一致，内容已按 GPU 复验更新）、P2（TQNet-720 上验证损失分不出好坏种子）和 P6（种子波动与方法间差距同量级）。
