# SQUEEZE_REPORT：三个时序对照在统一协议下的深挖（EXP-0005）

状态：完成（2026-10-24）。
报告 ID SQZ-EXP-0005 / 实验 ID EXP-0005 / 输入记录：analysis/实验报告/EXP-0005_record.md（2026-10-24 版，commit cd5d3c0）/ 日期 2026-10-24。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0005
squeeze_id: SQZ-EXP-0005
created_at: 2026-10-24
source_ids: [EXP-0005, EXP-0004, EXP-0003, SQZ-EXP-0003, PAPER-0001, PAPER-0002, PAPER-0003, PAPER-0007]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0005_record.md
- code/output/EXP-0005/analysis/（tables.md、summary.json、compare_tfb_ett.json、windows_stats.md、windows_summary.json；SHA256SUMS 的 sha256 9440032b…）
- code/output/EXP-0005/<RUN>/stdout.log（训练过程）、pred*.npy / true*.npy（逐窗口预测）
missing_evidence:
- 只有 ETTh1；DUET 只有 96 档
- 测试期“难段”的数据特征（与训练期、验证期的分布差异）还没有量化
- WPMixer 与论文差距的来源（GPU/AMP、测试集调参的超参数）没有拆开
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0005, EXP-0004]
claim_scope: ETTh1，回看 96，四个预测长度，三个种子，全部测试窗口；iTransformer 取自 EXP-0004 阶段 A
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“主要结果”
process_dynamics: 见“深度分析”第 2 行
mechanism_and_intermediate_findings: 见“深度分析”第 3 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 5 行
new_P: [P-SQZ5-1, P-SQZ5-2]
new_F: [F-TS8, F-TS9]（另更新 F-TS2、F-TS4）
new_M: [M-SQZ5-1（AI）]
evidence_index: 见 mode_C_input_bundle.E
open_questions: [测试期中难段（96 档第 7 个十分位、长预测的最后一个十分位）的数据特征是什么, 验证集为什么预测不了长预测长度上的测试表现, 方法排序在其他 ETT 数据集上是否相同]
next_routes: [量化难段与训练期的分布差异（CPU 分析，用已保存的窗口）, EXP-0004 阶段 B 完成后在 ETTh2/ETTm1/ETTm2 上复查逐窗口相关, 建立时序方向的 Mode C]
mode_C_input_bundle:
  source_ids: [EXP-0005, SQZ-EXP-0005]
  P:
  - id: P-SQZ5-1
    text: "四种架构的误差集中在测试期的同一些时段（逐窗口相关约 0.97）：这些难段的数据特征是什么，是否与训练期的分布差异有关"
    evidence: [E-SQZ5-3, E-SQZ5-4]
    status: UN
  - id: P-SQZ5-2
    text: "验证集损失几乎相同的种子，测试 MSE 可以相差 0.037（TQNet-720）：按验证集早停和选模型，在长预测长度上是否可靠"
    evidence: [E-SQZ5-5]
    status: DE（现象）；原因 UN
  F:
  - id: F-TS2（更新）
    text: "测试期末尾的窗口对所有方法都更难：最后 97 个窗口与其余窗口的 MSE 之比在 iTransformer、TQNet、WPMixer、DUET 的 4 个预测长度上为 1.09～1.37（三种子平均），全部大于 1。EXP-0003 中只在 WPMixer 上由两种计分反推，这里直接测到，且不限于 WPMixer"
    evidence: [E-SQZ5-4]
    status: DE
  - id: F-TS4（更新）
    text: "GPU 上 TQNet 的种子波动随预测长度变化很大：96 档三个种子 0.3730 / 0.3727 / 0.3727（标准差 0.0002），720 档 0.4881 / 0.5255 / 0.4921（标准差 0.0206，主要来自种子 2022）"
    evidence: [E-SQZ5-1]
    status: DE
  - id: F-TS8（新）
    text: "不同架构在同一批窗口上出错：ETTh1-96 上 TQNet、WPMixer、DUET、iTransformer 的逐窗口 MSE 相关系数约 0.97～0.98（同一方法不同种子为 0.994）；按十分位的误差曲线四种方法几乎相同（96 档最难的是第 7 个十分位，约为第 2 个十分位的 2.8 倍；336、720 档最难的是最后一个十分位）；两种方法逐窗口比较时，较好的一方只在 54%～58% 的窗口上更好（2026-10-29 完整阅读补注：按 code/output/EXP-0005/analysis/windows_stats.md 的 ETTh1-96 部分，上文“同一方法不同种子为 0.994”只对 TQNet 成立（0.992～0.994）；iTransformer 为 0.991～0.992，WPMixer 为 0.977～0.987，DUET 为 0.969～0.973，与跨方法的 0.961～0.981 相当。“较好的一方只在 54%～58% 的窗口上更好”只对 TQNet（种子 2021）与 DUET、WPMixer 的比较，以及 DUET 对 iTransformer（54.6%～58.2%）成立；WPMixer 对 iTransformer 为 55.8%～61.8%，TQNet 对 iTransformer 为 64.9%～67.9%。EXP-0005 记录“方法之间逐窗口高度相关”一条写的范围是对的。）"
    evidence: [E-SQZ5-3]
    status: DE
  - id: F-TS9（新）
    text: "验证集预测不了长预测长度上的测试表现：TQNet-720 三个种子的最好验证损失为 1.5577 / 1.5595 / 1.5554（相差 0.004），测试 MSE 却为 0.4881 / 0.5255 / 0.4921（相差 0.037）；WPMixer-336 同样（验证 1.3054 / 1.2992 / 1.2998，测试 0.4674 / 0.4659 / 0.4848）。验证期的 MSE 约为测试期的 1.8 倍（96 档 0.665 vs 0.373），两段数据的难度不同"
    evidence: [E-SQZ5-5]
    status: DE
  M:
  - id: M-SQZ5-1
    text: "误差主要由数据中的少数非平稳时段决定，而不是由架构决定；各方法学到的是相似的平滑外推，遇到同样的变化时同样失败"
    evidence: [E-SQZ5-3, E-SQZ5-4]
    status: AI（推断；难段的数据特征未量化）
  E:
  - id: E-SQZ5-1
    text: "各方法 × 预测长度的三种子 MSE、均值 ± 标准差、MAE、WPMixer 的 drop_last 计分"
    location: code/output/EXP-0005/analysis/tables.md、summary.json
  - id: E-SQZ5-2
    text: "TFB 与 ETDataset 的 ETTh1 逐值相同（最大差 3.6e-15）；DUET 与 iTransformer 的 2785 个测试窗口逐值相同（最大差 2.2e-07）"
    location: code/output/EXP-0005/analysis/compare_tfb_ett.json
  - id: E-SQZ5-3
    text: "逐窗口 MSE 的相关系数与胜率（所有窗口数相同的运行对），按十分位的 MSE"
    location: code/output/EXP-0005/analysis/windows_stats.md、windows_summary.json
  - id: E-SQZ5-4
    text: "末 97 / 末 113 个窗口与其余窗口的 MSE 之比（每次运行）"
    location: windows_summary.json
  - id: E-SQZ5-5
    text: "各次运行的早停轮数与最好验证损失（TQNet 8～18 轮，WPMixer 6～10 轮）"
    location: code/output/EXP-0005/EXP-0005-R*/stdout.log
  evidence_locations: [code/output/EXP-0005/analysis/SHA256SUMS]
```

## 证据审计

- **完整性**：27 次计划的运行全部有结果（WPMixer 的 12 次在补装 thop 后用新 ID 重跑）；每次运行都保存了逐窗口的预测与真值。
- **跨框架可比性**：TFB 数据与 ETDataset 逐值相同，DUET 与 TSLib 式框架计分的是同一批测试窗口（E-SQZ5-2）。
- **血缘**：所有运行开始时本仓库无未提交改动（commit 2a1b499）；外部仓库为 clean，WPMixer、DUET 从导出副本运行。
- **预先计划与事后提出**：方法 × 预测长度的表与逐窗口分析在计划 v1/v1.1 中写定；十分位曲线、方法间相关与早停/验证损失的比较是看到结果后补充的描述。

## 主要结果

| 结果 | 对照 | 差异 | 不确定性 | 实际意义 | 证据 |
| --- | --- | --- | --- | --- | --- |
| ETTh1 四档平均 MSE | iTransformer 0.4576 | TQNet 0.4460（−0.0116），WPMixer 0.4370（−0.0206） | 三个种子；TQNet-720 的标准差 0.0206 | 在本项目的统一协议下，两种 2025 年的方法都好于 iTransformer | E-SQZ5-1 |
| 与作者报告值 | TQNet 0.441、WPMixer 0.422、DUET 0.377 | +0.005、+0.015（drop_last 计分 +0.011）、+0.0025 | 同上 | WPMixer 的差距最大，与它在测试集上调参（E-WP-C04）一致，但本实验不能区分原因 | E-SQZ5-1 |
| 数据核对 | — | TFB 与 ETDataset 逐值相同，测试窗口一致 | 确定性 | 跨框架比较成立 | E-SQZ5-2 |

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 见上表；96 档 TQNet 0.3728 < WPMixer 0.3778 < DUET 0.3795 < iTransformer 0.3858，336、720 档 WPMixer 最好 | 方法间差距 0.005～0.04，与种子波动同量级或略大 | — | 只有 ETTh1 | EXP-0004 阶段 B 后在其他数据集上复查 |
| 训练动态 | TQNet 8～18 轮早停，WPMixer 6～10 轮（10 轮上限）；TQNet-720 三个种子的最好验证损失只差 0.004，测试 MSE 差 0.037（F-TS9） | 验证期与测试期难度不同（验证 MSE 约为测试的 1.8 倍），验证损失在长预测上几乎不区分模型好坏 | 测试期的少数难段放大了模型之间本来很小的差别 | 只看了早停时的验证损失 | 按窗口比较验证期与测试期的误差结构 |
| 机制、中间量 | 四种架构逐窗口 MSE 相关约 0.97，十分位误差曲线几乎重合（F-TS8） | 误差主要由数据中的少数非平稳时段决定（M-SQZ5-1，推断） | 各方法都用 RevIN 类实例归一化，可能在同样的水平突变处失败 | 难段的数据特征未量化 | 计算每个窗口输入段与训练期的分布距离，看它能否预测窗口误差 |
| 子组、切片和边界 | 末 97 个窗口的 MSE 是其余的 1.09～1.37 倍，所有方法、所有预测长度都如此（F-TS2 更新）；96 档最难的是第 7 个十分位 | 测试期末尾与第 7 个十分位附近有分布变化 | — | 只有一个数据集 | 同上 |
| 稳健性及替代解释 | 三个种子、四种方法、四个预测长度方向一致；同一方法不同种子之间的胜率约 50% | 方法排序在 ETTh1 上相对稳定，但胜出的窗口只占 54%～58%（2026-10-29 完整阅读补注：54%～58% 只对 TQNet（种子 2021）与 DUET、WPMixer 的比较，以及 DUET 对 iTransformer 的比较成立；WPMixer 对 iTransformer 为 55.8%～61.8%，TQNet 对 iTransformer 为 64.9%～67.9%，见上面交接包里 F-TS8 的补注与 windows_stats.md。） | — | — | — |
| 局部收益、整体收益和代价 | TQNet 每次约 20 秒、WPMixer 约 30 秒、DUET 约 2 分钟、iTransformer 约 2 分钟（RTX 3090） | — | — | 未单独计时推理 | — |

## 学到什么

- **假设**：EXP-0005 没有方向性假设；它确认了 iTransformer 在 ETTh1 上是“效果稍弱”的 baseline，并把 EXP-0003 的 CPU 单种子结论放到 GPU、三种子、统一窗口下复核。
- **已排除与仍无法区分的解释**：
  - 已排除：DUET 与其他方法的分数差来自数据或窗口不同（E-SQZ5-2）；末尾窗口更难只是 WPMixer 的特点（F-TS2 更新）。
  - 仍无法区分：WPMixer 与论文的差距来自 GPU/AMP 还是测试集调参；难段来自何种分布变化。
- **保留 / 复验 / 回退 / 停止**：保留“全部窗口计分 + 三种子 + 同一批窗口”的协议；比较方法时同时报告逐窗口相关与胜率，因为平均 MSE 的差别只来自少数窗口。
- **最小、最能区分的下一次分析**：用已保存的预测，计算每个测试窗口输入段与训练期的分布距离（例如均值、方差的标准化差），看它与窗口误差的相关（只用 CPU）。
- **回流 Deep Read**：F-TS2、F-TS4 的更新，F-TS8、F-TS9，P-SQZ5-1、P-SQZ5-2，M-SQZ5-1 交给时序方向的 Mode C（待建立）。

## 补充分析：难段是不是数据中的水平突变（2026-10-24，按“最小、最能区分的下一次分析”执行）

由 `code/ts_forecasting/analyze_hard_windows.py` 生成（`code/output/EXP-0005/squeeze/hard_windows.md`、hard_windows.json；SHA256SUMS 的 sha256 34b6112c…，附脚本副本，已备份）。脚本按 TSLib 式加载器的划分从 CSV 重建 ETTh1-96 的 2785 个测试窗口，与保存的 true.npy 最大差 2.2e-07；逐窗口误差取 ETTh1-96 的 14 次运行（iTransformer 5 次、TQNet、WPMixer、DUET 各 3 次）的平均。

| 数据特征（训练集标准化尺度） | 与逐窗口平均 MSE 的 Spearman 相关 |
| --- | --- |
| 目标段与输入段的水平差（各变量 \|均值差\| 的平均） | 0.587 |
| 目标段自身的波动 | 0.322 |
| 输入段均值偏离训练期的程度 | 0.278 |

- 四种方法各自的逐窗口 MSE 与水平差的相关为 0.571（DUET）、0.575（TQNet）、0.584（iTransformer）、0.596（WPMixer），几乎相同。
- 按水平差的五分位，平均 MSE 为 0.271、0.293、0.356、0.390、0.591：水平差最大的五分之一窗口，误差是最小五分之一的 2.2 倍。
- 解释：M-SQZ5-1 得到相关层面的支持——各方法在“未来一段的水平相对输入段发生变化”的窗口上一起失败。四种方法都对输入做实例归一化（RevIN 或同类），预测时基本沿用输入段的水平，所以无法预见水平突变（推断）。
- 限制：只是相关；iTransformer 的 5 次运行里有 3 次逐位相同，平均时它的权重偏大（不影响各方法分别计算的相关）。
- 状态更新：M-SQZ5-1 从 AI 改为“相关层面支持（DE），因果未验证”；新增 E-SQZ5-6（本节）。

## 同步清单

- [x] experiment-results.md：登记 EXP-0005 的 39 次运行与本报告（main 65b7335）。
- [x] 已发现的实验现象.md：更新 F-TS2、F-TS4，新增 F-TS8、F-TS9（main 65b7335）；F-TS8 按补充分析更新（2026-10-24）。
- [x] 论文素材库.md：候选图（四种方法的十分位误差曲线、逐窗口相关矩阵），MAT-0007（main 65b7335）。
- [x] 最新实验分析报告.md、index.md（main 65b7335）。
