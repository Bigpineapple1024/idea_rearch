# SQUEEZE_REPORT：DUET、TQNet、WPMixer 在 ETTh1 上的 CPU 复现（EXP-0003）

状态：完成（2026-10-23，实验室电脑补做；交接文档第 7 步）。只基于记录中抄录的数字：云端原始日志、预测和真值文件都没有保存。
报告 ID SQZ-EXP-0003 / 实验 ID EXP-0003 / 输入记录：analysis/实验报告/EXP-0003_record.md（2026-10-23 版）/ 日期 2026-10-23。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0003
squeeze_id: SQZ-EXP-0003
created_at: 2026-10-23
source_ids: [EXP-0003, PAPER-0001, PAPER-0002, PAPER-0003]
evidence_status: partial
evidence_locations:
- analysis/实验报告/EXP-0003_record.md（“原始结果”表与 drop_last 汇总）
- source/WPMixer@74104c9d data_provider/data_factory.py 第 22～23 行（测试集 shuffle=False、drop_last=True）
missing_evidence:
- 云端原始日志、checkpoint、逐窗口预测与真值均未保存，本报告的派生量只能由抄录的汇总数字推算
- WPMixer 每档只有一个种子；DUET 两个种子；只有 ETTh1
- CPU fp32 运行，官方 WPMixer 在 GPU 上用 AMP
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0003]
claim_scope: ETTh1、回看 96、各仓库官方参数下的 CPU 结果；不推广到其他数据集或方法优劣
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“深度分析”第 1 行
process_dynamics: 记录中只有 epoch 数与早停，没有逐 epoch 曲线；不分析
mechanism_and_intermediate_findings: 见“深度分析”第 3、5 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 6 行
new_P: [P-SQZ3-1]
new_F: none（F-TS2 获得定量更新；F-TS4、F-TS5、F-TS7 与本报告一致，不另立新 ID）
new_M: [M-SQZ3-1]
evidence_index: 见“证据审计”
open_questions: [ETTh1 测试期末尾的窗口为什么更难, 各方法的差距是否集中在这些窗口]
next_routes: [EXP-0005（GPU、统一协议、3 个种子）保存逐窗口误差并按时间顺序分析, iTransformer（EXP-0004）同样保存 pred/true, deep-read Mode C 回流]
mode_C_input_bundle:
  source_ids: [EXP-0003, SQZ-EXP-0003]
  P:
  - id: P-SQZ3-1
    text: "ETTh1 测试期最末的窗口误差显著更高；方法之间的差距是否集中在这些窗口，测试集截断会不会系统地改变排名"
    evidence: [F-TS2]
    status: SC（派生量，待逐窗口数据确认）
  F:
  - id: F-TS2（定量更新，台账已有此 ID）
    text: "WPMixer 的 drop_last 去掉的是测试期最末的 97/113/113 个窗口（3.48%/4.44%/5.23%）；由抄录的两种计分反推，这些窗口的平均 MSE 为 0.461/0.579/0.621，比保留窗口高 24.7%/23.5%/37.3%（96/336/720 档）"
    evidence: [EXP-0003 R8/R10/R11, WPMixer data_factory.py 第 22～23 行]
    status: DE（由汇总数字推算的派生量；逐窗口数据未保存）
  M:
  - id: M-SQZ3-1
    text: "测试期末尾的分布漂移（季节、负载变化）使最末窗口更难预测，drop_last 因而有利于报告值"
    evidence: []
    status: AI/UN（没有直接证据，需逐窗口误差随时间的曲线和外部变量）
  E:
  - id: E-SQZ3-1
    text: "反推公式：丢弃窗口均值 m_d = (全部 − (1−f)·保留) / f，f 为丢弃比例；数值见上面的 F-TS2 条目"
    location: 本报告“深度分析”第 3 行
  evidence_locations: [analysis/实验报告/EXP-0003_record.md]
```

## 证据审计

- **输入**：EXP-0003 记录的“全部运行”“原始结果”表。14 个运行中 R9 被手动停止（理由已记录），其余完成。
- **缺失**：原始日志、checkpoint、逐窗口预测都在云端临时目录，会话结束后丢失。因此本报告的所有新分析都是基于抄录数字的推算，不能做逐窗口、逐通道或逐 epoch 的分析。
- **分析命令**：派生量用 PowerShell 按 E-SQZ3-1 的公式计算（输入为记录中的 R8/R10/R11 数字与窗口数）；TQNet 的复现判定用 4 个种子的样本标准差。
- **预先计划与事后提出**：drop_last 影响是 EXP-0003 的预设问题；“丢弃窗口本身更难”的反推是本次看到数字后提出的。

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | TQNet 四档 MSE 与 result.txt 相差 ≤0.0025；96 档 4 个种子（2021～2024）均值 0.37289、标准差 0.00295，报告值 0.37122 落在均值 ± 2 倍标准差内（按研究流程的复现标准通过）。WPMixer 官方计分与 Table 9 相差 +0.0016/+0.0166/+0.0037（96/336/720，单种子）。DUET 两个种子均值 0.3844，报告 0.377 | TQNet 可以复现；WPMixer 336 档偏差较大，单种子无法判断；DUET 报告值落在两个种子之间 | WPMixer 偏差可能来自 CPU 不用 AMP、种子或并行运行 | WPMixer、DUET 种子不足 | EXP-0005 在 GPU 上每档 3 个种子 |
| 训练 / 运行动态 | 只有 epoch 数与早停点（WPMixer 336 档第 7 个 epoch 早停；DUET 12 个 epoch） | — | — | 没有曲线 | EXP-0005 保存逐 epoch 验证损失 |
| 错误和失败模式 | F-TS2（定量更新）：drop_last 去掉的是时间上最末的窗口（测试加载 shuffle=False），其 MSE 比保留窗口高 23.5%～37.3% | drop_last 不是随机地少算几个窗口，而是系统地去掉最难的窗口，所以官方计分偏低，而且预测越长偏得越多 | 末尾窗口数少（97～113 个），反推值有抽样误差；但三档方向一致 | 派生量，没有逐窗口数据 | EXP-0005 保存逐窗口误差直接验证 |
| 子组、切片和边界 | 96 档：WPMixer 官方 0.3696 < TQNet 0.3703 < WPMixer 全部窗口 0.3727 | 计分方式足以改变排名，但差距小于 TQNet 的种子标准差 | — | WPMixer 单种子 | 统一协议、多种子后再比较 |
| 机制、消融和交互 | M-SQZ3-1（AI/UN）：测试期末尾的分布漂移 | 若成立，末尾窗口是区分方法的关键切片 | 末尾也可能只是噪声更大，而不是漂移 | 无外部变量 | 画逐窗口误差随时间的曲线，与训练期统计量比较 |
| 稳健性及跨数据集 | F-TS4、F-TS5、F-TS7：种子波动与方法差距同量级；DUET 的方差主要来自训练随机性（种子差 0.011），不是测试 mask（标准差 0.0003） | 单种子结果不能用于排名；DUET 至少 3 个种子 | — | 只有 ETTh1 | EXP-0005；之后扩展到其他 ETT 数据集 |
| 局部收益、整体收益和代价 | CPU 上 TQNet 约 1 分钟/次，WPMixer 52～160 秒/epoch，DUET 约 11 分钟/次 | GPU 重跑成本低 | — | 与 CPU 负载有关 | — |

## 学到什么

- **假设**：EXP-0003 无预设方向。本次分析支持“WPMixer 的官方计分系统性偏低”，并给出了偏低的来源（最难的末尾窗口被丢掉）。
- **已排除与仍无法区分的解释**：排除了“drop_last 只是随机少算几个窗口”；仍无法区分末尾窗口更难的原因（漂移 vs 噪声）。
- **保留 / 复验 / 回退 / 停止**：保留 TQNet 的复现结论；WPMixer、DUET 待 EXP-0005 用 3 个种子复验。所有时序对比统一用全部窗口计分。
- **最小、最能区分的下一次实验**：EXP-0005 在 GPU 上按统一协议重跑时保存逐窗口误差，直接比较末尾窗口与其余窗口的误差，以及各方法在末尾窗口上的差距。
- **回流 Deep Read**：P-SQZ3-1、F-TS2 的定量更新、M-SQZ3-1（见交接包）；F-TS4、F-TS5、F-TS7 不变。

## 同步清单

- [ ] experiment-results.md：EXP-0003 分析报告一栏指向本报告。
- [ ] 已发现的实验现象.md：F-TS2 补入派生量与本报告链接（事实与解释分开）。
- [ ] 论文素材库.md：登记“drop_last 去掉最难窗口”作为评测协议素材（探索，待 EXP-0005 复验）。
- [ ] 最新实验分析报告.md：指向本报告。
- [x] version-evolution.md：无方向变化。
- [ ] index.md：登记本报告。
