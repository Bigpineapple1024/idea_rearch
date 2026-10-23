# EXPERIMENT_RECORD：DUET、TQNet、WPMixer 在 ETTh1 上的 CPU 复现与评测协议核验

状态：partial（计划内运行全部结束；WPMixer 192 档被手动停止；仅 ETTh1，多数为单种子）。
实验 ID EXP-0003 / IDEA ID none / 2026-10-23 / 执行者 Claude Code 云会话（4 CPU，无 GPU）。

## 结构化交接包

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
package_id: EXP-0003
experiment_id: EXP-0003
created_at: 2026-10-23
source_ids: [PAPER-0004, PAPER-0005, PAPER-0006]
evidence_status: partial
evidence_locations: [本文“全部运行”“原始结果”, code/ts_forecasting/]
missing_evidence:
- 只运行了 ETTh1；其他数据集未运行
- 除 TQNet-96 外均为单种子
- CPU fp32 运行，官方 WPMixer 脚本在 GPU 上用 AMP
next_skill: squeeze-experiment
source_idea_id: none
plan_version: "无 IDEA 包；协议 = 各仓库官方脚本参数 + 本项目的计分与随机性检查"
execution_status: partial
research_question: 三个官方仓库在 ETTh1 上能否复现报告值；评测协议差异（测试集丢窗口、测试时随机性）对分数和排名有多大影响
hypothesis: 无预设方向的核验
research_question_and_hypothesis:
  research_question: 三个官方仓库在 ETTh1 上能否复现报告值；评测协议差异对分数和排名有多大影响
  hypothesis: 无预设方向的核验
planned_vs_actual_design_and_controls: 见“计划与实际”
actual_configuration: 见“全部运行”
samples_materials_or_data_provenance: ETDataset ETTh1.csv（GitHub zhouhaoyi/ETDataset@1d16c8f4，sha256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066，17420 行）；标准 12/4/4 个月划分
all_runs_or_repetitions: 见“全部运行”
results_with_uncertainty: 见“原始结果”
observations_and_deviations: 见“分层记录”
known_limits_and_missing_evidence: 单数据集；多数为单种子
next_routes: [选定方向后在 GPU 上按统一协议多种子、多数据集重评, 建立时序方向的 MODE_C]
```

## 计划与实际

- **问题**：①复现报告值；②WPMixer 测试集 `drop_last=True` 对分数的影响；③DUET 测试时通道 mask 随机采样对分数的影响。
- **代码**：DUET dcc6e678、TQNet 15e19cb2、WPMixer 74104c9d；包装脚本见 code/ts_forecasting/（本次提交），不修改仓库文件。
- **环境**：Python 3.11、torch 2.1.2（CPU）、pandas 1.5.3、scikit-learn 1.3.2、PyWavelets 1.6.0、optuna 3.6.1、thop、ray、dash 等；OMP 线程 2～4。
- **数据**：TQNet 与 WPMixer 直接使用原始 CSV；DUET 用 to_tfb_format.py 截取前 14400 行转成 TFB 长表，按 TFB 的 0.6/0.2/0.2 划分恰好是 8640/2880/2880。
- **偏差**：
  - 官方环境为 Python 3.8/3.10、torch 2.0.1/2.4.1，本次统一用 3.11 + 2.1.2。
  - WPMixer 官方脚本带 `--use_amp`（GPU fp16），CPU 上不使用 AMP。
  - WPMixer 192 档运行到一半时被手动停止：该档官方计分只丢 1 个窗口，无法反映 drop_last 的影响；为节省时间改跑 336、720。停止的日志保留为 wpmixer_unified_etth1_192.stopped.log。
  - 中途一次 WPMixer 192/336/720 运行因 setuptools 被其他依赖升级、缺少 `pkg_resources` 而立即失败，修复后重跑。

## 全部运行

| 运行 ID | 方法 / 设置 | 种子 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| EXP-0003-R1 | TQNet ETTh1 96→96（官方 etth1.sh 参数） | 2024 | completed | 约 1 分钟 |
| EXP-0003-R2～R4 | TQNet ETTh1 96→192/336/720 | 2024 | completed | — |
| EXP-0003-R5～R7 | TQNet ETTh1 96→96 | 2021/2022/2023 | completed | 种子波动 |
| EXP-0003-R8 | WPMixer 统一设置 96→96，同一 checkpoint 两种计分 | 42 | completed | 每个 epoch 约 52 秒，10 个 epoch |
| EXP-0003-R9 | WPMixer 统一设置 96→192 | 42 | stopped（手动） | 理由见“偏差” |
| EXP-0003-R10 | WPMixer 统一设置 96→336，同一 checkpoint 两种计分 | 42 | completed | 每个 epoch 约 160 秒（与其他任务并行），第 7 个 epoch 早停 |
| EXP-0003-R11 | WPMixer 统一设置 96→720，同一 checkpoint 两种计分 | 42 | completed | 每个 epoch 约 70 秒，10 个 epoch |
| EXP-0003-R12 | DUET 统一设置 96→96 + 10 次测试 mask 重采样 | 2021（TFB 默认，deterministic=full） | completed | 12 个 epoch 后早停 |
| EXP-0003-R13 | DUET 随机权重 eval 模式两次前向（probe_duet_mask.py） | 0 | completed | 代码级探针 |
| EXP-0003-R14 | DUET 统一设置 96→96 + 1 次测试 mask 重采样 | 2022（strategy-args 覆盖） | completed | 约 11 分钟；用于判断与报告值的差距是否来自种子 |

## 原始结果

| 运行 ID | 指标 | 数值 | 对照（报告值） |
| --- | --- | --- | --- |
| R1 | MSE / MAE | 0.37035 / 0.39314 | result.txt：0.37122 / 0.39282 |
| R2 | MSE / MAE（192） | 0.42822 / 0.42364 | 0.42840 / 0.42609 |
| R3 | MSE / MAE（336） | 0.47536 / 0.44728 | 0.47571 / 0.44606 |
| R4 | MSE / MAE（720） | 0.48493 / 0.47048 | 0.48743 / 0.46977 |
| R5～R7 | MSE（96，种子 2021/2022/2023） | 0.37224 / 0.37183 / 0.37714 | 与 R1 合计 4 个种子：均值 0.3729，标准差 0.0030 |
| R8 | 官方计分（2688/2785 个窗口） | MSE 0.36956 / MAE 0.39563 | Table 9：0.368 / 0.394 |
| R8 | 全部窗口（2785 个） | MSE 0.37274 / MAE 0.39530 | — |
| R10 | 官方计分（2432/2545 个窗口） | MSE 0.46860 / MAE 0.43375 | Table 9：0.452 / 0.437 |
| R10 | 全部窗口（2545 个） | MSE 0.47349 / MAE 0.43876 | — |
| R11 | 官方计分（2048/2161 个窗口） | MSE 0.45271 / MAE 0.44731 | Table 9：0.449 / 0.449 |
| R11 | 全部窗口（2161 个） | MSE 0.46153 / MAE 0.45353 | — |
| R12 | mse_norm / mae_norm（全部 2785 个窗口；TFB test_report 与本项目计算一致） | 0.38980 / 0.39899 | 统一设置表：0.377 / 0.393 |
| R12 | 只重采样测试 mask 10 次 | MSE 0.38943～0.39045，均值 0.38983，标准差 0.00032 | — |
| R13 | 同一输入两次前向的最大输出差 | 0.1855；mask 196 个元素中 36 个不同；固定种子时差为 0 | — |
| R14 | mse_norm / mae_norm | 0.37897 / 0.39595 | 0.377 / 0.393；与 R12 合计两个种子均值 0.3844，相差 0.011 |

drop_last 的影响汇总（WPMixer 统一设置，同一 checkpoint，全部窗口 − 官方计分）：96 档 +0.0032，336 档 +0.0049，720 档 +0.0088（MSE）。丢弃的窗口比例分别为 3.5%、4.4%、5.2%。

## 分层记录

- **观察事实**：
  - TQNet 四个预测长度的复现 MSE 与官方相差 ≤0.0025。
  - WPMixer 全部窗口计分比官方计分高 0.0032 / 0.0049 / 0.0088（96/336/720 档 MSE）。预测越长，丢弃比例越高，差距越大。
  - DUET 在 eval 模式下输出随机，但训练后只重采样 mask 时 MSE 标准差仅 0.0003。
  - DUET 两个种子相差 0.011，报告值落在两者附近。
- **当前解释（推断）**：
  - 96 档、单种子时，计分方式足以让 WPMixer 与 TQNet 的排名互换（0.3696 < 0.3703 < 0.3727），但差距小于 TQNet 的种子标准差，不能据此判断谁更好。
  - 336、720 档在全部窗口计分下，WPMixer（0.4735、0.4615）仍低于 TQNet（0.4754、0.4849）。也就是说，在长预测上 WPMixer 的优势并非完全来自 drop_last；但它的超参数是在测试集上选的（E-WP-04），这部分影响本次无法分离。
- **待复验**：多种子、多数据集下 drop_last 与测试集调参对 WPMixer 报告优势的贡献。
- **不能得出的结论**：任何方法在其他数据集上的表现；方法之间的优劣。

## 交接

原始日志保存在本会话的临时目录（不随仓库保存），关键数字已全部抄录于上表；可用 code/ts_forecasting/README.md 中的命令重新生成。
