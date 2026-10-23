# Experiment Results：实验详细数据总表

状态：视觉 token 剪枝只有代码级核验（EXP-0001，CPU，随机权重）；时序预测有 ETTh1 上的 CPU 真实复现（EXP-0003）。空表表示无数据，绝不表示数值为零。

## 实验登记

| 实验 ID | IDEA ID | 核心问题 | 事实记录 | 分析报告 | 状态 |
| --- | --- | --- | --- | --- | --- |
| EXP-0001 | none | 三个剪枝仓库的代码实际行为、预算口径、能否在官方 LLaVA 推理中跑通 | [EXP-0001_record](实验报告/EXP-0001_record.md) | 待补做 squeeze（结论已先写入论文笔记与 MODE_C-0001） | completed |
| EXP-0002 | none | 视觉 token 剪枝 GPU 复现：主 baseline FastV，以及 vanilla / VisionZip / SparseVLM / 随机剪枝 × 192/128/64 × POPE、MME | 待执行 | — | planned |
| EXP-0003 | none | 时序预测：DUET、TQNet、WPMixer 在 ETTh1 上的 CPU 复现；WPMixer drop_last 与 DUET 测试随机性核验 | [EXP-0003_record](实验报告/EXP-0003_record.md) | 待补做 squeeze | partial |

## 全部运行

每个运行一行，包括失败、中断及无差异运行，不只收录最好结果。

| 实验 ID / 运行 ID | 组别 | commit / 工作区差异记录 | 配置与环境 | 数据版本 / 划分 | 种子 | 状态 / 退出原因 | 主指标（单位、方向） | 局部指标 | 副作用 / 成本 | 原始证据及校验和 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0001-R1 | FastV KV 版 × eager/sdpa × rotary 补丁 | FastV d1659729；本仓库 probes/probe_fastv.py | CPU，torch 2.1.2，transformers 4.37.2 | 随机输入 | 0 | completed（3 组按预期报错） | NA（代码核验） | 每层 KV 长度 | — | 输出全文见 EXP-0001 记录 |
| EXP-0001-R2 | SparseVLM 打分与预算表 | SparseVLM a9e71427；probes/probe_sparsevlm.py | 同上 | 随机输入 | 0 | completed | NA | batch 2 保留数；层平均预算；RoPE 除数 | — | 同上 |
| EXP-0001-R3 | VisionZip 编码器 | VisionZip 8f86b55c；probes/probe_visionzip.py | 同上 | 随机 CLIP | 0 | completed | NA | 输出 token 数、CLS、合并幅值 | — | 同上 |
| EXP-0001-R4 | FLOPs 口径计算 | probes/flops_budget.py | Python 标准库 | — | — | completed | NA | 理论 TFLOPs、KV 大小 | — | 同上 |
| EXP-0001-R5 | fastv_llava vs 官方 | check_fastv_equivalence.py | CPU | 随机输入 | 0/1 | completed | NA | 相对 L2 差 ≤3.23e-07 | — | 同上 |
| EXP-0001-R6 | 4 种配置走 LLaVA generate() | smoke_test.py，LLaVA c121f043 | CPU，小随机 LLaVA | 随机图像 | 0/1 | completed（两次一致） | NA | 每层视觉 token 数 | — | 同上 |
| EXP-0003-R1～R4 | TQNet ETTh1 96→96/192/336/720 | TQNet 15e19cb2；官方 etth1.sh 参数 | CPU，Python 3.11，torch 2.1.2 | ETDataset ETTh1（sha256 f18de3ad…），12/4/4 | 2024 | completed | MSE 0.3703 / 0.4282 / 0.4754 / 0.4849（越低越好） | MAE 0.3931 / 0.4236 / 0.4473 / 0.4705 | 约 1 分钟/次 | 数值抄录于 EXP-0003 记录；原始日志未备份 |
| EXP-0003-R5～R7 | TQNet ETTh1 96→96 | 同上 | 同上 | 同上 | 2021 / 2022 / 2023 | completed | MSE 0.3722 / 0.3718 / 0.3771 | 4 种子均值 0.3729，标准差 0.0030 | — | 同上 |
| EXP-0003-R8 | WPMixer 统一设置 96→96 | WPMixer 74104c9d；code/ts_forecasting/wpmixer_droplast.py | 同上（无 AMP） | 同上 | 42 | completed | 官方计分 MSE 0.3696；全部窗口 0.3727 | MAE 0.3956 / 0.3953 | 约 52 秒/epoch | 同上 |
| EXP-0003-R9 | WPMixer 统一设置 96→192 | 同上 | 同上 | 同上 | 42 | stopped（手动：该档只丢 1 个窗口） | NA | — | — | 日志未备份 |
| EXP-0003-R10 | WPMixer 统一设置 96→336 | 同上 | 同上 | 同上 | 42 | completed | 官方计分 MSE 0.4686；全部窗口 0.4735 | MAE 0.4337 / 0.4388 | 约 160 秒/epoch（并行） | 数值抄录于 EXP-0003 记录 |
| EXP-0003-R11 | WPMixer 统一设置 96→720 | 同上 | 同上 | 同上 | 42 | completed | 官方计分 MSE 0.4527；全部窗口 0.4615 | MAE 0.4473 / 0.4535 | 约 70 秒/epoch | 同上 |
| EXP-0003-R12 | DUET 统一设置 96→96 + 10 次测试 mask 重采样 | DUET dcc6e678；duet_test_stochasticity.py | 同上 | ETTh1 前 14400 行转 TFB 格式 | 2021 | completed | mse_norm 0.3898 | mae_norm 0.3990；mask 重采样 MSE 标准差 0.0003 | 12 个 epoch | 同上 |
| EXP-0003-R13 | DUET 随机权重 eval 两次前向 | probe_duet_mask.py | 同上 | 随机输入 | 0 | completed | NA（代码核验） | 最大输出差 0.186 | — | 同上 |
| EXP-0003-R14 | DUET 统一设置 96→96 | 同 R12 | 同上 | 同上 | 2022 | completed | mse_norm 0.3790 | mae_norm 0.3960 | 约 11 分钟 | 同上 |

## 汇总与比较

| 对比 ID | 运行 ID 集合 | 样本量 / 重复次数 | 汇总方法 | 效应量 / 不确定性及算法 | 实际意义阈值 | 排除与理由 | 结论范围 |
| --- | --- | --- | --- | --- | --- | --- | --- |

指标较多时在单次记录中使用长表，或引用可恢复的 CSV/JSON 并登记 schema、计算命令与校验和。缺失写 NA 及原因，未运行不写成失败或通过。见[事实记录模板](templates/EXPERIMENT_RECORD.md)。
