# Experiment Results：实验详细数据总表

状态：只有代码级核验（EXP-0001，CPU，随机权重），尚无真实模型的准确率或速度数据。空表表示无数据，绝不表示数值为零。

## 实验登记

| 实验 ID | IDEA ID | 核心问题 | 事实记录 | 分析报告 | 状态 |
| --- | --- | --- | --- | --- | --- |
| EXP-0001 | none | 三个剪枝仓库的代码实际行为、预算口径、能否在官方 LLaVA 推理中跑通 | [EXP-0001_record](实验报告/EXP-0001_record.md) | 无（代码核验，结论已写入论文笔记与 MODE_C-0001） | completed |
| EXP-0002 | none | GPU 复现 vanilla / VisionZip / FastV / SparseVLM / 随机剪枝 × 192/128/64 × POPE、MME | 待执行 | — | planned |

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

## 汇总与比较

| 对比 ID | 运行 ID 集合 | 样本量 / 重复次数 | 汇总方法 | 效应量 / 不确定性及算法 | 实际意义阈值 | 排除与理由 | 结论范围 |
| --- | --- | --- | --- | --- | --- | --- | --- |

指标较多时在单次记录中使用长表，或引用可恢复的 CSV/JSON 并登记 schema、计算命令与校验和。缺失写 NA 及原因，未运行不写成失败或通过。见[事实记录模板](templates/EXPERIMENT_RECORD.md)。
