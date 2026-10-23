# 单次实验报告

每个实验保存两份文件：

- `EXP-xxxx_record.md`：基于 [EXPERIMENT_RECORD](../templates/EXPERIMENT_RECORD.md)，记录实际发生了什么。
- `EXP-xxxx_squeeze.md`：基于 [SQUEEZE_REPORT](../templates/SQUEEZE_REPORT.md)，从这次实验中榨取尽可能多的知识（动态、错误、子组、边界、机制、稳健性），并整理成 deep-read 的输入。

**每一次实验都要做 squeeze**，包括 baseline 复现、组合实验和结果不理想的实验（[研究流程](../研究流程.md) 第 ④ 步）。

| 实验 | 内容 | record | squeeze |
| --- | --- | --- | --- |
| EXP-0001 | 视觉 token 剪枝三个仓库的代码级核验（CPU，随机权重） | [EXP-0001_record](EXP-0001_record.md) | 待补做 |
| EXP-0002 | 视觉 token 剪枝 GPU 复现（vanilla、FastV、VisionZip、SparseVLM、随机剪枝） | 待执行 | — |
| EXP-0003 | 时序预测三个仓库在 ETTh1 上的 CPU 复现与评测协议核验 | [EXP-0003_record](EXP-0003_record.md) | 待补做 |

原始产物放在 `code/output/<实验ID>/<运行ID>/`，并备份到实验室电脑 D 盘，记录中登记校验和与备份路径。

先保存事实，再做分析，不覆盖原始证据。完成后更新结果总表、现象台账、论文素材库、最新报告入口和索引。
