# 单次实验报告

每个实验保存两份文件：

- `EXP-xxxx_record.md`：基于 [EXPERIMENT_RECORD](../templates/EXPERIMENT_RECORD.md)，记录实际发生了什么。
- `EXP-xxxx_squeeze.md`：基于 [SQUEEZE_REPORT](../templates/SQUEEZE_REPORT.md)，从这次实验中榨取尽可能多的知识（动态、错误、子组、边界、机制、稳健性），并整理成 deep-read 的输入。

**每一次实验都要做 squeeze**，包括 baseline 复现、组合实验和结果不理想的实验（[研究流程](../研究流程.md) 第 ④ 步）。

全部实验的登记与当前状态以[实验详细数据总表](../experiment-results.md)的“实验登记”为准；下表只列最早的三个实验（2026-10-25 跑偏检查补注）。

| 实验 | 内容 | record | squeeze |
| --- | --- | --- | --- |
| EXP-0001 | 视觉 token 剪枝三个仓库的代码级核验（CPU，随机权重） | [EXP-0001_record](EXP-0001_record.md)（completed） | [SQZ-EXP-0001](EXP-0001_squeeze.md)（完成，2026-10-23；原写“待补做”） |
| EXP-0002 | 视觉 token 剪枝 GPU 复现（vanilla、FastV、VisionZip、SparseVLM、随机剪枝） | [EXP-0002_record](EXP-0002_record.md)（completed，2026-10-24；原写“待执行”） | [SQZ-EXP-0002](EXP-0002_squeeze.md)（完成，2026-10-24；原写“—”） |
| EXP-0003 | 时序预测三个仓库在 ETTh1 上的 CPU 复现与评测协议核验 | [EXP-0003_record](EXP-0003_record.md)（partial：计划内运行全部结束，WPMixer 192 档被手动停止） | [SQZ-EXP-0003](EXP-0003_squeeze.md)（完成，2026-10-23，只基于记录中抄录的数字；原写“待补做”） |
| EXP-0038 | IDEA-0009 的开发运行（编号相同时的第 0 层借看，开发集） | [EXP-0038_record](EXP-0038_record.md)（completed） | [SQZ-EXP-0038](EXP-0038_squeeze.md)（2026-11-01 补成单独的报告） |
| EXP-0041 | IDEA-0011 的开发运行（虚拟填补，CDPruner 底座，开发集） | [EXP-0041_record](EXP-0041_record.md)（completed） | [SQZ-EXP-0041](EXP-0041_squeeze.md)（2026-11-01 补成单独的报告） |
| EXP-0042 | IDEA-0011 的确认（新数据与 MME、POPE） | [EXP-0042_record](EXP-0042_record.md)（completed） | [SQZ-EXP-0042](EXP-0042_squeeze.md)（2026-11-01 补成单独的报告） |

原始产物放在 `code/output/<实验ID>/<运行ID>/`，并备份到实验室电脑的 `D:\6CJNM007\backup\<实验ID>\<运行ID>\`，记录中登记校验和与备份路径。

先保存事实，再做分析，不覆盖原始证据。完成后更新结果总表、现象台账、论文素材库、最新报告入口和索引。
