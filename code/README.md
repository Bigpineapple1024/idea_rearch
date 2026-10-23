# 实验代码

当前实现：[token_pruning/](token_pruning/README.md)，即视觉 token 剪枝 baseline 的复现脚本与代码级核验（只在 CPU 随机权重上验证过）。

在 code/<方向或实验编号> Git 分支开发，改造记录来源、许可证、基线 commit 与变更点。每次运行绑定 IDEA ID、实验 ID、代码 commit、配置、数据版本与划分、种子和环境。

原始产物放 output/<实验ID>/<运行ID>/，包括日志、指标、配置快照和诊断图。该目录已忽略，须登记可恢复备份。

所有运行登记到[实验结果总表](../analysis/experiment-results.md)，事实记录与深度报告放 analysis/实验报告/。选定框架后再添加真实依赖与复现命令，目前没有测试通过或复现成功的声明。
