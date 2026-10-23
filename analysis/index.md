# 项目文档索引

状态（2026-10-23，DEC-0009）：视觉 token 剪枝与时序预测两条方向并行，各处于“复现 baseline”阶段：FastV 待在 GPU 上复现（EXP-0002）；DUET 已在 CPU 上初步复现（EXP-0003）。流程规范为[研究流程 v2.0](研究流程.md)。新增、移动或归档文档时同步本索引。

## 当前工作入口

| 内容 | 入口 |
| --- | --- |
| **流程规范（唯一）** | [研究流程 v2.0](研究流程.md) |
| 实验室电脑新会话的交接 | [实验室会话交接](实验室会话交接.md) |
| 视觉 token 剪枝：三篇共性、差异与 baseline（FastV） | [所有论文在baseline骨架的改进内容](专题分析/所有论文在baseline骨架的改进内容.md) |
| 问题—现象—机制矩阵 | [MODE_C-0001](MODE_C-0001_视觉token剪枝.md) |
| 候选方案（待 baseline 复现与组合实验后重新排序） | [IDEA-0001](IDEA-0001_编码器侧压缩的位置编号.md) |
| baseline 复现代码与步骤 | [code/token_pruning](../code/token_pruning/README.md) |
| 时序预测：三篇对比、baseline（DUET）与组合实验候选 | [时序预测_DUET_TQNet_WPMixer对比](专题分析/时序预测_DUET_TQNet_WPMixer对比.md) |
| 时序方向复现记录与代码 | [EXP-0003](实验报告/EXP-0003_record.md)、[code/ts_forecasting](../code/ts_forecasting/README.md) |

## 必需文档定位

| 内容 | 入口 |
| --- | --- |
| 所有文档与定位 | 本文件 |
| 实验现象与问题 | [已发现的实验现象](已发现的实验现象.md) |
| 可用于论文的证据素材 | [论文素材库](论文素材库.md) |
| 每篇论文的结构化分析 | [论文笔记](论文笔记/README.md) |
| 最近实验报告 | [最新实验分析报告](最新实验分析报告.md) |
| 关键决策与方向调整 | [version-evolution](version-evolution.md) |
| 实验到论文的规划 | [paper-plan](paper-plan.md) |
| 实验详细数据 | [experiment-results](experiment-results.md) |
| 写作所需材料 | [新会话读取清单](../paper/论文写作-新会话读取清单.md) |

## 全部现有文档

| 文档 | 定位 |
| --- | --- |
| [.claude/README.md](../.claude/README.md) | Claude Code 运行时支持 |
| [AGENTS.md](../AGENTS.md) | 研究工作区协作规则 |
| [analysis/archied/README.md](../analysis/archied/README.md) | 归档方案 |
| [analysis/experiment-results.md](../analysis/experiment-results.md) | Experiment Results：实验详细数据总表 |
| [analysis/paper-plan.md](../analysis/paper-plan.md) | Paper Plan：实验到论文的宏观规划 |
| [analysis/templates/1a-4b机制分析.md](../analysis/templates/1a-4b机制分析.md) | 1a–4b 机制分析：<问题> |
| [analysis/templates/ANALYSIS_PACKAGE.md](../analysis/templates/ANALYSIS_PACKAGE.md) | ANALYSIS_PACKAGE：<论文标题> |
| [analysis/templates/EXPERIMENT_RECORD.md](../analysis/templates/EXPERIMENT_RECORD.md) | EXPERIMENT_RECORD：<实验标题> |
| [analysis/templates/IDEA_PACKAGE.md](../analysis/templates/IDEA_PACKAGE.md) | IDEA_PACKAGE：<想法> |
| [analysis/templates/MODE_C_PACKAGE.md](../analysis/templates/MODE_C_PACKAGE.md) | MODE_C_PACKAGE：<主题> |
| [analysis/templates/README.md](../analysis/templates/README.md) | 模板与交接约定 |
| [analysis/templates/SQUEEZE_REPORT.md](../analysis/templates/SQUEEZE_REPORT.md) | SQUEEZE_REPORT：<实验标题> |
| [analysis/templates/跨领域迁移方案.md](../analysis/templates/跨领域迁移方案.md) | 跨领域迁移方案：<方法> |
| [analysis/version-evolution.md](../analysis/version-evolution.md) | Version Evolution：关键决策与方向调整 |
| [analysis/毕业选题候选.md](../analysis/毕业选题候选.md) | 毕业选题候选：约束、候选方向与下一步 |
| [analysis/archied/2026-10-23_SOP-v1_从论文到融合方案.md](../analysis/archied/2026-10-23_SOP-v1_从论文到融合方案.md) | 已归档：SOP v1（被研究流程 v2.0 取代） |
| [analysis/实验室会话交接.md](../analysis/实验室会话交接.md) | 实验室电脑新会话的交接说明 |
| [analysis/MODE_C-0001_视觉token剪枝.md](../analysis/MODE_C-0001_视觉token剪枝.md) | MODE_C_PACKAGE：多模态大模型视觉 token 剪枝 |
| [analysis/IDEA-0001_编码器侧压缩的位置编号.md](../analysis/IDEA-0001_编码器侧压缩的位置编号.md) | IDEA_PACKAGE：编码器侧压缩的位置编号（融合路线候选，deferred） |
| [analysis/论文笔记/PAPER-0001_FastV.md](../analysis/论文笔记/PAPER-0001_FastV.md) | ANALYSIS_PACKAGE：FastV |
| [analysis/论文笔记/PAPER-0002_SparseVLM.md](../analysis/论文笔记/PAPER-0002_SparseVLM.md) | ANALYSIS_PACKAGE：SparseVLM / SparseVLM+ |
| [analysis/论文笔记/PAPER-0003_VisionZip.md](../analysis/论文笔记/PAPER-0003_VisionZip.md) | ANALYSIS_PACKAGE：VisionZip |
| [analysis/实验报告/EXP-0001_record.md](../analysis/实验报告/EXP-0001_record.md) | EXPERIMENT_RECORD：三个仓库的代码级核验（CPU） |
| [code/token_pruning/README.md](../code/token_pruning/README.md) | 视觉 token 剪枝 baseline 复现与代码级核验 |
| [analysis/论文笔记/PAPER-0004_DUET.md](../analysis/论文笔记/PAPER-0004_DUET.md) | ANALYSIS_PACKAGE：DUET |
| [analysis/论文笔记/PAPER-0005_TQNet.md](../analysis/论文笔记/PAPER-0005_TQNet.md) | ANALYSIS_PACKAGE：TQNet |
| [analysis/论文笔记/PAPER-0006_WPMixer.md](../analysis/论文笔记/PAPER-0006_WPMixer.md) | ANALYSIS_PACKAGE：WPMixer |
| [analysis/专题分析/时序预测_DUET_TQNet_WPMixer对比.md](../analysis/专题分析/时序预测_DUET_TQNet_WPMixer对比.md) | 时序预测三篇论文对比：DUET、TQNet、WPMixer |
| [analysis/实验报告/EXP-0003_record.md](../analysis/实验报告/EXP-0003_record.md) | EXPERIMENT_RECORD：DUET/TQNet/WPMixer 在 ETTh1 上的 CPU 复现 |
| [code/ts_forecasting/README.md](../code/ts_forecasting/README.md) | 多变量长期时序预测 baseline 复现与评测核验 |
| [analysis/专题分析/baseline骨架的全景图.md](../analysis/专题分析/baseline骨架的全景图.md) | baseline骨架的全景图 |
| [analysis/专题分析/外部工具应用现状.md](../analysis/专题分析/外部工具应用现状.md) | 外部工具应用现状 |
| [analysis/专题分析/所有论文创新类型全景分析.md](../analysis/专题分析/所有论文创新类型全景分析.md) | 所有论文创新类型全景分析 |
| [analysis/专题分析/所有论文在baseline骨架的改进内容.md](../analysis/专题分析/所有论文在baseline骨架的改进内容.md) | 所有论文在baseline骨架的改进内容 |
| [analysis/专题分析/顶会论文创新思路分析.md](../analysis/专题分析/顶会论文创新思路分析.md) | 顶会论文创新思路分析 |
| [analysis/专题分析/顶会论文完整规律汇总.md](../analysis/专题分析/顶会论文完整规律汇总.md) | 顶会论文完整规律汇总 |
| [analysis/专题分析/顶会论文完整解题思路.md](../analysis/专题分析/顶会论文完整解题思路.md) | 顶会论文完整解题思路 |
| [analysis/专题分析/顶会论文工作量分析.md](../analysis/专题分析/顶会论文工作量分析.md) | 顶会论文工作量分析 |
| [analysis/实验报告/README.md](../analysis/实验报告/README.md) | 单次实验报告 |
| [analysis/已发现的实验现象.md](../analysis/已发现的实验现象.md) | 已发现的实验现象 |
| [analysis/最新实验分析报告.md](../analysis/最新实验分析报告.md) | 最新实验分析报告 |
| [analysis/研究流程.md](../analysis/研究流程.md) | 研究流程 v2.0：从论文到 idea，再从 idea 到实现（唯一流程规范） |
| [analysis/论文笔记/README.md](../analysis/论文笔记/README.md) | 论文笔记 |
| [analysis/论文素材库.md](../analysis/论文素材库.md) | 论文素材库 |
| [CLAUDE.md](../CLAUDE.md) | Claude Code 项目入口 |
| [code/README.md](../code/README.md) | 实验代码 |
| [literature/README.md](../literature/README.md) | 论文解析材料 |
| [paper/README.md](../paper/README.md) | 论文写作 |
| [paper/论文写作-新会话读取清单.md](../paper/论文写作-新会话读取清单.md) | 论文写作：新会话读取清单 |
| [README.md](../README.md) | 研究项目工作区 |
| [sessions/README.md](../sessions/README.md) | 对话存档 |
| [source/README.md](../source/README.md) | 开源 baseline 参考 |
| [.claude/skills/5-skill-research-workflow.md](../.claude/skills/5-skill-research-workflow.md) | 五个研究 skill 的规范调用流程 |
| [.claude/skills/analyze-paper/SKILL.md](../.claude/skills/analyze-paper/SKILL.md) | Analyze Paper — 通用论文与研究分析 |
| [.claude/skills/deep-read/SKILL.md](../.claude/skills/deep-read/SKILL.md) | Deep Read — 模式 C：问题—现象—机制交叉矩阵 |
| [.claude/skills/discover-innovation/SKILL.md](../.claude/skills/discover-innovation/SKILL.md) | Discover Innovation — 通用研究创新发现 |
| [.claude/skills/record-experiment/SKILL.md](../.claude/skills/record-experiment/SKILL.md) | Record Experiment — 通用实验记录 |
| [.claude/skills/squeeze-experiment/SKILL.md](../.claude/skills/squeeze-experiment/SKILL.md) | Squeeze Experiment — 通用实验数据深挖 |
| [analysis/index.md](index.md) | 本索引 |

## 非文档与外部资源

- 根目录 .gitignore 定义外部资源、实验产物和本地运行时文件的忽略规则。
- literature/processed/.gitkeep 仅保留解析目录；.claude/skills/ 已包含实际技能文件。
- [.claude/skills/manifest.json](../.claude/skills/manifest.json) 保存上传包来源、协议版本与六份原文件的校验和。
- data/、checkpoints/、papers/、code/output/ 不跟踪；在对应笔记或实验记录中登记来源、校验和及可恢复位置。
- 外部代码（FastV、SparseVLM、VisionZip、LLaVA）只登记来源与 commit（见 [source/README](../source/README.md)），不纳入仓库；尚无真实实验数据或归档方案。
