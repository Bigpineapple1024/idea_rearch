# 研究项目工作区

从论文阅读、问题发现、实验验证到论文写作的可追溯工作区。

**当前状态（2026-10-23）：**两条方向并行，均处于“复现 baseline”阶段。

- **视觉 token 剪枝**：主 baseline 为 FastV（LLaVA-1.5-7B）。代码已在 CPU 上核验（EXP-0001），GPU 复现（EXP-0002）待执行。
- **时序预测**：主 baseline 为 DUET。ETTh1 已在 CPU 上真实复现（EXP-0003）。

流程规范见[研究流程 v2.0](analysis/研究流程.md)，入口见[文档索引](analysis/index.md)，在实验室电脑上开新会话先读[交接说明](analysis/实验室会话交接.md)。

## 目录与维护边界

| 路径 | 用途 | 维护策略 |
| --- | --- | --- |
| data/ | 外部数据集 | 外部下载，不跟踪 |
| checkpoints/ | 外部或训练权重 | 本地保存，不跟踪 |
| papers/ | PDF 原文 | 外部下载，不跟踪 |
| source/ | 自己 fork 的开源 baseline | 只读参考，保留来源与许可 |
| literature/processed/ | MinerU 解析全文 | 只读证据，保留来源和版本 |
| code/ | 实验代码 | code/<方向或实验编号> 分支开发 |
| code/output/ | 日志、指标、图表等实验产物 | 不跟踪；另行保存原始证据与备份 |
| paper/ | 论文正文 | main 分支维护 |
| analysis/ | 实验报告、笔记、规划和决策 | main 分支维护 |
| analysis/实验报告/ | 单次实验事实记录与深度报告 | 保留完整历史 |
| analysis/archied/ | 归档方案 | 保留指定拼写，不覆盖历史 |
| .claude/ | Claude Code 配置与 skills | 五个技能已安装 |
| sessions/ | 本地对话存档 | 默认忽略存档，仅跟踪说明 |

Git 不保存空目录。新检出后在仓库根目录恢复忽略的目录：

```bash
mkdir -p data checkpoints papers code/output
```

## 开始工作

1. 阅读[文档索引](analysis/index.md)、[宏观规划](analysis/paper-plan.md)与[研究流程](analysis/研究流程.md)。
2. 按研究流程第 ①② 步选择并复现 baseline，能复现才进入下一步。
3. 使用[模板](analysis/templates/README.md)保存论文、想法和实验的证据包。
4. 写作前读取[新会话清单](paper/论文写作-新会话读取清单.md)。

闭环：Analyze Paper → Deep Read（仅 Mode C）→ Discover Innovation → 实际执行实验 → Record Experiment → Squeeze Experiment → 回流 Deep Read。

技能入口见[技能说明](.claude/README.md)。缺少材料或运行证据时标注待补充，不生成虚构结果。

## 分支与复现约定

main 维护管理文档与论文，code/<方向或实验编号> 隔离实验开发。Git 分支覆盖整个仓库，并不自动限制目录。实验分支中的记录应随工作保留，再将核验后的报告同步到 main，并保留实验代码的确切 commit。

项目结构与共享技能以 main 为基础维护；实验分支为 code/token-pruning（视觉 token 剪枝）和 code/ts-forecasting（时序预测）。由 Claude Code 提交并推送。云任务已隔离，使用已有检出，除非明确要求，不另建 worktree。

外部材料记录来源 URL、版本、许可、获取方式和校验和。运行产物记录本地位置及可恢复备份；仅有本机路径不足以长期复现。依赖、环境与执行命令见 [code/README](code/README.md) 及各方向目录的 README。
