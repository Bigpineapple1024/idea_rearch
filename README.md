# 研究项目工作区

从论文阅读、问题发现、实验验证到论文写作的可追溯工作区。

**当前状态：**目录、文档模板与五个项目级 skills 已落盘，交接模板已对齐；研究主题、baseline、数据集和依赖环境待补充。尚未运行论文解析、复现或实验。

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
| .claude/ | Claude Code 配置与 skills | 五个技能已安装，运行时加载待验证 |
| sessions/ | 本地对话存档 | 默认忽略存档，仅跟踪说明 |

Git 不保存空目录。新检出后在仓库根目录恢复忽略的目录：

```bash
mkdir -p data checkpoints papers code/output
```

## 开始工作

1. 阅读[文档索引](analysis/index.md)、[宏观规划](analysis/paper-plan.md)与[研究流程](analysis/研究流程.md)。
2. 明确领域、资源和评价协议，选择并复现 baseline。
3. 使用[模板](analysis/templates/README.md)保存论文、想法和实验的证据包。
4. 写作前读取[新会话清单](paper/论文写作-新会话读取清单.md)。

闭环：Analyze Paper → Deep Read（仅 Mode C）→ Discover Innovation → 实际执行实验 → Record Experiment → Squeeze Experiment → 回流 Deep Read。

技能入口及验证范围见[技能说明](.claude/README.md)。当前完成文件安装与静态核验，未执行研究流程。缺少材料或运行证据时标注待补充，不生成虚构结果。

## 分支与复现约定

main 维护管理文档与论文，code/<方向或实验编号> 隔离实验开发。Git 分支覆盖整个仓库，并不自动限制目录。实验分支中的记录应随工作保留，再将核验后的报告同步到 main，并保留实验代码的确切 commit。

项目结构与共享技能以 main 为基础维护；开始具体实验时再创建对应的 code/<方向或实验编号> 分支。云任务已隔离，使用已有检出，除非明确要求，不另建 worktree。

外部材料记录来源 URL、版本、许可、获取方式和校验和。运行产物记录本地位置及可恢复备份；仅有本机路径不足以长期复现。依赖和执行命令待实际框架确定后再验证，不预设技术栈。
