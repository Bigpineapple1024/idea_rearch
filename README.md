# 研究项目工作区

从论文阅读、问题发现、实验验证到论文写作的可追溯工作区。

**当前状态（2026-10-25）：**视觉 token 剪枝在候选 1 的第 ⑫ 步，计划目标在 64 档达到（DEC-0033）：POPE 上 A′+B+V − A 在 64、128、192 三档都显著（+15.84、+5.21、+2.29，按图像重抽的区间 [13.88, 17.82]、[3.94, 6.47]、[1.38, 3.22]）；VQAv2 test2 上按题目句式路由 − A 在 64 档 +1.22 [+0.58, +1.82]、128 档 +0.72 [+0.29, +1.15]（EXP-0020）；MME-P 的差异不显著。EXP-0021 已完成；EXP-0022（GQA testdev-balanced，预注册的第三个 benchmark，DEC-0034）运行中；候选 1 的运行上限为 68 次（DEC-0034）。时序预测在第 ⑦ 步后暂停（DEC-0020）。baseline 按用户偏好选近三年、有源码、效果比当前最好的方法差一点的工作（DEC-0012）。最新结果见[最新实验分析报告](analysis/最新实验分析报告.md)。原状态行（2026-10-24）：视觉 token 剪枝在研究流程第 ⑧ 步第 3 轮（压误报，EXP-0013），之后按流程进入第 ⑨～⑫ 步（DEC-0023）；时序预测在第 ⑦ 步后暂停（DEC-0020）。baseline 按用户偏好选近三年、有源码、效果比当前最好的方法差一点的工作（DEC-0012）。最新结果见[最新实验分析报告](analysis/最新实验分析报告.md)。

- **视觉 token 剪枝**：主 baseline 为 FastV（LLaVA-1.5-7B），VisionZip、SparseVLM 为对照。GPU 复现 EXP-0002 完成；第 ④～⑦ 步的诊断见 EXP-0006～EXP-0010；第 ⑧ 步第 2 轮（EXP-0011）发现输入层的物体名 embedding 余弦能保住被问的小物体，同一预算下 POPE F1 从 64.66 升到 83.63，代价是误报增加（F30）。此后第 ⑧ 步第 3 轮压误报（EXP-0013），第 ⑨～⑫ 步依次试了 IDF 查询、可视性加权与按题目句式路由（EXP-0014～EXP-0022），第 ⑫ 步的计划目标在 64 档达到（DEC-0033），见 [paper-plan](analysis/paper-plan.md)（2026-10-25 跑偏检查补注）。
- **时序预测**：主 baseline 为 iTransformer（备选 CycleNet），DUET、TQNet、WPMixer 为对照。iTransformer 复现（EXP-0004，记为偏差，DEC-0018）、对照重跑（EXP-0005）、验证期预测（EXP-0009）与候选 1 的 CPU 诊断（EXP-0012）已完成；方向暂停中。

九篇论文（PAPER-0001～0009）已有原文笔记，其中七篇（PAPER-0001～0007）另有源码核验补充（PAPER-0008 CLASP、PAPER-0009 VIP-Router 于 2026-10-25 补读原文，代码未下载）（原写“六篇”，没有算上 PAPER-0007 iTransformer；2026-10-25 跑偏检查补注），见[论文笔记](analysis/论文笔记/README.md)、[六篇总结与趋势](analysis/六篇论文总结与视觉Token压缩趋势.md)、[视觉 Mode C](analysis/视觉Token压缩_ModeC.md)。

流程规范见[研究流程 v2.1](analysis/研究流程.md)，入口见[文档索引](analysis/index.md)。在实验室电脑上开新会话先读[交接说明](analysis/实验室会话交接.md)；实验室电脑上的所有文件只放在 `D:\6CJNM007`。

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

技能入口见[技能说明](.claude/README.md)。原文笔记按 analyze-paper/deep-read 指引写成，PDF 用 pdftotext/PyMuPDF 提取核对。2026-10-24 起另有 MinerU 解析的七篇论文全文（flash 档，只有文本，见 [literature/processed](literature/processed/README.md)），用于回查笔记中“待回查”的条目（原写“尚未使用 MinerU”；2026-10-25 跑偏检查补注）。缺少材料或运行证据时标注待补充，不生成虚构结果。

## 分支与复现约定

main 维护管理文档与论文，code/<方向或实验编号> 隔离实验开发。Git 分支覆盖整个仓库，并不自动限制目录。实验分支中的记录应随工作保留，再将核验后的报告同步到 main，并保留实验代码的确切 commit。

项目结构与共享技能以 main 为基础维护；实验分支为 code/token-pruning（视觉 token 剪枝）和 code/ts-forecasting（时序预测）。由 Claude Code 提交并推送。云任务已隔离，使用已有检出，除非明确要求，不另建 worktree。

外部材料记录来源 URL、版本、许可、获取方式和校验和。运行产物记录本地位置及可恢复备份；仅有本机路径不足以长期复现。依赖、环境与执行命令见 [code/README](code/README.md) 及各方向目录的 README。
