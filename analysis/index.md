# 项目文档索引

状态（2026-10-25，DEC-0012～0026）：视觉 token 剪枝方向完成第 ⑧ 步第 3 轮（EXP-0013，压误报，没有修正通过），第 ⑨～⑫ 步：IDF 加权的题目查询在 POPE 上提升、MME 上未提升（EXP-0014，DEC-0026），回到第 ⑨ 步处理词的可视性；时序预测方向在第 ⑦ 步后暂停（DEC-0020、DEC-0021）。七篇论文已有原文笔记与源码核验补充。实验室电脑是原生 Windows，文件只放在 D:\6CJNM007。流程规范为[研究流程 v2.1](研究流程.md)。新增、移动或归档文档时同步本索引。（此前这一行停在 DEC-0014 时的状态，2026-10-24 的跑偏检查中更新。）

## 当前工作入口

| 内容 | 入口 |
| --- | --- |
| **流程规范（唯一）** | [研究流程 v2.1](研究流程.md) |
| 实验室电脑新会话的交接 | [实验室会话交接](实验室会话交接.md) |
| 视觉 token 剪枝：六篇总结与趋势（原文） | [六篇论文总结与视觉Token压缩趋势](六篇论文总结与视觉Token压缩趋势.md) |
| 视觉 token 剪枝：三篇共性、差异与 baseline（FastV） | [所有论文在baseline骨架的改进内容](专题分析/所有论文在baseline骨架的改进内容.md) |
| 问题—现象—机制矩阵（合并版，回流 EXP-0002） | [MODE_C-0002](MODE_C-0002_视觉token剪枝.md)；旧版保留：[视觉Token压缩_ModeC](视觉Token压缩_ModeC.md)（原文）、[MODE_C-0001](MODE_C-0001_视觉token剪枝.md)（源码与探针） |
| 论文全文（MinerU 解析，只有文本） | [literature/processed](../literature/processed/README.md) |
| 候选方案（待 baseline 复现与组合实验后重新排序） | [IDEA-0001](IDEA-0001_编码器侧压缩的位置编号.md) |
| **视觉方向第 ⑨～⑫ 步：IDF 加权的题目查询（DEC-0025）** | [IDEA-0002](IDEA-0002_题目词按信息量加权.md)（迁移方案）；[EXP-0014](实验报告/EXP-0014_record.md)、[SQZ-EXP-0014](实验报告/EXP-0014_squeeze.md)（A / A′ / A′+B / A+B：POPE 提升，MME 未提升，F34、F35） |
| **视觉方向第 ⑨～⑫ 步第三轮：按题目句式路由（DEC-0031）** | [EXP-0018](实验报告/EXP-0018_record.md)、[SQZ-EXP-0018](实验报告/EXP-0018_squeeze.md)（VQAv2：差异不显著，yes/no 受益、other 受损，F38～F40）；[IDEA-0003](IDEA-0003_按题目句式路由.md)；[EXP-0020](实验报告/EXP-0020_record.md)（新数据上的预注册确认） |
| baseline 复现代码与步骤 | [code/token_pruning](../code/token_pruning/README.md) |
| 视觉方向 GPU 复现记录与深挖 | [EXP-0002](实验报告/EXP-0002_record.md)、[SQZ-EXP-0002](实验报告/EXP-0002_squeeze.md) |
| 视觉方向第 ⑤ 步：深挖意外现象 | [1a-4b_FastV不如随机选择](1a-4b_FastV不如随机选择.md)；[EXP-0006](实验报告/EXP-0006_record.md)、[SQZ-EXP-0006](实验报告/EXP-0006_squeeze.md)；[EXP-0008](实验报告/EXP-0008_record.md)、[SQZ-EXP-0008](实验报告/EXP-0008_squeeze.md)；[EXP-0007](实验报告/EXP-0007_record.md)、[SQZ-EXP-0007](实验报告/EXP-0007_squeeze.md)（层分布与选择规则）；[EXP-0010](实验报告/EXP-0010_record.md)、[SQZ-EXP-0010](实验报告/EXP-0010_squeeze.md)（覆盖对照） |
| **第 ⑦ 步：改进方向排序**（视觉：用户已选候选 1，DEC-0016；时序：候选 1 的 CPU 诊断完成，按规则升为高，时序方向暂停中，DEC-0019～DEC-0021） | [第7步_改进方向排序](第7步_改进方向排序.md)；[EXP-0012](实验报告/EXP-0012_record.md)、[SQZ-EXP-0012](实验报告/EXP-0012_squeeze.md)（时序候选 1 的诊断，F-TS15） |
| **视觉方向第 ⑧ 步：深挖到算子级（候选 1，第 2 轮 DEC-0022；第 3 轮误报 DEC-0024，完成）** | [1a-4b_选择与题目无关漏掉被问的小物体](1a-4b_选择与题目无关漏掉被问的小物体.md)（1A4B-0003；COCO 物体框分析在 code/output/EXP-0002/deepdive3，RePOPE 重算在 deepdive4）；[EXP-0011](实验报告/EXP-0011_record.md)、[SQZ-EXP-0011](实验报告/EXP-0011_squeeze.md)（D2、D2b、D3：输入层物体名余弦让 POPE +18.97，F28～F30）；[EXP-0013](实验报告/EXP-0013_record.md)、[SQZ-EXP-0013](实验报告/EXP-0013_squeeze.md)（第 3 轮：误报伴随区分力下降，同预算修正都未通过，F31～F33） |
| 第 ⑥ 步：骨架全景图 × 改进内容交叉分析（视觉与时序） | [baseline骨架的全景图](专题分析/baseline骨架的全景图.md)、[所有论文在baseline骨架的改进内容](专题分析/所有论文在baseline骨架的改进内容.md) |
| 时序预测：baseline（iTransformer，DEC-0012）、三篇对照对比与组合实验候选 | [时序预测_DUET_TQNet_WPMixer对比](专题分析/时序预测_DUET_TQNet_WPMixer对比.md) |
| 时序方向 GPU 复现与对照重评 | [EXP-0004](实验报告/EXP-0004_record.md)、[SQZ-EXP-0004](实验报告/EXP-0004_squeeze.md)（复现记为偏差，DEC-0018）；[EXP-0005](实验报告/EXP-0005_record.md)、[SQZ-EXP-0005](实验报告/EXP-0005_squeeze.md) |
| 时序方向第 ⑤ 步：深挖意外现象 | [1a-4b_四种架构在同一批窗口上出错](1a-4b_四种架构在同一批窗口上出错.md)；[EXP-0009](实验报告/EXP-0009_record.md)、[SQZ-EXP-0009](实验报告/EXP-0009_squeeze.md)（F-TS10 不跨数据集；F-TS13、F-TS14） |
| 时序方向的问题—现象—机制矩阵 | [MODE_C-0003](MODE_C-0003_时序预测.md) |
| 每篇论文的原文笔记与源码核验补充 | [论文笔记](论文笔记/README.md) |
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
| [六篇总结与趋势](六篇论文总结与视觉Token压缩趋势.md) | 六篇摘要、方法纠正、数字解释（文中的 VisionZip 开发定位已被 DEC-0014 取代） |
| [视觉Mode C](视觉Token压缩_ModeC.md) | P/F/M/E矩阵及候选连接点 |
| [PAPER-0001_DUET](论文笔记/PAPER-0001_DUET.md) | 双聚类时序预测证据包 |
| [PAPER-0002_TQNet](论文笔记/PAPER-0002_TQNet.md) | 周期查询证据包 |
| [PAPER-0003_WPMixer](论文笔记/PAPER-0003_WPMixer.md) | 多分辨率混合证据包 |
| [PAPER-0004_FastV](论文笔记/PAPER-0004_FastV.md) | 注意力剪枝与实现差异 |
| [PAPER-0005_SparseVLM](论文笔记/PAPER-0005_SparseVLM.md) | 文本raters、回收与阶段预算 |
| [PAPER-0006_VisionZip](论文笔记/PAPER-0006_VisionZip.md) | dominant/contextual压缩及边界 |
| [PAPER-0007_iTransformer](论文笔记/PAPER-0007_iTransformer.md) | 变量 token 与倒置 Transformer；ETT 上的报告值与限制 |
| [analysis/论文笔记/PAPER-0007_iTransformer_源码核验.md](../analysis/论文笔记/PAPER-0007_iTransformer_源码核验.md) | 源码核验补充：iTransformer（日历特征 token、脚本超参数与论文网格的出入） |
| [.claude/README.md](../.claude/README.md) | Claude Code 运行时支持 |
| [AGENTS.md](../AGENTS.md) | 研究工作区协作规则 |
| [analysis/archied/README.md](../analysis/archied/README.md) | 归档方案 |
| [analysis/baseline候选筛选.md](baseline候选筛选.md) | 3090 24GB 下的三篇候选、源码与许可核验、引用量待核实项 |
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
| [analysis/MODE_C-0001_视觉token剪枝.md](../analysis/MODE_C-0001_视觉token剪枝.md) | MODE_C_PACKAGE：多模态大模型视觉 token 剪枝（历史版本，已由 MODE_C-0002 取代） |
| [analysis/MODE_C-0002_视觉token剪枝.md](../analysis/MODE_C-0002_视觉token剪枝.md) | MODE_C_PACKAGE：视觉 token 剪枝合并版（原文版 + 源码版 + EXP-0001/0002 回流） |
| [analysis/IDEA-0002_题目词按信息量加权.md](../analysis/IDEA-0002_题目词按信息量加权.md) | 跨领域迁移方案：用 IDF 给题目加权，构造图像 token 的查询（第 ⑨～⑪ 步，领域 B 信息检索，领域 C BERTScore / SIF） |
| [analysis/IDEA-0001_编码器侧压缩的位置编号.md](../analysis/IDEA-0001_编码器侧压缩的位置编号.md) | IDEA_PACKAGE：编码器侧压缩的位置编号（融合路线候选，deferred） |
| [analysis/论文笔记/PAPER-0004_FastV_源码核验.md](../analysis/论文笔记/PAPER-0004_FastV_源码核验.md) | 源码核验补充：FastV |
| [analysis/论文笔记/PAPER-0005_SparseVLM_源码核验.md](../analysis/论文笔记/PAPER-0005_SparseVLM_源码核验.md) | 源码核验补充：SparseVLM / SparseVLM+ |
| [analysis/论文笔记/PAPER-0006_VisionZip_源码核验.md](../analysis/论文笔记/PAPER-0006_VisionZip_源码核验.md) | 源码核验补充：VisionZip |
| [analysis/实验报告/EXP-0001_record.md](../analysis/实验报告/EXP-0001_record.md) | EXPERIMENT_RECORD：三个仓库的代码级核验（CPU） |
| [analysis/实验报告/EXP-0001_squeeze.md](../analysis/实验报告/EXP-0001_squeeze.md) | SQUEEZE_REPORT：EXP-0001 的跨平台复验与预算口径 |
| [analysis/实验报告/EXP-0002_record.md](../analysis/实验报告/EXP-0002_record.md) | EXPERIMENT_RECORD：视觉 token 剪枝 GPU 复现（LLaVA-1.5-7B，POPE、MME，49 次运行） |
| [analysis/实验报告/EXP-0002_squeeze.md](../analysis/实验报告/EXP-0002_squeeze.md) | SQUEEZE_REPORT：EXP-0002（随机优于 FastV、“no”偏置、SparseVLM 实测 token） |
| [analysis/实验报告/EXP-0005_record.md](../analysis/实验报告/EXP-0005_record.md) | EXPERIMENT_RECORD：DUET、TQNet、WPMixer 在统一协议下的 GPU 重跑（ETTh1）与 TFB 数据核对 |
| [analysis/实验报告/EXP-0005_squeeze.md](../analysis/实验报告/EXP-0005_squeeze.md) | SQUEEZE_REPORT：EXP-0005（四种架构在同一批窗口上出错、验证集分不出长预测的好坏种子） |
| [analysis/实验报告/EXP-0006_record.md](../analysis/实验报告/EXP-0006_record.md) | EXPERIMENT_RECORD：FastV 与随机选择保留了图像的哪些位置（F8 的机制诊断） |
| [analysis/实验报告/EXP-0006_squeeze.md](../analysis/实验报告/EXP-0006_squeeze.md) | SQUEEZE_REPORT：EXP-0006（FastV 的选择由位置主导、几乎不看问题，F15） |
| [analysis/实验报告/EXP-0007_record.md](../analysis/实验报告/EXP-0007_record.md) | EXPERIMENT_RECORD：把层分布与选择规则拆开（MODE_C-0002 的 C11） |
| [analysis/实验报告/EXP-0007_squeeze.md](../analysis/实验报告/EXP-0007_squeeze.md) | SQUEEZE_REPORT：EXP-0007（SparseVLM 的优势来自规则 F19；前多后少的分布没有收益 F20） |
| [analysis/实验报告/EXP-0008_record.md](../analysis/实验报告/EXP-0008_record.md) | EXPERIMENT_RECORD：去掉位置项后 FastV 的选择是否恢复（1A4B-0001 的 3a） |
| [analysis/实验报告/EXP-0008_squeeze.md](../analysis/实验报告/EXP-0008_squeeze.md) | SQUEEZE_REPORT：EXP-0008（与内容无关的位置曲线 F16；位置校正补回 88% 的差距 F17） |
| [analysis/实验报告/EXP-0010_record.md](../analysis/实验报告/EXP-0010_record.md) | EXPERIMENT_RECORD：位置校正的收益来自“去位置”还是“更分散” |
| [analysis/实验报告/EXP-0010_squeeze.md](../analysis/实验报告/EXP-0010_squeeze.md) | SQUEEZE_REPORT：EXP-0010（只强制覆盖就追平随机，去位置 + 覆盖超过随机，F24） |
| [analysis/实验报告/EXP-0011_record.md](../analysis/实验报告/EXP-0011_record.md) | EXPERIMENT_RECORD：第 1 层的题目注意力能否保住被问的小物体（第 ⑧ 步的 D2、D2b、D3） |
| [analysis/实验报告/EXP-0011_squeeze.md](../analysis/实验报告/EXP-0011_squeeze.md) | SQUEEZE_REPORT：EXP-0011（第 1 层注意力不看题目 F28；输入层物体名余弦 F29；误报增加 F30） |
| [analysis/实验报告/EXP-0012_record.md](../analysis/实验报告/EXP-0012_record.md) | EXPERIMENT_RECORD：96 步输入下未来窗口的水平能否用简单统计量预测（时序候选 1 的 CPU 诊断） |
| [analysis/实验报告/EXP-0012_squeeze.md](../analysis/实验报告/EXP-0012_squeeze.md) | SQUEEZE_REPORT：EXP-0012（收益与风险都来自向训练期均值回归，F-TS15） |
| [analysis/实验报告/EXP-0013_record.md](../analysis/实验报告/EXP-0013_record.md) | EXPERIMENT_RECORD：题目相关的选择为什么多出误报，能否在同一预算下压下去（第 ⑧ 步第 3 轮） |
| [analysis/实验报告/EXP-0013_squeeze.md](../analysis/实验报告/EXP-0013_squeeze.md) | SQUEEZE_REPORT：EXP-0013（选择丢掉“没有”的证据 F31；F1 下最优阈值偏向 yes F32；对比打分 F33） |
| [analysis/实验报告/EXP-0014_record.md](../analysis/实验报告/EXP-0014_record.md) | EXPERIMENT_RECORD：IDF 加权的题目查询，第 ⑫ 步的 A / A′ / A′+B / A+B（POPE、MME） |
| [analysis/实验报告/EXP-0014_squeeze.md](../analysis/实验报告/EXP-0014_squeeze.md) | SQUEEZE_REPORT：EXP-0014（POPE 上复现物体名查询 F34；MME 名字类受损 F35） |
| [analysis/审计记录_第12步_2026-10-25.md](../analysis/审计记录_第12步_2026-10-25.md) | 第 ⑫ 步结论的独立审计与修正（DEC-0030） |
| [analysis/IDEA-0003_按题目句式路由.md](../analysis/IDEA-0003_按题目句式路由.md) | IDEA_PACKAGE：按题目句式路由（助动词或 how many 开头的题用题目查询，其余用强对照；P11） |
| [analysis/实验报告/EXP-0020_record.md](../analysis/实验报告/EXP-0020_record.md) | EXPERIMENT_RECORD：按题目句式路由在新抽的 VQAv2 test2 上的预注册确认（第 ⑫ 步第六轮） |
| [analysis/实验报告/EXP-0021_record.md](../analysis/实验报告/EXP-0021_record.md) | EXPERIMENT_RECORD：VQAv2 test2 的补全（192 档路由；128 档参照） |
| [analysis/实验报告/EXP-0018_squeeze.md](../analysis/实验报告/EXP-0018_squeeze.md) | SQUEEZE_REPORT：EXP-0018（答案类型上效应相反 F38；损失集中在问点名物体属性的题 F39；一般题目上 FastV 一族低于 VisionZip F40） |
| [analysis/实验报告/EXP-0019_record.md](../analysis/实验报告/EXP-0019_record.md) | EXPERIMENT_RECORD：审计要求的补充（A+B 的对照；最终方法的区分力） |
| [analysis/实验报告/EXP-0018_record.md](../analysis/实验报告/EXP-0018_record.md) | EXPERIMENT_RECORD：VQAv2 验证集子集上的一般题目检验（第 ⑫ 步第五轮） |
| [analysis/实验报告/EXP-0017_record.md](../analysis/实验报告/EXP-0017_record.md) | EXPERIMENT_RECORD：192 档复验（第 ⑫ 步第四轮） |
| [analysis/实验报告/EXP-0015-0016_squeeze.md](../analysis/实验报告/EXP-0015-0016_squeeze.md) | SQUEEZE_REPORT：EXP-0015、EXP-0016（收益主要来自切分；POPE 提升随预算缩小 F37） |
| [analysis/实验报告/EXP-0016_record.md](../analysis/实验报告/EXP-0016_record.md) | EXPERIMENT_RECORD：可视性与切分的拆分，以及 128 档复验（第 ⑫ 步第三轮） |
| [analysis/实验报告/EXP-0015_record.md](../analysis/实验报告/EXP-0015_record.md) | EXPERIMENT_RECORD：按词的可视性分配名额的题目查询（第 ⑨～⑫ 步第二轮，P10） |
| [analysis/1a-4b_FastV不如随机选择.md](../analysis/1a-4b_FastV不如随机选择.md) | 1a–4b 机制分析：同预算下 FastV 为什么不如随机选择（第 1 轮，到 2b） |
| [analysis/1a-4b_选择与题目无关漏掉被问的小物体.md](../analysis/1a-4b_选择与题目无关漏掉被问的小物体.md) | 1a–4b 机制分析（第 ⑧ 步，候选 1）：选择与题目无关时，被问的小物体为什么被删掉（第 1 轮，到 2b；COCO 物体框） |
| [analysis/1a-4b_四种架构在同一批窗口上出错.md](../analysis/1a-4b_四种架构在同一批窗口上出错.md) | 1a–4b 机制分析：时序四种架构为什么在同一批窗口上出错（第 1 轮，到 2b；误差的水平/形状拆分） |
| [analysis/实验报告/EXP-0009_record.md](../analysis/实验报告/EXP-0009_record.md) | EXPERIMENT_RECORD：iTransformer 检查点在验证期上的预测（不训练；计划） |
| [code/token_pruning/README.md](../code/token_pruning/README.md) | 视觉 token 剪枝 baseline 复现与代码级核验 |
| [analysis/论文笔记/PAPER-0001_DUET_源码核验.md](../analysis/论文笔记/PAPER-0001_DUET_源码核验.md) | 源码核验补充：DUET |
| [analysis/论文笔记/PAPER-0002_TQNet_源码核验.md](../analysis/论文笔记/PAPER-0002_TQNet_源码核验.md) | 源码核验补充：TQNet |
| [analysis/论文笔记/PAPER-0003_WPMixer_源码核验.md](../analysis/论文笔记/PAPER-0003_WPMixer_源码核验.md) | 源码核验补充：WPMixer |
| [analysis/专题分析/时序预测_DUET_TQNet_WPMixer对比.md](../analysis/专题分析/时序预测_DUET_TQNet_WPMixer对比.md) | 时序预测三篇论文对比：DUET、TQNet、WPMixer |
| [analysis/实验报告/EXP-0003_record.md](../analysis/实验报告/EXP-0003_record.md) | EXPERIMENT_RECORD：DUET/TQNet/WPMixer 在 ETTh1 上的 CPU 复现 |
| [analysis/实验报告/EXP-0003_squeeze.md](../analysis/实验报告/EXP-0003_squeeze.md) | SQUEEZE_REPORT：EXP-0003（drop_last 去掉最难的末尾窗口） |
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
| [analysis/研究流程.md](../analysis/研究流程.md) | 研究流程 v2.1：从论文到 idea，再从 idea 到实现（唯一流程规范） |
| [analysis/论文笔记/README.md](../analysis/论文笔记/README.md) | 论文笔记 |
| [analysis/论文素材库.md](../analysis/论文素材库.md) | 论文素材库 |
| [CLAUDE.md](../CLAUDE.md) | Claude Code 项目入口 |
| [code/README.md](../code/README.md) | 实验代码 |
| [literature/README.md](../literature/README.md) | 论文解析材料 |
| [literature/processed/README.md](../literature/processed/README.md) | MinerU 解析全文（flash 档，只有文本）与校验和 |
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

- 根目录 .gitignore 定义外部资源、实验产物和本地运行时文件的忽略规则；.gitattributes 固定 *.sh 为 LF 换行（Windows 上 CRLF 的脚本在 Git Bash 中无法运行）。
- [code/env/windows_env.sh](../code/env/windows_env.sh)：实验室电脑（原生 Windows）每条命令前加载的环境脚本。
- literature/processed/.gitkeep 仅保留解析目录；.claude/skills/ 已包含实际技能文件。
- [.claude/skills/manifest.json](../.claude/skills/manifest.json) 保存上传包来源、协议版本与六份原文件的校验和。
- data/、checkpoints/、papers/、code/output/ 不跟踪；在对应笔记或实验记录中登记来源、校验和及可恢复位置。
- [论文来源清单](论文笔记/sources-20261023.json) 保存 PDF 来源、版本、校验和与源码 commit；PDF 在被忽略的 papers/，临时源码核验副本未纳入项目。
- [引用量](论文笔记/citations-20261023.json)：8 篇论文（含 CycleNet）的 Semantic Scholar 引用量原始返回，2026-10-23 查询。
- 外部代码（LLaVA、FastV、SparseVLM、VisionZip、iTransformer、DUET、TQNet、WPMixer 等）只登记来源与 commit（见 [source/README](../source/README.md)），不纳入仓库。真实实验数据：EXP-0002 的 GPU 运行在 code/output/EXP-0002/（不跟踪；每个运行有 SHA256SUMS，备份在 D:\6CJNM007\backup\EXP-0002\）；EXP-0003 的 CPU 运行原始日志未保留，关键数字已抄录。尚无归档方案以外的归档。
