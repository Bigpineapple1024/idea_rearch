# 项目文档索引

状态（2026-10-31 上午 11 点后，补到 DEC-0074）：EXP-0038 完成：编号相同时第 0 层借看没有让先剪后送变好（peekc − DivPrune-64 −0.18 / −0.33）；Q2 的 +1.18 / +0.90 是补回 nopeekc 第 0 层稀疏编号造成的偏“是”（F98～F100）；IDEA-0009 停在开发集结论，确认与下一步交用户决定（DEC-0074）。前一状态行（2026-10-31 上午，补到 DEC-0073）：下一轮 IDEA-0009（编号相同时的第 0 层借看，MODE_C-0002 回流 27 的 C34）与开发实验 EXP-0038 计划 v1（6 次）已写定，排队等开门。前一状态行（2026-10-31 上午，补到 DEC-0072）：查新：剪枝后的编号与定位已有前人工作（GAP、Layer-Aware），不做 RefCOCO 诊断；第 0 层只算 K / V 的借看没有找到直接先例；PAPER-0010～0020 笔记的第二人复核完成（3560 处核对、64 处改正，DEC-0072）。前一状态行（2026-10-31 上午，补到 DEC-0071）：用户决定：IDEA-0008 停在现有结论、作为分析；下一步转向定位类任务（先查新，下载另行许可）；PAPER-0010～0020 笔记的第二人复核运行中（DEC-0071）。前一状态行（2026-10-31 8 点 45 分，补到 DEC-0070）：EXP-0037 完成：16 档时借看（留 13）比 DivPrune-16 差（−3.12 / −1.32），增益不随预算变小而变大（F97，DEC-0070）。IDEA-0008 用了 29 / 30 次：只在 64 档、GQA 一类题上稳定为正，去留交用户决定。前一状态行（2026-10-31 8 点前，补到 DEC-0069）：EXP-0036 完成：IDEA-0008 在 MME、POPE 上比 DivPrune-64 低 1.5～1.8，V* 上 +3.66 没有检出；POPE 上借看把判断推向“否”；原始编号在 MME position、V* 相对位置上没有检出作用（F94～F96，DEC-0069）。IDEA-0008 只在 GQA 一类题上稳定为正，去留交用户决定。EXP-0037（16 档）07:44 起运行。前一状态行（2026-10-31 6 点半，补到 DEC-0067）：EXP-0035（IDEA-0008 的确认，新抽的数据）完成：peek − DivPrune-64 GQA +1.53、VQAv2 +0.28，“部分成立”，按计划交用户决定是否作为方法；借看的作用（+2.17 / +2.37）、偏“是”的纠正与原始编号帮位置题（GQA 位置题 +11.85）都在新数据上重现（F91～F93，DEC-0067）；EXP-0036（MME、V*、POPE）06:12 起运行。前一状态行（2026-10-31 4 点，补到 DEC-0065）：EXP-0035（确认）03:09 起运行（代码 88cb1cb；计划 v1.1 在看到结果之前加了两项二级分析）；SQZ-EXP-0034 完成：原始位置编号、第 0 层看不到被剪的会偏答“是”，借看去掉它；GQA 位置题上连续编号低 18.8、原始编号补回 13.9（F88～F90）；DivPrune、CDPruner、MMTok、VScan 的官方代码已按许可克隆到 source/（DEC-0063）。前一状态行（2026-10-31 3 点，补到 DEC-0065）：EXP-0034 完成（6 次）：peek（第 0 层借看全部、第 1 层起留 61 个，FLOPs 约等于 DivPrune-64）在 GQA 上 +2.34、VQAv2 上 +0.18，按计划读作“部分”，送确认 EXP-0035；第 0 层看过被剪的 patch 在两个集上都约 +2（是非类题），GQA 属性、二选一题的增益来自原始位置编号（F84～F87，DEC-0065）。前一状态行（原写“2026-10-31 3 点，补到 DEC-0064”，“3 点”有误，实为 1 点 18 分前后）：下一轮 IDEA-0008（第 0 层借看全部 patch，按 FLOPs 对齐多留；MODE_C-0002 回流 25 的 C29）与开发实验 EXP-0034计划 v1（VQAv2 dev2、GQA val400，6 次）已写定，排队等开门；回流 25 已写。前一状态行（2026-10-31 1 点，补到 DEC-0063）：EXP-0033 完成（12 次）：四个组合都没能稳定超过 DivPrune-64；不剪枝 77.91 / 70.08，GQA 上的缺口集中在属性、二选一题，K = 1 的多样性在这两类上补回约一半（F77～F83，DEC-0062）；SQZ-EXP-0033 完成；代码分支的推迟修复已提交（9dae438）；PAPER-0010～0020 的笔记已提交（第二人复核待用户醒来后做，DEC-0063）。前一状态行（2026-10-30 20 点半，补到 DEC-0061）：EXP-0033（第 ④ 步第二轮的探索性组合，12 次）20:08:58 起运行（代码 0eed416）；SQZ-EXP-0032 完成（F75、F76：DivPrune 与 VisionZip 选的 patch 只重合约四分之一，DivPrune 铺得更开）。前一状态行（2026-10-30 20 点，补到 DEC-0061）：EXP-0032 完成：本流程中的 DivPrune-64 明显强于 VisionZip-64（开发集 +2.20 / +1.63），一层全看对多样性选择没有帮助；下一轮以 DivPrune-64 为底座（DEC-0061，F72～F74）。EXP-0033 开门前的阅读写在其记录。前一状态行（2026-10-30 19 点半，补到 DEC-0059）：EXP-0032 19:05:17 起运行（代码 e1daaab；开门前的阅读见其记录）。SQZ-EXP-0031 完成（F69～F71：K = 1 路由 + 覆盖 16 在 GQA 的两组新数据上都比 VisionZip-64 高约 2 分；两种排序只重合约三分之一、准确率分不开），MODE_C-0002 回流 24。前一状态行（2026-10-30 19 点后，补到 DEC-0059）：EXP-0031（IDEA-0007 的确认）完成：Q1、Q2 部分成立（只在 GQA val400b 上），Q3 不成立，IDEA-0007 新加的编码器排序不作为方法的贡献；拼接的二级分析 S1、S2 部分成立（DEC-0059，F66～F68）。新会话按用户要求完整阅读了仓库（DEC-0058）；EXP-0032 开门前按 DEC-0055 读改动部分，其余文档问题在 EXP-0032 运行期间修、EXP-0033 开门前修完。前一状态行（2026-10-30 17:35，补到 DEC-0057）：用户说“恢复”，EXP-0031 17:29:54 起从 R07 接着跑（chain_0031d），EXP-0032、EXP-0033 的排队在后面等。前一状态行（2026-10-29 14:25，补到 DEC-0057）：**按用户要求暂停**。EXP-0031 的 R01～R06 完成并封存，R07～R09、EXP-0032、EXP-0033 没有开始；恢复步骤见 EXP-0031 记录的“暂停”与 `D:\6CJNM007\tmp\RESUME_20261029.md`。EXP-0030 的深挖 SQZ-EXP-0030 已写（F63～F65）。前一状态行（2026-10-29 12:55，补到 DEC-0057）：EXP-0031（IDEA-0007 的确认）12:50:40 开始运行（代码 889f287；第一次推送被 GitHub 拒绝，重试成功后由续跑脚本 chain_0031c 接着跑，见 EXP-0031 记录）。前一状态行（2026-10-29 12:45（原写“下午 1 点”，按提交时间更正），补到 DEC-0057）：EXP-0030（IDEA-0007 的开发集运行）判读“有希望”：所选 k1qcls_c16 比 VisionZip-64 +1.18 / +1.38（两集区间都在 0 以上），增益主要在被路由的题，没有达到“提升大”（DEC-0057，F61、F62）；进入确认 EXP-0031，开门前按 DEC-0055 只读改动部分。前一状态行（2026-10-29 下午，补到 DEC-0056）：EXP-0033 计划 v1.1 写定（X1 的配置名、代码位置与实现核对的分工；组合不变），代码在 tmp 自检、空跑完毕，排队脚本 chain_0033 排在 chain_0032 之后；经用户许可下载的 11 篇论文 PDF 已在本地解析并登记为 PAPER-0010～PAPER-0020（笔记未写）。前一状态行（2026-10-29 中午，补到 DEC-0056）：第 ④ 步第二轮的组合实验 EXP-0033 计划 v1 写定，排在 EXP-0032 之后（DEC-0056）。前一状态行（2026-10-29 11 点，补到 DEC-0055）：完整阅读已修完（跑偏检查_2026-10-26），EXP-0030 于 10-29 11:09:59 开始运行；之后开门前只读改动部分（DEC-0055）。前一状态行（2026-10-26 13 点半，补到 DEC-0054）：EXP-0029（IDEA-0006 的确认）Q1 不成立，IDEA-0006 结束（DEC-0054，F59、F60）；用户新规则：每开启下一步之前完整阅读仓库并修好问题，需要时可以启动多代理工作流（DEC-0053，研究流程 v2.2 第 12 条），EXP-0030 等读完再开始。之前的状态行见文末附录（2026-10-26 移出，原文未改）。

## 当前工作入口

| 内容 | 入口 |
| --- | --- |
| **流程规范（唯一）** | [研究流程 v2.2](研究流程.md)（原写 v2.1；2026-10-26 补注） |
| 实验室电脑新会话的交接 | [实验室会话交接](实验室会话交接.md) |
| 视觉 token 剪枝：六篇总结与趋势（原文） | [六篇论文总结与视觉Token压缩趋势](六篇论文总结与视觉Token压缩趋势.md) |
| 视觉 token 剪枝：三篇共性、差异与 baseline（FastV） | [所有论文在baseline骨架的改进内容](专题分析/所有论文在baseline骨架的改进内容.md) |
| 问题—现象—机制矩阵（合并版，回流 EXP-0002；之后回流 1～13，2026-10-25 跑偏检查补注；2026-10-29 补注：现为回流 1～23；2026-10-31 补注：现为回流 1～27） | [MODE_C-0002](MODE_C-0002_视觉token剪枝.md)；旧版保留：[视觉Token压缩_ModeC](视觉Token压缩_ModeC.md)（原文）、[MODE_C-0001](MODE_C-0001_视觉token剪枝.md)（源码与探针） |
| 论文全文（MinerU 解析，只有文本） | [literature/processed](../literature/processed/README.md) |
| 候选方案（待 baseline 复现与组合实验后重新排序；2026-10-26 补注：这是最早的候选，之后的方案见下面第 ⑨～⑫ 步各轮的 IDEA-0002～IDEA-0007） | [IDEA-0001](IDEA-0001_编码器侧压缩的位置编号.md) |
| **视觉方向第 ⑨～⑫ 步：IDF 加权的题目查询（DEC-0025）** | [IDEA-0002](IDEA-0002_题目词按信息量加权.md)（迁移方案）；[EXP-0014](实验报告/EXP-0014_record.md)、[SQZ-EXP-0014](实验报告/EXP-0014_squeeze.md)（A / A′ / A′+B / A+B：POPE 提升，MME 未提升，F34、F35） |
| **视觉方向第 ⑨～⑫ 步第三轮：按题目句式路由（DEC-0031）** | [EXP-0018](实验报告/EXP-0018_record.md)、[SQZ-EXP-0018](实验报告/EXP-0018_squeeze.md)（VQAv2：差异不显著，yes/no 受益、other 受损，F38～F40）；[IDEA-0003](IDEA-0003_按题目句式路由.md)；[EXP-0020](实验报告/EXP-0020_record.md)（新数据上的预注册确认：64、128 档显著高于强对照，计划目标在 64 档达到，DEC-0033）、[SQZ-EXP-0020](实验报告/EXP-0020_squeeze.md)（F42、F43） |
| **视觉方向第 ⑨～⑫ 步第四轮：预算内放大（DEC-0035）** | [EXP-0023](实验报告/EXP-0023_record.md)（CPU 诊断：放大法满足“有希望”，F44～F46）；[IDEA-0004](IDEA-0004_预算内放大.md)（新模块）；[EXP-0024](实验报告/EXP-0024_record.md)（开发集运行：判读“不像”，DEC-0039）、[SQZ-EXP-0024](实验报告/EXP-0024_squeeze.md)（F49～F51：放大不成立；作为对照的覆盖 16 与 K = 1 有正的探索性结果）；[EXP-0025](实验报告/EXP-0025_record.md)（看细节的检验：T1～T3 不成立，IDEA-0004 结束，DEC-0041；F52） |
| **视觉方向第 ⑨～⑫ 步第五轮：定位后覆盖（DEC-0040）** | [IDEA-0005](IDEA-0005_定位后覆盖.md)（来自 MODE_C-0002 回流 17 的 C22，K = 1 作为变体）；[EXP-0026](实验报告/EXP-0026_record.md)（开发集运行：判读“有希望”，证据偏弱，DEC-0042；F53、F54）；[EXP-0027](实验报告/EXP-0027_record.md)（确认，预注册：Q1 部分成立、Q2 成立，增益主要来自 K = 1，覆盖不作为方法的贡献，DEC-0045；F55、F56） |
| **视觉方向第 ⑨～⑫ 步第六轮：强底座上的定位后覆盖（DEC-0044）** | [IDEA-0006](IDEA-0006_强底座上的定位后覆盖.md)（把 IDEA-0005 的定位后覆盖移到 VisionZip-64 上，只替换主导 token 的选择）；[EXP-0028](实验报告/EXP-0028_record.md)（开发集运行，计划 v1、v1.1：不围框对照；覆盖运行 R03–R12 因 dtype 错误失败，按 v1.2 以 R13–R22 重跑，DEC-0047；判读“有希望”，选每题覆盖 16，增益来自定位，DEC-0048；F57、F58；查新：VisionTrim 在 VisionZip 式选择上加文本引导的合并，方向不新）；[EXP-0029](实验报告/EXP-0029_record.md)（确认，计划 v1、v1.1，在 EXP-0028 出结果之前写定；10 次；DEC-0046；结果：Q1～Q3 不成立，IDEA-0006 结束，DEC-0054；F59、F60）、[SQZ-EXP-0029](实验报告/EXP-0029_squeeze.md)（开发集的增益为什么没有复现：题型内的效应变了，每题覆盖只改动约 9% 的题；TextVQA 的损失在读字题上）（原写“2026-10-26 09:04 起运行中”） |
| **视觉方向第 ⑨～⑫ 步第七轮：一层全看，再按编码器排序（DEC-0049）** | [IDEA-0007](IDEA-0007_一层全看再按编码器排序.md)（来自 EXP-0028 的记录内深挖：两种底座各有所长，按题型拼接 +1.96 / +1.17，探索性；层平均 63.5：第 0 层看全部 576 个，第 1 层起按视觉编码器的 CLS 排序留 47 个）；[EXP-0030](实验报告/EXP-0030_record.md)（开发集运行，计划 v1，6 次，排在 EXP-0029 之后；判读“有希望”，DEC-0057；深挖 [SQZ-EXP-0030](实验报告/EXP-0030_squeeze.md)）；[EXP-0031](实验报告/EXP-0031_record.md)（确认，计划 v1，在 EXP-0030 出结果之前写定，新抽的数据，9 次，只在 EXP-0030 判读“有希望”时运行；DEC-0051；结果：Q1、Q2 部分成立（只在 GQA val400b 上）、Q3 不成立，拼接 S1、S2 部分成立，DEC-0059；深挖 [SQZ-EXP-0031](实验报告/EXP-0031_squeeze.md)，F69～F71）；[EXP-0032](实验报告/EXP-0032_record.md)（机制检验：本流程中的 DivPrune-64 与一层全看对多样性选择的作用，4 次，DEC-0052；结果：DivPrune-64 明显强于 VisionZip-64，一层全看没有帮助，下一轮以 DivPrune-64 为底座，DEC-0061）；[查新补充](专题分析/查新补充_覆盖与多样性_2026-10-26.md)（覆盖 / 多样性一类与浅层剪枝） |
| **视觉方向第 ④ 步第二轮：探索性组合（制造意外，DEC-0056）** | [EXP-0033](实验报告/EXP-0033_record.md)（计划 v1、v1.1：四个组合，每个与两个单独方法比；X1 VisionZip + FastV（MODE_C-0002 的 C4）、X2 CLS 候选 × DivPrune、X3 句式路由 × DivPrune、X4 DivPrune + 上下文合并；12 次，排在 EXP-0032 之后；2026-10-30 20:08:58 起运行；已完成：没有组合稳定超过 DivPrune-64，DEC-0062） |
| **视觉方向第 ⑨～⑫ 步第八轮：第 0 层借看全部 patch，按 FLOPs 对齐多留（DEC-0064）** | [IDEA-0008](IDEA-0008_第0层借看全部patch.md)（MODE_C-0002 回流 25 的 C29：K = 1 时被剪的 patch 在第 0 层只作 key / value，第 1 层起留 DivPrune 的 61 个）；[EXP-0034](实验报告/EXP-0034_record.md)（开发集，计划 v1，6 次：peek、nopeek、peektext；已完成：Q1“部分”（GQA +2.34），Q2 两集都有作用，DEC-0065）；[EXP-0035](实验报告/EXP-0035_record.md)（确认，新抽的数据，10 次；已完成：Q1 部分成立（GQA +1.53），DEC-0067）；[EXP-0036](实验报告/EXP-0036_record.md)（MME、V*、POPE，9 次） |
| baseline 复现代码与步骤（2026-10-26 补注：main 上的是 2026-10-24 的旧副本；最新代码与说明在分支 code/token-pruning 的同一路径） | [code/token_pruning](../code/token_pruning/README.md) |
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
| 写作所需材料 | [新会话读取清单](../paper/论文写作-新会话读取清单.md)；[论文主张与证据（草稿）](论文主张与证据_草稿.md) |

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
| [analysis/论文笔记/PAPER-0008_CLASP.md](../analysis/论文笔记/PAPER-0008_CLASP.md) | 原文笔记：CLASP（按 9 类意图调预算比例与层融合；最接近的前人工作，与本项目逐项对照） |
| [analysis/论文笔记/PAPER-0009_VIP-Router.md](../analysis/论文笔记/PAPER-0009_VIP-Router.md) | 原文笔记：VIP-Router（学习逐样本在五种剪枝策略之间选择；与本项目对照） |
| [analysis/论文笔记/PAPER-0010_DivPrune.md](../analysis/论文笔记/PAPER-0010_DivPrune.md) | 原文笔记：DivPrune（最大最小多样性，在 LLM 之前选；本项目 DivPrune-64 的依据；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0011_CoverPruner.md](../analysis/论文笔记/PAPER-0011_CoverPruner.md) | 原文笔记：CoverPruner（按题目加权的覆盖（facility location）；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0012_MMTok.md](../analysis/论文笔记/PAPER-0012_MMTok.md) | 原文笔记：MMTok（题目覆盖 + 图像覆盖，次模贪心；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0013_CDPruner.md](../analysis/论文笔记/PAPER-0013_CDPruner.md) | 原文笔记：CDPruner（指令相关性调制的 DPP，条件多样性；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0014_VScan.md](../analysis/论文笔记/PAPER-0014_VScan.md) | 原文笔记：VScan（编码器侧扫描合并 + LLM 中间层按指令剪；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0015_ShaRP.md](../analysis/论文笔记/PAPER-0015_ShaRP.md) | 原文笔记：ShaRP（第一个解码层剪，去偏与去重，视频；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0016_FSR.md](../analysis/论文笔记/PAPER-0016_FSR.md) | 原文笔记：FSR（聚焦—扫描—细化：相关种子 + 条件最远点采样 + 合并；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0017_VisionTrim.md](../analysis/论文笔记/PAPER-0017_VisionTrim.md) | 原文笔记：VisionTrim（两阶段，层平均口径，文本引导的合并；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0018_SCOPE.md](../analysis/论文笔记/PAPER-0018_SCOPE.md) | 原文笔记：SCOPE（显著性 × 覆盖；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0019_HoloV.md](../analysis/论文笔记/PAPER-0019_HoloV.md) | 原文笔记：HoloV（按块分配名额的整体上下文保留；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/PAPER-0020_ZOO-Prune.md](../analysis/论文笔记/PAPER-0020_ZOO-Prune.md) | 原文笔记：ZOO-Prune（零阶梯度估计的灵敏度 × 多样性；2026-10-30 多代理工作流写，第二人复核待做） |
| [analysis/论文笔记/domainB_原文核对_LiRoth_Moldovan.md](../analysis/论文笔记/domainB_原文核对_LiRoth_Moldovan.md) | 领域 B 原文核对：Li & Roth（COLING 2002）、Moldovan 等（ACL 2000）对 IDEA-0003 的支持程度 |
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
| [analysis/MODE_C-0002_视觉token剪枝.md](../analysis/MODE_C-0002_视觉token剪枝.md) | MODE_C_PACKAGE：视觉 token 剪枝合并版（原文版 + 源码版 + EXP-0001/0002 回流；之后回流 1～13，2026-10-25 跑偏检查补注；2026-10-31 补注：现为回流 1～27；2026-10-29 补注：现为回流 1～23，回流 20～23 是 SQZ-EXP-0027、SQZ-EXP-0028、查新补充与 SQZ-EXP-0029） |
| [analysis/MODE_C-0003_时序预测.md](../analysis/MODE_C-0003_时序预测.md) | MODE_C-0003：时序预测的问题—现象—机制矩阵（iTransformer 为 baseline） |
| [analysis/IDEA-0002_题目词按信息量加权.md](../analysis/IDEA-0002_题目词按信息量加权.md) | 跨领域迁移方案：用 IDF 给题目加权，构造图像 token 的查询（第 ⑨～⑪ 步，领域 B 信息检索，领域 C BERTScore / SIF） |
| [analysis/IDEA-0001_编码器侧压缩的位置编号.md](../analysis/IDEA-0001_编码器侧压缩的位置编号.md) | IDEA_PACKAGE：编码器侧压缩的位置编号（融合路线候选，deferred） |
| [analysis/论文笔记/PAPER-0004_FastV_源码核验.md](../analysis/论文笔记/PAPER-0004_FastV_源码核验.md) | 源码核验补充：FastV |
| [analysis/论文笔记/PAPER-0005_SparseVLM_源码核验.md](../analysis/论文笔记/PAPER-0005_SparseVLM_源码核验.md) | 源码核验补充：SparseVLM / SparseVLM+ |
| [analysis/论文笔记/PAPER-0006_VisionZip_源码核验.md](../analysis/论文笔记/PAPER-0006_VisionZip_源码核验.md) | 源码核验补充：VisionZip |
| [analysis/实验报告/EXP-0001_record.md](../analysis/实验报告/EXP-0001_record.md) | EXPERIMENT_RECORD：三个仓库的代码级核验（CPU） |
| [analysis/实验报告/EXP-0001_squeeze.md](../analysis/实验报告/EXP-0001_squeeze.md) | SQUEEZE_REPORT：EXP-0001 的跨平台复验与预算口径 |
| [analysis/实验报告/EXP-0002_record.md](../analysis/实验报告/EXP-0002_record.md) | EXPERIMENT_RECORD：视觉 token 剪枝 GPU 复现（LLaVA-1.5-7B，POPE、MME，49 次运行） |
| [analysis/实验报告/EXP-0002_squeeze.md](../analysis/实验报告/EXP-0002_squeeze.md) | SQUEEZE_REPORT：EXP-0002（随机优于 FastV、“no”偏置、SparseVLM 实测 token） |
| [analysis/实验报告/EXP-0004_record.md](../analysis/实验报告/EXP-0004_record.md) | EXPERIMENT_RECORD：iTransformer 在 ETT 上的 GPU 复现（主 baseline，DEC-0012） |
| [analysis/实验报告/EXP-0004_squeeze.md](../analysis/实验报告/EXP-0004_squeeze.md) | SQUEEZE_REPORT：EXP-0004（iTransformer 在 ETT 上的 GPU 复现，阶段 A + B） |
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
| [analysis/跑偏检查_2026-10-25.md](../analysis/跑偏检查_2026-10-25.md) | 跑偏检查记录（2026-10-25）：对照研究流程、paper-plan 与 index 的全仓库检查，发现的问题与修正 |
| [analysis/跑偏检查_2026-10-30.md](../analysis/跑偏检查_2026-10-30.md) | 跑偏检查与完整阅读仓库（2026-10-30 新会话，DEC-0058）：EXP-0030、EXP-0031 两个阶段的检查，11 个只读代理与 Claude 读出的问题与修复，补封存的 tmp 脚本，推迟到排队跑完之后的代码分支修复（2026-10-31 补：已提交 9dae438，见文末补记） |
| [analysis/跑偏检查_2026-10-26.md](../analysis/跑偏检查_2026-10-26.md) | 跑偏检查与完整阅读仓库（2026-10-26 读，10-29 修完；DEC-0053）：EXP-0025～EXP-0029 各阶段的检查，读者报告的 120 条问题与修复，关机与额度中断后的恢复 |
| [analysis/IDEA-0003_按题目句式路由.md](../analysis/IDEA-0003_按题目句式路由.md) | IDEA_PACKAGE：按题目句式路由（助动词或 how many 开头的题用题目查询，其余用强对照；P11） |
| [analysis/实验报告/EXP-0020_record.md](../analysis/实验报告/EXP-0020_record.md) | EXPERIMENT_RECORD：按题目句式路由在新抽的 VQAv2 test2 上的预注册确认（第 ⑫ 步第六轮） |
| [analysis/实验报告/EXP-0021_record.md](../analysis/实验报告/EXP-0021_record.md) | EXPERIMENT_RECORD：VQAv2 test2 的补全（192 档路由；128 档参照） |
| [analysis/实验报告/EXP-0022_record.md](../analysis/实验报告/EXP-0022_record.md) | EXPERIMENT_RECORD：按题目句式路由在 GQA 上的预注册检验（第三个 benchmark；Q1、Q2 成立，Q3 不成立） |
| [analysis/实验报告/EXP-0022_squeeze.md](../analysis/实验报告/EXP-0022_squeeze.md) | SQUEEZE_REPORT：EXP-0022（GQA 上复现，增益来自 yes/no 与二选一题 F47；GQA 与 VQAv2 的差别 F48） |
| [analysis/实验报告/EXP-0023_record.md](../analysis/实验报告/EXP-0023_record.md) | EXPERIMENT_RECORD：开放式题的答案在哪里（为新模块做的 CPU 诊断） |
| [analysis/IDEA-0004_预算内放大.md](../analysis/IDEA-0004_预算内放大.md) | IDEA_PACKAGE：预算内放大（用题目查询定位被问的区域，裁剪后重新编码，用细节 token 替换一部分名额；P11、F44～F46） |
| [analysis/IDEA-0005_定位后覆盖.md](../analysis/IDEA-0005_定位后覆盖.md) | IDEA_PACKAGE：定位后覆盖（同一预算下，在题目查询定位的框里按基线的规则多留 token；C22、F50、F51） |
| [analysis/实验报告/EXP-0024_record.md](../analysis/实验报告/EXP-0024_record.md) | EXPERIMENT_RECORD：预算内放大的开发集运行（VQAv2 dev2、GQA val400；判读“不像”） |
| [analysis/实验报告/EXP-0024_squeeze.md](../analysis/实验报告/EXP-0024_squeeze.md) | SQUEEZE_REPORT：EXP-0024（放大为什么输 F49；覆盖 16 的增益依赖框盖住话题物体 F50；K = 1 的预算杠杆 F51） |
| [analysis/实验报告/EXP-0025_record.md](../analysis/实验报告/EXP-0025_record.md) | EXPERIMENT_RECORD：预算内放大的确认实验（计划 v0：TextVQA、V\* 的判据在开发结果之前写定；v1：所选变体、同取法的对照、10 次运行；结果：T1～T3 不成立） |
| [analysis/实验报告/EXP-0026_record.md](../analysis/实验报告/EXP-0026_record.md) | EXPERIMENT_RECORD：定位后覆盖的开发集运行（计划 v1、v1.1：居中框对照、K = 1、覆盖 8、不围框，13 次；判读“有希望”） |
| [analysis/实验报告/EXP-0027_record.md](../analysis/实验报告/EXP-0027_record.md) | EXPERIMENT_RECORD：定位后覆盖的确认实验（计划 v1：VQAv2 test3、GQA testdev，6 次；结果：Q1 部分成立、Q2 成立，DEC-0045） |
| [analysis/实验报告/EXP-0027_squeeze.md](../analysis/实验报告/EXP-0027_squeeze.md) | SQUEEZE_REPORT：EXP-0027（2026-10-26 事后补写：覆盖的增益都在不路由的题上，开发集的估计在新数据上保持；K = 1 路由与 VisionZip-64 持平或更好；F55、F56） |
| [analysis/IDEA-0006_强底座上的定位后覆盖.md](../analysis/IDEA-0006_强底座上的定位后覆盖.md) | IDEA_PACKAGE：强底座上的定位后覆盖（VisionZip-64 的主导 token 中，框内 = VisionZip 的框内个数 + n，框与 IDEA-0005 相同） |
| [analysis/实验报告/EXP-0028_record.md](../analysis/实验报告/EXP-0028_record.md) | EXPERIMENT_RECORD：VisionZip 上的定位后覆盖，开发集运行（计划 v1、v1.1：VQAv2 dev2、GQA val400，12 次，含不围框对照；v1.2：R03–R12 失败，以 R13–R22 重跑；结果：判读“有希望”，选每题覆盖 16） |
| [analysis/实验报告/EXP-0028_squeeze.md](../analysis/实验报告/EXP-0028_squeeze.md) | SQUEEZE_REPORT：EXP-0028（2026-10-26 事后补写：VisionZip-64 上定位的增益在开发集上集中于颜色题；两种底座的拼接是 IDEA-0007 的来由，补登为 C24；F57、F58） |
| [analysis/实验报告/EXP-0029_record.md](../analysis/实验报告/EXP-0029_record.md) | EXPERIMENT_RECORD：VisionZip 上的定位后覆盖的确认实验（计划 v1、v1.1：VQAv2 test3、GQA testdev、TextVQA，8～10 次；只在 EXP-0028 判读“有希望”时运行；2026-10-26 09:04 起运行，所选每题覆盖 16，10 次；结果：Q1～Q3 不成立，IDEA-0006 结束，DEC-0054） |
| [analysis/实验报告/EXP-0029_squeeze.md](../analysis/实验报告/EXP-0029_squeeze.md) | SQUEEZE_REPORT：EXP-0029（开发集的增益在新数据上消失来自题型内的效应而不是构成；每题覆盖只改动约 9% 的题；TextVQA 的损失集中在读字题；开发集上的定位模式没有复现；F59、F60） |
| [analysis/实验报告/EXP-0030_squeeze.md](../analysis/实验报告/EXP-0030_squeeze.md) | SQUEEZE_REPORT：EXP-0030（一层全看 + CLS 排序与 VisionZip 留的几乎是同一批 patch，VQAv2 上的差距来自 token 数而不是选法 F64；VisionZip 的主导 patch 约 4 个落在填充里 F63；GQA 语义类型上方向不同 F65，探索性） |
| [analysis/IDEA-0007_一层全看再按编码器排序.md](../analysis/IDEA-0007_一层全看再按编码器排序.md) | IDEA_PACKAGE：一层全看，再按编码器排序（K = 1：第 0 层看全部 patch，第 1 层起按视觉编码器的 CLS 注意力留 47 个；可加路由与覆盖） |
| [analysis/IDEA-0008_第0层借看全部patch.md](../analysis/IDEA-0008_第0层借看全部patch.md) | IDEA_PACKAGE：第 0 层借看全部 patch，按 FLOPs 对齐多留（被剪的 patch 在第 0 层只作 key / value；同样的 61 个、第 0 层看不到被剪的作对照） |
| [analysis/IDEA-0009_编号相同时的第0层借看.md](../analysis/IDEA-0009_编号相同时的第0层借看.md) | IDEA_PACKAGE：第 1 层起与 DivPrune 一样连续编号时，第 0 层只算 K / V 的借看能否加在先剪后送之上（peekc；对照 nopeekc、DivPrune-61、DivPrune-64） |
| [analysis/实验报告/EXP-0030_record.md](../analysis/实验报告/EXP-0030_record.md) | EXPERIMENT_RECORD：一层全看、再按编码器排序的开发集运行（计划 v1：VQAv2 dev2、GQA val400，6 次） |
| [analysis/实验报告/EXP-0031_record.md](../analysis/实验报告/EXP-0031_record.md) | EXPERIMENT_RECORD：一层全看、再按编码器排序的确认实验（计划 v1：新抽的 VQAv2 test4、GQA val400b 与 TextVQA，9 次；只在 EXP-0030 判读“有希望”时运行；结果：Q1、Q2 部分成立（只在 GQA val400b 上），Q3 不成立，拼接 S1、S2 部分成立，DEC-0059） |
| [analysis/实验报告/EXP-0031_squeeze.md](../analysis/实验报告/EXP-0031_squeeze.md) | SQUEEZE_REPORT：EXP-0031（K = 1 路由 + 覆盖 16 在 GQA 的两组新数据上都比 VisionZip-64 高约 2 分 F69；两种排序只重合约三分之一、准确率分不开 F70；每题覆盖 16 两组确认集合并 F71） |
| [analysis/实验报告/EXP-0032_record.md](../analysis/实验报告/EXP-0032_record.md) | EXPERIMENT_RECORD：本流程中的 DivPrune-64 与一层全看对多样性选择的作用（计划 v1：VQAv2 dev2、GQA val400，4 次；结果：DivPrune-64 明显强于 VisionZip-64（+2.20 / +1.63），一层全看没有帮助，DEC-0061） |
| [analysis/实验报告/EXP-0032_squeeze.md](../analysis/实验报告/EXP-0032_squeeze.md) | SQUEEZE_REPORT：EXP-0032（DivPrune 与 VisionZip 选的 patch 只重合约四分之一、DivPrune 铺得更开、留约 5 个填充 patch F75；K = 1 时多样性的优势不随两种选法的分歧变大 F76） |
| [analysis/实验报告/EXP-0033_record.md](../analysis/实验报告/EXP-0033_record.md) | EXPERIMENT_RECORD：第 ④ 步第二轮，探索性组合实验（计划 v1、v1.1：VQAv2 dev2、GQA val400，12 次；预先写定“意外”的三类，没有判据；2026-10-31 补：已完成，DEC-0062） |
| [analysis/实验报告/EXP-0033_squeeze.md](../analysis/实验报告/EXP-0033_squeeze.md) | SQUEEZE_REPORT：EXP-0033（脚本写于看到结果之前；X2 少铺开而准确率不变 F80；DivPrune 底座上的题目查询没有帮助 F79；64 档的缺口与 K = 1 在 GQA 属性、二选一题上的补回 F77、F78；按题目文字挑方法几乎没有提升 F83） |
| [analysis/实验报告/EXP-0034_record.md](../analysis/实验报告/EXP-0034_record.md) | EXPERIMENT_RECORD：IDEA-0008 的开发运行（计划 v1：VQAv2 dev2、GQA val400，6 次；Q1 peek − DivPrune-64，Q2 peek − nopeek；2026-10-31 补：已完成，DEC-0065） |
| [analysis/实验报告/EXP-0034_squeeze.md](../analysis/实验报告/EXP-0034_squeeze.md) | SQUEEZE_REPORT：EXP-0034（脚本写于看到结果之前；nopeek 偏答“是”、借看去掉它 F88；连续编号丢掉空间布局 F89；借看的增益与被点名的物体留下多少无关 F90；官方代码的位置编号） |
| [analysis/实验报告/EXP-0035_record.md](../analysis/实验报告/EXP-0035_record.md) | EXPERIMENT_RECORD：IDEA-0008 的确认（计划 v1，条件式：新抽的 VQAv2 test5、GQA val400c，TextVQA 描述，10 次；在看到 EXP-0034 的结果之前写定）；2026-10-31 补：已完成，Q1 部分成立，DEC-0067） |
| [analysis/实验报告/EXP-0036_record.md](../analysis/实验报告/EXP-0036_record.md) | EXPERIMENT_RECORD：IDEA-0008 与 F88、F89 在 MME、V*、POPE 上（计划 v1：9 次；Q1 推广，Q2 POPE 上的偏“是”，Q3 位置题；在看到 EXP-0035 的结果之前写定）；2026-10-31 补：已完成，Q1 没有一个更高，DEC-0069） |
| [analysis/实验报告/EXP-0037_record.md](../analysis/实验报告/EXP-0037_record.md) | EXPERIMENT_RECORD：IDEA-0008 在 16 档（计划 v1：开发集 4 次，探索；peek 留 13 对 DivPrune-16；在看到 EXP-0036 的结果之前写定）；2026-10-31 补：已完成，不像，DEC-0070） |
| [analysis/实验报告/EXP-0038_record.md](../analysis/实验报告/EXP-0038_record.md) | EXPERIMENT_RECORD：IDEA-0009 的开发运行（计划 v1：VQAv2 dev2、GQA val400，6 次；Q1 peekc − DivPrune-64，Q2 peekc − nopeekc；2026-10-31 补：已完成，Q1 不像，Q2 有希望但是纠正；R02 由 R07 重跑（计划 v1.1）；DEC-0074） |
| [analysis/实验报告/EXP-0020_squeeze.md](../analysis/实验报告/EXP-0020_squeeze.md) | SQUEEZE_REPORT：EXP-0020（增益来自 yes/no 与 number F42；与 VisionZip 的差距在开放式题，是底座的差距 F43） |
| [analysis/论文主张与证据_草稿.md](../analysis/论文主张与证据_草稿.md) | 论文主张与证据对照（草稿）：可以写的主张 CL-1～CL-10、不能写的主张、共同的限定 |
| [analysis/实验报告/EXP-0018_squeeze.md](../analysis/实验报告/EXP-0018_squeeze.md) | SQUEEZE_REPORT：EXP-0018（答案类型上效应相反 F38；损失集中在问点名物体属性的题 F39；一般题目上 FastV 一族低于 VisionZip F40） |
| [analysis/实验报告/EXP-0019_record.md](../analysis/实验报告/EXP-0019_record.md) | EXPERIMENT_RECORD：审计要求的补充（A+B 的对照；最终方法的区分力） |
| [analysis/实验报告/EXP-0018_record.md](../analysis/实验报告/EXP-0018_record.md) | EXPERIMENT_RECORD：VQAv2 验证集子集上的一般题目检验（第 ⑫ 步第五轮） |
| [analysis/实验报告/EXP-0017_record.md](../analysis/实验报告/EXP-0017_record.md) | EXPERIMENT_RECORD：192 档复验（第 ⑫ 步第四轮） |
| [analysis/实验报告/EXP-0015-0016_squeeze.md](../analysis/实验报告/EXP-0015-0016_squeeze.md) | SQUEEZE_REPORT：EXP-0015、EXP-0016（收益主要来自切分；POPE 提升随预算缩小 F37）（审计修正，DEC-0030 AU-12：应为“在切分之外，可视性没有可检出的贡献”；2026-10-25 跑偏检查补注） |
| [analysis/实验报告/EXP-0016_record.md](../analysis/实验报告/EXP-0016_record.md) | EXPERIMENT_RECORD：可视性与切分的拆分，以及 128 档复验（第 ⑫ 步第三轮） |
| [analysis/实验报告/EXP-0015_record.md](../analysis/实验报告/EXP-0015_record.md) | EXPERIMENT_RECORD：按词的可视性分配名额的题目查询（第 ⑨～⑫ 步第二轮，P10） |
| [analysis/1a-4b_FastV不如随机选择.md](../analysis/1a-4b_FastV不如随机选择.md) | 1a–4b 机制分析：同预算下 FastV 为什么不如随机选择（第 1 轮，到 2b） |
| [analysis/1a-4b_选择与题目无关漏掉被问的小物体.md](../analysis/1a-4b_选择与题目无关漏掉被问的小物体.md) | 1a–4b 机制分析（第 ⑧ 步，候选 1）：选择与题目无关时，被问的小物体为什么被删掉（第 1 轮，到 2b；COCO 物体框） |
| [analysis/1a-4b_四种架构在同一批窗口上出错.md](../analysis/1a-4b_四种架构在同一批窗口上出错.md) | 1a–4b 机制分析：时序四种架构为什么在同一批窗口上出错（第 1 轮，到 2b；误差的水平/形状拆分） |
| [analysis/第7步_改进方向排序.md](../analysis/第7步_改进方向排序.md) | 第 ⑦ 步：改进方向排序（视觉与时序两个方向的候选排序） |
| [analysis/实验报告/EXP-0009_record.md](../analysis/实验报告/EXP-0009_record.md) | EXPERIMENT_RECORD：iTransformer 检查点在验证期上的预测（不训练；已完成，48 次运行；原写“计划”，2026-10-25 跑偏检查补注） |
| [analysis/实验报告/EXP-0009_squeeze.md](../analysis/实验报告/EXP-0009_squeeze.md) | SQUEEZE_REPORT：EXP-0009（iTransformer 检查点在验证期上的预测，四个 ETT 数据集） |
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
| [analysis/专题分析/查新补充_覆盖与多样性_2026-10-26.md](../analysis/专题分析/查新补充_覆盖与多样性_2026-10-26.md) | 查新补充：覆盖 / 多样性一类在 64 个时的表现（CoverPruner 的表）、DivPrune 的放置消融、浅层剪枝与深层撤除（2026-10-26） |
| [analysis/专题分析/查新补充_位置编号与借看_2026-10-31.md](../analysis/专题分析/查新补充_位置编号与借看_2026-10-31.md) | 查新补充：剪枝后视觉 token 的位置编号（GAP、Layer-Aware 已做；两篇主代理核实摘要）、第 0 层借看（没有找到直接先例，最近 ShortV）、定位评测、是非偏移（偏否已有报告）；对本项目能与不能声称的（DEC-0072） |
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


## 附录：之前的状态行（2026-10-26 从开头移来，逐字未改）

前一状态行（2026-10-26 11 点 45 分，补到 DEC-0052）：查新补充：留 64 个 token 时，覆盖 / 多样性一类（DivPrune、MMTok、CoverPruner）比 VisionZip 高约 +1.7～+3.5，我们目前的方法比它小；机制检验 EXP-0032（本流程中的 DivPrune-64，以及一层全看对多样性选择的作用，4 次）排在 EXP-0031 之后（DEC-0052）。前一状态行（2026-10-26 11 点 20 分，补到 DEC-0051）：用户确认 EXP-0028 失败的 10 次不计入 IDEA-0006 的上限（DEC-0050）。IDEA-0007 的确认计划 EXP-0031（新抽的 VQAv2 test4、GQA val400b，9 次）在 EXP-0030 的任何结果之前写定，只在 EXP-0030 判读“有希望”时运行（DEC-0051）。EXP-0029 运行中。前一状态行（2026-10-26 9 点半，补到 DEC-0049）：EXP-0028 的记录内深挖（探索性）发现覆盖的增益主要在颜色题上，两种底座各有所长；据此提出 IDEA-0007（一层全看，再按编码器排序：层平均 63.5，第 0 层看全部 576 个，第 1 层起按视觉编码器的 CLS 排序留 47 个），开发实验 EXP-0030（6 次）的计划已写定，排在 EXP-0029 之后。前一状态行（2026-10-26 9 点，EXP-0028 完成时更新，补到 DEC-0048）：IDEA-0006（VisionZip-64 上的定位后覆盖）的开发实验 EXP-0028 判读“有希望”，选中每题覆盖 16：比 VisionZip-64 高 +1.36 [+0.73, +2.02]（VQAv2 dev2）/ +0.52 [−0.38, +1.52]（GQA val400）；与 FastV 一族不同，增益来自定位而不是集中（F57、F58）。R03–R12 因代码错误失败，以 R13–R22 重跑（DEC-0047）。确认实验 EXP-0029（10 次）09:04 起运行中（DEC-0048）。前一状态行（2026-10-26，EXP-0027 完成时更新，补到 DEC-0045）：视觉方向在第 ⑨～⑫ 步第六轮。IDEA-0004（预算内放大）在一般题目与看细节的题上都不成立，已结束（DEC-0039、DEC-0041）。IDEA-0005（定位后覆盖）的开发集判读“有希望”（DEC-0042）；确认实验 EXP-0027（预注册）Q1 部分成立、Q2 成立：K = 1 + 覆盖 16 比 IDEA-0003 的路由高 +3.44 / +1.75（VQAv2 test3 / GQA testdev），其中 K = 1 本身 +2.87 / +1.33，覆盖只再加 +0.57 / +0.42，覆盖不作为方法的贡献（DEC-0045，F55、F56）。按用户的选择（DEC-0043），区域覆盖移到 VisionZip-64 上（IDEA-0006，DEC-0044），开发实验 EXP-0028（12 次）2026-10-26 05:52 起运行中。前一状态行（2026-10-25 跑偏检查更新，DEC-0012～0034；IDEA-0004 时补到 DEC-0035；EXP-0022 完成时补到 DEC-0038）：视觉 token 剪枝方向在候选 1 的第 ⑫ 步，**计划目标在 64 档达到**（DEC-0033）。POPE 上 A′+B+V − A 在 64、128、192 三档都显著（+15.84、+5.21、+2.29，按图像重抽的区间 [13.88, 17.82]、[3.94, 6.47]、[1.38, 3.22]，DEC-0030）；VQAv2 test2 上按题目句式路由 − A 在 64 档 +1.22 [+0.58, +1.82]、128 档 +0.72 [+0.29, +1.15]（EXP-0020）；MME-P 的差异都不显著。EXP-0021（VQAv2 test2 的 192 档与 128 档参照，只作描述）已完成；EXP-0022（GQA testdev-balanced，预注册的第三个 benchmark，DEC-0034）已完成：路由 − A 64 档 +0.87 [+0.37, +1.40]、128 档 +0.68 [+0.30, +1.08]，与 VisionZip-64 无可检出差别（DEC-0038），候选 1 的 68 次运行用完。第 ⑨～⑫ 步第四轮：新模块 [IDEA-0004](IDEA-0004_预算内放大.md)（预算内放大，DEC-0035）设计完成，运行前经多代理代码审查与设计评审，单独计预算（30 次运行或 20 GPU 小时）；开发集运行 [EXP-0024](实验报告/EXP-0024_record.md)（17 次，计划 v1.1：TextVQA、V\* 的检验不以开发集判读为门槛，DEC-0037）2026-10-25 16:33 起运行中。候选 1 的运行上限为 68 次（DEC-0034；此前 DEC-0028 为 60 次，默认 30 次）。时序预测方向在第 ⑦ 步后暂停（DEC-0020、DEC-0021）。九篇论文（PAPER-0001～0009）已有原文笔记，其中七篇（PAPER-0001～0007）另有源码核验补充（PAPER-0008 CLASP、PAPER-0009 VIP-Router 只读原文，代码未下载），另有领域 B 两篇问答论文的原文核对；MinerU 解析的全文在 literature/processed（2026-10-24，flash 档，只有文本）。实验室电脑是原生 Windows，文件只放在 D:\6CJNM007。流程规范为[研究流程 v2.1](研究流程.md)。新增、移动或归档文档时同步本索引。（2026-10-25 跑偏检查更新本行，并在“全部现有文档”补上漏列的文件。）原状态行（2026-10-25，DEC-0012～0026）：视觉 token 剪枝方向完成第 ⑧ 步第 3 轮（EXP-0013，压误报，没有修正通过），第 ⑨～⑫ 步：IDF 加权的题目查询在 POPE 上提升、MME 上未提升（EXP-0014，DEC-0026），回到第 ⑨ 步处理词的可视性；时序预测方向在第 ⑦ 步后暂停（DEC-0020、DEC-0021）。七篇论文已有原文笔记与源码核验补充。实验室电脑是原生 Windows，文件只放在 D:\6CJNM007。流程规范为[研究流程 v2.1](研究流程.md)。新增、移动或归档文档时同步本索引。（此前这一行停在 DEC-0014 时的状态，2026-10-24 的跑偏检查中更新。）
