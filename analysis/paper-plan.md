# Paper Plan：实验到论文的宏观规划

状态：目录与模板初始化完成，五个 skills 已安装并静态核验。研究主题、baseline、数据集、资源与时间表待确定，下表不代表已经完成研究。

## 项目定义

| 项目 | 内容 |
| --- | --- |
| 研究领域与核心问题 | 多模态大模型推理加速：免训练视觉 token 剪枝（T1，见[毕业选题候选](毕业选题候选.md)）；当前问题见 [MODE_C-0001](MODE_C-0001_视觉token剪枝.md) |
| 投稿定位 / 目标读者 | 修士毕业论文为主，约 1 年 |
| baseline 与选择依据 | 骨架 LLaVA-1.5-7B；开发 baseline VisionZip；对照 FastV、SparseVLM（DEC-0005） |
| 数据来源、许可与划分 | POPE（COCO val2014）、MME 为主，GQA、TextVQA 用于独立验证；按 LLaVA docs/Evaluation.md 获取 |
| 主指标、方向与实际意义阈值 | POPE 准确率、MME 感知分（越高越好），同时报告理论 FLOPs 与实测延迟；阈值见 IDEA-0001 |
| 算力、存储、时间与预算 | 单张消费级 GPU（≤24GB）；存储与预算待补充 |
| 跨数据集 / 场景普遍性要求 | 待补充 |

## 阶段与交付

| 阶段 | 交付物 | 进入下一阶段的依据 | 状态 |
| --- | --- | --- | --- |
| 明确范围与复现 baseline | 固定版本、环境与复现记录 | 可复现基准，偏差原因可解释 | 进行中：版本已固定、CPU 代码核验完成（EXP-0001），GPU 复现（EXP-0002）待执行 |
| 阅读与交叉分析 | ANALYSIS_PACKAGE、MODE_C_PACKAGE | 原文可追溯，连接点有条件和证据 | 进行中：PAPER-0001～0003 与 MODE_C-0001 为 partial，论文全文待回查 |
| 设计最小验证 | IDEA_PACKAGE | 可证伪假设、对照与预算明确 | IDEA-0001 已设计，待 EXP-0002 通过后执行 |
| 执行与记录 | EXPERIMENT_RECORD | 保存所有运行和原始证据 | 待开始 |
| 深挖、归因与复验 | SQUEEZE_REPORT、1a–4b 分析 | 检验关键替代解释 | 待开始 |
| 迁移、消融与边界验证 | 适配对照、交互与普遍性结果 | 分别验证局部机制和整体净收益 | 待开始 |
| 写作与复核 | 正文、图表与复现材料 | 主张有证据，范围准确 | 待开始 |

## 下一步

1. 在 GPU 上按 [code/token_pruning/README](../code/token_pruning/README.md) 执行 EXP-0002：vanilla、VisionZip、FastV × 192/128/64 × POPE、MME，另跑 SparseVLM（v1.5）与同预算随机剪枝。
2. 回查三篇论文全文，补齐论文笔记中的 SC 条目，并补读 DART、VScan 等 3～5 篇相关工作做新颖性检索。
3. EXP-0002 通过后执行 IDEA-0001，之后按融合路线建立 IDEA-0002（合并幅值与 CLS）。

每份 IDEA_PACKAGE 预先填写成功标准、最大预算与停止/回退条件。证据不足时回到缺证据的环节，不因投入时间而不断叠加机制。计划变化记录到 [version-evolution](version-evolution.md)。
