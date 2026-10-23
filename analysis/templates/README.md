# 模板与交接约定

五个核心产物模板已对齐 [research-pipeline-contract v1.0](../../.claude/skills/5-skill-research-workflow.md)。技能入口与兼容说明见[技能安装说明](../../.claude/README.md)。模板用于保存产物，不是技能实现或已完成的研究。

| 模板 | 用途 | 实例位置 |
| --- | --- | --- |
| [ANALYSIS_PACKAGE](ANALYSIS_PACKAGE.md) | 单篇论文事实和解释边界 | analysis/论文笔记/ |
| [MODE_C_PACKAGE](MODE_C_PACKAGE.md) | P/F/M/E、P×F 与连接点 | analysis/ |
| [IDEA_PACKAGE](IDEA_PACKAGE.md) | 可执行、可证伪的方案 | analysis/ |
| [EXPERIMENT_RECORD](EXPERIMENT_RECORD.md) | 实际运行与原始证据 | analysis/实验报告/ |
| [SQUEEZE_REPORT](SQUEEZE_REPORT.md) | 深度分析和下一步 | analysis/实验报告/ |
| [1a-4b机制分析](1a-4b机制分析.md) | 因果假设与干预验证 | analysis/ |
| [跨领域迁移方案](跨领域迁移方案.md) | A/B/C 条件与适配对照 | analysis/ |

实例统一记录 ID、状态、日期、输入版本、作者/执行者、来源及证据位置。ID 在项目内唯一且稳定：P 问题、F 现象、M 机制、E 证据、C 连接点，论文另用 paper_id。

每个核心模板开头有一份 YAML 交接包，包含统一元数据和类型专属字段。填写真实实例时使用具体值或稳定的正文/表格位置，不保留含义不明的空字段；缺失项同时登记 missing_evidence。package_id 与本阶段 analysis_id / matrix_id / idea_id / experiment_id / squeeze_id 相同；source_ids 回链上游包或材料，created_at 使用实际生成日期。模板中的 null/空列表仅表示尚未填写，不能直接交付为已完成产物。

证据包状态与条目证据类型分别记录：

| 标记 | 含义 |
| --- | --- |
| verified | 当前材料存在可定位直接证据，关键边界已记录；不表示机制被普遍证明 |
| partial | 有证据，但条件、替代解释或数据缺口尚未解决 |
| unknown | 当前材料不足以判断，不等于阴性结果 |
| SC | 来源明确主张 |
| DE | 可直接核验的证据 |
| AI | 分析推断，写明推理链 |
| UN | 未知，材料不足 |

连接点状态另用 supported / plausible / diagnostic-needed / unsupported，不与包状态混用。发现后的路由为 SQUEEZE_REPORT → MODE_C_PACKAGE → IDEA_PACKAGE，不直接把深挖报告改名为新方案。

空字段是待补充；未运行、失败、中断、完成与证据不足须区分。复制模板创建实例，不直接把模板改成一份具体实验。新增实例登记索引，每次交接注明缺口与下一步。
