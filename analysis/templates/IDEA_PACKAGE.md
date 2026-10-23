# IDEA_PACKAGE：<想法>

状态：模板，未设计/执行。
IDEA ID / 日期 / MODE_C_PACKAGE 版本 / C ID：待补充。

## 结构化交接包

此块是未填写的模板；字段可填写内容或精确的正文位置。缺失项须登记 missing_evidence，不将占位值视为研究结果。

```yaml
package_type: IDEA_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: null
idea_id: null
created_at: null
source_ids: []
evidence_status: unknown
evidence_locations: []
missing_evidence:
- 模板尚未填写，需提供真实来源、记录和证据
next_skill: record-experiment
source_matrix_id: null
selected_C_id: null
source_P_ids: null
source_F_ids: null
source_M_ids: null
source_E_ids: null
research_question: null
hypothesis_and_mechanism: null
testable_predictions: null
design_and_controls: null
sample_materials_or_experimental_units: null
primary_outcome_and_estimand: null
secondary_outcomes: null
analysis_plan_and_uncertainty: null
resources_time_cost_and_risk: null
ethics_safety_privacy_or_compliance: null
stopping_and_interpretation_rules: null
status: null
```

status 填 exploration / diagnostic / pilot / formal-validation / deferred。一个包只选择一个 selected_C_id，并引用具体矩阵版本。

## 问题与预测

- 本次要验证或排除的一个问题：
- 假设、可观察预测及可能推翻它的结果：
- 直接证据、当前推断与未知：
- 探索 / 确认性研究；新颖性检索状态：
- 生效条件及 baseline 是否满足：

## 可执行设计

- baseline、代码版本、材料与数据划分：
- 改动、对照、控制变量和潜在混杂：
- 单变量检验；必要时组合/因子对照：
- 主指标、单位、方向、实际意义阈值：
- 局部诊断、整体收益、潜在损失与资源成本：
- 样本量、重复次数、种子及其选择依据：
- 不确定性估计、统计方法及排除规则：
- 跨数据集/场景和独立验证计划：
- 执行命令、配置、环境及输出路径（未验证须注明）：
- 算力/时间/存储预算和最大运行次数：

## 预先制定的解释规则

| 情形 | 支持 / 削弱什么 | 仍不能排除什么 | 下一步 |
| --- | --- | --- | --- |
| 局部改善，整体达阈值 | 待制定 | 待制定 | 待制定 |
| 局部改善，整体无净收益 | 待制定 | 待制定 | 待制定 |
| 局部与整体均未改善 | 待制定 | 待制定 | 待制定 |
| 噪声大、实现失效或运行不完整 | 待制定 | 待制定 | 待制定 |

成功 / 停止 / 回退标准和最晚决策点：待补充。
执行前冻结的方案版本与执行后偏差记录方式：待补充。
