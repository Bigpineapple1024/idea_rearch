# EXPERIMENT_RECORD：<实验标题>

状态：模板，未执行。
实验 ID / IDEA ID / 日期 / 执行者：待补充。
实际执行后标注部分完成 / 完成 / 失败 / 中断及原因。

## 结构化交接包

此块是未填写的模板；字段可填写内容或精确的正文位置。缺失项须登记 missing_evidence，不将占位值视为研究结果。

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
package_id: null
experiment_id: null
created_at: null
source_ids: []
evidence_status: unknown
evidence_locations: []
missing_evidence:
- 模板尚未填写，需提供真实来源、记录和证据
next_skill: squeeze-experiment
source_idea_id: null
plan_version: null
execution_status: null
research_question: null
hypothesis: null
research_question_and_hypothesis:
  research_question: null
  hypothesis: null
planned_vs_actual_design_and_controls: null
actual_configuration: null
samples_materials_or_data_provenance: null
all_runs_or_repetitions: null
results_with_uncertainty: null
observations_and_deviations: null
known_limits_and_missing_evidence: null
next_routes: null
```

execution_status 按事实填 completed / partial / interrupted / pilot / exploratory / data-incomplete / completed-no-clear-difference，逐次失败原因保留在运行表。无 Idea 包时 source_idea_id 填 none，并引用实际协议或来源；不得事后伪造计划。组合字段 research_question_and_hypothesis 与分列字段保持一致。

## 计划与实际

- 计划版本与链接：
- 实际问题、组别、变量和对照：
- 材料/数据来源、版本、划分、许可、校验和：
- 代码 commit、未提交差异补丁的保存路径及校验和：
- 环境、依赖、硬件、配置、命令和种子：
- 与计划的偏差、发生时间和原因：

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |

## 原始结果长表

| 运行 ID | 指标 / 数据切片 | 数值或 NA | 单位 / 方向 | 计算命令 / 版本 | 原始证据定位 |
| --- | --- | --- | --- | --- | --- |

## 汇总

- 纳入的全部运行及排除理由：
- 主结果、局部指标、整体表现和代价：
- 重复次数、效应量、不确定性及计算方法：
- 未完成项、丢失数据、失败和中断的影响：

## 分层记录

- 观察事实（直接证据 E）：
- 当前解释（推断）：
- 待复验假设与替代解释：
- 不能由本次数据得出的结论：

## 交接

核验原始证据可恢复、覆盖全部运行、总表已同步。将本记录与真实原始证据交给 Squeeze Experiment。
