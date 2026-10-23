# SQUEEZE_REPORT：<实验标题>

状态：模板，未分析。
报告 ID / 实验 ID / 输入记录版本 / 日期：待补充。

## 结构化交接包

此块是未填写的模板；字段可填写内容或精确的正文位置。缺失项须登记 missing_evidence，不将占位值视为研究结果。

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: null
squeeze_id: null
created_at: null
source_ids: []
evidence_status: unknown
evidence_locations: []
missing_evidence:
- 模板尚未填写，需提供真实来源、记录和证据
next_skill: deep-read Mode C
input_experiment_record_ids: null
claim_scope: null
data_quality_and_provenance_gaps: null
primary_result_with_uncertainty: null
process_dynamics: null
mechanism_and_intermediate_findings: null
errors_heterogeneity_and_boundary_conditions: null
robustness_sensitivity_and_alternative_explanations: null
new_P: null
new_F: null
new_M: null
evidence_index: null
open_questions: null
next_routes: null
mode_C_input_bundle:
  source_ids: []
  P: []
  F: []
  M: []
  E: []
  evidence_locations: []
```

mode_C_input_bundle 的每个 P/F/M/E 条目附 ID、来源 experiment_id 或 squeeze_id、证据标记和稳定定位。没有直接证据的机制标 AI/UN；确无新增时明确记录无新增及依据，再回流 Mode C。

## 证据审计

- EXPERIMENT_RECORD 与全部原始证据：
- 缺失、异常、测量问题和排除：
- 分析命令、版本与输出：
- 哪些分析预先计划，哪些在看到数据后提出：

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |
| 训练 / 运行动态 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |
| 错误和失败模式 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |
| 子组、切片和边界 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |
| 机制、消融和交互 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |
| 稳健性及跨数据集 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |
| 局部收益、整体收益和代价 | 待分析 | 待分析 | 待分析 | 待分析 | 待制定 |

不适用或证据不足时写明原因，不为填满表格编造分析。探索切片考虑样本量、多重比较和数据泄漏；筛选出的模式需独立复验。

## 学到什么

- 支持/削弱的假设、负结果和无明显差异：
- 已排除与仍无法区分的解释：
- 保留 / 复验 / 回退 / 停止及依据：
- 最小、最能区分剩余解释的下一次实验：
- 新 P/F/M/E 与回流 Deep Read 的连接点：

## 同步清单

- [ ] experiment-results.md 与所有运行相符。
- [ ] 已发现的实验现象.md 更新，事实与解释分开。
- [ ] 论文素材库.md 标注证据及边界。
- [ ] 最新实验分析报告.md 指向本报告与事实记录。
- [ ] 关键方向变化写入 version-evolution.md。
- [ ] 新文档登记 index.md。
