# SQUEEZE_REPORT：FastV 与随机选择保留位置的深挖（EXP-0006）

状态：完成（2026-10-24）。
报告 ID SQZ-EXP-0006 / 实验 ID EXP-0006 / 输入记录：analysis/实验报告/EXP-0006_record.md（2026-10-24 版，commit 488d551）/ 日期 2026-10-24。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0006
squeeze_id: SQZ-EXP-0006
created_at: 2026-10-24
source_ids: [EXP-0006, EXP-0002, SQZ-EXP-0002, MODE_C-0002]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0006_record.md
- code/output/EXP-0006/analysis/kept_stats.md、kept_summary.json（SHA256SUMS 的 sha256 dc4ac5c0…，附 analyze_kept_tokens.py 副本）
- code/output/EXP-0006/EXP-0006-R01～R04/kept_tokens.json（每道题保留的 patch 索引）
missing_evidence:
- 没有 COCO 物体框：不能确认被删掉的是题目物体
- 只测了 K=2；位置偏差的来源（RoPE 衰减、attention sink、因果注意力）没有拆开
- 子集只有 20 张图、358 道题，一个随机种子
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0006]
claim_scope: LLaVA-1.5-7B，FastV（K=2，上一层最后一个 token 的 head 平均注意力）与同数量的均匀随机选择，keep 30 与 166，pope_img20 子集
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“主要结果”
process_dynamics: 不适用（推理诊断）
mechanism_and_intermediate_findings: 见“深度分析”第 3 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 2、4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 5 行
new_P: [P-SQZ6-1, P-SQZ6-2]
new_F: [F15]
new_M: [M1 状态更新：结果层面得到支持，来源未拆开]
evidence_index: 见 mode_C_input_bundle.E
open_questions: [位置分数曲线来自 RoPE 衰减还是 attention sink, 去掉位置项后保留分布与召回是否恢复, 更深的 K 是否仍与问题无关]
next_routes: [用空白图像测 FastV 的纯位置分数曲线并据此校正打分（子集，2～3 次诊断运行）, 补 COCO 物体框后检验被删 patch 与题目物体的重叠, deep-read Mode C 回流]
mode_C_input_bundle:
  source_ids: [EXP-0006, SQZ-EXP-0006]
  P:
  - id: P-SQZ6-1
    text: "FastV 打分的位置曲线（最后一行最高，第 0 行有小峰）来自 RoPE 远程衰减、attention sink，还是因果注意力的其他位置效应"
    evidence: [E-SQZ6-1, E-SQZ6-2]
    status: UN
  - id: P-SQZ6-2
    text: "把 FastV 的分数除以与内容无关的位置分数曲线后，保留集合能否覆盖整幅图像，POPE 召回能否回升"
    evidence: [E-SQZ6-1, E-SQZ6-3]
    status: UN（诊断待做）
  F:
  - id: F15（新）
    text: "FastV（K=2）保留的 patch 由位置主导且几乎与问题无关：平均行号 15.9～16.6（均匀 11.5），底部 4 行占 44%～45%（均匀 17%），最后一个 patch 在 100% 的题目中被保留；同一张图换问题，保留集合的 Jaccard 为 0.95；同一 keep 下覆盖的 4×4 块比随机少（45% vs 58%，85% vs 100%）"
    evidence: [E-SQZ6-1, E-SQZ6-2, E-SQZ6-4]
    status: DE
  M:
  - id: M1（状态更新）
    text: "“注意力打分带位置偏差”在结果层面得到直接支持（保留集中在序列上最靠近文本的图像底部）；偏差来自 RoPE 衰减的部分仍是推断"
    evidence: [E-SQZ6-1]
    status: 结果 DE；来源 AI
  E:
  - id: E-SQZ6-1
    text: "保留 patch 的平均行号、底部 1/4 占比、4×4 块覆盖率、最近邻距离与归一化熵（四次运行）"
    location: code/output/EXP-0006/analysis/kept_stats.md
  - id: E-SQZ6-2
    text: "逐行分布（每行占保留 patch 的比例，均匀为 4.2%）：FastV-64 最后一行 16.1%、第 20～23 行合计 44.2%、上半幅 26.4%，第 0 行 4%（比第 1～15 行的 1%～3% 高）；FastV-192 最后一行 13.6%（每道题约 23/24 个最后一行的 patch 被保留）、第 20～23 行 45.3%、上半幅 22.4%；随机两档每行都是 4%"
    location: code/output/EXP-0006/analysis/kept_summary.json（row_histogram）
  - id: E-SQZ6-3
    text: "FastV-64 在子集上答对的正样本 24/180，yes 比例 6.7%（随机 77/180、22.3%）；FastV 答对与答错的正样本块覆盖率只差 4.5 个百分点（49.3% vs 44.8%）"
    location: kept_summary.json
  - id: E-SQZ6-4
    text: "同图不同题 Jaccard：FastV 0.948 / 0.957，随机 0.026 / 0.169；不同图 Jaccard：FastV 0.108 / 0.359，随机 0.027 / 0.169（2811 对同图不同题）"
    location: kept_stats.md
  evidence_locations: [code/output/EXP-0006/analysis/SHA256SUMS]
```

## 证据审计

- **完整性**：4 次运行都记录到 358 次预填充，等于题数；kept_tokens.json 中每个集合的大小都等于 keep（metrics.json 的 all_sets_full）。
- **对照的有效性**：随机选择的统计量都与独立均匀抽取的期望一致（平均行号 11.5、每行 4%、keep 166 时 Jaccard 0.169 对期望 0.168），说明分析脚本和钩子没有引入位置偏差。
- **血缘**：子集由 make_pope_subset.py 生成（题目文件 sha256 f3ae79a4…，标签文件 0dedbc18…）；代码 commit 1eee940，运行时无未提交改动。
- **预先计划与事后提出**：H1～H4 和解释规则在计划 v1 中写定；逐行分布（E-SQZ6-2）和答对/答错的覆盖率比较是看到结果后补充的描述。

## 主要结果

| 结果 | 对照 | 差异 | 不确定性 | 实际意义 | 证据 |
| --- | --- | --- | --- | --- | --- |
| H1 位置偏差 | 随机 | 平均行号 +4.4 / +5.1；底部 1/4 占比 57% vs 25% | 358 道题；方向在两档 keep 上一致 | FastV 的保留集合偏向序列上最靠近文本的图像底部 | E-SQZ6-1、E-SQZ6-2 |
| H2 空间集中 | 随机 | 4×4 块覆盖率 45.4% vs 57.7%，85.4% vs 99.6% | 同上 | 同样数量的 token，FastV 覆盖的区域更少 | E-SQZ6-1 |
| H3 固定位置 | 随机 | 最后一个 patch 100% 被保留；熵 0.84 vs 1.00 | 同上 | 有一组与图像无关、几乎总被保留的位置 | E-SQZ6-1 |
| H4 与问题的关系 | 随机 | 同图不同题 Jaccard 0.95 | 2811 对 | K=2 时 FastV 的选择基本不看问题 | E-SQZ6-4 |

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | H1～H3 成立，H4 显示选择几乎与问题无关（见上表） | 支持 MODE_C-0002 的 M1：注意力打分带位置偏差 | — | 单一子集、单一种子 | — |
| 错误和失败模式 | FastV-64 在子集上只答对 24/180 道正样本，yes 比例 6.7%；答对与答错的块覆盖率只差 4.5 个百分点（E-SQZ6-3） | 底部几行之外的物体证据被删，模型默认回答“不存在”（与 F10 一致） | 覆盖率本身不是足够的解释：答对的样本覆盖率也不高，可能是物体恰好落在底部 | 没有物体框 | 补 COCO 物体框，按“题目物体是否落在保留区域”分组 |
| 机制、消融和交互 | 位置曲线：最后一行最高、越往上越低，第 0 行有一个小峰（E-SQZ6-2）；不同图之间也有明显重叠（Jaccard 0.108 / 0.359） | 最后一个 token 在序列上离图像底部最近，RoPE 衰减使它更关注这些 token；第 0 行的小峰可能是 attention sink | 因果注意力本身的位置效应；第 1 层的注意力还没有形成跨模态对齐 | 没有“纯位置”的对照 | 用空白图像测分数曲线；按 SparseVLM+ 的 RoPE 引力公式校正后再测（P-SQZ6-1、P-SQZ6-2） |
| 子组、切片和边界 | keep 30 与 166 的结论一致；FastV-192 每道题几乎保留整行最后一行（约 23/24），上半幅只保留 13% 的 patch | 预算越大，位置偏差占用的“固定名额”越多 | — | 只测了 K=2 | 更深的 K（例如 8、16）是否仍与问题无关 |
| 稳健性及替代解释 | 随机对照的统计量全部符合期望；两档 keep 方向一致 | 结论不是分析工具造成的 | 子集的图像构图（物体多在画面中部）可能放大“底部”的效果 | 20 张图 | 在更多图像上复验（可直接用 EXP-0002 的全集重跑一次 FastV-64 记录位置，约 20 分钟） |
| 局部收益、整体收益和代价 | 不涉及速度；FastV 与随机的速度相同 | — | — | — | — |

## 学到什么

- **假设**：H1、H2、H3 成立；H4 显示 FastV 在 K=2 时的选择基本与问题无关。这与 FastV 论文“用最后一个 token 的注意力选与问题相关的视觉 token”的直觉相反（论文正文描述的打分方式本身也与代码不同，见 MODE_C-0002 的 E-RC-2）。
- **已排除与仍无法区分的解释**：
  - 已排除：分析工具或钩子造成的位置假象（随机对照符合期望）。
  - 仍无法区分：RoPE 衰减、attention sink、因果注意力的其他位置效应。
- **保留 / 复验 / 回退 / 停止**：保留 F15 作为本项目的直接证据；P5（多轮复用）对 FastV 的担心在 K=2 时基本不成立，因为它的选择本来就不看问题。
- **最小、最能区分的下一次实验**：用空白（或噪声）图像在同一批问题上跑 FastV，得到与内容无关的分数曲线；再把真实图像的分数除以这条曲线后选 token，比较保留分布与子集上的 yes 比例（2～3 次诊断运行）。
- **回流 Deep Read**：F15、P-SQZ6-1、P-SQZ6-2、M1 的状态更新与 E-SQZ6-1～4 交给 MODE_C-0002（C9）。

## 同步清单

- [ ] experiment-results.md：登记 EXP-0006 的 4 次运行与本报告。
- [ ] 已发现的实验现象.md：新增 F15。
- [ ] 论文素材库.md：登记候选图（FastV 与随机保留位置的热力图）。
- [ ] MODE_C-0002：C9 状态更新，M1 状态更新，新增 F15。
- [ ] 最新实验分析报告.md、index.md。

（2026-10-29 完整阅读补注：上面五项实际都已完成，只是没有勾选：experiment-results 的 EXP-0006 行与 EXP-0006-R01～R04 四行；已发现的实验现象 F15；论文素材库 MAT-0006（保留位置热力图）；MODE_C-0002 回流 1（F15、M1、C9、P-SQZ6-1/2）；最新实验分析报告“EXP-0006”一节与 index 中的两条登记。）
