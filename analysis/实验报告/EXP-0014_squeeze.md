# SQUEEZE_REPORT：EXP-0014（IDF 加权的题目查询：第 ⑫ 步的 A / A′ / A′+B / A+B）

深挖 ID：SQZ-EXP-0014 / 输入实验记录：[EXP-0014](EXP-0014_record.md)（[IDEA-0002](../IDEA-0002_题目词按信息量加权.md)）/ 日期：2026-10-25 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 0. 交接元数据

- 包类型：SQUEEZE_REPORT（research-pipeline-contract v1.0）。
- 来源 IDs：EXP-0014-R01～R07；参照 EXP-0011-R06、R07，EXP-0002、EXP-0010 的运行。
- 证据状态：verified。7 次运行都有 SHA256SUMS 并已备份；analysis（45d8ede7…）已封存；探索性检查有三版：squeeze（a226336d…，MME 部分为空）、squeeze_v2（34e956b7…，MME 的词用错了 prompt）、squeeze_v3（1f4257c3…，采用），原因写在各目录的 notes.md。
- 缺失证据：第二个一般题目的 benchmark（VQAv2 验证集等需下载许可）；其他预算与模型。

## 1. 事实与证据边界

- **研究问题**：不知道被问物体在题目里的位置时，用 IDF 加权全部题目 token 作查询（A′+B），能否接近物体名查询，并在一般题目（MME）上不变差。
- **设计**：A、A′、A′+B、A+B 四组，POPE 与 MME，预算与 EXP-0011 相同；IDF 在被评测的题目文件上计（不用答案与标签）；判据 S1～S3 在运行前写定。
- **已知偏差**：无；探索性检查的脚本错了两次，三版输出都保留。

## 2. 主要结果

| 结果 | 对照 | 差异或效应 | 不确定性 | 实际意义 | 证据位置 |
| --- | --- | --- | --- | --- | --- |
| POPE：A′+B 83.59 | A′ 65.85 / A 66.90 | +17.74 / +16.69 | [16.55, 19.06] / [15.44, 17.98]；RePOPE 同向 | S1 成立 | analysis/query.md |
| POPE：A′+B | 物体名查询 83.63 | −0.04 | [−0.19, +0.11]；保留集合重合 97.2% | 不用 POPE 句式就得到同样结果 | 同上 |
| POPE：A+B 66.61 | A | −0.29 | [−1.27, +0.67] | S2 成立：不适配时 B 无效 | 同上 |
| MME-P：A′+B 1236.2 | A 1288.3 | −52.1 | [−133.2, +19.1] | S3 按字面成立（区间含 0），但点估计为负 | 同上 |
| MME-P：A′+B | A′ 1028.3 | +207.8 | [+130.0, +281.6] | 适配后 B 有效 | 同上 |
| MME-P：A′+B | VisionZip-64 1364.2 | −128.1 | [−210.2, −49.8] | 一般题目上不如 VisionZip | 同上 |

## 3. 深挖发现

| 维度 | 发现 | 证据 | 稳健性 | 新信息 / 已知 |
| --- | --- | --- | --- | --- |
| 数据质量 | 4 个 IDF 运行写出的 df 表与离线重算相同；全部 17 个参照运行的重算主指标与登记值相同 | analysis/query.md | 高 | — |
| 机制 | POPE 上 IDF 把权重几乎全给物体名，A′+B 与物体名查询的保留集合重合 98.6%（物体名权重 ≥ 0.9 的 8664 题）；拼写 “imange” 的 246 题重合只有 49.6%，答案不同 4.5%（其余 0.3%），但对总分影响在区间内 | squeeze_v3 | 高 | 新（F34） |
| 机制 | 等权查询（A′）在 POPE 上与 A 相当、在 MME 上大幅变差（−260）：模板词主导查询时，按输入 embedding 选择比与题目无关的注意力还差 | analysis/query.md | 高 | 新（F34） |
| 错误 / 异质性 | MME 上 A′+B 相对 A：点名物体的子任务 existence +5、count +11.7、position +20、color +8.3（逐题：position 只有 A′+B 错 1 题、只有 A 错 7 题）；名字类 celebrity −31.5、landmark −25、posters −16.7、artwork −9.3，scene −7.2、OCR −7.5 | analysis、squeeze_v3 | 中（子任务 40～400 题） | 新（F35） |
| 错误 / 异质性 | 名字类子任务里“只有 A′+B 答错”的题，IDF 最高的 token 都是人名、地名、片名的词片（Hoff、Graham、Leon、Arena、Bor 等）：IDF 选出了“最有信息量”的词，但这些词在输入 embedding 空间里没有对应的图像区域 | squeeze_v3 | 中 | 新（F35，M14） |
| 边界 | A+B 在 MME 上 1313.3（比 A +25.0 [−25.2, +76.8]），OCR +15、position +8.3；在 POPE 上无效。第 1 层注意力行加权对一般题目有小的、不显著的帮助 | analysis/query.md | 低 | 新（描述） |

## 4. 解释与不确定性

- **最支持的解释**（AI）：IDF 衡量的是“这个词在题目集合里有多少区分度”，不是“这个词在图像里能不能看到”。点名物体的题目里两者一致；名字、标题、文字类题目里，最有区分度的词（人名、地名）没有可见的对应区域，查询就变成噪声，选出的 token 反而不如与题目无关的选择。
- **替代解释**：名字类题目本来就靠整体外观（人脸、建筑整体）回答，任何局部选择都吃亏；A 的显著性选择恰好保住了整体结构。
- **不支持的强结论**：方法在一般题目上有效；“两个 benchmark 整体稳定提升”。

## 5. 知识沉淀

- **新现象**：F34（IDF 查询在 POPE 上复现物体名查询；等权查询无效；不适配时 B 无效）、F35（MME 上点名物体的子任务受益、名字类受损）。
- **已回答**：Q1（是）、Q2（是，适配必要）、Q3（不变差按字面成立，但点估计为负，名字类受损）。
- **更新位置**：已发现的实验现象、MODE_C-0002 回流 7、结果总表、论文素材库、DEC-0026。

## 6. 下一步决策

1. **回到第 ⑨ 步**解决 P10：题目里最有信息量的词不一定在图像里看得见。机制方向：
   - 心理语言学的词“可意象性 / 具体性”（imageability / concreteness）：词能否唤起视觉形象；
   - 从图文数据估计词的视觉具体性（领域 C 候选：Hessel 等，NAACL 2018）；
   - 本项目的做法：只用开发集的 COCO 图像，估计每个 token 在图像上的相似度图是否集中（可视性），查询权重 = IDF × 可视性。人名、地名的可视性低，权重自然缩小；全部词都不可见时退回与题目无关的选择。
2. **预算**：候选 1 已用 22 次运行（上限 30），下一轮控制在 4 次以内。
3. 若下一轮仍不能让 MME 不低于 A：方法限定为“题目点名可见物体时使用”，论文中写清范围。

## 7. 回流到 deep-read Mode C 的输入包

```text
package_type: SQUEEZE_REPORT
squeeze_id: SQZ-EXP-0014
input_experiment_record_ids: [EXP-0014, EXP-0011, EXP-0002, EXP-0010]
claim_scope: LLaVA-1.5-7B，层平均 64；POPE 与 MME；IDF 在被评测的题目文件上计算
data_quality_and_provenance_gaps: 探索性检查脚本错两次，三版保留；无其他缺口
primary_result_with_uncertainty: POPE A'+B 83.59，比 A' +17.74、比 A +16.69（区间都在 0 以上），与物体名查询差 −0.04 [−0.19, +0.11]；MME-P A'+B − A −52.1 [−133.2, +19.1]
process_dynamics: 无（推理）
mechanism_and_intermediate_findings: IDF 把权重给物体名（POPE）或名字词片（MME 名字类）；后者在输入空间无对应区域
errors_heterogeneity_and_boundary_conditions: MME 点名物体子任务 +45，名字类四项 −82.4（加 scene、OCR −97.1，审计修正）；POPE 拼写 imange 的 246 题保留集合重合 49.6%
robustness_sensitivity_and_alternative_explanations: 两种 POPE 标注一致；MME 区间宽；名字类题本来就依赖整体外观
new_P: [P10：题目里最有信息量的词不一定在图像里看得见，IDF 查询在这类题上变成噪声]
new_F: [F34, F35]
new_M: [M14：IDF 衡量区分度，不衡量可见性；名字词片在输入 embedding 空间里没有对应区域（AI，squeeze_v3 支持）]
evidence_index: [code/output/EXP-0014/analysis/query.md, squeeze_v3/squeeze_query.md]
open_questions: [按可视性加权能否在 MME 名字类上退回 A 的水平而保住物体类的收益, 第二个一般题目 benchmark]
next_routes: [deep-read Mode C 回流, 第 ⑨ 步：词的可视性]
next_skill: deep-read Mode C
```
