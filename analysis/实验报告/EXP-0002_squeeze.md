# SQUEEZE_REPORT：视觉 token 剪枝 GPU 复现的深挖（EXP-0002）

状态：完成（2026-10-24）。
报告 ID SQZ-EXP-0002 / 实验 ID EXP-0002 / 输入记录：analysis/实验报告/EXP-0002_record.md（2026-10-24 版，commit b333391）/ 日期 2026-10-24。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0002
squeeze_id: SQZ-EXP-0002
created_at: 2026-10-24
source_ids: [EXP-0002, SQZ-EXP-0001, MODE_C-0001, PAPER-0004, PAPER-0005, PAPER-0006]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0002_record.md
- code/output/EXP-0002/analysis/（tables.md、summary.json；SHA256SUMS 的 sha256 bf1e369c…）
- code/output/EXP-0002/squeeze/checks.txt（由 code/token_pruning/squeeze_exp0002.py 生成；SHA256SUMS 的 sha256 d26e736c…）
- source/SparseVLMs@734d9003：llava/model/language_model/modelling_sparse_llama.py 第 265～313 行、score.py 第 5～9 行、llava/model/builder.py 第 114～121 行
- 备份：D:\6CJNM007\backup\EXP-0002\（49 个运行目录、analysis、analysis_v1、squeeze，复制后逐文件核对）
missing_evidence:
- 被保留 token 的空间位置（patch 索引）没有记录，FastV 劣于随机选择的机制无法直接检验
- 没有 COCO 物体框（大小、位置），按物体的分析分不开“物体小”和“物体不显著”
- 随机选择的 192、128 档只有一个种子
- SparseVLM 只在 SDPA 替代下运行，没有官方 FlashAttention-2 的逐答案对照
- 只有 POPE、MME 两个 benchmark
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0002]
claim_scope: LLaVA-1.5-7B，POPE（8910 题）与 MME（2374 题），贪心解码，batch 1，32 层平均预算 192/128/64；FastV 只限 K=2、keep 按层平均换算的设置；本环境的 sdpa 注意力（SparseVLM 为 SDPA 替代）
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 见“主要结果”
process_dynamics: 无训练过程；6 小时内重复运行的答案逐条相同，没有漂移（深度分析第 2 行）
mechanism_and_intermediate_findings: 见“深度分析”第 5 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 3、4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 6 行
new_P: [P-SQZ2-1, P-SQZ2-2, P-SQZ2-3, P-SQZ2-4]
new_F: [F10, F11]（另更新 F4、F8）
new_M: [M-SQZ2-1（AI）]
evidence_index: 见 mode_C_input_bundle.E
open_questions: [FastV 选中的 token 在空间上是否集中, “no”偏置来自证据被删还是语言先验, 按实际 token 数对齐后 SparseVLM 与 VisionZip 的比较, 各选择规则分别保住哪类能力]
next_routes: [记录 FastV 与随机选择保留的 patch 索引（50 道 POPE，2 次诊断运行）, 按实际 token 数对齐的 VisionZip 对照（POPE，3 次运行）, deep-read Mode C 回流（合并 视觉Token压缩_ModeC.md 与 MODE_C-0001）]
mode_C_input_bundle:
  source_ids: [EXP-0002, SQZ-EXP-0002]
  P:
  - id: P-SQZ2-1
    text: "同一层平均预算下，FastV（K=2，按第 1 层最后一个 token 的注意力选）为什么不如均匀随机选择：是选中的 token 在空间上集中，还是选中了与内容无关的 sink 类 token"
    evidence: [E-SQZ2-1, E-SQZ2-3]
    status: UN（现象 DE，机制未测）
  - id: P-SQZ2-2
    text: "剪枝后模型更常答 no：是物体证据被删掉，还是视觉 token 太少时语言先验占了主导"
    evidence: [E-SQZ2-2]
    status: UN
  - id: P-SQZ2-3
    text: "SparseVLM 实际每层 token 多于名义预算；按实际 token 数对齐后，它与 VisionZip 的比较是否改变"
    evidence: [E-SQZ2-5, E-SQZ2-7]
    status: DE（口径差异是事实）；对比较结论的影响 UN
  - id: P-SQZ2-4
    text: "方法排序随 benchmark 和子任务改变：不同选择规则分别保住哪类视觉能力（物体存在、计数、颜色、人脸、文字）。是 MODE_C-0001 P4 的细化"
    evidence: [E-SQZ2-4, E-SQZ2-7, E-SQZ2-8]
    status: DE（现象）；机制 AI
  F:
  - id: F4（更新，台账已有此 ID）
    text: "实测每层视觉 token（EXP-0002-R47～R49，50 道 POPE 完全相同）：SparseVLM 192 为 576×3、309×4、204×9、113×16；128 为 576×3、312×4、117×9、39×16；64 为 576×3、82×4、32×9、18×16。层平均 206.5/145.4/82.25，比名义预算多 7.6%/13.6%/28.5%。SQZ-EXP-0001 按预算表推算的 188.2/123.9/61.7（比名义少 2～4%）被实测取代，原因有两个：剪枝层本身用剪枝前的 token 计算，剪枝从下一层起生效；每次剪枝后追加 int((int(0.3d)+1)/10)+1 个回收聚类 token（d 为被删数）。公式与 9 个分段全部吻合。FastV-64 实测 576×2、30×30，层平均 64.125，与设计一致"
    evidence: [EXP-0002-R46～R49, E-SQZ2-5]
    status: DE
  - id: F8（更新，MODE_C-0001 中为 SC）
    text: "本实验中随机选择在 POPE 三档都优于 FastV（K=2），在 MME 的 64 档也更好，192、128 档没有可分辨的差异；但随机选择在所有预算、两个 benchmark 上都不如 VisionZip 和 SparseVLM（例如 64 档 POPE 比 VisionZip 低 16.8）。F8 中“随机优于 SparseVLM”的部分在本设置下不成立"
    evidence: [E-SQZ2-1, E-SQZ2-7]
    status: DE（限本设置）
  - id: F10（新）
    text: "剪枝使 LLaVA-1.5-7B 在 POPE 上更常答 no：yes 比例随预算单调下降（vanilla 42.6%；FastV 32.7/27.3/11.6%；随机 37.5/34.3/23.5%；VisionZip 43.3/40.4/34.7%；SparseVLM 41.9/41.5/35.0%）。F1 的下降主要来自召回（FastV-64 召回 22.3%，精确率 93.8%～98.5%）。全部 POPE 答案都以 yes/no 开头，不是格式问题"
    evidence: [E-SQZ2-2, E-SQZ2-6]
    status: DE
  - id: F11（新）
    text: "方法与子任务有交互：同预算下随机选择在 MME celebrity（64 档 −57.9～−66.2）和 OCR（−17.5～−27.5）上的损失远大于 FastV（celebrity −32.4；OCR 0/−2.5/−15），FastV 则在 existence、count、color、position 上损失更大（64 档 −73.3～−95）。方法排序也随 benchmark 改变：192 档 FastV 的 POPE 比 VisionZip 低 9.7，MME-P 却高 21.7（区间跨 0）"
    evidence: [E-SQZ2-4, E-SQZ2-7]
    status: DE（OCR 只有 40 题、existence 等各 60 题，子任务差异的不确定性大）
  M:
  - id: M-SQZ2-1
    text: "均匀随机覆盖保住了较多物体的“存在”证据（召回更高），但把人脸、文字这类需要连续局部细节的区域切碎；注意力选择集中在少数位置，保住这些位置的细节，却丢掉其他物体的存在证据"
    evidence: [E-SQZ2-3, E-SQZ2-4, E-SQZ2-8]
    status: AI（推断；保留 token 的空间分布未测）
  E:
  - id: E-SQZ2-1
    text: "FastV − 随机，POPE 配对 bootstrap 95% 区间：192 [−5.51, −3.66]，128 [−9.43, −7.31]，64 三个种子 [−26.05, −22.30]/[−26.03, −22.31]/[−25.44, −21.71]；MME-P：192 [−20.73, 108.74]，128 [−63.02, 77.73]，64 [−264.03, −116.43]/[−259.79, −113.55]/[−220.90, −74.24]"
    location: code/output/EXP-0002/analysis/tables.md
  - id: E-SQZ2-2
    text: "POPE 各子集的 TP/FP/TN/FN、精确率、召回率与 yes 比例；例如 vanilla 召回 79.13%、FastV-64 22.27%、随机-64（种子 2021）43.5%/44.5%/44.7%"
    location: code/output/EXP-0002/analysis/summary.json（configs.*.pope.splits）
  - id: E-SQZ2-3
    text: "同一批 POPE 题上 FastV 与随机（种子 2021）的逐题对照：正样本只有随机答对 591/826/1339 题、只有 FastV 答对 247/299/352 题（192/128/64）；负样本只有 FastV 答对 119/122/104 题、只有随机答对 33/28/26 题"
    location: code/output/EXP-0002/squeeze/checks.txt
  - id: E-SQZ2-4
    text: "MME 14 个子任务相对 vanilla 的变化（全部配置）"
    location: code/output/EXP-0002/analysis/tables.md（MME 子任务表）
  - id: E-SQZ2-5
    text: "SparseVLM 每层视觉 token 的实测值（R47～R49），以及剪枝位置与回收 token 数的代码依据"
    location: code/output/EXP-0002/EXP-0002-R47～R49/layer_tokens.json；modelling_sparse_llama.py 第 265～313 行；score.py 第 5～9 行
  - id: E-SQZ2-6
    text: "数据质量：POPE 全部答案以 yes/no 开头，MME 每个配置最多 3 个答案不以 yes/no 开头；R02 与 R05、R04 与 R34（相隔约 6 小时、worker 数不同）答案逐条相同，R42 与 R46 的逐层计数相同"
    location: code/output/EXP-0002/squeeze/checks.txt；EXP-0002_record.md
  - id: E-SQZ2-7
    text: "方法间配对比较（探索性，未校正）：FastV-192 − VisionZip-192 POPE −9.72 [−10.60, −8.90]、MME-P +21.68 [−30.21, 69.88]；VisionZip − SparseVLM POPE +0.20/−1.95/−0.64（192/128/64），64 档 MME-P +84.64 [26.48, 146.78]；随机-64 − VisionZip-64 POPE −16.76 [−18.09, −15.35]"
    location: code/output/EXP-0002/analysis/tables.md（方法间比较表）
  - id: E-SQZ2-8
    text: "64 档按物体的召回率（random 子集的 1500 道正样本题，79 种物体）：VisionZip 与 SparseVLM 整体相同（除 person 外 0.593 vs 0.603，McNemar p=0.43），但 dining table（0.26 vs 0.44，p=0.007）、spoon（0.07 vs 0.34，p=0.008）上 SparseVLM 更高；VisionZip 在 15 种常见物体中 11 种高于随机、3 种低于随机（car、dining table、chair）、1 种相同，car 上 0.37 vs 0.52（p=0.03）。dining table、spoon、car 这三项按 15 次比较的 Bonferroni 阈值（0.0033）都不显著"
    location: code/output/EXP-0002/squeeze/checks.txt
  evidence_locations: [code/output/EXP-0002/analysis/SHA256SUMS, code/output/EXP-0002/squeeze/SHA256SUMS]
```

## 证据审计

- **输入**：EXP-0002 记录与 49 个运行目录。42 次完成的评测都有完整答案、评分日志与校验和；7 次失败没有答案（R01、R16、R18～R23，均已重跑）；R43～R45 的逐层计数有缺陷，不使用。
- **数据质量**：
  - POPE 的全部答案都以 yes 或 no 开头，每个配置的 MME 最多 3 个答案不以 yes/no 开头。分数下降不是答案格式问题。
  - 同一配置重复运行的答案逐条相同：R02 与 R05（0 个与 4 个 worker）、R04 与 R34（相隔约 6 小时）。R42 与 R46 的逐层计数也相同。运行之间没有漂移。
  - 49 个 git.txt 中，只有 R04 记录了一个当时未提交的 `bootstrap_ci.py`（评测不使用它）；其余运行开始时本仓库没有未提交改动，外部仓库都是 clean。
- **数据血缘**：COCO val2014、POPE 标注与 MME 图像的校验和在 `D:\6CJNM007\data\downloads\SHA256SUMS`；题目文件来自 LLaVA eval.zip；评分用官方 eval_pope.py 与 MME calculation.py。官方 13B 预测打分与 MODEL_ZOO 一致。
- **预先计划与事后提出**：
  - H0、H1、主指标、bootstrap 和随机种子规则在计划 v1 中写定。
  - 以下都是看到结果后提出的，属于探索性分析：逐层计数诊断、方法间配对比较、FastV 与随机的逐题对照、按物体的召回率。多数没有做多重比较校正，在下文标为“探索性”。

## 主要结果

| 结果 | 基线/对照 | 差异或效应 | 不确定性 | 实际意义 | 证据位置 |
| --- | --- | --- | --- | --- | --- |
| H0：vanilla POPE | 官方 85.9 | 85.90，差 0.00 | [85.09, 86.66] | 管线在 POPE 上可靠 | R04、R34 |
| H0：vanilla MME-P | 官方 1510.7 | 1507.63，差 −3.07（eager 为 +3.86） | 区间约 ±65；只换注意力内核就移动 6.93 | 字面门槛（1 分）未通过，但差异小于这项评测的数值分辨率 | R02、R03、R05 |
| 复现：VisionZip、SparseVLM 的 POPE | 各自论文 | 三档都相差 ≤0.2 | 区间约 ±0.8 | 两种方法在本环境可复现 | R13、R15、R17、R37、R39、R41 |
| H1：FastV − 随机（POPE） | 同预算随机 | −4.6 / −8.4 / −23.6～−24.2（192/128/64） | 区间全在 0 以下（E-SQZ2-1） | 注意力选择在 POPE 上比随机差，预算越低差得越多 | E-SQZ2-1 |
| H1：FastV − 随机（MME-P） | 同预算随机 | +47.3 / +8.2 / −142.9～−188.6 | 192、128 档区间跨 0 | MME 上只有 64 档可分辨（随机更好） | E-SQZ2-1 |
| 代价：每题秒数 | vanilla 0.239（POPE） | 剪枝后 0.128～0.185，快 1.3～1.9 倍 | 单次运行；含 Python 开销 | 同档预算下 SparseVLM 最慢；POPE 上 VisionZip 最快 | tables.md |

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 见“主要结果”。随机-64 三个种子的 POPE 标准差 0.32、MME-P 标准差 26.28；FastV 与随机的差（约 24 分）是种子波动的 70 多倍 | H1 的结论不依赖随机种子 | 192、128 档随机只有一个种子 | MME 的区间很宽（约 ±65），192、128 档 MME 上的方法差异多在噪声内 | — |
| 运行动态 | 没有训练；重复运行答案逐条相同，没有随时间或加载方式的漂移（E-SQZ2-6） | 贪心解码在同一环境下是确定的 | — | 只测了 sdpa 与 eager 两种内核 | — |
| 错误和失败模式 | 剪枝的主要错误是漏报（F10）。所有方法的 yes 比例都随预算下降；FastV-64 召回 22.3%，精确率仍为 93.8%～98.5%。逐题对照（E-SQZ2-3）：随机对 FastV 的优势全部来自正样本（64 档只有随机答对 1339 题，只有 FastV 答对 352 题）；负样本上 FastV 略好（104 vs 26），因为它更常答 no。adversarial 子集的误报从 vanilla 的 132 降到 FastV-64 的 22 | 视觉证据不足时，模型默认答“不存在” | 语言先验在视觉 token 少时占主导（P-SQZ2-2）；两者也可能同时成立 | 没有物体框，无法区分“物体被删”与“物体还在但证据太弱” | 用 COCO 物体框比较“删物体区域”与“删背景”（P-SQZ2-2） |
| 子组、切片和边界 | MME 子任务（F11）：随机选择在 celebrity、OCR 上损失最大，FastV 在 existence、count、color、position 上损失最大；scene 几乎不受影响（除 FastV-64 外都 +0.2～+6.8）。按物体（E-SQZ2-8，探索性）：VisionZip-64 与 SparseVLM-64 整体召回相同，但丢失的物体不同；FastV-64 在 chair、dining table、spoon 上召回为 0～0.02 | 不同选择规则保住不同的信息：均匀覆盖保住“存在”，集中选择保住少数位置的细节（M-SQZ2-1，推断） | 子任务样本量小（OCR 40 题、existence 等 60 题），部分差异可能是噪声；按物体的 15 次比较未校正 | 按物体的分析没有物体大小信息 | 记录保留 patch 的空间分布（P-SQZ2-1）；补 COCO 物体框后按物体大小分层 |
| 机制、消融和交互 | SparseVLM 每层实际 token 比名义预算多 7.6%/13.6%/28.5%（F4 更新，E-SQZ2-5）：剪枝层本身用剪枝前的 token，再加回收 token；回收数只由预算决定，与图像无关。按实际 token 数看，SparseVLM-64 用了 82 个 token，POPE 只比 VisionZip-64（64 个）高 0.64（区间跨 0），MME-P 反而低 84.6；SparseVLM-128 用了 145 个，POPE 比 VisionZip-128 高 1.95 | SparseVLM 在名义预算下占了口径的便宜；按实际 token 对齐后，它在 128 档的 POPE 优势可能缩小或消失 | 文本引导打分本身更好（P-SQZ2-3） | 没有在 82/145/206 个 token 下跑 VisionZip | 按实际 token 数对齐的 VisionZip 对照（3 次 POPE 运行） |
| 稳健性及跨数据集 | 加载方式（0 vs 4 个 worker）不改变答案；注意力内核（sdpa vs eager）使 MME-P 移动 6.93；随机种子影响很小（见第 1 行）；只有 POPE、MME | 方法间比较在同一环境内，固定的数值偏移不影响排序 | 其他 benchmark（GQA、TextVQA）上排序可能不同（F11 已显示排序随 benchmark 改变） | 两个 benchmark；一个模型 | 至少再加一个以 OCR/细节为主的 benchmark（TextVQA）再下结论 |
| 局部收益、整体收益和代价 | 每题快 1.3～1.9 倍；POPE 上同档预算 VisionZip 最快（在 LLM 之前就压缩），两个 benchmark 上 SparseVLM 都最慢，且多占约 4 GiB 显存（每层复制一份注意力权重）。“误报下降”只是少答 yes 的副作用，不是幻觉减少 | 速度主要由 LLM 中的 token 数决定；batch 1 下生成循环是瓶颈（GPU 利用率约 38%） | — | 速度含 Python 开销，只适合同一环境内比较 | 需要准确的预填充延迟时，单独计时 |

## 解释与不确定性

- **最支持的解释**：在 K=2、keep 按层平均换算的设置下，FastV 的注意力打分选中的 token 不能覆盖题目所问的物体；均匀随机至少保留了每个区域的一部分 token，所以召回更高。
- **仍可成立的替代解释**：
  - 注意力打分偏向固定位置（MODE_C-0001 的 M1，RoPE 远程衰减）或 sink 类 token，与题目无关。
  - K=2 太早，第 1 层的注意力还没有形成有意义的跨模态对齐；换成更深的 K 结论可能不同。
- **当前证据不支持的强结论**：
  - “FastV 方法本身不好”：只测了一种设置。
  - “剪枝减少幻觉”：误报下降来自整体少答 yes。
  - “SparseVLM 优于 VisionZip”：128 档 POPE 的优势是在多用 13.6% token 的情况下得到的，64 档 MME 上 VisionZip 反而更好。

## 学到什么

- **假设**：H0 在 POPE 上通过；MME-P 字面未通过，但差异在数值分辨率之内。H1 得到支持，方向是“同预算下随机不比 FastV 差，POPE 上更好”。
- **已排除与仍无法区分的解释**：
  - 已排除：答案格式错误、数据加载方式、运行漂移、随机种子波动。
  - 仍无法区分：注意力选择“空间集中” vs “位置偏差” vs “选中 sink token”；“no”偏置来自证据被删还是语言先验。
- **保留 / 复验 / 回退 / 停止**：
  - 保留 FastV 作为主 baseline，但以后每次都配随机选择对照。
  - 比较 SparseVLM 时同时报告名义预算和实测层平均。
  - POPE 结果同时报告 F1、yes 比例和召回率。
- **最小、最能区分的下一次实验**：在 50 道 POPE 上记录 FastV 与随机选择保留的 patch 索引（24×24 网格），比较空间分布和与题目物体的关系。只需 2 次约 1 分钟的诊断运行，可以直接检验 M-SQZ2-1 和 P-SQZ2-1。
- **回流 Deep Read**：P-SQZ2-1～4、F4 与 F8 的更新、F10、F11、M-SQZ2-1 和 E-SQZ2-1～8 交给 Mode C 合并（见交接包）。

## 同步清单

- [x] experiment-results.md：登记 EXP-0002 的 49 次运行与本报告（main 3aa5d6b 起）。
- [x] 已发现的实验现象.md：F4 写入实测每层计数（取代按预算表的推算）；新增 F8（本实验证据）、F10、F11。
- [x] 论文素材库.md：登记候选图表（MAT-0001～0005）。
- [x] 最新实验分析报告.md：指向本报告。
- [x] version-evolution.md：研究方向尚未改变，方向选择在研究流程第 ⑦ 步，不需要新条目。
- [x] index.md：登记本报告与 EXP-0002 记录。

## 补充分析：不同方法是否漏掉同一批物体（2026-10-24，针对 F10）

由 `code/token_pruning/analyze_missed_objects.py` 生成（`code/output/EXP-0002/deepdive/missed_objects.md`、.json；SHA256SUMS 的 sha256 f773f035…，附脚本副本，已备份）。POPE 三个 split 的 1500 个正样本完全相同（同一组图像—物体对），一个正样本被“漏答”指回答 no（与 eval_pope.py 同样计分）。集合比较用 adversarial split 的回答；按物体的漏答率取三个 split 的平均。

- vanilla 本身漏掉 20.9% 的正样本；VisionZip-192（20.7%）、SparseVLM-192（21.9%）与它相同，64 档各方法漏掉 34.2%（SparseVLM）～77.7%（FastV）。
- 剪枝后漏掉的正样本几乎包含 vanilla 漏掉的：vanilla 的漏答中，各方法也漏掉 84.0%～96.8%。
- 不同方法的漏答高度重叠，远超独立漏答的期望：VisionZip-64 与 SparseVLM-64 的 Jaccard 0.652（独立时期望 0.209），vanilla 与 VisionZip-64 0.498（0.151），随机-64 与 SparseVLM-64 0.475（0.271）。
- 按物体（正样本 ≥ 10 个的 43 类），64 档最难的是勺子、杯子、椅子、钟、餐桌、叉子、长椅、背包，狗、公交车、披萨几乎不漏；各方法按物体漏答率的 Spearman 相关 0.50～0.85，与 vanilla 0.34～0.72。
- 解释：剪枝造成的“答 no”集中在 vanilla 本来就容易漏掉的正样本上，多是杂乱场景里的小物体。与“物体证据被删”一致：小物体只占少数 patch，最容易被剪掉；也与“这些正样本本来就在判断边界上，任何扰动都会翻转”一致（推断，两者需要答案概率或物体框才能区分）。
- 新现象 F18（待登记）；新问题 P-SQZ2-4：剪枝方法的差别是否主要体现在“边界正样本”上，若用 vanilla 的 yes 概率分层，剪枝的损失是否集中在低置信度的那一层。
