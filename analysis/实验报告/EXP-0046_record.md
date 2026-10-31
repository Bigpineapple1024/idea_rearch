# EXPERIMENT_RECORD：IDEA-0011：修好 VisionZip-64 上的 fill 再检验；POPE 上 CDPruner-64 的分辨力（开发集与 POPE）

状态：计划 v1（2026-11-01 凌晨 3 点半后，用户决定之后、任何运行之前写定；DEC-0092）。写时已看过 EXP-0041～EXP-0043 的全部结果与 EXP-0044 的 GPU 冒烟测试（VisionZip-64 加 fill 在 GPU 上跑不通，DEC-0091）；EXP-0044 正在运行，没有看过它的任何结果。

- 实验 ID：EXP-0046。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)（C40、C42）；用户决定 DEC-0092（原话：“1和2都可以做”）。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（换底座：VisionZip-64，修好 vzfill.py 之后）**：VisionZip-64 加 fill − VisionZip-64（EXP-0044 计划 v1 的 Q1，原样搬过来；那里因 GPU 上的数据类型问题没有运行）。
2. **Q2（POPE 上的分辨力）**：CDPruner-64 在 POPE 上重跑并记录第一个答案 token 的对数几率，与 fill（EXP-0042-R07，已记录）比：是非题上 gt yes 对 gt no 的 AUC，fill − CDPruner-64。
3. **Q3（重跑是否相同）**：CDPruner-64 的 POPE 重跑（R05）与 EXP-0042-R08 的答案是否相同；VisionZip-64 的重跑（R01、R02）与 EXP-0028-R01、R02 的答案是否相同（描述）。
4. **同时报告（描述）**：Q1 的同 EXP-0044 计划 v1（不剪枝 − VisionZip-64 加 fill；按题型、GQA 位置题；是非题按标准答案的答“是”比例；与不剪枝的回答相同的比例；VisionZip-64 与它加 fill 的 AUC；VisionZip 布局的统计；FLOPs（577 个槽）；图像 KV 缓存：普通实现是缓存里的 64 个加旁边 577 个槽，(64 + 577) / 64 倍；每题秒数与峰值显存）。Q2 的：三个分集各自的 AUC；“对齐”的比较——fill 答“是”的题数对齐到 CDPruner-64，以及反过来把 CDPruner-64 对齐到 fill（阈值在同一批题上定，只作描述），对齐后与对方的准确率差；两个运行的对数几率正负与计分的是 / 否是否一致。

### 设计

- **vzfill.py 的修复**：投影之前把 CLIP 第 −2 层的状态转成投影层参数的数据类型（GPU 上是半精度；VisionZip 自己送进 LLM 的 token 也是半精度投影的）；其余不变。check_vzfill 加一项：状态的数据类型与投影层不同时，得出与相同数据类型时一样的投影（CPU 上用 float64 的状态对 float32 的投影层）。
- 其余同 EXP-0044 计划 v1 的设计（VisionZip 上的 fill：CLS 占网格之前自己的槽，主要 patch 与上下文 token 的目标在自己的槽，其余槽放最相似的非 CLS token，共 577 个槽；VisionZip 送进 LLM 的 64 个 token 不变）。
- Q2：fill 的 POPE 运行（EXP-0042-R07，代码 cea78da）已记录对数几率；之后 vfill.py 只加了紧凑实现与选法自给的布局，不加新参数时普通 fill 的计算不变（check_vfill 核对），所以 R07 的对数几率可直接比较。CDPruner-64 的重跑用 `cdpruner64_logits`（只多记对数概率，不改计算）。

### 数据

- VQAv2 dev2（5192 题）与 GQA val400（4786 题），开发集（P14）；POPE（8910 题，三个分集）。

### 配置与运行（5 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `visionzip64_logits`（--method visionzip --dominant 54 --contextual 10，记对数概率） | vqav2_dev2、gqa_val400 |
| R03、R04 | `visionzip64_fill_logits`（同上加 --layout fill，经修好的 vzfill.py） | 同上 |
| R05 | `cdpruner64_logits`（--method cdpruner --cdpruner-keep 64，记对数概率） | pope |

- 复用：fill 的 POPE（EXP-0042-R07）、CDPruner-64 的 POPE（EXP-0042-R08，Q3 的对照）、不剪枝（EXP-0033-R03、R04；POPE 为 EXP-0002-R34）；VisionZip-64 的旧运行（EXP-0028-R01、R02）只作 Q3 的对照。失败的运行保留，用新编号（R06 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- Q2：AUC(fill) − AUC(CDPruner-64)（POPE 全部 8910 题；按图像重抽 1000 次，种子 0，analyze_0036 的图像顺序）：95% 区间整个在 0 以上 →“fill 的分辨力更高”；整个在 0 以下 →“fill 的分辨力更低”；否则 →“分辨力的差别没有检出”。
- Q3：答案每题都相同 →“逐位相同”；每个比较 ≥ 99.5% 的题相同 →“等价”；否则 →“不等价”（Q2 照样用 R05 比，结果里写明）。
- **之后**：Q1“有希望”，或“部分”且两集平均 ≥ +0.5 → 在新数据上确认（[EXP-0047](EXP-0047_record.md)，计划 v1 与本计划同时写定）；否则记录“在 VisionZip-64 上没有检出”。Q2 写进 F112 的更新：“没有检出”且对齐后的准确率差的区间含 0 →“POPE 上 fill 对 CDPruner-64 的损失与门槛移动一致，没有检出分辨力的差别（不等于相同）”；“更低” →“损失里有分辨力的下降”；“更高” →“fill 分辨得更好，损失来自门槛”。
- 区间：每个集内按图像重抽 1000 次（种子 0；开发集按 analyze_0028 的图像顺序，POPE 按 analyze_0036）。

### 可检验的预测（推断，只为事后对照，不是判据）

- Q1：同 EXP-0044 计划 v1：VQAv2 0～+2、GQA +1～+4，GQA 位置题 > +5；方向不确定的部分在 VQAv2（上下文 token 是合并出来的）。
- Q2：AUC 差在 ±0.01 之内，“没有检出”（开发集上 CDPruner-128、-192 的 fill 的 AUC 都略高于底座，F116；EXP-0042 里 fill 对齐到 CDPruner-64 后 −0.34 [−1.04, +0.33]，F112）。
- Q3：两处都“逐位相同”。

### 实现核对（门槛 ≥ 99%，analyze_0046.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率（POPE：analyze_0036 的计分）；VisionZip 加 fill 的布局日志是 fill、每题一条（35 个 token 之后 64 个图像 token、577 个槽、位置是预期的、32 层都用了布局、槽 0 是 CLS、53 个主要与 10 个目标的槽、每个 token 的槽指向自己）；比较的两个运行提示相同；记录对数概率的运行每题一条；POPE 上 R05 留下的 patch 与 EXP-0042-R08 相同（每题）；两个 POPE 运行的对数几率正负与计分的是 / 否一致（≥ 99%）。
- CPU（tmp 的代码副本）：check_vzfill 14 / 14（加一项）；check_vfill 56 / 56 与回归 check_cdpruner 43 / 43、check_divprune 26 / 26、check_vzcover 45 / 45 照旧。
- GPU 冒烟测试（EXP-0045 结束后、开门前；tmp，不是 EXP 运行）：VQAv2 dev2 前 8 题的 VisionZip-64 与它加 fill，POPE 前 8 题的 cdpruner64_logits；布局日志正确；答案不计分、不看。analyze_0046.py 先用替身空跑。

### 代码（分支 code/token-pruning，在 EXP-0045 的代码提交之上，`D:\6CJNM007\tmp\exp0046\code\`）

- vzfill.py（投影前转数据类型）、check_vzfill.py（加一项）、analyze_0046.py（新）。其余不变。

### 预算与顺序

- 5 次，约 1.5 GPU 小时。IDEA-0011 合计（EXP-0041～EXP-0046）最多 53 次，加 EXP-0047 最多 4 次，合计最多 57 次；GPU 小时估计最多约 18.7（上限 60 次、20 GPU 小时，DEC-0085、DEC-0086）；累计要超过上限之前先停下问用户。
- 顺序：EXP-0044 → EXP-0045 → 本实验 → EXP-0047（条件性）。
