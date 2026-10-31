# EXPERIMENT_RECORD：IDEA-0011 的推广与实现（VisionZip-64 加 fill；紧凑的 KV 缓存，开发集）

状态：计划 v1（2026-11-01 凌晨（更正：实际写于 2026-10-31 晚上 11 点半前后，原写“2026-11-01 凌晨”）），在 EXP-0044 的任何运行之前写定；写时 EXP-0043 正在运行（只看过运行号、状态与答案数），没有看过 EXP-0043 的任何结果。

- 实验 ID：EXP-0044。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)（C40）；用户决定 DEC-0083；DEC-0085。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（换底座：VisionZip-64）**：VisionZip-64 加 fill − VisionZip-64。
2. **Q2（紧凑的 KV 缓存）**：CDPruner-64 加 fill 的紧凑实现（`--layout-compact`）对普通实现（EXP-0041-R01、R02）：答案与第一个答案 token 的对数概率是否相同。
3. **同时报告（描述）**：不剪枝 − VisionZip-64 加 fill；按题型、GQA 位置题；是非题按标准答案的答“是”比例；与不剪枝的回答相同的比例；是非题上第一个答案 token 的对数几率的 AUC（VisionZip-64 与它加 fill 都记录）；VisionZip 布局的统计；FLOPs（577 个槽）；按本流程的实现的图像 KV 缓存（普通实现：缓存里 64 个保留 token 加旁边 576 个槽，10 倍；紧凑实现：1 倍）；紧凑实现、普通实现与 CDPruner-64 的每题秒数与峰值显存；本流程中 VisionZip-64 重跑与 EXP-0028 的答案是否相同。

### 设计

- **VisionZip 上的 fill（vzfill.py）**：VisionZip-64（官方代码 source/VisionZip，只读、不改）送进 LLM 的 64 个 token 是 CLS、53 个主要 patch（按 CLS 注意力，按编号顺序）、10 个上下文 token（目标 patch 加与它最相似的其余 patch 的平均）。布局：CLS 占网格之前自己的槽（槽 0，位置 start），patch p 占槽 1 + p（位置 start + 1 + p）；主要 patch 在自己的槽，上下文 token 在它的目标 patch 的槽；其余 patch 槽放投影后特征（第 −2 层的状态过投影层，即不剪枝的 LLaVA 给这个 patch 的输入）与 64 个 token 余弦最大的非 CLS token 的 K、V；CLS 不代表任何 patch；图像之后的文字按 577 个图像 token 编号。这是一个设计选择：VisionZip 的 CLS 与合并出来的 token 在 LLaVA 训练时没有 patch 位置，上面是与 fill 最接近的做法。VisionZip 送进 LLM 的 64 个 token 不变（check_vzfill 核对）。
- **紧凑的 KV 缓存**：fill 里 576 个槽在每一次注意力里代替图像 token 的 K，所以缓存里图像部分旋转后的 K 从来不读；紧凑实现把那里换成旋转前的 K，每个解码步用与预填充相同的取值与旋转现算 576 个槽的 K（V 取缓存里的）；旁边不再另存槽。图像部分的 KV 缓存回到 n 个（与底座相同），运算相同，所以输出相同（CPU 上逐位相同，check_vfill）。

### 数据

- VQAv2 dev2（5192 题）与 GQA val400（4786 题），开发集（P14）。

### 配置与运行（6 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `visionzip64_logits`（--method visionzip --dominant 54 --contextual 10，记对数概率） | vqav2_dev2、gqa_val400 |
| R03、R04 | `visionzip64_fill_logits`（同上加 --layout fill，经 vzfill.py） | 同上 |
| R05、R06 | `cdpruner64_fillc_logits`（--method cdpruner --cdpruner-keep 64 --layout fill --layout-compact） | 同上 |

- 复用：普通实现的 fill（EXP-0041-R01、R02）、CDPruner-64（EXP-0039-R01、R02）、不剪枝（EXP-0033-R03、R04）；VisionZip-64 的旧运行（EXP-0028-R01、R02）只作重跑的对照。失败的运行保留，用新编号（R07 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- Q2：两个集每一题的答案与对数概率都相同 →“逐位相同”；每个集 ≥ 99.5% 的题答案相同 →“等价”（个别不同来自 GPU 运算顺序）；否则 →“不等价”。
- **之后**：Q1“有希望”，或“部分”且两集平均 ≥ +0.5 → 在新数据上确认（[EXP-0045](EXP-0045_record.md) 的 D 部分，计划 v1.1，在 EXP-0044 的任何结果之前写定）；否则记录“在 VisionZip-64 上没有检出”。Q2“逐位相同”或“等价” → 报告显存时用紧凑实现（图像 KV 缓存 1 倍）；“不等价” → 不用，查原因。
- 区间：每个集内按图像重抽 1000 次（种子 0，analyze_0028 的读法与计分）。

### 可检验的预测（推断，只为事后对照，不是判据）

- Q1：VisionZip-64 在本流程中较弱（EXP-0028：71.93 / 62.20），也是连续编号、丢布局；fill 给回布局：VQAv2 0～+2、GQA +1～+4，GQA 位置题 > +5；但上下文 token 是合并出来的，复制到空槽可能不如保留的 patch 忠实，方向不确定的部分在 VQAv2。
- Q2：“逐位相同”；紧凑实现的峰值显存接近 CDPruner-64（约 14.44 GiB），每题秒数接近普通实现。

### 实现核对（门槛 ≥ 99%，analyze_0044.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；VisionZip 加 fill 的布局日志是 fill、每题一条：35 个 token 之后 64 个图像 token、577 个槽、位置是预期的、32 层都用了布局、槽 0 是 CLS、53 个主要与 10 个目标的槽、每个 token 的槽指向自己；紧凑实现的布局日志标为 compact、每题一条（64 个、576 个槽、32 层），分配与普通实现相同，留下的 patch 与普通实现相同（每题）；比较的两个运行提示相同；记录对数概率的运行每题一条。
- CPU（tmp 的代码副本）：check_vfill 56 / 56（新加 12 项：紧凑实现在 fill、shuffle 上预填充 logits 与生成与普通实现逐位相同、旁边不存槽、缓存长度与底座相同、图像部分是旋转前的 K，参数与配置）；check_vzfill 13 / 13（新：VisionZip 的 64 个 token 不变、主要 patch 与目标与这里独立算的相同、代表、577 个槽的手算参照、贪心解码、紧凑实现、拒绝）；回归 check_cdpruner 43 / 43、check_divprune 26 / 26、check_vzcover 45 / 45。
- GPU 冒烟测试（EXP-0043 结束后、开门前；tmp，不是 EXP 运行）：VQAv2 dev2 的前 8 题，VisionZip-64、它加 fill、CDPruner-64 加 fill 的普通与紧凑实现；紧凑与普通的答案与对数概率相同；布局日志正确；答案不计分、不看。analyze_0044.py 先用替身空跑。

### 代码（分支 code/token-pruning，在 EXP-0043 的代码 ae7242e 之上，`D:\6CJNM007\tmp\exp0044\code\`）

- vfill.py（compact；选择可以直接交出布局）、vzfill.py（新）、run_eval.py（--layout-compact；--layout fill 可配 plain --method visionzip）、exp0002.py（两个配置）、check_vfill.py（加 12 项）、check_vzfill.py（新）、analyze_0044.py（新）。不加新参数时其余方法的计算不变。

### 预算

- 6 次，约 2.2 GPU 小时；IDEA-0011 合计 34 次（另有 EXP-0045 最多 20 次），上限 60 次、20 GPU 小时（DEC-0085）。
