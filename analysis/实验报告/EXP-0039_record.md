# EXPERIMENT_RECORD：本流程中的 CDPruner-64（强对照的校准，开发集）

状态：已完成（2026-10-31 11:20:32～11:55:19，2 次，代码 fc76f1a；分析 11:55:59 封存；结果见文末“结果”，DEC-0076）。前一状态：计划 v1（原写“2026-10-31 中午”，按提交时间为上午 11 点 18 分），在 EXP-0039 的任何运行之前写定。写计划时已看过 EXP-0034～EXP-0038 的全部结果。

- 实验 ID：EXP-0039。
- 来源：用户 2026-10-31 中午的选择“接入 CDPruner 做强对照”（DEC-0075）；MODE_C-0002 回流 25 的 C30（要先在本流程中实现 CDPruner）；P13（文献报告的最强免训练结果）。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（主要）**：CDPruner-64 − DivPrune-64（都在 LLM 之前留 64 个、按原顺序连续编号）。
2. **同时报告（描述）**：CDPruner-64 − VisionZip-64、DivPrune-64 − VisionZip-64（复现 F72）、不剪枝 − CDPruner-64；按题型（VQAv2 答案类型，GQA 结构与语义类型，被路由与不路由的题）；GQA 位置题（细分题型以 position 开头，505 题，F89、F100）；是非题按标准答案分开的答“是”比例（F88、F99）；每题秒数与峰值显存。

### 数据

- VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图），开发集（P14）。

### 配置与运行（2 次；每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `cdpruner64`（--method cdpruner --cdpruner-keep 64） | vqav2_dev2、gqa_val400 |

- CDPruner 的做法（cdpruner.py，从官方代码 source/CDPruner 9541616c 移植，Apache-2.0）：视觉塔倒数第 2 层的 patch 特征经 CLIP 的 post_layernorm 与 visual_projection，与题目的 CLIP 文本嵌入算负余弦，在本图的 patch 上做 min-max 归一，得到相关性；核为“相关性 × 投影后特征的余弦 × 相关性”；条件 DPP 的贪心 MAP 选 64 个；留下的按原顺序、连续编号送进 LLM（与官方相同，position_ids 为 None）。题目文字：官方 model_vqa_loader 把去掉 “\nAnswer the question using a single word or phrase.” 的题目交给文本塔；本移植从提示的 token 找回题目文字（combo._stash_text），与官方相同。精度与官方在 device_map="auto" 下相同：视觉塔、文本塔与 visual_projection 为 float32，投影后的特征为 float16，归一后转 float32 算核。
- 复用：DivPrune-64（EXP-0032-R01、R02）、VisionZip-64（EXP-0028-R01、R02）、不剪枝（EXP-0033-R03、R04）。失败的运行保留，用新编号（R03 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“更强”：下一轮 IDEA 以 CDPruner-64 为底座；只有一个 →“部分”：下一轮 IDEA 按它针对的题型选底座，并写明理由；否则 →“不更强”：DivPrune-64 仍是底座（不用文本塔，更简单）。两集平均 ≥ +1.5 另记“提升大”。
- 区间：每个集内按图像重抽 1000 次（种子 0）。

### 可检验的预测（推断，只为事后对照，不是判据）

- CDPruner 论文 Table 1（LLaVA-1.5-7B，64 档，各自的评测划分，SC，PAPER-0013 的笔记）：CDPruner 75.4 / 58.6，DivPrune 74.1 / 57.5（VQAv2 / GQA），差 +1.3 / +1.1。本项目预测 Q1 在两个集上约 +0.5～+1.5；按题型，增益在问某个物体属性的题上较大；GQA 位置题上 CDPruner-64 与 DivPrune-64 一样远低于不剪枝（都连续编号，F89）；“否”题上答“是”的比例与 DivPrune-64 相近（都连续编号，F99）。

### 实现核对（门槛 ≥ 99%，analyze_0039.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；CDPruner 的日志每题一条，留 64 个不重复、升序、在 0..575 内，选择顺序是同一组，文字是去掉答题格式后缀的题目。
- CPU 核对 check_cdpruner.py（43 项）：选择与官方 encode_images 第 150～185 行逐个相同（随机输入：4 个种子 × 留 16 / 64 / 128 个 × 一段或两段文字；秩不足时与官方同样退化）；短题一段、长文字每 77 个 token 一段；visual_projection 与文本塔从本地 CLIP-L/336 快照离线载入（float32）；七个题目文件的“提示 token → 题目”映射没有冲突；参数解析（合法的通过，三种不合法的被拒绝，vanilla、divprune64 照常）。回归（新代码对已提交代码）check_peekc 19/19、check_peek 29/29、check_divprune 26/26、check_k1cls 32/32。
- GPU 冒烟测试（tmp，不是 EXP 运行；VQAv2 dev2 的前 40 题）：40 个答案，每题一条日志，留 64 个，文字正确，提示长度约 125（文字约 61 + 64），37 秒（含载入）。layer_tokens 日志的“视觉 token”一栏对在 LLM 之前剪枝的方法按 576 计（只适用于 FastV 一类），正式运行不记它。
- analyze_0039.py 用替身空跑（VisionZip-64 当 CDPruner-64，日志核对用冒烟测试的 40 题）：35 项都通过；DivPrune-64 − VisionZip-64 重算为 +2.20 / +1.63，与 EXP-0032 封存的相同（替身不是结果）。

### 代码（分支 code/token-pruning，在 81db8d7 之上，`D:\6CJNM007\tmp\exp0039\code\`）

- cdpruner.py（新）、run_eval.py（--method cdpruner、--cdpruner-keep、--cdpruner-log）、exp0002.py（cdpruner64 与它的日志）、check_cdpruner.py（新）、analyze_0039.py（新）。sha256 记在 `D:\6CJNM007\tmp\exp0039\MANIFEST`。
- 排队脚本 `D:\6CJNM007\tmp\chain_0039.sh`：读完改动部分、写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0039.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 81db8d7；MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 在仓库里重跑 check_cdpruner.py（全部通过）→ 核对配置 → 运行 R01、R02 → 都 completed、GPU 空闲后用已提交的 analyze_0039.py 分析并封存。

### 预算

- 2 次，约 0.6 GPU 小时；作为对照的复现，不计入任何 IDEA 的预算（研究流程：baseline 复现一周）。

### 许可与版本

- 官方代码 source/CDPruner（9541616c，Apache-2.0；2026-10-31 按 DEC-0063 的许可克隆，只读，登记见 source/README.md）。CLIP-L/336 用本地快照（含文本塔，不需要下载）。

## 开门前的阅读（DEC-0055，2026-10-31 中午）

- **读的范围**：上一道门（EXP-0038，2026-10-31 09:25）以来的改动：main 的 82a2fae（EXP-0038 计划 v1.1）、66d9850（EXP-0038 的结果、DEC-0074、F98～F100）、9447b4c（DEC-0075、本计划 v1）；code/token-pruning 的 81cb815（EXP-0038 的代码，开门前已读，由 chain_0038 核对 MANIFEST 后提交）、81db8d7（analyze_0038.py v1.1，由 chain_0038b 核对 MANIFEST 后提交：与 v1 只差 peekc 在 GQA 上的运行号与说明，check_v11 5 / 5）；新的运行产物：EXP-0038 的 R01～R07（R02 failed）、分析与深挖（都已封存，verify_seal ALL OK，深挖的备份已核对）。本实验要提交的代码：run_eval.py（cdpruner 的参数、校验、分支与日志，共 32 行）、exp0002.py（cdpruner64 与它的日志，5 行）相对 81db8d7 的差异；cdpruner.py（与 2026-10-31 03:46 核对过的移植逐字节相同）、check_cdpruner.py、analyze_0039.py 为新文件；sha256 记在 `D:\6CJNM007\tmp\exp0039\MANIFEST`；run_analyze_0039.sh、seal_0039.sh、提交说明 msg_code73.txt 与排队脚本 chain_0039.sh（bash -n 通过）。
- **核对**：不加 --method cdpruner 时其余方法的计算不变（新分支只在 method 为 cdpruner 时走；回归 check_peekc 19 / 19、check_peek 29 / 29、check_divprune 26 / 26、check_k1cls 32 / 32）；check_cdpruner 43 / 43；GPU 冒烟测试 40 题正常；analyze_0039.py 的替身空跑 35 项都通过。
- **发现**：代码与计划 v1 一致，没有要改的。
- **开门**：本节提交之后写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0039.done`。

## 结果（2026-10-31 中午，看过封存的分析之后写）

- **运行**：R01、R02 都 completed（代码 fc76f1a），每个配置 1 次、贪心解码，没有失败、没有重跑；评测共约 0.58 GPU 小时（计划约 0.6）。
- **分析与封存**：chain_0039 用已提交的 analyze_0039.py 在 CPU 上分析，11:55:59 封存（code/output/EXP-0039/analysis，SHA256SUMS b3755ecd…，已备份）；独立核对（verify_seal.sh EXP-0039）ALL OK。
- **实现核对**：38 项全部 100%：两个运行都是计划的配置 × 数据且 completed；每题一个答案，重算的得分等于计分文件与登记的准确率；CDPruner 的日志每题一条，留 64 个不重复、升序、在 0..575 内，选择顺序是同一组，文字是去掉答题格式后缀的题目。

### 准确率与配对差（VQAv2 dev2 5192 题 / GQA val400 4786 题）

| 配置 | VQAv2 dev2 | GQA val400 | 每题秒数 | 峰值显存（GiB） |
| --- | --- | --- | --- | --- |
| 不剪枝（EXP-0033-R03、R04） | 77.91 | 70.08 | 0.2713 / 0.2671 | 14.52 |
| VisionZip-64（EXP-0028-R01、R02） | 71.93 | 62.20 | 0.1606 / 0.1550 | 14.35 |
| DivPrune-64（EXP-0032-R01、R02） | 74.13 | 63.83 | 0.1647 / 0.1602 | 13.96 |
| **CDPruner-64**（R01、R02） | 74.87 | 66.32 | 0.2087 / 0.2000 | 14.44 |

| 比较（百分点，按图像重抽 1000 次） | VQAv2 dev2 | GQA val400 |
| --- | --- | --- |
| **Q1 CDPruner-64 − DivPrune-64** | +0.74 [+0.08, +1.46] | +2.49 [+1.38, +3.57] |
| CDPruner-64 − VisionZip-64 | +2.94 [+2.11, +3.80] | +4.12 [+2.89, +5.36] |
| DivPrune-64 − VisionZip-64（复现 F72） | +2.20 [+1.38, +3.03] | +1.63 [+0.34, +2.92] |
| 不剪枝 − CDPruner-64 | +3.04 [+2.35, +3.79] | +3.76 [+2.57, +4.94] |

- **按题型**（描述）：CDPruner-64 − DivPrune-64 在 VQAv2 的颜色题 +6.23 [+3.49, +9.10]、其他题 +1.75 [+0.70, +2.85]，计数 +0.59、是非 −0.53 没有检出；GQA 属性 +4.61 [+2.35, +6.85]、查询 +3.57 [+1.67, +5.32]、是非 +1.64 [+0.12, +3.12]，位置题 +2.77 [−1.75, +7.42] 没有检出。离不剪枝还差：VQAv2 计数 +5.18、其他 +3.21、是非 +2.10，颜色只剩 +0.80 [−1.11, +2.84]；GQA 二选一 +10.76、属性 +5.31、位置题 +16.04 [+11.75, +20.00]。
- **是非题的方向**（描述）：标准答案为“否”的题上答“是”的比例 CDPruner-64 0.246 / 0.228、DivPrune-64 0.221 / 0.201、不剪枝 0.166 / 0.181；“否”题上 CDPruner-64 − DivPrune-64 −2.19 [−3.55, −0.85] / −2.72 [−4.65, −0.74]，“是”题上 +1.14 [+0.04, +2.34] / +6.27 [+4.30, +8.24]。
- **成本**：每题秒数比 DivPrune-64 多 27% / 25%（CLIP 文本塔与 DPP 选择；官方称每个样本额外不到 10 ms，SC）；峰值显存多 0.48 GiB（文本塔与视觉投影，float32）。

### 按计划写定的读法

- **Q1**：两个集的区间都在 0 以上 →“更强”；两集平均 +1.61 ≥ +1.5，另记“提升大”。
- **计划写定的之后**：下一轮 IDEA 以 CDPruner-64 为底座（DEC-0076）。
- **预测对照**（只为事后对照）：Q1 约 +0.5～+1.5：VQAv2 +0.74 在范围内，GQA +2.49 超出；增益在问物体属性的题上较大：符合（颜色、属性）；GQA 位置题上与 DivPrune-64 一样远低于不剪枝：符合（差 16.04，都连续编号，F89）；“否”题上答“是”的比例与 DivPrune-64 相近：不符，CDPruner-64 多答“是”（F102）。论文 Table 1（各自的评测划分，SC）CDPruner − DivPrune 为 +1.3 / +1.1，本流程中方向相同、GQA 更大。
