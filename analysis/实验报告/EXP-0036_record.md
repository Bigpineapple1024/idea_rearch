# EXPERIMENT_RECORD：IDEA-0008 与 SQZ-EXP-0034 的两条发现在 MME、V*、POPE 上

状态：计划 v1（2026-10-31 凌晨 4 点前），**在看到 EXP-0035 与 EXP-0036 的任何结果之前写定**（写计划时 EXP-0035 正在运行，进度只看了运行号与答题数）。写计划时已看过 EXP-0034 的结果与 SQZ-EXP-0034（开发集，DEC-0065、F84～F90）。

- 实验 ID：EXP-0036。
- IDEA ID：[IDEA-0008](../IDEA-0008_第0层借看全部patch.md)（推广检验）；MODE_C-0002 回流 26 的 P18、P19（C31 的前提）；DEC-0066。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（推广）**：peek − DivPrune-64 在三个没用来设计 IDEA-0008 的 benchmark 上（逐题准确率）：MME（2374 题）、V\* Bench（191 题）、POPE（8910 题）。
2. **Q2（F88 在 POPE 上）**：POPE 标准答案为“否”的题上，nopeek − DivPrune-64（偏“是”是否重现）、peek − nopeek（借看是否去掉它）；“是”题上 peek − nopeek（代价）。
3. **Q3（F89 在 MME、V\* 上）**：MME 的 position 子任务（60 题）与 V\* 的相对位置（76 题）合并：nopeek − DivPrune-64（原始位置编号是否帮位置题）。
4. 同时报告（描述）：MME 每个子任务的官方分数（acc + acc+）与 existence、count、color 子任务的逐题差；POPE 每个划分的准确率、答“是”的比例、F1；V\* 的直接属性；与不剪枝的差；POPE 上 peek、nopeek 第一个答案 token 的对数几率；预填充 FLOPs（analyze_0034.macs）。

### 写计划时已知与未知

- 已知：开发集上 peek − DivPrune-64 +0.18 / +2.34（VQAv2 dev2 / GQA val400），nopeek 在“否”题上偏答“是”、借看去掉它（F88），GQA 位置题上原始编号补回 DivPrune-64 缺口的大部分（F89）。三个 benchmark 上的不剪枝结果（EXP-0002-R05、R34，EXP-0025-R08，DEC 记录里早已报告）。
- 未知：peek、nopeek、DivPrune-64 在这三个 benchmark 上的任何结果；EXP-0035 的任何结果。

### 数据

- MME（2374 题，1187 张图，14 个子任务；官方评测工具）；V\* Bench（191 题：直接属性 115、相对位置 76）；POPE（8910 题，COCO val2014 的图，adversarial、popular、random 三个划分；标准答案为“是”的 4500 题、“否”的 4410 题）。都已在本项目中用过（EXP-0002 起，V\* 自 EXP-0025），但没有用来设计或挑选 IDEA-0008 的任何设置。

### 配置与运行（9 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02、R03 | `fastv61k1_div_peek_kll`、`divprune64`、`fastv61k1_div_nopeek_kll` | mme |
| R04、R05、R06 | 同上 | vstar |
| R07、R08、R09 | 同上 | pope |

- 复用（不剪枝，参照）：EXP-0002-R05（MME）、EXP-0025-R08（V\*）、EXP-0002-R34（POPE）；它们与 EXP-0002-R02、R04 逐条相同，代码路径（不剪枝）之后没有改。
- 失败的运行保留，用新编号（R10 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- **Q1**：三个 benchmark 中 peek − DivPrune-64 的 95% 区间在 0 以上的个数 → “三个 / 两个 / 一个 / 没有一个”。只作推广的证据，不改变 EXP-0035 对 IDEA-0008 的判断。
- **Q2**：nopeek − DivPrune-64 在“否”题上区间在 0 以下 →“偏‘是’在 POPE 上重现”；peek − nopeek 在“否”题上区间在 0 以上 →“借看在 POPE 上去掉偏‘是’”；peek − nopeek 在“是”题上区间在 0 以下 →“代价重现”。
- **Q3**：合并 136 题，nopeek − DivPrune-64 区间在 0 以上 →“原始编号在 MME、V\* 的位置题上也有帮助”，否则“没有检出”（题少，检不出不等于没有）。
- 区间：每个 benchmark 内按图像重抽 1000 次（种子 0）；合并时在每个 benchmark 内各自重抽再合在一起。
- **之后**：Q2、Q3 都重现 → C31（先剪后送的方法保留原始编号 + 第 0 层借看，作为插件）进入 discover-innovation，在 VisionZip、CDPruner、MMTok 上检验；只重现一个 → 如实记录，C31 只按重现的那条设计；都不重现 → F88、F89 只作开发集上的现象。

### 可检验的预测（推断，只为事后对照，不是判据）

- MME position：nopeek − DivPrune-64 约 +10～+20（60 题，即 6～12 题）；V\* 相对位置约 +5～+15。
- POPE“否”题：nopeek − DivPrune-64 约 −5～−10，peek − nopeek 约 +5～+10；“是”题 peek − nopeek 约 −1～−3；POPE 整体 peek − DivPrune-64 约 −1～+1。
- MME 逐题整体、V\* 整体：peek − DivPrune-64 约 0～+2。

### 实现核对（门槛 100%，analyze_0036.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；POPE 第 k 题对应其划分标注文件的第 k 行（图与题目相同），按 LLaVA eval_pope.py 的规则重算的每个划分的准确率、答“是”的比例等于 metrics.json；MME 按官方 calculation.py 的规则在其逐子任务文件上重算的每个子任务分数（acc + acc+）等于 metrics.json；V\* 的准确率等于 metrics.json；peek、nopeek 每题一条带模式的选择记录，留下的 61 个等于记录、等于 DivPrune-64 的前 61 个（≥ 99%），第 0 层看到完整提示、第 1～31 层少 515 个，两者的提示长度相同。tmp 里用 EXP-0002、EXP-0025 的旧运行作替身空跑：48 项核对都通过（替身不是结果）。

### 代码（分支 code/token-pruning，在 EXP-0035 的代码 88cb1cb 之上，`D:\6CJNM007\tmp\exp0036\code\`）

- 只新增 analyze_0036.py（sha256 记在 `D:\6CJNM007\tmp\exp0036\MANIFEST`）；模型代码、配置、数据集登记都不改（三个配置、三个 benchmark 都已登记）。
- 排队脚本 `D:\6CJNM007\tmp\chain_0036.sh`：等 chain_0035 结束（EXP-0035 的分析封存）；按 DEC-0055 读改动部分、写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0036.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 EXP-0035 的代码提交；MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 核对配置 → 运行 R01–R09 → 9 次都 completed、GPU 空闲后用已提交的 analyze_0036.py 分析并封存。任何一步不满足就写“CHAIN STOP”并停下。

### 预算

- 9 次：MME 每次约 8 分钟、V\* 约 1 分钟、POPE 约 27 分钟，约 1.8 GPU 小时。IDEA-0008 共 25 / 30 次，约 6.2 / 20 GPU 小时。
