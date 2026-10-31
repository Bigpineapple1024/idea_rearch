# EXPERIMENT_RECORD：IDEA-0010 的开发运行（多留、冻住，CDPruner 底座）

状态：计划 v1（2026-10-31 中午 12 点后），在 EXP-0040 的任何运行之前写定。写计划时已看过 EXP-0034～EXP-0039 的全部结果。

- 实验 ID：EXP-0040。
- IDEA ID：[IDEA-0010](../IDEA-0010_多留冻住.md)（MODE_C-0002 回流 29 的 C37）；DEC-0077。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（主要，FLOPs 约相等）**：f127 − b64（b64 = CDPruner-64，EXP-0039）。
2. **Q2（127 个时冻住的代价）**：f127 − b127。
3. **同时报告（描述）**：b127 − b64（多留本身）、f64 − b64（64 个时冻住的代价）、不剪枝 − f127；按题型（VQAv2 答案类型，GQA 结构与语义类型，被路由与不路由的题）；GQA 位置题；是非题按标准答案分开的答“是”比例；第一个答案 token 的对数几率（f127、b127、f64 记录）；FLOPs（按结构、逐题，文字 token 计入）；视觉部分的 KV 缓存；每题秒数与峰值显存。

### 数据

- VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图），开发集（P14）；要写成方法的结论要在新抽的数据与 MME、POPE 上确认。

### 配置与运行（6 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `cdpruner127_f19_logits`（--method cdpruner --cdpruner-keep 127 --freeze-layers 0,2,3,12,17-31） | vqav2_dev2、gqa_val400 |
| R03、R04 | `cdpruner127_logits`（--method cdpruner --cdpruner-keep 127） | 同上 |
| R05、R06 | `cdpruner64_f19_logits`（--method cdpruner --cdpruner-keep 64 --freeze-layers 0,2,3,12,17-31） | 同上 |

- 冻住（freeze.py）：在所列的层，图像 token 保持该层的输入状态（不算注意力输出与 MLP），其他 token 照常把它们当 key / value；预填充时缓存，生成的 token 也看得到。层表是 ShortV 为 LLaVA-1.5-7B 选的 19 层（其 Table 8），不在本项目的数据上调。
- 复用：CDPruner-64（EXP-0039-R01、R02）、不剪枝（EXP-0033-R03、R04）。失败的运行保留，用新编号（R07 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- Q2：两集平均 ≥ −0.5 →“便宜”；否则 →“代价大”。
- **之后**：Q1“有希望”，或“部分”且两集平均 ≥ +0.5 → 确认：新抽的 VQAv2、GQA 各一组（与已用的全部不重合），加 MME 与 POPE（推广；IDEA-0008 在这两个上比底座差，F94、F95），写确认计划、提交后运行；否则 → 记录多留的增益与冻住的代价，IDEA-0010 停在开发集，交用户决定下一步（若 Q2“代价大”而 b127 − b64 大，可另立预注册检验按剪枝后的模型重选冻住的层）。
- 区间：每个集内按图像重抽 1000 次（种子 0）。

### 可检验的预测（推断，只为事后对照，不是判据）

- b127 − b64 约 +0.8～+1.5；Q2 约 −0.3～−1.0；Q1 约 0～+1（见 IDEA-0010“机制与预测”）。

### 实现核对（门槛 ≥ 99%，analyze_0040.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；CDPruner 的日志每题一条，留 127（或 64）个，127 个的选择顺序的前 64 个等于 CDPruner-64 的顺序；冻住的日志列出 ShortV 的 19 层，每题一条，图像在 35 个 token 的系统提示之后、127（或 64）个、每题还原 19 次；f127 与 f64 的提示相同。
- CPU 核对 check_freeze.py（24 项，随机初始化的小 LLaVA，6 层，冻住第 0、2、3 层）：冻住的模型的预填充 logits 等于手算参照（冻住的层只让非图像行当 query，K、V 来自每一行的输入，图像行不变）；generate 等于不带缓存的手算贪心解码；图像行在冻住的层前后完全相同、在其他层改变；不冻住时等于原方法；参照所用的“只算非图像行”在每一层都等于整层计算的非图像行；层表解析、参数与配置。回归（新代码对已提交代码）check_cdpruner 43 / 43、check_peekc 19 / 19、check_peek 29 / 29、check_divprune 26 / 26、check_k1cls 32 / 32。
- GPU 冒烟测试（tmp，不是 EXP 运行；VQAv2 dev2 的前 40 题，f127 与 f64）：各 40 个答案；冻住的日志 40 条，图像 127 / 64 个，每条还原 19 次；各 37 秒（含载入）。
- analyze_0040.py 用替身空跑（准确率部分用已封存的运行当 f127 / b127 / f64，日志核对用冒烟测试的 40 题）：44 项都通过（替身不是结果）。按冒烟测试的提示长度算的 FLOPs（相对 CDPruner-64）：f127 1.0016、b127 1.5066、f64 0.7459、不剪枝 5.154。

### 代码（分支 code/token-pruning，在 fc76f1a 之上，`D:\6CJNM007\tmp\exp0040\code\`）

- freeze.py（新）、run_eval.py（--freeze-layers、--freeze-log）、exp0002.py（六个配置与日志：CDPruner、DivPrune 各三个，本实验只用 CDPruner 的）、check_freeze.py（新）、analyze_0040.py（新）。sha256 记在 `D:\6CJNM007\tmp\exp0040\MANIFEST`。
- 排队脚本 `D:\6CJNM007\tmp\chain_0040.sh`：读完改动部分、写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0040.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 fc76f1a；MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 在仓库里重跑 check_freeze.py 与 check_cdpruner.py（全部通过）→ 核对配置 → 运行 R01～R06 → 都 completed、GPU 空闲后用已提交的 analyze_0040.py 分析并封存。

### 预算

- 6 次，约 2 GPU 小时；IDEA-0010 上限 30 次或 20 GPU 小时。

## 开门前的阅读（DEC-0055，2026-10-31 中午 12 点后）

- **读的范围**：上一道门（EXP-0039，2026-10-31 11:19）以来的改动：main 的 41dd99e（MODE_C-0002 回流 28）、3754844（EXP-0039 的结果、DEC-0076、F101、F102）、dc52811（回流 29、IDEA-0010、本计划 v1、DEC-0077）；code/token-pruning 的 fc76f1a（EXP-0039 的代码，开门前已读，由 chain_0039 核对 MANIFEST 后提交，仓库里 check_cdpruner 43 / 43）；新的运行产物：EXP-0039 的 R01、R02 与分析（已封存，verify_seal ALL OK）。本实验要提交的代码：run_eval.py（--freeze-layers 的参数、校验、施加与日志，27 行）、exp0002.py（六个配置与冻住的日志，9 行）相对 fc76f1a 的差异；freeze.py、check_freeze.py、analyze_0040.py 为新文件（BASE = cdpruner）；sha256 记在 `D:\6CJNM007\tmp\exp0040\MANIFEST`；run_analyze_0040.sh、seal_0040.sh、提交说明 msg_code74.txt 与排队脚本 chain_0040.sh（bash -n 通过）。
- **核对**：不加 --freeze-layers 时其余方法的计算不变（新代码只在这个参数存在时施加钩子；回归 check_cdpruner 43 / 43、check_peekc 19 / 19、check_peek 29 / 29、check_divprune 26 / 26、check_k1cls 32 / 32）；check_freeze 24 / 24；GPU 冒烟测试 40 题正常；analyze_0040.py 的替身空跑 44 项都通过。
- **发现**：代码与计划 v1 一致，没有要改的。
- **开门**：本节提交之后写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0040.done`。
