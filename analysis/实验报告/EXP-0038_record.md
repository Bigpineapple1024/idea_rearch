# EXPERIMENT_RECORD：IDEA-0009 的开发运行（编号相同时的第 0 层借看）

状态：计划 v1（2026-10-31 上午），在 EXP-0038 的任何运行之前写定。写计划时已看过 EXP-0034～EXP-0037 的全部结果与查新（DEC-0065～DEC-0072）。

- 实验 ID：EXP-0038。
- IDEA ID：[IDEA-0009](../IDEA-0009_编号相同时的第0层借看.md)（MODE_C-0002 回流 27 的 C34）；DEC-0073。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（主要，FLOPs 约相等）**：peekc − DivPrune-64。
2. **Q2（借看本身，每层编号方式相同）**：peekc − nopeekc。
3. **同时报告（描述）**：nopeekc − DivPrune-61（第 0 层原来的位置对连续）、DivPrune-61 − DivPrune-64（少 3 个）、peekc − peek 与 nopeekc − nopeek（第 1 层起连续对原始编号，EXP-0034 的 R01～R04）、不剪枝 − peekc；按题型（VQAv2 答案类型，GQA 结构与语义类型，被路由与不路由的题）；FLOPs（analyze_0034.macs）。

### 数据

- VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图），开发集（P14）；要写成方法的结论要在新抽的数据上确认。

### 配置与运行（6 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `fastv61k1_div_peekc_kll`（--fastv-k 1 --fastv-keep 61 --fastv-score div --fastv-div-peek prefill --fastv-pos-after contiguous） | vqav2_dev2、gqa_val400 |
| R03、R04 | `fastv61k1_div_nopeekc_kll`（--fastv-div-peek none --fastv-pos-after contiguous） | 同上 |
| R05、R06 | `divprune61`（--method divprune --divprune-keep 61） | 同上 |

- 复用：DivPrune-64（EXP-0032-R01、R02）、peek 与 nopeek（EXP-0034-R01～R04）、不剪枝（EXP-0033-R03、R04）。失败的运行保留，用新编号（R07 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1、Q2：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- **之后**：Q2“有希望”，或“部分”且两集平均 ≥ +0.5 → 借看能加在先剪后送之上：抽新的确认集（VQAv2、GQA 各一组，与已用的全部不重合），写确认计划、提交后运行；否则 → 借看在 EXP-0034～EXP-0036 里的作用记为“纠正原始编号的偏‘是’”，IDEA-0009 结束，交用户决定下一步。
- 区间：每个集内按图像重抽 1000 次（种子 0）。

### 可检验的预测（推断，只为事后对照，不是判据）

- 见 IDEA-0009 的“两种解释的预测”。本项目倾向“只是纠正”：peekc − nopeekc 约 −0.5～+0.5，peekc − DivPrune-64 约 −0.5～+0.3。

### 实现核对（门槛 ≥ 99%，analyze_0038.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；peekc、nopeekc 每题一条带模式的选择记录，留下的 61 个等于记录、等于 DivPrune-64 的前 61 个，第 0 层看到完整提示、第 1～31 层少 515 个，两者提示长度相同；DivPrune-61 的选择等于 DivPrune-64 的前 61 个。模型内部用 check_peekc.py 查（CPU，随机初始化的小 LLaVA）：两种模式的预填充 logits 等于手算参照（第 0 层整段、原来的位置、带模式的遮罩；第 1 层起留下的按连续位置）；generate 等于不带缓存的手算贪心解码；偏移为 515；与原始编号的结果不同；参数、绑定与配置。tmp 里的结果：check_peekc 19/19，回归 check_peek 29/29、check_divprune 26/26、check_k1cls 32/32；analyze_0038.py 用 EXP-0034 / EXP-0032 的运行作替身空跑：72 项都通过，算出的与 EXP-0034 封存的逐位相同（替身不是结果）。

### 代码（分支 code/token-pruning，在 008bbee 之上，`D:\6CJNM007\tmp\exp0038\code\`）

- fastv_llava.py（pos_after）、run_eval.py（--fastv-pos-after）、exp0002.py（三个配置）、check_peekc.py（新）、analyze_0038.py（新）。sha256 记在 `D:\6CJNM007\tmp\exp0038\MANIFEST`。
- 排队脚本 `D:\6CJNM007\tmp\chain_0038.sh`：读完改动部分、写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0038.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 008bbee；MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 在仓库里重跑 check_peekc.py 与 check_peek.py（全部通过）→ 核对配置 → 运行 R01–R06 → 都 completed、GPU 空闲后用已提交的 analyze_0038.py 分析并封存。

### 预算

- 6 次，约 1.5 GPU 小时；IDEA-0009 上限 30 次或 20 GPU 小时。

## 开门前的阅读（DEC-0055，2026-10-31 上午）

- **读的范围**：上一道门（EXP-0037，2026-10-31 早上 7 点前）以来的改动：main 的 971db0a（EXP-0036 的结果）、54bf807（EXP-0037 的结果）、ea84350（DEC-0071）、77f1f0c（笔记的第二人复核、查新补充、DEC-0072）、5624546（MODE_C-0002 回流 27）、d5811d2（本计划 v1、IDEA-0009、DEC-0073）；code/token-pruning 的 f981196（EXP-0037 的代码，由 chain_0037 核对 MANIFEST 后提交）、008bbee（后封存的 8 个脚本收入 sealed_copies，与封存副本逐字节相同）；新的运行产物：EXP-0036、EXP-0037 的运行与分析（都已封存并独立核对 ALL OK）。本实验要提交的代码：fastv_llava.py（pos_after：剪枝点的重新编号、解码时第 1 层起减去偏移）、run_eval.py（--fastv-pos-after）、exp0002.py（三个配置）相对 008bbee 的差异，check_peekc.py、analyze_0038.py 为新文件；sha256 记在 `D:\6CJNM007\tmp\exp0038\MANIFEST`；run_analyze_0038.sh、seal_0038.sh、提交说明 msg_code71.txt 与排队脚本 chain_0038.sh（bash -n 通过）。
- **核对**：不加 --fastv-pos-after 时 fastv_llava 的计算不变（新分支只在 peek 与 pos_after 都有值时走；解码时 fastv_pos_shift 为 None 就不改位置），回归 check_peek 29/29、check_divprune 26/26、check_k1cls 32/32（新代码对已提交代码）；check_peekc 19/19；analyze_0038.py 的替身空跑 72 项都通过、数字与 EXP-0034 封存的逐位相同。
- **发现**：代码与计划 v1 一致，没有要改的。
- **开门**：本节提交之后写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0038.done`。
