# EXPERIMENT_RECORD：IDEA-0008 的开发运行（第 0 层借看全部 patch，按 FLOPs 对齐多留）

状态：计划 v1（2026-10-31 凌晨），在 EXP-0034 的任何运行之前写定。写计划时已看过 EXP-0032、EXP-0033 的全部结果（开发集，DEC-0061、DEC-0062），没有 EXP-0034 的任何结果。

- 实验 ID：EXP-0034。
- IDEA ID：[IDEA-0008](../IDEA-0008_第0层借看全部patch.md)（MODE_C-0002 回流 25 的 C29）；DEC-0064。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（主要，FLOPs 约相等）**：peek（K = 1、DivPrune 的 61 个，第 0 层每个 token 都看得到被剪的 515 个 patch）− DivPrune-64。
2. **Q2（机制，同样的 61 个与位置）**：peek − nopeek（第 0 层谁也看不到被剪的 patch）。
3. **同时报告（描述）**：nopeek − DivPrune-64；peektext（只有题目看得到）对 nopeek、对 peek；peek − K = 1 留 47（EXP-0032-R03、R04）；对 VisionZip-64、不剪枝；按题型（VQAv2 的答案类型与颜色题，GQA 的结构与语义类型，被路由与不路由的题）。

### 写计划时已知与未知

- 已知：DivPrune-64、K = 1 留 47、VisionZip-64、不剪枝在这两个开发集上的结果（F72～F78）；K = 1 留 47 在 GQA 属性、二选一题上高、在 VQAv2 计数、颜色题上低（F78，挑出这两类是看过 EXP-0032 的分组之后）。
- 未知：peek、nopeek、peektext 的任何结果。

### 数据

- VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图），开发集，已多次使用（P14）；要写成方法的结论都要在新抽的数据上预注册确认。

### 配置与运行（6 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 | 作用 |
| --- | --- | --- | --- |
| R01、R02 | `fastv61k1_div_peek_kll`（--fastv-k 1 --fastv-keep 61 --fastv-score div --fastv-div-peek prefill） | vqav2_dev2、gqa_val400 | peek（方法） |
| R03、R04 | `fastv61k1_div_nopeek_kll`（--fastv-div-peek none） | 同上 | nopeek（对照：同样的 61 个与位置，第 0 层看不到被剪的） |
| R05、R06 | `fastv61k1_div_peektext_kll`（--fastv-div-peek text） | 同上 | peektext（只有题目看得到） |

- 复用（配置与代码不变，从已封存的文件夹读）：DivPrune-64 EXP-0032-R01、R02；K = 1 留 47 EXP-0032-R03、R04；VisionZip-64 EXP-0028-R01、R02；不剪枝 EXP-0033-R03、R04。
- `_kll` 记下每题保留的 patch、第一个答案 token 的对数概率与进入每一层的序列长度，都不改变计算。每次运行设 PYTHONFAULTHANDLER=1。失败的运行保留，用新编号（R07 起）重跑，每个配置最多重跑 1 次，先记录再重跑。

### 计算量的口径（三种都报）

- **每层的 token 数**（本项目一直用的口径）：DivPrune-64 每层 64；peek、nopeek、peektext 第 0 层 576 个、之后 61 个，层平均 (576 + 31 × 61) / 32 = 77.1。
- **FLOPs**（按 LLaMA-7B 的结构计算，不是计时）：每个 token 每层的乘加 4d² + 3dm（d = 4096，m = 11008），只算 K、V 的 2d²（约 1/6），注意力按因果对数 × 2d。peek（被剪的只算 K、V）约为 DivPrune-64 的 0.998 倍，nopeek 0.976 倍，本实验实际运行的实现（第 0 层全部完整计算）1.105 倍，不剪枝 5.08 倍（按替身数据空跑时每题 63 个非图像 token 估算；分析脚本按每题的实际长度重算）。
- **KV 缓存**：本实现第 0 层缓存保留全部（解码时遮住，见 fastv_llava 的说明）；只算 K / V 的实现预填充后可删掉，其余层每层 61 个图像 token。
- 本实验不计时；每题秒数只作记录（第 0 层完整计算，比只算 K / V 的实现慢）。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 → “有希望”；只有一个 → “部分”；都含 0 而两集平均 > 0 → “方向为正，未检出”；否则 → “不像”。两集平均 ≥ +1.5 另记“提升大”（报告，不是判据）。
- Q2：同样的读法，写成“第 0 层看过被剪的 patch 有作用（两集 / 一集 / 正但未检出 / 没有作用）”。
- **下一步**：Q1 “有希望”，或“部分”且两集平均 ≥ +0.5 → 抽新的确认集（VQAv2、GQA 各一组，新种子，与已用的全部不重合），写确认计划并提交到 main，之后再运行；Q1 “方向为正，未检出” → 记录，交用户决定是否扩大开发集；Q1 “不像” → 不确认，Q2 的机制结果照常写进 Mode C。
- 区间：每个集内按图像重抽 1000 次（种子 0），与 analyze_0028 相同。

### 实现核对（不满足就停止排查，门槛 ≥ 99%）

- 在运行留下的日志上查（analyze_0034.py）：每题一个答案；重算的得分与计分文件、登记的准确率一致；每题一条选择记录、带对应模式；留下的 61 个等于记录；选择等于 DivPrune-64 的前 61 个（同样的特征）；第 0 层看到完整的提示，第 1～31 层少 515 个；三种模式的提示长度相同。
- 要用到模型内部的，在 check_peek.py 里查（CPU，随机初始化的小 LLaVA；排队脚本在仓库里重跑，全部通过才运行）：三种模式的预填充 logits 等于手算的参照（第 0 层带模式的遮罩算整段提示，之后只留系统提示、61 个 patch 与题目，按原始位置）；peek 的预填充等于原来 K = 1 留 61 的路径（逐值相同）；nopeek 等于把 DivPrune 的 61 个按原始位置交给 LLM；三种模式彼此不同；解码时第 0 层看不到被剪的 patch（generate 等于不带缓存的手算贪心解码）；参数、配置与运行记录里 .npy 输入表的哈希。tmp 里的结果：check_peek.py 29/29；回归 check_divprune.py 26/26、check_k1cls.py 32/32、check_zoom.py 77/77；analyze_0034.py 用替身数据空跑通过（替身不是结果）。

### 代码（分支 code/token-pruning，在门文件写下之后由排队脚本提交）

- 在 9dae438 之上（`D:\6CJNM007\tmp\exp0034\code\`）：fastv_llava.py（div_peek：选择提前到第 0 层之前、第 0 层的遮罩、解码时第 0 层的遮罩）、run_eval.py（--fastv-div-peek）、exp0002.py（三个配置；运行记录加 input_tables_sha256，记 method_args 里 .npy 表的哈希）、check_peek.py（新）、analyze_0034.py（新）。sha256 记在 `D:\6CJNM007\tmp\exp0034\MANIFEST`。
- 排队脚本 `D:\6CJNM007\tmp\chain_0034.sh`：按 DEC-0055 只读改动部分，写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0034.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 9dae438、MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 在仓库里重跑 check_peek.py（全部通过）与 check_divprune.py（26/26）→ 核对配置 → 运行 R01–R06 → 6 次都 completed、GPU 空闲后用已提交的 analyze_0034.py 在 CPU 上分析并封存。任何一步不满足就写“CHAIN STOP”并停下。

### 预算

- 6 次，每次约 15 分钟（VQAv2 dev2）、13 分钟（GQA val400），约 1.6 GPU 小时；IDEA-0008 上限 30 次或 20 GPU 小时。
