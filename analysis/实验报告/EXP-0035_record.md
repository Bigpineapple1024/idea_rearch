# EXPERIMENT_RECORD：IDEA-0008 的确认（新抽的 VQAv2 test5、GQA val400c；TextVQA 描述）

状态：计划 v1（2026-10-31 凌晨），**在看到 EXP-0034 的任何结果之前写定**（写计划时 EXP-0034 正在运行，进度只看了运行号与答题数）。只在 EXP-0034 的读法把 IDEA-0008 送进确认时运行（EXP-0034 计划 v1：Q1“有希望”，或“部分”且两集平均 ≥ +0.5；读 code/output/EXP-0034/analysis/0034.json 的 criteria.confirmation_next）；否则不运行，本记录写明原因。

- 实验 ID：EXP-0035。
- IDEA ID：[IDEA-0008](../IDEA-0008_第0层借看全部patch.md)；DEC-0064（确认用新抽的数据）。

## 计划 v1（运行前写定）

### 数据（新抽，2026-10-31 01:22，EXP-0034 的 R01 才答了 145 题）

- `make_confirm_sets2.py`（`D:\6CJNM007\tmp\exp0035\code\`，随确认的代码提交）每个只抽一次、拒绝覆盖：
  - **VQAv2 test5**：从 make_vqav2_sets.py 的图像池（39704 张）去掉 test、test2、dev2、test3、test4 的图（每个先按原种子重抽并与已有文件核对）后，按种子 5 抽 1000 张的全部 5356 题；与之前所有 VQAv2 子集（含 dev）不重合。题目文件 sha256 b87ec941…、标注 be6415e5…（vqav2_val/SHA256SUMS_test5）。
  - **GQA val400c**：从 val-balanced 的 10234 张图去掉 val400、val400b 的图（各自先重抽核对）后，按种子 2 抽 400 张的全部 5106 道 balanced 题；与 testdev 不重合。题目 f412b0b5…、标注 03975edc…（gqa/SHA256SUMS_val400c），400 张图的 sha256 在 gqa/images_val400c_SHA256SUMS（669f4004…），从已下载的官方 images.zip 解出。
- TextVQA v0.5.1 val（不给 OCR，5000 题），只作描述（IDEA-0004 起已多次使用）。

### 运行（10 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 | 作用 |
| --- | --- | --- | --- |
| R01、R02 | `fastv61k1_div_peek_kll` | vqav2_test5、gqa_val400c | peek（方法） |
| R03、R04 | `divprune64` | 同上 | DivPrune-64（底座，FLOPs 约相等） |
| R05、R06 | `fastv61k1_div_nopeek_kll` | 同上 | nopeek（机制对照） |
| R07、R08 | `fastv61k1_div_peek_kll`、`divprune64` | textvqa_noocr | 描述 |
| R09、R10 | `vanilla` | vqav2_test5、gqa_val400c | 参照：不剪枝（F77 的缺口在新数据上是否一样） |

- 失败的运行保留，用新编号（R11 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- **Q1（主要）**：peek − DivPrune-64，两个新集的 95% 区间都在 0 以上 → 成立；只有一个 → 部分成立；都没有 → 不成立。两集平均 ≥ +1.5 另记“提升大”（报告）。
- **Q2（机制）**：peek − nopeek，同样的读法。
- **Q3（描述）**：TextVQA peek − DivPrune-64。
- 同时报告：按题型（GQA 的属性、二选一题，即 F78 在新数据上是否重现）；不剪枝与各配置的差；计算量三种口径（见 EXP-0034 计划）。
- **之后**：Q1 成立 → IDEA-0008 作为方法写进论文的候选（主表还要与 CDPruner / MMTok 在本流程中比，官方代码已获许可下载，DEC-0063），并做只算 K / V 的实现与计时；部分成立 → 如实写出在哪个集上成立，交用户决定；不成立 → IDEA-0008 不作为方法的贡献，Q2 的机制结果照常报告。
- 区间：每个集内按图像重抽 1000 次（种子 0）。

### 实现核对（门槛 ≥ 99%，analyze_0035.py）

- 每题一个答案；重算的得分与计分文件、登记的准确率一致；peek、nopeek 每题一条带模式的选择记录，留下的 61 个等于记录、等于 DivPrune-64 的前 61 个，第 0 层看到完整提示、第 1～31 层少 515 个。
- 新数据集的计分路径：排队脚本先用标准答案合成的答案文件在两个新集上计分，n_scored = n_answers = 题数才运行（tmp 里用合成答案空跑过：5356 / 5106 题全部计分）。

### 代码（分支 code/token-pruning，在 EXP-0034 的代码 e3f721e 之上，`D:\6CJNM007\tmp\exp0035\code\`）

- exp0002.py：登记 vqav2_test5、gqa_val400c；score_pope_mme.py：两个新 bench；make_confirm_sets2.py（新）；analyze_0035.py（新，替身数据空跑通过）。sha256 记在 `D:\6CJNM007\tmp\exp0035\MANIFEST`。
- 排队脚本 `D:\6CJNM007\tmp\chain_0035.sh`：等 chain_0034 结束（EXP-0034 的分析封存）；读 0034.json 的 criteria.confirmation_next，为 false 就写“EXP-0035 not run”并结束；为 true 时按 DEC-0055 只读改动部分，写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0035.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 EXP-0034 的代码提交；MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 计分的冒烟测试 → 核对配置 → 运行 R01–R10 → 10 次都 completed 后用已提交的 analyze_0035.py 分析并封存。

### 预算

- 10 次：peek / nopeek / DivPrune-64 每次约 15 分钟，TextVQA 约 18 分钟，不剪枝约 24 分钟；约 2.9 GPU 小时。IDEA-0008 共 16 / 30 次。
