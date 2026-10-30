# EXPERIMENT_RECORD：IDEA-0008 的确认（新抽的 VQAv2 test5、GQA val400c；TextVQA 描述）

状态：计划 v1（2026-10-31 凌晨），**在看到 EXP-0034 的任何结果之前写定**（写计划时 EXP-0034 正在运行，进度只看了运行号与答题数）。只在 EXP-0034 的读法把 IDEA-0008 送进确认时运行（EXP-0034 计划 v1：Q1“有希望”，或“部分”且两集平均 ≥ +0.5；读 code/output/EXP-0034/analysis/0034.json 的 criteria.confirmation_next）；否则不运行，本记录写明原因。（2026-10-31 3 点补：EXP-0034 的读法是“部分”、两集平均 +1.26，送进确认，DEC-0065；开门前的阅读见文末；03:20 加计划 v1.1：两项二级分析，不改 Q1～Q3，见文末。）

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

## 开门前的阅读（DEC-0055，2026-10-31 凌晨 3 点）

- **读的范围**：上一道门（EXP-0034，2026-10-31 01:20）以来的改动：main 的 7fb5ab9（EXP-0034 开门前的阅读）、2fdba1f（本计划 v1）、482ffdf（EXP-0034 的结果、F84～F87、DEC-0065；提交前用脚本核对，文中 58 个“差 [区间]”与 22 个准确率都能在已封存的分析里找到原样，新加的表格行与相邻行的单元格数相同）；code/token-pruning 的 e3f721e（EXP-0034 的代码，由 chain_0034 提交，开门前已读，与 exp0034/MANIFEST 一致）；新的运行产物：EXP-0034 的 R01～R06 与 analysis（已封存并独立核对）。本实验要提交的代码：exp0002.py 相对 e3f721e +11 行（只加 vqav2_test5、gqa_val400c 两个 bench 的登记），score_pope_mme.py +4 / −3（两个 bench 名，gqa_val400c 与 val400 同样计分），analyze_0035.py、make_confirm_sets2.py 为新文件；run_analyze_0035.sh、seal_0035.sh、提交说明 msg_code67.txt 与排队脚本 chain_0035.sh。代码目录里其余文件与 HEAD 逐字节相同。
- **核对**：四个文件的 sha256 与 `D:\6CJNM007\tmp\exp0035\MANIFEST` 相同；HEAD 的 exp0002.py、score_pope_mme.py 与 EXP-0034 的副本相同（排队脚本的前提）；EXP-0035 的输出目录与计分冒烟测试的目录都还不存在；GPU 空闲；工作区干净、在 code/token-pruning。analyze_0035.py：先读 EXP-0034 封存的 criteria.confirmation_next，不为真就停；10 个运行的编号、配置与数据与计划相同；核对项与计划相同；Q1、Q2 按两个新集的 95% 区间读，两集平均 ≥ +1.5 另记“提升大”，Q3 只作描述，与计划写定的一致；区间每个集内按图像重抽 1000 次（种子 0）。make_confirm_sets2.py 01:22 已运行一次（日志在 tmp），新数据的校验和由排队脚本在提交之前再核。chain_0035.sh：等 EXP-0034 的分析封存、读 confirmation_next（02:55 读到 True）、等本门，之后的步骤与计划一致。
- **发现**：①计划写“同时报告……计算量三种口径（见 EXP-0034 计划）”，analyze_0035.py 没有算计算量；结果出来之后用 analyze_0034.py 的同一公式（macs()）在新数据每题的实际长度上补算，只作描述，不影响读法，记为偏差。②其余与计划 v1 一致，没有要改的代码。
- **开门**：本节提交之后写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0035.done`。

## 计划 v1.1（2026-10-31 03:20，EXP-0035 运行中、看到任何 EXP-0035 结果之前；只加二级分析，不改 Q1～Q3 与运行）

- **来由**：SQZ-EXP-0034（开发集，事后、探索性；code/output/EXP-0034/squeeze，SHA256SUMS 17af06d5…）发现两件事：①同样的 61 个、原始位置编号、第 0 层看不到被剪的（nopeek）在标准答案为“否”的是非题上偏向答“是”（准确率 nopeek − DivPrune-64 −9.13 [−11.06, −7.17] / −7.48 [−10.21, −4.68]），第 0 层看过被剪的 patch 把它去掉（peek − nopeek +9.55 [+7.70, +11.32] / +11.00 [+8.32, +13.66]），代价是“是”题略降（−2.25 [−3.58, −0.93] / −3.61 [−6.38, −1.03]）；②GQA 的位置题（细分题型以 position 开头，505 题）上 DivPrune-64 比不剪枝低 18.81 [+13.67, +23.91]，原始位置编号补回 13.86 [+9.09, +18.99]（nopeek − DivPrune-64）。两条都是看过开发集的分组之后找到的，要在新数据上检验。
- **二级分析**：脚本 `D:\6CJNM007\tmp\exp0035\secondary\analyze_0035_secondary.py`（sha256 508829fc…，03:19 写定，来历见同目录 notes.md）。同一代码用 --dev 在开发集运行上空跑，S1 的数字与 SQZ-EXP-0034 逐项相同。EXP-0035 的分析封存之后在 CPU 上运行一次，输出写到 code/output/EXP-0035/secondary 并封存。
  - **S1 是非题的方向**（两个新集；标准答案为“是”或“否”的题：GQA 用答案，VQAv2 用答案类型 yes/no 且十个答案中最多的是 yes 或 no）：nopeek − DivPrune-64 在“否”题上的 95% 区间在 0 以下 →“偏‘是’重现（两个集 / 一个集 / 不重现）”；peek − nopeek 在“否”题上在 0 以上 →“第 0 层看被剪的去掉偏‘是’（两个集 / 一个集 / 不）”；peek − nopeek 在“是”题上在 0 以下 →“代价重现（两个集 / 一个集 / 不）”。同时报告答“是”的比例与 peek、nopeek 第一个答案 token 的对数几率。
  - **S2 GQA val400c 的位置题**（细分题型以 position 开头，合并）：nopeek − DivPrune-64 的区间在 0 以上 →“原始位置编号帮位置题（重现）”，否则“不重现”；同时报告 peek − DivPrune-64、不剪枝 − DivPrune-64 与每个细分题型。
  - **S3**：至少 60 题的细分题型，三个配对差，只作描述。
- 区间：每个集内按图像重抽 1000 次（种子 0）。二级分析不改 Q1～Q3 的读法，也不影响 IDEA-0008 是否作为方法的判断（那只看 Q1）。
