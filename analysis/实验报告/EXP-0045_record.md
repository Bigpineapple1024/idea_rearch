# EXPERIMENT_RECORD：EXP-0043 的确认（条件性；新抽的 VQAv2 test7、GQA val400e 与 MME、POPE）

状态补注（2026-11-01 凌晨 3 点后，DEC-0090、DEC-0091）：A、B、C 的条件满足（EXP-0043 的 Q1、Q2“有希望”，Q3“部分”且两集平均 +0.81）；D 的条件（EXP-0044 的 Q1）因 EXP-0044 计划 v1.2 拿掉 Q1 而不成立，D 不运行（D 的问题改由 [EXP-0047](EXP-0047_record.md) 在 EXP-0046 之后做，条件是 EXP-0046 的 Q1，DEC-0092）。本实验在 EXP-0044 之后运行 R01～R16（最多 16 次）。

状态：计划 v1.1（2026-11-01 凌晨（更正：实际写于 2026-10-31 晚上 11 点半前后，原写“2026-11-01 凌晨”），加 D 部分：VisionZip-64，见文末），在 EXP-0043、EXP-0044 的任何结果之前写定。前一状态：计划 v1（2026-10-31 晚上 11 点后），在 EXP-0043 的任何结果之前写定（写时 EXP-0043 的 R01 正在运行，只看过运行号、状态与答案数）。只对 EXP-0043 里读作“有希望”、或“部分”且两集平均 ≥ +0.5 的设置运行；都不满足则不运行，本记录注明。

- 实验 ID：EXP-0045（EXP-0044 留给 VisionZip-64 加 fill 与省显存的实现，DEC-0083）。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)；开发实验 [EXP-0043](EXP-0043_record.md)；DEC-0083、DEC-0084。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题（每一部分只在 EXP-0043 的对应问题满足条件时运行）

1. **A：DivPrune-64**（EXP-0043 的 Q1）。**A1**：新抽的数据上 fill − DivPrune-64（VQAv2 test7、GQA val400e）。**A2**：MME、POPE 上 fill − DivPrune-64（逐题准确率）是否更差。
2. **B：CDPruner-128**（EXP-0043 的 Q2）：新抽的数据上 fill − CDPruner-128。
3. **C：CDPruner-192**（EXP-0043 的 Q3）：新抽的数据上 fill − CDPruner-192。
4. **同时报告（描述）**：不剪枝 − fill（新数据）；按题型、GQA 位置题；是非题按标准答案的答“是”比例；与不剪枝的回答相同的比例；MME 子任务分数；POPE 各分集的准确率、F1 与答“是”比例，POPE 上 fill − 不剪枝；填补的分配；每题秒数与峰值显存。

### 数据

- VQAv2 test7（1000 张 val2014 图、5229 题）与 GQA val400e（400 张 GQA val 图、5037 题）：make_confirm_sets4.py 在 2026-10-31 23:15 抽出（EXP-0043 的 R01 运行时、在它的任何结果之前），之后没有用过；脚本断言它们与已用的全部 VQAv2 子集（test、dev、test2、dev2、test3～test6）、GQA 子集（val400、val400b～val400d、testdev）不重合；sha256 见 vqav2_val/SHA256SUMS_test7、gqa/SHA256SUMS_val400e 与 gqa/images_val400e_SHA256SUMS。
- MME（2374 题）、POPE（8910 题）：本流程的标准版本。B、C 不在 MME、POPE 上跑（64 个时的推广已由 EXP-0042 检验；预算的问题看一般题目）。

### 配置与运行（按此顺序；只运行条件满足的部分；编号固定，不运行的部分编号空着）

| 运行 | 配置 | 数据 | 部分 |
| --- | --- | --- | --- |
| R01、R02 | `divprune64_fill_logits`、`divprune64` | vqav2_test7 | A |
| R03、R04 | 同上 | gqa_val400e | A |
| R05、R06 | `divprune64_fill_logits` | mme、pope（DivPrune-64 复用 EXP-0036-R02、R08） | A |
| R07、R08 | `cdpruner128_fill_logits`、`cdpruner128_logits` | vqav2_test7 | B |
| R09、R10 | 同上 | gqa_val400e | B |
| R11、R12 | `cdpruner192_fill_logits`、`cdpruner192_logits` | vqav2_test7 | C |
| R13、R14 | 同上 | gqa_val400e | C |
| R15、R16 | `vanilla` | vqav2_test7、gqa_val400e（任一部分运行时） | — |

- fill 与 EXP-0041～EXP-0043 的完全相同（vfill.py 不改）。复用：不剪枝 MME（EXP-0002-R05）、POPE（EXP-0002-R34）；DivPrune-64 MME、POPE（EXP-0036-R02、R08，之后 divprune.py 只多记一个属性，核对留下的 patch 相同）。失败的运行保留，用新编号（R17 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- A1、B、C：两个新集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- A2：MME、POPE 各自的 95% 区间整个在 0 以下 →“更差”，否则 →“不更差”（与 EXP-0042 相同）。（2026-11-01 审查补注：“不更差”只是判定标签，意为未检出更差，不是非劣效证明；判定规则不变，DEC-0089。）另外预先写定一个次要读法（不进结论，因为 F112、P23）：POPE 上 fill − 不剪枝的区间整个在 0 以下 →“比不剪枝差”，否则 →“不比不剪枝差”。
- **结论**：A：A1“有希望”且 A2 两个都“不更差” →“确认”；A1“部分”且 A2 两个都“不更差” →“部分确认”；否则 →“未确认”。B、C：“有希望” →“在该预算上确认（一般题目）”；“部分” →“部分确认”；否则 →“未确认”。
- **之后**：记录，交用户决定（写成方法时的推广性证据）。
- 区间：每个 benchmark 内按图像重抽 1000 次（种子 0；新数据按 analyze_0028 的图像顺序，MME、POPE 按 analyze_0036）。

### 实现核对（门槛 ≥ 99%，analyze_0045.py，在看到 EXP-0045 的任何结果之前写好并用替身空跑）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；新数据上重算的得分等于计分文件与登记的准确率；MME 每个子任务的分数等于 metrics.json；POPE 第 k 题是第 k 个标注，每个分集的准确率与答“是”比例等于 metrics.json；fill 与底座留下的 patch 相同；fill 的布局日志每题一条（n 个图像 token、576 个槽、位置正确、32 层用了布局）。

### 代码（若运行：分支 code/token-pruning，在 EXP-0043 的代码之上）

- exp0002.py（登记 vqav2_test7、gqa_val400e）、score_pope_mme.py（让它们能计分）、make_confirm_sets4.py（原样提交，已于 23:15 运行过）、analyze_0045.py（新）。开门前另做 GPU 冒烟测试，另写 MANIFEST 与排队脚本。

### 预算

- 最多 16 次，约 5.5 GPU 小时；IDEA-0011 合计最多 28 + EXP-0044 + 16 次（上限 50 次、20 GPU 小时，DEC-0083）。超出时先停下复盘。

## 计划 v1.1（2026-11-01 凌晨（更正：实际写于 2026-10-31 晚上 11 点半前后，原写“2026-11-01 凌晨”），在 EXP-0043、EXP-0044 的任何结果之前写定；DEC-0085）

- **加 D 部分：VisionZip-64**（[EXP-0044](EXP-0044_record.md) 的 Q1 满足条件时运行：“有希望”，或“部分”且两集平均 ≥ +0.5）：新抽的数据上 fill − VisionZip-64，只看一般题目（同 B、C）。

| 运行 | 配置 | 数据 | 部分 |
| --- | --- | --- | --- |
| R17、R18 | `visionzip64_fill_logits`、`visionzip64_logits` | vqav2_test7 | D |
| R19、R20 | 同上 | gqa_val400e | D |

- 读法与 B、C 相同；R15、R16（不剪枝）在任一部分运行时运行。失败的运行重跑从 R21 起。
- 最多 20 次，约 6.8 GPU 小时；IDEA-0011 合计最多 54 次（上限 60 次、20 GPU 小时，DEC-0085）。v1 的其余内容不变。

## 计划 v1.2（2026-10-31 深夜，EXP-0043 运行中，在 EXP-0043、EXP-0044 的任何结果之前；只补写实现核对，问题、数据、运行与读法都不改）

- v1.1 加 D 部分时没有改“实现核对”一节，那里写的“fill 与底座留下的 patch 相同”“576 个槽”不适用于 D：VisionZip 没有选择日志，fill 的布局是 577 个槽。
  D 部分的核对改为：fill 的布局日志每题一条（64 个图像 token、577 个槽、位置正确、32 层用了布局），fill 与底座的提示相同；VisionZip 送进 LLM 的 64 个 token
  不因 fill 改变由 check_vzfill 在 CPU 上核对（EXP-0044）。analyze_0045.py 按此写（替身空跑 98 项通过）。
- 已知的局限（写在这里，结果里同样报告）：A 部分 MME、POPE 的 DivPrune-64 复用 EXP-0036（代码 8cd98f3），不是同一份代码重跑；核对留下的 patch 相同，
  LLM 部分的运算没有逐题重跑核对。

## 开门前的阅读（DEC-0055，2026-11-01 凌晨 4 点半后）

- **读的范围**：上一道门（EXP-0044，6f05801）以来的改动：main 的 578f8d5（EXP-0046、EXP-0047 的计划 v1，DEC-0092）、55f457f（SQZ-EXP-0043 的单独报告、F117、F118、回流 34）、fa27945（EXP-0044 的结果、DEC-0093、F119）；code/token-pruning 的 a23e1d2（EXP-0044 的代码，开门前已读，由 chain_0044 核对 MANIFEST 后提交）；新的运行产物：EXP-0044 的 R05～R08 与分析（b09dd146…，verify_seal ALL OK）、SQZ-EXP-0043（576f323e…，已备份）。
- **运行的部分**：A、B、C（EXP-0043 的 Q1、Q2“有希望”，Q3“部分”且两集平均 +0.81，DEC-0090）；D 不运行（它的条件是 EXP-0044 的 Q1，计划 v1.2 拿掉了，DEC-0091；D 的问题改由 EXP-0047 做，DEC-0092）。`tmp/exp0045/PARTS` = “A B C”，`BASE_HEAD` = a23e1d2；运行 R01～R16（16 次）。
- **本实验要提交的代码**（相对 a23e1d2 的代码副本；其余 102 个 .py 文件与 EXP-0044 的代码副本逐字节相同；sha256 记在 `D:\6CJNM007\tmp\exp0045\MANIFEST`）：exp0002.py（+12 行，登记 vqav2_test7、gqa_val400e）；score_pope_mme.py（3 处，让两个新集能计分）；make_confirm_sets4.py（2026-10-31 23:15 运行过的原样）；analyze_0045.py（新：按 runs.tsv 里实际有的运行判断哪些部分运行；A1、B、C 的读法同 EXP-0043；A2 的 MME、POPE 读法同 EXP-0042，另有 POPE 对不剪枝的次要读法；实现核对；替身空跑 98 项）。排队脚本 `chain_0045.sh`（计划提交 6825b9f、c1c8794、52c9079；只运行 PARTS 里的部分）、run_analyze_0045.sh、seal_0045.sh、提交说明 msg_code80.txt（写明这次只跑 A、B、C）：都 bash -n 通过。
- **核对**：不改任何方法的计算（只登记数据集、加分析）。GPU 冒烟测试（新集前 8 题，A 另加 MME、POPE 前 8 题；答案不计分、不看）：14 项里 13 项第一次就正常结束；GQA val400e 上的 DivPrune-64 加 fill 第一次在答第 1 题时中止（退出码 127，日志里没有报错、没有回溯），用同样的 8 题与参数单独重跑一次正常（8 题、布局正确、留下的 patch 与底座相同 8 / 8、对数概率每题一条），判断为偶发退出（EXP-0038 的 R02 也遇到过），不是代码路径的问题；正式运行中若再出现，按计划用新编号（R21 起）重跑，在分析之前写计划 v1.3 说明读哪一号。其余：fill 与底座留下的 patch 相同（比较的五处各 8 / 8）；布局日志是 fill、每题一条、64 / 128 / 192 个图像 token 在 35 个 token 之后、576 个槽、位置正确、32 层；记录对数概率的运行每题一条；用全错的替身答案查两个新集能计分（vqav2_test7 5229 题、gqa_val400e 5037 题都计上）。仓库里的 check_vfill、check_vzfill、check_cdpruner、check_divprune 由排队脚本在提交后重跑。
- **发现**：代码与计划 v1～v1.2 一致，没有要改的。
- **开门**：本节提交之后写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0045.done`。
