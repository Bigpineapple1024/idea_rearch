# EXPERIMENT_RECORD：按题目句式路由的确认实验（第 ⑫ 步第六轮）

状态：计划 v1（2026-10-25），在任何运行之前写定；v1.1（同日，仍在任何运行之前）见文末。

- 实验 ID：EXP-0020。
- IDEA ID：[IDEA-0003](../IDEA-0003_按题目句式路由.md)；DEC-0031。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，运行前写定）

### 问题

EXP-0018 的 test1 上，A′+B+V 在 yes/no 题上显著好于 A，在 other 题上显著差（F38）。路由规则只按题目句式选择：助动词开头或“how many”开头的题用 A′+B+V，其余用 A。在新抽的 VQAv2 子集上，这个规则是否比 A 好？规则是看过 test1 的结果后定的，所以确认必须在新数据上做。

### 数据

- **test2**：`make_vqav2_sets.py test2` 生成。
  - 来源池与 test1 相同：POPE 与 coco_dev 都没用到、且本地有的 val2014 图，共 39704 张。先去掉 test1 的 1000 张（重抽后核对与现有文件一致），再按种子 1 抽 1000 张，取其全部题目，共 5415 题（yes/no 2020、number 680、other 2715）。
  - 与 test1、dev、POPE、coco_dev 的图像都不重叠。
  - sha256：题目 c086f656…，标注 4eb4e5c2…。
- prompt、计分与 EXP-0018 相同（m4c 评估器；逗号处理的影响 ≤0.03 个百分点，见 EXP-0018 的复核）。
- IDF 在 test2 的题目文件上计；可视性表与 EXP-0015 相同。
- 路由比例：test2 上 0.506，即 2740 题（check_route.py）。

### 运行

| 运行 | 配置 × 数据 | 组别 |
| --- | --- | --- |
| R01 | `fastv64_prior_strat_nopad_kl` × vqav2_test2 | A（64） |
| R02 | `fastv64_qroute_kl` × vqav2_test2 | 路由（64） |
| R03 | `fastv64_qvis_kl` × vqav2_test2 | A′+B+V（64） |
| R04 | `fastv128_prior_strat_nopad_kl` × vqav2_test2 | A（128） |
| R05 | `fastv128_qroute_kl` × vqav2_test2 | 路由（128） |
| R06 | `fastv128_qvis_kl` × vqav2_test2 | A′+B+V（128） |

POPE、MME 不重跑。这两个 benchmark 的题目（MME 只算十个感知子任务）全部被路由，路由在那里就是 A′+B+V。依据：

- check_route.py：从 prompt ids 得到的决定与题目文字上的规则逐题相同；
- 本实验的核对：被路由的题，路由的保留集合与答案与 A′+B+V 相同。

### 判据

| 编号 | 判据 |
| --- | --- |
| Q1（主） | 64 档：路由 − A 的 VQA 准确率配对差（按图像重抽 1000 次，种子 0）区间在 0 以上 |
| Q2 | 128 档：同上 |
| Q3（复验 F38） | 64 档：A′+B+V − A 在 yes/no 类上区间在 0 以上，在 other 类上区间在 0 以下 |

- **Q1 成立**：POPE（三档显著）与 VQAv2（64 档显著）两个 benchmark 在 64 档都提升，MME-P 差异不显著。计划目标在 64 档达到，“不变差”不使用。
- **Q1、Q2 都成立**：跨预算。
- **Q1 不成立**：方法限定为“答案依赖被点名物体本身的题目”。
- **同时报告**：按答案类型的差、路由 − A′+B+V、路由比例。
- **预测**：64 档为正，比 test1 的事后估计 +1.15 小；128 档接近 0。

### 实现核对（不满足就停止排查）

- R02、R05 记录的路由决定，等于题目文字上的规则（100%）。
- R02、R05 的保留集合与答案：被路由的题等于 A′+B+V 的（R03、R06），其余等于 A 的（R01、R04），至少 99%。
- 只有不路由的题，按查询选的部分为空。
- R03、R06 按查询选的个数等于 round(keep × S)，至少 99%。
- 离线重算的准确率与 metrics.json 一致。
- 分析脚本 `analyze_0020.py` 已在 EXP-0018 的输出上空跑通过，路由输出由两次运行拼出，只用于检查脚本，不是结果。

### 预算

- 6 次运行，约 1.7 GPU 小时（剪枝约 0.18 秒/题）。
- 候选 1 累计：EXP-0018 之后 43 次，EXP-0019 3 次，本实验 6 次，共 52 次（上限 60）。

### 代码（分支 code/token-pruning，运行前提交）

- `fastv_llava.py`、`run_eval.py`：`route` / `--fastv-route yesno_howmany`，以及 kept_tokens.json 的 `routed`。
- `exp0002.py`：`fastv{64,128}_qroute` 与 vqav2_test2。
- `make_vqav2_sets.py`：`test2`。
- `score_pope_mme.py`：vqav2_test2。
- `check_route.py`（新，17 项全部通过）。
- `analyze_0020.py`（新）。
- 原有 check_query_vis、check_mix_names、check_query_idf、check_object_cos 在改动后的代码上全部通过。

## 计划 v1.1（2026-10-25，任何运行之前；判据不变）

文献检索见 [IDEA-0003](../IDEA-0003_按题目句式路由.md) 的“相近工作与新颖性”。最接近的是 CLASP（按 9 个意图类别设预算比例）与 VIP-Router（学到的逐样本策略路由），据此做两处改动：

- **增加报告项**（不是判据）：逐题 oracle，即每题取 A 与 A′+B+V 中较好的一路，以及它与 A 的配对差，作为“这两路上任何路由”的上界。analyze_0020.py 已加入，并在 EXP-0018 的输出上重新空跑：test1 上 oracle − A 为 +6.93（64 档）、+3.61（128 档）。
- **配置**：exp0002.py 同时登记 `fastv192_qroute`。本实验不运行它，只为以后的 192 档复验准备；check_route.py 增加对应的配置核对，18 项全部通过。
- **术语**：被点名的物体写作题目的话题，不写 focus；A 的注意力排序写作“几乎不随题目变化”（F15），不写“与文本无关”。

## 计划 v1.2（2026-10-25，任何运行之前；判据不变）

为论文的 VQAv2 表补同一批题上的参照，排在 R01～R06 之后，GPU 不留空档：

| 运行 | 配置 × 数据 | 组别 |
| --- | --- | --- |
| R07 | `vanilla` × vqav2_test2 | 不剪枝参照 |
| R08 | `visionzip64` × vqav2_test2 | 参照 |
| R09 | `sparsevlm64` × vqav2_test2 | 参照 |

- 只作描述，不设判据。analyze_0020.py 报告路由 − VisionZip-64、路由 − SparseVLM-64、A − VisionZip-64 与路由 − vanilla，已在 EXP-0018 的输出上空跑。
- 预算：9 次运行，约 2.6 GPU 小时；候选 1 累计 55 次（上限 60）。
