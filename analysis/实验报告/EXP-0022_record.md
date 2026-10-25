# EXPERIMENT_RECORD：按题目句式路由在 GQA 上的预注册检验（第三个 benchmark）

状态：计划 v1（2026-10-25），在任何 GQA 运行之前、在看到任何 GQA 答案之前写定。

- 实验 ID：EXP-0022。
- IDEA ID：[IDEA-0003](../IDEA-0003_按题目句式路由.md)；DEC-0033、DEC-0034。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。
- 来由：用户在三个后续方向（拼接组合、第二个模型、GQA / TextVQA）中同意先做 GQA（2026-10-25，“同意”），并选定官方数据（“官方全套（推荐）”）。

## 计划（v1，运行前写定）

### 问题

路由规则（助动词或 how many 开头的题用 A′+B+V，其余用 A）是在 VQAv2 test1 上看过结果后选的，只在 VQAv2 test2 上确认过一次（EXP-0020）。规则原样冻结、一字不改，在题目分布不同的 GQA 上，路由是否仍比 A 好？

### 数据

- **GQA testdev-balanced**：12578 题、398 张图（LLaVA 评测用的 `llava_gqa_testdev_balanced.jsonl`，来自 eval.zip，本地已有；prompt 与 VQAv2 相同：“Answer the question using a single word or phrase.”）。
- **答案**：官方 `questions1.2.zip` 中的 `testdev_balanced_questions.json`（1.50 GB 压缩包，downloads.cs.stanford.edu/nlp/data/gqa/）。
- **图像**：官方 `images.zip`（21.8 GB），只解压这 398 张，逐张记录 sha256。
- 两个压缩包放在 `D:\6CJNM007\data\downloads\gqa\`，下载后核对大小（HEAD 的 Content-Length：21817965542、1498616372 字节）并记录 sha256。
- **数据核对（不满足就停止排查）**：
  - 答案覆盖 12578 个 question_id，且一一对应；
  - LLaVA 题目文件的题目文字与官方文件相同；
  - 398 张图都能找到并能打开。
- **计分**：按 LLaVA 的 `convert_gqa_for_eval.py`（回答去掉句末句点、转小写）与官方 `eval.py --tier testdev_balanced` 的口径，逐题与标准答案完全匹配，求准确率。官方 eval.py 在 eval.zip 中（0.82 GB，未下载），所以在 `score_pope_mme.py` 中按同一口径实现，逐题分数写入 `gqa_scores.jsonl`。
- IDF 在 GQA 的题目文件上计（与 VQAv2 相同，不用标签）；可视性表与 EXP-0015 相同。

### 数据准备的实际情况（2026-10-25，任何 GQA 运行之前）

- `questions1.2.zip` 下载完整（1498616372 字节，与 HEAD 一致；sha256 cffe9277…）。
- **398 张图**：用 HTTP 分段请求从同一个官方 `images.zip` 读出目录和这 398 个成员（`make_gqa_set.py`），每个成员都经 zipfile 的 CRC-32 校验。逐张 sha256 在 `D:\6CJNM007\data\gqa\images_SHA256SUMS`（该文件的 sha256 9fcfa63b…），398 张都能用 PIL 打开（宽 288～640 像素）。这样做是为了不让 GPU 空等完整包下载。完整的 `images.zip`（21.8 GB）仍在后台下载，下完后核对大小、记录 sha256，并逐张比对两条路径取到的图（结果补记在本节）。**补记（2026-10-25 12:07）**：完整的 images.zip 下载完成，21817965542 字节（与 HEAD 一致），sha256 02ce5c49…；从中读出的 398 张图与分段取到的逐张 sha256 都相同（398 / 398，`D:\6CJNM007\tmp\exp0022\verify_full_zip.py`）。
- **数据核对全部通过**：官方答案覆盖 LLaVA 题目文件的 12578 个 question_id，一一对应；图像名与题目文字逐题相同。
- 逐题标注 `llava_gqa_testdev_balanced_annotations.jsonl`，sha256 23a5c89c…；LLaVA 题目文件的 sha256 515aab31…。
- **只由标注得到的事实**（不涉及任何模型输出）：
  - 标准答案为 yes / no 的题 4525 题（36.0%），全部被路由；其余 8053 题中 10.1% 被路由（二选一题）。
  - 结构类型：verify 2252、logical 1803、choose 1129、compare 589、query 6805。
- `analyze_0022.py` 已用合成的替身运行空跑：答案由标准答案按题号的固定规律构造，路由运行按规则由 A 与 A′+B+V 拼出。全部核对项的代码路径走通，替身的数字没有意义。

### 只用题目文字、运行前已知的事实

route_share_gqa.py（不读答案）：

- 路由比例 0.425，共 5340 题。开头词：is 2708、are 1145、does 987、do 497、was 2、could 1。GQA 没有 how many 题。
- 被路由的题中，**1279 题（24%）是二选一题**，题目中含 or，如 “Is the shirt red or blue?”：句式是助动词开头，答案却是属性或位置。
- 不路由的题开头词：what 3994、which 991、who 917、how 432 等。
- 每张图 3～92 题，中位数 29。

### 运行（gqa_testdev，按此顺序；主判据的运行在前）

| 运行 | 配置 | 组别 |
| --- | --- | --- |
| R01 | `fastv64_prior_strat_nopad_kl` | A（64） |
| R02 | `fastv64_qroute_kl` | 路由（64） |
| R03 | `fastv64_qvis_kl` | A′+B+V（64） |
| R04 | `fastv128_prior_strat_nopad_kl` | A（128） |
| R05 | `fastv128_qroute_kl` | 路由（128） |
| R06 | `vanilla` | 不剪枝参照 |
| R07 | `visionzip64` | 参照 |
| R08 | `sparsevlm64` | 参照 |

配置都已在 commit 9ceacd1 中，只新增数据集 `gqa_testdev`。

### 判据

| 编号 | 判据 |
| --- | --- |
| Q1（主） | 64 档：路由 − A 的 GQA 准确率配对差（按图像重抽 1000 次，种子 0）区间在 0 以上 |
| Q2 | 128 档：同上 |
| Q3（复验 F38） | 64 档：A′+B+V − A 在标准答案为 yes / no 的题上区间在 0 以上，在其余题上区间在 0 以下 |

- **Q1 成立**：冻结的规则在第三个 benchmark（与 VQAv2 题目分布不同）上复现，“路由高于强对照”在两个一般题目 benchmark 上成立。
- **Q1 不成立**：论文的主张限定为 VQAv2 式的题目分布，GQA 作为边界如实报告。**不在 GQA 上改规则**：GQA testdev 不能再用来选规则。
- **同时报告（预先写定，不是判据）**：
  - 按答案类型（yes / no 与其余）、按 GQA 的结构类型（verify、logical、choose、compare、query）的差；
  - 被路由的题中二选一与非二选一两组的 A′+B+V − A；
  - 路由 − A′+B+V；
  - 路由比例；
  - 逐题 oracle 与 A-64 / VisionZip-64 的 oracle 对照（同 SQZ-EXP-0020 S5）；
  - yes/no 题上答 yes 的比例。
- **预测**：
  - 64 档为正，但比 VQAv2 test2 的 +1.22 小，因为被路由的题中有 24% 是二选一题；区间可能含 0。
  - 二选一组的 A′+B+V − A ≤ 0。
  - 128 档比 64 档小。
- **参照**：VisionZip-64、SparseVLM-64、vanilla 只作描述。

### 实现核对（不满足就停止排查）

- R06（vanilla）的准确率与 LLaVA-1.5 论文报告的 GQA 62.0 相差不超过 1 个点（复现标准）。
- R02、R05 记录的路由决定等于题目文字上的规则（100%）。
- R02 的保留集合与答案：被路由的题等于 R03 的，其余等于 R01 的，至少 99%。R05 只核对路由决定与“不路由时按查询选的部分为空”（128 档没有 A′+B+V 运行）。
- R03 按查询选的个数等于 round(30 × S)，至少 99%。
- 离线重算的准确率与 metrics.json 一致。
- 分析脚本 `analyze_0022.py` 在运行前用替身数据空跑，只检查代码路径。

### 预算

- 8 次运行。剪枝约 0.17 秒/题，每次约 36 分钟；vanilla 约 57 分钟；共约 5.2 GPU 小时。
- 候选 1 累计：EXP-0021 后 59 次，加本实验 8 次，共 67 次。上限按 DEC-0034 从 60 调为 68，留 1 次给失败重跑。

### 代码（分支 code/token-pruning，运行前提交：commit 1d95df0）

`exp0002.py` 同时改为用 LF 写 SHA256SUMS，并在每次运行后备份 runs.tsv（跑偏检查，2026-10-25）；不影响方法与计分。

- `make_gqa_set.py`（新）：从官方压缩包写出逐题标注 `llava_gqa_testdev_balanced_annotations.jsonl`，解压 398 张图并记录 sha256，做上面的数据核对。
- `exp0002.py`：数据集 `gqa_testdev`。
- `score_pope_mme.py`：`gqa_testdev` 计分（按 LLaVA 的口径完全匹配，逐题分数写入 `gqa_scores.jsonl`）。
- `analyze_0022.py`（新）：判据、实现核对与同时报告的各项。
