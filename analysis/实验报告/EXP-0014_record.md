# EXPERIMENT_RECORD：用 IDF 加权的题目查询选择图像 token（第 ⑫ 步：A / A′ / A′+B / A+B，POPE 与 MME）

状态：completed（2026-10-25）。计划 v1 在任何运行之前写定；7 次运行全部完成，核对全部通过。S1、S2 成立，S3 按字面成立：IDF 加权的题目查询在 POPE 上复现了物体名查询（83.59 对 83.63），比 A′、A 高约 17；不搬到输入空间（A+B）无效；MME-P 比 A 低 52（区间含 0），名字类子任务受损。结论见 DEC-0026。

- 实验 ID：EXP-0014。
- IDEA ID：IDEA-0002（[迁移方案](../IDEA-0002_题目词按信息量加权.md)，第 ⑨～⑪ 步）。
- 日期：2026-10-24。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-24，运行前写定）

### 要回答的问题

- **Q1（POPE）**：不知道物体名的位置时，用 IDF 加权的题目查询（A′+B），能否接近物体名查询（EXP-0011 的变体），并且明显好于等权的题目查询（A′）与强对照（A）？
- **Q2（适配是否必要）**：把同样的 IDF 权重用在第 1 层注意力行上（A+B），是否无效？
- **Q3（MME，第二个 benchmark）**：A′+B 在一般题目上是否不比 A 差？哪些子任务受益、哪些受损？

### 设计

- 模型、解码、预算与 EXP-0011 相同：LLaVA-1.5-7B，贪心解码，FastV 的 K = 2、保留 30 个（层平均 64）；全部配置都用 6×6 分块轮流取并排除填充区。
- IDF 在被评测的题目文件上算（每题一个文档，按 LLaVA 的 prompt 取图像之后的 token；log((N+1)/(df+1))），不用答案、标签或图像（2026-10-29 补注：ln 为自然对数，题目文件里没出现过的 token 记 ln(N + 1)；精确公式与代码行号见 [IDEA-0002 卡片](../IDEA-0002_题目词按信息量加权.md)的“实现公式”一节）；每次运行把 df 表写到 `answers.jsonl.idf.json`。
- 没有可调的超参数，所以不需要开发集上的挑选；POPE 与 MME 只作最终评测。

| 运行 | 配置 × 数据 | 组别 |
| --- | --- | --- |
| R01 | `fastv64_qmean_strat_nopad_kl` × POPE | A′ |
| R02 | `fastv64_qidf_strat_nopad_kl` × POPE | A′+B |
| R03 | `fastv64_attnidf_strat_nopad_kl` × POPE | A+B |
| R04 | `fastv64_prior_strat_nopad_kl` × MME | A |
| R05 | `fastv64_qmean_strat_nopad_kl` × MME | A′ |
| R06 | `fastv64_qidf_strat_nopad_kl` × MME | A′+B |
| R07 | `fastv64_attnidf_strat_nopad_kl` × MME | A+B |

POPE 上的 A 用 EXP-0011-R07（同一配置，只是没有记录对数概率）。参照：EXP-0011-R06（物体名查询）、EXP-0002 的 vanilla、FastV-64、VisionZip-64、SparseVLM-64。

### 实现核对（不满足就停止排查）

- 每次运行写出的 df 表与离线重算的相同（题目数与每个 token 的 df）。
- 只作报告：第一个回答 token 是 Yes / No 的比例；R02 与 EXP-0011-R06 保留集合的平均重合。

### 判据（与 IDEA-0002 相同）

| 编号 | 判据 | 读法 |
| --- | --- | --- |
| S1 | POPE：A′+B − A′ 与 A′+B − A 的配对差（题目在各子集内重抽 1000 次）区间都在 0 以上，原始标注与 RePOPE 都成立 | B 在适配后的系统中有效 |
| S2 | POPE：A+B − A 在 ±1 以内，或区间含 0 | 不适配时 B 无效，适配是必要的 |
| S3 | MME-P：A′+B − A 的配对差（按子任务内的图像重抽 1000 次）区间不在 0 以下 | 一般题目上不变差 |

- S1 与 S3 都成立：记为“POPE 上提升、MME 上不变差”。MME 的区间在 0 以上才记为“两个 benchmark 都提升”。
- 同时报告：POPE 的准确率、误报数、margin AUC；MME 的各子任务分数（描述）；与 VisionZip-64、SparseVLM-64 的差。
- 停止 / 回退：S1 不成立 → B 在本系统中无效，回到第 ⑨ 步；S3 不成立 → 方法限定在点名物体的题目，按题目类型分析损失来源。

### 预算

- 7 次运行，约 1.6 GPU 小时（POPE 每次约 22 分钟，MME 约 7 分钟）。
- 候选 1 累计：EXP-0011 的 7 次、EXP-0013 的 8 次加本实验 7 次，共 22 次，约 4.4 GPU 小时（上限 30 次或 20 GPU 小时）。

### 留存

- 运行目录：`code/output/EXP-0014/<RUN>/`；备份 `D:\6CJNM007\backup\EXP-0014\<RUN>\`。

### 代码（分支 code/token-pruning，与本计划同一次提交）

- `fastv_llava.py`：新增 `score_kind` 为 `query_mean`、`query_idf`（按权重平均图像之后所有 token 的输入 embedding 作查询）和 `attn_idf`（按同样的权重平均第 1 层注意力行）；`token_weights` 给出每个 token 的权重。
- `run_eval.py`：`--fastv-score query_mean / query_idf / attn_idf`；`question_idf` 在题目文件上计 df；df 表写到答案文件旁。
- `exp0002.py`：新增 `fastv64_{qmean,qidf,attnidf}_strat_nopad`；`_kl`（保留集合 + 对数概率）扩展到所有 FastV 配置。
- **CPU 自检**（`check_query_idf.py`，17 项全部通过）：
  - query_idf 的打分等于重算的“与加权平均输入 embedding 的余弦”，保留集合等于打分的分块轮流取；
  - 等权的 query_idf 与 query_mean 相同；全部权重放在物体名上时与 object_cos 相同；attn_idf 全部权重放在最后一个 token 上时与 FastV 的打分相同；
  - 用真实 tokenizer：每题都出现的 token 权重为 0（coco_dev3 22 个、POPE 21 个）；
  - 报告（不作要求）：物体名拿到的权重，coco_dev3 中位数 1.000、109 题低于 0.9（冠词 “an”）；POPE 中位数 0.995、246 题低于 0.9（原题拼写 “imange”）。
- **回归**：check_mix_names 38/38、check_object_cos 12/12、check_layer_rows_log 12/12、check_text_rows_log 13/13；check_fastv_equivalence 相对差 ≤ 3.13e-7；smoke_test 五项正常。

## 结果（R01～R07，2026-10-24～25）

分析脚本 `analyze_query.py`（code/token-pruning aff722b），输出 `code/output/EXP-0014/analysis/query.{md,json}`（SHA256SUMS 的 sha256 45d8ede7…，备份 `D:\6CJNM007\backup\EXP-0014\analysis\`）。核对全部通过：4 个 IDF 运行写出的 df 表与离线重算相同；17 个运行的重算主指标与登记值相同。

### POPE（三个子集 F1 均值）

| 组别 | 配置（运行） | F1 原始 / RePOPE | 准确率 | 召回 | 误报 adv / pop / rand |
| --- | --- | --- | --- | --- | --- |
| A | 强对照 + 排除填充区（EXP-0011-R07） | 66.90 / 72.81 | 74.44 | 51.1% | 53 / 13 / 12 |
| A′ | 等权题目查询（R01） | 65.85 / 69.80 | 72.89 | 51.7% | 135 / 62 / 47 |
| **A′+B** | **IDF 加权题目查询（R02）** | **83.59 / 86.77** | 84.51 | 78.0% | 212 / 123 / 57 |
| A+B | IDF 加权第 1 层注意力行（R03） | 66.61 / 73.22 | 74.31 | 50.7% | 48 / 17 / 7 |
| 参照 | 物体名查询（EXP-0011-R06） | 83.63 / 86.85 | 84.56 | 78.0% | 211 / 121 / 56 |
| 参照 | VisionZip-64 / SparseVLM-64 / vanilla | 77.03 / 77.67 / 85.90 | — | — | — |

| 配对差（题目在各子集内重抽 1000 次） | 原始标注 | RePOPE |
| --- | --- | --- |
| A′+B − A′（S1） | +17.74 [16.55, 19.06] | +16.97 [15.70, 18.20] |
| A′+B − A（S1） | +16.69 [15.44, 17.98] | +13.96 [12.77, 15.30] |
| A+B − A（S2） | −0.29 [−1.27, +0.67] | +0.41 [−0.64, +1.48] |
| A′+B − 物体名查询 | −0.04 [−0.19, +0.11] | −0.07 [−0.24, +0.09] |
| A′+B − VisionZip-64 / SparseVLM-64 | +6.57 / +5.93（区间都在 0 以上） | +4.99 / +4.27 |

（2026-10-29 完整阅读补注：按 DEC-0030 / AU-1，POPE 的配对差改按图像重抽（code/output/AUDIT-20261025/pope_cluster_bootstrap/cluster_boot.md，2000 次，SHA256SUMS 的 sha256 2ae290cb…）：A′+B − A′ +17.74 [15.82, 19.79]（RePOPE +16.97 [14.98, 18.96]）；A′+B − A +16.69 [14.61, 18.73]（RePOPE +13.96 [11.94, 16.06]）；A+B − A −0.29 [−1.94, +1.36]（RePOPE +0.41 [−1.31, +2.18]）；A′+B − 物体名查询 −0.04 [−0.33, +0.23]（RePOPE −0.07 [−0.34, +0.16]）。上表中的区间是按题重抽的旧口径；A′+B 与 VisionZip-64、SparseVLM-64 的差没有按图像重算。S1、S2 的判读不变。）

A′+B 与物体名查询的保留集合平均重合 97.2%。

### MME（Perception）

| 组别 | MME-P | existence | count | position | color | posters | celebrity | scene | landmark | artwork | OCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A（R04） | 1288.3 | 165.0 | 115.0 | 103.3 | 125.0 | 134.4 | 114.1 | 160.2 | 146.8 | 107.0 | 117.5 |
| A′（R05） | 1028.3 | 130.0 | 73.3 | 81.7 | 120.0 | 112.2 | 72.4 | 131.8 | 108.2 | 91.3 | 107.5 |
| **A′+B（R06）** | **1236.2** | 170.0 | 126.7 | 123.3 | 133.3 | 117.7 | 82.6 | 153.0 | 121.8 | 97.7 | 110.0 |
| A+B（R07） | 1313.3 | 165.0 | 113.3 | 111.7 | 135.0 | 131.6 | 116.2 | 161.0 | 142.0 | 105.0 | 132.5 |
| VisionZip-64 / SparseVLM-64 / vanilla | 1364.2 / 1279.6 / 1507.6 | | | | | | | | | | |

| 配对差（按子任务内的图像重抽 1000 次） | 差 [95% 区间] |
| --- | --- |
| A′+B − A（S3） | −52.1 [−133.2, +19.1] |
| A′+B − A′ | +207.8 [+130.0, +281.6] |
| A+B − A | +25.0 [−25.2, +76.8] |
| A′+B − VisionZip-64 | −128.1 [−210.2, −49.8] |

### 判读（计划 v1）

- **S1 成立**：POPE 上 A′+B 比 A′、A 都高约 17 个点，两种标注都成立；并且与需要 POPE 句式的物体名查询相同（−0.04）。
- **S2 按字面成立**：A+B 与 A 无可检出差别。审计修正：A+B 没有位置先验而 A 有，A+B − A 同时改了两件事，“适配是必要的”没有被识别；对照见 EXP-0019。（2026-10-29 完整阅读补注：EXP-0019 已补无位置先验的强对照 A₀：A+B − A₀ = +0.56 [−0.70, +1.96]（按图像重抽 2000 次），而 A′+B − A′ = +17.74，“适配是必要的”得到支持，AU-2 已关闭，见 DEC-0032。）
- **S3 按字面成立**（区间含 0），但点估计为 −52，比 VisionZip-64 低 128。点名物体的子任务合计 +45，四个名字类（celebrity、landmark、posters、artwork）合计 −82.4，加上 scene、OCR 为 −97.1（F35；原写 −96）（审计修正，DEC-0030）。
- 结论：B 在适配后有效，A′+B 可以替代需要句式的物体名查询；**“两个 benchmark 整体指标稳定提升”还没有达到**：MME 上没有提升。下一步见 DEC-0026。

### 全部运行

| 运行 ID | 配置 × 数据 | 开始 / 结束 | 状态 | 实际配置 | 原始输出 / 校验和 / 备份 | 检查 |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0014-R04 | `fastv64_prior_strat_nopad_kl` × MME | 10-24 22:43 / 22:50 | completed | commit 2a407fd，无未提交改动；`tp`；每题 0.149 秒，峰值 13.96 GiB | `code/output/EXP-0014/EXP-0014-R04/`，7f824677…；备份同名目录 | 主指标重算一致 |
| EXP-0014-R05 | `fastv64_qmean_strat_nopad_kl` × MME | 22:50 / 22:56 | completed | 同上；0.151 秒 | 96546624… | 同上 |
| EXP-0014-R06 | `fastv64_qidf_strat_nopad_kl` × MME | 22:56 / 23:02 | completed | 同上；0.149 秒 | 1e0d6e3c… | df 表一致 |
| EXP-0014-R07 | `fastv64_attnidf_strat_nopad_kl` × MME | 23:02 / 23:09 | completed | 同上；0.150 秒 | c2949d2f… | df 表一致 |
| EXP-0014-R01 | `fastv64_qmean_strat_nopad_kl` × POPE | 23:09 / 23:30 | completed | 同上；0.144 秒，峰值 13.93 GiB | 9c8854d2… | 主指标重算一致 |
| EXP-0014-R02 | `fastv64_qidf_strat_nopad_kl` × POPE | 23:30 / 23:52 | completed | 同上；0.143 秒 | 368f521a… | df 表一致 |
| EXP-0014-R03 | `fastv64_attnidf_strat_nopad_kl` × POPE | 23:52 / 10-25 00:13 | completed | 同上；0.143 秒 | 5e4ace0f… | df 表一致 |

- **预算**：7 次运行，约 1.6 GPU 小时；候选 1 累计 22 次，约 4.4 GPU 小时（上限 30 次或 20 GPU 小时）。
- **偏差**（审计补记）：POPE 运行进行时提前看过 MME 的结果（tmp/exp0014/peek_mme.py）（2026-10-29 完整阅读补注：截至该日，这份脚本只在 tmp/exp0014/peek_mme.py（849 字节，文件时间 2026-10-24 23:15:58，sha256 daa18182…），没有封存进 code/output/EXP-0014，也不在 D:\6CJNM007\backup\EXP-0014 里；这条偏差的证据还没有封存，待补），分析脚本在此之前已写好、未改。探索性检查（squeeze）的脚本错了两次（v1 的 MME 部分为空，v2 用错了 MME 的 prompt），三版都保留，采用 v3，见 [SQZ-EXP-0014](EXP-0014_squeeze.md)。（2026-10-29 完整阅读补注：三个目录封存的 squeeze_query.py 是同一个文件（sha256 8b4fd191…，即 v3），tmp/exp0014/run_sq3.sh 用它一起封存了三个目录；v1、v2 的脚本没有保留，squeeze/、squeeze_v2/ 的输出不能用目录里的脚本复现，只作为出错版本的输出保留，结论只用 v3。）

### 跑偏检查（2026-10-25）

- 当前步骤：第 ⑫ 步的第一次验证完成；按计划，MME 上没有提升，回到第 ⑨ 步处理 P10（DEC-0026）。
- 产出已同步：记录、squeeze、现象台账（F34、F35）、MODE_C-0002 回流 7、结果总表、素材库、索引、最新报告、paper-plan。
- 实验回答的是计划中的问题；POPE 与 MME 只作最终评测，没有用来挑选。
- 边界：只用 D:\6CJNM007；没有下载；候选 1 剩 8 次运行的预算。
