# EXPERIMENT_RECORD：用 IDF 加权的题目查询选择图像 token（第 ⑫ 步：A / A′ / A′+B / A+B，POPE 与 MME）

状态：计划 v1（2026-10-24），在任何运行之前写定。

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
- IDF 在被评测的题目文件上算（每题一个文档，按 LLaVA 的 prompt 取图像之后的 token；log((N+1)/(df+1))），不用答案、标签或图像；每次运行把 df 表写到 `answers.jsonl.idf.json`。
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
