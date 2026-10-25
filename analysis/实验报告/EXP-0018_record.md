# EXPERIMENT_RECORD：VQAv2 验证集子集上的一般题目检验（第 ⑫ 步第五轮）

状态：计划 v1（2026-10-25），在任何运行之前写定。2026-10-25 完成（completed）：结果、偏差与复核见文末；深挖见 [SQZ-EXP-0018](EXP-0018_squeeze.md)。

- 实验 ID：EXP-0018。
- IDEA ID：IDEA-0002；DEC-0029。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，运行前写定）

### 问题

MME 区间太宽（半宽约 35～65 分；审计修正，DEC-0030；原写 50～65），检不出约 ±30 的差别（DEC-0029）。在题量大的一般题目 benchmark 上，A′+B+V（IDF × 可视性、名额按可见占比分配）是否比 A（强对照 + 排除填充区）好？

### 数据

- VQAv2 验证集（用户 2026-10-25 许可下载；`data/downloads/vqav2/` 的两个 zip，sha256 0caae7c8…、e71f6c5c…；解出的 JSON 在 `data/vqav2/`，校验和见 SHA256SUMS）。
- `make_vqav2_sets.py`：
  - test：从 POPE 与 coco_dev 都没用到的 val2014 图像中按种子 0 抽 1000 张，取其全部题目，共 5246 题；
  - dev：coco_dev 的 300 张图上的全部题目，共 1485 题（本实验不用，留给以后的挑选）。
  - 题目文件的 sha256：test 65362002…，dev 04facedd…；标注 test 4d5ba51a…，dev e75e4559…。
- prompt 与 LLaVA 的 VQAv2 评测相同（题目 + “Answer the question using a single word or phrase.”）。LLaVA-1.5 训练用的是 VQAv2 的 train，不是 val。
- 计分：LLaVA 的 `m4c_evaluator.TextVQAAccuracyEvaluator`，即标准 VQA 准确率（EvalAI 答案归一化，min(1, 匹配数 / 3) 在 10 个留一子集上取平均）；逐题分数写到 `vqa_scores.jsonl`。

### 运行

| 运行 | 配置 × 数据 | 组别 |
| --- | --- | --- |
| R01 | `vanilla` × vqav2_test | 不剪枝参照 |
| R02 | `fastv64_prior_strat_nopad_kl` × vqav2_test | A（64） |
| R03 | `fastv64_qvis_kl` × vqav2_test | A′+B+V（64） |
| R04 | `fastv64_qidf_strat_nopad_kl` × vqav2_test | A′+B（64，描述：只用 IDF 查询） |
| R05 | `visionzip64` × vqav2_test | 参照 |
| R06 | `sparsevlm64` × vqav2_test | 参照 |
| R07 | `fastv128_prior_strat_nopad_kl` × vqav2_test | A（128） |
| R08 | `fastv128_qvis_kl` × vqav2_test | A′+B+V（128） |

IDF 在 test 题目文件上计；可视性表与 EXP-0015 相同。

### 判据

| 编号 | 判据 |
| --- | --- |
| Q1 | 64 档：A′+B+V − A 的 VQA 准确率配对差（按图像重抽 1000 次，种子 0）区间在 0 以上 |
| Q2 | 128 档：同上 |

- Q1、Q2 都成立：一般题目上也提升，与 POPE 合起来达到“两个 benchmark 整体指标都提升”（本设置）。只有 Q1 成立：只在小预算上提升。都不成立但区间含 0：一般题目上不变差。区间在 0 以下：一般题目上变差，方法限定为点名物体的题目。
- 同时报告：按答案类型（yes/no、number、other）的准确率；与 VisionZip-64、SparseVLM-64、vanilla 的差。
- 预测：number 类（数数要看到物体）与提到物体的 other 类受益；与物体无关的题（例如“这是在博物馆吗”）无差别。

### 实现核对（不满足就停止排查）

- R03、R08：按查询选的个数等于离线重算的 round(keep × S)（至少 99%）；
- 分析时离线重算的准确率与 metrics.json 一致；答案覆盖全部 5246 题。
- 计分实现与数据划分另由多代理复核（只读）；若复核在运行后才发现计分问题，答案不受影响，离线重算并记为偏差。

### 预算

- 8 次运行，约 3 GPU 小时（vanilla 约 0.28 秒/题，剪枝约 0.15 秒/题，SparseVLM 约 0.16 秒/题）；候选 1 累计 42 次（上限 60）。

### 代码（分支 code/token-pruning，与本计划同一次提交）

- `make_vqav2_sets.py`（新）、`score_pope_mme.py`（新增 vqav2_test / vqav2_dev 的计分）、`exp0002.py`（新增两个 bench）。

## 结果（R02～R09，2026-10-25）

分析 `analyze_0018.py`，输出 `code/output/EXP-0018/analysis/0018.{md,json}`（SHA256SUMS 的 sha256 a909c868…）。核对全部通过：

- R03、R08 按查询选的个数等于 round(30 S)、round(98 S)（100%）；
- 8 个运行的逐题分数离线重算，与 vqa_scores.jsonl、主指标一致；
- 答案覆盖全部 5246 题（1000 张图；yes/no 1967、number 691、other 2588）。

| 配置（vqav2_test） | 全部 | yes/no | number | other |
| --- | --- | --- | --- | --- |
| vanilla（R09） | 76.69 | 91.85 | 57.31 | 70.34 |
| A-64（R02） | 67.01 | 84.74 | 42.63 | 60.03 |
| **A′+B+V-64（R03）** | **66.26** | 86.44 | 44.98 | 56.61 |
| A′+B-64（R04，只用 IDF 查询） | 63.11 | 85.48 | 41.85 | 51.78 |
| VisionZip-64（R05） | 70.57 | 86.84 | 49.02 | 63.96 |
| SparseVLM-64（R06） | 68.43 | 86.14 | 47.99 | 60.44 |
| A-128（R07） | 73.55 | 89.55 | 51.65 | 67.23 |
| **A′+B+V-128（R08）** | **73.35** | 89.67 | 51.69 | 66.73 |

| 配对差（百分点，按图像重抽 1000 次） | 全部 | yes/no | number | other |
| --- | --- | --- | --- | --- |
| A′+B+V − A，64（Q1） | −0.74 [−1.70, +0.20] | +1.70 [+0.24, +3.12] | +2.34 [−0.76, +5.52] | −3.43 [−4.90, −1.94] |
| A′+B+V − A，128（Q2） | −0.20 [−0.91, +0.53] | +0.13 [−0.79, +1.03] | +0.04 [−2.42, +2.46] | −0.51 [−1.55, +0.54] |
| A′+B − A，64 | −3.90 [−5.02, −2.80] | +0.74 | −0.78 | −8.25 [−10.06, −6.44] |
| A′+B+V − A′+B，64 | +3.15 [+2.35, +4.00] | +0.96 | +3.13 | +4.83 [+3.52, +6.15] |
| A′+B+V − VisionZip-64 | −4.31 [−5.41, −3.16] | −0.40 | −4.04 | −7.36 |
| A′+B+V − SparseVLM-64 | −2.17 [−3.18, −1.05] | +0.30 | −3.01 | −3.83 |
| A − vanilla，64 | −9.68 [−10.72, −8.72] | −7.11 | −14.67 | −10.31 |

- **判读**：
  - Q1、Q2 都不成立。
  - 计划 v1 的字面规则是“都不成立但区间含 0：一般题目上不变差”。但两档点估计都为负，也没有设非劣效界限，按 DEC-0030 的口径写作“**差异不显著**”：64 档区间 [−1.70, +0.20]，128 档 [−0.91, +0.53]。
- **按答案类型（预设的同时报告项）**：64 档的总差接近 0，是两个相反的显著效应相抵：
  - yes/no 显著受益（+1.70）；
  - other 显著受损（−3.43）；
  - number 方向为正、不显著。
- **与预测对照**：预测“number 类与提到物体的 other 类受益”。number 方向一致但不显著；提到物体的 other 类反而受损最多（−5.14，见 SQZ-EXP-0018 S3）。所以预测的后一半被推翻。
- **只用 IDF 查询（A′+B）在一般题目上明显变差**（−3.90，主要在 other −8.25）。按可见占比分配名额（V）收回其中 3.15，与 MME 上的方向相同（F36）。
- **参照**：64 档上 A 与 A′+B+V 都低于 VisionZip-64（−3.56、−4.31）和 SparseVLM-64（−1.42、−2.17）。这与 POPE 相反：POPE 上 A′+B+V-64 比 VisionZip-64 高 5.7（82.74 vs 77.03）。在一般题目上，FastV 一族在第 2 层按注意力选（含本方法），整体弱于在视觉编码器里按 CLS 注意力选的 VisionZip。
- **全部运行**（runs.tsv 按时间排）：
  - R02 03:50 → R03 → R07 → R08 → R04 → R05 → R06 都 completed，代码 commit 0ee8b57；
  - R01 `vanilla` 05:43 **failed**：eval exit 127，27 秒后退出，stdout 无 traceback，判断为一次性的启动失败；
  - R09 是 R01 的重试（新 ID），代码 commit 0c3f70c，completed（76.69）。0c3f70c 与 0ee8b57 之间只改了 EXP-0019 的配置、bench 的校验和指针与注释，不影响 vanilla 的计算；
  - 各运行的工作区都干净（git.txt），备份在 `D:\6CJNM007\backup\EXP-0018\`。
- **预算**：9 次运行（含 1 次失败），约 2.3 GPU 小时；候选 1 累计 43 次（上限 60）。

## 偏差与复核（2026-10-25）

- **运行顺序**：vanilla 原定第一个，实际最后跑；失败的 R01 保留文件夹，重试用新 ID R09。
- **分析脚本**：从 `D:\6CJNM007\tmp\exp0018\analyze_0018.py` 运行，因为当时 EXP-0019 在 GPU 上，运行期间不改工作区。之后原样提交为 `code/token_pruning/analyze_0018.py`。
- **只读复核**：多代理工作流在运行期间做了只读复核，结论都不影响结果，记录如下。
  - **计分**：m4c 的答案归一化会删掉逗号，官方 vqaEval.py 不删，所以“标准 VQA 准确率”不完全准确。按 lmms-eval 与 LLaVA 服务端的定义重算 8 个运行：每个运行只有 0～3 题不同，准确率差 ≤0.03 个百分点，所有配置的差同向抵消（`D:\6CJNM007\tmp\verify_comma_metric\score_runs.py`）。
  - **dev 不是 test 的同分布样本**：dev 的 300 张图都有至少 3 个 COCO 类别（平均 4.51），test 只有 46.6% 有（平均 2.92）。答案类型比例 dev 为 34.8 / 14.7 / 50.4，test 为 37.5 / 13.2 / 49.3。已写进 make_vqav2_sets.py 的说明。以后若用 dev 预测 test，要另抽同分布的开发样本。EXP-0020 改用新抽的 test2 做确认。
  - **coco_dev3 的 adversarial 负样本**：类别频率与共现计数用了全部非 POPE 的 val2014 图，包括后来划进 vqav2_test 的图（989 张在计数池中，占 2.5%）。去掉这些图后，300 张图中有 3 张的 adversarial 负样本集合改变（5 张的排序改变）。按原样重建 coco_dev3 与现有文件逐字节相同（45fbc759…、ba6cd852…）。所以 vqav2_test 的图像不是严格“未被开发碰过”，影响只在 EXP-0013 的开发集负样本上，不需要重建。
  - **数据来源记录**：bench 的 `data_archives_sha256_file` 原来指向不含 VQAv2 的 `downloads/SHA256SUMS`。已在 0c3f70c 改为 `downloads/vqav2/SHA256SUMS`，所以 R02～R08 的 config.json 里是旧指针，R09 是新指针。
  - **训练数据重叠**：LLaVA-1.5 没有用 VQAv2 val 的题与答案训练，但 val2014 的图像可能出现在它基于 COCO train2017 的指令数据中（POPE 同样如此）。这影响绝对准确率，不影响同一模型上的配对比较。
