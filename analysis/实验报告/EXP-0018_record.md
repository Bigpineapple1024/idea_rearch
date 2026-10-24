# EXPERIMENT_RECORD：VQAv2 验证集子集上的一般题目检验（第 ⑫ 步第五轮）

状态：计划 v1（2026-10-25），在任何运行之前写定。

- 实验 ID：EXP-0018。
- IDEA ID：IDEA-0002；DEC-0029。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，运行前写定）

### 问题

MME 区间太宽（半宽约 50～65 分），检不出约 ±30 的差别（DEC-0029）。在题量大的一般题目 benchmark 上，A′+B+V（IDF × 可视性、名额按可见占比分配）是否比 A（强对照 + 排除填充区）好？

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
