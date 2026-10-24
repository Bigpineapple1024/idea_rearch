# EXPERIMENT_RECORD：VQAv2 test2 的补全（192 档路由；128 档参照）

状态：计划 v1（2026-10-25），在任何运行之前写定。EXP-0020 运行期间写定，内容不依赖 EXP-0020 的结果。

- 实验 ID：EXP-0021。
- IDEA ID：IDEA-0003；DEC-0031。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，运行前写定）

### 问题

论文的 VQAv2 表需要和 POPE、MME 一样覆盖三档预算，并在 128 档有同预算的参照。本实验只作描述：

- 192 档上路由与 A 的差；
- 128 档上 VisionZip、SparseVLM 与 A、路由（EXP-0020 的 R04、R05）在同一批题上的差。

### 运行（vqav2_test2，与 EXP-0020 相同的 5415 题）

| 运行 | 配置 | 组别 |
| --- | --- | --- |
| R01 | `fastv192_prior_strat_nopad_kl` | A（192） |
| R02 | `fastv192_qroute_kl` | 路由（192） |
| R03 | `visionzip128` | 参照 |
| R04 | `sparsevlm128` | 参照 |

配置都已在 commit 9ceacd1 中。`fastv192_qroute` 已由 check_route.py 核对，即 192 档的 A′+B+V 加路由。

### 报告（没有判据）

- 路由 − A（192），按答案类型；
- 路由-128 − VisionZip-128、路由-128 − SparseVLM-128、A-128 − VisionZip-128；
- 路由-192 − vanilla，A-192 − A-128。
- 配对差按图像重抽 1000 次，种子 0。
- **预期**：192 档路由 − A 接近 0（test1 上 128 档的三类题都接近 0）。

### 实现核对

- 离线重算的准确率与 metrics.json 一致；
- R02 记录的路由决定等于题目文字上的规则（100%）；
- R02 中只有不路由的题，按查询选的部分为空。

### 分析脚本

`D:\6CJNM007\tmp\exp0021\analyze_0021.py`。EXP-0020 运行期间工作区不能改动，所以先放在 tmp，全部运行结束后原样提交到 code/token_pruning。提交前先用 EXP-0020 的输出空跑。

### 预算

- 4 次运行，约 1.2 GPU 小时。
- 候选 1 累计 59 次（上限 60），剩 1 次留给失败重跑。用满后停下 GPU 实验，转入写作与复核，是否再加预算等用户决定。
