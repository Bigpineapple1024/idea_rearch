# EXPERIMENT_RECORD：审计要求的两项补充（A+B 的对照；最终方法的区分力）

状态：计划 v1（2026-10-25），在任何运行之前写定（DEC-0030 的 AU-2、AU-16）。

- 实验 ID：EXP-0019。
- IDEA ID：IDEA-0002；DEC-0030。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，运行前写定）

### 问题

- **Q1（AU-2）**：EXP-0014 的 A+B（IDF 加权第 1 层注意力行）没有位置先验，而 A 有。换成改动只有一处的对照“A 去掉位置先验”（最后一个 token 行 + 分块 + 排除填充区），IDF 加权注意力行（B）在不适配时有没有作用？
- **Q2（AU-16，落实 DEC-0024）**：最终方法 A′+B+V 的 yes/no 区分力（margin AUC）与 vanilla、A 相比如何？误报在三个子集上是多少？

### 运行

| 运行 | 配置 × 数据 | 用途 |
| --- | --- | --- |
| R01 | `fastv64_strat_nopad_kl` × POPE | A 去掉位置先验（A₀）；与 EXP-0014-R03（A+B）只差打分行 |
| R02 | `fastv64_prior_strat_nopad_kl` × POPE | A 带对数概率；答案必须与 EXP-0011-R07 逐题相同（复现核对） |
| R03 | `vanilla_logits` × POPE | vanilla 的对数概率 |

A′+B+V 的对数概率来自 EXP-0015-R03（`fastv64_qvis_kl`）。

### 判据与报告

- **Q1**：A+B（EXP-0014-R03）− A₀（R01）的 POPE F1 配对差，按图像重抽 2000 次（DEC-0030 的新规则）。区间在 0 以上：B 在不适配时也有作用，“适配必要”不成立；区间含 0：B 只在适配后起作用（与 A′+B − A′ +17.74 对照）；区间在 0 以下：B 在不适配时有害。同时报告 A₀ − A（位置先验的作用）。
- **Q2**：三个子集上 margin（log P(Yes) − log P(No)）区分正负样本的 AUC，A′+B+V、A、vanilla；配对差按图像重抽；三个子集的误报数与准确率。只作描述，没有门槛。
- 实现核对：R02 的答案与 EXP-0011-R07 逐题相同（不同就停止排查）；所有运行第一个回答 token 都是 Yes / No（报告比例）。

### 预算

- 3 次运行，约 1.3 GPU 小时；候选 1 累计 45 次（上限 60）。

### 代码

- `exp0002.py`：新增 `fastv64_strat_nopad`（`fastv64_strat` 加排除填充区）。
