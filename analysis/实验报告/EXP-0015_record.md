# EXPERIMENT_RECORD：按词的可视性分配名额的题目查询（第 ⑨～⑫ 步第二轮，P10）

状态：计划 v1（2026-10-25），在任何运行之前写定。

- 实验 ID：EXP-0015。
- IDEA ID：IDEA-0002 的第二轮（[迁移方案](../IDEA-0002_题目词按信息量加权.md)）；MODE_C-0002 的 C17。
- 日期：2026-10-25。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-25，运行前写定）

### 来源与第 ⑨ 步

- **问题 P10**（F35）：MME 上，IDF 查询（A′+B）在名字类子任务受损；这些题里 IDF 最高的是人名、地名的词片，在输入 embedding 空间里没有对应的图像区域。
- **领域 B**：心理语言学的词的具体性 / 可意象性：具体词有可见的指称（原文 PDF 未读，按用户许可后再核对）。
- **领域 C**：Hessel 等，Quantifying the visual concreteness of words and topics in multimodal datasets（NAACL 2018，arXiv 1804.06786，ar5iv 全文已读）：
  - 从图文数据集里按最近邻统计每个词的“视觉具体性”；
  - 它能解释跨模态检索效果的方差，词频几乎不能；
  - 具体性依赖数据集（例：“London” 在 COCO 中具体，在 Flickr 中不具体）。
- **迁移（第 ⑪ 步）**：不用图文对，改用本系统自己的空间。一个词的可视性 = 它的输入 embedding 能否在图像上“挑出一块区域”。

### 规则（全部在运行前写定，没有可调参数）

1. **可视性**（`make_visibility.py`，只用开发集 coco_dev 的 300 张 COCO 图，不碰 POPE、MME 的图像）：
   - 每张图、每个词表 id：peak = 该 id 的输入 embedding 与图像 token 输入 embedding 余弦的最大值 − 均值；
   - 原始可视性 = peak 在 300 张图上的 95% 分位；
   - 使用值 = 原始可视性在全部 32000 个词表 id 中的百分位（0～1）。
2. **查询**：每个图像之后的 token 的权重 = IDF（在被评测的题目文件上计，同 EXP-0014）× 可视性；查询 = 加权平均的输入 embedding。
3. **名额**：可见信息量占比 S = Σ(IDF × 可视性) / Σ IDF；round(30 × S) 个 token 按与查询的余弦选（分块、排除填充区），其余按强对照的分数（去位置、分块、排除填充区），第二轮先填空块。S = 0 时等于 A，S = 1 时与 A′+B 的查询相同。

### 运行

| 运行 | 配置 × 数据 | 用途 |
| --- | --- | --- |
| R01 | `vanilla_vocabvis` × coco_dev | 记录每张图上全部词表 id 的最大、平均余弦（并记录答案的对数概率） |
| — | `make_visibility.py`（CPU） | 生成可视性表 `code/output/EXP-0015/visibility/` |
| R02 | `fastv64_qvis_kl` × MME | A′+B+V |
| R03 | `fastv64_qvis_kl` × POPE | A′+B+V |

可视性表做好后、R02 之前，只报告（不改规则）：POPE、MME 各子任务的 S 分布，以及几个词的可视性。

### 判据

| 编号 | 判据 | 读法 |
| --- | --- | --- |
| V1（MME，主） | A′+B+V − A（EXP-0014-R04）的 MME-P 配对差（子任务内图像重抽 1000 次）区间在 0 以上 | MME 上提升 |
| V1′ | 点估计 > 0、区间含 0 | MME 上不变差，但提升未被证明 |
| V2（POPE） | A′+B+V − A（EXP-0011-R07）的 F1 配对差区间在 0 以上，原始标注与 RePOPE 都成立 | POPE 上提升 |

- V1 与 V2 都成立：在 POPE 与 MME 两个 benchmark 上，整体指标都比 A 提升，达到 paper-plan 第 ⑨～⑫ 步的目标（本设置：LLaVA-1.5-7B、层平均 64）。
- 同时报告：与 A′+B（EXP-0014-R02、R06）的差（可视性的代价与收益）；MME 各子任务；与 VisionZip-64、SparseVLM-64 的差；POPE 的准确率与误报。
- **停止 / 回退**：V1 不成立时，方法的适用范围限定为“题目点名可见物体”，论文中写清；剩余预算（5 次）用于在 128 档复验 POPE 的结论。

### 实现核对（不满足就停止排查）

- R02、R03：每题记录的“按查询选的部分”的个数等于 round(30 × S)，S 用离线的 IDF 与可视性表重算（至少 99% 的题）。
- 可视性表的值在 [0, 1]，与原始值同序。

### 预算

- 3 次运行，约 0.7 GPU 小时；候选 1 累计 25 次，约 5.1 GPU 小时（上限 30 次或 20 GPU 小时）。

### 代码（分支 code/token-pruning，与本计划同一次提交）

- `fastv_llava.py`：新增 `score_kind="query_vis"`（IDF × 可视性的查询；名额按 S 分给查询与强对照的分数）。
- `run_eval.py`：`--fastv-score query_vis`、`--fastv-visibility`、`--vocab-visibility-log`。
- `exp0002.py`：`vanilla_vocabvis`、`fastv64_qvis`（及 `_kl`）。
- `make_visibility.py`：可视性表。
- **CPU 自检**（`check_query_vis.py`，14 项全部通过）：可视性全为 1 时与 query_idf 相同；全为 0 时与强对照相同；随机可视性下保留集合等于离线两轮规则，记录的个数与占比正确；词表记录每张图一行、数值等于重算；可视性表在 [0, 1] 且与原始值同序；参数检查。
- **回归**：check_query_idf 17/17、check_mix_names 38/38、check_object_cos 12/12、check_layer_rows_log 12/12、check_text_rows_log 13/13；FastV 等价相对差 ≤ 3.13e-7；smoke_test 五项正常。
