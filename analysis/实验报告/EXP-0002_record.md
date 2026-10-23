# EXPERIMENT_RECORD：视觉 token 剪枝 GPU 复现（LLaVA-1.5-7B，POPE、MME）

状态：计划 v1 已写定（2026-10-23，第一次 GPU 运行之前），执行中。
实验 ID：EXP-0002 / IDEA ID：none（baseline 复现）/ 日期：2026-10-23 起 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，2026-10-23，运行前写定）

- **要回答的问题**：
  - 在实验室电脑（RTX 3090、原生 Windows）上，LLaVA-1.5-7B vanilla 能否复现官方 POPE 85.9、MME 1510.7（LLaVA docs/MODEL_ZOO.md）。
  - 复现通过后：在同一预算口径（32 层平均视觉 token 数 192/128/64）下，FastV、VisionZip、SparseVLM 与随机选择的 POPE、MME 分数和实测速度如何。
- **假设**：
  - H0（复现门槛）：vanilla 与官方值相差不超过 1 个点。
  - H1（MODE_C-0001 的 P1/F8）：同预算下，FastV 的注意力选择不一定优于随机选择。方向不确定：S-04 报告随机在部分数据集上更好（SC）。
- **指标**：
  - 主指标：POPE 为 adversarial/popular/random 三个子集 F1 的平均（百分数，越高越好）；MME 为 Perception 总分（越高越好）。
  - 局部指标：POPE 各子集的准确率、F1、yes 比例；MME 14 个子任务分（含 Cognition）。
  - 代价：每题评测秒数（不含模型加载）、评测期间峰值显存。
- **评分管线已核验**：用同一套官方工具给 eval.zip 自带的 LLaVA-1.5-13B 官方预测打分，得 POPE 85.95、MME Perception 1531.32，与 MODEL_ZOO 的 85.9、1531.3 一致。证据在 `code/output/EXP-0002/precheck_official_13b/`。
- **组别与顺序**（run ID 按执行顺序编号）：
  - 阶段 A（门槛）：vanilla × MME、vanilla × POPE。
  - 阶段 B（阶段 A 通过后才执行）：
    1. FastV（主 baseline）：K=2，keep 166/98/30；
    2. VisionZip：dominant/contextual 162/30、108/20、54/10；
    3. SparseVLM v1.5：retained_tokens 192/128/64，batch 1；
    4. 随机选择：FastV 框架，K=2，同样的 keep 数，种子 2021/2022/2023。

    每个配置都跑 POPE 和 MME，计划共 38 次运行。
- **预算与停止/回退**：
  - 阶段 A 任一指标相差超过 1 个点：不进入阶段 B，按“环境 → 数据 → 配置 → 实现”排查；排查中的重跑用新的 run ID。复现上限一周。
  - 阶段 B 的论文对照值：从 VisionZip、SparseVLM 原文表格回查（含其中的 FastV 行），在阶段 B 开始前补入本计划（v1.1）。某个方法与论文值相差超过 1 个点时，记录并排查口径，不停止其他配置。
  - 若单次 POPE 评测超过 40 分钟：随机选择的 192、128 预算只跑种子 2021，64 预算保留 3 个种子（变更记入偏差）。
- **重复与不确定性**：
  - 贪心解码，结果确定；POPE、MME 用按样本 bootstrap（1000 次）估计 95% 区间。
  - 随机选择报告 3 个种子的均值与标准差。
- **材料与版本**：
  - 模型：`liuhaotian/llava-v1.5-7b@4481d270`、`openai/clip-vit-large-patch14-336@ce19dc91`。
  - 评测文件：LLaVA eval.zip（docs/Evaluation.md 的 Google Drive 链接）。
  - POPE 标注：AoiDragon/POPE@e3e39262；COCO val2014 来自 S3，ETag `3925cc4d…-793`，6,645,013,297 字节。
  - MME：图像 `darkyarding/MME@7056f44d`，评分工具 BradyFU 仓库 Evaluation 分支 @dd295090。
  - 所有下载文件的校验和在 `D:\6CJNM007\data\downloads\SHA256SUMS`；获取方式见 `code/token_pruning/prepare_eval_data.sh`。
- **代码与环境**：
  - 本分支的 `code/token_pruning/`（run_eval.py、fastv_llava.py、exp0002.py、score_pope_mme.py）；每次运行的 commit 记在它的 git.txt。
  - 外部仓库：LLaVA c121f043、SparseVLMs 734d9003（v1.5）、VisionZip 8f86b55c。
  - 环境：`tp`、`sparsevlm`（见 code/README 的安装结果）；RTX 3090，驱动 595.95。
- **留存**：每次运行在 `code/output/EXP-0002/<RUN>/`，按研究流程保存全部文件和 SHA256SUMS，并复制到 `D:\6CJNM007\backup\EXP-0002\<RUN>\`（复制后重新核对校验和）。

## 计划 v1.1：论文对照值（2026-10-23，阶段 B 开始前补入）

来源是与论文笔记相同版本的 PDF（sha256 与 `analysis/论文笔记/sources-20261023.json` 一致），放在 `papers/`（不提交），用 PyMuPDF 取出表格文字：

- VisionZip：CVPR 2025 正式版，第 5 页 Table 1；
- SparseVLM：arXiv:2410.04417v4（ICML 2025，对应 v1.5 分支），第 6 页 Table 1。

两篇的 MME 都是 Perception + Cognition 总分（vanilla 分别为 1862、1864），所以本实验同时报告 Perception（与 LLaVA 官方 1510.7 比）和 P+C 总分（与两篇论文比）。POPE 的 vanilla 都是 85.9，与 LLaVA 官方一致。

| 预算 | 方法 | VisionZip 表（MME P+C / POPE） | SparseVLM v4 表（MME P+C / POPE） |
| --- | --- | --- | --- |
| 576 | vanilla | 1862 / 85.9 | 1864 / 85.9 |
| 192 | FastV | 1612 / 64.8 | 1605 / 64.8 |
| 192 | SparseVLM | 1721 / 83.6（引用的 2024.10 版） | 1787 / 85.3 |
| 192 | VisionZip | 1782.6 / 85.3 | — |
| 128 | FastV | 1490 / 59.6 | 1490 / 53.4 |
| 128 | SparseVLM | 1696 / 80.5（同上） | 1746 / 85.0 |
| 128 | VisionZip | 1761.7 / 83.2 | — |
| 64 | FastV | 1256 / 48.0 | 1255 / 38.2 |
| 64 | SparseVLM | 1505 / 75.1（同上） | 1589 / 77.5 |
| 64 | VisionZip | 1690 / 77.0 | — |

判断标准：

- VisionZip 与它自己的表比较，SparseVLM 与 v4 表比较；相差超过 1 个点（POPE）就记录并排查口径。
- 两篇论文中的 FastV 行彼此不一致（例如 128 预算的 POPE 为 59.6 与 53.4），而且都没写明 K 和保留数。本实验的 FastV 按“32 层平均相等”取 K=2、keep 166/98/30（code/token_pruning/README.md），所以 FastV 只报告，不设复现门槛。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0002-R01 | vanilla × MME | 10-23 19:28 / 19:28 | failed（评测退出码 1） | commit 9bbc249，tp 环境 | 0 | `code/output/EXP-0002/EXP-0002-R01/`，SHA256SUMS 的 sha256 d80f67d2…；备份同名目录 | LLaVA 的 LlamaTokenizer（legacy=False）需要 protobuf，LLaVA 依赖清单里没有；在 tp、sparsevlm 装 protobuf 4.25.3 后重跑为 R02 |
| EXP-0002-R02 | vanilla × MME | 10-23 19:30 / 19:42 | completed | commit e20b1d1，sdpa 注意力（transformers 4.37.2 默认） | 2374 | 同上目录规则，sha256 626c24bf… | MME-P 1507.63，比官方低 3.07，超出计划的 1 分门槛 |
| EXP-0002-R03 | vanilla（eager 注意力）× MME | 10-23 19:43 / 19:55 | completed | commit c467c24，`--attn-implementation eager` | 2374 | sha256 58499555… | 门槛诊断（环境排查）：只换注意力实现 |
| EXP-0002-R04 | vanilla × POPE | 10-23 19:56 / 20:38 | completed | commit c467c24，sdpa，DataLoader worker 0 | 8910 | sha256 见 runs.tsv | F1 均值 85.899，与官方 85.9 一致；评测 2492.7 秒（0.280 秒/题） |
| EXP-0002-R05 | vanilla × MME（4 个 DataLoader worker） | 10-23 20:39 / 20:50 | completed | commit 109134d | 2374 | sha256 184c9e8f… | 加载方式核验：2374 个答案与 R02 逐条相同（题号、提示、答案文本）；MME-P 1507.63；0.265 秒/题，比 R02 的 0.286 快约 7% |

**阶段 B 的执行设置（偏差记录）**：

- 加载方式：从 R05 起统一用 4 个 worker。答案与 0 个 worker 时完全相同，只影响速度；瓶颈是 batch 1 的 generate 循环本身，GPU 利用率约 38%。
- 种子数：按计划的 40 分钟规则（R04 的 POPE 用时 41.5 分钟），随机选择在 192、128 预算只跑种子 2021，64 预算跑 2021/2022/2023，阶段 B 共 14 个配置。
- 速度参照：最后补一次 4 个 worker 的 vanilla × POPE，作为 POPE 上的速度参照。

## 阶段 A：vanilla 门槛与排查

- **MME（R02、R03）**：
  - R02（sdpa）：Perception 1507.63、Cognition 357.86、P+C 1865.49。bootstrap 95% 区间 [1440.6, 1569.3]（1000 次，按图像重采样）。
  - R03（eager）：Perception 1514.56、Cognition 357.86、P+C 1872.41。
  - 两者只差 4 个答案（2374 题中），集中在 position（128.33→133.33）、celebrity（136.18→137.35）、landmark（163.00→163.75）。官方 1510.7 落在两者之间（−3.07 / +3.86）。
  - 按计划的字面标准（相差不超过 1 分），两种注意力实现都没通过。
- **排查（按“环境 → 数据 → 配置 → 实现”）**：
  - 环境：只换注意力内核就使 MME-P 变化 6.93 分（4 个答案翻转），说明这个量级的差异来自浮点数值。官方数字产生于 2023 年 10 月的 transformers 4.31，原环境无法完全复原。
  - 数据：MME 图像包的 sha256 与 Hugging Face 的 LFS 记录一致；题目文件来自官方 eval.zip。
  - 评分：同一管线给官方 13B 预测打分，得到与 MODEL_ZOO 完全一致的 1531.3。
  - 配置：与官方 mme.sh 相同（温度 0、vicuna_v1、max_new_tokens 128）。
  - 实现：vanilla 走未修改的 LLaVA 代码，只把 DataLoader worker 设为 0。
- **POPE（R04）**：三子集 F1 均值 85.899（官方 85.9），准确率均值 86.871。
  - adversarial：F1 84.21、准确率 85.17、yes 比例 43.97%；
  - popular：F1 86.14、准确率 87.27、yes 比例 41.87%；
  - random：F1 87.34、准确率 88.18、yes 比例 41.86%；
  - 三个子集的召回率都是 79.13%（正样本题相同，只有负样本不同）。
- **门槛判定（2026-10-23，偏离计划 v1，理由如下）**：
  - POPE 通过（差 0.00）。MME-P 按字面标准未通过（sdpa −3.07）。
  - 进入阶段 B 的理由：MME-P 是 10 个子任务 (acc + acc+)×100 之和，一个答案的变化就移动约 1～5 分；只换注意力内核就移动 6.93 分（4/2374 个答案）。因此 1 分门槛低于这项评测在数值上的分辨率。数据、评分、配置、实现已逐项核对。
  - 阶段 B 的所有比较都在同一环境（sdpa，与 R02/R04 相同）内进行，固定的数值偏移不影响方法间比较。
  - 记录的是事实结果（MME-P 1507.63，未达字面门槛），不改写为通过。若需要逐比特复原官方环境（transformers 4.31 等），另行安排。
