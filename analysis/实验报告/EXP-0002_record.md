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

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
