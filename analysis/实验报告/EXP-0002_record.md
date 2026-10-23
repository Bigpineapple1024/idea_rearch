# EXPERIMENT_RECORD：视觉 token 剪枝 GPU 复现（LLaVA-1.5-7B，POPE、MME）

状态：completed（2026-10-24）。计划 v1 写定于 2026-10-23，在第一次 GPU 运行之前。阶段 A 的门槛有一项偏离计划（MME-P，见“阶段 A”）；阶段 B 的 14 个配置全部完成；另做了计划外的 SparseVLM 逐层视觉 token 诊断。
实验 ID：EXP-0002 / IDEA ID：none（baseline 复现）/ 日期：2026-10-23～24 / 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 结构化交接包

```yaml
package_type: EXPERIMENT_RECORD
package_version: research-pipeline-contract v1.0
package_id: EXP-0002
experiment_id: EXP-0002
created_at: 2026-10-24
source_ids: [PAPER-0004, PAPER-0005, PAPER-0006, MODE_C-0001, EXP-0001]
evidence_status: verified
evidence_locations:
- code/output/EXP-0002/<RUN>/（每次运行一个目录：cmd.txt、config.json、git.txt、seed.txt、env.txt、gpu.txt、stdout.log、answers.jsonl(+.meta.json)、score.log、metrics.json、status.json、SHA256SUMS）
- code/output/EXP-0002/runs.tsv（全部 49 次运行，含每个目录 SHA256SUMS 的 sha256）
- code/output/EXP-0002/analysis/tables.md、summary.json（由 code/token_pruning/analyze_exp0002.py 生成，目录内附生成时的脚本副本；analysis/SHA256SUMS 的 sha256 bf1e369c…）。第一版（还没有方法间比较）保留为 analysis_v1（SHA256SUMS 的 sha256 1fa8a21f…），新版只在末尾追加了一张表，前面的内容逐字相同
- 备份：D:\6CJNM007\backup\EXP-0002\（49 个运行目录 + precheck_official_13b + analysis + analysis_v1，复制后逐文件核对校验和）
missing_evidence:
- 未逐比特复原 LLaVA 官方 2023-10 的环境（transformers 4.31），MME-P 的 −3.07 无法进一步归因
- SparseVLM 用 SDPA 代替 FlashAttention-2 运行（Windows 没有 flash-attn），未与官方内核逐答案比较
- 只有 POPE、MME；GQA、TextVQA 等未运行
- 速度是 batch 1 下的每题秒数（含 Python 与生成开销），不是单独测量的预填充延迟
- 被保留 token 的空间位置未记录（解释 FastV 劣于随机选择所需）
next_skill: squeeze-experiment
source_idea_id: none
plan_version: "v1（2026-10-23，第一次运行前）+ v1.1（论文对照值，阶段 B 开始前）"
execution_status: completed
research_question: 实验室电脑上 vanilla LLaVA-1.5-7B 能否复现官方 POPE/MME；同一层平均预算下 FastV、VisionZip、SparseVLM 与随机选择的 POPE、MME 与速度
hypothesis: H0 vanilla 与官方相差 ≤1 个点；H1 FastV 的注意力选择不一定优于同预算随机选择（方向未定）
research_question_and_hypothesis:
  research_question: 同上
  hypothesis: 同上
planned_vs_actual_design_and_controls: 见“计划”“全部运行”“偏差与限制”
actual_configuration: 见“全部运行”及每次运行的 config.json、cmd.txt、git.txt
samples_materials_or_data_provenance: POPE 8910 题（COCO val2014）、MME 2374 题；版本与校验和见“计划”中的材料与版本
all_runs_or_repetitions: 49 次（R01～R49）。失败 7 次（R01、R16、R18～R23）；诊断 8 次（R42～R49），其中 R43～R45 有缺陷、不用于分析
results_with_uncertainty: 见“结果”（bootstrap 95% 区间；随机选择 64 档为 3 个种子的均值 ± 标准差）
observations_and_deviations: 见“观察与解释”“偏差与限制”
known_limits_and_missing_evidence: 见 missing_evidence 与“偏差与限制”
next_routes: [squeeze-experiment（EXP-0002_squeeze）, deep-read Mode C（合并两份视觉 Mode C 并回流本记录）]
```

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

主指标：POPE 为三子集 F1 均值（%），MME 为 Perception 总分，诊断运行（pope50）为 32 层平均视觉 token 数。49 个 git.txt 中，只有 R04 记录了一个未跟踪文件（`code/token_pruning/bootstrap_ci.py`，当时尚未提交）；其余运行开始时本仓库没有未提交改动，外部仓库均为 clean。R06～R25 的 commit 为 f5087a9；R26 起为 e4636ec，这次提交加入了 SDPA 替代，只改变 sparsevlm 路径。

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0002-R01 | vanilla × MME | 10-23 19:28 / 19:28 | failed（评测退出码 1） | commit 9bbc249，tp 环境 | 0 | `code/output/EXP-0002/EXP-0002-R01/`，SHA256SUMS 的 sha256 d80f67d2…；备份同名目录 | LLaVA 的 LlamaTokenizer（legacy=False）需要 protobuf，LLaVA 依赖清单里没有；在 tp、sparsevlm 装 protobuf 4.25.3 后重跑为 R02 |
| EXP-0002-R02 | vanilla × MME | 10-23 19:30 / 19:42 | completed | commit 9bbc249，sdpa 注意力（transformers 4.37.2 默认） | 2374 | 同上目录规则，sha256 626c24bf… | MME-P 1507.63，比官方低 3.07，超出计划的 1 分门槛 |
| EXP-0002-R03 | vanilla（eager 注意力）× MME | 10-23 19:43 / 19:55 | completed | commit c467c24，`--attn-implementation eager` | 2374 | sha256 58499555… | 门槛诊断（环境排查）：只换注意力实现 |
| EXP-0002-R04 | vanilla × POPE | 10-23 19:56 / 20:38 | completed | commit c467c24，sdpa，DataLoader worker 0 | 8910 | sha256 18ff3da6… | F1 均值 85.899，与官方 85.9 一致；评测 2492.7 秒（0.280 秒/题） |
| EXP-0002-R05 | vanilla × MME（4 个 DataLoader worker） | 10-23 20:39 / 20:50 | completed | commit 109134d | 2374 | sha256 184c9e8f… | 加载方式核验：2374 个答案与 R02 逐条相同（题号、提示、答案文本）；MME-P 1507.63；0.265 秒/题，比 R02 的 0.286 快约 7% |
| EXP-0002-R06 | fastv192 × mme | 10-23 20:52 / 20:59 | completed | commit f5087a9；主指标 1,464.39 | 2374 | sha256 711f73d1… |  |
| EXP-0002-R07 | fastv192 × pope | 10-23 20:59 / 21:24 | completed | commit f5087a9；主指标 75.76 | 8910 | sha256 67e9f907… |  |
| EXP-0002-R08 | fastv128 × mme | 10-23 21:24 / 21:31 | completed | commit f5087a9；主指标 1,363.21 | 2374 | sha256 a324c870… |  |
| EXP-0002-R09 | fastv128 × pope | 10-23 21:31 / 21:53 | completed | commit f5087a9；主指标 68.29 | 8910 | sha256 863c8eba… |  |
| EXP-0002-R10 | fastv64 × mme | 10-23 21:53 / 21:59 | completed | commit f5087a9；主指标 979.58 | 2374 | sha256 e9afa151… |  |
| EXP-0002-R11 | fastv64 × pope | 10-23 21:59 / 22:20 | completed | commit f5087a9；主指标 36.19 | 8910 | sha256 664b3a77… |  |
| EXP-0002-R12 | visionzip192 × mme | 10-23 22:20 / 22:27 | completed | commit f5087a9；主指标 1,442.71 | 2374 | sha256 8e49f79d… |  |
| EXP-0002-R13 | visionzip192 × pope | 10-23 22:27 / 22:51 | completed | commit f5087a9；主指标 85.49 | 8910 | sha256 70fb10d9… |  |
| EXP-0002-R14 | visionzip128 × mme | 10-23 22:51 / 22:57 | completed | commit f5087a9；主指标 1,432.45 | 2374 | sha256 8329444a… |  |
| EXP-0002-R15 | visionzip128 × pope | 10-23 22:57 / 23:19 | completed | commit f5087a9；主指标 83.12 | 8910 | sha256 47a6eca1… |  |
| EXP-0002-R16 | visionzip64 × mme | 10-23 23:19 / 23:19 | failed（退出码 127） | commit f5087a9；主指标 NA | 0 | sha256 6b7c0f89… | 评测开始时进程退出（退出码 127），无 Python 报错，应用事件日志无记录；同配置 R17 正常；重跑为 R35 |
| EXP-0002-R17 | visionzip64 × pope | 10-23 23:19 / 23:39 | completed | commit f5087a9；主指标 77.03 | 8910 | sha256 af1447f5… |  |
| EXP-0002-R18 | sparsevlm192 × mme | 10-23 23:39 / 23:40 | failed（退出码 1） | commit f5087a9；主指标 NA | 0 | sha256 fce2e177… | NameError: flash_attn_func 未定义（SparseVLM 每层挂 FlashAttention-2，Windows 无 flash-attn）；重跑为 R36 |
| EXP-0002-R19 | sparsevlm192 × pope | 10-23 23:40 / 23:41 | failed（退出码 1） | commit f5087a9；主指标 NA | 0 | sha256 bfd86c5f… | 同 R18；重跑为 R37 |
| EXP-0002-R20 | sparsevlm128 × mme | 10-23 23:41 / 23:41 | failed（退出码 1） | commit f5087a9；主指标 NA | 0 | sha256 b61c3000… | 同 R18；重跑为 R38 |
| EXP-0002-R21 | sparsevlm128 × pope | 10-23 23:41 / 23:42 | failed（退出码 1） | commit f5087a9；主指标 NA | 0 | sha256 01e1b765… | 同 R18；重跑为 R39 |
| EXP-0002-R22 | sparsevlm64 × mme | 10-23 23:42 / 23:42 | failed（退出码 1） | commit f5087a9；主指标 NA | 0 | sha256 ce3eabad… | 同 R18；重跑为 R40 |
| EXP-0002-R23 | sparsevlm64 × pope | 10-23 23:42 / 23:43 | failed（退出码 1） | commit f5087a9；主指标 NA | 0 | sha256 865af7cc… | 同 R18；重跑为 R41 |
| EXP-0002-R24 | random192_s2021 × mme | 10-23 23:43 / 23:50 | completed | commit f5087a9；主指标 1,417.13 | 2374 | sha256 0f1c360f… |  |
| EXP-0002-R25 | random192_s2021 × pope | 10-23 23:50 / 00:15 | completed | commit f5087a9；主指标 80.39 | 8910 | sha256 02fb004d… |  |
| EXP-0002-R26 | random128_s2021 × mme | 10-24 00:15 / 00:21 | completed | commit e4636ec；主指标 1,355.03 | 2374 | sha256 79216c52… |  |
| EXP-0002-R27 | random128_s2021 × pope | 10-24 00:21 / 00:43 | completed | commit e4636ec；主指标 76.65 | 8910 | sha256 31390f04… |  |
| EXP-0002-R28 | random64_s2021 × mme | 10-24 00:43 / 00:49 | completed | commit e4636ec；主指标 1,168.17 | 2374 | sha256 eb4384df… |  |
| EXP-0002-R29 | random64_s2021 × pope | 10-24 00:49 / 01:09 | completed | commit e4636ec；主指标 60.26 | 8910 | sha256 16b6cede… |  |
| EXP-0002-R30 | random64_s2022 × mme | 10-24 01:09 / 01:15 | completed | commit e4636ec；主指标 1,167.88 | 2374 | sha256 29dd87a3… |  |
| EXP-0002-R31 | random64_s2022 × pope | 10-24 01:15 / 01:35 | completed | commit e4636ec；主指标 60.38 | 8910 | sha256 dceca042… |  |
| EXP-0002-R32 | random64_s2023 × mme | 10-24 01:35 / 01:41 | completed | commit e4636ec；主指标 1,122.50 | 2374 | sha256 7b91b0a8… |  |
| EXP-0002-R33 | random64_s2023 × pope | 10-24 01:41 / 02:01 | completed | commit e4636ec；主指标 59.77 | 8910 | sha256 efc07a8a… |  |
| EXP-0002-R34 | vanilla × pope | 10-24 02:01 / 02:37 | completed | commit e4636ec；主指标 85.90 | 8910 | sha256 91cbb650… | POPE 速度参照（4 个 worker）；8910 个答案与 R04 逐条相同 |
| EXP-0002-R35 | visionzip64 × mme | 10-24 02:38 / 02:44 | completed | commit e4636ec；主指标 1,364.22 | 2374 | sha256 1a13b8a8… | R16 的重跑 |
| EXP-0002-R36 | sparsevlm192 × mme | 10-24 02:44 / 02:52 | completed | commit e4636ec；主指标 1,447.98 | 2374 | sha256 d1d7e395… | SDPA 替代 FlashAttention-2（sparsevlm_sdpa.py），meta.json 记录 |
| EXP-0002-R37 | sparsevlm192 × pope | 10-24 02:52 / 03:20 | completed | commit e4636ec；主指标 85.29 | 8910 | sha256 94fa3d7c… | SDPA 替代 FlashAttention-2 |
| EXP-0002-R38 | sparsevlm128 × mme | 10-24 03:20 / 03:27 | completed | commit e4636ec；主指标 1,434.98 | 2374 | sha256 b03d76d8… | SDPA 替代 FlashAttention-2 |
| EXP-0002-R39 | sparsevlm128 × pope | 10-24 03:27 / 03:52 | completed | commit e4636ec；主指标 85.07 | 8910 | sha256 658c1edb… | SDPA 替代 FlashAttention-2 |
| EXP-0002-R40 | sparsevlm64 × mme | 10-24 03:52 / 03:59 | completed | commit e4636ec；主指标 1,279.59 | 2374 | sha256 b7fde804… | SDPA 替代 FlashAttention-2 |
| EXP-0002-R41 | sparsevlm64 × pope | 10-24 03:59 / 04:22 | completed | commit e4636ec；主指标 77.67 | 8910 | sha256 3be2af55… | SDPA 替代 FlashAttention-2 |
| EXP-0002-R42 | fastv64_layers × pope50 | 10-24 04:23 / 04:24 | completed | commit 97bda7b；主指标 64.13 | 50 | sha256 dc698c8d… | 逐层 token 诊断（50 题）；钩子核验：FastV-64 已知答案 64.125 |
| EXP-0002-R43 | sparsevlm192_layers × pope50 | 10-24 04:24 / 04:24 | completed | commit 97bda7b；主指标 190.31 | 50 | sha256 8f289551… | 诊断有缺陷：钩子漏记 SparseVLM 用关键字调用的 3 个剪枝层，只记 29 层；不用于分析 |
| EXP-0002-R44 | sparsevlm128_layers × pope50 | 10-24 04:24 / 04:25 | completed | commit 97bda7b；主指标 125.79 | 50 | sha256 c40cb558… | 同 R43 |
| EXP-0002-R45 | sparsevlm64_layers × pope50 | 10-24 04:25 / 04:26 | completed | commit 97bda7b；主指标 66.97 | 50 | sha256 3eed52d0… | 同 R43 |
| EXP-0002-R46 | fastv64_layers × pope50 | 10-24 04:28 / 04:28 | completed | commit a2fa660；主指标 64.13 | 50 | sha256 dc9c89ed… | 修复后的钩子（with_kwargs）；FastV-64 核验 64.125 |
| EXP-0002-R47 | sparsevlm192_layers × pope50 | 10-24 04:28 / 04:29 | completed | commit a2fa660；主指标 206.50 | 50 | sha256 63b656ad… | SparseVLM 每层实际视觉 token，32 层全部记录 |
| EXP-0002-R48 | sparsevlm128_layers × pope50 | 10-24 04:29 / 04:30 | completed | commit a2fa660；主指标 145.41 | 50 | sha256 c5b1b6df… | 同 R47 |
| EXP-0002-R49 | sparsevlm64_layers × pope50 | 10-24 04:30 / 04:30 | completed | commit a2fa660；主指标 82.25 | 50 | sha256 6856ab69… | 同 R47 |

**阶段 B 的执行设置（偏差记录）**：

- 加载方式：从 R05 起统一用 4 个 worker。答案与 0 个 worker 时完全相同，只影响速度；瓶颈是 batch 1 的 generate 循环本身，GPU 利用率约 38%。
- 种子数：按计划的 40 分钟规则（R04 的 POPE 用时 41.5 分钟），随机选择在 192、128 预算只跑种子 2021，64 预算跑 2021/2022/2023，阶段 B 共 14 个配置。
- 速度参照：最后补一次 4 个 worker 的 vanilla × POPE（R34），作为 POPE 上的速度参照。
- 失败与重跑（全部保留原目录，重跑用新 ID）：
  - R16（VisionZip-64 × MME）：评测刚开始时进程以退出码 127 结束，没有 Python 报错，Windows 应用事件日志无记录；同配置的 R17（POPE）紧接着正常完成。原因未知，按偶发处理，重跑为 R35。
  - R18～R23（SparseVLM 全部 6 次）：`NameError: name 'flash_attn_func' is not defined`。SparseVLM 的 `load_pretrained_model` 默认给 32 层都挂上 `LlamaDynamicvitFlashAttention2`（`llava/model/builder.py` 第 114～121 行），每层注意力都调用 FlashAttention-2；此前“v1.5 评测不需要 flash-attn”的判断是错的（导入发生在 transformers 内部）。flash-attn 没有官方 Windows 版本，于是写 `code/token_pruning/sparsevlm_sdpa.py`，运行时只把这一个函数换成 SDPA（与显式 softmax 注意力的差 ≤2.1e-07），重跑为 R36～R41。
- 计划外的诊断（回答 SQZ-EXP-0001 的 P-SQZ1-2）：
  - R42～R45：50 道 POPE（每 178 题取 1 道，三个子集 17/16/17），用前向 hook 记录每层输入的序列长度。R42（FastV-64）给出已知答案 64.125，钩子在这条路径上正确。
  - R43～R45 有缺陷：SparseVLM 用关键字参数调用它的 3 个剪枝层（第 2、6、15 层），只看位置参数的钩子漏记了这 3 层（每次只记到 29 层），32 层平均值因此是错的。保留记录，不用于分析。
  - R46～R49：钩子改为 `with_kwargs=True` 并要求 32 层全部记到（否则判为失败）后重跑；R46 再次给出 64.125。

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

## 结果

全部来自 `code/output/EXP-0002/analysis/tables.md`。vanilla 取 R34（POPE）与 R05（MME）：与阶段 B 同为 sdpa、4 个 worker，答案与 R04、R02 逐条相同。区间为按样本 bootstrap 1000 次的 95% 区间（种子 0）；“相对 vanilla”为同一批样本上的配对 bootstrap。

### 主结果

| 配置 | 32 层平均视觉 token | POPE F1 均值 [95% 区间] | 相对 vanilla [95% 区间] | POPE yes 比例（三子集平均） | MME-P [95% 区间] | MME P+C | 秒/题（POPE / MME） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | 576 | 85.90 [85.09, 86.66] | — | 42.6% | 1507.63 [1440.63, 1569.25] | 1865.49 | 0.239 / 0.265 |
| FastV-192（K=2，keep 166） | 191.6（设计） | 75.76 [74.67, 76.79] | [−11.01, −9.27] | 32.7% | 1464.39 [1402.94, 1527.15] | 1820.10 | 0.164 / 0.170 |
| FastV-128（keep 98） | 127.9（设计） | 68.29 [67.03, 69.51] | [−18.72, −16.51] | 27.3% | 1363.21 [1290.40, 1432.26] | 1692.50 | 0.150 / 0.156 |
| FastV-64（keep 30） | 64.1（R46 实测） | 36.19 [34.59, 37.82] | [−51.39, −48.16] | 11.6% | 979.58 [918.39, 1037.85] | 1288.15 | 0.135 / 0.145 |
| 随机-192（种子 2021） | 191.6 | 80.39 [79.37, 81.30] | [−6.22, −4.84] | 37.5% | 1417.13 [1352.23, 1482.11] | 1780.35 | 0.163 / 0.169 |
| 随机-128（种子 2021） | 127.9 | 76.65 [75.59, 77.64] | [−10.08, −8.48] | 34.3% | 1355.03 [1287.01, 1426.12] | 1688.96 | 0.148 / 0.154 |
| 随机-64（3 个种子） | 64.1 | 60.14 ± 0.32（60.26 / 60.38 / 59.77） | 约 −25.8（三个种子的区间均在 [−27.5, −24.2] 内） | 23.5% | 1152.85 ± 26.28（1168.17 / 1167.88 / 1122.50） | 1460.6（均值） | 0.133 / 0.139 |
| VisionZip-192（162+30） | 192 | 85.49 [84.63, 86.26] | [−0.92, 0.10] | 43.3% | 1442.71 [1377.38, 1506.68] | 1783.78 | 0.158 / 0.186 |
| VisionZip-128（108+20） | 128 | 83.12 [82.22, 83.95] | [−3.43, −2.08] | 40.4% | 1432.45 [1366.92, 1493.25] | 1759.23 | 0.143 / 0.153 |
| VisionZip-64（54+10） | 64 | 77.03 [75.95, 78.05] | [−9.74, −7.99] | 34.7% | 1364.22 [1295.10, 1430.84] | 1696.01 | 0.128 / 0.146 |
| SparseVLM-192 | **206.5**（R47 实测） | 85.29 [84.47, 86.10] | [−1.15, −0.10] | 41.9% | 1447.98 [1379.67, 1510.01] | 1786.20 | 0.185 / 0.195 |
| SparseVLM-128 | **145.4**（R48 实测） | 85.07 [84.25, 85.87] | [−1.39, −0.31] | 41.5% | 1434.98 [1368.75, 1501.35] | 1762.12 | 0.166 / 0.173 |
| SparseVLM-64 | **82.25**（R49 实测） | 77.67 [76.64, 78.64] | [−9.12, −7.34] | 35.0% | 1279.59 [1212.12, 1342.06] | 1589.59 | 0.152 / 0.161 |

峰值显存（各运行）：vanilla 14.50～14.52 GiB；FastV/随机 13.93～14.11；VisionZip 14.35；SparseVLM 18.08～18.18。SparseVLM 多出的约 4 GiB 与它的模型构造一致：`builder.py` 第 114～121 行为每层新建一个 `flash_attn` 模块并复制 `self_attn` 的权重，原模块保留，32 层的 q/k/v/o 投影（fp16）合计 4 GiB。

### 与论文报告值的比较（计划 v1.1）

| 方法 | 预算 | POPE：本实验 / 论文 | MME P+C：本实验 / 论文 | 结论 |
| --- | --- | --- | --- | --- |
| VisionZip（CVPR Table 1） | 192 / 128 / 64 | 85.49 / 85.3；83.12 / 83.2；77.03 / 77.0 | 1783.78 / 1782.6；1759.23 / 1761.7；1696.01 / 1690 | POPE 三档都在 0.2 以内，复现通过 |
| SparseVLM（arXiv v4 Table 1） | 192 / 128 / 64 | 85.29 / 85.3；85.07 / 85.0；77.67 / 77.5 | 1786.20 / 1787；1762.12 / 1746；1589.59 / 1589 | POPE 三档都在 0.2 以内，复现通过（SDPA 代替 FlashAttention-2） |
| FastV（两篇论文中的 FastV 行，口径未写明） | 192 / 128 / 64 | 75.76 / 64.8；68.29 / 59.6 或 53.4；36.19 / 48.0 或 38.2 | 1820.10 / 1612 或 1605；1692.50 / 1490；1288.15 / 1256 或 1255 | 不设门槛。192、128 档本实验两项都明显高于论文行；64 档 POPE 低于两篇，MME 高于两篇 |

### H1：同预算下 FastV 的注意力选择 vs 随机选择

FastV − 随机的配对 bootstrap 95% 区间（同一批题目）：

| 预算 | POPE | MME-P |
| --- | --- | --- |
| 192（随机种子 2021） | [−5.51, −3.66] | [−20.73, 108.74] |
| 128（种子 2021） | [−9.43, −7.31] | [−63.02, 77.73] |
| 64（种子 2021 / 2022 / 2023） | [−26.05, −22.30] / [−26.03, −22.31] / [−25.44, −21.71] | [−264.03, −116.43] / [−259.79, −113.55] / [−220.90, −74.24] |

POPE 上三档都是随机更好（区间全在 0 以下）；MME 上 192、128 档区间跨 0（没有可分辨的差异），64 档随机更好。

### 方法间的补充配对比较（探索性，记录后计算）

tables.md 的最后一张表（`analyze_exp0002.py` 的 `METHOD_PAIRS`，在记录初稿之后加入）。用同一套 bootstrap（1000 次，种子 0），未做多重比较校正，只作描述。

| 比较（A − B） | POPE：差值 [95% 区间] | MME-P：差值 [95% 区间] |
| --- | --- | --- |
| FastV-192 − VisionZip-192 | −9.72 [−10.60, −8.90] | +21.68 [−30.21, 69.88] |
| FastV-192 − SparseVLM-192 | −9.53 [−10.33, −8.71] | +16.40 [−31.23, 63.47] |
| VisionZip-192 − SparseVLM-192 | +0.20 [−0.24, 0.63] | −5.28 [−41.68, 28.59] |
| VisionZip-128 − SparseVLM-128 | −1.95 [−2.51, −1.38] | −2.53 [−49.14, 43.36] |
| VisionZip-64 − SparseVLM-64 | −0.64 [−1.42, 0.20] | +84.64 [26.48, 146.78] |
| 随机-64（种子 2021）− VisionZip-64 | −16.76 [−18.09, −15.35] | −196.06 [−282.02, −111.92] |

### SparseVLM 每层实际看到的视觉 token（R46～R49，50 道 POPE）

| 配置 | 逐层视觉 token（值 × 层数，从第 0 层起） | 32 层平均 | 相对名义预算 |
| --- | --- | --- | --- |
| FastV-64（钩子核验） | 576×2，30×30 | 64.125 | 与设计一致 |
| SparseVLM-192 | 576×3，309×4，204×9，113×16 | 206.50 | +7.6% |
| SparseVLM-128 | 576×3，312×4，117×9，39×16 | 145.41 | +13.6% |
| SparseVLM-64 | 576×3，82×4，32×9，18×16 | 82.25 | +28.5% |

- 50 次预填充的逐层视觉 token 数完全相同（每个配置只有 1 种逐层分布；题目文本长度 61～63 个 token）。
- 与预算表的差别有两个来源，都已在代码中核实（`source/SparseVLMs/llava/model/language_model/modelling_sparse_llama.py` 第 265～313 行；预算表在 `score.py` 第 7～9 行）：
  - 剪枝层本身用剪枝前的 token 计算，剪枝结果从下一层起生效。所以 576 持续 3 层（第 0～2 层），而不是 EXP-0001 推算时假设的 2 层。
  - 每次剪枝后追加回收的聚类 token：被删 token 数为 d 时，追加 int((int(0.3d)+1)/10)+1 个。例如 192 档第 2 层删 276 个，追加 9 个，得 300+9=309。三档九个分段都与这个公式完全吻合，所以回收数只由预算决定，与图像无关。
- EXP-0001 按预算表推算的层平均（188.2/123.9/61.7）因此偏低。实际值比名义预算多，而不是少。

### 局部指标（摘要；完整表见 tables.md）

- **POPE 的 yes 比例与召回率**：
  - vanilla 的 yes 比例为 42.6%（真值 50%）。所有剪枝方法都更少答 yes：FastV 最严重（32.7% → 27.3% → 11.6%），随机次之（37.5% → 34.3% → 23.5%）。VisionZip、SparseVLM 在 192/128 档与 vanilla 相差不超过 2.2 个百分点，64 档降到 34.7%、35.0%。
  - F1 的下降主要来自召回：FastV-64 的召回率为 22.3%（vanilla 79.1%），精确率仍有 93.8%～98.5%。
  - FastV 与随机选择改变的答案以“对→错”为主（FastV-64：2616 vs 254；随机-64 种子 2021：1690 vs 237）。VisionZip-192、SparseVLM-192 两者接近（239 vs 195、236 vs 194）。
- **MME 子任务（相对 vanilla）**：
  - FastV-64 在 existence −90、count −95、color −78.3、position −73.3 上大幅下降。
  - 随机选择在 celebrity 上的下降（64 档 −57.9～−66.2）远大于同预算的 FastV（−32.4）、VisionZip（−22.1）和 SparseVLM（−25.0）；192、128 档 FastV 的 celebrity 不降（+0.6、+2.4），随机为 −7.4、−18.2。
  - OCR 上，随机选择三档都下降 17.5～27.5，FastV 为 0 / −2.5 / −15。
  - VisionZip-64 的最大降幅在 count −41.7、color −25；SparseVLM-64 在 color −45、position −33.3、landmark −27.8。
  - scene 是唯一几乎不受影响的子任务：除 FastV-64（−30）外，所有配置都略有上升（+0.2～+6.8）。
  - 样本量提醒：OCR 与 Cognition 中的 numerical_calculation、text_translation、code_reasoning 各只有 20 张图（40 道题），一道题就能移动 2.5～7.5 分；existence、count、position、color 各 30 张图（60 道题）。这些子任务的差异不单独解读。
- **速度**：
  - 所有剪枝方法每题比 vanilla 快 1.3～1.9 倍（POPE：0.239 → 0.128～0.185 秒）。
  - 同档预算下 SparseVLM 最慢。可能的原因（未单独测量）：逐层打分的开销、实际 token 更多、SDPA 代替 FlashAttention-2。

## 观察与解释

| 类型 | 观察（E） | 当前解释（推断） | 待复验 / 替代解释 | 证据位置 |
| --- | --- | --- | --- | --- |
| 主要观察 | VisionZip、SparseVLM 的 POPE 三档都与论文相差 ≤0.2；vanilla POPE 与官方一致 | 两种方法可在本环境复现，阶段 B 的比较有可靠基线 | MME 上 SparseVLM-128 比论文高 16.1、VisionZip-64 高 6.0，来源未知 | R12～R15、R17、R35～R41；tables.md |
| 主要观察 | 同名义预算下，FastV（主 baseline）在 POPE 上大幅落后于 VisionZip、SparseVLM（192 档 −9.7、−9.5），也落后于随机选择 | FastV 只按第 K−1 层最后一个 token 的注意力挑 token，覆盖不到物体所在区域（待验证） | 本实验的 FastV 设置（K=2、按层平均换算 keep）与原论文不同 | R06～R11、R24～R33；补充配对比较 |
| 意外现象 | 随机选择在 POPE 三档都显著优于 FastV；64 档优势约 24 分，远大于种子间波动（标准差 0.32） | 注意力高的 token 在空间上集中，丢掉了物体所在区域（待验证） | 注意力偏向图像下部或特定位置（位置偏差，S-04 的说法）；需要看被选 token 的空间分布 | R29、R31、R33 vs R11 |
| 意外现象 | 剪枝越狠，模型越倾向答 “no”：FastV-64 的 yes 比例只有 11.6%，召回率 22.3%，精确率仍在 93.8% 以上 | 关键物体的 token 被删后，模型默认“不存在” | 也可能是视觉 token 太少时语言先验占主导；需要比较删掉物体区域与删掉背景区域 | tables.md 子集表；summary.json 的 splits |
| 意外现象 | 方法排序随 benchmark 改变：192 档 FastV 的 POPE 比 VisionZip 低 9.7 分，MME-P 却高 21.7 分（区间 [−30.2, 69.9]，跨 0） | POPE 考物体存在，对删掉的局部区域敏感；MME 的多数子任务更依赖整体语义 | MME 的区间很宽，192 档 MME 上的差异在噪声内，不能说 FastV 在 MME 上更好 | R06、R07 vs R12、R13；补充配对比较 |
| 意外现象 | 随机选择在 celebrity、OCR 上损失远大于 FastV，而 FastV 在 existence、count、color、position 上损失更大 | 随机均匀覆盖保住了物体存在信息，却把人脸、文字这类局部细节切碎；注意力选择则相反 | 需要看这两类题中被保留的 token 位置 | tables.md MME 子任务表 |
| 意外现象 | SparseVLM 每层实际看到的视觉 token 比名义预算多 7.6%/13.6%/28.5% | 剪枝层仍处理剪枝前的 token，加上回收 token；两项都已在代码中核实，公式与实测完全吻合 | 按实际 token 数对齐预算后，SparseVLM 与 VisionZip 的比较可能改变 | R46～R49；modelling_sparse_llama.py 第 265～313 行 |
| 技术 / 质量 | SparseVLM 的每层注意力都走 FlashAttention-2（此前判断错误）；Windows 上用 SDPA 替代后，POPE 与论文一致 | 替代只改变内核，不改变算法 | 与官方 flash-attn 的逐答案差异未测 | R18～R23、R36～R41；sparsevlm_sdpa.py |
| 技术 / 质量 | 4 个 DataLoader worker 与 0 个 worker 的答案逐条相同（MME 2374 题、POPE 8910 题） | 只影响速度 | — | R05 vs R02；R34 vs R04 |

## 偏差与限制

- **与计划的差异**：
  - 阶段 A 的 MME-P 门槛按字面未通过，仍进入阶段 B（理由见“阶段 A”）。
  - 随机选择按 40 分钟规则减少为 5 个配置。
  - SparseVLM 用 SDPA 代替 FlashAttention-2。
  - 增加了计划外的速度参照（R34）、逐层 token 诊断（R42～R49）和方法间的补充配对比较。
- **失败与排除**：失败 7 次（R01、R16、R18～R23），均保留目录并已用新 ID 重跑；R43～R45 的诊断数据有缺陷，不用于分析。
- **可能影响结果的因素**：
  - 注意力内核（sdpa vs eager）本身就能使 MME-P 移动约 7 分。
  - SparseVLM 每层实际 token 数多于名义预算，按名义预算并列比较时它占了便宜。
  - 本实验的 FastV 设置与原论文不同；结论只适用于“K=2、按层平均换算 keep”的 FastV。
  - 随机选择的 192、128 档只有一个种子。随机生成器在整次运行中按题目顺序连续抽取，所以同一道正样本题在三个子集里抽到的 token 不同（三个子集的召回率因此不同，例如种子 2021：43.5%/44.5%/44.7%）。
  - 速度测量受 Python 开销影响，只适合同一环境下的相对比较。
- **伦理与合规**：只用公开 benchmark；数据和权重按各自许可在本机使用，未再分发。

## 结论边界与下一步

- **当前证据支持**：
  - 本环境的评测管线可复现官方 vanilla POPE 和两种方法论文的 POPE。
  - 在“32 层平均预算相等”的口径下，FastV 的注意力选择不如随机选择（POPE 三档、MME 64 档），H1 得到支持。
  - SparseVLM 的名义预算不等于它实际处理的 token 数，差值可由代码中的公式精确算出。
- **当前证据不能支持**：
  - 任何方法在 POPE、MME 以外的 benchmark 上的结论。
  - 注意力选择差的机制（位置偏差、空间覆盖等都还只是假设）。
  - 用原论文 FastV 设置时的结论。
- **下一项最低成本检查**：在 50 道 POPE 上加一个诊断开关，记录 FastV 与随机选择保留的图像 patch 索引（24×24 网格），比较两者的空间分布，检验“注意力集中在少数位置”的假设。只需 2 次约 1 分钟的诊断运行，不需要完整评测。
- **后续路由**：squeeze-experiment（EXP-0002_squeeze）→ deep-read Mode C（合并两份视觉 Mode C，回流 P/F/M/E）。
- **下一 skill**：squeeze-experiment。

## 交接

原始证据可从 `code/output/EXP-0002/` 与备份恢复（每个运行目录有 SHA256SUMS，备份已逐文件核对）。runs.tsv 覆盖全部 49 次运行；分析表由 `code/token_pruning/analyze_exp0002.py` 从运行目录重新生成。交给 squeeze-experiment。
