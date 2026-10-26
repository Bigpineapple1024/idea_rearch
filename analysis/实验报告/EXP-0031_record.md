# EXPERIMENT_RECORD：一层全看、再按编码器排序的确认实验（IDEA-0007）

状态：计划 v1（2026-10-26 11 点 20 分），在 EXP-0030 的任何运行与任何结果之前写定。写计划时 EXP-0029 正在运行，也没有看过它的任何结果。只在 EXP-0030 的判读为“有希望（进入确认）”时运行。

- 实验 ID：EXP-0031。
- IDEA ID：[IDEA-0007](../IDEA-0007_一层全看再按编码器排序.md)；DEC-0049、DEC-0051。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。
- 来由：EXP-0030 计划 v1 写定“判读‘有希望’时写确认计划并提交，然后才运行”，并且确认要用新抽的数据（VQAv2 test3、GQA testdev 上 K = 1 路由的结果已经看过）。为了不受 EXP-0030 结果的影响，按 IDEA-0006 的做法（EXP-0029），确认计划与确认数据在 EXP-0030 出结果之前准备好。所选配置由 EXP-0030 写定的选择规则决定，分析脚本从 EXP-0030 已封存的分析结果中读出，不手工填写。

## 计划 v1（运行前写定）

### 问题

1. **Q1（主）**：在新抽的数据上，所选配置是否高于 VisionZip-64？
2. **Q2**：是否高于 IDEA-0006 的所选（VisionZip-64 + 每题覆盖 16）？EXP-0030 的判读要求开发集上高于它，这里在新数据上再检验。
3. **Q3**：同样的路由与覆盖下，按视觉编码器的 CLS 注意力排序，是否好于 FastV 按 LLM 注意力的排序？这是 IDEA-0007 与 FastV 一族的区别。对照是所选的“对应的 FastV 配置”：
   - `k1cls`、`k1qcls` 对 K = 1 路由（`fastv64k1_qroute_kl`，与 EXP-0024-R06、R07 相同）；
   - `k1qcls_c16` 对 K = 1 路由 + 覆盖 16（`fastv64k1_qroute_c16_kl`，与 EXP-0026-R03、R04 相同）。
   - `k1cls` 的被路由题也换成了 CLS 排序，所以它这一对还包含路由的差别，结论写成“好于 K = 1 路由”。
4. **Q4（描述）**：在看细节的题（TextVQA，不给 OCR）上，所选比 VisionZip-64、比 IDEA-0006 的所选高多少？

### 写计划时已知与未知

- **已知**：
  - 开发集上 IDEA-0003～IDEA-0006 的全部结果，以及 EXP-0028 的记录内深挖；
  - VQAv2 test3、GQA testdev 上 FastV 一族与 VisionZip-64 的结果（EXP-0022、EXP-0027）；
  - TextVQA（不给 OCR）上 K = 2 路由与不剪枝的结果（EXP-0025）。
- **未知**：
  - EXP-0030 的任何结果，包括会选中哪个候选、判读是什么；
  - EXP-0029 的任何结果，包括 TextVQA 上 VisionZip-64 与每题覆盖 16 的结果（本实验复用这两次运行）；
  - 新数据上任何配置的结果。

### 条件

- 只有 EXP-0030 的判读为“有希望（进入确认）”时运行。其余判读（不比 IDEA-0006 好、不像、中间情况）按 EXP-0030 计划交用户决定，本计划不运行。
- **所选** = EXP-0030 分析的 `criteria.selected`：`k1cls`、`k1qcls` 或 `k1qcls_c16`，配置分别为 `fastv64k1_cls_kl`、`fastv64k1_qroute_cls_kl`、`fastv64k1_qroute_cls_c16_kl`（与 EXP-0030 相同；`_kl` 只加记录，不改变计算）。

### 数据（新抽）

`make_confirm_sets.py` 在 2026-10-26 上午、EXP-0030 的任何运行之前抽好，每个只抽一次：

- **VQAv2 test4**：val2014 上另外 1000 张图的全部 5333 题，种子 4。
  - 图像池同 make_vqav2_sets.py（去掉 POPE 与 coco_dev 的图像），再去掉 test、test2、dev2、test3 的图像（35704 张中抽）；
  - 抽之前先按原来的种子重抽这四个子集，与现有文件逐一核对；断言与 dev 也不重叠。
- **GQA val400b**：GQA val-balanced 中另外 400 张图的全部 5165 题，种子 1。
  - 从去掉 val400 图像的 val-balanced 图像中抽（先按种子 0 重抽 val400 并与现有文件核对）；断言与 val400、testdev 都不重叠；
  - 图像从官方 images.zip 解出。
- **校验和**：`data/llava_eval/vqav2_val/SHA256SUMS_test4`、`data/llava_eval/gqa/SHA256SUMS_val400b`、`data/gqa/images_val400b_SHA256SUMS`（400 张图），副本与抽取日志 make.log 在 `D:\6CJNM007\tmp\sets4\`。抽之前核对了 VQAv2 两个 JSON 与 GQA questions1.2.zip 的 sha256。
- **TextVQA** v0.5.1 val（5000 题，不给 OCR）：只作描述。IDEA-0004、IDEA-0006 的确认用过它（EXP-0025、EXP-0029）。

### 运行（9 次，按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 | 作用 |
| --- | --- | --- | --- |
| R01 | 所选 | vqav2_test4 | Q1–Q3 |
| R02 | 所选 | gqa_val400b | Q1–Q3 |
| R03 | `visionzip64` | vqav2_test4 | Q1 的对照 |
| R04 | `visionzip64` | gqa_val400b | 同上 |
| R05 | `visionzip64_c16all` | vqav2_test4 | Q2 的对照：IDEA-0006 的所选 |
| R06 | `visionzip64_c16all` | gqa_val400b | 同上 |
| R07 | 对应的 FastV 配置 | vqav2_test4 | Q3 的对照 |
| R08 | 对应的 FastV 配置 | gqa_val400b | 同上 |
| R09 | 所选 | textvqa_noocr | Q4 |

- **复用**（TextVQA，配置与代码不变，从已封存的文件夹读）：VisionZip-64 EXP-0029-R03；每题覆盖 16 EXP-0029-R04；K = 2 路由 EXP-0025-R01；不剪枝 EXP-0025-R07。
- 每次运行设 PYTHONFAULTHANDLER=1（不改变计算）。失败的运行保留，用新编号（R10 起）重跑，先记录再重跑。

### 判据（预注册）

| 编号 | 判据 |
| --- | --- |
| Q1（主） | 所选 − VisionZip-64，在 VQAv2 test4、GQA val400b 上分别计全部题准确率的配对差，按图像重抽 1000 次（种子 0）。两个 95% 区间都在 0 以上：成立；只有一个：部分成立；都不：不成立 |
| Q2 | 所选 − 每题覆盖 16，同样的读法 |
| Q3 | 所选 − 对应的 FastV 配置，同样的读法 |
| Q4（描述） | TextVQA 上所选 − VisionZip-64、所选 − 每题覆盖 16；区间在 0 以上才写“高于” |

- **提升大**（只报告，不改判读）：Q1 的两集平均 ≥ +2.0。
- **判读**：
  - Q1 成立或部分成立：写成“IDEA-0007 在新数据上高于 VisionZip-64”（部分成立时写明在哪个集上）。Q2 成立才写“好于 IDEA-0006 的所选”；Q3 成立才写“K = 1 时按编码器排序好于 FastV 的注意力排序”（`k1cls` 写“好于 K = 1 路由”）；部分成立写明在哪个集上。
  - Q1 不成立：IDEA-0007 不作为方法的贡献；记录，交用户决定。
- **预测**（推断，不是判据）：确认集上的增益一般不大于开发集。IDEA-0005、IDEA-0006 的覆盖增益在确认集上与开发集相近或更小。

### 同时报告（预先写定，不是判据）

- 每题覆盖 16 − VisionZip-64；对应的 FastV 配置 − VisionZip-64；每题覆盖 16 − 对应的 FastV 配置。
- TextVQA：所选 − K = 2 路由；不剪枝 − 所选；不剪枝 − VisionZip-64。
- 按题分组：VQAv2 的答案类型与颜色题；GQA 的结构类型、语义类型；被路由与不路由的题。
- 覆盖记录、每题秒数、峰值显存。

### 分析

- `analyze_0031.py` v1：从 `code/output/EXP-0030/analysis/0030.json` 读所选与判读，判读不是“有希望”就停止。
- 三种选择各用替身数据空跑过（`D:\6CJNM007\tmp\exp0031\dryrun_0031.py`；都 exit 0，核对 66、73、84 项都 100%；日志 `dry_k1cls.log`、`dry_k1qcls.log`、`dry_k1qcls_c16.log`）。替身数据里故意让 0.5% 的题的分数最高的 53 个与 VisionZip 的不同、0.5% 的框与参照不同，门槛照常通过。第一次空跑之后只给 score_pope_mme.py 的说明加了一行用法，又重跑一次，结果相同（第一次的日志在 `dry_v1a\`）。
- 计分的冒烟测试：tmp 中的 score_pope_mme.py 从命令行给两个新数据集的假答案（全部答 Yes）计分，5333、5165 题都计上（`D:\6CJNM007\tmp\exp0031\precheck.sh`，日志 `precheck.log`）。
- sha256 记在 `D:\6CJNM007\tmp\exp0031\MANIFEST`（analyze_0031.py 为 c39adc3b…，make_confirm_sets.py 为 99f33a05…）。

### 实现核对（不满足就停止排查，门槛 ≥ 99%）

- **每个运行**：每题一个答案；重算的准确率与计分文件、登记的准确率一致；runs.tsv 中的配置与计划相同（包括复用的 EXP-0029-R03、R04）。
- **每题覆盖 16**（R05、R06 与复用的 EXP-0029-R04）：analyze_0028 的同一套记录核对；对应的 FastV 配置有框时（K = 1 路由 + 覆盖 16），每题覆盖 16 的框与它相同（不路由的题，至少 99%）。
- **所选**（R01、R02、R09），与 EXP-0030 相同的 CLS 运行核对：
  - 每题一条 CLS 记录、一份保留列表，顺序与题目文件相同；按 CLS 排序的题恰为计划的题；保留记录里的路由决定等于题目文字上的规则；
  - 每题保留 47 个、不重复、没有整块落在填充里的；
  - 分数最高的 53 个等于 VisionZip 自己的主导 token（同一个集上每题覆盖 16 的记录）的题至少 99%；
  - 按 CLS 排序的题，保留的就是 CLS 排序自己的选择（`k1cls`、`k1qcls`）；
  - 新数据上，被路由的题保留的 token 与答案都与对应的 FastV 配置相同（`k1qcls`、`k1qcls_c16`，各至少 99%）；
  - `k1qcls_c16` 的覆盖：至少 99% 的不路由题被覆盖、被路由的题都不覆盖；框与参照相同至少 99%（新数据上参照为对应的 FastV 配置，TextVQA 上为每题覆盖 16）；框内个数 = CLS 排序自己的框内个数 + 16（上限 47 与框内可选的 patch 数），框内包含原来在框内的，框外是原来在框外的子集；没被覆盖的不路由题保留的就是 CLS 排序自己的选择。

### 代码（分支 code/token-pruning，在 EXP-0030 分析封存、判读为“有希望”之后、GPU 空闲时提交）

- `exp0002.py`：BENCHES 加 `vqav2_test4`、`gqa_val400b`（基于 IDEA-0007 的代码提交中的版本）。
- `score_pope_mme.py`：计分加 `vqav2_test4`（走原有的 VQAv2 分支）、`gqa_val400b`（与 val400 相同的 GQA 计分）。
- `make_confirm_sets.py`、`analyze_0031.py`（新）。
- 模型代码与配置不变。四个文件的 sha256 记在 `D:\6CJNM007\tmp\exp0031\MANIFEST`。

### 排队

排队脚本 `D:\6CJNM007\tmp\chain_0031.sh`（2026-10-26 补注：用户要求每开启下一步之前完整阅读一遍仓库（DEC-0053），排队改用 `D:\6CJNM007\tmp\chain_0031b.sh`：与原脚本逐字相同，只在第 2 步之后多等一道门 `D:\6CJNM007\tmp\read_repo\gate\EXP-0031.done`，读完、修好、提交之后才写）：

1. 等 `D:\6CJNM007\tmp\run_vision_0030.log` 写出 “EXP-0030 analysis sealed”；
2. 从 0030.json 读判读：不是“有希望”就停下，不运行本实验；
3. 确认 GPU 空闲、在 code/token-pruning、HEAD 是 IDEA-0007 的代码提交（父提交 7ba3657，提交说明与 `msg_code62.txt` 相同）、工作区干净；核对 HEAD 中 exp0002.py、score_pope_mme.py 与本计划改动的底本相同，核对 MANIFEST 与三份数据校验和，确认本计划已在 main 上；
4. 复制、提交、推送；在仓库里核对配置与数据集能读到，用全部答“yes”的假答案冒烟测试两个新数据集的计分（写在 tmp，不进仓库）；
5. 开始 R01–R09；全部 completed、GPU 空闲后，用已提交的 analyze_0031.py 在 CPU 上分析并封存。

### 预算

- 9 次，约 2.4 GPU 小时。IDEA-0007 共 15 次（EXP-0030 6 次 + 本实验 9 次），上限 30 次。
