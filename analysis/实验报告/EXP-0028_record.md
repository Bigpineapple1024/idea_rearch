# EXPERIMENT_RECORD：VisionZip 上的定位后覆盖，开发集运行（IDEA-0006）

状态：计划 v1（2026-10-26 4 点）与 v1.1（4 点半），都在 EXP-0028 的任何运行之前写定；写计划时 EXP-0027 正在运行，没有看过 EXP-0027 的任何结果。v1.1 只在 R10 之后加 2 次运行和一条归因规则，见“计划 v1.1”。运行中：R01、R02 completed；R03–R12 因代码错误失败，修正后按计划 v1.2 以 R13–R22 重跑（见文末，DEC-0047）。

- 实验 ID：EXP-0028。
- IDEA ID：[IDEA-0006](../IDEA-0006_强底座上的定位后覆盖.md)；DEC-0043、DEC-0044。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。
- 来由：用户选择先确认 IDEA-0005、再把区域覆盖移到更强的底座上（DEC-0043）。确认实验 EXP-0027 结束后接着运行本实验，不等 EXP-0027 的判读。

## 计划 v1（运行前写定）

### 问题

1. 在 VisionZip-64 上（同为 64 个 token），把一部分主导名额挪进题目定位的框里，准确率是否提高？
2. 增益来自“找对了区域”，还是只来自“把名额集中到一块”？用边长相同、放在图像中央的框作对照。
3. 挪得更多（覆盖 32）、或者每道题都覆盖（包括 yes/no 与 how many 题），会怎样？

### 写计划时已知与未知

- **已知**：
  - FastV 一族上的全部开发集结果（EXP-0024、EXP-0026）；
  - VisionZip-64 在 VQAv2 test1 70.57、test2 70.70，GQA testdev 55.18（EXP-0018-R05、EXP-0020-R08、EXP-0022-R07）。
- **未知**：
  - VisionZip-64 在 VQAv2 dev2、GQA val400 上的结果（没有运行过）；
  - VisionZip 上任何覆盖的结果；
  - EXP-0027 的任何结果。
- **CPU 核对**（小的随机 LLaVA，`check_vzcover.py` 40/40，日志 `D:\6CJNM007\tmp\exp0028\check_vzcover2.log`）：
  - 不覆盖时与 visionzip.visionzip 逐值相同；复制的视觉塔前向在没有题目查询时与 VisionZip 的前向逐值相同；
  - 被路由的题与 VisionZip 的 token、答案相同；
  - 不路由的题：框等于在普通 LLaVA 的图像输入 embedding 上按题目查询定位的框；主导 token 按覆盖规则选；LLM 收到的 64 个 token 等于投影层作用在“CLS + 这些 patch + 其余 patch 按 VisionZip 合并”上（核对脚本独立写了一遍合并）；
  - 居中框、每题覆盖、覆盖 32、可见权重为 0 的题；参数与配置。
  - 第一次运行 39/40：失败的一项是核对本身把 tmp 目录下算出的仓库路径当成了仓库（配置在仓库里运行时路径正确），改了这项核对的写法，并给记录加上框内可选的 patch 数（region）后重跑。
  - FastV 一族的回归核对 check_zoom.py 77/77（新代码不改 FastV 的路径）。

### 数据

与 IDEA-0004、IDEA-0005 相同的开发集：VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图）。VQAv2 test3、GQA testdev 不用于开发。

### 运行（10 次，按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 | 作用 |
| --- | --- | --- | --- |
| R01、R02 | `visionzip64` | dev2、val400 | 底座：VisionZip-64（CLS + 53 个主导 + 10 个上下文） |
| R03、R04 | `visionzip64_c16` | dev2、val400 | 覆盖 16：不路由的题上，框内 = VisionZip 的框内个数 + 16 |
| R05、R06 | `visionzip64_c16ctr` | dev2、val400 | 居中框覆盖 16：定位的对照 |
| R07、R08 | `visionzip64_c32` | dev2、val400 | 覆盖 32：挪得更多 |
| R09、R10 | `visionzip64_c16all` | dev2、val400 | 每题都覆盖 16（包括被路由的题） |

- 覆盖只在不路由的题上做（`c16all` 除外）；被路由的题与 VisionZip-64 逐题相同。可见权重为 0 的题不覆盖。
- 居中框：边长与定位框相同，位置在图像内容带的中央（与 IDEA-0004、IDEA-0005 的居中框同一规则）。
- 每次运行设 PYTHONFAULTHANDLER=1（不改变计算）。失败的运行保留，用新编号（R11 起）重跑，先记录再重跑。

### 选择规则（写定）

- **候选**：覆盖 16、覆盖 32、每题覆盖 16。
- **主指标**：Δ = 候选 − VisionZip-64，即全部题准确率的配对差，两个开发集分别计，取平均 Δ̄。
- **选法**：选 Δ̄ 最大的候选。与最大值相差不到 0.1 的，按此顺序取第一个：覆盖 16、覆盖 32、每题覆盖 16。

### 判读（写定）

- **覆盖的增益** Δc = 所选 − VisionZip-64，两个开发集分别计。
- **定位的增益** Δl = 覆盖 16 − 居中框覆盖 16，取两集平均。所选不是覆盖 16 时仍用这一对，并写明这一限制。
- **有希望（进入确认）**：Δc 在两个开发集上都 > 0，且 Δl > 0。
- **集中而非定位**：Δc 都 > 0，但 Δl ≤ 0。记录后交用户决定是否继续。
- **不像**：三个候选的 Δ̄ 都 ≤ 0。定位后覆盖在 VisionZip 上不成立，记录，回到第 ⑨ 步，等用户决定下一个方向。
- **中间情况**：其余情形，交用户决定。
- **证据强度**（只报告，不改判读）：Δc 的区间在 0 以上的开发集个数（0、1、2）。

### 同时报告（预先写定，不是判据）

- 居中框覆盖 16 − VisionZip-64（集中本身的作用）；覆盖 32 − 覆盖 16；每题覆盖 16 − 覆盖 16。
- 与 FastV 一族比较（描述）：VisionZip-64 − K = 2 路由（EXP-0024-R02、R03）；VisionZip-64、各候选 − FastV K = 1 + 覆盖 16（EXP-0026-R03、R04，IDEA-0005 所选）。
- 按题分组：VQAv2 的答案类型与颜色题；GQA 的结构类型、语义类型；被路由与不路由的题。
- 框内保留：VisionZip 自己的平均框内个数、覆盖后的平均框内个数、全部 53 个都在框里的比例、平均改动的 token 数、框平均占网格的比例。
- 每题秒数、峰值显存。

### 实现核对（不满足就停止排查，门槛 ≥ 99%）

- **每个运行**：每题一个答案；重算的准确率与计分文件、登记的准确率一致。
- **覆盖运行**：
  - 每题一条覆盖记录，顺序与题目文件相同；
  - 路由决定等于题目文字上的规则（`c16all` 没有题被当作路由跳过），不路由的题至少 99% 被覆盖；
  - 主导 token 恒为 53 个、不重复；框内个数等于记录的个数，等于“VisionZip 的框内个数 + n”（上限 53 与框内可选的 patch 数）；改动个数与记录一致；框是正方形，边长与记录一致；
  - 覆盖 16、居中框、覆盖 32 在被路由的题上的答案与 VisionZip-64 相同；
  - 覆盖 16 的框与 EXP-0026 中 FastV K = 1 覆盖 16 的框逐题相同（定位只用输入 embedding，两种底座上应当相同），至少 99%；
  - 覆盖 32、每题覆盖 16 的框与覆盖 16 的框相同（两者都覆盖的题）；
  - 居中框的边长与覆盖 16 相同，位置等于按图像大小重算的居中框（分析脚本独立写了一遍居中的规则）。
- `analyze_0028.py` 在运行前用替身数据空跑过，只检查代码路径（`D:\6CJNM007\tmp\exp0028\dryrun_0028.py`；替身数据里故意让 0.5% 的题与 EXP-0026 的框不同，门槛照常通过）。

### 代码（分支 code/token-pruning，在 EXP-0027 的最后一个运行之后、GPU 空闲时提交）

- `visionzip_cover.py`（新）：题目查询与定位在预填充前算好；视觉塔的前向复制自 VisionZip，只在有题目查询时改主导 token 的选择。
- `run_eval.py`：`--vz-cover`、`--vz-cover-on`、`--vz-cover-centre`、`--vz-cover-box`、`--vz-visibility`、`--vz-log`；IDF 表照 FastV 一族的做法按评测的题目文件计算并写出。
- `exp0002.py`：`visionzip64_c16`、`visionzip64_c16ctr`、`visionzip64_c32`、`visionzip64_c16all`；覆盖配置自动加 `--vz-log`。
- `check_vzcover.py`（新）、`analyze_0028.py`（新）。
- 提交前在 CPU 上重跑 check_vzcover.py 与 check_zoom.py，都通过才提交；排队脚本确认本计划已在 main 上、工作区干净、GPU 空闲后才开始 R01。

### 预算

- 10 次运行，约 2.5 GPU 小时（VisionZip-64 每题约 0.16 秒）。
- IDEA-0006 单独计：试验阶段 30 次运行或 20 GPU 小时。

### 确认（之后，EXP-0029）

判读“有希望”时写确认计划并提交，然后才运行：在 VQAv2 test3 与 GQA testdev 上比较所选配置与 VisionZip-64（EXP-0027-R04、EXP-0022-R07 可复用）；判据在确认计划中写定。

## 计划 v1.1（2026-10-26 4 点半，EXP-0028 的任何运行之前）

### 为什么改

v1 提交之后做了查新（Claude 自己检索网页与摘要，没有开子代理，没有下载；笔记 `D:\6CJNM007\tmp\exp0028\novelty_check_idea0006.md`）。最接近的是 VisionTrim（arXiv 2601.22674，ICLR 2026）：在 VisionZip 式的主导 token 选择之上，按逐 token 的文本相似度挑出聚类中心，把其余 token 合并进去，报告 64 个 token 时 GQA 58.8、VQAv2 76.8（同表 VisionZip 55.1、72.4）。CDPruner（NeurIPS 2025）也在 LLM 之前按指令相关性选 token。所以审稿人最先会问：框有没有用，还是只要留下与题目最相似的 token 就够了。FastV 一族上这个对照已做过（EXP-0026 v1.1，F53：按区域多留好于按逐 token 相似度多留），VisionZip 上 v1 没有。

### 加的运行（在 R10 之后）

| 运行 | 配置 | 数据 | 作用 |
| --- | --- | --- | --- |
| R11、R12 | `visionzip64_t16` | dev2、val400 | 不围框：不路由的题上，与题目最相似的 16 个非填充 patch 必留，其余 37 个主导名额按 CLS 注意力在它们之外取 |

- 新选项 `--vz-cover-top 16`：覆盖的区域换成与题目查询最相似的 16 个 patch，覆盖的规则不变（框内名额 = min(VisionZip 的区域内个数 + 16, 16)，所以这 16 个全部保留）。框照常定位并记录，用于核对。
- 代码与核对：check_vzcover.py 加不围框的离线规则核对与参数、配置核对，45/45（`D:\6CJNM007\tmp\exp0028\check_vzcover3.log`）；run_eval.py 改了参数，check_zoom.py 重跑 77/77（`check_zoom3.log`）。analyze_0028.py v1.1 加入 R11、R12 的核对与下面的归因，用替身数据重新空跑。代码随 v1 的代码一起，在 EXP-0027 之后、R01 之前提交。

### 归因规则（写定，不改选择规则与判读）

- **区域还是逐 token 相关性**：Δr = 覆盖 16 − 不围框，取两集平均。
  - Δr > 0：按区域多留比按逐 token 相似度多留更好，写成“区域的作用”；
  - Δr ≤ 0：增益来自与题目相似的 token 本身，不需要框，论文按此写。
- **同时报告**：不围框 − VisionZip-64。

### 实现核对（加）

- 不围框的运行：每条被覆盖的记录里，与题目最相似的 16 个 patch 都在主导 token 里，区域内个数恰为 16；被路由的题与 VisionZip-64 的答案相同；记录的框与覆盖 16 的框相同。

### 预算

- EXP-0028 共 12 次（v1 的 10 次 + 2 次），约 3 GPU 小时；IDEA-0006 的上限 30 次不变。

## 运行中的记录与偏差（2026-10-26）

- **开始**：EXP-0027 结束后，排队脚本 `D:\6CJNM007\tmp\chain_0028.sh` 核对 `D:\6CJNM007\tmp\exp0028\MANIFEST_v11` 的 sha256，提交 analyze_0027.py（6dfc3f5）与 IDEA-0006 的代码（15e75ed），确认计划在 main 上，05:52:00 开始 R01。每次运行都设了 PYTHONFAULTHANDLER=1（不改变计算）。
- **R01、R02**（VisionZip-64，不覆盖）：completed（05:52–06:18）。这两个配置不加载 visionzip_cover，不受下面的错误影响。
- **R03–R12 全部失败（06:19–06:25）**：
  - 每次在开始后约 31 秒退出，eval_exit 1，stdout.log 里是同一个错误：`RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half`，位置在 visionzip_cover._cover 调用投影层的一行。
  - VQAv2 dev2 的 5 次与 R10（每题覆盖，GQA）在第一题出错，answers.jsonl 为空；R04、R06、R08、R12（GQA val400）的第一题被路由（不覆盖），写出 1 个答案后在第二题出错。这 4 个答案没有看，也没有计分。
  - 这是代码错误，与以前的偶发退出（exit 127、没有 traceback）不同。
- **原因**：
  - LLaVA 按 `device_map="auto"` 加载时，视觉塔保持 float32（llava/model/builder.py 只在其他 device_map 下才转成 float16），投影层与 LLM 是 float16。
  - 普通 LLaVA 在投影之前把第 −2 层的特征转成图像的 dtype（CLIPVisionTower.forward 的 `.to(images.dtype)`），VisionZip 输出前也这样转；`_cover` 却把 float32 的第 −2 层直接送进了 float16 的投影层。
  - check_vzcover.py 的小模型全是 float32，所以没有发现。
- **修正**：`_cover` 多一个参数，取图像的 dtype，在投影之前转换，与普通 LLaVA 的算法相同。其他代码不变。
- **核对（都在重跑之前）**：
  - check_vzcover.py 用修正后的代码重跑 45/45（`D:\6CJNM007\tmp\exp0028b\check_vzcover4.log`）。
  - 新的 check_vzcover_dtype.py 11/11（`check_vzcover_dtype.log`）：小模型按真实的 dtype 布局设置（视觉塔 float32；投影层、LLM、图像用 bfloat16 代替 float16，因为 CPU 上没有 float16 的矩阵乘）。已提交的版本在这个布局上报与 GPU 相同的错，修正版能运行；被路由的题与 VisionZip 相同；框等于按普通 LLaVA 的做法算出的图像输入 embedding 上的框；主导 token 按规则；LLM 收到的 CLS 与主导 token 逐值相同，上下文 token 在 bfloat16 的舍入内相同。
  - GPU 冒烟测试（不是实验运行，不计分，`D:\6CJNM007\tmp\exp0028b\gpu_smoke\`）：修正后的代码在 LLaVA-1.5-7B 上跑 VQAv2 dev2 的前 20 题（覆盖 16，IDF 按完整的 dev2 题目文件）。被覆盖的 10 题的框与 EXP-0026-R03 的框 10/10 相同，被路由的题一致，每题 53 个主导 token 不重复，框内个数 = VisionZip 的框内个数 + 16（上限 53）。答案没有看。第一次尝试因驱动脚本缺少 `__main__` 保护（Windows 上 DataLoader 用 spawn 启动子进程）而卡住，手动结束，没有产生记录。
- **没有看过 R01、R02 的任何 headline**：进度检查只显示运行号与状态。

## 计划 v1.2（2026-10-26 6 点半，重跑之前写定）

- **重跑**：R03–R12 用新编号 R13–R22 重跑，配置、数据、顺序都不变。

| 重跑 | 原运行 | 配置 | 数据 |
| --- | --- | --- | --- |
| R13 | R03 | `visionzip64_c16` | vqav2_dev2 |
| R14 | R04 | `visionzip64_c16` | gqa_val400 |
| R15 | R05 | `visionzip64_c16ctr` | vqav2_dev2 |
| R16 | R06 | `visionzip64_c16ctr` | gqa_val400 |
| R17 | R07 | `visionzip64_c32` | vqav2_dev2 |
| R18 | R08 | `visionzip64_c32` | gqa_val400 |
| R19 | R09 | `visionzip64_c16all` | vqav2_dev2 |
| R20 | R10 | `visionzip64_c16all` | gqa_val400 |
| R21 | R11 | `visionzip64_t16` | vqav2_dev2 |
| R22 | R12 | `visionzip64_t16` | gqa_val400 |

- 失败的 10 次保留，文件夹与 runs.tsv 的行都不动。VisionZip-64 仍用 R01、R02（它们不经过修正的代码）。
- **代码**：修正后的 visionzip_cover.py、check_vzcover_dtype.py（新）、analyze_0028.py v1.2 一起提交到 code/token-pruning；提交前 check_vzcover.py 45/45、check_vzcover_dtype.py 11/11。
- **分析**：analyze_0028.py v1.2 只把运行表中覆盖运行的编号从 R03–R12 改为 R13–R22；选择规则、判读、归因与实现核对不变。用替身数据重新空跑通过（114 项核对都 100%，`D:\6CJNM007\tmp\exp0028b\dry.log`）。
- **若重跑又失败**：偶发退出（exit 127、没有 traceback）按以前的做法用 R23 起的新编号重跑一次；若是代码错误，就停下排查。
- **预算**：EXP-0028 共 22 次（12 次 + 10 次重跑）。失败的 10 次合计用了约 5 分钟 GPU，没有产生计分的答案。IDEA-0006 的运行数怎样计见 DEC-0047。
