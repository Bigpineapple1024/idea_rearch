# EXPERIMENT_RECORD：IDEA-0011 的推广与实现（VisionZip-64 加 fill；紧凑的 KV 缓存，开发集）

状态：已完成（2026-11-01 03:19:14～04:28:36，4 次，代码 a23e1d2；分析 04:29:19 封存；结果见文末“结果”，DEC-0093：Q2、Q3 逐位相同）。前一状态：计划 v1.2（2026-11-01 凌晨 3 点后，GPU 冒烟测试之后、任何运行之前写定：拿掉 Q1，VisionZip-64 加 fill 在 GPU 上跑不通，见文末；DEC-0091）。前一状态：计划 v1.1（2026-11-01 00:20，加 Q3：DivPrune-64 在现在的代码上重跑，R07、R08，见文末；DEC-0086；在 EXP-0044 的任何运行之前、EXP-0043 的任何结果之前写定）。前一状态：计划 v1（2026-11-01 凌晨（更正：实际写于 2026-10-31 晚上 11 点半前后，原写“2026-11-01 凌晨”）），在 EXP-0044 的任何运行之前写定；写时 EXP-0043 正在运行（只看过运行号、状态与答案数），没有看过 EXP-0043 的任何结果。

- 实验 ID：EXP-0044。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)（C40）；用户决定 DEC-0083；DEC-0085。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（换底座：VisionZip-64）**：VisionZip-64 加 fill − VisionZip-64。
2. **Q2（紧凑的 KV 缓存）**：CDPruner-64 加 fill 的紧凑实现（`--layout-compact`）对普通实现（EXP-0041-R01、R02）：答案与第一个答案 token 的对数概率是否相同。
3. **同时报告（描述）**：不剪枝 − VisionZip-64 加 fill；按题型、GQA 位置题；是非题按标准答案的答“是”比例；与不剪枝的回答相同的比例；是非题上第一个答案 token 的对数几率的 AUC（VisionZip-64 与它加 fill 都记录）；VisionZip 布局的统计；FLOPs（577 个槽）；按本流程的实现的图像 KV 缓存（普通实现：缓存里 64 个保留 token 加旁边 576 个槽，10 倍；紧凑实现：1 倍）；紧凑实现、普通实现与 CDPruner-64 的每题秒数与峰值显存；本流程中 VisionZip-64 重跑与 EXP-0028 的答案是否相同。

### 设计

- **VisionZip 上的 fill（vzfill.py）**：VisionZip-64（官方代码 source/VisionZip，只读、不改）送进 LLM 的 64 个 token 是 CLS、53 个主要 patch（按 CLS 注意力，按编号顺序）、10 个上下文 token（目标 patch 加与它最相似的其余 patch 的平均）。布局：CLS 占网格之前自己的槽（槽 0，位置 start），patch p 占槽 1 + p（位置 start + 1 + p）；主要 patch 在自己的槽，上下文 token 在它的目标 patch 的槽；其余 patch 槽放投影后特征（第 −2 层的状态过投影层，即不剪枝的 LLaVA 给这个 patch 的输入）与 64 个 token 余弦最大的非 CLS token 的 K、V；CLS 不代表任何 patch；图像之后的文字按 577 个图像 token 编号。这是一个设计选择：VisionZip 的 CLS 与合并出来的 token 在 LLaVA 训练时没有 patch 位置，上面是与 fill 最接近的做法。VisionZip 送进 LLM 的 64 个 token 不变（check_vzfill 核对）。
- **紧凑的 KV 缓存**：fill 里 576 个槽在每一次注意力里代替图像 token 的 K，所以缓存里图像部分旋转后的 K 从来不读；紧凑实现把那里换成旋转前的 K，每个解码步用与预填充相同的取值与旋转现算 576 个槽的 K（V 取缓存里的）；旁边不再另存槽。图像部分的 KV 缓存回到 n 个（与底座相同），运算相同，所以输出相同（CPU 上逐位相同，check_vfill）。

### 数据

- VQAv2 dev2（5192 题）与 GQA val400（4786 题），开发集（P14）。

### 配置与运行（6 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `visionzip64_logits`（--method visionzip --dominant 54 --contextual 10，记对数概率） | vqav2_dev2、gqa_val400 |
| R03、R04 | `visionzip64_fill_logits`（同上加 --layout fill，经 vzfill.py） | 同上 |
| R05、R06 | `cdpruner64_fillc_logits`（--method cdpruner --cdpruner-keep 64 --layout fill --layout-compact） | 同上 |

- 复用：普通实现的 fill（EXP-0041-R01、R02）、CDPruner-64（EXP-0039-R01、R02）、不剪枝（EXP-0033-R03、R04）；VisionZip-64 的旧运行（EXP-0028-R01、R02）只作重跑的对照。失败的运行保留，用新编号（R07 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- Q2：两个集每一题的答案与对数概率都相同 →“逐位相同”；每个集 ≥ 99.5% 的题答案相同 →“等价”（个别不同来自 GPU 运算顺序）；否则 →“不等价”。
- **之后**：Q1“有希望”，或“部分”且两集平均 ≥ +0.5 → 在新数据上确认（[EXP-0045](EXP-0045_record.md) 的 D 部分，计划 v1.1，在 EXP-0044 的任何结果之前写定）；否则记录“在 VisionZip-64 上没有检出”。Q2“逐位相同”或“等价” → 报告显存时用紧凑实现（图像 KV 缓存 1 倍）；“不等价” → 不用，查原因。
- 区间：每个集内按图像重抽 1000 次（种子 0，analyze_0028 的读法与计分）。

### 可检验的预测（推断，只为事后对照，不是判据）

- Q1：VisionZip-64 在本流程中较弱（EXP-0028：71.93 / 62.20），也是连续编号、丢布局；fill 给回布局：VQAv2 0～+2、GQA +1～+4，GQA 位置题 > +5；但上下文 token 是合并出来的，复制到空槽可能不如保留的 patch 忠实，方向不确定的部分在 VQAv2。
- Q2：“逐位相同”；紧凑实现的峰值显存接近 CDPruner-64（约 14.44 GiB），每题秒数接近普通实现。

### 实现核对（门槛 ≥ 99%，analyze_0044.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；VisionZip 加 fill 的布局日志是 fill、每题一条：35 个 token 之后 64 个图像 token、577 个槽、位置是预期的、32 层都用了布局、槽 0 是 CLS、53 个主要与 10 个目标的槽、每个 token 的槽指向自己；紧凑实现的布局日志标为 compact、每题一条（64 个、576 个槽、32 层），分配与普通实现相同，留下的 patch 与普通实现相同（每题）；比较的两个运行提示相同；记录对数概率的运行每题一条。
- CPU（tmp 的代码副本）：check_vfill 56 / 56（新加 12 项：紧凑实现在 fill、shuffle 上预填充 logits 与生成与普通实现逐位相同、旁边不存槽、缓存长度与底座相同、图像部分是旋转前的 K，参数与配置）；check_vzfill 13 / 13（新：VisionZip 的 64 个 token 不变、主要 patch 与目标与这里独立算的相同、代表、577 个槽的手算参照、贪心解码、紧凑实现、拒绝）；回归 check_cdpruner 43 / 43、check_divprune 26 / 26、check_vzcover 45 / 45。
- GPU 冒烟测试（EXP-0043 结束后、开门前；tmp，不是 EXP 运行）：VQAv2 dev2 的前 8 题，VisionZip-64、它加 fill、CDPruner-64 加 fill 的普通与紧凑实现；紧凑与普通的答案与对数概率相同；布局日志正确；答案不计分、不看。analyze_0044.py 先用替身空跑。

### 代码（分支 code/token-pruning，在 EXP-0043 的代码 ae7242e 之上，`D:\6CJNM007\tmp\exp0044\code\`）

- vfill.py（compact；选择可以直接交出布局）、vzfill.py（新）、run_eval.py（--layout-compact；--layout fill 可配 plain --method visionzip）、exp0002.py（两个配置）、check_vfill.py（加 12 项）、check_vzfill.py（新）、analyze_0044.py（新）。不加新参数时其余方法的计算不变。

### 预算

- 6 次，约 2.2 GPU 小时；IDEA-0011 合计 34 次（另有 EXP-0045 最多 20 次），上限 60 次、20 GPU 小时（DEC-0085）。

## 计划 v1.1（2026-11-01 00:20，在 EXP-0044 的任何运行之前、EXP-0043 的任何结果之前写定；DEC-0086）

- **加 Q3（DivPrune-64 的重跑）**：EXP-0043 的 Q1 复用了 EXP-0032 的 DivPrune-64（代码 e1daaab，之后 divprune.py 只多记一个属性、不改计算，只核对了留下的 patch 相同），用户要求在现在的代码上重跑底座。R07、R08：`divprune64` × vqav2_dev2、gqa_val400（各 1 次，贪心）。读法：两个集每一题的答案都与 EXP-0032 相同 →“逐位相同”；每个集 ≥ 99.5% 的题相同 →“等价”（个别不同来自 GPU 运算顺序）；否则 →“不等价”（查原因；EXP-0043 的 Q1 以重跑的底座另算一次，记入 EXP-0043 的结果）。
- **同时报告（描述）**：重跑与 EXP-0032 留下的 patch 是否逐题相同；重跑 − EXP-0032 的配对差；EXP-0043 的 fill（R01、R02）− 重跑的 DivPrune-64，与 EXP-0043 的 Q1 对照。
- **核对加**：R07、R08 是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；与 EXP-0032 的提示相同。analyze_0044.py v1.1 按此改，替身空跑（EXP-0043 的 fill 用 EXP-0041 的运行当替身，不读 EXP-0043）后记入 MANIFEST。GPU 冒烟测试加 DivPrune-64 的 8 题（留下的 patch 与 EXP-0032-R01 的前 8 题比）。
- **预算**：8 次，约 2.9 GPU 小时；IDEA-0011 合计 36 次（另有 EXP-0045 最多 20 次，合计最多 56 次；上限 60 次、20 GPU 小时）。v1 的其余内容不变。

## 计划 v1.2（2026-11-01 凌晨 3 点后，GPU 冒烟测试之后、EXP-0044 的任何运行之前写定；DEC-0088 ②、DEC-0091）

- **拿掉 Q1（VisionZip-64 加 fill）**：开门前的 GPU 冒烟测试（VQAv2 dev2 前 8 题；tmp 的代码副本，不是 EXP 运行）里，VisionZip-64 加 fill 跑不通：vzfill.py 第 85 行把 CLIP 第 −2 层的状态（float32）直接送进半精度的投影层，报 “mat1 and mat2 must have the same dtype, but got Float and Half”，没有答案；CPU 上的 check_vzfill 全程用 float32，没有测出。按用户睡前的决定（DEC-0088 ②：冒烟测试有部分出问题时，在任何运行之前写 v1.2 拿掉那部分、写明原因，其余照跑，拿掉的部分等用户醒来再查），这里不修，拿掉 Q1 的四次运行：R03、R04（VisionZip-64 加 fill）与 R01、R02（VisionZip-64 底座，只为 Q1 与它的描述服务）；编号空着。看起来的修法（投影之前转成投影层的数据类型）与 Q1 是否另做，等用户决定。
- **其余照跑，读法不变**：R05、R06（紧凑实现，Q2）与 R07、R08（DivPrune-64 的重跑，Q3）。冒烟测试里这两部分都通过：紧凑实现与普通实现的答案、第一个答案 token 的对数概率 8 / 8 相同，留下的 patch 8 / 8 相同，布局日志正确（紧凑实现标为 compact）；峰值显存紧凑 14.44 GiB、普通 14.72 GiB；DivPrune-64 重跑留下的 patch 与 EXP-0032-R01 的前 8 题相同（8 / 8）。冒烟脚本的汇总在读 VisionZip 加 fill 的布局日志时中断，其后的每题秒数与显存另行补读（tmp/exp0044/smoke_meta.log）。
- **分析**：analyze_0044.py v1.2 只加一个开关（Q1 = False），把 VisionZip 的两个运行、它的布局核对、Q1 的判读与 VisionZip 的描述放到开关后面；Q2、Q3 与其余描述不变。替身空跑之后记入 MANIFEST。
- **提交的代码**：照开门前读过的 7 个文件提交；vzfill.py 不用于本实验的运行，在 CPU 上照旧通过 check_vzfill，GPU 上的问题记在这里与提交说明里，修复另做（等用户）。
- **EXP-0045 的 D 部分**：条件是本实验的 Q1；Q1 没有运行，D 不运行。EXP-0045 运行 A、B、C 与不剪枝，最多 16 次（DEC-0090、DEC-0091）。
- **预算**：4 次，约 1.2 GPU 小时；IDEA-0011 合计 32 次，加 EXP-0045 最多 16 次，合计最多 48 次（上限 60 次、20 GPU 小时）。
- 写时已看过 EXP-0043 的结果（与本计划的 Q2、Q3 无关）；没有看过 EXP-0044 的任何结果（还没有运行）。

## 开门前的阅读（DEC-0055，2026-11-01 凌晨）

- **读的范围**：上一道门（EXP-0043，3fb0898）以来的改动：main 的 6825b9f（EXP-0045 计划 v1、DEC-0084）、c1c8794（本计划 v1、EXP-0045 v1.1、DEC-0085）、a3c9f10（时间更正）、52c9079（EXP-0045 v1.2）、ba8f903（SQZ-EXP-0041、SQZ-EXP-0042 的单独报告，1A4B-0004）、c19fadd（DEC-0086、本计划 v1.1）、1e63748（素材库、README）、fd92c8c（DEC-0087）、60e101a（DEC-0088）、f99a591（DEC-0089：外部审查的核对与措辞更正）、6017307（查新第 6 节）、182f82b（SQZ-EXP-0038 的单独报告）、bcb772c（素材库 MAT-0049～MAT-0053）、56bb299（EXP-0043 的结果，DEC-0090）、eb5bdfc（本计划 v1.2，DEC-0091）；code/token-pruning 的 ae7242e（EXP-0043 的代码，开门前已读）；新的运行产物：EXP-0043 的 R01～R10 与分析（已封存，ec1736f4…，verify_seal ALL OK）。
- **本实验要提交的代码**（相对 ae7242e；导出的其余 97 个 .py 文件与 ae7242e 逐字节相同；sha256 记在 `D:\6CJNM007\tmp\exp0044\MANIFEST`）：
  - vfill.py：①紧凑实现（`compact=True`，只用于 fill、shuffle）：预填时算槽与普通实现相同，之后把缓存里图像那一段换成这一层旋转前的 K（槽代替了图像 token 的 K，缓存的这一段不会以旋转后的样子被读到）；解码时从缓存的这一段用同样的取数与旋转现算 576 个槽（V 用缓存里的）；旁边不存槽，缓存长度与底座相同，算术与普通实现相同。②接受选法自己给的布局（prune_last 带 rep、kept、n_slots）；槽数从固定的 576 改为每题的 n_slots（编号、掩码、参考计算都随之改）；VisionZip 上只接受 fill。
  - vzfill.py（新）：VisionZip-64 的布局（CLS 占网格之前的 0 号槽，主导 patch 占自己的槽，上下文 token 占它的目标 patch 的槽，其余槽放投影后特征余弦最大的非 CLS token，共 577 个槽）。GPU 上把 float32 的第 −2 层状态直接送进半精度的投影层而报错（计划 v1.2）；本实验不用它，照读过的样子提交。
  - run_eval.py：`--layout-compact`（只与 fill、shuffle 一起）；`--layout fill` 可与普通的 `--method visionzip` 一起（先接 vzfill 再接 vfill）。exp0002.py：登记 visionzip64_fill_logits、cdpruner64_fillc_logits（名字带 _logits 自动记第一个答案 token 的对数几率，带 --layout 自动记布局日志）。
  - check_vfill.py（加 12 项）、check_vzfill.py（新）、analyze_0044.py v1.2（计划 v1.1 的 Q2、Q3 与实现核对；Q1 放在开关 Q1 = False 后面）。
  - 排队脚本 `D:\6CJNM007\tmp\chain_0044.sh`（计划提交 c1c8794、c19fadd、eb5bdfc；只运行 R05～R08）、run_analyze_0044.sh、seal_0044.sh、提交说明 msg_code79.txt（都 bash -n 通过；说明写了 vzfill.py 的问题）。
- **核对**：不改任何已有方法的计算（DivPrune、CDPruner、普通 fill 的路径不变，check_vfill 原有的各项照旧通过）；tmp 的代码副本上 check_vfill 56 / 56（紧凑实现：fill、shuffle 的预填对数几率与普通实现 torch.equal、生成相同、旁边不存槽、缓存长度与底座相同且图像段是旋转前的 K；参数与配置的登记）、check_vzfill 13 / 13（CPU、float32）；回归 check_cdpruner 43 / 43、check_divprune 26 / 26、check_vzcover 45 / 45。GPU 冒烟测试（VQAv2 dev2 前 8 题；答案不计分、不看）：VisionZip-64 正常；**VisionZip-64 加 fill 跑不通**（见计划 v1.2）；CDPruner-64 加 fill 的普通与紧凑实现都正常，答案与第一个答案 token 的对数概率 8 / 8 相同、留下的 patch 8 / 8 相同，布局日志正确（64 个、576 个槽、位置正确、32 层、紧凑实现标为 compact），峰值显存紧凑 14.44 GiB、普通 14.72 GiB；DivPrune-64 重跑留下的 patch 与 EXP-0032-R01 的前 8 题相同（8 / 8）。analyze_0044.py v1.2 的替身空跑（已封存的 EXP-0032、EXP-0039、EXP-0041、EXP-0033 运行当替身，日志核对用冒烟测试的 8 题）73 项都通过（替身不是结果）。
- **发现**：VisionZip-64 加 fill 在 GPU 上的数据类型问题——按 DEC-0088 ② 不修，计划 v1.2 拿掉 Q1，修复等用户决定；其余代码与计划 v1、v1.1、v1.2 一致，没有要改的。
- **开门**：本节提交之后写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0044.done`。

## 结果（2026-11-01 凌晨 4 点半，看过封存的分析之后写）

- **运行**：R05～R08 都 completed（2026-11-01 03:19:14～04:28:36，代码 a23e1d2；每个运行的 git.txt 是 a23e1d2、没有未提交的改动）；R01～R04 按计划 v1.2 没有运行（编号空着）。每个配置 1 次、贪心解码；评测共 1.13 GPU 小时（计划约 1.2）。IDEA-0011 共用了 32 / 60 次、约 10.6 / 20 GPU 小时。
- **分析与封存**：chain_0044 用已提交的 analyze_0044.py v1.2 在 CPU 上分析，04:29:19 封存（code/output/EXP-0044/analysis，SHA256SUMS b09dd146…，已备份）；独立核对（verify_seal.sh EXP-0044）ALL OK。
- **实现核对**：76 项全部 100%：四个新运行与复用的运行都是计划的配置 × 数据且 completed；每题一个答案，重算的得分等于计分文件与登记的准确率；紧凑实现的布局日志标为 compact、每题一条（64 个、576 个槽、32 层、位置正确），分配与普通实现相同，留下的 patch 与普通实现逐题相同；DivPrune-64 的重跑留下的 patch 与 EXP-0032 逐题相同；比较的两个运行提示相同；记录对数概率的运行每题一条。

### 结果（VQAv2 dev2 5192 题 / GQA val400 4786 题）

| 比较 | VQAv2 dev2 | GQA val400 |
| --- | --- | --- |
| **Q2 紧凑实现（R05、R06）对普通实现（EXP-0041-R01、R02）**：答案相同 / 对数概率相同 | 100% / 100%（最大差 0） | 100% / 100%（最大差 0） |
| **Q3 DivPrune-64 的重跑（R07、R08）对 EXP-0032-R01、R02**：答案相同 | 100%（准确率差 +0.000） | 100%（+0.000） |
| EXP-0043 的 fill − 重跑的 DivPrune-64（描述） | +1.37 [+0.60, +2.13] | +3.20 [+2.08, +4.30] |

| 配置 | 准确率（VQAv2 / GQA） | 每题秒数 | 峰值显存（GiB） | 图像 KV 缓存（相对 CDPruner-64，本流程的实现） |
| --- | --- | --- | --- | --- |
| CDPruner-64（EXP-0039） | 74.87 / 66.32 | 0.2087 / 0.2000 | 14.44 / 14.44 | 1 |
| fill，普通实现（EXP-0041） | 75.88 / 68.95 | 0.2363 / 0.2257 | 14.74 / 14.74 | 10 |
| **fill，紧凑实现**（R05、R06） | 75.88 / 68.95 | 0.2540 / 0.2393 | 14.46 / 14.46 | 1 |
| DivPrune-64 的重跑（R07、R08） | 74.13 / 63.83 | 0.1636 / 0.1575 | 13.96 / 13.96 | — |

- 紧凑实现的每题秒数比普通实现多约 7.5% / 6.0%（普通实现的秒数来自 EXP-0041，不同时间的运行），比 CDPruner-64 多约 22% / 20%；峰值显存与 CDPruner-64 只差 0.02 GiB。

### 按计划写定的读法

- **Q1**：计划 v1.2 拿掉，没有运行（VisionZip-64 加 fill 在 GPU 冒烟测试里跑不通，DEC-0091）；修好之后在 [EXP-0046](EXP-0046_record.md) 重做（DEC-0092）。
- **Q2**：两个集每一题的答案与对数概率都相同 →“逐位相同”。计划写定的之后：报告显存时用紧凑实现（图像 KV 缓存 1 倍）。
- **Q3**：两个集每一题的答案都与 EXP-0032 相同 →“逐位相同”；EXP-0043 的 Q1（复用 EXP-0032 的底座）不受影响，用重跑的底座另算一次与原来相同（+1.37 / +3.20）。
- **预测对照**（只为事后对照）：Q2“逐位相同”：符合；“紧凑实现的峰值显存接近 CDPruner-64（约 14.44 GiB）”：符合（14.46）；“每题秒数接近普通实现”：大致符合（多约 6%～7.5%）。

### 解释与边界

- 紧凑实现在缓存里存旋转前的 K、每个解码步现算 576 个槽，算术与普通实现相同（设计如此，CPU 上也逐位相同），所以输出逐位相同；显存省下的就是旁边那 576 个槽，代价是每步多一次取数与旋转（秒数多约 6%～7.5%）。
- 边界：只在 CDPruner-64、batch 1 上；秒数是本流程的实现，不代表优化后的速度；普通实现的秒数来自另一时间的运行。
