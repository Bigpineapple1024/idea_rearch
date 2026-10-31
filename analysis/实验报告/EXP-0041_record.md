# EXPERIMENT_RECORD：IDEA-0011 的开发运行（虚拟填补，CDPruner 底座）

状态：计划 v1（2026-10-31 下午），在 EXP-0041 的任何运行之前写定。写计划时已看过 EXP-0034～EXP-0040 的全部结果。

- 实验 ID：EXP-0041。
- IDEA ID：[IDEA-0011](../IDEA-0011_虚拟填补.md)（MODE_C-0002 回流 30 的 C38）；DEC-0079。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划 v1（运行前写定）

### 问题

1. **Q1（主要，FLOPs 约相等）**：fill − b64（b64 = CDPruner-64，EXP-0039）。
2. **Q2（复制的内容是否要紧）**：fill − shuffle。
3. **Q3（填上缺口是否比只换编号好）**：fill − orig。
4. **同时报告（描述）**：orig − b64、mass − b64、shuffle − b64、不剪枝 − fill；按题型（VQAv2 答案类型，GQA 结构与语义类型，被路由与不路由的题）；GQA 位置题；是非题按标准答案分开的答“是”比例；第一个答案 token 的对数几率（四个新运行记录；fill − orig、fill − shuffle、fill − mass）；填补的分配（剪掉的 patch 与代表的平均余弦、一个保留 token 最多代表的槽数、不代表任何剪掉的 patch 的保留 token 数）；FLOPs（按结构、逐题，文字 token 与多出的注意力对、旋转计入）；视觉部分的 KV 缓存（按本流程的实现）；每题秒数与峰值显存。

### 数据

- VQAv2 dev2（5192 题、1000 张图）与 GQA val400（4786 题、400 张图），开发集（P14）；要写成方法的结论要在新抽的数据与 MME、POPE 上确认。

### 配置与运行（8 次；按此顺序，每个 1 次，贪心解码）

| 运行 | 配置 | 数据 |
| --- | --- | --- |
| R01、R02 | `cdpruner64_fill_logits`（--method cdpruner --cdpruner-keep 64 --layout fill） | vqav2_dev2、gqa_val400 |
| R03、R04 | `cdpruner64_shuffle_logits`（--layout shuffle，种子 0） | 同上 |
| R05、R06 | `cdpruner64_orig_logits`（--layout orig） | 同上 |
| R07、R08 | `cdpruner64_mass_logits`（--layout mass） | 同上 |

- 填补（vfill.py）：CDPruner 的选择不变（同样的 64 个，按原顺序送进 LLM）；每个剪掉的 patch 的代表 = 投影后特征余弦（float32）最大的保留 token；在每一层，576 个槽的 K、V 是代表的 K、V（RoPE 之前），按槽自己的位置（35 + patch 编号）旋转；保留的图像 token 在原始位置（35 + patch 编号），图像之后的文字按不剪枝的编号（35 + 576 + t）；每个 query 看位置不大于自己的槽与 token（按位置因果，保留的图像 token 也看代表在它之后的槽）；生成的 token 也看 576 个槽。KV 缓存只存真实 token（generate 的计数不变），每层 576 个槽的 K、V 另存，供同一题的解码步用。
- shuffle：剪掉的槽的代表在剪掉的槽之间打乱（每次运行一个 CPU 生成器，种子 0，按题目顺序抽），每个保留 token 代表的槽数与 fill 相同、位置不同。orig：只换编号（保留 token 在原始位置，文字按不剪枝的编号），缺口留着，注意力不变。mass：编号与 b64 相同（连续）；每个保留 token 作为 key 的注意力 logit 在每一层、对每个 query 加 log(c_i)，c_i = 1 + 以它为代表的剪掉的 patch 数。
- 复用：CDPruner-64（EXP-0039-R01、R02）、不剪枝（EXP-0033-R03、R04）。失败的运行保留，用新编号（R09 起）重跑，每个配置最多重跑 1 次。每次运行设 PYTHONFAULTHANDLER=1。

### 读法（预先写定）

- Q1：两个集的 95% 区间都在 0 以上 →“有希望”；只有一个 →“部分”；都含 0 而两集平均 > 0 →“方向为正，未检出”；否则 →“不像”。两集平均 ≥ +1.5 另记“提升大”。
- Q2、Q3：同样的四级读法；“有希望”或“部分”读作“内容要紧”“填上缺口要紧”。
- **之后**：Q1“有希望”，或“部分”且两集平均 ≥ +0.5 → 确认：新抽的 VQAv2 test6（1000 张图、5057 题）与 GQA val400d（400 张图、5397 题；2026-10-31 12:07 为 IDEA-0010 的确认抽出，在 EXP-0040 的任何结果之前，从未用过，与已用的全部不重合），加 MME 与 POPE，fill 对 CDPruner-64，写确认计划、提交后运行；否则 → 记录 orig、mass、shuffle 各自的作用，IDEA-0011 停在开发集，交用户决定下一步。
- 区间：每个集内按图像重抽 1000 次（种子 0）。

### 可检验的预测（推断，只为事后对照，不是判据）

- orig − b64：GQA 位置题约 +8～+14（F89 在 DivPrune 上 +13.86），“否”题上答“是”变多，VQAv2 整体约 −0.5～−2。
- fill − b64：GQA 位置题约 +6～+12，“否”题上答“是”不比 b64 多；VQAv2 约 −0.5～+1，GQA 约 +0.5～+2。
- fill − shuffle：两个集都 > 0。fill − orig：VQAv2 > 0，GQA 约 0～+1。mass − b64：方向不确定。

### 实现核对（门槛 ≥ 99%，analyze_0041.py）

- 每个运行是计划的配置 × 数据且 completed；每题一个答案；重算的得分等于计分文件与登记的准确率；CDPruner 的日志每题一条，留 64 个，留下的 patch 与 CDPruner-64（EXP-0039）相同；布局日志是本运行的模式，每题一条：35 个 token 之后 64 个图像 token、576 个槽、位置是预期的、fill / shuffle / mass 在 32 层都用了布局（orig 为 0）、保留的槽指向自己；orig、mass 的分配与 fill 相同；shuffle 的分配是 fill 的在剪掉的槽上的排列（每题），且与它不同；四个运行的提示相同；每题一条第一个答案 token 的记录。
- CPU 核对 check_vfill.py（44 项，随机初始化的小 LLaVA，6 层，DivPrune 留 20）：四种模式的预填充 logits 等于手算参照（vfill.reference_logits：eager 运算、float32 softmax；fill、shuffle 的槽是代表的 K、V 按槽的位置旋转，query 按位置因果；mass 的保留 key 加 log 代表数）；generate 等于不带缓存的手算贪心解码；全都留下（576 个）时四种模式都等于不剪枝；分配是最相似的保留 token、保留的槽指向自己，shuffle 是在剪掉的槽上的排列（代表数相同）；层看到的位置是预期的；每次预填充一条日志（图像位置、提示长度、用了布局的层数、解码步数）；参数与配置。回归（新代码对已提交代码）check_freeze 24 / 24、check_cdpruner 43 / 43、check_peekc 19 / 19、check_peek 29 / 29、check_divprune 26 / 26、check_k1cls 32 / 32。
- GPU 冒烟测试（tmp，不是 EXP 运行；VQAv2 dev2 的前 40 题，四种模式）：各 40 个答案；布局日志 40 条，64 个图像 token、576 个槽、位置正确，fill / shuffle / mass 在 32 层都用了布局；剪掉的 patch 与代表的平均余弦 0.697，一个保留 token 平均最多代表 168.0 个槽，平均 16.5 个保留 token 不代表任何剪掉的 patch；各 36～38 秒（含载入）。
- analyze_0041.py 用替身空跑（准确率部分用已封存的运行当四种模式，日志核对用冒烟测试的 40 题）：72 项都通过（替身不是结果）。按冒烟测试的提示算的 FLOPs（相对 CDPruner-64）：fill、shuffle 1.0106，orig、mass 1.0000，不剪枝 5.154。

### 代码（分支 code/token-pruning，在 a1ae3da 之上，`D:\6CJNM007\tmp\exp0041\code\`）

- vfill.py（新）、divprune.py 与 cdpruner.py（记下当前图像的特征与留下的 patch，不改计算）、run_eval.py（--layout、--layout-log、--layout-seed）、exp0002.py（divprune / cdpruner 64 的四种布局配置，本实验只用 CDPruner 的）、check_vfill.py（新）、analyze_0041.py（新）。sha256 记在 `D:\6CJNM007\tmp\exp0041\MANIFEST`。
- 排队脚本 `D:\6CJNM007\tmp\chain_0041.sh`：读完改动部分、写门文件 `D:\6CJNM007\tmp\read_repo\gate\EXP-0041.done` 后才继续；GPU 空闲；在 code/token-pruning、工作区干净、HEAD 是 a1ae3da；MANIFEST 相同；本计划在 main 上 → 复制、提交、推送 → 在仓库里重跑 check_vfill.py、check_cdpruner.py、check_divprune.py（全部通过）→ 核对配置 → 运行 R01～R08 → 都 completed、GPU 空闲后用已提交的 analyze_0041.py 分析并封存。

### 预算

- 8 次，约 2.8 GPU 小时；IDEA-0011 上限 30 次或 20 GPU 小时。
