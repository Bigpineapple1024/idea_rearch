# SOP：从分析新论文到输出融合方案

版本 v1.0，2026-10-23。依据 [研究流程](研究流程.md) 与五个 skill 的 [交接协议](../.claude/skills/5-skill-research-workflow.md)，并吸收了第一轮（视觉 token 剪枝：FastV、SparseVLM、VisionZip）的教训。每一步都有输入、输出、负责的 skill 和“进入下一步”的检查点；检查点不通过就回到对应步骤，不往下推。

## 总览

```text
S1 选论文 → S2 逐篇分析（含源码核验）→ S3 代码级试跑 → S4 共性/差异 + 选 baseline
   → S5 GPU 复现 baseline → S6 深挖复现结果 → S7 Mode C 矩阵 + 新颖性检索
   → S8 融合方案（IDEA 包）→ S9 执行 → 记录 → 深挖 → 回到 S7
```

| 步骤 | 输入 | 输出（位置） | skill / 工具 | 检查点 |
| --- | --- | --- | --- | --- |
| S1 选论文 | 方向与约束（[毕业选题候选](毕业选题候选.md)） | 候选清单 + 固定 commit（[source/README](../source/README.md)） | WebSearch、git | ≥3 篇有可运行源码、同一骨架模型、单卡可跑 |
| S2 逐篇分析 | 论文 + 源码 | ANALYSIS_PACKAGE（analysis/论文笔记/PAPER-xxxx） | analyze-paper | 每个关键主张都有定位；代码事实标 DE、作者主张标 SC |
| S3 代码级试跑 | 源码 | EXPERIMENT_RECORD（analysis/实验报告/EXP-xxxx） | Python 探针、CPU 冒烟测试 | 所有方法能在同一推理流程中跑通；可疑实现已用小实验确认 |
| S4 共性/差异 + 选 baseline | S2、S3 | [改进内容映射](专题分析/所有论文在baseline骨架的改进内容.md)、[骨架全景图](专题分析/baseline骨架的全景图.md) | 人工 + Claude | 差异写成“维度 × 取值”的表；按评分表选出开发 baseline |
| S5 复现 baseline | baseline 代码、权重、数据 | EXPERIMENT_RECORD + [结果总表](experiment-results.md) | GPU、record-experiment | vanilla 与官方值相差 ≤1 个点；baseline 与论文值相差 ≤1 个点 |
| S6 深挖复现结果 | S5 的原始输出 | SQUEEZE_REPORT | squeeze-experiment | 失败案例、子任务差异已登记为 F |
| S7 Mode C | S2、S3、S6 | MODE_C_PACKAGE（analysis/MODE_C-xxxx） | deep-read（仅 Mode C） | 每个 C 有机制、证据、可区分预测和新颖性检索记录 |
| S8 融合方案 | 一个 C | IDEA_PACKAGE（analysis/IDEA-xxxx） | discover-innovation | 单一可证伪假设、单变量设计、预先写好的解释规则和预算 |
| S9 执行与回流 | IDEA 包 | 记录 → 深挖 → 更新矩阵 | record-experiment → squeeze-experiment → deep-read | 所有运行（含失败）入表；新 F 回流 S7 |

## S1 选论文

1. 用 3～5 组关键词检索近两年的工作（例：“training-free visual token pruning LLaVA 2025”），同时检索“质疑类”论文（例：“are we solving the right problem”“random pruning baseline”）。
2. 每篇记录：会议/年份、代码地址、许可证、最近提交、骨架模型、显存需求、被后续工作比较的频率。
3. 选 3 篇：至少 1 篇开山工作、1 篇当前最强、1 篇思路不同（例如剪枝位置或打分方式不同）。另选 1～2 篇质疑类论文作为 Mode C 的外部证据。
4. `git clone --depth 1` 并记录 commit；在 source/README 登记。

**检查点**：3 篇共用同一骨架（如 LLaVA-1.5-7B），在 24GB 单卡上能推理；有许可证（没有就注明风险）。

## S2 逐篇分析（必须读代码）

调用 `analyze-paper`，输入论文链接和代码 commit。除论文内容外，必须完成以下源码核验，并逐条写入 E 表：

- [ ] 方法核心在哪个文件、哪几行；实际选择规则 / 公式是否与论文一致。
- [ ] 写死的常量（层号、token 数、prompt 长度、路径、`.cuda()`）。
- [ ] “预算”的口径：输入 LLM 的数量、K 层之后的数量，还是层平均。
- [ ] 版本差异：论文版本（arXiv v1 / 会议版）、代码分支、环境变量开关。
- [ ] batch size、padding、KV cache、注意力实现（eager / SDPA / Flash）的限制。
- [ ] 许可证、依赖版本、安装方式是否真的可用（例：PyPI 包是否有代码）。
- [ ] 论文说了、代码里找不到的东西（例：SparseVLM 的秩自适应）。

**检查点**：作者主张（SC）与代码事实（DE）分开；全文没读到的部分在 missing_evidence 中写明。

## S3 代码级试跑（CPU 即可）

对 S2 中的可疑点写最小探针：随机权重小模型 + 固定种子，只检验代码行为，不产生性能结论。

- 冒烟测试：所有方法走同一个官方推理入口（例：LLaVA `generate()`），打印每层看到的 token 数。
- 等价性测试：自己的复现或改写，与官方实现在相同权重下逐步对比输出（第一轮：fastv_llava 与官方实现的相对误差 ≤3e-7）。
- 口径计算：用统一公式计算各方法在“相同预算”下的 FLOPs 和每层 token 数。

参考实现：`code/token_pruning/`（smoke_test.py、check_fastv_equivalence.py、probes/）。结果写入 EXPERIMENT_RECORD，原始输出全文保存在记录中。

## S4 共性/差异与 baseline 选择

1. **差异写成“维度 × 取值”的表**，不写成“A 有 x，B 有 y”。第一轮用到的维度：在哪里剪、谁来打分、是否看问题、被丢弃信息怎么处理、位置编号、预算口径、前几层是否处理全部 token、多轮复用、工程状态。
2. **共性优先找“共同假设”**（例：三篇都把注意力当作重要性）。共同假设往往是最好的问题来源（P）。
3. **在骨架图上标出每篇改在哪个模块**，同时标出“三篇都没改”的模块（例：投影层、位置编号）。
4. **baseline 评分表**（各 1～5 分）：

| 维度 | 看什么 |
| --- | --- |
| 可运行性 | 单卡 24GB、依赖可装、无写死路径 |
| 改动成本 | 代码量、接入方式（补丁 / 分叉）、能否用 SDPA |
| 结果强度 | 作者报告的结果、被后续工作比较时的表现 |
| 许可证 | 是否明确允许修改和发表 |
| 可切入点 | 代码里是否有可以单变量检验的问题 |

   总分最高者作为开发 baseline；开山工作和另一类代表作为对照方法。vanilla 模型永远是骨架 baseline。

## S5 复现 baseline（GPU）

1. 先跑 vanilla，与官方报告值比较（LLaVA-1.5-7B：POPE 85.9、MME 1510.7、GQA 62.0、TextVQA 58.2）。偏差超过 1 个点就先排查环境、数据、prompt 和解码设置。
2. 再跑开发 baseline 和对照方法，三档预算，统一口径。
3. 加一组随机剪枝（同预算）作为下限对照。
4. 每次运行登记到结果总表，包括失败和中断。

## S6～S7 深挖与 Mode C

- `squeeze-experiment`：按子任务、问题类型、失败案例拆分复现结果；新现象登记为 F。
- `deep-read`（仅 Mode C）：列 P/F/M/E，填满 P×F 矩阵（每个组合都要判断），提出 C 并按六个维度评分。
- **每个 C 都要做新颖性检索**：检索“方法组合关键词 + 年份”，读到的最接近工作写进 C 的评分理由。第一轮中“VisionZip + LLM 内文本引导剪枝”就是在这一步发现已被 VScan（ICML 2025）实现。

## S8 融合方案（IDEA 包）

“融合”的定义：把一篇论文中**有证据的机制**迁移到另一篇的框架中，并写清迁移条件。不是把差异点拼在一起。

必须满足：

1. 只选一个 C，对应一个可证伪假设。
2. 迁移类方案至少比较：原方法 A、只做适配的 A′、A′ + 迁移机制 B（见 [研究流程 §5](研究流程.md)）；怀疑有交互时再加组合对照。
3. 能拆开的因素拆开（例：IDEA-0001 用三组把“空间间隔”和“文本距离”分开）。
4. 写清主指标、实际意义阈值、不确定性方法、最大预算、停止规则和各种结果的解释。
5. 多步融合写成路线表，每一步都有自己的 IDEA 包，前一步出结论后再做下一步。

**禁止**：

- 把三篇论文的差异点直接叠成一个方法。
- 没做新颖性检索就宣称创新。
- 一次改多个变量后声称定位了机制。
- 用同一个测试集反复调参后报告最好结果。

## 用 Claude Code 执行时的提示词

```text
/analyze-paper <论文链接或 PDF> 代码 <仓库>@<commit>，按 SOP S2 的源码核验清单逐项完成，写入 analysis/论文笔记/PAPER-xxxx
/deep-read 输入 PAPER-xxxx 与 EXP-xxxx，按 SOP S7 生成 MODE_C 包，每个 C 做新颖性检索
/discover-innovation 输入 MODE_C-xxxx 的 C<n>，按 SOP S8 生成 IDEA 包
/record-experiment 输入 IDEA-xxxx 与 code/output/<实验ID>/ 的原始输出
/squeeze-experiment 输入 EXP-xxxx_record.md 与原始输出
```

## 第一轮的教训（2026-10-23）

| 早期做法 | 问题 | SOP 中的对应步骤 |
| --- | --- | --- |
| 只比较论文的共性和差异 | 差异不等于问题；作者自述不等于代码行为 | S2 源码核验、S3 代码级试跑 |
| 差异点拼成融合方案 | 直接拼接已被 VScan 实现；无法归因 | S7 新颖性检索、S8 融合定义 |
| 直接比较论文表格中的“64 token” | 三篇口径不同（层平均 / K 层后 / 输入），FLOPs 相同但每层信息不同 | S2 预算口径、S3 口径计算 |
| 相信 README 的安装方式 | PyPI 上的 visionzip 包没有代码；FastV KV 版需要额外补丁 | S2 工程核验 |
