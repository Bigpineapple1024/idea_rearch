# MODE_C_PACKAGE：多模态大模型视觉 token 剪枝（LLaVA-1.5-7B）

状态：partial。模式：仅 Mode C。
包 ID MODE_C-0001 / 2026-10-23 / 输入：PAPER-0001～0003（partial）、EXP-0001（代码级核验）、6 篇相关工作的检索摘要 / 材料范围：免训练视觉 token 剪枝，LLaVA-1.5 为主。

## 结构化交接包

```yaml
package_type: MODE_C_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: MODE_C-0001
matrix_id: MODE_C-0001
created_at: 2026-10-23
source_ids: [PAPER-0001, PAPER-0002, PAPER-0003, EXP-0001, S-04, S-05, S-06, S-07, S-08, S-09]
evidence_status: partial
evidence_locations: [本文“来源地图”“P/F/M/E 清单”, analysis/论文笔记/PAPER-000{1,2,3}_*.md, analysis/实验报告/EXP-0001_record.md]
missing_evidence:
- 所有论文全文未读，SC 条目来自检索摘要
- 没有真实权重下的注意力、范数或准确率数据（等待 EXP-0002）
- S-04～S-09 只读了摘要级信息
next_skill: discover-innovation
scope_and_inclusion_rules: "纳入：有源码、免训练、以 LLaVA-1.5 为主要实验对象的视觉 token 剪枝/合并工作，以及直接质疑其假设的工作。排除：需要重新训练的压缩方法、视频专用方法。"
source_map: 见“来源地图”
P: [P1, P2, P3, P4, P5]
F: [F1, F2, F3, F4, F5, F6, F7, F8, F9]
M: [M1, M2, M3, M4]
E: 见各论文笔记 E-FV/E-SV/E-VZ 与本文 S 条目
P_by_F_matrix: 见“P×F 交叉矩阵”
C: [C1, C2, C3, C4, C5]
recommended_C: C1
first_low_cost_check: "EXP-0002 复现通过后，在 POPE 与 MME 上用 VisionZip 64/128/192 比较三种位置编号（IDEA-0001 的 A/B/C 组），同时在 200 个样本上记录最后一个文本 token 对视觉 token 的注意力总量"
```

## 来源地图

| S ID | 来源 / 版本 / 上游包 ID | 类型 | 对象 / 条件 | 可用性 | 证据定位与缺口 |
| --- | --- | --- | --- | --- | --- |
| S-01 | PAPER-0001 FastV（ECCV 2024），代码 d1659729 | 论文 + 代码 | LLaVA-1.5 7B/13B | 代码已核验，全文未读 | E-FV-01～10 |
| S-02 | PAPER-0002 SparseVLM（ICML 2025）/ SparseVLM+（TPAMI 2026），代码 a9e71427、734d9003 | 论文 + 代码 | LLaVA-1.5 等 | 代码已核验，全文未读 | E-SV-01～12 |
| S-03 | PAPER-0003 VisionZip（CVPR 2025），代码 8f86b55c | 论文 + 代码 | LLaVA-1.5、NeXT 等 | 代码已核验，全文未读 | E-VZ-01～12 |
| S-04 | Wen et al., “Token Pruning in MLLMs: Are We Solving the Right Problem?”，Findings of ACL 2025，arXiv 2502.11501 | 论文 | LLaVA 系列 | 检索摘要 | FastV、SparseVLM 在部分数据集上不如随机丢弃或池化；注意力分数有位置偏差 |
| S-05 | Wen et al., “Stop Looking for Important Tokens in MLLMs: Duplication Matters More”（DART），arXiv 2502.11494，代码 ZichenWen1/DART@0c888673 | 论文 + 代码（未细读） | LLaVA、Qwen2-VL | 检索摘要 | 靠近最后一个 token 的视觉 token 注意力偏高；按重复度剪枝 |
| S-06 | VScan，ICML 2025 / TMLR 2026，arXiv 2505.22654 | 论文 | LLaVA-1.5 / NeXT | 检索摘要 | 编码器侧全局+局部扫描与合并，再在 LLM 中间层做文本相关剪枝 |
| S-07 | PyramidDrop，arXiv 2410.17247 | 论文 | LLaVA-1.5 / NeXT | 检索摘要 | 浅层对剪枝敏感，第 16 层只保留 10% 也不明显掉点；分阶段金字塔剪枝 |
| S-08 | “The Unseen Bias: How Norm Discrepancy in Pre-Norm MLLMs Leads to Visual Information Loss”，ICLR 2026，arXiv 2512.08374 | 论文 | LLaVA-1.5 | 检索摘要 | 投影后视觉 token 范数约 40，文本嵌入约 1.08；范数差异导致视觉 token 更新更慢 |
| S-09 | “Statistically Optimal Scaling for Token Merging in Transformers”，ICML 2026 | 论文 | Transformer 通用 | 检索摘要 | 合并造成残差流的协变量偏移和能量漂移，激进合并时更明显 |
| S-10 | EXP-0001 代码级核验 | 本项目实验 | 随机权重小模型 | 原始输出全文在记录中 | 只说明代码行为 |

## P/F/M/E 清单

| ID | 类型 | 定义 / 观察 | 对象与条件 | SC/DE/AI/UN | 来源定位 / E ID | 证据强度及理由 | 替代解释 / 未知 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 问题 | 用注意力分数代表视觉 token 重要性是否可靠 | 三篇方法共同假设 | SC | S-04、S-05、E-SV-08 | 中：外部论文报告随机有时更好；SparseVLM 作者自己加了位置校正 | 随机更好可能只在高预算或特定数据集成立 |
| P2 | 问题 | 压缩后的视觉 token 是否偏离 LLaVA 训练时的输入分布（幅值、CLS、位置），从而损失性能 | VisionZip（编码器侧压缩） | DE + AI | E-VZ-07、E-VZ-08、E-VZ-09；S-08、S-09 | 中：偏离本身是代码事实，影响大小未知 | LLM 可能对这些偏离不敏感 |
| P3 | 问题 | 不同论文的“保留 N 个 token”口径不同，对比结论可能不公平 | 三篇方法 | DE | E-SV-06、E-FV-10、EXP-0001 R4 | 高：口径差异是代码事实 | 作者在论文中可能已说明口径，待回查 |
| P4 | 问题 | 文本无关（VisionZip）与文本引导（SparseVLM）在相同预算下各自丢失什么信息 | 两类方法 | SC（互相矛盾） | E-VZ-04、E-SV-01 | 低：双方都是作者主张 | 可能取决于任务（OCR、计数、小目标） |
| P5 | 问题 | 文本引导剪枝按首轮问题剪，多轮对话中后续问题需要的信息可能已被丢弃 | FastV、SparseVLM | AI | E-VZ-05、E-FV-09 | 低：推断 | 多轮评测集缺乏 |
| F1 | 现象 | 深层对视觉 token 的注意力很低；深层只保留 10% 也不明显掉点 | LLaVA-1.5 | SC | E-FV-01；S-07 | 中：两篇独立工作一致 | 注意力低不等于信息不需要 |
| F2 | 现象 | SparseVLM+ 用“全 1 q/k 的 RoPE 点积”作除数校正注意力；该除数为正、随距离减小，即放大远处视觉 token；同时保留原始位置 | LLaVA-1.5-7B 维度 | DE | E-SV-08、E-SV-09 | 高（计算事实）；效果为 SC | 校正是否真的有效需消融 |
| F3 | 现象 | 作者报告：低预算下文本无关的 VisionZip 优于文本引导方法 | LLaVA-1.5，576→64 | SC | E-VZ-02、E-VZ-04 | 低：作者自报，口径未统一 | 可能来自每层 token 分布差异（F4） |
| F4 | 现象 | 同样平均 64 个 token，三者理论 FLOPs 相同，但 FastV 在第 2 层后只剩 30 个，SparseVLM 在最后 17 层只剩 17～20 个，VisionZip 每层 64 个 | LLaVA-1.5-7B 维度 | DE | EXP-0001 R4、R6 | 高 | — |
| F5 | 现象 | SparseVLM v2 取消了 v1 的回收机制 | 代码 | DE | E-SV-10 | 高（事实）；原因未知 | 可能因速度而非准确率 |
| F6 | 现象 | 位置编号处理不同：FastV、SparseVLM v2 保留原始位置，SparseVLM v1、VisionZip 压缩为连续编号 | 代码 | DE | E-FV-05、E-SV-09、E-VZ-09 | 高 | — |
| F7 | 现象 | VisionZip 的合并 token 在成员相同时幅值为 2 倍；第 0 个视觉 token 是 LLaVA 训练时没见过的 CLS | 代码 | DE | E-VZ-07、E-VZ-08 | 高（事实）；影响未知 | — |
| F8 | 现象 | 随机丢弃或池化在部分数据集上优于 FastV、SparseVLM | LLaVA 系列 | SC | S-04 | 中 | 预算与数据集范围待回查 |
| F9 | 现象 | LLaVA-1.5 投影后视觉 token 范数约 40，文本约 1.08；论文称范数差异使视觉 token 更新更慢 | LLaVA-1.5 | SC | S-08 | 中 | — |
| M1 | 机制 | RoPE 的远程衰减使文本 query 更关注位置上靠近它的视觉 token（图像末尾几行），基于注意力的打分因此带位置偏差 | LLM 内剪枝 | SC + DE | S-04、S-05、E-SV-08 | 中 | 也可能来自 attention sink |
| M2 | 机制 | Pre-norm LLM 中视觉 token 的范数影响它们被注意和被更新的方式；合并改变幅值会改变这种行为 | 编码器侧合并 | SC + AI | S-08、S-09、E-VZ-07 | 低—中 | LayerNorm 可能抵消大部分影响 |
| M3 | 机制 | 压缩位置编号会同时改变（a）视觉 token 之间的空间间隔和（b）文本与视觉 token 的距离；在 RoPE 下，（b）变短会增加文本对视觉的注意力，（a）丢失会损害空间推理 | VisionZip | AI | E-VZ-09、F2、M1 | 低：两个方向的效应可能相互抵消 | 需要拆开两个因素 |
| M4 | 机制 | 浅层需要完整视觉信息完成跨模态融合，深层只需少量 token | LLM 内剪枝 | SC | F1、S-07 | 中 | — |

## P×F 交叉矩阵

状态记号：S = supported，P = plausible，D = diagnostic-needed，U = unsupported（当前证据下无联系）。每个 P 与每个 F 都已判断。

| | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 注意力≠重要性 | U | S | U | U | U | P | U | S | U |
| P2 输入分布偏离 | U | P | P | U | U | D | D | U | P |
| P3 预算口径 | P | U | P | S | U | U | U | P | U |
| P4 文本无关 vs 文本引导 | P | U | D | P | U | U | U | P | U |
| P5 多轮 | U | U | P | U | U | U | U | U | U |

非 U 组合的说明：

| P ID | F ID | 状态 | 联系或理由 | 对象条件兼容性 | 候选 M | 支持 / 反驳 E | 替代解释 | 最小验证 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | F2 | supported | SparseVLM 作者自己为注意力加位置校正，说明打分受位置影响 | LLM 内剪枝 | M1 | E-SV-08；S-04 | 校正可能只是另一种启发式 | 比较 v1/v2 打分选中 token 的空间分布 |
| P1 | F6 | plausible | 保留原始位置会让 RoPE 衰减继续作用于打分 | LLM 内剪枝 | M1 | E-FV-05 | — | — |
| P1 | F8 | supported | 随机优于注意力打分是 P1 最直接的证据 | LLaVA 系列 | M1 | S-04 | 只在特定数据集 | 在 EXP-0002 中加入随机剪枝对照 |
| P2 | F2 | plausible | 位置变化会改变 RoPE 下的注意力分配 | 编码器侧压缩 | M3 | E-SV-08 | — | IDEA-0001 |
| P2 | F3 | plausible | VisionZip 虽有分布偏离仍表现最好，说明偏离影响可能不大，或者修正后还有提升空间 | LLaVA-1.5 | M2、M3 | E-VZ-02 | F4 的层分布差异 | — |
| P2 | F6 | diagnostic-needed | VisionZip 压缩位置，FastV/SparseVLM+ 保留位置；位置处理对编码器侧压缩的影响未被单独测试 | VisionZip | M3 | E-VZ-09、E-SV-09 | 影响可能很小 | **C1** |
| P2 | F7 | diagnostic-needed | 合并幅值翻倍和 CLS 注入是代码事实，对输出的影响未知 | VisionZip | M2 | E-VZ-07、E-VZ-08 | LayerNorm 抵消 | **C2** |
| P2 | F9 | plausible | LLaVA 对视觉 token 范数敏感（外部主张），合并改变范数 | LLaVA-1.5 | M2 | S-08 | — | C2 中测量范数分布 |
| P3 | F1 | plausible | 深层 token 少影响小，所以层平均口径可能低估了深层只剩极少 token 的代价 | LLM 内剪枝 | M4 | S-07 | — | C5 |
| P3 | F3 | plausible | “VisionZip 更好”可能部分来自口径 | LLaVA-1.5 | M4 | E-VZ-02、F4 | — | C5 |
| P3 | F4 | supported | 口径差异的直接证据 | 三篇方法 | M4 | EXP-0001 R4 | — | C5 |
| P3 | F8 | plausible | 随机对照同样需要统一口径 | — | — | S-04 | — | C5 |
| P4 | F1 | plausible | 若深层不需要视觉信息，文本引导的优势只能体现在浅中层 | LLM 内剪枝 | M4 | F1 | — | C4 |
| P4 | F3 | diagnostic-needed | 两方主张矛盾，需要相同口径下按任务类型比较 | LLaVA-1.5 | — | E-VZ-04、E-SV-01 | 任务依赖 | C4 |
| P4 | F4 | plausible | 两类方法的层分布不同，影响比较 | — | M4 | F4 | — | C4/C5 |
| P4 | F8 | plausible | 随机对照可以检验“文本引导是否真的选到了相关 token” | — | M1 | S-04 | — | C4 |
| P5 | F3 | plausible | 文本无关方法天然适合多轮 | — | — | E-VZ-05 | 缺多轮评测 | 暂缓 |

## 候选连接点

六维评分：机制完整度 / 证据质量 / 跨设置潜力 / 非平凡性 / 新颖性状态 / 可验证性，各 0–2 分。

| C ID | 对应 P/F/M/E | 连接逻辑 | 状态 | 必要条件 | 已有证据 | 未证实环节 | 可区分预测 | 评分与理由 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **C1** | P2 × F6（+F2）；M3；E-VZ-09、E-SV-09、E-FV-05 | 把 FastV / SparseVLM+ 保留原始位置的做法迁移到编码器侧压缩（VisionZip），并拆开“空间间隔”和“文本距离”两个因素 | diagnostic-needed | 复现 VisionZip 的官方数字（EXP-0002） | 位置处理差异是代码事实；SparseVLM+ 作者把保留位置作为改进（SC） | 编码器侧压缩时位置是否重要；方向未知 | 若空间间隔重要：保留间隔的组在位置、计数类问题上更好；若文本距离重要：文本距离变远的组对视觉的注意力总量下降、准确率下降 | 1/1/1/1/1/2 = 7。新颖性：SparseVLM+ 只在 LLM 内剪枝时保留位置，编码器侧压缩的位置处理尚未查到专门研究（检索范围有限） |
| **C2** | P2 × F7（+F9）；M2；E-VZ-07、E-VZ-08；S-08、S-09 | 合并 token 的幅值翻倍和 CLS 注入可能让压缩后的输入偏离 LLaVA 训练分布；改为保持范数的合并、去掉 CLS | diagnostic-needed | 同 C1 | 幅值翻倍与 CLS 注入是代码事实；LLaVA 对范数敏感（SC） | 对真实模型输出的影响 | 低预算（64）时 contextual 占比大，修正后提升应大于高预算（192）；若 LayerNorm 抵消，各组无差异 | 1/1/1/1/1/2 = 7。新颖性：S-09 讨论了通用合并的能量漂移，未见针对 VLM 视觉 token 压缩的研究（检索范围有限） |
| C3 | P1 × F8；M1 | VisionZip 用 CLS 注意力选 dominant，可能选中 ViT 中携带全局信息的高范数“伪影 token”（Darcet et al., ICLR 2024，“Vision Transformers Need Registers”，SC，未回查） | diagnostic-needed | 真实 CLIP 权重 | 无本项目证据 | 全部 | dominant token 的范数分布显著高于其他 token | 1/0/1/2/1/1 = 6 |
| C4 | P4 × F3/F4；M4 | 编码器侧压缩 + LLM 内文本引导剪枝 | plausible | — | VScan（S-06）已经实现了这种组合 | 按样本或任务自适应分配两部分预算 | — | 1/1/1/0/0/1 = 4。直接拼接已被 VScan 覆盖，只有“自适应分配”可能还有空间 |
| C5 | P3 × F4/F8 | 在统一口径（相同层平均预算 + 实测延迟 + 随机对照）下重新评估三种方法 | supported | GPU 复现 | 口径差异是代码事实 | 统一口径后排名是否改变 | 若 VisionZip 的优势主要来自层分布，同样每层 64 个 token 的随机/均匀池化会接近 VisionZip | 0/2/2/1/1/2 = 8。S-04 已部分做过类似评估，是论文的分析章节而非方法创新 |

## 交接与回流

- **交给 Discover Innovation 的 C ID**：C1（IDEA-0001）。C2 排在第二，C5 作为论文的分析章节与 EXP-0002 同步完成。
- **为什么不是“三篇差异点拼接”**：差异点拼接对应 C4，已被 VScan（ICML 2025）实现，新颖性为 0；而且一次叠加多个模块无法归因。C1、C2 都是单变量、能在一张 24GB 卡上用 VisionZip 免训练完成的检验。
- **可比性、材料缺口和选择偏差**：三篇论文都是本项目选出的代表作，2025–2026 年的大量后续工作（如 DART、VScan、HoliTom 等）只读了摘要；新颖性判断必须在 IDEA 执行前用 analyze-paper 补读最相关的 3～5 篇。
- **新实验产生的 P/F/M/E**：EXP-0002 完成后回流本矩阵，更新 F3、F8 的状态（从 SC 变为本项目 DE），并重算 C 评分。
- **被削弱或淘汰的连接点**：C4（被 VScan 覆盖）。
