# 所有论文在baseline骨架的改进内容

状态：partial，2026-10-23；2026-10-24 补入时序方向（文末）；2026-11-01 在视觉部分末尾补入“剪枝之后 LLM 里的编号与布局（2026-11-01 补）”一节（外部审查指出本文落后，DEC-0089）。材料范围：PAPER-0004 FastV、PAPER-0005 SparseVLM、PAPER-0006 VisionZip 的代码（固定 commit 见各论文笔记）与检索摘要；论文全文未读（2026-10-29 完整阅读补注：“论文全文未读”只指 2026-10-23 写本文时用的材料。PAPER-0004～0006 现已有原文笔记，核验状态为“原文+源码”，见[论文笔记 README](../论文笔记/README.md)；下文“幅值、CLS、位置编号三篇都没有讨论”一句另写有“2026-10-24 已回查论文”。baseline 选择表里 FastV 一行的“引用量待…实查”也已查，数字见该行补注）。代码层面的核验见 [EXP-0001](../实验报告/EXP-0001_record.md)。（2026-10-26 补注：视觉部分末尾补入“查新补充中的论文（2026-10-26）”一节，列 DivPrune、MMTok、CoverPruner、ShaRP、VTW、PyramidDrop 六篇：只读了摘要与 HTML 页，没有原文笔记，也没有读代码。“论文全文未读”说的是本文三行的材料；PAPER-0004～0006 另有原文笔记（[论文笔记/README](../论文笔记/README.md) 的核验状态为“原文+源码”），2026-10-24 回查过论文的地方见共性第 5 条。）

目的：将文献改造映射到 baseline 的具体位置，与[骨架全景图](baseline骨架的全景图.md)交叉阅读。

| 论文 ID | baseline 版本 | 模块 / 代码定位 | 改了什么 | 为何改 | 收益与代价 | 证据 / 条件 |
| --- | --- | --- | --- | --- | --- | --- |
| PAPER-0004 FastV | LLaVA-1.5，transformers 修改版 | B4 第 K 层前（fastv_kvcache.py） | 按第 K-1 层最后一个 token 的注意力保留前 R 个视觉 token，保留原始位置 | 深层对视觉注意力很低（SC） | FLOPs 降低；前 K 层不省；打分层不能用 FlashAttention | E-FV-C01～10 |
| PAPER-0005 SparseVLM | LLaVA-1.5 分叉，transformers 4.37.0 + flash_attn 2.3.3 | B4 第 2/6/15 层（modelling_sparse_llama.py） | 文本评分者打分，分阶段保留固定数；v1 回收被剪 token；v2 做 RoPE 校正、保留原始位置、取消回收 | 应由问题引导剪枝（SC） | 文本相关；前 2 层不省；只支持 batch 1 | E-SV-C01～12 |
| PAPER-0006 VisionZip | 官方 LLaVA，猴子补丁 | B1 CLIP 第 22 层（clip_encoder.py） | CLS 注意力选 dominant + key 相似度合并 contextual，送入 LLM 前完成压缩 | 视觉 token 大量冗余（SC） | 所有 LLM 层都变短，多轮可复用；不看问题；CLS 注入、合并幅值翻倍、位置压缩 | E-VZ-C01～12 |

（2026-10-26 补注：查新补充中的六篇只读了摘要与 HTML 页，单列在视觉部分末尾的“查新补充中的论文（2026-10-26）”一节，不并入上表。）

## 共性

1. **都不需要训练**，都插在 LLaVA-1.5 官方推理流程中，都用 LLaVA-1.5 的评测集，都只支持 batch size 1 且无 padding。
2. **都把注意力当作“重要性”**：FastV 用最后一个文本 token → 图像 token 的注意力，SparseVLM 用评分者文本 → 图像，VisionZip 用 CLS → patch。这个共同假设恰恰被 2025 年的工作质疑：注意力有位置偏差，随机选择有时更好（MODE_C S-04、S-05）。本项目的直接证据（2026-10-24）：同一层平均预算下随机选择在 POPE 三档都优于 FastV（EXP-0002，F8）；FastV 在 K=2 时保留的 patch 集中在图像底部、几乎不随问题改变（EXP-0006，F15）。
3. **都需要显式注意力矩阵**：打分那一层无法使用融合注意力内核。
4. **都是硬选择 + 预设预算**：发布的代码中没有按样本自适应的预算（SparseVLM 论文说有，代码中没有，E-SV-C05）。
5. **都改变了 LLM 看到的输入分布**（被删 token、合并 token、CLS、位置编号）。只有 VisionZip 讨论了 token 数骤减造成的不对齐，并用只微调投影层的 VisionZip‡ 处理；FastV、SparseVLM 没有讨论；幅值、CLS、位置编号三篇都没有讨论（2026-10-24 已回查论文）。

## 差异

| 维度 | FastV | SparseVLM（v1 / v2） | VisionZip |
| --- | --- | --- | --- |
| 在哪里剪 | LLM 第 K 层（单点） | LLM 第 2/6/15 层（多阶段） | 视觉编码器第 22 层（进入 LLM 前） |
| 谁来打分 | 最后一个 prompt token | 第 0 层选出的文本评分者 | CLS token |
| 是否看问题 | 间接（最后一个 token 经过前几层）；实测 K=2 时几乎不看（同一张图换问题，保留集合 95% 相同，EXP-0006） | 是，显式 | 否 |
| 被丢弃的信息 | 直接丢 | v1 聚类回收 30%；v2 直接丢 | 合并为 contextual token |
| 位置编号 | 保留原始位置 | v1 压缩；v2 保留并做 RoPE 校正 | 压缩为连续编号 |
| “64 token”的口径 | K 层后保留数（本项目按 32 层平均换算为 30） | 名义为 32 层平均；实测（v1.5，EXP-0002）前 3 层 576，之后 82、32、18，层平均 82.25 | 每层都是 64 |
| 前几层是否处理全部 576 | 是（前 K 层） | 是（实测前 3 层） | 否 |
| 多轮对话 / KV 复用 | 按首轮问题剪 | 按首轮问题剪 | 文本无关，可复用 |
| 工程状态 | 无 LICENSE；KV 版写死常量、重复写 cache | 写死 .cuda() 与路径；batch>1 出错；代码中缺秩自适应 | PyPI 包为空；写死 24 层 CLIP；默认参数不是论文预算 |

同样“平均 64 个视觉 token”时，三者的理论 FLOPs 几乎相同（vanilla 的 18～23%，EXP-0001 flops_budget.py）（2026-10-29 完整阅读补注：这是按 EXP-0001 的名义分布算的。上表“64 token 的口径”一行已写明 SparseVLM 的实测层分布（前 3 层 576，之后 82、32、18，层平均 82.25）；按它用同一公式重算，SparseVLM-64 约为 vanilla 的 21.49%（文本 20 token）到 26.05%（文本 60 token），比 FastV-64、VisionZip-64 高约 12%～16%，三者不再“几乎相同”）。差别在于**每层看到的视觉信息量**：FastV 在第 2 层以后只剩 30 个，SparseVLM 在最后 17 层只剩 17～20 个，VisionZip 每层 64 个。这可以部分解释为什么作者报告的低预算结果中 VisionZip 最好（推断，需在统一口径下实测）。

2026-10-24 统一口径实测（EXP-0002）：64 档 POPE 为 VisionZip 77.03、SparseVLM 77.67（实际层平均 82.25）、同分布随机 60.14、FastV 36.19。SparseVLM 的实际 token 比名义多，FastV 的打分由位置主导（EXP-0006）；层分布与选择规则各自的作用已由 EXP-0007 拆分：同一层分布下 SparseVLM 的规则比随机高 15.5，层平均相同时前多后少的分布没有收益。FastV 加上 FEATHER 式的去位置与覆盖后升到 64.66（EXP-0010）。第 ⑥ 步的交叉结论见[骨架全景图](baseline骨架的全景图.md)文末。

## baseline 选择

2026-10-23 按[研究流程 v2.0](../研究流程.md) 第 ① 步修订（DEC-0011）。旧版按自设评分表选了 VisionZip，与用户方法论不符，已改正。

| 用途 | 选择 | 理由 |
| --- | --- | --- |
| 骨架 baseline（必须） | LLaVA-1.5-7B vanilla | 所有结论的参照；先复现官方 POPE 85.9、MME 1510.7 |
| **主 baseline** | **FastV**（ECCV 2024 Oral） | 方法论的选择标准：近两年引用最高、有源码、衍生工作多。FastV 是本方向的开山工作，SparseVLM、VisionZip 以及质疑类工作 S-04 都以它为对照，后续改进工作最多，便于制造意外实验和统计骨架改造位置（引用量待在实验室电脑上实查并注明日期）（2026-10-29 完整阅读补注：已查。2026-10-23 21:48 用 Semantic Scholar 查询：FastV 799（有影响的引用 225）、SparseVLM 467、VisionZip 379，见 [citations-20261023.json](../论文笔记/citations-20261023.json)；该口径合并预印本与正式版本，Google Scholar 未查）。复现用 `code/token_pruning/fastv_llava.py`（已证明与官方 lmms-eval 实现输出一致，并修正了 KV cache 重复写入） |
| 对照 + 组合实验素材 | VisionZip | 编码器侧、文本无关压缩的代表；作者报告的低预算结果最好；本项目在它身上发现了三个有代码证据的问题（合并幅值、CLS 注入、位置压缩） |
| 对照 + 组合实验素材 | SparseVLM（v1.5 分支，ICML 版） | 文本引导 + 逐层剪枝的代表；在它自己的仓库里跑，batch 1 |

注意事项：

- FastV 仓库根目录没有 LICENSE 文件。本项目使用的是自己的重新实现（fastv_llava.py），发表前仍需确认引用与许可。
- 低预算下 FastV 的报告结果弱于 VisionZip（E-VZ-C02，SC）。这正好说明它的骨架上还有可改进、可制造意外的地方，不影响它作为 baseline。

交叉结论：三篇论文的差异是同一问题在“剪枝位置、打分 query、丢弃信息处理、位置编号、预算口径”五个维度上的不同取值。

- **作为论文方法主张**：直接把差异点拼起来（例如 VisionZip 压缩 + LLM 内文本引导剪枝）已有 VScan（ICML 2025）做过，新颖性低。
- **作为第 ④ 步“制造意外实验”的组合实验**：仍然可以做，目的是观察有没有两篇论文都没发现的现象，见 [MODE_C-0001](../MODE_C-0001_视觉token剪枝.md)。

## 查新补充中的论文（2026-10-26）

（2026-10-26 13 点半补）来源：[查新补充：覆盖 / 多样性一类方法与浅层剪枝](查新补充_覆盖与多样性_2026-10-26.md)（DEC-0052）。那份笔记是 Claude 自己做的网页搜索与摘要页、HTML 全文页的读取（WebSearch 4 次、WebFetch 5 次），没有开子代理，没有下载 PDF 或代码。

- **阅读级别**：下表六篇只读了摘要与 HTML 页。MMTok 只看了摘要或检索结果；VTW、PyramidDrop 在笔记里各只有一两句，读到哪一级没有写明（推断为检索摘要级）。六篇都没有 PAPER 编号、原文笔记和源码核验。
- **证据性质**：“改了什么”“为何改”“收益与代价”都是作者主张（SC）。数字来自 WebFetch 对 HTML 页的摘录；CoverPruner 表中其他方法的数字是 CoverPruner 自己的对比，来源与实现未逐一核对。引用到论文之前要对照 PDF 原表核对。
- **骨架位置**按[骨架全景图](baseline骨架的全景图.md)的 B1～B5 写；笔记没有写明的标“推断”或“笔记未记”。

| 论文 | baseline 版本 | 模块 / 位置（没有读代码） | 改了什么 | 为何改 | 收益与代价（论文报告，SC） | 证据 / 条件 |
| --- | --- | --- | --- | --- | --- | --- |
| DivPrune（arXiv 2503.02175 v2，CVPR 2025） | LLaVA-1.5（论文说视觉 token 是 4096 维） | 默认在进入 LLM 之前选（Layer 0）；推断用的是 B2 投影层的输出（论文没有明说）。Table 3 也在 LLM 之内选，用解码层特征（没有写清） | 最大-最小多样性：余弦距离 d = 1 − cos；第一个取“最近邻最远”的 token，之后每次加入“到已选集合的最小距离最大”的候选；距离矩阵只算一次 | 笔记未记 | 自己的表（约 13%～16% TFLOPs，约 56 个）：VQAv2 71.55（EM，不剪枝 76.65）、GQA 56.85（不剪枝 61.96），FastV、VTW 在同样的算力下很低。Table 3：同样的 TFLOPs（19.61%～19.80%）下，Layer 0、1、2、3 四种放置的平均分（MMB、MMMU、POPE、SQA 四列）为 62.34、60.44、56.99、38.60。CoverPruner 的 Table 1（留 64 个）：VQAv2 74.1、GQA 57.5、TextVQA 54.5、POPE 85.5、MME 1334.7 | 读了 arXiv HTML 页（v2）。本项目按论文的 Algorithm 1 自己实现（投影层输出，在 LLM 之前留 64 个），EXP-0032 计划中、未运行；没有下载官方代码，可能与官方实现有差别（DEC-0052） |
| MMTok（arXiv 2508.18264，ICLR 2026） | 笔记未记 | 笔记未记 | 视觉与文本的覆盖最大化 | 笔记未记 | CoverPruner 的 Table 1（留 64 个，CoverPruner 自己的对比）：VQAv2 75.2、GQA 58.3、TextVQA 56.0、POPE 85.8、MME 1371.5 | 只看了摘要或检索结果 |
| CoverPruner（arXiv 2609.03158 v1，EMNLP 2026 main） | LLaVA-1.5-7B（Table 1） | 在 B2 投影层的输出空间上、进入 LLM 之前选一次；题目权重用 B4 第 0 层的 input LayerNorm 与各头的 W_Q、W_K 算题目 token 对视觉 token 的注意力（“第一层注意力探针”，不跑整层） | 按题目加权的覆盖最大化（facility location：`f(S) = Σ_i w_i max_{j∈S} A_ij`，A 为去均值后截到 0 的余弦相似度；贪心按边际增益选） | 笔记未记 | Table 1（576 个中留 64 个）：VQAv2 75.9、GQA 58.4、TextVQA 55.6、POPE 86.1、MME 1432.4；同表 VisionZip 72.4、55.1、55.5、77.0、1365.2，不剪枝 78.5、61.9、58.2、85.9、1506.5 | 读了 arXiv 摘要页与 HTML 页（v1） |
| ShaRP（arXiv 2512.05385 v2） | 视频模型（笔记未记具体模型） | B4 的第一个解码层（同一笔记：FastV 在第 2 层、PDrop 在第 4 层起剪） | 按最后一个文本 token 的注意力排序；用全黑视频估计位置偏差并减去（λ = 0.6）；段内聚合；对高注意力 token 去冗余 | 笔记未记 | 按算力对齐的保留比例比较，报告保留约 97.2% 的性能；只测了视频模型 | 读了 arXiv 摘要页与 HTML 页（v2） |
| VTW（arXiv 2405.05803，AAAI 2025） | 笔记未记 | B4 的某一层之后（按 KL 选层） | 某一层之后撤除全部视觉 token | 深层不需要视觉 token | DivPrune 的表中，同样算力下 VTW 很低 | 笔记只有一句，读到哪一级没有写明（推断为检索摘要级） |
| PyramidDrop（arXiv 2410.17247） | 笔记未记 | B4 多段（B4-b，分阶段丢弃） | 分阶段按比例丢弃 | 作者的观察：浅层对丢弃敏感，第 16 层只留 10% 没有明显下降 | CoverPruner 的 Table 1 中 PDrop（留 64 个）：VQAv2 56.3、GQA 46.1、TextVQA 49.2、POPE 40.8、MME 982.2（推断 PDrop 即 PyramidDrop，笔记没有写明） | 同上 |

**与本项目的对照（推断，2026-10-26）**：

- **改在哪里**：DivPrune、CoverPruner 都在 B2 投影层输出与进入 LLM 之间选（DivPrune 用投影层输出是推断），既不在 CLIP 内部（VisionZip，B1），也不在 LLM 内部（FastV，B4）。ShaRP 在 B4 的第一个解码层；VTW、PyramidDrop 在 B4 的中深层撤除或分段丢弃。MMTok 的位置笔记未记。
- **谁来打分**：DivPrune、CoverPruner、MMTok 按多样性或覆盖选，不把注意力直接当作“重要性”，与上文“共性”第 2 条的三篇不同；CoverPruner 只用第 0 层的注意力探针给题目加权。ShaRP 仍用 LLM 注意力，但先减去用全黑视频估计的位置偏差，与 FEATHER、本项目 EXP-0008 的“去位置”同类。
- **64 个时的差距**：CoverPruner 的 Table 1 中，DivPrune、MMTok、CoverPruner 比 VisionZip 高 +1.7、+2.8、+3.5（VQAv2）与 +2.4、+3.2、+3.3（GQA）（DEC-0052）。IDEA-0006 的所选（VisionZip-64 + 每题覆盖 16）在开发集上比 VisionZip-64 高 +1.36 / +0.52（EXP-0028），在确认集上只有 +0.40 [−0.26, +1.10] / +0.10 [−0.47, +0.67]（EXP-0029）。同表 VisionZip 的 GQA 55.1、POPE 77.0，与本项目测得的 VisionZip-64（GQA testdev 55.18，EXP-0022；POPE 77.03，EXP-0002）相近；VQAv2 上本项目用的是自己抽的验证集子集，不能直接比。
- **预算口径**：在 LLM 之前剪的方法每层都是 64 个，与本项目按 32 层平均的计法相同（IDEA-0005 查新）；DivPrune 的 Table 3 按 TFLOPs 对齐。
- **对 IDEA-0007**：ShaRP 说明“第 0 层全看、第 1 层起少留”的放置本身不新；DivPrune 的 Table 3 中 Layer 1 比 Layer 0 低约 1.9，是不利于“一层全看有帮助”的旁证，但放置与特征没有分开。EXP-0032 用同样的特征（投影层输出、同一多样性选择）检验放置（DEC-0052），未运行。
- **PyramidDrop 与 EXP-0007**：层平均相同时，前多后少的分布没有收益（F20）；[骨架全景图](baseline骨架的全景图.md)已写“与 PyramidDrop 的金字塔直觉相反”。
- **查新补充里出现、这里没有单列的**：SCOPE（arXiv 2510.24214，显著性 + 覆盖，64 个时保留 96.0%）、HoloV（arXiv 2510.02912，64 个时 95.8%）、ZOO-Prune（arXiv 2509.24837，64 个时 95.2%，同处 DivPrune 94.4%），都只看了摘要或检索结果；FEATHER、HiDrop 已在骨架全景图的 B4-a、B4-b 中出现。

## 剪枝之后 LLM 里的编号与布局（2026-11-01 补）

（外部审查指出本文落后于当前对编号、填补与缓存的认识，DEC-0089 ⑥。）来源：PAPER-0010～PAPER-0020 的原文笔记；[查新补充：位置编号与借看](查新补充_位置编号与借看_2026-10-31.md)（子代理读的摘要与正文，主代理核实了 GAP、Layer-Aware 的摘要）；[IDEA-0011 卡片](../IDEA-0011_虚拟填补.md)的“相近工作与新颖性”；2026-10-31 晚上读的五篇 arXiv HTML 页（WebFetch，由小模型摘要；见[查新补充](查新补充_位置编号与借看_2026-10-31.md)第 6 节）。除标“原文笔记”的外都是作者主张的摘述（SC），细节没有对照 PDF（UN），引用之前要读原文。骨架位置按[骨架全景图](baseline骨架的全景图.md)的 B1～B5。

| 工作 | 骨架位置 | 剪掉的位置在 LLM 里放什么 | 留下的 token 的编号 | 与本项目的关系 | 阅读级别 |
| --- | --- | --- | --- | --- | --- |
| DivPrune（PAPER-0010）、MMTok（PAPER-0012）、CDPruner（PAPER-0013） | B2 之后、进入 LLM 之前 | 什么都不放 | 论文正文没有写；官方代码按缩短后的序列连续编号（SQZ-EXP-0034 的代码阅读） | 本项目的底座（DivPrune-64、CDPruner-64），fill 加在它们之上 | 原文笔记；官方代码 |
| CoverPruner（PAPER-0011） | 同上 | 什么都不放（留下的 token 不合并、不改表示）；覆盖只决定留谁 | 全文没有写 | 以“每个剪掉的 token 由谁代表”为选择目标，与 fill 的“把代表放回剪掉的位置”互补；LLaVA-1.5-7B 的 POPE 64 个 86.1、不剪枝 85.9 | 原文笔记；HTML 页 |
| CRISP（arXiv 2607.16326） | 进入 LLM 之前（按名词选 + 多样性补足） | 什么都不放 | 没有写 | LLaVA-1.5-7B 的 POPE 192 / 128 / 64 个 87.7 / 87.8 / 87.2，高于不剪枝 85.9，不报答“是”的比例（与 F112、C42 有关） | HTML 页 |
| IVC-Prune（ICLR 2026，arXiv 2602.03060） | B4 中间层选一次 | 从 KV 缓存去掉，什么都不放 | 保留原始编号（把 RoPE 位置当作隐式的视觉坐标） | 本流程的 orig 一类；没有原始编号与重排编号的单独消融 | HTML 页 |
| GAP（arXiv 2506.21873）、Layer-Aware（arXiv 2609.23715） | B3 | 什么都不放 | GAP 调整为原始编号；Layer-Aware 在对定位敏感的层用稀疏（原始）编号、其余层用连续编号 | 已报告“连续编号伤定位、原始编号补回”（F89 的主体）与“原始编号伤感知”；没有报是非题偏“是” | 摘要（主代理核实）；正文（子代理） |
| VPRune（arXiv 2609.24485）、ATP-LLaVA（CVPR 2025）、ST-Pruner、NVIDIA DART 的实现 | B3 | 什么都不放 | 保留原始的光栅位置或原顺序（DART 的实现只用于 mRoPE 模型） | orig 一类；本流程中只换原始编号比连续编号差（F107），与 VPRune 的方向不同（模型、数据不同，未核对原文） | 检索摘要 |
| Nüwa（arXiv 2602.02951） | B1 之后按区域留空间锚点 | 不填缺口 | — | 动机（剪枝丢掉全局空间参照系）与 P22 相近，机制不同 | 检索摘要 |
| DyMU（arXiv 2504.17040，NeurIPS 2025） | B1（DToMe 合并）+ B4（VTU） | 在 LLM 里用记下的位置把合并的 token 虚拟地还原到全长算注意力，每层之后再按分组平均回去 | 全长 | 最近的工作；fill 是 VTU 思路向先剪后送的选择方法的迁移；DyMU 只和 ToMe、DToMe 组合测过 | 网页摘述 |
| ERA（arXiv 2606.31982） | B4 注意力 logit | 把剪掉的 token 回收到锚点，logit 加 log 偏置 | 不知道（UN） | 与本流程的 mass 对照同类（只补注意力质量；F107：不行） | 检索摘要 |
| ASAP（arXiv 2603.14549） | B4，两阶段 | 把空出的名额补给被剪的高注意力 token（补名额，不复制 K、V） | 没有写 | 不是 fill 的做法 | HTML 页 |
| **本项目 fill（IDEA-0011）** | B3 + B4 + B5 | 每一层放最相似的保留 token 的 K、V，按槽的位置旋转；只有保留 token 计算 query 与 MLP | 原始编号，没有缺口 | CDPruner-64 上开发集 +1.02 / +2.63、新数据 +1.04 / +2.98；POPE −1.43（“未确认”，DEC-0082）；图像 KV 缓存普通实现 (576 + n) / n 倍、紧凑实现 1 倍（EXP-0044 验证） | 本项目实测 |

**与本项目的对照（推断，2026-11-01）**：

- **改在哪里**：剪枝文献改的是“留谁”（B1、B2 之后或 B4）与“留下的怎么编号”（B3）；剪掉的位置在 LLM 里放什么，只见合并一侧的 DyMU（VTU）处理过。fill 把这件事放到先剪后送的选择方法上，原则上可以加在任何选法之上（别的底座与预算在 EXP-0043～EXP-0045 检验）。
- **编号的两难**：连续编号伤定位（GAP、Layer-Aware、F89），原始编号伤感知（Layer-Aware）、在本流程中让是非题偏“是”（F88、F107）；fill 用原始编号又不留缺口，在本流程中两面都补回（F106～F108）。Layer-Aware 的逐层切换是另一种折中，没有在同一流程中比较。
- **评价**：CRISP、CoverPruner、arXiv 2604.12035（CDPruner-64 88.0 对不剪枝 87.0）都报告剪枝在 POPE 上高于或接近不剪枝，都不报答“是”的比例；MI-Pruner（arXiv 2604.03072）报告留 64 个时 FastV、VisionZip 等偏“否”（子代理读的表）。本流程中 CDPruner-64 在 POPE 上的领先都在“是”题上（F112；门槛与分辨力没有分开）。比较 POPE 时要同时给答“是”的比例（C42）。
- **成本**：VTU 与 fill 都要在每层多算注意力对；fill 的普通实现多存 576 个槽的 K、V，紧凑实现只存真实 token、每步现算（DEC-0083 ②），与普通实现在 GPU 上是否答案相同在 EXP-0044 验证。

# 时序方向：各论文在 iTransformer 骨架上的改造（2026-10-24）

状态：partial。材料范围：PAPER-0007 iTransformer、PAPER-0001 DUET、PAPER-0002 TQNet、PAPER-0003 WPMixer 的原文笔记与源码核验（固定 commit 见 code/ts_forecasting/README.md）。模块编号 T1～T6 见[骨架全景图](baseline骨架的全景图.md)的时序一节。三篇对照不是在 iTransformer 上修改，而是各自的完整模型；这里按“相对于 iTransformer 骨架，它在哪个位置换了什么”来记录。

| 论文 ID | 模块 | 改了什么（相对于 iTransformer） | 为何改（作者主张，SC） | 证据 / 代码位置 |
| --- | --- | --- | --- | --- |
| PAPER-0007 iTransformer（baseline 本身，相对于 Transformer） | T2、T3 | 每个变量的整段输入作为一个 token，注意力在变量之间而不是时间步之间；FFN 逐变量 | 时间步 token 混合了不同变量、感受野小 | model/iTransformer.py；layers/Embed.py 第 127～140 行 |
| PAPER-0001 DUET | T3 | 频域距离 → 学习的稀疏通道 mask → 只在 mask 内做通道间注意力（测试时 mask 也随机采样） | 只让相关的通道互相影响 | E-DU-C06、E-DU-C12 |
| PAPER-0001 DUET | T2 / T4 | 按窗口路由到若干 DLinear 式线性专家（时间聚类的混合专家），代替单一的嵌入与投影 | 不同时间模式用不同专家 | E-DU-C04（路由器的输入没有做实例归一化，与论文写的 X_norm 不同） |
| PAPER-0002 TQNet | T3 | 按周期移位的可学习 query 与变量做一层注意力，代替变量 token 之间的自注意力 | 用周期先验刻画变量间的关系，更轻 | E-TQ-C04（周期长度需人为指定） |
| PAPER-0002 TQNet | T2 / T4 | 线性投影 + 两层 MLP | 轻量 | models/TQNet.py |
| PAPER-0003 WPMixer | T1 | 除整体的 RevIN 外，在每个小波分量上各做一次 RevIN | 不同频带的非平稳性不同 | models/decomposition.py 第 49、96～122 行 |
| PAPER-0003 WPMixer | T2 / T3 | 小波多分辨率分解，每个分辨率一个 patch + MLP Mixer；没有变量间混合（通道独立） | 多尺度时间模式 | E-WP-C01 |
| PAPER-0003 WPMixer | T5 / T6 | 超参数用 Optuna 以测试集损失为目标搜索；测试集 drop_last | — | E-WP-C04（Optuna 的目标）；E-WP-C03、E-WP-C07（drop_last，F-TS2） |
| RevIN 系列（RevIN、Non-stationary Transformer、Dish-TS、SAN、FAN） | T1 | 归一化方式与“未来统计量”的估计 | 非平稳性 | **本项目尚未读（UN）**，第 ⑦ 步前按 analyze-paper 补读 |

共性：四种方法都保留 T1 的实例归一化（iTransformer 第 43～69 行；TQNet models/TQNet.py 第 44～73 行；WPMixer models/wavelet_patch_mixer.py 第 74～106 行；DUET linear_extractor_cluster.py 第 135、277～280 行），都用线性或 MLP 头一次输出整个预测段；改造集中在 T3（变量间关系）和 T2（时间表示）。T4、T5、T6 很少被改，T1 在三篇对照里只有 WPMixer 做了扩展，但在专门研究非平稳性的文献里被大量改造。

交叉结论：本项目测到的损失（F-TS10、F-TS11）落在三篇对照都没有改的位置（T1 的水平 + T4），而它们改得最多的 T3 在 ETTh1 上没有表现出差别（F-TS8）。这与视觉方向的情况相反：视觉方向的问题出在被改造最多的打分规则上。
