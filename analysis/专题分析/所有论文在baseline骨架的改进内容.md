# 所有论文在baseline骨架的改进内容

状态：partial，2026-10-23。材料范围：PAPER-0001 FastV、PAPER-0002 SparseVLM、PAPER-0003 VisionZip 的代码（固定 commit 见各论文笔记）与检索摘要；论文全文未读。代码层面的核验见 [EXP-0001](../实验报告/EXP-0001_record.md)。

目的：将文献改造映射到 baseline 的具体位置，与[骨架全景图](baseline骨架的全景图.md)交叉阅读。

| 论文 ID | baseline 版本 | 模块 / 代码定位 | 改了什么 | 为何改 | 收益与代价 | 证据 / 条件 |
| --- | --- | --- | --- | --- | --- | --- |
| PAPER-0001 FastV | LLaVA-1.5，transformers 修改版 | B4 第 K 层前（fastv_kvcache.py） | 按第 K-1 层最后一个 token 的注意力保留前 R 个视觉 token，保留原始位置 | 深层对视觉注意力很低（SC） | FLOPs 降低；前 K 层不省；打分层不能用 FlashAttention | E-FV-01～10 |
| PAPER-0002 SparseVLM | LLaVA-1.5 分叉，transformers 4.37.0 + flash_attn 2.3.3 | B4 第 2/6/15 层（modelling_sparse_llama.py） | 文本评分者打分，分阶段保留固定数；v1 回收被剪 token；v2 做 RoPE 校正、保留原始位置、取消回收 | 应由问题引导剪枝（SC） | 文本相关；前 2 层不省；只支持 batch 1 | E-SV-01～12 |
| PAPER-0003 VisionZip | 官方 LLaVA，猴子补丁 | B1 CLIP 第 22 层（clip_encoder.py） | CLS 注意力选 dominant + key 相似度合并 contextual，送入 LLM 前完成压缩 | 视觉 token 大量冗余（SC） | 所有 LLM 层都变短，多轮可复用；不看问题；CLS 注入、合并幅值翻倍、位置压缩 | E-VZ-01～12 |

## 共性

1. **都不需要训练**，都插在 LLaVA-1.5 官方推理流程中，都用 LLaVA-1.5 的评测集，都只支持 batch size 1 且无 padding。
2. **都把注意力当作“重要性”**：FastV 用最后一个文本 token → 图像 token 的注意力，SparseVLM 用评分者文本 → 图像，VisionZip 用 CLS → patch。这个共同假设恰恰被 2025 年的工作质疑：注意力有位置偏差，随机选择有时更好（MODE_C S-04、S-05）。
3. **都需要显式注意力矩阵**：打分那一层无法使用融合注意力内核。
4. **都是硬选择 + 预设预算**：发布的代码中没有按样本自适应的预算（SparseVLM 论文说有，代码中没有，E-SV-05）。
5. **都改变了 LLM 看到的输入分布**（被删 token、合并 token、CLS、位置编号），但都没有显式处理这种变化（推断，待回查论文）。

## 差异

| 维度 | FastV | SparseVLM（v1 / v2） | VisionZip |
| --- | --- | --- | --- |
| 在哪里剪 | LLM 第 K 层（单点） | LLM 第 2/6/15 层（多阶段） | 视觉编码器第 22 层（进入 LLM 前） |
| 谁来打分 | 最后一个 prompt token | 第 0 层选出的文本评分者 | CLS token |
| 是否看问题 | 间接（最后一个 token 经过前几层） | 是，显式 | 否 |
| 被丢弃的信息 | 直接丢 | v1 聚类回收 30%；v2 直接丢 | 合并为 contextual token |
| 位置编号 | 保留原始位置 | v1 压缩；v2 保留并做 RoPE 校正 | 压缩为连续编号 |
| “64 token”的口径 | K 层后保留数（本项目按 32 层平均换算为 30） | 32 层平均（前 2 层 576，最后 17 层 17～20） | 每层都是 64 |
| 前几层是否处理全部 576 | 是（前 K 层） | 是（前 2 层） | 否 |
| 多轮对话 / KV 复用 | 按首轮问题剪 | 按首轮问题剪 | 文本无关，可复用 |
| 工程状态 | 无 LICENSE；KV 版写死常量、重复写 cache | 写死 .cuda() 与路径；batch>1 出错；代码中缺秩自适应 | PyPI 包为空；写死 24 层 CLIP；默认参数不是论文预算 |

同样“平均 64 个视觉 token”时，三者的理论 FLOPs 几乎相同（vanilla 的 18～23%，EXP-0001 flops_budget.py）。差别在于**每层看到的视觉信息量**：FastV 在第 2 层以后只剩 30 个，SparseVLM 在最后 17 层只剩 17～20 个，VisionZip 每层 64 个。这可以部分解释为什么作者报告的低预算结果中 VisionZip 最好（推断，需在统一口径下实测）。

## baseline 选择

2026-10-23 按[研究流程 v2.0](../研究流程.md) 第 ① 步修订（DEC-0009）。旧版按自设评分表选了 VisionZip，与用户方法论不符，已改正。

| 用途 | 选择 | 理由 |
| --- | --- | --- |
| 骨架 baseline（必须） | LLaVA-1.5-7B vanilla | 所有结论的参照；先复现官方 POPE 85.9、MME 1510.7 |
| **主 baseline** | **FastV**（ECCV 2024 Oral） | 方法论的选择标准：近两年引用最高、有源码、衍生工作多。FastV 是本方向的开山工作，SparseVLM、VisionZip 以及质疑类工作 S-04 都以它为对照，后续改进工作最多，便于制造意外实验和统计骨架改造位置（引用量待在实验室电脑上实查并注明日期）。复现用 `code/token_pruning/fastv_llava.py`（已证明与官方 lmms-eval 实现输出一致，并修正了 KV cache 重复写入） |
| 对照 + 组合实验素材 | VisionZip | 编码器侧、文本无关压缩的代表；作者报告的低预算结果最好；本项目在它身上发现了三个有代码证据的问题（合并幅值、CLS 注入、位置压缩） |
| 对照 + 组合实验素材 | SparseVLM（v1.5 分支，ICML 版） | 文本引导 + 逐层剪枝的代表；在它自己的仓库里跑，batch 1 |

注意事项：

- FastV 仓库根目录没有 LICENSE 文件。本项目使用的是自己的重新实现（fastv_llava.py），发表前仍需确认引用与许可。
- 低预算下 FastV 的报告结果弱于 VisionZip（E-VZ-02，SC）。这正好说明它的骨架上还有可改进、可制造意外的地方，不影响它作为 baseline。

交叉结论：三篇论文的差异是同一问题在“剪枝位置、打分 query、丢弃信息处理、位置编号、预算口径”五个维度上的不同取值。

- **作为论文方法主张**：直接把差异点拼起来（例如 VisionZip 压缩 + LLM 内文本引导剪枝）已有 VScan（ICML 2025）做过，新颖性低。
- **作为第 ④ 步“制造意外实验”的组合实验**：仍然可以做，目的是观察有没有两篇论文都没发现的现象，见 [MODE_C-0001](../MODE_C-0001_视觉token剪枝.md)。
