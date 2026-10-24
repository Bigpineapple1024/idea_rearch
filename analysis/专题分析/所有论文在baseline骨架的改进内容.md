# 所有论文在baseline骨架的改进内容

状态：partial，2026-10-23；2026-10-24 补入时序方向（文末）。材料范围：PAPER-0004 FastV、PAPER-0005 SparseVLM、PAPER-0006 VisionZip 的代码（固定 commit 见各论文笔记）与检索摘要；论文全文未读。代码层面的核验见 [EXP-0001](../实验报告/EXP-0001_record.md)。

目的：将文献改造映射到 baseline 的具体位置，与[骨架全景图](baseline骨架的全景图.md)交叉阅读。

| 论文 ID | baseline 版本 | 模块 / 代码定位 | 改了什么 | 为何改 | 收益与代价 | 证据 / 条件 |
| --- | --- | --- | --- | --- | --- | --- |
| PAPER-0004 FastV | LLaVA-1.5，transformers 修改版 | B4 第 K 层前（fastv_kvcache.py） | 按第 K-1 层最后一个 token 的注意力保留前 R 个视觉 token，保留原始位置 | 深层对视觉注意力很低（SC） | FLOPs 降低；前 K 层不省；打分层不能用 FlashAttention | E-FV-C01～10 |
| PAPER-0005 SparseVLM | LLaVA-1.5 分叉，transformers 4.37.0 + flash_attn 2.3.3 | B4 第 2/6/15 层（modelling_sparse_llama.py） | 文本评分者打分，分阶段保留固定数；v1 回收被剪 token；v2 做 RoPE 校正、保留原始位置、取消回收 | 应由问题引导剪枝（SC） | 文本相关；前 2 层不省；只支持 batch 1 | E-SV-C01～12 |
| PAPER-0006 VisionZip | 官方 LLaVA，猴子补丁 | B1 CLIP 第 22 层（clip_encoder.py） | CLS 注意力选 dominant + key 相似度合并 contextual，送入 LLM 前完成压缩 | 视觉 token 大量冗余（SC） | 所有 LLM 层都变短，多轮可复用；不看问题；CLS 注入、合并幅值翻倍、位置压缩 | E-VZ-C01～12 |

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

同样“平均 64 个视觉 token”时，三者的理论 FLOPs 几乎相同（vanilla 的 18～23%，EXP-0001 flops_budget.py）。差别在于**每层看到的视觉信息量**：FastV 在第 2 层以后只剩 30 个，SparseVLM 在最后 17 层只剩 17～20 个，VisionZip 每层 64 个。这可以部分解释为什么作者报告的低预算结果中 VisionZip 最好（推断，需在统一口径下实测）。

2026-10-24 统一口径实测（EXP-0002）：64 档 POPE 为 VisionZip 77.03、SparseVLM 77.67（实际层平均 82.25）、同分布随机 60.14、FastV 36.19。SparseVLM 的实际 token 比名义多，FastV 的打分由位置主导（EXP-0006）；层分布与选择规则各自的作用已由 EXP-0007 拆分：同一层分布下 SparseVLM 的规则比随机高 15.5，层平均相同时前多后少的分布没有收益。FastV 加上 FEATHER 式的去位置与覆盖后升到 64.66（EXP-0010）。第 ⑥ 步的交叉结论见[骨架全景图](baseline骨架的全景图.md)文末。

## baseline 选择

2026-10-23 按[研究流程 v2.0](../研究流程.md) 第 ① 步修订（DEC-0011）。旧版按自设评分表选了 VisionZip，与用户方法论不符，已改正。

| 用途 | 选择 | 理由 |
| --- | --- | --- |
| 骨架 baseline（必须） | LLaVA-1.5-7B vanilla | 所有结论的参照；先复现官方 POPE 85.9、MME 1510.7 |
| **主 baseline** | **FastV**（ECCV 2024 Oral） | 方法论的选择标准：近两年引用最高、有源码、衍生工作多。FastV 是本方向的开山工作，SparseVLM、VisionZip 以及质疑类工作 S-04 都以它为对照，后续改进工作最多，便于制造意外实验和统计骨架改造位置（引用量待在实验室电脑上实查并注明日期）。复现用 `code/token_pruning/fastv_llava.py`（已证明与官方 lmms-eval 实现输出一致，并修正了 KV cache 重复写入） |
| 对照 + 组合实验素材 | VisionZip | 编码器侧、文本无关压缩的代表；作者报告的低预算结果最好；本项目在它身上发现了三个有代码证据的问题（合并幅值、CLS 注入、位置压缩） |
| 对照 + 组合实验素材 | SparseVLM（v1.5 分支，ICML 版） | 文本引导 + 逐层剪枝的代表；在它自己的仓库里跑，batch 1 |

注意事项：

- FastV 仓库根目录没有 LICENSE 文件。本项目使用的是自己的重新实现（fastv_llava.py），发表前仍需确认引用与许可。
- 低预算下 FastV 的报告结果弱于 VisionZip（E-VZ-C02，SC）。这正好说明它的骨架上还有可改进、可制造意外的地方，不影响它作为 baseline。

交叉结论：三篇论文的差异是同一问题在“剪枝位置、打分 query、丢弃信息处理、位置编号、预算口径”五个维度上的不同取值。

- **作为论文方法主张**：直接把差异点拼起来（例如 VisionZip 压缩 + LLM 内文本引导剪枝）已有 VScan（ICML 2025）做过，新颖性低。
- **作为第 ④ 步“制造意外实验”的组合实验**：仍然可以做，目的是观察有没有两篇论文都没发现的现象，见 [MODE_C-0001](../MODE_C-0001_视觉token剪枝.md)。

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
