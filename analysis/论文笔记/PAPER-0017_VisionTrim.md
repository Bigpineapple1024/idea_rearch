# VisionTrim：研究分析笔记

> 与本项目最相关的一点：它在 LLaVA-1.5-7B 64 档报告的数字，是“LLM 之前剪一次 + LLM 第 2 层后再剪一次”、按“Retain Averaged”计的结果；它自己的消融里，只在 LLM 之前剪时，GQA、VQAv2 与 VisionZip-64 相当。本项目引用过的说法逐条核对见「本项目引用过的说法：核对」，与本项目的对照见「与当前项目的关系」。没有源码核验补充（代码未下载）。

分析日期：2026-10-30。本笔记由工作流子代理用 Read 工具读 MinerU flash 档的解析全文后写成，从第 1 页读到第 31 页。PDF 不在只读导出（main_337305f）里，本次没有对照 PDF。没有下载任何材料，没有联网，没有运行代码或实验。

## 来源与阅读范围

- 标题：VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration（p.1）。
- 作者/机构：Hanxun Yu、Wentong Li、Xuan Qu（三人共同一作）、Song Wang、Junbo Chen、Jianke Zhu（后两位是通讯作者）；State Key Lab of CAD & CG, Zhejiang University、NUAA、Udeer.ai、Shenzhen Loop Area Institute（p.1，按原文英文名）。资助：国家自然科学基金 62376244（p.10 致谢）。
- 版本与会议：arXiv:2601.22674v3 [cs.CV]，2026-03-28（p.1）；每页页眉写明是 ICLR 2026 的会议论文（p.1 起）。没有联网核对 OpenReview。
- PDF：`papers/VisionTrim-arxiv-2601.22674.pdf`，31 页，7,588,458 字节，SHA-256 `794614de3a8c1d290d7ac12d846cd8765dfd49af805753ef5f507fed03304552`（见[论文笔记 README](README.md)“2026-10-29 下载的十一篇 PDF 的登记”；PDF 不在只读导出里，本次没有复核）。能访问原文不等于获得再分发许可，全文不提交 Git。
- 解析全文：`literature/processed/PAPER-0017/VisionTrim-arxiv-2601.22674.md`，MinerU 4.0.12 flash 档（见 [literature/processed/README](../../literature/processed/README.md)）；101,960 字节，SHA-256 `0d83d80811f6e1082b3a7c27e3a02cdec8816805416fe4a31ee0a35b42175d38`（本次用 Get-FileHash 复核，前 16 位与登记一致）；31 个页标记。同目录没有 README，只有这一个文件。
- 解析的限制：
  - 公式都是图片引用：式 1–5、7、9–12 的内容读不到；式 6、式 8 只有错位的文字，本笔记按上下文读出，以 PDF 为准。解析里只看得到 (6)、(8) 两个式号，其余式号按出现顺序推定。
  - 表格被压成文本。Table 1、4、5、13、15 的列能对上（逐格读过；64 档的平均分我按十个基准对 vanilla 的比值复算为 98.7% 与 94.4%，与表中 98.8%、94.4% 相符）。Table 2 的百分比与 POPE 粘在一起；Table 3（视频）的 Ours 一行错位；Table 7 与 Fig. 4 的文字交错；Table 26 的参数名与取值顺序错位。这些表只引用读得清的数字。
  - 图只有图中的文字（Fig. 1–15），没有看图像。
  - 页码用解析文件的页标记（与 PDF 页号相同）。数字以 PDF 为准（literature/processed/README 的引用规则）。
- 代码：https://github.com/hanxunyu/VisionTrim（p.1 摘要）。代码未下载、未核验（需许可）。代码的许可证全文没有写；Table 27 只列了所用数据集与模型的许可证（p.23）。
- 实际阅读：p.1–31 全部。正文 p.1–10，参考文献 p.10–14，附录 p.15–31（A 个案、B LLM 使用声明、C 计算量估计、D 更多实验、E 影响、F 许可；p.25–31 是 Fig. 9–15 的注意力可视化，只有标题）。
- 标签：SC = 作者主张，DE = 全文里直接写着的设置、表格或数字，AI = 本笔记的推断，UN = 不知道 / 全文没有。DE 不代表我们复现过。

## 研究问题

已有的免训练剪枝多数只管流水线的一段（视觉编码之后，或 LLM 解码之中），而且常常不看文本。能不能做一个统一的免训练框架：在视觉编码阶段和 LLM 解码阶段都压缩视觉 token，先按全局语义与局部连续性选出主要的 token，再用文本引导把被丢掉的 token 合并成补充 token？（p.1 摘要；p.1–2 §1）

## 方法与贡献

- **两个模块**（p.4 Fig. 3；p.3–6 §3）：主导 token 选择 DVTS、文本引导的视觉补充 TGVC。作者说两者都可以插在视觉编码器或 LLM 的任意两层之间（p.2；p.4）。
- **DVTS：每个 token 算一个分，取前 K 个**（p.4–5 §3.1）。
  - 全局分：CLIP 视觉编码器倒数第二层里，[CLS] 对各视觉 token 的注意力，H 个头取平均，再归一化成概率分布（p.4；式 1–4 是图片）。作者说这一做法借自 VisionZip 与 FasterVLM（p.4）。
  - 局部分（LTAM）：每个 token 与 k × k 邻域里的 token 算亲和度，是“特征相似”与“位置接近”两项的加权组合，σ_f、σ_p 是特征差与位置差的标准差；对邻域取平均后归一化（p.5；式 5 是图片）。默认 3 × 3，(w1, w2, w3) = (0.3, 0.3, 0.5)（p.22 Table 26）。三个权重怎样作用在两项上，读不到（UN）。[AI] 按文字描述，与邻居越像的 token 局部分越高。
  - 融合：按方差反比加权，S_i = α·Ŝ^g_i + (1 − α)·S^l_i，α = σ_l² / (σ_g² + σ_l²)（式 6，解析错位，按此读）；取前 K 个作主导 token（p.5）。
- **TGVC：把剩下的 N − K 个合并成 R 个补充 token**（p.5–6 §3.2）。
  - 用 CLIP 的文本编码器得到文本特征 T（L 个 token）；算剩余视觉 token 与 T 的相似度，对 L 个文本 token 取平均，得到每个视觉 token 的分，取前 R 个作聚类中心（p.5 式 7）。
  - 分配：每个剩余 token 对各文本 token 的 softmax 分布，与每个中心对各文本 token 的 softmax 分布做内积，归到内积最大的中心（p.5 式 8–9）。
  - 合并：每簇按文本引导的相似度加权平均（式 10，图片），重复 T 次（p.6）。最后把 K 个主导 token 和 R 个补充 token 拼起来（式 11）。
  - [UN] 与文本比的视觉特征取自哪一层、哪一种（ViT 第 23 层的隐状态，还是投影之后），怎样与 CLIP 文本特征对齐到同一维度，全文没写（正文只写两者都是 d 维）。编码的是整段 prompt 还是只有问题，没写。T 取几，没写（Fig. 6 画了 TGVC×1 与 TGVC×3，p.9）。
- **两阶段（§3.3，p.6）**：
  - 视觉编码阶段：进 LLM 之前，把 N 个视觉 token 压成 K + R 个（p.6）。默认插在 ViT 第 23 层（共 24 层）（p.22 Table 26）。[AI] LLaVA 本来就取倒数第二层的输出，所以这一段相当于在送进投影层的特征上剪，位置与 VisionZip 相同，省不了多少 ViT 的计算。
  - LLM 解码阶段：默认只插在 LLM 第 2 层与第 3 层之间（p.19 D.3.1；p.22 Table 26），位置相当于本项目说的 K = 2。全局分改用“第一个生成 token”对视觉 token 的注意力；TGVC 改用视觉 token 与文本 token 之间的跨模态注意力（α_i 是第 i 个视觉 token 的平均跨模态注意力）；局部分仍用 LTAM（p.6 式 12）。[AI] 预填充时第一个生成 token 还没有生成，能用的只有 prompt 的最后一个位置（它的输出就是第一个生成 token），这与 FastV 用最后一个 token 的注意力是同一种量；Table 9 报告了预填充加速，说明剪枝发生在预填充里。原文没有说清（UN）。
  - 名额：DVTS : TGVC = 3 : 1；LLM 之前的保留率 α1 = 50%；LLM 内的保留率 α2“动态调整”（p.22 Table 26）。α1 相对什么（576 个，还是目标预算），没写（UN；见「预算口径」）。
- **是否免训练**：主结果不训练（标题；摘要；p.2；p.10 §5）。另有微调版 VisionTrim‡：微调 TGVC 中的跨模态注意力模块和投影层，用 1/10 的 LLaVA-1.5 数据（66K 条）（p.17 D.2.1；p.24 F），只出现在 Table 13 与 Fig. 2(c)。[UN] 免训练版里这个“跨模态注意力模块”有没有参数、用什么权重，没写。DINOv2 的实验先把 LLaVA 的视觉编码器换成 DINOv2 并微调 LLaVA，再用 VisionTrim（p.21 D.3.6）。
- **是否用到题目文字**：用到。视觉编码阶段的 TGVC 用 CLIP 文本编码器编码文本指令；LLM 阶段的 TGVC 用 LLM 里的视觉—文本注意力；LLM 阶段 DVTS 用的注意力也随题目变化（AI）。视觉编码阶段的 DVTS 不看题目。每道题都同样处理，不按题型区分（E-VT-18）。作者说：没有题目时退回只用 DVTS 的不看文本模式（Table 10 的“只 DVTS”一行）；题目无关或误导时，相似度普遍很低，中心近乎随机，相当于 ToMe 式的无监督聚类，又因为是合并而不是丢掉，语义能保住（p.9）。后一半没有实验（AI）。
- **作者列出的三点贡献**（p.2）：①统一的免训练视觉 token 压缩框架，覆盖整个 MLLM 流水线；②两个即插即用的模块 DVTS、TGVC，可插在任意两层之间；③在标准与高分辨率、图像与视频的多种 MLLM 上的实验。

## 预算口径

- **“保留 N 个”指什么**：LLaVA-1.5 的表头写“Retain Averaged 192 / 128 / 64 Tokens”（p.6 Table 1；Table 13 同，p.18）。“Averaged”全文没有定义（UN）。
- [AI] 最可能是 32 层平均（与本项目的层平均口径相同），理由：
  1. 同表比较的 VScan 写法相同（“Retain 64 Tokens in Average”），VScan 原文说是在 LLaMA 全部 32 层上平均保留 192、128、64 个视觉 token（PAPER-0014 解析全文 p.8 正文、p.9 Table 2）；
  2. 主方法在 LLM 之前和 LLM 第 2 层后各剪一次，一个数字只能是某种平均；
  3. Table 15 调 α2 来保持“整个 LLM 解码阶段的总保留率 75%”（p.19–20），表中各行满足 (k + (32 − k)·α2) / 32 = 75%，即相对进入 LLM 的个数、按 32 层平均（k = 2 一行例外：按此式 α2 应为 73.3%，表中写 77.3%，可能是笔误或解析误差）。
- **各档的层间日程没有给出**（UN）：进 LLM 时有多少个、第 2 层后剩多少个，全文没有表。Table 11 的 48 + 16 是一次 DVTS + TGVC 的名额，两阶段时每一段怎么分，没写。[AI] 若 α1 = 50% 指 576 的一半（288 个进 LLM），层平均 64 时第 3 层起约留 49 个（(2 × 288 + 30 × n) / 32 = 64）；但这种读法下层平均 16、8、4、1 的档（Table 13）做不到（光前 2 层就已是 2 × 288 / 32 = 18 个），所以 α1 的含义不确定。
- **CLS**：DVTS 用 [CLS] 的注意力打分（p.4）；[CLS] 本身是否送进 LLM、是否占名额，没写（UN）。[AI] Table 11 的 48 + 16 = 64 里没有 [CLS] 的名额；VisionZip 的 64 个里含 [CLS]（54 个主导含 [CLS]、10 个上下文，[PAPER-0006 笔记](PAPER-0006_VisionZip.md) E-VZ-1）。
- **合并 token**：TGVC 的每个簇合成 1 个 token、记 1 个（Table 11 的 16，p.17）；VisionZip 的 10 个上下文 token 也各记 1 个。合并 token 在 LLM 里用什么位置编号，没写（UN）。
- **名额表**（p.17 Table 11–12、D.1）：LLaVA-1.5 的 32 / 64 / 128 / 192 = 24 + 8 / 48 + 16 / 96 + 32 / 144 + 48（3 : 1）；LLaVA-NeXT 的 80 / 160 / 320 / 640 = 70 + 10 / 140 + 20 / 280 + 40 / 560 + 80（7 : 1）；Video-LLaVA 每帧 17 个、8 帧共 136 个。[AI] NeXT 是 7 : 1，与 Table 26 的 3 : 1 和“各模型、各基准超参数一致”（p.21 D.3.7）不符。
- **与 FastV 一族怎么对齐**：LLaVA-1.5-7B 的表里没有 FastV；对照是 SparseVLM、VisionZip、PDrop 与 VScan（p.7 §4.1；复核时改正：原写“SparseVLM、PDrop 与 VScan”，漏了 VisionZip），其中 SparseVLM、PDrop 在 LLM 内剪，VisionZip 在 LLM 之前剪，VScan 两阶段（AI，按各自论文），作者说沿用各自论文的默认设置与评测指标（p.7 §4.1）。实验表格里 FastV 只出现在 Qwen2-VL / Qwen2.5-VL（Table 6，约 1/3 的 token）与 InternVL2-2B（Table 14，削减 70%）上，K 与剪除比例没写（UN）；Fig. 2(b)(c) 的曲线图例在解析里读不到，图中有没有 FastV 不知道（UN）。[AI] 对照方法的 token 数沿用各自论文的计法：VScan、SparseVLM 是层平均，VisionZip 每层都是 N 个。
- **FLOPs 与时延的口径**：
  - C.1 给了“第 K 层之后”的 FLOPs 估算式，C.2 给了理论削减比（都是图片，读不到；p.16–17）。
  - Table 8（LLaVA-NeXT-7B，SQA）报 CUDA 时间、FLOPs、KV cache；Table 9（单张 A100，POPE）报总推理时间、预填充时间、FLOPs；Table 20（A40，LLaVA-NeXT-7B / 13B）报预填充毫秒数、总时间、显存。
  - FLOPs 是否算进视觉编码器、CLIP 文本编码器与 TGVC 的聚类，批大小、解码方式，都没写（UN）。Table 8 的硬件没写（UN）。
- **几处对不上的地方**（AI）：
  - Table 8：同为 320 个，VisionTrim 的 FLOPs 是 VisionZip 的一半（0.8 对 1.6 T），作者写同样 token 数下比 VisionZip 少 50% 计算（p.9）。同样的层平均 token 数下，LLM 的计算应当相近；差一倍说明两边的“320”不是同一个计算口径，或 FLOPs 另有算法。
  - KV 粗算：每个 token 的 KV 约为 32 层 × K、V × 4096 维 × 2 字节 ≈ 0.524 MB。vanilla 的 303.6 MB 约合 579 个、LLaVA-NeXT 的 1512.1 MB 约合 2884 个、SparseVLM-320 的 168.0 MB 约合 320 个，都对得上“只计视觉 token、按层平均”的假设。照此，Table 4 里三种都写 64 个的设置，KV 约合 48 个（只在 ViT，25.4 MB）、83 个（只在 LLM，43.5 MB）、58 个（两阶段，30.2 MB）的层平均；Table 8 中 VisionZip-320 约合 344 个，VisionTrim-320 约合 194 个。KV 的测法没写，这只是粗算。
  - 同一个 LLaVA-NeXT-7B，vanilla 的 FLOPs 在 Table 8 是 9.6 T、在 Table 9 是 12.6 T；VisionTrim-320 是 0.8 T 与 1.2 T。基准不同、文本长度不同可能有差别，原文没说明。

## 结果及不确定性

### LLaVA-1.5-7B 主表（Table 1，p.6）

| 档 | 方法 | GQA | MMB | MME | POPE | SQA | VQAv2 | TextVQA | SEED | MMVet | VizWiz | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 576 | vanilla | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 78.5 | 58.2 | 58.6 | 31.1 | 50.1 | 100.0% |
| 192 | SparseVLM | 57.6 | 62.5 | 1721 | 83.6 | 69.1 | 75.6 | 56.1 | 55.8 | 30.2 | 50.0 | 96.4% |
| 192 | VisionZip | 59.3 | 63.0 | 1783 | 85.3 | 68.9 | 76.8 | 57.3 | 56.4 | 31.7 | 50.5 | 98.3% |
| 192 | PDrop | 57.3 | 63.3 | 1797 | 84.8 | 69.2 | 76.4 | 56.5 | 57.2 | 30.8 | 50.0 | 97.6% |
| 192 | VScan | 60.6 | 63.9 | 1806 | 86.2 | 68.6 | 77.8 | 57.7 | — | — | 50.4 | 98.9% |
| 192 | VisionTrim | 61.0 | 64.4 | 1798 | 86.8 | 70.8 | 78.4 | 58.4 | 58.3 | 33.2 | 51.3 | 100.6% |
| 128 | SparseVLM | 56.0 | 60.0 | 1696 | 80.5 | 67.1 | 73.8 | 54.9 | 53.4 | 30.0 | 51.0 | 94.2% |
| 128 | VisionZip | 57.6 | 62.0 | 1762 | 83.2 | 68.9 | 75.6 | 56.8 | 54.9 | 32.6 | 50.0 | 97.2% |
| 128 | PDrop | 57.1 | 61.6 | 1761 | 82.6 | 68.4 | 76.0 | 56.6 | 56.2 | 31.0 | 49.6 | 96.5% |
| 128 | VScan | 59.8 | 63.0 | 1792 | 86.1 | 68.9 | 77.1 | 57.3 | — | — | 51.7 | 98.6% |
| 128 | VisionTrim | 60.3 | 64.0 | 1788 | 86.6 | 69.7 | 78.2 | 58.2 | 57.5 | 32.7 | 51.4 | 99.9% |
| 64 | SparseVLM | 52.7 | 56.2 | 1505 | 75.1 | 62.2 | 68.2 | 51.8 | 51.1 | 23.3 | 49.6 | 86.7% |
| 64 | VisionZip | 55.1 | 60.1 | 1690 | 77.0 | 69.0 | 72.4 | 55.5 | 52.2 | 31.7 | 51.9 | 94.4% |
| 64 | PDrop | 47.5 | 58.8 | 1561 | 76.2 | 69.0 | 73.3 | 50.6 | 53.0 | 30.5 | 50.2 | 90.8% |
| 64 | VScan | 58.3 | 62.1 | 1698 | 85.0 | 69.1 | 75.4 | 55.6 | — | — | 51.8 | 96.8% |
| 64 | VisionTrim | 58.8 | 63.0 | 1780 | 86.2 | 71.0 | 76.8 | 56.8 | 54.8 | 32.2 | 52.3 | 98.8% |

- 表头是“Retain Averaged N Tokens”，VisionTrim 一行是两阶段的结果（Table 4 的“两阶段”一行与 64 档相同）。表里没有 FastV、DivPrune（E-VT-17）；VScan 没有 SEED、MMVet。
- 平均分是十个基准对 vanilla 比值的平均（复算见「来源与阅读范围」）。VScan 的平均只有八个基准，与其他行不能直接比（AI）。
- 作者的说法：三档都超过之前的方法；POPE、SQA、TextVQA 上不降反升（p.7）。（复核时补：按 Table 1，POPE、SQA 三档都高于 vanilla；TextVQA 只有 192 档高于 vanilla（58.4 对 58.2），128 档持平（58.2），64 档低（56.8）。）

**VisionTrim − VisionZip（同表相减，AI 算术）**

| 档 | GQA | MMB | MME | POPE | SQA | VQAv2 | TextVQA | SEED | MMVet | VizWiz | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 192 | +1.7 | +1.4 | +15 | +1.5 | +1.9 | +1.6 | +1.1 | +1.9 | +1.5 | +0.8 | +2.3 |
| 128 | +2.7 | +2.0 | +26 | +3.4 | +0.8 | +2.6 | +1.4 | +2.6 | +0.1 | +1.4 | +2.7 |
| 64 | +3.7 | +2.9 | +90 | +9.2 | +2.0 | +4.4 | +1.3 | +2.6 | +0.5 | +0.4 | +4.4 |

### 增益从哪里来：64 档的分解（Table 1、4、7、10）

| 设置（都写 64 个） | GQA | MMB | POPE | VQAv2 | KV cache（MB） | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| vanilla（576 个） | 61.9 | 64.7 | 85.9 | 78.5 | 303.6 | Table 1、4 |
| VisionZip-64（引用的数字） | 55.1 | 60.1 | 77.0 | 72.4 | — | Table 1 |
| 只在 ViT，只 DVTS | 52.8 | 56.9 | 76.1 | 68.6 | 25.4 | Table 4 |
| 只在 ViT，DVTS + TGVC | 55.6 | 60.2 | 80.2 | 72.2 | （同组） | Table 4（Table 7 的 64 档同） |
| 只在 LLM，只 DVTS | 52.0 | 57.4 | 75.8 | 70.2 | 43.5 | Table 4 |
| 只在 LLM，DVTS + TGVC | 54.7 | 61.1 | 79.2 | 73.6 | （同组） | Table 4 |
| 两阶段，只 DVTS（推断为两阶段） | — | — | 84.6 | — | — | Table 10 |
| 两阶段，只 TGVC（推断为两阶段） | — | — | 85.0 | — | — | Table 10 |
| 两阶段，DVTS + TGVC | 58.8 | 63.0 | 86.2 | 76.8 | 30.2 | Table 1、4、9、10 |

- KV 值在解析里只写在每组的“只 DVTS”一行，是两行共用还是只属于这一行，看不出（UN）。Table 10 的基准没写，vanilla 的分数与时间与 Table 9（POPE）相同，推断为 POPE（AI）；作者说 Table 10 是在最终模型上加不同组件（p.21 D.3.8），所以推断为两阶段。
- 只在 LLM 之前用时：DVTS + TGVC 与 VisionZip-64 在 GQA、VQAv2、MMB 上相当（+0.5、−0.2、+0.1），POPE 高 3.2。
- 加上 LLM 第 2 层后的第二次剪枝：GQA 55.6 → 58.8、VQAv2 72.2 → 76.8、MMB 60.2 → 63.0、POPE 80.2 → 86.2。
- [AI] 所以 VisionTrim 在 GQA、VQAv2 上比 VisionZip 高的 3～4 分，大部分来自两阶段的层间分配，不是来自文本引导的合并本身。限制：这是跨表比较；VisionZip 的数字是引用的（E-VT-8）；三种设置的 KV 不同（E-VT-12），实际预算可能不完全相同。
- 文本引导的合并本身（Table 7，只在 LLM 之前，p.8、p.9）：加 TGVC 后 POPE +3.1 / +3.3 / +4.1 / +4.4，MMB +2.3 / +2.4 / +3.3 / +4.2，TextVQA +2.5 / +2.8 / +3.4 / +4.0（192 / 128 / 64 / 32 个），token 越少增益越大。对照是只用 DVTS；这时是 64 个主导 token，还是去掉 TGVC 的 16 个后只剩 48 个，没写（UN）。Table 4 的 KV 粗算约合 48 个（E-VT-12），所以后一种可能存在；若是 48 个，TGVC 的“增益”里含 16 个名额的作用（AI）。
- 只在 LLM 之前时与 VisionZip 比（Table 7 对 Table 1）：POPE +0.8 / +1.5 / +3.2，MMB +0.4 / +0.9 / +0.1，TextVQA +0.5 / +0.4 / +0.5（192 / 128 / 64）。
- [AI] 只用 DVTS（不合并被丢掉的 token）在 LLM 之前时低于 VisionZip（64 档 GQA 52.8 对 55.1、VQAv2 68.6 对 72.4）；加上合并后才与 VisionZip 相当。VisionZip 也把其余 token 合并（上下文 token）。这提示“合并被丢掉的 token”占了很大一部分作用，文本引导在其上再加的不多。没有“不看文本的合并”这一对照，分不开（UN）。

### 其他消融（LLaVA-1.5-7B）

- **DVTS 的融合方式**（Table 5，p.7；64 档）：GQA / MMB / MME / POPE / SQA / TextVQA，只用 [CLS] 52.8 / 55.3 / 1536 / 74.2 / 67.9 / 51.2；逐元素最大 53.4 / 58.7 / 1702 / 77.6 / 70.0 / 52.4；几何平均 55.2 / 56.5 / 1631 / 80.2 / 71.1 / 54.3；方差加权 58.8 / 63.0 / 1780 / 86.2 / 71.0 / 56.8。[AI] 方差加权一行与 Table 1 的 64 档相同，所以这是在两阶段全流程上的消融。局部项让 GQA 高 6.0、POPE 高 12.0，效应很大，原文只用 Fig. 5 的可视化来解释（p.9）。
- **插入层**（Table 15，p.18；α1 = 50%，LLM 内总保留率 75%）：k = 2 / 8 / 12 / 16 / 20 / 24 时，GQA 60.4 / 60.2 / 60.5 / 58.4 / 57.3 / 56.5，MMB 64.2 / 64.0 / 63.8 / 62.9 / 63.3 / 60.5，POPE 86.2 / 86.0 / 85.5 / 84.4 / 84.1 / 83.2。作者：插在浅层通常更好（p.20）。没有 k = 1。这个消融不是 64 档。
- **更少的 token**（Table 13，p.18）：层平均 16 / 8 / 4 / 1 个时，平均 90.0% / 87.4% / 84.5% / 82.8%（免训练），微调版 94.3% / 91.8% / 88.0% / 86.5%。16 个时 GQA 51.5、VQAv2 68.4、POPE 72.6。[AI] 按层平均计，“1 个”不一定只有 1 个 token 进 LLM，日程没写。
- **32 个时与随机剪比**（Table 24，p.22）：GQA / MMB / POPE / SQA / VQAv2，随机 40.8 / 51.4 / 62.8 / 60.5 / 56.7，VisionTrim 55.3 / 60.2 / 77.5 / 70.1 / 72.6。
- **DINOv2 编码器**（Table 25，p.22；32 个）：SEED 60.6 → 58.5、MMB 64.2 → 63.0、MME 1404 → 1368、POPE 86.7 → 83.4、SQA 70.0 → 70.7。
- **“直接文本引导”对照**（Table 23，p.22；LLaVA-1.5-13B，32 个）：POPE 59.5 对 84.4，MME 1240 对 1685，OCRBench 154 对 297，TextVQA 47.4 对 57.6。这个对照怎样实现，没写（UN）。
- **指代物体分类 ROC**（Table 18，p.19）：Box 一项 vanilla 55.4、VisionZip 40.4、VisionTrim 55.1；token 数没写（UN）。
- **个案**（Fig. 7，p.15–16）：64 个时，问木凳颜色、问左边那瓶饮料陈酿了多久两例（原文是 drink；复核时改正：原写“问酒的年份”），原模型（576 个）答对，VisionZip 与只用 DVTS 答错，加 TGVC 后改对（这两例的对错正文 p.15 明写了）；第 3 例（问女士在做什么）三种设置都答 Cooking，与 576 个时的回答相同；第 4 例（32 个，问汽水品牌）三种设置都答 Drink，576 个时答 Coca-Cola（复核时改正：原写“第 3 例三者都对；第 4 例三者都错”。正文只明写了前两例的对错，第 3、4 例的红字标注在解析里看不到，这里只按回答文字说）。作者称之为缓解“知识边界漂移”（p.15–16）。是挑出来的 4 个例子，不是统计。

### 效率（LLaVA-1.5-7B）

- Table 9（p.10，单张 A100，POPE）：vanilla 总时间 1303 s、预填充 494 s、3.8 T、85.9；SparseVLM-64 1068 s（1.22×）、377 s（1.31×）、1.3 T、75.1；VisionTrim-64 685 s（1.90×）、235 s（2.10×）、0.8 T、86.2。
- Table 10（p.16）：只 DVTS 670 s / 227 s / 84.6；只 TGVC 741 s / 263 s / 85.0；两者 685 s / 235 s / 86.2。作者说只用 TGVC 更慢，是因为聚类要算更多相似度（p.21 D.3.8）。
- [UN] CLIP 文本编码器的开销是否算在内，没写。LLaVA-1.5 本身不带 CLIP 的文本塔（AI）。

### 其他模型（只列总的情况）

- LLaVA-NeXT-7B（Table 2，p.7）：640 / 320 / 160 个时平均 99.9% / 97.0% / 94.0%，VisionZip 97.8% / 94.1% / 90.7%。Table 8（p.9）：320 个时 CUDA 时间 26:34 → 10:16，FLOPs 9.6 → 0.8 T，KV 1512.1 → 101.8 MB，SQA 70.2 → 69.6。
- Video-LLaVA-7B（Table 3，p.7）：136 个时平均 98.0%，VisionZip 91.3%，SparseVLM 86.5%（Ours 一行解析错位，各项数字没有逐一核对）。
- Qwen2-VL-7B、Qwen2.5-VL-7B（Table 6，p.8；约 1/3 的 token）：Qwen2-VL 上 MMB 82.8（vanilla 80.7）、POPE 86.3（86.4），FastV 77.8、82.0；Qwen2.5-VL 上 MMB 83.2（83.5）、RefCOCO 86.8（89.5），FastV 77.5、73.6。
- InternVL2-2B（Table 14，p.18；削减 70%）：OCRBench 75.0（vanilla 75.6），FastV 68.8，TopV 72.9。
- LLaVA-1.5-13B 读字类（Table 21，p.20；32 个）：平均 96.3%，VisionZip 71.4%；推理 704 s，vanilla 2394 s。
- 另有 LLaVA-OneVision-7B 与 Qwen2-VL 的多图、视频基准（Table 16）、BLIP2 的描述与检索（Table 17、19）、视频帧间压缩（Table 22）、A40 上 LLaVA-NeXT-7B / 13B 的效率（Table 20）。

### 不确定性

- 所有表都是单个数字：没有重复运行、方差、区间或种子；解码方式没写（E-VT-18）。
- VQAv2、GQA 用哪个划分，没写（UN）；vanilla 的 78.5、61.9 与 VisionZip 表中的 vanilla 相同。
- POPE 的指标：Table 9 的表头写 Accuracy；但 vanilla 85.9 与本项目实测的 POPE F1 均值 85.90 相同，本项目 vanilla 的准确率均值是 86.87（EXP-0002-R04，见 [PAPER-0008 笔记](PAPER-0008_CLASP.md)），所以很可能是三个子集 F1 的平均（AI）。
- MMVet 由 GPT 打分、题少，相差 1 分以内不宜当作差别（AI）。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-VT-1 | DE | DVTS：CLIP 倒数第二层 [CLS] 注意力（头平均、归一化）+ LTAM 局部亲和度（3 × 3，w = 0.3 / 0.3 / 0.5），按方差反比融合，取前 K 个 | p.4–5 §3.1、式 6；p.22 Table 26 | 式 1–5 是图片，亲和度的具体形式与三个权重的作用读不到 |
| E-VT-2 | DE | TGVC：CLIP 文本编码器；与文本平均相似度最高的 R 个作中心；按“经由文本”的相似度分配；加权平均合并，迭代 T 次 | p.5–6 §3.2、式 7–11 | 视觉特征取自哪层、怎样与文本特征对齐、T 取几，都没写（UN） |
| E-VT-3 | DE | 两阶段：ViT 第 23 层（共 24）+ LLM 第 2 层与第 3 层之间；LLM 阶段改用“第一个生成 token”的注意力与跨模态注意力；DVTS : TGVC = 3 : 1；α1 = 50%，α2 动态 | p.6 §3.3；p.19 D.3.1；p.22 Table 26 | α 相对什么没写；“第一个生成 token”在预填充时指什么没说清 |
| E-VT-4 | DE | 名额：LLaVA-1.5 的 64 = 48 + 16（128 = 96 + 32，192 = 144 + 48，32 = 24 + 8）；NeXT 的 320 = 280 + 40；视频每帧 17 个 | p.17 Table 11–12、D.1 | 两阶段时每一段怎么分没写；NeXT 是 7 : 1，与“超参数一致”（p.21）不符（AI） |
| E-VT-5 | AI | “Retain Averaged N”是 32 层平均 | p.6 Table 1；p.18 Table 15、p.19–20；PAPER-0014 解析全文 p.8 正文、p.9 Table 2 | 全文没有定义；依据是 VScan 的同类写法与 Table 15 的算式 |
| E-VT-6 | UN | 各档进 LLM 的个数、第 2 层后的个数；α1 的基数；[CLS] 是否保留；合并 token 的位置编号；KV 与 FLOPs 的测法 | p.6；p.16–17；p.22 | 影响与 FastV 一族、VisionZip 的对齐 |
| E-VT-7 | DE | LLaVA-1.5-7B 主表：64 档 VisionTrim GQA 58.8、VQAv2 76.8、POPE 86.2、平均 98.8%；VisionZip 55.1、72.4、77.0、94.4% | p.6 Table 1 | 单次，无区间；VQAv2、GQA 划分没写 |
| E-VT-8 | AI | 对照数字是引用的：VisionZip 三档在共有列上与 VisionZip CVPR Table 1 逐格相同；SparseVLM 与该表的 SparseVLM(2024.10) 一行相同（192 档 MMVet 一格除外：30.2 对 31.5）；VScan 与 VScan Table 2 相同；VisionZip 的 VizWiz 在 VisionZip 正文与补充材料的免训练结果里没有找到 | p.6 Table 1；p.7 §4.1；PAPER-0006 解析全文 p.5 Table 1；PAPER-0014 解析全文 p.9 Table 2 | 原文只说沿用各自论文的设置与指标；SparseVLM 引的是 ICML 2025 版（p.14），数字却是 2024.10 版；PDrop 的 POPE、VQAv2 与 VScan 表中的 PDrop 不同（64 档 76.2 对 55.9、73.3 对 69.2），来源没写 |
| E-VT-9 | DE | 64 档分解：只在 ViT（DVTS 52.8 / 56.9 / 76.1 / 68.6，加 TGVC 55.6 / 60.2 / 80.2 / 72.2）、只在 LLM（52.0 / 57.4 / 75.8 / 70.2，加 TGVC 54.7 / 61.1 / 79.2 / 73.6）、两阶段 58.8 / 63.0 / 86.2 / 76.8（GQA / MMB / POPE / VQAv2）；KV 25.4 / 43.5 / 30.2 MB | p.7 Table 4；p.8 §4.3 | 单次；三组的 KV 不同（E-VT-12） |
| E-VT-10 | AI | 只在 LLM 之前时，VisionTrim 与 VisionZip-64 在 GQA、VQAv2、MMB 上相当（+0.5、−0.2、+0.1），POPE +3.2；比 VisionZip 高的 3～4 分主要来自加上 LLM 内的第二次剪枝 | p.7 Table 4 对 p.6 Table 1 | 跨表比较；VisionZip 数字是引用的；各行实际预算可能不同 |
| E-VT-11 | DE | 只在 LLM 之前，加 TGVC 的增益随 token 减少变大：POPE +3.1 ～ +4.4，MMB +2.3 ～ +4.2，TextVQA +2.5 ～ +4.0（192 → 32 个） | p.8 Table 7；p.9 | 对照“只 DVTS”是 64 个还是 48 个没写；没有“不看文本的合并”对照 |
| E-VT-12 | AI | KV 粗算：Table 4 的三个“64”约合 48 / 83 / 58 个 token 的层平均；Table 8 的 SparseVLM、VisionZip、VisionTrim 的“320”约合 320 / 344 / 194 个 | p.7 Table 4；p.9 Table 8 | 假设只计视觉 token、fp16、每 token 0.524 MB；该假设在 vanilla（约 579、2884 个）与 SparseVLM-320 上吻合；测法没写 |
| E-VT-13 | DE | DVTS 融合方式消融：只用 [CLS] GQA 52.8、POPE 74.2；方差加权 58.8、86.2 | p.7 Table 5；p.8–9 | 方差加权一行即 Table 1 的 64 档（AI：是两阶段全流程上的消融）；局部项的效应很大，没有解释 |
| E-VT-14 | DE | 插入层消融（k = 2 ～ 24），插在浅层更好；没有 k = 1 | p.18 Table 15；p.19–20 D.3.1 | 不是 64 档；k = 2 一行的 α2（77.3%）与算式（73.3%）不符（AI） |
| E-VT-15 | DE | 效率（单张 A100，POPE）：总时间 1303 → 685 s（1.90×），预填充 494 → 235 s（2.10×），FLOPs 3.8 → 0.8 T；只 DVTS 670 s，只 TGVC 741 s | p.10 Table 9；p.16 Table 10；p.21 D.3.8 | 批大小、解码方式没写；CLIP 文本编码器是否计入没写 |
| E-VT-16 | AI | 内部不一致：NeXT vanilla 的 FLOPs 9.6 T（Table 8）对 12.6 T（Table 9），VisionTrim-320 0.8 T 对 1.2 T；NeXT 的 POPE，SparseVLM-320 78.6（Table 2）对 78.5（Table 9），VisionTrim-320 83.6 对 84.8；3 : 1 对 NeXT 的 7 : 1；同为 320 个 FLOPs 差一倍 | p.7 Table 2；p.9 Table 8；p.10 Table 9；p.17 Table 12；p.21 D.3.7 | 基准不同可能解释 FLOPs 的一部分差别，原文没说明 |
| E-VT-17 | DE | LLaVA-1.5-7B 的表里没有 FastV、DivPrune；参考文献里没有 DivPrune；实验表格里 FastV 只在 Table 6（Qwen2-VL / Qwen2.5-VL）与 Table 14（InternVL2-2B）出现，设置没写（复核时补：Fig. 2(b)(c) 图例读不到，图中有没有不知道） | 全文 p.1–31；参考文献 p.10–14；p.8 Table 6；p.18 Table 14 | 通读判断，并检索核对过 DivPrune、diversity 等词 |
| E-VT-18 | DE | 没有按题型的分析（全文未见 yes/no、answer type、counting、how many 等）；没有重复运行、方差或区间；VQAv2 与 GQA 的划分、解码方式没写 | 全文 p.1–31 | 通读判断，并检索核对 |
| E-VT-19 | AI | POPE 的指标很可能是 F1（三个子集平均），不是表头写的 Accuracy | p.6 Table 1；p.10 Table 9；[PAPER-0008 笔记](PAPER-0008_CLASP.md)（EXP-0002-R04） | 依据：vanilla 85.9 等于本项目实测 F1 85.90，本项目准确率是 86.87 |
| E-VT-20 | SC | 没有题目时退回只用 DVTS（POPE 84.6 对 86.2）；题目无关或误导时退化成无监督聚类，因为是合并，语义能保住 | p.9 §4.3；p.16 Table 10 | 后半句没有实验（AI） |
| E-VT-21 | DE | “直接文本引导”对照在 LLaVA-1.5-13B、32 个时 POPE 59.5（VisionTrim 84.4）、MME 1240（1685） | p.20 D.3.4；p.22 Table 23 | 对照怎样实现没写（UN） |
| E-VT-22 | DE | 个案 4 例：颜色题、陈酿年数题上加 TGVC 后回答与原模型一致（正文称原模型答对）；第 3 例三者回答与原模型相同；第 4 例（32 个）三者都答 Drink，原模型答 Coca-Cola（复核时改正：原写“年份题”“第 4 例三者都错”；第 3、4 例的对错标注解析里读不到） | p.15 Fig. 7；p.15–16 附录 A | 挑选的个案，不是统计 |
| E-VT-23 | DE / SC | 作者自述局限：64 档保留 98.8%，仍不是无损；部署到高风险应用前要做个案测试；错误输出可能造成误导或安全问题 | p.10 §5；p.16 附录 A；p.23–24 附录 E | 没有讨论预算口径、对照来源、文本编码器的开销（AI） |
| E-VT-24 | DE | 代码链接 https://github.com/hanxunyu/VisionTrim；代码许可证没写；Table 27 列出所用数据集与模型的许可证；LLM 只用于润色语言（附录 B） | p.1；p.16 附录 B；p.23 Table 27 | 代码未下载、未核验（需许可） |
| E-VT-25 | DE | 微调版 VisionTrim‡：微调 TGVC 的跨模态注意力模块与投影层，用 1/10 的 LLaVA-1.5 数据（66K 条）；16 个时平均 90.0% → 94.3% | p.17–18 D.2.1、Table 13；p.24 附录 F | 主表是免训练版 |

没有源码核验：代码未下载、未核验（需许可）。

## 限制与替代解释

- **三件事一起变，主表分不开**：VisionTrim 同时改了选择分数（加 LTAM）、被丢 token 的处理（文本引导的合并）和层间分配（两阶段）。主表只报三者合起来的结果；Table 4 显示层间分配占大头（E-VT-10，AI）。所以“文本引导让 VisionZip 式选择高 3～4 分”的读法不对。
- **预算口径不透明**：“Averaged”没有定义（E-VT-5）；各档的层间日程没有给（E-VT-6）；同写 64 个的三种设置，KV 粗算差别不小（E-VT-12）；同为 320 个时 FLOPs 是 VisionZip 的一半（E-VT-16）。与每层都留 N 个的 VisionZip 比较时，“同一预算”只在层平均意义上成立（AI）。
- **对照的来源**：VisionZip、SparseVLM、VScan 的数字与各自（或 VisionZip）论文的表逐格相同，应是引用而不是重跑（E-VT-8，AI）。引用的数字与 VisionTrim 自己的数字不一定出自同一套评测代码。SparseVLM 用的是旧版的数字。
- **缺强对照**：LLaVA-1.5 上没有 FastV，也没有 DivPrune 等多样性方法（E-VT-17）。
- **与 VScan 的区别可能比作者说的小**：作者批评 VScan 在 LLM 阶段直接用视觉 token 与最后一个指令 token 之间的注意力（p.1）；VisionTrim 在 LLM 阶段用的“第一个生成 token”的注意力，在预填充时也只能取 prompt 的最后一个位置（AI，E-VT-3）。
- **统计**：单次、无区间；MMVet 等小基准的差别不稳（AI）。
- **文本编码器**：要另加 CLIP 的文本塔；视觉特征怎样与它对齐没写（E-VT-2）。
- **LTAM 的效应偏大**：局部项让 GQA 高 6.0、POPE 高 12.0（Table 5），而它只是 3 × 3 邻域里的相似度平均；没有消融说明它留下了哪些 token，只有 Fig. 5 的可视化（AI）。
- **多轮对话**：作者批评直接用文本引导的方法妨碍多轮对话（p.3；p.20）。TGVC 的补充 token 也随题目变化，多轮时是否要重算、开销多大，没有测（AI）。
- **其他内部不一致**（AI）：Table 15 k = 2 一行的 α2；NeXT 的 7 : 1 与 3 : 1；Table 8 与 Table 9 的 FLOPs；NeXT 的 POPE 在 Table 2 与 Table 9 不同（E-VT-16）。
- **作者自述的局限**（p.10 §5）：64 档保留 98.8%，仍不是无损，今后想做无损的方法。附录 A（p.16）说部署到高风险应用（如 AI 辅助诊断）前要做个案测试；附录 E（p.23–24）说错误输出可能造成误导或安全问题，建议加安全规程。

## 本项目引用过的说法：核对

行号按只读快照 main_337305f。

| # | 本项目的说法（出处） | 核对结果 | 依据（全文位置）与说明 |
| --- | --- | --- | --- |
| 1 | VisionTrim 是 ICLR 2026（[IDEA-0006](../IDEA-0006_强底座上的定位后覆盖.md) 第 73 行；EXP-0028 记录第 111 行；实验室会话交接第 120 行） | 成立 | 页眉写明 ICLR 2026 会议论文（p.1 起）；arXiv v3，2026-03-28（p.1）。没有联网核对 OpenReview |
| 2 | 免训练、即插即用（IDEA-0006 第 73 行） | 成立 | 标题、摘要、p.2 贡献②。另有微调版 VisionTrim‡（p.17），主表不用 |
| 3 | 在 VisionZip 式选择上加文本引导的合并（index 第 20 行；实验室会话交接第 120 行；EXP-0028 记录第 111 行） | 部分成立 | 选择是 [CLS] 注意力（作者说借自 VisionZip 与 FasterVLM，p.4）再加 LTAM 局部亲和度、按方差反比融合（p.4–5），不只是 VisionZip 式；文本引导的合并成立（p.5–6）。漏了一点：主结果还在 LLM 第 2 层后再剪一次（p.6 §3.3；p.19；p.22 Table 26），而这一段占了相对 VisionZip 增益的大头（E-VT-10） |
| 4 | 主导 token 按倒数第二层的 CLS 注意力加局部亲和度选（IDEA-0006 第 73 行） | 成立 | p.4 §3.1；p.5 式 6；p.22 Table 26（3 × 3 邻域） |
| 5 | 其余 token 与文本比相似度，取最相似的作聚类中心，把其余合并进去（IDEA-0006 第 73 行；EXP-0028 记录第 111 行） | 成立 | p.5 §3.2 式 7–10：中心是与 CLIP 文本特征平均相似度最高的 R 个；分配按“经由文本”的相似度（式 8），合并是加权平均，迭代 T 次 |
| 6 | 不围框，中心可以散在全图（IDEA-0006 第 73 行） | 成立（推断） | 方法里没有区域或框，中心在全部剩余 token 中按文本相似度取（p.5）；LTAM 的 3 × 3 邻域只用于打分 |
| 7 | 64 个 token 时 GQA 58.8、VQAv2 76.8，同表 VisionZip 55.1、72.4（IDEA-0006 第 73 行；EXP-0028 记录第 111 行） | 成立 | p.6 Table 1。条件：表头是“Retain Averaged 64 Tokens”，VisionTrim 一行是两阶段的结果；VisionZip 的数字与 VisionZip 论文 Table 1 相同（引用，E-VT-8） |
| 8 | 报告比 VisionZip 高 3～4 分（IDEA-0006 第 77、83 行；EXP-0028 记录第 259 行、squeeze 第 92 行；version-evolution 中 DEC-0048 ⑤；实验室会话交接第 120 行） | 部分成立 | 只在 64 档的 GQA、VQAv2 上成立：+3.7、+4.4（平均分 +4.4；POPE +9.2，TextVQA +1.3，MMVet +0.5）。128 档 +2.7、+2.6，192 档 +1.7、+1.6。条件：两阶段、层平均（推断）；VisionZip 数字是引用的；是否全量、哪个划分都没写（复核时改正：原写“全量基准”，全文没有说用全量还是子集）；单次。只在 LLM 之前用时（Table 4 对 Table 1）：GQA +0.5、VQAv2 −0.2、MMB +0.1、POPE +3.2 |
| 9 | 页面没有给代码地址；要复现 VisionTrim 但没有代码地址（IDEA-0006 第 73、83 行） | 与 v3 不符 | v3 摘要给了 https://github.com/hanxunyu/VisionTrim（p.1）。当时读的 HTML 是哪个版本，不知道（UN）。代码仍未下载、未核验（需许可） |
| 10 | VisionTrim 属于“逐 token 文本相关性”一类（version-evolution 中 DEC-0046 ②；论文主张与证据草稿 CL-15；论文素材库 MAT-0032） | 成立，需补一句 | 中心按逐 token 的文本相似度取（p.5）；但它把其余 token 合并进来，不只是“留下最相似的”。本项目的“不围框”对照只留与题目最相似的 16 个、不合并，与它不同 |
| 11 | GQA testdev 上本项目的 VisionZip-64 为 55.18，与 VisionTrim 表中的 55.1 相近（IDEA-0006 第 83 行） | 部分成立 | 55.1 在 Table 1（p.6），但它与 VisionZip 论文自己的数字相同，是引用的；所以这是在与 VisionZip 原文比，不是与 VisionTrim 的重跑比。VisionTrim 的 GQA 划分没写 |
| 12 | EXP-0028 的 +0.5～+1.4 远小于 VisionTrim 报告的 +3～4（EXP-0028 记录第 259 行、squeeze 第 92 行；DEC-0048 ⑤） | 部分成立 | 对两阶段的数字成立。与 EXP-0028 设置相同的“只在 LLM 之前”一行（Table 4）只有 +0.5 / −0.2，与 EXP-0028 同一量级（评测集不同，AI）。这不改变 DEC-0054 的结论（确认集上不成立） |

小结：会议、做法的大体描述、64 档的四个数字都对；“高 3～4 分”要加条件（64 档、GQA 与 VQAv2、两阶段的层平均预算），而且这一增益主要不来自文本引导的合并；“没有代码地址”与 v3 不符。

## 与当前项目的关系

本项目的事实来自本任务给出的项目摘要，以及只读快照中的[论文素材库](../论文素材库.md)（MAT-0029～MAT-0035）、[已发现的实验现象](../已发现的实验现象.md)（F67～F71）、[IDEA-0003](../IDEA-0003_按题目句式路由.md)、[IDEA-0006](../IDEA-0006_强底座上的定位后覆盖.md) 与 EXP-0032 记录。F72～F74 与 DEC-0061 在只读快照里还没有记录，相关数字照任务给出的摘要写。

### 逐项对照

| 维度 | VisionTrim（全文位置） | 本项目 | 判断（AI，理由） |
| --- | --- | --- | --- |
| 剪在哪里 | 两次：ViT 第 23 层（LLM 之前）+ LLM 第 2 层与第 3 层之间（p.6；p.19；p.22 Table 26） | FastV 一族在 LLM 第 1 层（K = 1）或第 2 层（K = 2）之后剪一次；VisionZip-64、DivPrune-64 在 LLM 之前剪一次 | VisionTrim 的 LLM 内插入点相当于 K = 2；它没有试 k = 1（Table 15） |
| 预算口径 | “Retain Averaged N”，没有定义，最可能是 32 层平均（E-VT-5） | 32 层平均：K = 1 时第 0 层 576 个、之后 47 个；K = 2 时前 2 层 576 个、之后 30 个 | 口径可能相同，但它的层间日程不明（E-VT-6）；我们的日程是明写的 |
| 前几层看多少 | 进 LLM 之前已剪过（α1 = 50%，基数不明），没有哪一层看全部 576 个（AI） | K = 1：第 0 层看全部 576 个 | 不同。本项目 F70：第 0 层看全部之后，留哪 47 个影响很小；VisionTrim 的 Table 5：在它的流程里，选择分数影响很大（GQA 52.8 对 58.8）。两者可能不矛盾：没有一层看全部时，选择更要紧（AI，未检验） |
| 选 token 的分数 | ViT 阶段：[CLS] 注意力 + LTAM；LLM 阶段：第一个生成 token 的注意力 + LTAM（p.4–6） | 强对照 A：FastV 末 token 注意力去位置先验 + 6×6 分块轮流取 + 排除填充区；DivPrune：投影层输出上的最大最小多样性；IDEA-0007：编码器 [CLS] 排序 | 都是“重要性之外加某种覆盖”：LTAM 偏向局部连续，A 用空间分块，DivPrune 用特征空间的分散 |
| 题目怎样进入 | ViT 阶段用 CLIP 文本编码器，LLM 阶段用跨模态注意力；每道题都用（p.5–6；p.9） | 输入层词向量的 IDF × 可视性加权查询，与图像 token 算余弦；只对助动词或 how many 开头的题用（IDEA-0003） | 不同：它不路由，也没有按题型的分析（E-VT-18） |
| 文本相关的名额 | 64 档 16 个（3 : 1），补充 token 是合并出来的，中心可在全图任何位置（p.17 Table 11；p.5） | 覆盖 16：不路由题的 16 个名额挪进题目查询定位的框里，不合并（IDEA-0005、IDEA-0006） | 名额相同（16 / 64）；它不围框、合并其余，我们围框、不合并。“不围框 + 合并”我们没试过 |
| 被丢掉的 token | 合并进 R 个补充 token（p.5–6） | FastV 一族、DivPrune-64：丢掉；VisionZip-64：并成 10 个上下文 token | 它自己的数字提示合并占了很大作用（「增益从哪里来」的 AI 条） |
| 只在 LLM 内（K = 2 位置） | 64 档 GQA 54.7、VQAv2 73.6（Table 4），比引用的 VisionZip-64 低 0.4、高 1.2 | K = 2 路由 − VisionZip-64：VQAv2 test3 −2.89，GQA testdev +0.29（MAT-0030） | 评测集不同、它还做了合并，不能比大小；只记下：VQAv2 上方向相反（+1.2 对 −2.89），GQA 上两边都接近 0 |
| 额外的模型与计算 | CLIP 文本编码器（LLaVA-1.5 不带）；TGVC 的聚类（只 TGVC 741 s 对只 DVTS 670 s，Table 10） | 无额外模型；路由耗时 +0.7% | 我们更轻；它报告的端到端 1.90× 是否含文本编码器没写 |
| 训练 | 免训练；另有微调版‡ | 免训练 | 相同 |
| 对照 | SparseVLM、VisionZip、PDrop、VScan，数字多为引用；LLaVA-1.5 上没有 FastV、DivPrune | 同一流程里实测 VisionZip-64、DivPrune-64（按论文实现）、FastV 一族 | 我们的对照是自己跑的 |
| 评测 | 十个基准，是否全量、哪个划分都没写（复核时改正：原写“全量基准”），单次，无区间 | 开发集与确认集子集，按图像重抽区间，确认预注册 | 数字不能直接比 |
| 64 档相对 VisionZip-64 | 两阶段：GQA +3.7、VQAv2 +4.4（Table 1）；只在 LLM 之前：+0.5、−0.2（Table 4） | K = 1 路由 + 覆盖 16：GQA 两组新数据合并 +2.16，VQAv2 合并 +0.19（F69）；DivPrune-64：开发集 VQAv2 +2.20、GQA +1.63；VisionZip-64 + 每题覆盖 16：两组确认集合并 +0.83 / +0.38（F71） | 方向上：只在 LLM 之前加文本引导，两边都只有小增益；让 LLM 前几层看更多再剪（两阶段 / K = 1），GQA 上都更高（VisionTrim 约 +3～4，本项目约 +1.6～+2.2）。评测集不同，只能比方向 |

### 重叠（AI）

1. **层平均口径下，在 LLM 浅层再剪的做法在 GQA 上胜过只在 LLM 之前剪**：VisionTrim 比 VisionZip-64 高的 3～4 分大部分来自 LLM 第 2 层后的第二次剪枝（E-VT-10）。本项目同样发现，层平均 64 下让第 0 层看全部 576 个、之后留 47 个（K = 1），在 GQA 上比 VisionZip-64 高（K = 1 路由 testdev +1.62，MAT-0030；加覆盖 16 后两组新数据合并 +2.16，F69），在 VQAv2 上持平。方向相同、大小不同，选择准则也不同，不能当作同一现象的复现。它的 Table 15（插得越浅越好）与本项目 K = 1 好于 K = 2（MAT-0029）同向，但它没试 k = 1，而且不在 64 档。
2. **16 个文本相关名额**：它 64 档的 16 个补充 token 与本项目“覆盖 16”的名额相同（Table 11）。
3. **在 LLM 之前“让 VisionZip 看题目”只有小增益**：它只在 LLM 之前用时，GQA、VQAv2 与 VisionZip-64 相当（Table 4）；本项目 VisionZip-64 上的每题覆盖 16，确认集 +0.40 / +0.10（EXP-0029），两组确认集合并 +0.83 / +0.38（F71）。两边都说明：在 LLM 之前、同样 64 个的条件下，加文本引导的增益不大（评测集不同）。
4. **只用 [CLS] 排序偏弱**：它的 Table 5 中只用 [CLS] 比方差加权低很多；本项目 K = 1 时多样性选择比编码器 [CLS] 排序高约 2 分（F72～F74，据任务给出的摘要），DivPrune-64 比 VisionZip-64 高（开发集）。两边都提示：低预算下，[CLS] 注意力之外要加某种覆盖（局部连续、空间分块或特征多样性）。

### 可辩护的差别（AI，附理由）

1. **按题型路由与按答案类型的证据**：它对每道题都用文本引导，全文没有按题型的分析（E-VT-18）；唯一相关的对照是“直接文本引导”（Table 23），实现没写。本项目有“题目查询对 yes/no、计数有利，对问点名物体属性的开放题有害”的逐类型证据（F38、F39）和预注册的句式路由。
2. **不加模型**：本项目的题目查询在 LLM 输入层的词向量空间里算，与投影后的图像 token 在同一空间；它要另跑 CLIP 文本编码器，视觉特征怎样与之对齐没写（E-VT-2）。
3. **预算日程明写**：本项目每个配置都写明哪几层看多少个（K = 1：576 → 47）；它的“Averaged”没有定义，层间日程没给，同写 64 的设置 KV 粗算差别不小（E-VT-12）。
4. **对照在同一流程里实测，结果有区间与预注册**：它的对照数字多为引用（E-VT-8），全部单次。
5. **本项目的弱点**：只有 LLaVA-1.5-7B、以 64 档为主；它覆盖八个以上模型、高分辨率与视频；它在十个基准上报告的 64 档数字（GQA 58.8、VQAv2 76.8；复核时改正：原写“全量基准”，全文没写是否全量）比我们能写的高，我们没有复现它。审稿人可能要求在同一流程里跑 VisionTrim（有代码，下载要用户许可）。

### 是否已经做了我们想做的事（新颖性威胁，AI）

- **句式路由（IDEA-0003）**：没有。威胁仍来自 CLASP 与 VIP-Router（[PAPER-0008](PAPER-0008_CLASP.md)、[PAPER-0009](PAPER-0009_VIP-Router.md)）。
- **K = 1（层平均口径下在浅层剪）**：本项目已把它定为预算计法的事实、不作贡献（DEC-0045）。VisionTrim 是“在 LLM 浅层（第 2 层后）再剪一次、按层平均计预算”的已发表实例，Table 15 给出“越浅越好”的方向；它没有在同一层平均预算下比较 k = 1 与 k = 2，所以 MAT-0029 的比较本身它没有做过。写论文时应把 VisionTrim（以及 VScan）列为“两阶段在层平均口径下占优”的先例。
- **覆盖 16 / 定位后覆盖**：它已在 64 档给文本相关的 token 16 个名额（不围框、合并其余），对每道题都做。本项目的围框版本在 VisionZip 上没有在新数据上成立（DEC-0054），在 FastV 一族上的增益主要来自 K = 1（DEC-0045）。这与“覆盖不作为贡献”的既有决定一致，VisionTrim 是又一条理由。
- **DivPrune 底座的下一轮（DEC-0061）**：
  - DivPrune + 合并其余 token（EXP-0033 的 X4）：不看文本的合并有 VisionZip 的上下文 token 在先，看文本的合并有 VisionTrim 的 TGVC 在先；“先取与文本对齐的 token、再按多样性补上下文”有 CRISP（IDEA-0006 的补充检索，只读过摘要）。新颖性是增量的。
  - DivPrune + 在 LLM 内再剪一次：VisionTrim、VScan 都已做两阶段。本项目把同样的多样性选择放到 K = 1 没有平均的帮助（VQAv2 −1.51、GQA +0.88，据任务摘要），而 VisionTrim 在它较弱的前段上两阶段大幅占优。一个可检验的说法是：LLM 之前的选择越强，两阶段的好处越小（AI，未检验）。
  - DivPrune + 句式路由（EXP-0033 的 X3）：它没有路由，不构成威胁。

### 能借鉴什么（AI；都是候选，要经 Mode C 与查新，不是计划）

1. **同一预算下的分解表**：照 Table 4 的样子，在同一层平均预算下并排报告“只在 LLM 之前”“只在 LLM 内”“两段 / K = 1”，同时列实测的 KV 与预填充时间；并写清每层多少个，补上它没写的日程。
2. **合并被丢掉的 token**：它的数字（只用 DVTS 低于 VisionZip，加合并后与之相当，见「增益从哪里来」）提示，在 DivPrune-64 上把没选中的 token 合并成几个补充 token，可能比丢掉好。若要按题目合并，可以用本项目输入层的题目查询代替 CLIP 文本编码器（同一空间、不加模型）。这对应 EXP-0033 的 X4，新颖性有限（见上）。
3. **扫插入层时保持层平均不变**：它的 Table 15 调 α2 保持平均，做法与本项目扫 K 相同，可以引用它“浅层更好”的结论作背景。

### 对写作与实验的影响（AI）

- **改写本项目的说法**（下次更新这些文件时改；本笔记不改它们）：IDEA-0006、EXP-0028 的记录与 squeeze、DEC-0048 ⑤、实验室会话交接、index 中“VisionTrim 报告比 VisionZip 高 3～4 分”，改为“VisionTrim 的两阶段版本（LLM 之前 + LLM 第 2 层后，层平均 64）比 VisionZip-64 高 GQA +3.7、VQAv2 +4.4（Table 1）；只在 LLM 之前用时 GQA +0.5、VQAv2 −0.2（Table 4）；VisionZip 的数字引自 VisionZip 论文”。IDEA-0006 中“没有代码地址”改为：v3 给了 https://github.com/hanxunyu/VisionTrim（p.1）。
- **EXP-0028 / EXP-0029 的定位**：VisionZip-64 上的每题覆盖 16，开发集 +1.36 / +0.52、确认集 +0.40 / +0.10，与 VisionTrim 自己“只在 LLM 之前”的 +0.5 / −0.2 同一量级（评测集不同）。“远小于 VisionTrim 的 +3～4”这一比较口径不对；但 DEC-0054 的结论（确认集上不成立）不变。
- **相关工作的写法**：VisionTrim = 免训练、两阶段（ViT 第 23 层 + LLM 第 2 层），[CLS] 注意力加局部亲和度选主导 token，按文本把其余 token 合并成补充 token，每道题都用。本项目的区别写在“路由与逐类型证据”“不加模型”“预算日程明写”上；不写“首次同时在 LLM 之前与之内剪”。
- **数字对比**：不要把本项目子集上的数字与它的 Table 1 直接比；要比就在同一流程里跑它的代码（下载要用户许可）。引用它的 64 档数字时注明“两阶段、层平均（推断）、对照数字为引用”。

## 交接 Deep Read

- 候选 P：两阶段或 LLM 内剪枝的方法在“层平均 N 个”的口径下与 LLM 之前剪枝的方法比较时，选择准则、被丢 token 的处理与层间分配三件事一起变，论文里分不开；VisionTrim 自己的消融显示层间分配占大头（E-VT-9、E-VT-10）。
- 候选 F：LLaVA-1.5-7B 64 档，VisionTrim 两阶段 GQA 58.8、VQAv2 76.8，只在 LLM 之前 55.6、72.2，VisionZip-64 55.1、72.4（Table 1、Table 4）；LLM 之前加文本引导的合并，POPE +3.1～+4.4、MMB +2.3～+4.2、TextVQA +2.5～+4.0，token 越少增益越大（Table 7，E-VT-11）。
- 候选 F：同写“64 个”的三种设置，KV 粗算约合 48 / 83 / 58 个 token 的层平均（Table 4，AI，E-VT-12）。
- 机制候选见「方法与贡献」；适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功或失败结果。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-VisionTrim-20261030
analysis_id: AP-VisionTrim-20261030
created_at: '2026-10-30'
source_ids:
- PAPER-0017
- arXiv:2601.22674v3
- https://github.com/hanxunyu/VisionTrim（代码未下载、未核验（需许可））
evidence_status: partial
evidence_locations:
- 'E-VT-1: p.4–5 §3.1 式6；p.22 Table 26'
- 'E-VT-2: p.5–6 §3.2 式7–11'
- 'E-VT-3: p.6 §3.3；p.19 D.3.1；p.22 Table 26'
- 'E-VT-4: p.17 Table 11–12、D.1；p.21 D.3.7'
- 'E-VT-5: p.6 Table 1；p.18 Table 15；p.19–20 D.3.1；PAPER-0014 解析全文 p.8 正文、p.9 Table 2（推断）'
- 'E-VT-6: p.6；p.16–17；p.22 Table 26（未知）'
- 'E-VT-7: p.6 Table 1；p.7 §4.2'
- 'E-VT-8: p.6 Table 1；p.7 §4.1；p.14 参考文献；PAPER-0006 解析全文 p.5 Table 1、PAPER-0014 解析全文 p.9 Table 2（推断）'
- 'E-VT-9: p.7 Table 4；p.8 §4.3'
- 'E-VT-10: p.7 Table 4 对 p.6 Table 1（推断）'
- 'E-VT-11: p.8 Table 7；p.9'
- 'E-VT-12: p.7 Table 4；p.9 Table 8（推断）'
- 'E-VT-13: p.7 Table 5；p.8–9'
- 'E-VT-14: p.18 Table 15；p.19–20 D.3.1'
- 'E-VT-15: p.10 Table 9；p.16 Table 10；p.21 D.3.8'
- 'E-VT-16: p.7 Table 2；p.9 Table 8；p.10 Table 9；p.17 Table 12；p.21 D.3.7（推断）'
- 'E-VT-17: 全文 p.1–31；参考文献 p.10–14；p.8 Table 6；p.18 Table 14'
- 'E-VT-18: 全文 p.1–31 通读与检索'
- 'E-VT-19: p.6 Table 1；p.10 Table 9；PAPER-0008 笔记（推断）'
- 'E-VT-20: p.9 §4.3；p.16 Table 10'
- 'E-VT-21: p.20 D.3.4；p.22 Table 23'
- 'E-VT-22: p.15 Fig.7；p.15–16 附录 A'
- 'E-VT-23: p.10 §5；p.16 附录 A；p.23–24 附录 E'
- 'E-VT-24: p.1；p.16 附录 B；p.23 Table 27'
- 'E-VT-25: p.17–18 D.2.1、Table 13；p.24 附录 F'
missing_evidence:
- 代码未下载、未核验（需许可）；代码许可证全文没有写
- 没有对照 PDF；数字来自 MinerU flash 解析，式1–5、7、9–12 读不到
- 各档的层间 token 日程、α1 的基数、CLS 是否保留、合并 token 的位置编号、KV 与 FLOPs 的测法没有写
- TGVC 的视觉特征来源与文本特征的对齐方式、迭代次数 T、LTAM 三个权重的作用没有写
- 对照数字是重跑还是引用没有写；VQAv2 与 GQA 的划分、POPE 的指标、解码方式没有写
- 没有重复运行、方差或区间
- 没有本项目独立运行 VisionTrim 的结果
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」
research_question: 能否用一个免训练的统一框架，在视觉编码与 LLM 解码两个阶段都压缩视觉 token，并用文本引导把被丢掉的 token 合并成补充 token？
study_type_and_context: 免训练的视觉 token 压缩方法论文；见「方法与贡献」与「结果及不确定性」
design_units_samples_or_materials: LLaVA-1.5-7B 等多个模型与十余个基准，全部单次；见「结果及不确定性」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「预算口径」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」与「本项目引用过的说法：核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 复现等级未知（代码未核验）；可迁移的原则与必须重验的部分见「与当前项目的关系」
candidate_P: 两阶段或 LLM 内剪枝的方法在层平均口径下与 LLM 之前剪枝的方法比较时，选择准则、被丢 token 的处理与层间分配一起变，论文里分不开（E-VT-9、E-VT-10）。
candidate_F: LLaVA-1.5-7B 64 档，两阶段 GQA 58.8、VQAv2 76.8，只在 LLM 之前 55.6、72.2，VisionZip-64 55.1、72.4（Table 1、Table 4）；LLM 之前加文本引导的合并，token 越少增益越大（Table 7）。
evidence_index:
- E-VT-1
- E-VT-2
- E-VT-3
- E-VT-4
- E-VT-5
- E-VT-6
- E-VT-7
- E-VT-8
- E-VT-9
- E-VT-10
- E-VT-11
- E-VT-12
- E-VT-13
- E-VT-14
- E-VT-15
- E-VT-16
- E-VT-17
- E-VT-18
- E-VT-19
- E-VT-20
- E-VT-21
- E-VT-22
- E-VT-23
- E-VT-24
- E-VT-25
unresolved_questions:
- Retain Averaged 64 的层间日程：进 LLM 多少个、第 2 层后多少个
- 只在 LLM 之前时，文本引导的合并相对不看文本的合并（VisionZip 式）还剩多少增益
- LLM 之前的选择较强（如 DivPrune）时，两阶段还有多少好处
- KV 与 FLOPs 的测法；Table 8 中同为 320 个而 FLOPs 差一倍的原因
- TGVC 在 LLM 之前用哪一层的视觉特征、怎样与 CLIP 文本特征对齐
- 代码是否与论文一致
```

## 主代理抽查（2026-10-31）

- 写笔记的工作流（2026-10-30）原定由另一个子代理逐条复核，复核因用量上限没有完成；2026-10-31 用户睡觉期间不开子代理（DEC-0063），第二人复核留到之后。本笔记目前只经过主代理的抽查：Table 1 “Retain Averaged 64 Tokens” 一段的 VisionZip（55.1、72.4）与 Ours（GQA 58.8、VQAv2 76.8）两行、消融表的 “w/ DVTS+TGVC” 55.6 / 72.2 都在解析全文里找到；表头写的是 Averaged（层平均口径）。没有抽查到的数字以 PDF 为准。

## 第二人复核（2026-10-31）

复核方式：独立从头读完解析全文（`literature/processed/PAPER-0017/VisionTrim-arxiv-2601.22674.md`，只读导出 main_54bf807，1280 行，p.1–31 全部读完），只对照这份文本，没有对照 PDF，没有联网、没有运行 Python 或 git。MinerU flash 档只有文本，公式是图片，表格可能错位。算术（平均分复算、KV 换算、Table 15 的 α2）用只读的 PowerShell 表达式算了一遍，没有写文件。

### 核对了多少处

约 360 处，包括：Table 1 的 LLaVA-1.5-7B 全部格子（vanilla 与 192 / 128 / 64 三档的 SparseVLM、VisionZip、PDrop、VScan、VisionTrim，逐格对）；“VisionTrim − VisionZip”差值表的 33 个差值（全部重算，无误）；64 档平均分复算（VisionTrim 98.73%、VisionZip 94.38%，与笔记写的 98.7% / 94.4% 和表中 98.8% / 94.4% 相符）；Table 4、5、7、9、10、11、12、13、14、15、18、21、23、24、25、26 与 Table 2、3、6、8 的数字；KV 粗算（0.524288 MB 每 token：579、2884、320、344、194、48、83、58 个都算对）；Table 15 每行的 α2 算式（只有 k = 2 一行不符，笔记写对了：算式应得 73.3%，表中 77.3%）；作者、机构、基金号、版本、日期、代码链接、页码、附录结构；各条 DE / SC 的出处；“本项目引用过的说法：核对”12 条与“与当前项目的关系”里关于这篇论文本身的说法。文件大小（101,960 字节）与 SHA-256 前 16 位（0d83d80811f6e1082b3a7c27e3a02cdec8816805416fe4a31ee0a35b42175d38）与笔记登记一致。VScan 论文的“Retain N Tokens in Average”与“32 层平均”（PAPER-0014 解析全文 p.8 正文、p.9 Table 2）也对了一遍，一致。

### 改了哪些（改前、改后、依据）

1. 个案（「其他消融」的“个案”一条，及 E-VT-22）。
   - 改前：问“酒的年份”；“第 3 例三者都对；第 4 例三者都错”。
   - 改后：问左边那瓶饮料陈酿了多久（原文 drink）；第 3 例三种设置都答 Cooking、与 576 个时相同；第 4 例三种设置都答 Drink、576 个时答 Coca-Cola；第 3、4 例的对错不下结论。
   - 依据：p.15 Fig. 7 图中文字；p.15 正文只明写了前两例“原模型答对，VisionZip 或只用 DVTS 答错，加 TGVC 后恢复”；红字标注（错误）在解析里看不到。“酒”是原笔记的推测，原文只写 drink。
2. 对照方法（「预算口径」的“与 FastV 一族怎么对齐”）。
   - 改前：对照是 SparseVLM、PDrop 与 VScan。
   - 改后：对照是 SparseVLM、VisionZip、PDrop 与 VScan。
   - 依据：p.7 §4.1 “We compare our approach with SparseVLM, VisionZip, PyramidDrop, and VScan”；Table 1 也有 VisionZip 行。
3. FastV 的出现范围（同一条及 E-VT-17）。
   - 改前：“FastV 只出现在 Table 6 与 Table 14”。
   - 改后：加限定“实验表格里”，并注明 Fig. 2(b)(c) 图例读不到，图中有没有 FastV 不知道。
   - 依据：全文检索 FastV：正文 p.1、p.3 的引言与相关工作提到，表格里只在 Table 6（p.8）与 Table 14（p.18）；Fig. 2 图中文字没有方法名（p.3）。
4. “全量基准”（「本项目引用过的说法：核对」第 8 条、「与当前项目的关系」的“评测”行、「可辩护的差别」第 5 条“本项目的弱点”）。
   - 改前：写“全量基准，划分没写”。
   - 改后：是否全量、哪个划分都没写。
   - 依据：全文检索 subset、full set、official 等，没有任何一处说明用全量测试集还是子集（p.7 §4.1 只写“10 widely-used image-based benchmarks”）。
5. 作者说法的补注（「LLaVA-1.5-7B 主表」后的“作者的说法”）。
   - 改前：只转述“POPE、SQA、TextVQA 上不降反升”。
   - 改后：补一句按 Table 1 核对的结果：POPE、SQA 三档都高于 vanilla；TextVQA 只有 192 档高（58.4 对 58.2），128 档持平，64 档低（56.8）。
   - 依据：p.6 Table 1 与 p.7 §4.2 正文。这不是笔记的转述错误，是补上作者说法与自己表格不完全一致之处。

除以上五处，其余数字、页码与 DE / SC / AI 标签核对后未发现错误。

### 还有哪些读不清

- 公式 1–5、7、9–12 是图片，读不到；式 6、式 8 只有错位文字。尤其“按内积分配”（笔记写“做内积”）：解析里只看到 a_ij = S_v2t^i · S_t2c^j 的乘积形式，是否对文本维度求和（内积）靠上下文推断，以 PDF 为准。式的编号是按出现顺序推定的。
- Table 3（视频）Ours 一行错位，各项数字仍未能逐一核对；平均 98.0% 与 SparseVLM 86.5%、VisionZip 91.3% 读得清。
- Table 4 的 KV cache 数字是否两行共用；Table 10 的基准没写（POPE 是推断）。
- Fig. 1–15 只有图中文字，没有看图像；Fig. 7 的红字标注、Fig. 2(b)(c) 的图例读不到。
- 笔记里引用其他论文的说法（PAPER-0006 VisionZip、PAPER-0014 VScan 里的数字对比，E-VT-8 的 SparseVLM、PDrop 数字差异）这次只抽查了 VScan 的 Table 2 与“32 层平均”两处，其余没有逐格复核。
- 关于本项目自己实验的数字，按要求没有核对。

### (a) 剪枝或合并之后留下的视觉 token 的位置编号（position ids / RoPE 位置）

全文未见。全文检索 position、RoPE、rotary，只在两处出现“位置”：p.5 §3.1 的 LTAM 用 token 在视觉网格里的二维坐标 P_xy ∈ R²（“position (x, y)”）算“位置接近”那一项，同页 σ_p 的说明里“positional differences”同指这个；它们是打分用的网格坐标，不是 LLM 里的位置编号。剪枝或合并之后，留下的 K 个主导 token 与 R 个补充 token 进 LLM 时用什么 position ids、合并 token 的位置怎么定、第 2 层后再剪时留下的 token 是否保持原编号，正文、附录、Table 26 都没有写。式 11、式 12 是图片读不到，不能排除图片里有，但文字没有任何说明。这与笔记「预算口径」里“合并 token 在 LLM 里用什么位置编号，没写（UN）”一致。

### (b) 定位类任务的评测

有一处，只有一个数字，且不是专门的定位实验。p.8 Table 6 右半（Qwen2.5-VL-7B，约 1/3 的 token）最后一列是 RefCOCO：vanilla 89.5，FastV 73.6，SparseVLM 78.4，PDrop 77.5，VisionTrim 86.8（比 vanilla 低 2.7）。指标（是否为 Acc@0.5）、用哪个划分（val / testA / testB）、是否有 RefCOCO+ / RefCOCOg、输出框坐标的提示格式，全文都没有写；也没有目标检测或其他 visual grounding 评测。LLaVA-1.5 上没有 RefCOCO 类评测。p.19 Table 18 的“Referring Object Classification”（Box / Mask / Scribble / Point 是指代的提示方式）是指代物体分类，不输出框坐标，不算定位；其中 Box 一列 vanilla 55.4、VisionZip 40.4、VisionTrim 55.1，token 数没写。正文 p.8 把 Table 6 概括为“几个情况下只掉约 0.1%，偶尔超过原模型”，没有单独讨论 RefCOCO 的 2.7 分下降。
