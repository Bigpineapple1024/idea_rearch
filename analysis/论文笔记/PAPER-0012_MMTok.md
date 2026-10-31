# MMTok：研究分析笔记

> 与本项目下一轮（以 DivPrune-64 为底座，DEC-0061；在底座上加题目信息是 EXP-0033 的 X3 一类组合）最接近的前人工作：在 LLM 之前、免训练，同时“覆盖”题目的文本 token 和全部视觉 token。没有源码核验补充笔记：代码未下载、未核验（需许可）。与本项目的逐项对照见「与当前项目的关系」。

分析日期：2026-10-30。本笔记由工作流子代理用 Read 工具分段读 MinerU 解析全文（p.1–25）后整理。没有读 PDF（本次可读的只读导出 main 337305f 里没有 papers/），没有联网，没有下载任何材料，没有运行代码或实验。还没有经过第二个子代理逐条复核。

## 来源与阅读范围

- 标题：MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs（解析全文 p.1）。
- 作者/机构：Sixun Dong（Arizona State University）、Juhua Hu（University of Washington）、Mian Zhang（University of Texas at Dallas）、Ming Yin（Duke University）、Yanjie Fu（Arizona State University）、Qi Qian（Zoom Communications，通讯作者）；Dong、Zhang、Yin 的工作在 Zoom 实习期间完成（p.1 脚注）。
- 版本：arXiv:2508.18264v2 [cs.CV]，2026-03-03（p.1 页边）；每页页眉写 “Published as a conference paper at ICLR 2026”。v1 的日期、v1 与 v2 的差别未知（UN；编号 2508 说明 v1 在 2025 年 8 月提交，AI）。
- PDF：`papers/MMTok-arxiv-2508.18264.pdf`，25 页，11,995,795 字节，SHA-256 `706b0792bfff8edce3e6b69e66fd4924baa840970110e7a4fcea6d347f2a34da`；2026-10-29 经用户许可从 https://arxiv.org/pdf/2508.18264 下载（见[论文笔记 README](README.md) 的登记）。本次没有拿到 PDF，没有复算这个校验和。能访问原文不等于获得再分发许可，全文不提交 Git。
- 解析全文：`literature/processed/PAPER-0012/MMTok-arxiv-2508.18264.md`，MinerU 4.0.12 的 flash 档（只有文本），2026-10-29 在本机解析（[literature/processed/README.md](../../literature/processed/README.md)）。本次用 Get-FileHash 复算的 SHA-256 为 `4a93bcea…d88766`，与登记的前 16 位 `4a93bceae515e06a` 一致；页标记 25 个，与 PDF 页数相同。PAPER-0012 目录下没有单独的 README。
- 解析误差（引用到论文之前要对照 PDF）：
  - Eq.1、Eq.2、Prop.1–2 的不等式和附录 B.1 的优化式都是图块，解析里没有文字。本笔记对目标函数的写法是按上下文推断的（AI）。
  - Fig.1–5 是图块。Fig.1 只带出一串错乱的数字，没有使用；图像都没看。
  - Table 1、3、5、11、12、21、22 和 Table 7、18、19、20 的一部分是图块，数字来自 MinerU 抽出的 “image content” 文字，表头和行名有错位。Table 1 与附录里文字版的 Table 16 逐项相同。
  - 主表有两行都叫 “VisionZip”。第二行是微调投影层的版本，原文在名字后有一个符号，解析时丢了。依据：p.6 正文说 64 档比微调的 VisionZip 高 1.6%，只对第二行成立（96.6 − 95.0）；两行的原始分数与 VisionZip 论文 Table 1 中 VisionZip、VisionZip‡ 两行逐项相同（PAPER-0006 解析全文 p.5）。本笔记把第二行记作 VisionZip‡（AI）。
- 实际阅读：解析全文 p.1–25 全部读过：正文 p.1–10，参考文献 p.11–13（浏览），附录 p.14–25。表格数字逐表看过；没有复算论文的实验。
- 页码写法：p.N 是解析文件中 `<!-- page N of 25 -->` 页标记的页号，按解析说明与 PDF 页码一致。
- 代码：摘要给出 https://github.com/Ironieser/mmtok（p.1）；复现声明说已发布代码（p.10 §7）。许可证全文未见（UN）。代码未下载、未核验（需许可）。
- SC=作者主张，DE=解析全文中直接写着的设置、表格或数字，AI=本笔记的推断，UN=未知或全文没有。DE 不代表我们复现过。

## 研究问题

多数视觉 token 剪枝方法只用一种模态的信息：VisionZip 主要靠视觉的 [CLS] 注意力，SparseVLM 主要靠题目文字对图像的注意力，DivPrune 只看视觉 token 之间的差异；也缺一个能同时用在两种模态上的统一准则（p.1 摘要；p.2 §1；p.3 §2）。能不能用“覆盖”作为准则，同时用题目文字和图像本身来挑视觉 token？（p.2 §1）

## 方法与贡献

- **在哪里剪、是否训练**：LLaVA-1.5 的一张 336×336 图经 CLIP ViT-L-336px 得到 576 个视觉 token。在送进 LLM 之前，从 n 个视觉 token 中选 k 个（|S| ≪ n），减少 LLM 的输入 token（p.3 §3）。作者说常见做法是在视觉编码器之后选，在解码层里再选一次只做了初步实验（p.16 B.6）。全程免训练（p.2 §1；p.10 §5）。
- **“覆盖”是什么**：给定“目标 token × 源 token”的相似度矩阵 M（m×n），选一个源 token 子集 S，让目标 token 与所选 token 的相似度最大，作者称为“用所选的源 token 覆盖目标 token”（p.3–4 §3.1，Eq.1）。
  - Eq.1 在解析里只有图块。按 “maximize the similarity between the target and selected tokens”（p.4）、附录 “greedy algorithm only with max operations”（p.16 B.5）和 B.1 残缺的式子（像是对目标行取平均，p.14），推断为 f(S; M) = Σ_i max_{j∈S} M_ij，可能再除以目标个数（AI）。
  - 白话：每个目标 token 在已选的 token 里找一个最像的，把这些“最像的程度”加起来。所选集合越能让每个目标都找到一个像的，分越高。
- **为什么能用贪心**：作者引用 Leskovec 等（2007），说 Eq.1 是次模函数，即同一个 token 加进越大的已选集合，分数增加得越少（Prop.1，p.4）。一般情形下求最优是 NP 难；每次加入使目标增加最多的那个 token 的贪心法，结果不低于最优值的 (1 − 1/e)，约 63%（Prop.2，Nemhauser 等 1978；p.2、p.4）。两个次模函数之和仍是次模（Corollary 1，p.5）。
- **题目覆盖（T-V）**：目标是题目的 m 个文本 token t_i，源是 n 个视觉 token v_j；M^tv_ij = t_i · v_j，两者都归一化到单位长度，也就是余弦（p.4 §3.1.1）。v 用投影层之后的视觉 token，也就是和文本 token 拼在一起送进 LLM 的那些（p.4；p.5 实现细节）。
  - 文本 token 用哪一层的表示，原文没有直接写。推断是 LLM 输入层的词向量（AI）。理由：p.4 说用投影后的视觉 token 是为了与文本对齐（“those concatenated with text tokens as input for LLMs”）；附录 B.2 讨论把一个词的几个子词 token 在算相似度之前或之后合并（复核时改正：原写“在算相似度之前合并”；p.14 B.2 比较的是先合再算与算完再合两种）（p.14）。
  - “题目”包括哪些 token（系统提示、“Answer the question using a single word or phrase.” 这类说明句是否算进去）：UN。
- **图像覆盖（V-V）**：目标是全部 n 个视觉 token，源也是视觉 token；M^vv_ij = v′_i · v′_j（n×n），v′ 是投影层之前的视觉 token。作者的理由：投影前的特征更适合衡量视觉 token 之间的相似，不混入文本信息（p.4–5 §3.1.2；p.5 实现细节）。投影前取视觉塔的哪一层没写（UN；推断是 LLaVA 送进投影层的那一层，AI）。动机：题目含糊时（例如 “Please describe the image”）只靠题目覆盖不够（p.4）。
- **两项怎么合**：两个矩阵形状不同、数值范围也不同，先对每一行（每个目标 token）在全部视觉 token 上做 softmax，温度分别是 τ_t、τ_v；再相加，图像覆盖一项乘权重 α（Eq.2，p.5 §3.1.3）。仍用贪心法逐个选（Alg.2，p.4）。
- **默认参数**：τ_t = 0.02，τ_v = 0.2，α = 0.5，所有实验固定；作者说方法对超参数不敏感（p.5 实现细节）。Qwen-2.5-VL 上 τ_t 改为 0.01（p.7）。
- **代价**：作者说贪心只用 max 运算，选 k 个是 O(kn)（p.16 B.5）。实测见 E-MT-15。
- **变体（附录）**：MMTok_Adapt，每个样例自动定 τ_v（作者先说可以二分搜索，实际在 {0.05, 0.1, 0.15, 0.2} 里搜，p.14 B.1）；MMTok++，排除 LLaVA-1.5 的填充 patch，并修正一个“浪费一个 token”的溢出错误（p.16 附录 C）；在解码层里再选一次（p.16 B.6）。
- **作者列出的三点贡献（SC，p.2）**：① 把视觉 token 选择写成最大覆盖问题，它是次模函数，贪心法有近似保证；② 同时覆盖文本 token 与全部视觉 token，显式利用两种模态的信息；③ 在多个 VLM、多个 benchmark 上实验，并能压到极少的 token。

## 研究设计

- **模型**：LLaVA-1.5-7B、LLaVA-1.5-13B、LLaVA-NeXT-7B、LLaVA-NeXT-13B、Qwen-2.5-VL-7B（Instruct）（p.5 §4；p.22 Table 20 标题）。
- **数据集**：沿用 VisionZip 用的：GQA、MMBench、MME、POPE、ScienceQA(IMG)、VQAv2-Test-Dev、TextVQA、MMMU、SEED-Bench（p.5 §4）。Qwen 上另加 OCRBench（p.7 Table 3）；附录另加 MMStar（p.14–15 B.3）。VQAv2 以外各数据集用哪个划分没写（UN）。
- **指标**：POPE 用 F1，MME 用感知 + 认知（P+C），其余是准确率（p.8 Table 6、p.18 Table 16 的表头）。“Avg.” 是各项相对不剪枝的百分比的平均（p.18 Table 16 标题）。
  - LLaVA-1.5-7B 主表的 SEED 列不剪枝为 58.6，与 Table 4、21 的 SEED-I（不剪枝 66.14）不是同一个指标。按 Table 17 的表注（SEED 指 SEED-ALL），推断主表的 SEED 是 SEED-ALL（AI；p.8 Table 4、p.19 Table 17、p.23 Table 21）。
- **对照**：FastV（作者称“基于视觉”）、SparseVLM（基于语言）、VisionZip（基于 [CLS] 重要性）、DivPrune（基于多样性），外加一个微调的方法 VisionZip‡（p.5 §4）。DivPrune 用它的官方代码跑；其余对照的数字直接取自 VisionZip 论文（p.5）。作者说之后的实验只与 VisionZip、DivPrune 比（p.6），但附录 Table 18（LLaVA-NeXT-7B）仍列了 SparseVLM（p.20）。
- **评测框架**：lmms-eval（p.5）。
- **硬件**：效率表用单张 H100（p.9 Table 8）和单张 A6000（p.16 Table 12、13）；主表用什么硬件没写（UN）。
- **统计**：全文没有运行次数、方差或区间；解码设置没写（UN）。

## 预算口径

- **“保留 N 个”是什么（DE）**：选择在 LLM 之前，N 是送进 LLM 的视觉 token 个数（p.3 §3）。LLaVA-1.5 从 576 个中留 192 / 128 / 64 个（↓67% / 78% / 89%，p.6 Table 1），附录另测 32、16、8、4、2 个（p.23 Table 21）。
- **换成本项目的层平均口径（AI）**：在 LLM 之前选，32 层每层都只看 N 个，所以 MMTok 的 64 个就是本项目口径下的层平均 64，与 DivPrune-64、VisionZip-64 的算法相同。
- **CLS 与合并 token**：MMTok 只从 n 个视觉 token 里挑，不合并，方法描述里没有 CLS（DE：p.3–5）。LLaVA-1.5 的 576 个本来就不含 CLS（AI，依据是 LLaVA 的设计，不是 MMTok 原文）。表中 VisionZip 的 64 个按 VisionZip 自己的定义含 CLS 和合并的上下文 token（本项目的记录，见[已发现的实验现象](../已发现的实验现象.md) F62、F64），MMTok 没有讨论这一点（UN）。
- **与 FastV 一族怎么对齐**：没有对齐。FastV、SparseVLM 的数字直接取自 VisionZip 论文（p.5），MMTok 没写 FastV 的剪枝层 K 和保留数（UN）。VisionZip 对 FastV、SparseVLM 也只说沿用原设置（本项目 [MODE_C-0002](../MODE_C-0002_视觉token剪枝.md) 的 E-RC-4）。所以表中的“FastV-64”换到本项目的层平均口径是多少，不知道。旁证（AI）：MMTok 表中 FastV-64 的 POPE 为 48.0（p.6 Table 1），CoverPruner 表中为 35.5（CoverPruner 解析全文 p.5 Table 1），两篇论文的“FastV-64”不是同一个设置。
- **LLaVA-NeXT**：按每个子图的固定比例选，最多 5 个子图。上限 640 / 320 / 160 就是每个子图 128 / 64 / 32 个；只取 4 个子图时，上限 160 档实际是 128 个（p.6–7）。各数据集实际的平均子图数和 token 数见 Table 18（p.20），例如上限 640 档的 GQA 平均 627 个。
- **Qwen-2.5-VL-7B**：动态分辨率，按每个数据集的平均视觉 token 数 T̄ 留 20% / 10% / 5%；例如 MME 的 T̄ = 867.6，20% 是 173.5 个（p.7 Table 3）。
- **FLOPs 与时延**：
  - 没有整个模型的理论 FLOPs（全文未见）。
  - 时延是整套评测的墙钟时间（秒），另报 GPU 利用率和运行时多占的显存：LLaVA-NeXT-13B、单张 H100（p.9 Table 8）；Qwen-2.5-VL-7B、单张 A6000、MME（p.16 Table 12）。
  - 选择这一步本身的耗时和计算量单独给出（p.16 B.5、Table 13）。
  - LLaVA-1.5-7B 上没有时延或 FLOPs（全文未见）。

## 结果及不确定性

### LLaVA-1.5-7B（p.6 Table 1；p.18 Table 16，两表数字相同）

| 预算 | 方法 | GQA | MMB | MME | POPE | SQA | VQAv2 | TextVQA | MMMU | SEED | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 576 | 不剪枝 | 61.90 | 64.70 | 1862.00 | 85.90 | 69.50 | 78.50 | 58.20 | 36.30 | 58.60 | 100% |
| 192 | FastV | 52.70 | 61.20 | 1612.00 | 64.80 | 67.30 | 67.10 | 52.50 | 34.30 | 57.10 | 89.6% |
| 192 | SparseVLM | 57.60 | 62.50 | 1721.00 | 83.60 | 69.10 | 75.60 | 56.10 | 33.80 | 55.80 | 95.5% |
| 192 | VisionZip | 59.30 | 63.00 | 1782.60 | 85.30 | 68.90 | 76.80 | 57.30 | 36.60 | 56.40 | 97.9% |
| 192 | DivPrune | 59.97 | 62.54 | 1762.23 | 87.00 | 68.66 | 76.87 | 56.97 | 35.44 | 58.71 | 98.0% |
| 192 | VisionZip‡（微调） | 60.10 | 63.40 | 1834.00 | 84.90 | 68.20 | 77.40 | 57.80 | 36.20 | 57.10 | 98.4% |
| 192 | MMTok | 60.07 | 63.40 | 1773.86 | 86.42 | 68.76 | 77.11 | 57.68 | 36.33 | 59.21 | 98.7% |
| 128 | FastV | 49.60 | 56.10 | 1490.00 | 59.60 | 60.20 | 61.80 | 50.60 | 34.90 | 55.90 | 84.4% |
| 128 | SparseVLM | 56.00 | 60.00 | 1696.00 | 80.50 | 67.10 | 73.80 | 54.90 | 33.80 | 53.40 | 92.9% |
| 128 | VisionZip | 57.60 | 62.00 | 1761.70 | 83.20 | 68.90 | 75.60 | 56.80 | 37.90 | 54.90 | 96.8% |
| 128 | DivPrune | 59.25 | 62.03 | 1718.22 | 86.72 | 68.66 | 75.96 | 56.06 | 35.56 | 56.98 | 96.9% |
| 128 | VisionZip‡（微调） | 58.90 | 62.60 | 1823.00 | 83.70 | 68.30 | 76.60 | 57.00 | 37.30 | 55.80 | 97.7% |
| 128 | MMTok | 59.29 | 62.29 | 1779.14 | 86.25 | 68.82 | 76.35 | 57.03 | 35.67 | 58.59 | 97.8% |
| 64 | FastV | 46.10 | 48.00 | 1256.00 | 48.00 | 51.10 | 55.00 | 47.80 | 34.00 | 51.90 | 75.6% |
| 64 | SparseVLM | 52.70 | 56.20 | 1505.00 | 75.10 | 62.20 | 68.20 | 51.80 | 32.70 | 51.10 | 86.9% |
| 64 | VisionZip | 55.10 | 60.10 | 1690.00 | 77.00 | 69.00 | 72.40 | 55.50 | 36.20 | 52.20 | 93.2% |
| 64 | DivPrune | 57.78 | 59.28 | 1674.40 | 85.56 | 68.07 | 74.11 | 54.69 | 35.56 | 55.13 | 94.8% |
| 64 | VisionZip‡（微调） | 57.00 | 61.50 | 1756.00 | 80.90 | 68.80 | 74.20 | 56.00 | 35.60 | 53.40 | 95.0% |
| 64 | MMTok | 58.29 | 61.17 | 1715.33 | 85.77 | 69.16 | 75.20 | 56.01 | 36.11 | 57.15 | 96.6% |

- 表注：POPE 为 F1，MME 为 P+C，VQAv2 为 test-dev，SQA 为 ScienceQA(IMG)，SEED 见「研究设计」（p.5；p.18 Table 16 表头）。DivPrune 是作者用官方代码跑的；FastV、SparseVLM、VisionZip、VisionZip‡ 取自 VisionZip 论文，9 项原始分数与 VisionZip 论文 Table 1 逐项相同（PAPER-0006 解析全文 p.5）。平均只算这 9 项，所以与 VisionZip 论文的 11 项平均不同，例如 VisionZip-64 是 93.2% 对 94.0%（AI 核对）。
- 作者的说法（SC，p.6 §4.1）：三档的平均都是最好（98.7 / 97.8 / 96.6%）；64 档比 DivPrune 高 1.8 个百分点；192 档比所有免训练的对照至少高 0.7；64 档比微调的 VisionZip‡ 高 1.6。这几个差都能由表中的平均值对上（AI 核对）。
- 按单项看（DE 读表；归纳是 AI）：
  - VQAv2：64 档 MMTok 75.20，比 DivPrune 高 1.09，比 VisionZip 高 2.80；128、192 档比 DivPrune 只高 0.39、0.24；192 档低于 VisionZip‡（77.11 对 77.40）。
  - GQA：64 档 58.29，比 DivPrune 高 0.51，比 VisionZip 高 3.19；128、192 档与 DivPrune 只差 0.04、0.10；192 档与 VisionZip‡ 基本相同（60.07 对 60.10）。
  - POPE：64 档 85.77，比 DivPrune 高 0.21；128、192 档低于 DivPrune（86.25 对 86.72，86.42 对 87.00）。DivPrune 与 MMTok 在 128、192 档都高于不剪枝的 85.90。
  - MME：三档都低于 VisionZip‡（1773.86 / 1779.14 / 1715.33 对 1834 / 1823 / 1756）；192 档也低于免训练的 VisionZip（1782.6）。MMTok 自己 128 档的 MME 高于 192 档。
  - SEED：MMTok 三档都最高；192 档 59.21，超过不剪枝（101.0%）。
  - 所以 192、128 档上平均的领先来自少数几项（AI，按 Table 16 的百分比计算）：相对 VisionZip‡ 主要来自 POPE 与 SEED（MME 反而更低）；相对 DivPrune，128 档主要来自 MME、SEED、TextVQA，192 档分散在 MMMU、MMB、TextVQA 等几项。
- **不确定性**：没有区间，没有写运行次数。192、128 档 VQAv2、GQA 上不到 0.4 的差分辨不出（AI）。MME 上 128 档反而高于 192 档，说明 MME 的单次波动可能有几十分（AI）。

### 其他模型（p.6 Table 2；p.7 Table 3；p.19–22 Table 17–20）

- LLaVA-1.5-13B 的平均（192 / 128 / 64）：MMTok 98.7 / 97.5 / 96.4%，DivPrune 98.2 / 96.9 / 95.3%，VisionZip 97.9 / 97.0 / 93.7%，VisionZip‡ 98.7 / 97.4 / 94.8%（p.6 Table 2；p.19 Table 17）。
- LLaVA-NeXT-7B（上限 640 / 320 / 160）：MMTok 98.7 / 97.3 / 95.1%，VisionZip‡ 98.9 / 97.6 / 95.0%；LLaVA-NeXT-13B：MMTok 98.2 / 96.4 / 95.1%，VisionZip‡ 98.8 / 97.8 / 94.6%（p.6 Table 2）。NeXT 的 640、320 档上，微调的 VisionZip‡ 高于 MMTok（DE）。
- Qwen-2.5-VL-7B（20% / 10% / 5%，5 项平均）：MMTok 94.6 / 88.5 / 79.0%，VisionZip 94.2 / 87.5 / 75.4%，DivPrune 91.5 / 84.7 / 76.3%（p.7 Table 3）。平均没有算 SQA 和 OCRBench，作者的理由是 SQA 的图像贡献低、所有对照在 OCRBench 上都差（p.7）。不给视觉 token 时，Qwen 对 POPE 全答 “No”，F1 为 0（p.7 Table 3 注）。

### 图像贡献与极少 token（p.8 §4.2、Table 4–5；p.23–24 Table 21–22）

- 图像贡献 IC = (不剪枝的分数 − 不给视觉 token 的分数) / 不给视觉 token 的分数（p.8）。LLaVA-1.5-7B：MMB 2.347、POPE 0.924、MME 0.918、SEED-I 0.786、GQA 0.644、TextVQA 0.397、SQA 0.094、MMMU 0.089（p.8 Table 4）。VQAv2 没有 IC 值（UN）。不给视觉 token 时，LLaVA-1.5 在 MMMU 上仍保留 92%、在 ScienceQA 上 91%（p.8）。作者据此为 LLaVA 选出 5 个高 IC 数据集（MMB、POPE、MME、SEED、GQA），为 NeXT 选出 6 个（再加 TextVQA）（p.8）。
- LLaVA-1.5-7B 在 5 个高 IC 数据集上的平均（64 / 32 / 16 / 8 / 4 / 2 个）：MMTok 94.7 / 91.0 / 86.4 / 79.8 / 71.4 / 62.1%，DivPrune 93.1 / 89.6 / 81.4 / 68.4 / 54.3 / 50.1%，VisionZip 90.0 / 83.5 / 69.7 / 48.9 / 43.8 / 43.0%；不给视觉 token 为 50.2%（p.8 Table 5；p.23 Table 21）。
- 留得极少时比不给图还差（DE 读表）：不给视觉 token 时 POPE F1 为 44.64；VisionZip 留 8 / 4 / 2 个只有 23.66 / 24.48 / 26.48，DivPrune 留 4 / 2 个为 33.33 / 37.60；MMTok 留 2 个仍有 68.95（p.23 Table 21）。作者说 MMTok 只用 2 个 token 在 POPE 上保留约 80%（p.8；68.95 / 85.9 = 80.3%，AI 计算）。
- MMStar（推理题，p.15 Table 11）：不剪枝平均 33.69；64 个时 VisionZip 32.47、DivPrune 30.99、MMTok 33.33；32 个时 MMTok 33.50。作者说不剪枝本身就低，差别难显出来（p.15）。

### 消融（LLaVA-1.5-7B，64 个，p.8 Table 6）

| 变体 | GQA | MMB | MME | POPE | SQA | TextVQA | MMMU | SEED | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 只用题目覆盖 T-V（M^tv） | 56.82 | 59.62 | 1632.47 | 83.56 | 68.72 | 51.97 | 35.33 | 56.36 | 93.8% |
| 只用图像覆盖 V-V（M^vv） | 58.14 | 59.88 | 1662.34 | 83.43 | 67.67 | 53.93 | 35.33 | 56.90 | 94.7% |
| Softmax T-V | 56.66 | 58.85 | 1674.11 | 83.69 | 68.57 | 52.01 | 35.33 | 56.37 | 93.9% |
| Softmax V-V | 57.97 | 60.31 | 1684.33 | 84.31 | 68.07 | 55.90 | 35.89 | 56.88 | 95.7% |
| MMTok（两项） | 58.29 | 61.17 | 1715.33 | 85.77 | 69.16 | 56.01 | 36.11 | 57.15 | 96.7% |

- 8 项，没有 VQAv2（DE）。两项合起来比只用一项高约 1 个百分点，作者据此说两项互补（p.8）。
- 从 Softmax V-V 加上题目覆盖，8 项都上升：POPE +1.46、SQA +1.09、MMB +0.86、GQA +0.32、SEED +0.27、MMMU +0.22、TextVQA +0.11、MME +31.00（DE 读表）。
- 只用题目覆盖时，8 项平均最低；比只用图像覆盖好的只有 SQA（两种版本都是）和 POPE（只在不做 softmax 时，83.56 对 83.43）（DE 读表）。
- 与 DivPrune 比（AI 计算，同样 8 项）：DivPrune-64 按 Table 16 的百分比平均为 94.8%，低于 Softmax V-V 的 95.7%；逐项只有 POPE 低于 DivPrune（84.31 对 85.56）（复核时补：SQA 两者同为 68.07，持平；Table 16 的 94.8% 是 9 项平均，复核时去掉 VQAv2 按 8 项重算，DivPrune-64 仍约 94.8%，VisionZip-64 约 93.2%）。
- 没有 α、τ_t 的扫描（全文未见）。作者说“不敏感”，依据只有 16 档的自适应 τ_v（p.14 Table 9：88.6% 对 88.7%）。

### 其他附录结果

- 子词合并（p.14–15 B.2、Table 10）：把一个词的几个子词 token 合起来（先合再算或算完再合；取平均、取最大或取第一个），没有一致的好处，平均为不合并的 99.7%～99.8%，所以不合并。
- 缩小图像（p.8–9、Table 7，Qwen、MME）：同样的 token 数，从原图里选比缩小图像好；先缩到 1/4 再留 25%（87.4 个）得 2170，高于在原图上留 10%（86.8 个）的 2051。
- 解码层里再选一次（p.16 B.6、Table 14，LLaVA-NeXT-13B、MME）：进 LLM 时 160 个、第 24 层再减到 80 个，MME 1846；只留 160 个是 1811，只留 80 个是 1717。作者称是初步实验。
- MMTok++（p.16 附录 C；p.17 Table 15，5 个高 IC 数据集的平均）：32 / 16 / 8 / 4 / 2 个时从 91.0 / 86.4 / 79.8 / 71.4 / 62.1% 升到 91.6 / 87.5 / 81.4 / 73.6 / 67.3%；token 越少，好处越大（AI 归纳）。64 档没有报（UN）。正文的结果都是修正前的版本（p.16）。

### 效率（p.9 Table 8；p.16 Table 12–13）

- LLaVA-NeXT-13B、单张 H100、上限 160（Table 8）：POPE 的推理总时间 1705 s → MMTok 913 s（VisionZip 866 s，DivPrune 1060 s）；6 项合计 15204 s → 7768 s（VisionZip 7551 s，DivPrune 8186 s）；GPU 利用率 86.7% → 58.0%；运行时显存（模型本身 25.42 GB 之外）4.59 GB → 1.78 GB；6 项的平均保留 MMTok 93.7%、DivPrune 90.5%、VisionZip 89.6%。表注说这些结果都在同一硬件和评测设置下重跑（p.9）。作者说 DivPrune 慢一些，是因为它的官方代码处理多子图的方式不同（p.9）。
- Qwen-2.5-VL-7B、单张 A6000、MME、留 86.8 个（Table 12）：675 s → MMTok 419 s（VisionZip 508 s，DivPrune 423 s）。
- 选择本身（Table 13，LLaVA-NeXT-7B，A6000，100 次平均）：576 个中选 32 个 1.267 ms、选 16 个 0.774 ms；2880 个中选 160 个 6.417 ms；2880 个输入时约 13.93 GFLOPs（p.16）。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-MT-1 | DE | 在 LLM 之前从 n 个视觉 token 中选 k 个，减少 LLM 的输入；免训练；正文不在解码层里剪 | p.2 §1；p.3 §3；p.10 §5；p.16 B.6 | B.6 只有一个解码层二次选择的初步实验 |
| E-MT-2 | DE / AI | 选择写成最大覆盖：选 S 使目标 token 与所选源 token 的相似度最大；作者称是次模函数，贪心不低于最优的 (1 − 1/e)；两个次模函数之和仍是次模 | p.3–4 §3.1、Eq.1、Prop.1–2；p.5 Corollary 1 | Eq.1、Eq.2 在解析里只有图块；f(S; M) = Σ_i max_{j∈S} M_ij 的写法是 AI 推断 |
| E-MT-3 | DE / AI | 题目覆盖：M^tv_ij = t_i · v_j，单位长度；v 是投影层之后的视觉 token | p.4 §3.1.1；p.5 实现细节 | 文本表示推断为 LLM 输入层词向量（AI，依据 p.4、p.14 B.2）；是否含系统提示、说明句 UN |
| E-MT-4 | DE | 图像覆盖：M^vv_ij = v′_i · v′_j（n×n），v′ 是投影层之前的视觉 token，目标是全部视觉 token | p.4–5 §3.1.2；p.5 实现细节 | 取视觉塔哪一层 UN |
| E-MT-5 | DE | 每行 softmax（τ_t、τ_v）后相加，图像覆盖乘 α；默认 τ_t = 0.02、τ_v = 0.2、α = 0.5；Qwen 上 τ_t = 0.01 | p.5 §3.1.3、实现细节；p.7 | α、τ_t 没有扫描；选参数用的数据 UN |
| E-MT-6 | DE | “保留 N 个”= 送进 LLM 的视觉 token 数；LLaVA-1.5 为 192/128/64（另有 32～2）；NeXT 按每个子图（上限 640/320/160）；Qwen 按数据集平均的 20%/10%/5% | p.3 §3；p.6 Table 1；p.6–7；p.7 Table 3；p.20 Table 18；p.23 Table 21 | 换成本项目的层平均是同一个数（AI） |
| E-MT-7 | DE / AI | DivPrune 用官方代码跑，其余对照直接取自 VisionZip 论文，用 lmms-eval；9 项原始分数与 VisionZip 论文 Table 1 相同；第二行 VisionZip 是 VisionZip‡ | p.5 §4；p.6 §4.1；PAPER-0006 解析全文 p.5 Table 1 | FastV、SparseVLM 的 K 与保留数没写（UN） |
| E-MT-8 | DE | LLaVA-1.5-7B 三档的全部数字（见上表）；64 档 MMTok VQAv2 75.20、GQA 58.29、POPE 85.77 | p.6 Table 1；p.18 Table 16 | 运行次数未写，无区间；VQAv2 为 test-dev，POPE 为 F1，MME 为 P+C |
| E-MT-9 | SC | 三档平均最好；64 档比 DivPrune 高 1.8、比 VisionZip‡ 高 1.6，192 档比免训练对照至少高 0.7（百分点）；免训练能超过微调的方法 | p.2 §1；p.6 §4.1 | 平均值对得上（AI 核对）；单项上 MME 三档都低于 VisionZip‡，128、192 档 POPE 低于 DivPrune |
| E-MT-10 | DE | 跨模型平均：LLaVA-1.5-13B 98.7/97.5/96.4%；NeXT-7B 98.7/97.3/95.1%（VisionZip‡ 98.9/97.6/95.0%）；NeXT-13B 98.2/96.4/95.1%（VisionZip‡ 98.8/97.8/94.6%）；Qwen（5 项）94.6/88.5/79.0% | p.6 Table 2；p.7 Table 3；p.19–22 Table 17–20 | NeXT 的 640、320 档 VisionZip‡ 更高 |
| E-MT-11 | DE | 消融（64 档，8 项，没有 VQAv2）：T-V 93.8%，V-V 94.7%，Softmax T-V 93.9%，Softmax V-V 95.7%，两项 96.7% | p.8 Table 6 | 无区间；没有 α 消融，也没有“随机文本 token 当目标”的对照 |
| E-MT-12 | DE | IC 的定义；LLaVA-1.5-7B：POPE 0.924、MME 0.918、GQA 0.644、TextVQA 0.397、SQA 0.094、MMMU 0.089、MMB 2.347、SEED-I 0.786 | p.8 §4.2、Table 4 | 没有 VQAv2；不给视觉 token 时怎样输入没写 |
| E-MT-13 | DE | 极少 token（5 个高 IC 数据集平均）：MMTok 94.7→62.1%（64→2 个），DivPrune 93.1→50.1%，VisionZip 90.0→43.0%，不给视觉 token 50.2%；POPE 上 VisionZip 留 8/4/2 个只有 23.66/24.48/26.48，低于不给视觉 token 的 44.64 | p.8 Table 5；p.23 Table 21 | 作者没有解释为什么会比不给图还差 |
| E-MT-14 | DE | 效率：NeXT-13B、H100、上限 160：POPE 1705→913 s，6 项 15204→7768 s，运行时显存 4.59→1.78 GB；Qwen、A6000、MME：675→419 s | p.9 Table 8；p.16 Table 12 | 整套评测的墙钟时间；LLaVA-1.5-7B 上没有 |
| E-MT-15 | DE / AI | 选择本身：576 选 32 为 1.267 ms，2880 选 160 为 6.417 ms，约 13.93 GFLOPs；作者称 O(kn) | p.16 B.5、Table 13 | O(kn) 没有把目标行数算进去（AI，见「限制与替代解释」） |
| E-MT-16 | SC / UN | 摘要：POPE 上 1.87 倍加速且保留 98.7%（LLaVA-NeXT-13B）；LLaVA-1.5-7B 只留 4 个仍保留 87.7% | p.1 摘要 | 前者与 Table 8 对得上（AI 计算）；87.7% 在解析的表格里找不到出处（UN） |
| E-MT-17 | DE | MMTok++：排除填充 patch、修正溢出错误；32～2 个时平均从 91.0/86.4/79.8/71.4/62.1% 升到 91.6/87.5/81.4/73.6/67.3%；正文结果是修正前的 | p.16 附录 C；p.17 Table 15 | 64 档没有报；错误的细节、如何认出填充 patch 没写 |
| E-MT-18 | DE | 解码层二次选择：NeXT-13B、MME，160 个进 LLM、第 24 层再减到 80 个为 1846；只留 160 个 1811，只留 80 个 1717 | p.16 B.6、Table 14 | 一个数据集，作者称初步 |
| E-MT-19 | DE / AI | 子词合并没有一致的好处（99.7%～99.8%）；自适应 τ_v 88.6% 对 88.7%（16 档） | p.14–15 B.1–B.2、Table 9–10 | Table 9 的 SEED 分母疑似混用（AI，见「限制与替代解释」） |
| E-MT-20 | DE / AI | Qwen 的平均去掉 SQA 与 OCRBench；按 Table 20 的 7 项百分比自己平均，20% 档 MMTok 91.1%、VisionZip 91.2% | p.7；p.22 Table 20 | 7 项平均是 AI 计算 |
| E-MT-21 | DE | 缩小图像（Qwen、MME）：同 token 数下从原图选好于缩小；先缩到 1/4 再留 25%（87.4 个）2170，高于原图留 10%（86.8 个）的 2051 | p.8–9、Table 7 | 一个模型、一个数据集；是缩小整张图，不是本项目 IDEA-0004 的裁剪后放大 |
| E-MT-22 | SC | 作者自述的局限与未来方向：题目含糊、文本信息有限、按题目难度定 token 数、缩小与选择结合、推理题的上限低 | p.4 §3.1.2；p.9；p.10 §4.3、§5；p.15 B.3 | 全文没有专门的 Limitations 一节 |
| E-MT-23 | DE / UN | 代码链接 https://github.com/Ironieser/mmtok；复现声明说已发布代码；伦理声明说没有要披露的问题；LLM 使用声明说 idea 与写作阶段没有用 LLM | p.1；p.10 §6–7；p.14 附录 A | 许可证全文未见；代码未下载、未核验（需许可） |
| E-MT-24 | AI | CoverPruner（PAPER-0011）Table 1 中 MMTok-64 的 VQAv2 75.2、GQA 58.3、SQA 69.2、TextVQA 56.0、POPE 85.8、MMB 61.2 与 MMTok 自己的数字相同（四舍五入）；MME 1371.5 不同，CoverPruner 的不剪枝 MME 为 1506.5，推断它只报感知分 | CoverPruner 解析全文 p.5 Table 1；MMTok p.6 Table 1 | 像是引用而不是重跑；没有看到独立复现 |
| E-MT-25 | AI | GQA 64 档：MMTok 表中 VisionZip 55.10，本项目 testdev 实测 55.18；本项目已确认最好的配置（K = 1 路由 + 覆盖 16）为 57.22，比 MMTok 表中的 MMTok 58.29 低 1.07、比 DivPrune 57.78 低 0.56 | MMTok p.6 Table 1；本项目 EXP-0022（[所有论文在baseline骨架的改进内容](../专题分析/所有论文在baseline骨架的改进内容.md)）、EXP-0027-R05（[论文主张与证据_草稿](../论文主张与证据_草稿.md)） | MMTok 的 GQA 划分没写（按 lmms-eval 的默认推断为 testdev-balanced）；单次，无区间；实现不同 |
| E-MT-26 | AI | 同样 8 项上，Softmax V-V 单独 95.7%，DivPrune-64 94.8%，VisionZip-64 93.2%；逐项只有 POPE 上 V-V 低于 DivPrune（84.31 对 85.56） | p.8 Table 6；p.18 Table 16 | 由两张表的百分比算出；DivPrune 是作者用官方代码跑的 |
| E-MT-27 | UN | 运行次数与区间；超参数的选取数据；文本 token 的范围；V-V 用视觉塔哪一层；Eq.1、Eq.2 的精确形式；不给视觉 token 时的输入方式；VQAv2 以外的划分；MMTok++ 在 64 档的结果；v1 与 v2 的差别；许可证 | — | 需要 PDF 与代码 |

没有源码核验：代码未下载、未核验（需许可）。

## 限制与替代解释

- **对照数字的来源不一**：DivPrune 是作者用官方代码跑的，FastV、SparseVLM、VisionZip、VisionZip‡ 照抄 VisionZip 论文（p.5）。不同来源之间可能有评测环境带来的系统差（AI）。一个旁证：LLaVA-NeXT-13B 上作者自己重跑的 VisionZip（Table 8：GQA 57.77、POPE 76.32、MME 1738.24）与 Table 19 中按作者说法取自 VisionZip 论文的数字（57.8、76.6、1739）相比，准确率一类相差都在 0.3 以内，MME 差 0.76 分（p.9；p.21），至少在这一处两种来源对得上（AI）。
- **FastV 的预算口径不明**（见「预算口径」）。表中的 FastV 很弱（64 档平均 75.6%），可能部分来自口径（AI）。LLaVA-1.5-7B 之后，作者不再比 FastV（p.6）。
- **没有不确定性**：没有区间、运行次数和种子（全文未见），小的差分辨不出（见「结果及不确定性」）。
- **平均值的构成**：9 项相对百分比的平均会被超过 100% 的项拉动；192、128 档上 MMTok 的平均领先来自少数几项，例如相对 VisionZip‡ 主要是 POPE 与 SEED（AI，见「结果及不确定性」）。Qwen 上作者去掉了 SQA（MMTok 三档都低于 VisionZip：81.61 / 80.47 / 77.19 对 84.23 / 82.30 / 79.57）和 OCRBench（MMTok 高于 VisionZip）。按 Table 20 的 7 项百分比自己平均，20% 档 MMTok 91.1%、VisionZip 91.2%；10%、5% 档 MMTok 仍然更高（83.8% 对 82.2%，74.3% 对 70.2%）（AI 计算，p.22 Table 20）。
- **“最好”的范围**：LLaVA-NeXT 的 640、320 档，微调的 VisionZip‡ 高于 MMTok（p.6 Table 2）；LLaVA-1.5-7B 的 MME 三档都低于 VisionZip‡（p.6 Table 1）。作者的 “overall best”（p.2）对免训练的对照和低预算更成立（AI）。
- **超参数**：τ_t、τ_v、α 固定，作者说不敏感（p.5），但全文没有 α 或 τ_t 的扫描，只有 16 档的自适应 τ_v（p.14 Table 9）；Qwen 上又换了 τ_t（p.7），说明不是完全不用调。这些值在哪些数据上定的，没写（UN）。如果是在评测的 benchmark 上调的，结果会偏乐观（AI）。
- **消融的范围**：Table 6 没有 VQAv2，也没有按题型拆分。没有“随机文本 token 当目标”之类的对照，所以题目覆盖的增益是来自题目的语义，还是只来自“多了一组目标行、选择变了”，分不开（AI）。
- **正文结果有已知问题**：正文结果包括填充 patch，还有一个浪费一个 token 的溢出错误；修正版只在附录、只报 32～2 个（p.16–17）。错误具体是什么、修正版实际用了几个 token，没写（UN）（复核时补：p.16 附录 C 写修正后“准确率明显提高、用的 token 更少”，但没给具体个数）。对比较的影响：正文的 MMTok 数字偏保守（AI）。
- **公式缺失**：解析里 Eq.1、Eq.2 只有图块；本笔记对目标函数的写法是推断（AI）。
- **复杂度的说法**：作者说 O(kn)（p.16）。按常见实现，每一步要对每个候选算它对全部目标行（m 个文本 + n 个视觉）的增益，量级应是 O(k·n·(m + n))（AI）。实测只有 1～7 ms（Table 13），不影响“开销很小”的结论。
- **表内不一致（AI）**：Table 9（p.14）的 SEED 列不剪枝写 58.6，但 MMTok 16 档的 56.67 在 Table 21 里属于 SEED-I（不剪枝 66.14，p.23）。按 66.14 算，Table 9 中 MMTok 的平均应是 86.4%（与 Table 15、21 一致），不是 88.6%。这不影响 Table 9 两行之间的比较。
- **摘要的数字**：1.87 倍和 98.7% 能由 Table 8 的 POPE 算出（1705 / 913 = 1.87；85.11 / 86.22 = 98.7%），条件是 LLaVA-NeXT-13B、上限 160、整套评测的墙钟时间（AI 计算）。“4 个 token 保留 87.7%”在解析的表格里找不到：Table 21 中 POPE 是 74.84 / 85.9 = 87.1%，MMTok++ 是 76.36 / 85.9 = 88.9%（p.17；p.23）；可能来自 Fig.1(a)（图未看）（UN）。
- **效率测量**：整套评测的墙钟时间，含数据读取、生成等；只在 NeXT-13B 和 Qwen 上测，LLaVA-1.5-7B 上没有（全文未见）。DivPrune 慢的原因作者归于它的官方代码（p.9），所以“和 VisionZip 一样快”的比较也取决于实现（AI）。
- **定性图**：Fig.3–5 只是少数例子（p.9–10、p.17、p.25），图像未看。
- **独立复现**：没有看到。CoverPruner 表中的 MMTok 数字与 MMTok 自己的相同（E-MT-24），像是引用（AI）。

### 作者自述的局限

- 全文没有专门的 Limitations 一节（全文未见）。
- 题目含糊时只靠题目覆盖不够（p.4 §3.1.2），这是加图像覆盖的理由。
- 文本作为覆盖目标可能信息有限，以后可以用一个轻量的 agent VLM 补充文本 token（p.10 §5）。
- 合适的 token 数取决于题目难度，按难度选 token 是以后的方向（p.10，Fig.4）。
- 缩小图像与选择的结合留作以后的工作（p.9）。
- 推理题上不剪枝的分数本身低，差别难显出来（p.15 B.3）。
- 伦理声明：没有要披露的伦理问题（p.10 §6）。LLM 使用声明：idea 与写作阶段完全没有用 LLM（p.14 附录 A）。

## 与当前项目的关系

本项目的事实来自工作流给出的项目事实（IDEA-0003、F69、F72～F74、DEC-0061），以及本仓库只读导出（main 337305f）中的记录：[IDEA-0002](../IDEA-0002_题目词按信息量加权.md) 的“实现公式”、[IDEA-0003](../IDEA-0003_按题目句式路由.md)、[IDEA-0005](../IDEA-0005_定位后覆盖.md)、[已发现的实验现象](../已发现的实验现象.md)（F62～F64、F69）、[EXP-0032](../实验报告/EXP-0032_record.md)（本项目的 DivPrune 实现）、[EXP-0033](../实验报告/EXP-0033_record.md)（计划中的组合，导出里没有结果）、[论文主张与证据_草稿](../论文主张与证据_草稿.md)。F72～F74 与 DEC-0061 在这个导出里还没有，按工作流给出的说法引用。这些都不是 MMTok 的内容。

### 逐项对照

| 维度 | MMTok（解析全文位置） | 本项目 | 判断（AI，理由） |
| --- | --- | --- | --- |
| 在哪里剪 | 进 LLM 之前，每层都是 k 个（p.3 §3）；只在 B.6 试过在解码层第 24 层再选一次（p.16） | A 与题目查询都在 LLM 内（K = 2 或 K = 1）；下一轮的 DivPrune-64 底座在 LLM 之前（DEC-0061；EXP-0032） | 与 DivPrune 底座的位置相同，与 K = 1 不同 |
| 预算口径 | 送进 LLM 的视觉 token 数（p.3；p.6 Table 1） | 32 层平均；LLM 之前剪的方法每层 64 个；K = 1 为第 0 层 576 个、之后 47 个，平均 63.5 | 对 LLM 之前的方法两边一致；MMTok 表中 FastV 的口径不明，不能与本项目的 FastV K = 1、K = 2 对齐 |
| 题目信号用的特征 | 文本 token 与投影后视觉 token 的余弦（p.4 §3.1.1）；文本表示推断为 LLM 输入层词向量 | 题目 token 的 LLM 输入层 embedding 与投影层输出的余弦（IDEA-0002 实现公式） | 基本是同一个空间，不新 |
| 题目里的词怎么组合 | 每个文本 token 是一行目标，行 softmax（τ_t = 0.02），对所选 token 取最大再相加（复核时补：Eq.1、Eq.2 在解析里是图块，“取最大再相加”是推断，AI；行 softmax 与 τ_t 是 p.5 正文直接写的）；全文未见按词加权；子词合并没有一致好处（p.5；p.14–15 B.2） | IDF × 可视性加权平均成一个查询 q，按余弦取 n_q = round(count × S) 个，分块轮流（IDEA-0002） | 真实差别：MMTok 没有 IDF，也没有可视性；本项目没有逐词覆盖 |
| 哪些题用题目 | 所有题，α 固定 0.5（p.5） | 只有助动词或 how many 开头的题（IDEA-0003） | 真实差别；MMTok 没有按题型的结果 |
| 只看图像的那一半 | 图像覆盖：让每个原视觉 token 都有一个相似的被选 token（投影前特征，p.4–5） | DivPrune 的最大最小多样性：让被选 token 彼此尽量远（投影层输出、余弦距离、不排除填充区；EXP-0032） | 准则不同；MMTok 自己的表中只用图像覆盖已略高于 DivPrune，POPE 除外（E-MT-26）（复核时改正：这只对做了 softmax 的 Softmax V-V 成立，95.7% 对 94.8%；不做 softmax 的 V-V 是 94.7%，略低于 DivPrune 的 94.8%，p.8 Table 6；原文未限定是哪一个） |
| 两部分怎么合 | 一个联合目标：题目覆盖 + α × 图像覆盖，一起贪心（p.5 Eq.2；p.4 Alg.2） | 手工名额：先按查询取 n_q 个，其余按底座规则；EXP-0033 的 X3 是“其余按最大最小多样性接着选” | MMTok 的做法更统一；本项目的名额要说明理由，或与联合目标比较 |
| “覆盖”一词 | 特征空间里的覆盖 | 覆盖 16：用题目查询定位一个方框，框内比底座多放 16 个（IDEA-0005） | 名字相同、意思不同；论文里要分开写 |
| 填充 patch | MMTok++ 排除（p.16 附录 C）；正文结果没有排除 | A 排除；DivPrune-64 的实现不排除（EXP-0032）；VisionZip-64 平均约 4 个主导 patch 落在填充里（F63） | 不新；可以借到 DivPrune 底座上 |
| K = 1 | 没有 | K = 1 好于 K = 2；DivPrune 放到 K = 1 没有平均的帮助（F72～F74） | 没有重叠 |
| 按题型分析 | 没有：VQAv2 只有总分，消融里没有 VQAv2 | 有：按 VQAv2 答案类型和句式分解 | 真实差别 |
| 64 档 VQAv2 | MMTok 75.20、DivPrune 74.11、VisionZip 72.40（test-dev，p.6 Table 1） | 用验证集里抽的子集；K = 1 路由 + 覆盖 16 与 VisionZip-64 持平（F69） | 划分不同，不能直接比；但 MMTok 表中两者都明显高于 VisionZip-64，而本项目最好的配置在 VQAv2 上没有超过它 |
| 64 档 GQA | MMTok 58.29、DivPrune 57.78、VisionZip 55.10（p.6 Table 1） | testdev：VisionZip-64 55.18；K = 1 路由 + 覆盖 16 为 57.22（比 VisionZip-64 高 2.04，F69） | 见 E-MT-25：VisionZip 两边对得上，本项目最好的已确认配置比 MMTok 表中的 MMTok 低约 1.1、比 DivPrune 低约 0.6 |
| 64 档 POPE | MMTok 85.77、DivPrune 85.56、VisionZip 77.0（F1，p.6 Table 1） | VisionZip-64 77.03；A′+B+V-64 82.74（K = 2，[PAPER-0008 笔记](PAPER-0008_CLASP.md) 引用的本项目数字）；DivPrune 底座的 POPE 没测 | 同为 F1；本项目在 FastV 底座上的题目查询明显低于 MMTok 表中的数字 |
| MME | P+C（p.8、p.18 表头） | MME-P | 不能直接比；CoverPruner 表中 MMTok 的 1371.5 很可能只是感知分（E-MT-24） |
| 统计 | 没有区间，没写次数 | 按图像重抽的区间，预注册确认 | 方法学上的差别，不是方法贡献 |
| 模型 | 5 个，含动态分辨率的 Qwen | 1 个 | 本项目的弱点 |

### 重叠（AI）

1. **如果下一轮在 DivPrune 底座上加题目信息，MMTok 在大方向上已经做了**：在 LLM 之前、免训练，把与题目相关的 token 和一个只看图像的准则（覆盖）合起来选，并在 LLaVA-1.5-7B 的 64 档报告高于 DivPrune（VQAv2 +1.09、GQA +0.51、POPE +0.21，p.6 Table 1）。EXP-0033 的 X3（被路由的题先按题目查询取 round(64 × S) 个，其余按最大最小多样性选）可以看成 MMTok 思路的“分题路由、按名额分配、按 IDF 加权”的版本。
2. **题目相关性的信号相同**：都是投影后视觉 token 与题目 token 表示的余弦（MMTok 的文本表示层是推断）。本项目不能写“首次在 embedding 空间用题目—图像余弦选 token”。
3. **排除填充 patch 也有先例**：MMTok++（v2 附录 C）。
4. **都不信任按注意力排序**：MMTok 说 VisionZip 依赖 [CLS]（p.2），并且在 64 档只用图像覆盖就高于 VisionZip（95.7% 对 93.2%，同 8 项，E-MT-26）（复核时补：95.7% 是 Softmax V-V；不做 softmax 的 V-V 为 94.7%，也高于 93.2%）；本项目测到 K = 1 时多样性选择比按 CLS 排序高约 2 分（F72～F74），IDEA-0007 新加的 CLS 排序也没有作用。方向一致，但设置不同（LLM 之前对 K = 1；平均百分比对准确率）。

### 可辩护的差别（AI，附理由）

1. **什么时候用题目**：MMTok 对所有题用同一个 α，没有按题型的结果。本项目有逐答案类型的证据：在 FastV 底座上，题目查询对 yes/no、计数有利，对问点名物体属性的开放题有害（[PAPER-0008 笔记](PAPER-0008_CLASP.md) 引用的 CL-7、CL-8：VQAv2 test2、64 档、K = 2，A′+B+V − A 的 yes/no +1.65、other −4.03）。理由：这是 MMTok 没有回答的问题。限制：这个证据是在 FastV 底座、手工名额下得到的；换到联合目标或 DivPrune 底座后是否还成立，没有测（UN）。MMTok 的消融中加上题目覆盖后 8 项都上升（Table 6），至少在 benchmark 层面没有看到“有害”。
2. **按词的信息量加权**：MMTok 对每个文本 token 同等对待，只试过子词合并（p.14–15）；本项目的 IDF 让每道题都有的模板词权重为 0。理由：如果说明句的 token 也算作 MMTok 的目标行，它们会各自拉走一些视觉 token（MMTok 是否算进说明句是 UN）。限制：MMTok 是逐词取最大，本来就比“先平均成一个查询”更不容易被模板词淹没（AI）；在逐词覆盖下 IDF 还有没有用，要测。
3. **K = 1 与层平均的计法**：MMTok 完全没有。但本项目把 DivPrune 放到 K = 1 没有平均的帮助（F72～F74），这一条对下一轮的价值有限。
4. **统计**：区间与预注册是方法学上的差别，不是方法贡献。

### 新颖性威胁（AI，分级）

| 本项目可能想写的点 | 威胁 | 理由 |
| --- | --- | --- |
| DivPrune（或别的只看图像的多样性、覆盖）底座 + 题目相关 token，在 LLM 之前、免训练 | 高 | MMTok 已做，并报告高于 DivPrune（p.6 Table 1；p.8 Table 6）；ICLR 2026，arXiv 2025-08（v2 为 2026-03） |
| 用 embedding 空间的题目—图像余弦选 token | 高 | MMTok 的 M^tv（p.4 §3.1.1） |
| 排除填充 patch | 中 | MMTok++（附录 C；只在附录，只报 32～2 个） |
| 按题目句式决定用不用题目 | 对 MMTok 低；对 CLASP、VIP-Router 仍是中（见 [PAPER-0008 笔记](PAPER-0008_CLASP.md)、[PAPER-0009 笔记](PAPER-0009_VIP-Router.md)） | MMTok 没有；Fig.4 只把“按难度定 token 数”列为以后的方向（p.10） |
| IDF × 可视性加权 | 低 | MMTok 没有（B.2 只试了子词合并） |
| K = 1、方框内的空间覆盖 | 低 | MMTok 没有 |
| 逐答案类型的机制证据 | 低 | MMTok 没有按题型的分析 |

### 可借鉴的内容（AI，都要在本项目的开发集上重新验证）

1. **用联合目标代替手工名额**：题目覆盖 + α × 图像覆盖，一起贪心。句式路由可以改写成“α（或 τ_t）随题型变”，可见占比 S 可以改写成每题自己的 α（候选连接，交 Mode C，未验证）。
2. **按行 softmax 加温度**，把两种来源的相似度放到同一尺度（p.5）。
3. **DivPrune 底座排除填充 patch**（MMTok++ 的方向；本项目 EXP-0032 的实现不排除）。按 Table 15 的趋势，64 档的好处可能小于 32 档的 +0.6 个百分点（AI），要测。另一个推断（AI，未检验）：填充 patch 彼此很像，覆盖式的目标会因为“覆盖了很多填充 patch”而倾向选一个填充 patch，所以排除填充对覆盖式准则可能比对 DivPrune 更要紧。
4. **用图像贡献 IC 说明 benchmark 的选择**（p.8）：本项目用的 POPE、MME、GQA 在 LLaVA-1.5-7B 上都是高 IC（0.924、0.918、0.644，Table 4）；VQAv2 没有 IC 值，可以自己算。
5. **极少 token（2～16 个）的压力测试**能拉开选择准则的差别（p.23 Table 21），可以放附录。
6. **多轮对话**：题目相关的选择要回答“换一个问题怎么办”；MMTok 用图像覆盖来论证可复用（Fig.3，p.9–10，定性）。

### 对写作与实验的影响（AI）

- **相关工作**：MMTok 是“题目 + 视觉覆盖、在 LLM 之前、免训练”的最近先例，与 CLASP（按类别调预算）、VIP-Router（学习式逐样本路由）、CoverPruner（按题目加权的覆盖）并列。不写“首次结合题目与视觉多样性”。
- **主表**：以 DivPrune-64 为底座时，MMTok 应作对照。代码有链接，下载要用户许可；没有代码时只能引用 Table 1，并注明 VQAv2 是 test-dev（本项目是验证集子集）、MME 是 P+C（本项目是 MME-P）、单次无区间、对照数字来源不一。
- **能直接比的一点**：GQA testdev 上 VisionZip-64 两边对得上（55.10 对 55.18），MMTok-64 的 58.29、DivPrune-64 的 57.78 都高于本项目已确认最好的 57.22（E-MT-25）。写结果时如实说。
- **一个低成本的检验（建议，未做）**：在开发集上把 MMTok 的两项目标（或只加题目覆盖一项）放到本项目的 DivPrune-64 上，按 VQAv2 答案类型和 GQA 题型拆分。若题目覆盖在开放题上也有害，按句式改 α 是可以辩护的扩展；若不伤，句式路由在这个底座上的余地可能很小。实现需要 Eq.1、Eq.2 的精确形式（解析里没有，要看 PDF 或代码）。
- **预期（AI，未检验）**：MMTok 的消融中，在已经很强的图像覆盖上再加题目项，每项只多 0.1～1.5（MME 为 31 分）（p.8 Table 6）。本项目在 FastV 底座上题目查询带来的大增益（例如 POPE），换到 DivPrune 底座后很可能小得多。
- **其他失败方向的旁证**：MMTok 的缩小图像实验（Table 7）是缩小整张图，与本项目已失败的“预算内放大”（[IDEA-0004](../IDEA-0004_预算内放大.md)，裁剪后重新编码）不是同一种操作，只能当背景。

## 交接 Deep Read

- 候选 P：题目引导的选择对哪些题型有利、对哪些有害，MMTok 没有检验：VQAv2 只有总分，题目覆盖与图像覆盖的消融里没有 VQAv2，也没有按题型拆分（p.6 Table 1；p.8 Table 6）。
- 候选 P：MMTok 的超参数（τ_t、τ_v、α）固定，选取用的数据没写，α 没有消融；“不敏感”只有 16 档自适应 τ_v 的一个对照（p.5；p.14 Table 9）。
- 候选 F：64 档上，只看图像的覆盖（Softmax V-V，95.7%）已高于 DivPrune（94.8%，同 8 项，AI 计算），只有 POPE 低于 DivPrune；再加题目覆盖，8 项都上升（+0.1～+1.5，MME +31）（p.8 Table 6；p.18 Table 16）。
- 候选 F：VisionZip 留 2～8 个、DivPrune 留 2～4 个 token 时，POPE F1 低于不给视觉 token（44.64），MMTok 没有（p.23 Table 21）。
- 候选 F：LLaVA-1.5 上排除填充 patch 的好处随 token 减少而增大（MMTok++，p.17 Table 15）。
- 机制候选见「方法与贡献」，适用条件见「限制与替代解释」。没有自有实验，不向实验台账写入成功或失败结果。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-MMTok-20261030
analysis_id: AP-MMTok-20261030
created_at: '2026-10-30'
source_ids:
- PAPER-0012
- arXiv:2508.18264v2
- https://github.com/Ironieser/mmtok（代码未下载、未核验（需许可））
evidence_status: partial
evidence_locations:
- 'E-MT-1: 解析全文 p.2 §1；p.3 §3；p.10 §5；p.16 B.6'
- 'E-MT-2: p.3–4 §3.1、Eq.1、Prop.1–2；p.5 Corollary 1（公式为图块，形式为推断）'
- 'E-MT-3: p.4 §3.1.1；p.5 实现细节；p.14 B.2'
- 'E-MT-4: p.4–5 §3.1.2；p.5 实现细节'
- 'E-MT-5: p.5 §3.1.3、实现细节；p.7'
- 'E-MT-6: p.3 §3；p.6 Table 1；p.6–7；p.7 Table 3；p.20 Table 18；p.23 Table 21'
- 'E-MT-7: p.5 §4；p.6 §4.1；PAPER-0006 解析全文 p.5 Table 1'
- 'E-MT-8: p.6 Table 1；p.18 Table 16'
- 'E-MT-9: p.2 §1；p.6 §4.1'
- 'E-MT-10: p.6 Table 2；p.7 Table 3；p.19–22 Table 17–20'
- 'E-MT-11: p.8 Table 6'
- 'E-MT-12: p.8 §4.2、Table 4'
- 'E-MT-13: p.8 Table 5；p.23 Table 21'
- 'E-MT-14: p.9 Table 8；p.16 Table 12'
- 'E-MT-15: p.16 B.5、Table 13'
- 'E-MT-16: p.1 摘要；p.9 Table 8；p.17 Table 15；p.23 Table 21'
- 'E-MT-17: p.16 附录 C；p.17 Table 15'
- 'E-MT-18: p.16 B.6、Table 14'
- 'E-MT-19: p.14–15 B.1–B.2、Table 9–10；p.23 Table 21'
- 'E-MT-20: p.7；p.22 Table 20（7 项平均为推断计算）'
- 'E-MT-21: p.8–9、Table 7'
- 'E-MT-22: p.4 §3.1.2；p.9；p.10 §4.3、§5；p.15 B.3'
- 'E-MT-23: p.1；p.10 §6–7；p.14 附录 A'
- 'E-MT-24: CoverPruner 解析全文 p.5 Table 1；MMTok p.6 Table 1（推断）'
- 'E-MT-25: MMTok p.6 Table 1；本项目 EXP-0022、EXP-0027-R05（推断）'
- 'E-MT-26: p.8 Table 6；p.18 Table 16（推断计算）'
missing_evidence:
- 没有读 PDF：只读导出里没有 papers/；数字来自 MinerU flash 解析，引用前要对照 PDF
- Eq.1、Eq.2、Prop.1–2 与附录 B.1 的公式在解析里只有图块，目标函数的精确形式未核对
- Fig.1–5 的图像未看
- 代码未下载、未核验（需许可）；许可证全文未见
- 运行次数、方差、区间、超参数的选取数据、VQAv2 以外的划分、文本 token 的范围、V-V 用的视觉层都没写
- 没有本项目独立运行 MMTok 的结果；没有按题型的拆分
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」
research_question: 能否用最大覆盖作为统一准则，同时用题目文字（覆盖文本 token）和图像本身（覆盖全部视觉 token）在 LLM 之前免训练地选视觉 token？
study_type_and_context: 见「研究设计」与「预算口径」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」
candidate_P: 题目引导的选择对哪些题型有利、哪些有害，MMTok 没有检验（VQAv2 只有总分，消融里没有 VQAv2，无按题型拆分）；超参数固定且选取数据未写，α 没有消融。
candidate_F: 64 档上只看图像的覆盖已高于 DivPrune（同 8 项 95.7% 对 94.8%，推断计算），只有 POPE 例外；加题目覆盖后 8 项都上升（0.1～1.5，MME 31 分）；VisionZip 留 2～8 个、DivPrune 留 2～4 个时 POPE F1 低于不给视觉 token；排除填充 patch 的好处随 token 减少而增大。
evidence_index:
- E-MT-1
- E-MT-2
- E-MT-3
- E-MT-4
- E-MT-5
- E-MT-6
- E-MT-7
- E-MT-8
- E-MT-9
- E-MT-10
- E-MT-11
- E-MT-12
- E-MT-13
- E-MT-14
- E-MT-15
- E-MT-16
- E-MT-17
- E-MT-18
- E-MT-19
- E-MT-20
- E-MT-21
- E-MT-22
- E-MT-23
- E-MT-24
- E-MT-25
- E-MT-26
- E-MT-27
unresolved_questions:
- MMTok 的题目覆盖按 VQAv2 答案类型（yes/no、number、other）与 GQA 题型拆分后，是否在开放题上有害
- 在逐词覆盖下，用 IDF（或 IDF × 可视性）给文本目标行加权是否还有用
- MMTok 的文本目标是否包含系统提示与说明句；V-V 用视觉塔的哪一层；Eq.1、Eq.2 是否对目标行取平均
- MMTok++（排除填充）在 64 档的增益；本项目 DivPrune-64 排除填充后的变化
- 超参数在哪些数据上选定；α 的敏感性
- 在本项目的划分与计分下，MMTok-64 相对 DivPrune-64 的差是否还在
- 摘要中“4 个 token 保留 87.7%”的出处
- 代码是否与论文一致、许可证是什么
```

## 主代理抽查（2026-10-31）

- 写笔记的工作流（2026-10-30）原定由另一个子代理逐条复核，复核因用量上限没有完成；2026-10-31 用户睡觉期间不开子代理（DEC-0063），第二人复核留到之后。本笔记目前只经过主代理的抽查：Table 1 的 MMTok（58.29、75.20、96.6%）、DivPrune（57.78、74.11、94.8%）、VisionZip（55.10、72.40、93.2%）三行与 Table 21 的 POPE 不给视觉 token 44.64 都在解析全文里找到。没有抽查到的数字以 PDF 为准。

## 第二人复核（2026-10-31）

复核方式：另一个子代理从头到尾读完解析全文 p.1–25（大文件分段读，全部读完），只读，没有联网、没有运行代码、没有读 PDF。解析全文的 SHA-256 用 Get-FileHash 复算，前 16 位 4A93BCEAE515E06A、末 6 位 D88766，与笔记一致；页标记 25 个，与笔记一致。

**核对了多少处**：约 260 处，包括：
- Table 1 / Table 16（LLaVA-1.5-7B）三档 × 6 个方法 × 9 项 + 平均，共 162 格逐格对照，笔记表格与解析全文全部一致；Table 1 与 Table 16 逐项相同。用 Table 16 的百分比重算了 MMTok-64（96.59%）、DivPrune-64（94.76%）、VisionZip-64（93.1%）的 9 项平均，与表值一致。
- 「按单项看」里的全部差值（VQAv2、GQA、POPE、MME、SEED 各档）、作者在 p.6 §4.1 的几个差（1.8、0.7、1.6）。
- Table 6 消融 5 行 × 9 个数 + 全部差值（+1.46、+1.09、+0.86、+0.32、+0.27、+0.22、+0.11、+31.00）。
- Table 2、17、18、19 的平均；Table 3、20（Qwen）的数字，并重算了笔记里的 7 项平均（91.1 / 91.2、83.8 / 82.2、74.3 / 70.2）；Table 4、5、21 的 IC 值、5 项平均、POPE 极少 token 的数；Table 7、8、9、10、11、12、13、14、15 的数；Table 9 的 SEED 分母问题（按 66.14 重算为 86.4%）。
- 方法、设置、作者主张、引用页码（p.1–p.25 的页码与章节）、作者自述的局限、伦理与复现声明。
- 附带核对了跨论文的几处：CoverPruner 解析全文 p.5 Table 1 中 MMTok-64 的数字、FastV-64 POPE 35.5、不剪枝 MME 1506.5；VisionZip 解析全文中 1834 / 1761.7 / 1690 / 1756 与 94.0%。这些与笔记一致。
- 没有核对：关于本项目自己实验的数字（按任务要求）；PDF 的字节数与校验和（没有 PDF）。

**改了哪些**（正文里已用“复核时改正 / 补”标出）：
1. 「方法与贡献」文本表示一条：原写“附录 B.2 讨论把一个词的几个子词 token 在算相似度之前合并”；改为“之前或之后”。依据：p.14 B.2 比较了先合再算（Pre）与算完再合（Post）两种。
2. 「消融」第 3 条（与 DivPrune 比）：原写“DivPrune-64 按 Table 16 的百分比平均为 94.8%……逐项只有 POPE 低于 DivPrune”；补充：SQA 两者同为 68.07（持平）；94.8% 是 9 项平均，去掉 VQAv2 重算仍约 94.8%（VisionZip-64 约 93.2%），结论不变。依据：p.6 Table 1、p.8 Table 6。
3. 「与当前项目的关系」逐项对照「只看图像的那一半」：原写“MMTok 自己的表中只用图像覆盖已略高于 DivPrune，POPE 除外”；改正为只对 Softmax V-V 成立（95.7% 对 94.8%），不做 softmax 的 V-V 是 94.7%，略低于 DivPrune 的 94.8%。依据：p.8 Table 6。
4. 「重叠」第 4 条：原写“在 64 档只用图像覆盖就高于 VisionZip（95.7% 对 93.2%）”；补充 95.7% 是 Softmax V-V，不做 softmax 的 V-V 94.7% 也高于 93.2%。依据：p.8 Table 6。
5. 「题目里的词怎么组合」一格：原写“对所选 token 取最大再相加；不按词加权”；改为：“取最大再相加”标为推断（Eq.1、Eq.2 是图块，AI），行 softmax 与 τ_t 是 p.5 正文直接写的；“不按词加权”改为“全文未见按词加权”（全文没有直接说过“不加权”，只是没有提到）。
6. 「限制与替代解释」“正文结果有已知问题”：补充 p.16 附录 C 原文写修正后“准确率明显提高、用的 token 更少”，只是没给具体个数。

没有发现数字抄错的地方；上面几处都是口径或措辞不够准。

**还有哪些读不清**：
- Eq.1、Eq.2、Prop.1–2、B.1 优化式是图块，目标函数的精确形式仍未核对（笔记已标 AI）。
- Table 1、3、5、18、19、20、21、22 的行名和表头是 MinerU 抽出的文字，有错位；本次靠与文字版 Table 16 / 17 的交叉对照确认了 LLaVA-1.5 的数，Qwen 与 NeXT 的行靠百分比重算交叉确认，没有发现矛盾，但仍以 PDF 为准。
- Fig.1–5 没有可读文字；摘要“4 个 token 保留 87.7%”在解析全文的表格里仍找不到（全文只有摘要一处写到 87.7）。笔记的“找不到出处”这句话成立。
- 笔记中关于本项目自己的数字、PDF 字节数与校验和没有核对。

**(a) 位置编号（position ids / RoPE）**：全文未见。全文没有出现 position id、RoPE、位置编码这类词；剪枝后留下的视觉 token 怎样编号（保留原编号、重新连续编号、插值）完全没写。全文里与“位置”字样有关的只有 p.15 Table 10 的 “Position” 一列，指子词合并发生在算相似度之前还是之后（Pre / Post），与位置编号无关。唯一相关的是 p.3 §3 把被选的视觉 token 子集 {v_s}, s∈S 与文本 token 拼接送进 LLM，没有说编号怎么处理。

**(b) 定位类评测**：全文未见。全文没有 RefCOCO / RefCOCO+ / RefCOCOg，没有输出框坐标，没有目标检测或 visual grounding 的评测。所测数据集只有：GQA、MMBench、MME、POPE、ScienceQA(IMG)、VQAv2、TextVQA、MMMU、SEED-Bench（p.5 §4）；Qwen 上另加 OCRBench（p.7 Table 3）；附录另加 MMStar（p.14–15 B.3）。p.17 附录 D 与 p.25 Fig.5 的“选中的 token 与最近的词”可视化是定性展示，不是定位评测。
