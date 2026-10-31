# SCOPE：研究分析笔记

> 没有源码核验补充：代码未下载、未核验（下载需用户许可）。与本项目的逐项对照见「与当前项目的关系」。

分析日期：2026-10-30。本笔记由工作流子代理用 Read 工具读 MinerU flash 档的解析全文后写成，没有看 PDF 本身，没有下载任何材料，没有运行代码或实验。尚未经另一个代理逐页复核。

## 来源与阅读范围

- 标题：SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs（“Multimodel”是原文的拼写，p.1）。方法名 SCOPE。
- 作者/机构：Jinhong Deng、Wen Li（通讯作者）、Joey Tianyi Zhou、Yang He；电子科技大学（UESTC）、电子科技大学深圳高等研究院、新加坡科技研究局（A\*STAR）的 CFAR 与 IHPC。第一作者在 CFAR 实习期间完成本工作（p.1）。
- 版本：arXiv:2510.24214v1 [cs.CV]，2025-10-28（p.1 页边）。页脚写“39th Conference on Neural Information Processing Systems (NeurIPS 2025)”（p.1），文末附 NeurIPS 检查表（p.20–26）。会议正式版与 v1 是否有差别，未知（UN）。
- PDF：`papers/SCOPE-arxiv-2510.24214.pdf`，26 页，2,516,112 字节，SHA-256 `50a1f9a32777dc0f7146ace8a1c49fe58a99e44c97018447a1aff06c81fde14e`，2026-10-29 经用户许可从 https://arxiv.org/pdf/2510.24214 下载（[论文笔记 README](README.md) 的登记）。PDF 不在本次读的只读导出里，没有重算校验和。能访问原文不等于获得再分发许可，全文不提交 Git。
- 解析全文：[`literature/processed/PAPER-0018/SCOPE-arxiv-2510.24214.md`](../../literature/processed/PAPER-0018/SCOPE-arxiv-2510.24214.md)，MinerU 4.0.12 flash 档（只有文本）。本次用 Get-FileHash 重算 SHA-256 为 `359c9ff16d9e4f998d08722dea5e4de701a73397281d8a7817240fd0b447858a`，与 [literature/processed/README](../../literature/processed/README.md) 登记的前 16 位一致。下文的页码 p.N 取解析文本中的 `<!-- page N of 26 -->` 标记。
- 解析质量：
  - 多数公式块是图片引用，没有文字（复核时改正：原写“公式块都是图片引用，没有文字”；式 (1)、(2) 在解析里是零散的文字，如 (1) 的 L(M(S, T), M(V, T)) 与 min S、(2) 的 QK⊤ 与 √d，见 p.4；式 (3)、(5)～(9) 等是图片引用）。可见的式号只有 (1)、(2)、(4)，正文提到式 (6)。每一步怎么算，只能从正文文字和 Algorithm 1（p.6，伪代码是文本）读出。
  - 表格：Table 1、2、3、6、7、8、9 逐格用“原始分数 ÷ 不剪枝分数”复算相对百分比。Table 1 全部对上（FastV-192 的 MMVet 一格除外，那一格是从 VisionZip 原表带来的，见 E-SC-27），说明行列没有错位；Table 2 的 MMMU 一列被解析到方法名前面，按相对值还原后全部对上；Table 6 有两格平均对不上（E-SC-21）。
  - 图（Fig. 1～8）只有图中的零散文字，曲线和柱的数值读不出；下文凡取自图的数字都注明“图中文字”。
- 实际阅读：解析全文从头到尾读完：正文 p.1–10，参考文献 p.11–13，附录 A～F p.14–19，NeurIPS 检查表 p.20–26。
- 为核对对照数字的来源，另读了同一只读导出中 VisionZip（[PAPER-0006 解析全文](../../literature/processed/PAPER-0006/VisionZip-cvpr.md) p.5–6 的 Table 1、2；补充材料只检索了 FastV 一词）与 SparseVLM（[PAPER-0005 解析全文](../../literature/processed/PAPER-0005/SparseVLM-arxiv-2410.04417.md) p.6 的 Table 1）。为对照本项目，读了 [experiment-results](../experiment-results.md)、[已发现的实验现象](../已发现的实验现象.md)、[EXP-0032](../实验报告/EXP-0032_record.md)、[EXP-0033](../实验报告/EXP-0033_record.md)、[查新补充](../专题分析/查新补充_覆盖与多样性_2026-10-26.md) 与 [MODE_C-0002](../MODE_C-0002_视觉token剪枝.md) 中的有关行。任务给出的 F72～F74、DEC-0061 在本次的只读导出（main 337305f）里还没有，下文按任务的表述引用，没有核对原记录。
- 检查表第 16 项：方法开发没有把 LLM 当作重要组成部分（p.26）。
- SC=作者主张，DE=全文中直接写着的设置、表格或数字，AI=本笔记的推断，UN=未知或全文没有。DE 不代表我们复现过。

## 研究问题

按注意力挑“最显著”token 的剪枝方法（例如 VisionZip 取视觉编码器 CLS 注意力最高的 token）会让留下的 token 挤在少数地方，丢掉回答需要的上下文（作者的例子：问“猫在哪里”时，注意力集中在猫身上，忽略了周围环境）；而且注意力分布很偏，排在后面的 token 注意力几乎一样，分不出哪个有用。能否在挑 token 时同时考虑“显著”和“覆盖”（留下的 token 能代表全部 token），在很少的 token 下保住语义完整？（p.1 摘要；p.2 §1；p.4 §3.2）

## 方法与贡献

- **在哪里剪**：视觉编码器之后、进入 LLM 之前剪一次（p.5 Fig. 3 图注写“在送进 LLM 之前减少视觉 token”；p.6 §3.3 “Integration into MLLMs”、p.7 §4.1 写“接在视觉编码器之后”）。[AI] 由此推出 LLM 的每一层都只看留下的 token，原文没有逐字这样写（复核时改正：原写“LLM 的每一层都只看留下的 token”，放在出处后面当作 DE，没有标 AI）。
- **免训练**：作者写明 train-free，不需要额外训练（p.3 贡献三；p.5 Fig. 3 图注；p.6）。
- **记号**：V = {v1, …, vN} 是一张图的全部视觉 token，vi ∈ R^d（p.3 §3.1）。LLaVA-1.5 有 576 个（p.3），从中选 K 个（p.4 Fig. 2 图注：576 个中选 64 个）。
- **每一步怎么算**（p.5–6 §3.3；p.6 Algorithm 1）：
  1. **显著性 A_v**：视觉编码器倒数第二层（layer −2）里，CLS token 对每个视觉 token 的注意力（p.7 §4.1）。多个注意力头怎么合并（求和还是平均），未写（UN）。
  2. **两两相似度**：所有 token 两两的余弦 S_uv，作为 Algorithm 1 的输入（p.6）。用哪一层、哪个空间的特征（编码器第 −2 层的输出，还是投影层之后的输出），没有写清（UN）；“integrated after the vision encoder”（p.6）两种读法都说得通。是否先去均值，未写（UN）。
  3. **覆盖**：一个 token u 被已选集合 S 覆盖的程度，是 u 与 S 中最相似的那个 token 的余弦；集合的覆盖是对全部 token 求和（p.5 式 (6) 前后的文字；Algorithm 1 第 2、9 行用 c_u 记下每个 token 目前最好的相似度）。[AI] 这种写法在文献里叫设施选址（facility location）函数，原文没有用这个名字，也没有给近似保证（检查表说本文没有理论结果，p.21）。c_u 从 0 开始，所以负的余弦不计入（Algorithm 1 第 2、5 行）。
  4. **覆盖增益**：Δ(v; S) = Σ_u [max(S_uv, c_u) − c_u]，即把 v 加进来后总覆盖增加多少（p.6 Algorithm 1 第 5 行；正文按 [14] 的边际增益定义）。
  5. **SCOPE 分数与贪心**：每轮在未选的 token 中取 Δ(v; S) · A_v^α 最大的一个加入 S，更新 c_u，共 K 轮（p.6 Algorithm 1 第 3–10 行）。α 默认 1.0（p.7 §4.1）。[AI] α = 0 时只看覆盖，即消融中的 Coverage-only。
  6. 把选出的 K 个 token 送进 LLM（p.5 Fig. 3 图注）。它们在 LLM 中的顺序与位置编号怎么定，未写（UN）。
- **是否用题目文字**：选择时不用。Algorithm 1 的输入只有视觉 token、K、两两相似度、注意力分数和 α（p.6），显著性取 CLS 注意力（p.7）。[AI] 所以同一张图的不同题目得到同一组 token，题目只进入 LLM。Fig. 3 画了题目 "Is it overcast?"（p.5），解析读不出箭头指向，不改变这个判断。
- **合并**：不合并，只挑原有的 token（Algorithm 1 返回 S ⊆ V、|S| = K，p.6）。原文把“合并相似 token”归于 VisionZip、SparseVLM、PruMerge（p.3 §2）。
- **其他模型上的做法**：LLaVA-NeXT 的 2880 个 token（576 × 5，p.8）是一起选还是按子图选；Video-LLaVA（“following VisionZIP”，p.9）按帧还是整体选；Qwen2-VL（p.16 Table 8）的显著性从哪里取（正文的定义依赖 CLS token），都未写（UN）。
- **作者列出的三点贡献**（p.2–3）：①指出显著性剪枝忽略所选 token 的语义完整，并受注意力分布偏斜之苦；②提出同时建模显著性与覆盖的 SCOPE；③不经训练接入 LLaVA-1.5 与 LLaVA-NeXT，在多个视觉语言基准上取得计算效率与性能的较好折中。

## 研究设计

- **模型**：LLaVA-1.5-7B、LLaVA-NeXT-7B（正文）；LLaVA-1.5-13B、LLaVA-NeXT-13B、Qwen2-VL-7B（附录 C）；视频用 Video-LLaVA（p.7 §4.1；p.9 Table 3）；附录的几个模型在 p.15–16（复核时改正：原写“视频用 Video-LLaVA（p.7 §4.1；p.15–16）”，p.15–16 讲的是 13B 与 Qwen2-VL，不是视频）。
- **图像基准与划分**（p.7 §4.1；p.14–15 附录 B）：
  - GQA：testdev_balanced_instructions，12,578 题。
  - MMBench：约 3,000 道选择题，划分未写。
  - MME：写“dev split including 4,377 samples”，指标写“accuracy”。
  - POPE：test，9,000 题，指标 F1。
  - ScienceQA：准确率，划分未写。
  - TextVQA：写“test split, including 5,000 samples”，指标 EM。
  - SEED-Bench：19,000 道选择题，准确率（Table 1 写 SEED，Table 6、7 写 SEED-I）。
  - MM-Vet：218 题，GPT 打分（哪个 GPT 未写）。
  - 附录另有 DocVQA、ChartQA、OCRBench（Table 9），LLaVA-NeXT 上有 MMMU（Table 2、7）。视频：TGIF、MSVD、MSRVTT、ActivityNet（p.7）。
  - **全文没有 VQAv2**：检索解析全文，“VQA”只出现在 TextVQA、DocVQA 和参考文献的标题里。
  - [AI] 几处与常见设置对不上：MME 的分数是 1862 这样的总分量级，不是准确率；本项目的 MME 是 2374 题，与 4,377 对不上；TextVQA 的 5,000 题与它的验证集题数相同；本项目的 POPE 是 8910 题，这里写 9,000。
- **对照方法**：FastV、SparseVLM、VisionZip、PDrop（p.7 §4.1）；附录在 Qwen2-VL 上对比 DivPrune（p.16 Table 8）。Table 1 只把 PDrop 标为“† adapted from [49]”（[49] 是 SparseVLM，p.7）；其余对照是重跑还是引用，原文没说（UN；数字核对见 E-SC-12）。
- **实现**：基于 lmms-eval；4 × A100；推理 batch size 1（p.7 §4.1）。
- **消融与分析**：Random、Saliency-only（只按 CLS 注意力取，p.5）、Coverage-only、完整方法（p.9 Table 4）；α 扫描（p.17 Fig. 6）；48 / 32 / 16 / 8 个 token 的极端情形（p.9 Fig. 4，只与 VisionZip 比）；θ-覆盖分析（p.4–5 Fig. 2）；选中 token 的可视化（p.10 Fig. 5；p.18–19 Fig. 7–8）。
- **统计**：检查表第 7 项答 [No]，“按以往工作的做法”不报误差条（p.22）。运行次数未写（UN）。[AI] 方法本身没有随机成分，但 MM-Vet 的 GPT 打分会波动。

## 预算与计算量口径

- **“保留 N 个 token”指什么**：SCOPE 在进 LLM 之前剪，N 就是进入 LLM 的视觉 token 数（p.5 Fig. 3 图注；p.7 §4.1）；[AI] 因此 LLM 每层看到的视觉 token 数都是 N（复核时改正：原写“LLM 的 32 层每层都是 N”并标 p.5、p.7；全文未见“32 层”，层数不是这篇论文写的）。表头写“Retain 64 Tokens (↓ 88.9%)”（p.7 Table 1），即 576 个中留 64 个。[AI] 按本项目的层平均口径，SCOPE-64 就是层平均 64，与 VisionZip-64、本项目的 DivPrune-64 同口径。
- **CLS**：只用来算显著性。LLaVA-1.5 送进 LLM 的是 576 个 patch token（p.3），SCOPE 从这 576 个里选（p.4 Fig. 2 图注）。[AI] 所以 CLS 不在候选里，也不占名额。对照的 VisionZip-64 在本项目的实现里是 53 个主导 patch + CLS + 10 个合并的上下文 token（VisionZip 源码核验与 F62、F64；不是本文的内容）。
- **合并 token**：SCOPE 不合并。VisionZip 的 64 个里含合并 token，本文只说 VisionZip 会合并（p.3），没写怎么拆分。
- **与 FastV 一族怎么对齐**：全文没有说明 FastV、SparseVLM、PDrop 这些在 LLM 内剪的方法“保留 64 个”是怎么定义的：在哪一层剪、前面几层看多少、是否按层平均（UN）。
  - [AI] 这几行数字与 VisionZip 论文 Table 1 逐格相同（E-SC-12）；VisionZip 正文的解析全文也只写“Following the setup in [6, 66]”。
  - [AI] 本项目按层平均口径实测的 FastV（K = 2，第 2 层后留 30 / 98 / 166 个）在 POPE 上 F1 为 36.19 / 68.29 / 75.76，准确率为 60.34 / 75.33 / 79.83（EXP-0002-R11 / R09 / R07）；本文表中 FastV 是 48.0 / 59.6 / 64.8（p.7 Table 1）。按 F1 比，64 档本项目更低、128 与 192 档更高；按准确率比，本项目三档都更高。两种指标都对不上，所以表中 FastV 的预算定义或实现与本项目不同，不能套进本项目的层平均口径。
- **FLOPs**：全文没有（检索解析全文没有这个词）。
- **时延**：只有 Table 5（p.10）一处：LLaVA-NeXT-7B，POPE，4 × A100，batch size 1（p.7）。
  - 不剪（2880 个）：POPE 86.4，时延 601.9 s。
  - PDrop 留 160 个：53.2，184.0 s（3.3×）。
  - SCOPE 留 160 个：81.3，188.8 s（3.2×）。
  - 时延量的是整个 POPE 的总时长还是别的、是否含视觉编码器和生成，未写（UN；[AI] 601.9 秒不可能是单题时延）。SCOPE 选择本身的耗时没有单独报告（UN）。没有与 VisionZip 比时延。
  - [AI] 贪心共 K 轮，每轮对每个候选算一遍它对全部 token 的增益，规模约 K × N²。LLaVA-1.5 上 N = 576、K = 64，相对 7B 的 LLM 很小；LLaVA-NeXT 上 N = 2880，N² 约是 LLaVA-1.5 的 25 倍。

## 结果及不确定性

### LLaVA-1.5-7B 主表（Table 1，p.7）

每格是原始分数（MME 是总分；POPE 按附录是 F1，p.14）；“平均”是 8 项“相对不剪枝百分比”的平均（作者说因为各基准的量纲不同，p.8 §4.2）。† 表示数字取自 SparseVLM 论文（p.7 表注）。所有对照都是本文表中的数，不是我们复现的。

| 预算 | 方法 | GQA | MMB | MME | POPE | SQA | TextVQA | SEED | MMVet | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 576 | 不剪 | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 58.2 | 58.6 | 31.1 | 100% |
| 192 | FastV | 52.7 | 61.2 | 1612 | 64.8 | 67.3 | 52.5 | 57.1 | 27.7 | 89.5% |
| 192 | SparseVLM | 57.6 | 62.5 | 1721 | 83.6 | 69.1 | 56.1 | 55.8 | 31.5 | 96.5% |
| 192 | VisionZip | 59.3 | 63.0 | 1783 | 85.3 | 68.9 | 57.3 | 56.4 | 31.7 | 98.0% |
| 192 | PDrop† | 57.1 | 63.2 | 1766 | 82.3 | 70.2 | 56.1 | 54.7 | 30.5 | 96.2% |
| 192 | **SCOPE** | 60.1 | 63.6 | 1804 | 86.4 | 68.8 | 57.7 | 58.7 | 32.5 | 99.5% |
| 128 | FastV | 49.6 | 56.1 | 1490 | 59.6 | 60.2 | 50.6 | 55.9 | 28.1 | 84.4% |
| 128 | SparseVLM | 56.0 | 60.0 | 1696 | 80.5 | 67.1 | 54.9 | 53.4 | 30.0 | 93.3% |
| 128 | VisionZip | 57.6 | 62.0 | 1761.7 | 83.2 | 68.9 | 56.8 | 54.9 | 32.6 | 96.9% |
| 128 | PDrop† | 56 | 61.1 | 1664 | 82.3 | 69.9 | 55.1 | 53.3 | 30.8 | 94.4% |
| 128 | **SCOPE** | 59.7 | 62.5 | 1776 | 86.1 | 68.4 | 57.2 | 57.8 | 31.4 | 98.1% |
| 64 | FastV | 46.1 | 48.0 | 1256 | 48 | 51.1 | 47.8 | 51.9 | 25.8 | 74.9% |
| 64 | SparseVLM | 52.7 | 56.2 | 1505 | 75.1 | 62.2 | 51.8 | 51.1 | 23.3 | 85.1% |
| 64 | VisionZip | 55.1 | 60.1 | 1690 | 77.0 | 69.0 | 55.5 | 52.2 | 31.7 | 93.5% |
| 64 | PDrop† | 41.9 | 33.3 | 1092 | 55.9 | 69.2 | 45.9 | 40.0 | 30.7 | 73.5% |
| 64 | **SCOPE** | 58.3 | 61.7 | 1698 | 83.9 | 68.6 | 56.6 | 56.3 | 30.4 | 96.0% |

- SCOPE-64 的相对值：GQA 94.2%、MMB 95.4%、MME 91.2%、POPE 97.7%、SQA 98.7%、TextVQA 97.3%、SEED 96.1%、MMVet 97.7%（p.7 Table 1）。
- **VQAv2：全文未见。DivPrune：LLaVA-1.5-7B 上全文未见**（只在 Qwen2-VL 上，p.16 Table 8）。
- 作者的概括（SC，p.8 §4.2）：192 个时平均 99.5%，比 FastV、SparseVLM、VisionZip 分别高 +6.0%、+3.0%、+1.5%；64 个时 96.0%，VisionZip 93.5%、SparseVLM 85.1%。[AI] 按 Table 1，192 档对 FastV 是 +10.0（99.5 − 89.5），正文的 +6.0 与表不符；正文说 192 档 POPE 的相对值是 100.2%，表中 192 档是 100.6%，100.2% 是 128 档。
- [AI] 不是每一项都赢：SQA 三档都低于 VisionZip 和 PDrop（64 档 68.6，对 69.0、69.2）；MMVet 在 128、64 档低于 VisionZip（31.4 对 32.6；30.4 对 31.7）；MME 64 档只比 VisionZip 高 8 分（1698 对 1690）。64 档相对 VisionZip 提高最多的是 POPE（+6.9）、SEED（+4.1）、GQA（+3.2）。
- “超过不剪枝”（192 档 POPE 86.4 > 85.9、MMVet 32.5 > 31.1）：作者解释为去掉了冗余信息的干扰（SC，p.8）。替代解释见「限制与替代解释」。

### 消融（Table 4，p.9）

| 方法 | GQA | MMB | MME | POPE | TextVQA |
| --- | --- | --- | --- | --- | --- |
| Random | 55.5 | 54.0 | 1556 | 75.2 | 48.4 |
| Saliency-only | 55.0 | 60.8 | 1665 | 76.8 | 55.4 |
| Coverage-only | 58.1 | 60.8 | 1687 | 82.1 | 56.3 |
| SCOPE | 58.3 | 61.7 | 1698 | 83.9 | 56.6 |

- 表注没有写模型和预算；SCOPE 一行的五个数与 Table 1 的 SCOPE-64 相同，[AI] 应是 LLaVA-1.5-7B、64 个。
- [AI] 只看覆盖（Coverage-only）已拿到 SCOPE 相对 VisionZip-64 的大部分提高：Coverage-only − VisionZip-64（Table 1）为 GQA +3.0、MMB +0.7、MME −3、POPE +5.1、TextVQA +0.8；SCOPE − Coverage-only 只有 +0.2、+0.9、+11、+1.8、+0.3。前一个差混了来源（Coverage-only 是本文自跑的，VisionZip 是引用的数）；都是单次、没有区间。
- [AI] Coverage-only − Saliency-only：GQA +3.1、MMB 0、MME +22、POPE +5.3、TextVQA +0.9。
- 作者说两个消融变体都好于随机（SC，p.9）。[AI] GQA 上 Saliency-only 55.0 低于 Random 55.5，与这句话不符；Random 的 GQA 还高于 Table 1 的 VisionZip-64（55.1）。Random 用了什么种子、跑了几次，未写（UN）。

### 其他结果

- **LLaVA-NeXT-7B**（p.8 Table 2；GQA、MMB、MME、SQA、TextVQA、MMMU 六项平均）：640 / 320 / 160 个时，SCOPE 98.9% / 97.1% / 95.1%，VisionZip 97.4% / 95.0% / 92.5%，SparseVLM 96.0% / 93.6% / 86.9%。160 个时 SCOPE 的原始分数：GQA 60.0、MMB 64.3、MME 1700、SQA 67.4、TextVQA 56.8、MMMU 35.6。
- **极少 token**（p.9 Fig. 4、§4.3）：48 / 32 / 16 / 8 个时在 GQA、POPE、SQA、MME 上与 VisionZip 比，作者说差距随 token 变少而变大；数值从解析文本读不出（UN）。
- **Video-LLaVA**（p.9 Table 3）：2048 → 136 个，四个视频基准的平均 SCOPE 100.5%、VisionZip 93.2%、SparseVLM 86.5%、FastV 52.1%。
- **LLaVA-1.5-13B**（p.15 Table 6，只与 VisionZip 比）：192 / 128 / 64 个时 SCOPE 100.2% / 98.7% / 96.9%，VisionZip 98.7% / 97.0% / 93.7%。[AI] 按表中 VisionZip-128 的 8 个相对值复算，平均约 98.0%，不是 97.0%；VisionZip-64 复算约 93.4%（表中 93.7%）。是原表的错还是解析的问题，要对 PDF。13B 上也不是每项都赢：128 档 MME 1735 对 1743，64 档 SQA 73.2 对 74.4。
- **LLaVA-NeXT-13B**（p.16 Table 7，只与 VisionZip 比）：640 / 320 / 160 个时 SCOPE 98.8% / 96.9% / 95.1%，VisionZip 97.8% / 94.8% / 91.7%。
- **Qwen2-VL-7B**（p.16 C.3、Table 8；GQA / MMB / MME / POPE 四项）：保留 25% 时 DivPrune 93.9%（59.4 / 72 / 2043 / 85.9），SCOPE 94.6%（59.8 / 72.5 / 2065 / 86.5）；保留 10% 时 DivPrune 85.9%（54.3 / 63.7 / 1874 / 80.8），SCOPE 89.6%（56.6 / 66.8 / 1953 / 84.3）。这是全文唯一的 DivPrune 对照；DivPrune 是重跑还是引用、用什么特征，未写（UN）。
- **OCR 类**（p.16–17 C.4、p.17 Table 9，LLaVA-1.5-7B，只与 VisionZip 比；DocVQA / ChartQA / OCRBench；复核时改正：原写“p.17 C.4”，C.4 小节标题在 p.16）：不剪 28.0 / 18.2 / 31.3。64 个时 VisionZip 21.1 / 16.0 / 28.2（84.4%），SCOPE 23.2 / 16.7 / 29.5（89.6%）；128 个时 93.1% 对 95.2%；192 个时 95.8% 对 96.6%。
- **α**（p.17 C.5、Fig. 6）：LLaVA-1.5-7B、64 个，在 GQA、SQA、TextVQA、MMB 上扫 α ∈ {0, 0.5, 1.0, 1.5, 2.0}，作者说多数基准在 α = 1.0 最好；数值读不出（UN）。[AI] 扫描用的就是报告结果的基准，没有单独的验证集；只有这一个超参数，默认值又是自然的 1.0，影响有限。
- **θ-覆盖**（p.4 Definition 1；p.4–5 Fig. 2）：一个 token 若与某个被选 token 的余弦不低于 θ，就算被覆盖；θ-覆盖是被覆盖 token 的比例。MME、64 个，θ 从 0.5 到 0.9：作者文字写：只按显著性选的覆盖在不同 θ 下都偏低，“甚至比随机还差”，SCOPE 明显高于只按显著性选（SC，p.5 §3.2 文字；图在 p.4）。曲线读不出，是否每个 θ 下都低于随机无法确认（复核时改正：原写“只按显著性选的覆盖在各个 θ 下都低于随机”并标 SC/DE；原文没有逐个 θ 说）。Coverage-only 与 SCOPE 哪个更高，曲线读不出（UN）。
- **注意力偏斜**（p.2 Fig. 1(b)）：MME 上按 CLS 注意力排前 128 个的平均分布；后 64 个（图中文字写“64 tail tokens”；“第 65～128 名”是按“前 128 个”推的，AI）的注意力均值 0.0021、标准差 0.0003、最大 0.0029、最小 0.0016（图中文字，解析可能错位）。
- **不确定性**：全文没有误差条或区间（p.22）。[AI] 作为量级参照：本项目在同一个 GQA testdev 划分（12578 题）上，逐题配对差的 95% 区间半宽约 0.5～0.9（例如 +0.87 [+0.37, +1.40]，experiment-results 的 C-0022-1）。SCOPE − Coverage-only 的 GQA +0.2 在这个量级之内。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-SC-1 | DE / AI | 在视觉编码器之后、进 LLM 之前剪一次；免训练（DE）。“LLM 每层都只看留下的 token”由此推出（AI，复核时补标：原整行标 DE） | p.5 Fig. 3 图注；p.6 §3.3；p.7 §4.1；p.3 贡献三 | 没有在 LLM 内剪的对照；“接在编码器后信息保留最多”（p.6）没有实验 |
| E-SC-2 | DE | 显著性 = 视觉编码器第 −2 层 CLS 对各视觉 token 的注意力；α 默认 1.0 | p.7 §4.1 | 多头如何合并未写 |
| E-SC-3 | DE | 贪心：Δ(v; S) = Σ_u [max(S_uv, c_u) − c_u]，每轮取 Δ · A_v^α 最大的，共 K 轮；输入里没有题目 | p.6 Algorithm 1；p.5 式 (6) 前后；p.6 正文 | 公式块解析为图片，只能按伪代码与正文读；没有理论保证（p.21） |
| E-SC-4 | UN | 余弦用哪层、哪个空间的特征，是否去均值；留下的 token 的顺序与位置编号；LLaVA-NeXT 是否按子图；Video-LLaVA 的做法；Qwen2-VL 的显著性从哪里取；是否避开填充区 | 全文通读 | 自己实现时要自己定 |
| E-SC-5 | SC | 只按显著性选的 θ-覆盖在不同 θ（图横轴 0.50～0.90）下都偏低，文字说甚至比随机还差；SCOPE 明显更高（复核时改正：原写“θ = 0.5～0.9 下都低于随机”） | p.4–5 §3.2、Definition 1、Fig. 2 | 只在 MME、64 个；曲线数值读不出；θ-覆盖与覆盖目标在同一空间（AI） |
| E-SC-6 | DE | 按 CLS 注意力排前 128 个中，后 64 个注意力均值 0.0021、标准差 0.0003（最大 0.0029、最小 0.0016） | p.2 Fig. 1(b) | 图中文字，解析可能错位；只在 MME 上 |
| E-SC-7 | DE / AI | “保留 N 个”= 进 LLM 的视觉 token 数，从 576 个 patch 中选；不合并；CLS 只用来算显著性、不占名额（后一句为 AI） | p.3；p.4 Fig. 2 图注；p.5 Fig. 3 图注；p.6 Algorithm 1；p.7 Table 1 | 按本项目口径即层平均 N（AI） |
| E-SC-8 | UN / AI | 在 LLM 内剪的对照（FastV、SparseVLM、PDrop）“保留 N 个”的定义未写；本项目按层平均实测的 FastV（K = 2，留 30 / 98 / 166）POPE F1 36.19 / 68.29 / 75.76、准确率 60.34 / 75.33 / 79.83，与表中 48.0 / 59.6 / 64.8 都对不上 | p.7 Table 1；EXP-0002-R11 / R09 / R07 | 表中 FastV 不能套用本项目的层平均口径 |
| E-SC-9 | DE | 全文没有 FLOPs；时延只有一处：LLaVA-NeXT-7B、POPE、160 个，SCOPE 188.8 s（3.2×）、PDrop 184.0 s（3.3×）、不剪 601.9 s；POPE 81.3 / 53.2 / 86.4 | p.10 Table 5；p.7 §4.1；全文检索 | 时延的范围未定义；选择本身的耗时未报告；没有 VisionZip 的时延 |
| E-SC-10 | DE | LLaVA-1.5-7B 平均：SCOPE 99.5 / 98.1 / 96.0%（192 / 128 / 64），VisionZip 98.0 / 96.9 / 93.5%，SparseVLM 96.5 / 93.3 / 85.1%，FastV 89.5 / 84.4 / 74.9%，PDrop 96.2 / 94.4 / 73.5% | p.7 Table 1；p.8 §4.2 | 8 项相对值的平均；逐格复算一致；无区间 |
| E-SC-11 | DE | Table 1 没有 VQAv2 列，也没有 DivPrune 行；DivPrune 只在 Qwen2-VL 上比 | p.7 Table 1；p.16 Table 8；全文检索 | 本项目的主基准无法对照 |
| E-SC-12 | AI | 不剪、FastV、SparseVLM、VisionZip 的数字与 VisionZip 论文 Table 1 逐格相同（少了 VQAv2、MMMU、LLaVA-Bench 三列；VisionZip-192 的 MME 1782.6 写成 1783）；Table 2 的 SparseVLM、VisionZip 与 VisionZip 论文 Table 2 相同（少了 VQAv2 列）；只有 PDrop 标了来源 | p.7 Table 1；p.8 Table 2；VisionZip 解析全文 p.5–6 的 Table 1、2 | 只有 SCOPE 的行与消融是本文自跑（lmms-eval，p.7）；它自己的不剪分数未报告；两处都删掉了 VQAv2 列，原因未说明（UN） |
| E-SC-13 | AI | SparseVLM 标为 ICML'25，数字却是 VisionZip 表中的“SparseVLM(2024.10)”；64 个时 8 项都低于 SparseVLM 自己 ICML 版（v4）的表（例如 GQA 52.7 对 53.8、POPE 75.1 对 77.5）；本项目复现的 SparseVLM POPE 与 v4 相差不超过 0.2 | p.7 Table 1；SparseVLM 解析全文 p.6 Table 1；[SparseVLM 源码核验](PAPER-0005_SparseVLM_源码核验.md) E-SV-C13 | 对 SparseVLM 不利 |
| E-SC-14 | DE | 消融（GQA / MMB / MME / POPE / TextVQA）：Random 55.5 / 54.0 / 1556 / 75.2 / 48.4，Saliency-only 55.0 / 60.8 / 1665 / 76.8 / 55.4，Coverage-only 58.1 / 60.8 / 1687 / 82.1 / 56.3，SCOPE 58.3 / 61.7 / 1698 / 83.9 / 56.6 | p.9 Table 4 | 表注未写模型与预算，按数字相同推断为 LLaVA-1.5-7B、64 个（AI）；单次、无区间 |
| E-SC-15 | AI | 相对 VisionZip-64 的提高大部分来自覆盖：Coverage-only − VisionZip-64 为 +3.0 / +0.7 / −3 / +5.1 / +0.8；SCOPE − Coverage-only 为 +0.2 / +0.9 / +11 / +1.8 / +0.3 | p.9 Table 4；p.7 Table 1 | 前者混了来源（自跑对引用） |
| E-SC-16 | SC / DE | 作者说两个消融变体都好于随机；GQA 上 Saliency-only 55.0 < Random 55.5 | p.9 正文与 Table 4 | 主张在 GQA 一项上与表不符 |
| E-SC-17 | DE | α ∈ {0, 0.5, 1.0, 1.5, 2.0}（LLaVA-1.5-7B、64 个；GQA、SQA、TextVQA、MMB），多数在 1.0 最好 | p.17 C.5、Fig. 6 | 数值读不出；在报告结果的基准上选 |
| E-SC-18 | SC | 48 / 32 / 16 / 8 个时对 VisionZip 的差距随 token 变少而变大 | p.9 Fig. 4、§4.3 | 数值读不出 |
| E-SC-19 | DE | LLaVA-NeXT-7B：640 / 320 / 160 个时 SCOPE 98.9 / 97.1 / 95.1%，VisionZip 97.4 / 95.0 / 92.5%，SparseVLM 96.0 / 93.6 / 86.9% | p.8 Table 2；p.9 | 6 项平均；MMMU 列按相对值还原；对照来自 VisionZip 论文（E-SC-12） |
| E-SC-20 | DE | Video-LLaVA 2048 → 136 个：SCOPE 100.5%、VisionZip 93.2%、SparseVLM 86.5%、FastV 52.1% | p.9 Table 3 | 视频上的选法未写 |
| E-SC-21 | DE / AI | LLaVA-1.5-13B（只对 VisionZip）：SCOPE 100.2 / 98.7 / 96.9%，VisionZip 98.7 / 97.0 / 93.7%；按 VisionZip 各项相对值复算，128 档约 98.0%、64 档约 93.4% | p.15 C.1、Table 6 | 复算不一致的原因要对 PDF；13B 上也不是每项都赢 |
| E-SC-22 | DE | LLaVA-NeXT-13B（只对 VisionZip）：SCOPE 98.8 / 96.9 / 95.1%，VisionZip 97.8 / 94.8 / 91.7%（640 / 320 / 160） | p.15–16 C.2、p.16 Table 7（复核时改正：原写 p.16 C.2；C.2 小节标题在 p.15） | — |
| E-SC-23 | DE | Qwen2-VL-7B：保留 25% / 10% 时 DivPrune 93.9 / 85.9%，SCOPE 94.6 / 89.6% | p.16 C.3、Table 8 | 唯一的 DivPrune 对照；DivPrune 的来源、实现与特征未写 |
| E-SC-24 | DE | OCR 类（LLaVA-1.5-7B；DocVQA、ChartQA、OCRBench）：SCOPE 96.6 / 95.2 / 89.6%，VisionZip 95.8 / 93.1 / 84.4%（192 / 128 / 64） | p.16–17 C.4、p.17 Table 9（复核时改正：原写 p.17 C.4；C.4 小节标题在 p.16） | 只对 VisionZip |
| E-SC-25 | DE | 不报误差条（检查表第 7 项 [No]） | p.22 | 运行次数未写 |
| E-SC-26 | DE | 划分与指标：GQA testdev_balanced 12,578；POPE test 9,000（F1）；MME “dev 4,377”、写作 accuracy；TextVQA “test 5,000”（EM）；MM-Vet 218（GPT 打分）；lmms-eval，4 × A100，batch 1 | p.14–15 附录 B；p.7 §4.1 | MME、TextVQA 的写法与常见设置不一致（AI） |
| E-SC-27 | AI | 文字与表不一致：192 档对 FastV“+6.0%”（表为 +10.0）；192 档 POPE“100.2%”（表为 100.6%）；p.2 写 9 倍、p.6 写 8 倍；FastV-192 的 MMVet 27.7 应为 89.1%，表写 89.7%（VisionZip 原表即如此）；局限一节只说两个模型 | p.2；p.6；p.7 Table 1；p.8 §4.2；p.18；VisionZip 解析全文 p.5 | 不改变主要结论；正文数字以表为准 |
| E-SC-28 | DE | 作者自述局限：剪得狠时细粒度、少见信息仍会丢；主要只在 LLaVA-1.5 与 LLaVA-NeXT 上验证 | p.18 附录 F | 与附录 C.3（Qwen2-VL）不一致（AI） |
| E-SC-29 | DE | 代码链接 https://github.com/kinredon/SCOPE（摘要说已公开）；检查表说录用后发布；许可证未写 | p.1；p.22 检查表第 5 项；p.24 第 12 项 | 未下载、未核验（需许可） |
| E-SC-30 | AI | 同一 GQA 划分与 POPE 上，本项目的不剪、VisionZip 与表中数字对得上；SCOPE-64 的 GQA 58.3 比本项目目前最好的 57.22 高约 1.1；POPE 83.9 / 86.1 / 86.4 比本项目 A′+B+V 的 82.74 / 84.71 / 85.41 高 1.0～1.4 | p.7 Table 1、p.9 Table 4；EXP-0022-R06 / R07、EXP-0027-R05、EXP-0002-R04 / R13 / R15 / R17、EXP-0015-R03、EXP-0016-R03、EXP-0017-R04 | SCOPE 的行用 lmms-eval，它的不剪分数未报告；双方都是单次；没有复现 SCOPE |

没有源码核验：代码未下载、未核验（需许可）。

## 限制与替代解释

### 作者自述的局限（p.18 附录 F）

1. 剪得很狠时，细粒度或少见的语义信息仍可能丢，影响需要细看的任务。
2. 实验主要在常用的视觉语言基准和两个模型（LLaVA-1.5、LLaVA-NeXT）上，推广到别的任务和模型架构还没有充分验证。[AI] 附录 C.3 已有 Qwen2-VL、正文有 Video-LLaVA，这一节没有跟着改。

另外：检查表第 7 项承认不报误差条（p.22）。附录 E 讨论了更高效的多模态模型可能被滥用（虚假信息、监控等）（p.17–18）；检查表第 11 项说只用公开数据集作学术用途（p.24）。

### 本笔记看到的问题（除标明外都是 AI）

- **对照数字来源不一**：
  - FastV、SparseVLM、VisionZip 与不剪枝的数字与 VisionZip 论文的 Table 1、Table 2 逐格相同（E-SC-12），只有 PDrop 标了来源；SCOPE 自己的行用 lmms-eval 跑（p.7）。SCOPE 在 lmms-eval 下的不剪枝分数没有报告，相对百分比用的是引来的不剪枝分数。
  - SparseVLM 标为“ICML'25”，数字却是 VisionZip 表中的“SparseVLM(2024.10)”，64 个时 8 项都低于 SparseVLM 自己 ICML 版的表（E-SC-13），对 SparseVLM 不利。
  - PDrop 的相对值按 VisionZip 的不剪枝分数算，而 SparseVLM 论文的不剪枝分数不同（例如 SEED 60.3 对 58.6），PDrop 的 SEED 相对值因此偏高（93.3%，SparseVLM 表中为 90.7%）。
  - 本项目复现的 VisionZip 与表中数字很接近（GQA testdev 55.18 对 55.1，POPE F1 77.03 / 83.12 / 85.49 对 77.0 / 83.2 / 85.3），所以 VisionZip 这一行本身可信；可信度低的是 FastV（口径不明，见「预算与计算量口径」）和 SparseVLM（旧版）。
- **缺最强的同类对照**：同类的 DivPrune（CVPR 2025，相关工作 p.3 引了）在 LLaVA-1.5 上没有比，只在 Qwen2-VL 上比（p.16）。本项目自己实现的 DivPrune-64 在开发集上也比 VisionZip-64 高约 2 分（任务给出的 F72～F74），所以 SCOPE 在 LLaVA-1.5 上相对“同类最强”的差距未知（UN）。
- **增益主要来自覆盖**：Coverage-only 与 SCOPE 很接近（E-SC-15），显著性只加 +0.2～+1.8，且是单次、没有区间。“显著性与覆盖互补”（p.9）的证据偏弱。
- **“语义完整”的论证有点循环**：θ-覆盖与 SCOPE 的目标用同一种余弦、在同一个特征空间里算，以覆盖为目标的方法自然得分高（Fig. 2）。它说明只按显著性选的 token 很集中，但不能证明覆盖高是答对的原因。
- **替代解释**：
  - 在 POPE（问某物体在不在）和 GQA 上提高最多，可能主要因为留下的 token 在图上铺得更开，被问的物体（常是小的、不显著的）更可能有 token 留下（Fig. 5 的可视化也是“铺开”，p.10），不一定是“语义完整”。这与本项目 [1a-4b_选择与题目无关漏掉被问的小物体](../1a-4b_选择与题目无关漏掉被问的小物体.md) 的现象同向。
  - 也可能是去掉了重复的背景 token，腾出了名额（与 DivPrune 的思路相同）。
  - “超过不剪枝”可能来自评测流程不同（SCOPE 的行用 lmms-eval，不剪枝分数是引来的）、MM-Vet 的 GPT 打分波动（218 题），或剪枝改变了 POPE 上答“是”的比例（F1 对此敏感）。本项目实测 VisionZip-192 的 POPE F1 85.49，低于不剪枝的 85.90，没有超过。
- **文字与表不一致**：E-SC-27 列出几处。不影响主要结论，但正文的数字要以表为准。
- **只选一次、与题目无关**：作者的局限（1）已承认细节可能丢。被问的小物体若既不显著、又被相似的大片区域“覆盖”，就会被丢掉（推断，未检验）。
- **填充区**：LLaVA-1.5 会把图补成正方形（本项目 F63）。原文没有提填充区。覆盖目标下，大片几乎相同的填充 token 很容易被一个 token 代表、因而增益大，多数补过边的图可能至少留一个填充 token；乘上 CLS 注意力后可能更多（F63：CLS 注意力最高的 53 个里平均约 4 个整块在填充里）。未检验。
- **实现细节缺失**：特征来源、多头合并、位置编号、余弦是否去均值、LLaVA-NeXT 是否按子图（E-SC-4）。按论文自己实现时这些都要自己定，结果会因此不同。
- **计算量证据少**：没有 FLOPs；只有一张时延表，只与 PDrop 比，时延的范围没有定义（E-SC-9）。

## 代码与许可

- 摘要写“Our code is available at https://github.com/kinredon/SCOPE”（p.1）；检查表第 5 项写数据公开，“We will release the code once this work is accepted”（p.22）。两处说法不同；代码现在是否公开、与论文是否一致，未知（UN）。
- 代码未下载、未核验（下载需用户许可）。许可证全文未写（UN）；检查表第 12 项只说引用了原始论文（p.24）。
- 可复现等级：未知（没有访问代码，不等于低）。按论文文字可以自己实现 Algorithm 1，但 E-SC-4 中的几项要自己定。

## 与当前项目的关系

本项目的事实来自任务给出的摘要（F72～F74、DEC-0061 按任务的表述），以及只读导出中的 [experiment-results](../experiment-results.md)、[已发现的实验现象](../已发现的实验现象.md)（F62～F70）、[EXP-0032](../实验报告/EXP-0032_record.md)、[EXP-0033](../实验报告/EXP-0033_record.md)（计划 v1.1；是否已运行，本次未核对）、[查新补充](../专题分析/查新补充_覆盖与多样性_2026-10-26.md) 与 [MODE_C-0002](../MODE_C-0002_视觉token剪枝.md)。不是这篇论文的内容。

### 逐项对照

| 维度 | SCOPE（全文位置） | 本项目 | 层级 |
| --- | --- | --- | --- |
| 在哪里剪 | 视觉编码器之后、LLM 之前，一次（p.5–7） | FastV 一族在 LLM 内（K = 2：第 2 层后；K = 1：第 1 层起）；DivPrune-64 底座与 VisionZip-64 对照在 LLM 之前 | DE |
| 预算口径 | 进 LLM 的视觉 token 数，每层相同；在 LLM 内剪的对照怎么算，未写（p.7） | 32 层平均：K = 2 第 2 层后 30 个（64），K = 1 第 1 层起 47 个（63.5） | DE / UN |
| 选择依据 | CLS 注意力^α × 覆盖增益，贪心（p.6–7） | 被路由的题（助动词或 how many 开头）：题目查询（IDF × 可视性加权的输入层词向量与图像 token 的余弦）；其余：A（去位置先验 + 6×6 分块轮流取 + 排除填充区）；DivPrune：最大最小多样性（投影层输出的余弦距离，EXP-0032） | DE |
| 是否用题目 | 不用（p.6 Algorithm 1） | 用（路由与查询） | DE |
| “覆盖”的意思 | 特征空间里，每个 token 都有一个相似的被选 token（p.4–5） | “覆盖 16”：把不路由题的 16 个名额挪进题目查询定位的框里，是图上局部的加密 | DE；同名不同义（AI） |
| 显著性 | 编码器第 −2 层的 CLS 注意力（p.7） | A：FastV 第 2 层末 token 注意力去位置先验；IDEA-0007 用的就是同一个 CLS 分数（F62～F64、F66） | DE |
| 相似度用的特征 | 未写（UN） | 投影层输出（EXP-0032；按 DivPrune 论文推断） | UN / DE |
| 填充区 | 未提 | A 排除填充区；DivPrune 本实现不排除（EXP-0032）；F63 | UN / DE |
| 按题型分析 | 没有，只按基准报告 | 有（VQAv2 答案类型、句式、GQA 语义类型） | DE |
| VQAv2 | 没有 | 主基准（dev2、test3、test4 等本地子集） | DE |
| GQA | testdev_balanced，12,578 题（p.14） | 同一划分 12578 题（EXP-0022、EXP-0027），另有 val400、val400b | DE |
| POPE | test 9,000 题，F1（p.14） | 8910 题，F1 均值 | DE |
| MME | 总分量级（不剪 1862），写作 accuracy，4,377 题（p.7、p.14） | 2374 题；不剪 P 1507.63、P+C 1865.49（EXP-0002-R02） | DE；1862 应是 P+C（AI） |
| 统计 | 无误差条（p.22） | 按图像重抽的区间；确认集预注册 | DE |
| 模型 | 5 个图像模型 + Video-LLaVA（p.7、p.15–16） | 只有 LLaVA-1.5-7B | DE |
| 代码 | 有链接，未核验（p.1、p.22） | 自写 | DE |

### 数字能不能放在一起看（AI）

- **GQA**：两边是同一个划分（testdev-balanced，12578 题）。本项目的不剪 61.96、VisionZip-64 55.18（EXP-0022-R06、R07），与本文表中（引自 VisionZip 论文）的 61.9、55.1 对得上，说明本项目的 GQA 计分与 VisionZip 论文一致。SCOPE 自己的行用 lmms-eval，它在 lmms-eval 下的不剪分数没有报告，所以下面的比较假设 lmms-eval 的 GQA 计分与此一致（未核对）。
  - 64 个时：SCOPE 58.3、Coverage-only 58.1、Saliency-only 55.0、Random 55.5（p.7 Table 1、p.9 Table 4）；本项目 K = 1 路由 + 覆盖 16 为 57.22（EXP-0027-R05，F69 的所选），K = 1 路由 56.80（EXP-0027-R06），K = 2 路由 55.47、A 54.60（EXP-0022-R02、R01）。
  - 也就是说，SCOPE 报告的数字比本项目目前最好的配置高约 1.1 分；连不看题目、只看覆盖的 Coverage-only 也高约 0.9 分。两边都是单次；本项目的区间只针对与自己的对照之差。
- **POPE（F1）**：本项目的不剪 85.90（EXP-0002-R04）、VisionZip-64 / 128 / 192 为 77.03 / 83.12 / 85.49（R17 / R15 / R13），与表中的 85.9、77.0 / 83.2 / 85.3 对得上。SCOPE 64 / 128 / 192 为 83.9 / 86.1 / 86.4；本项目 A′+B+V（POPE 的题都走路由，所以等于路由；K = 2）为 82.74 / 84.71 / 85.41（EXP-0015-R03、EXP-0016-R03、EXP-0017-R04），低 1.2 / 1.4 / 1.0。即不看题目的 SCOPE（显著性 × 覆盖），在存在类是非题上不比本项目按题目选差；只看覆盖的 Coverage-only 是 82.1（p.9 Table 4），略低于本项目 64 档的 82.74。K = 1 的路由在 POPE 上没有结果（只读导出里没有找到）。
- **MME**：本文的 MME 是总分（不剪 1862 与本项目 P+C 1865.49 相近；VisionZip-64 1690 与本项目 1696.01 相近，EXP-0002-R35），本项目平时用的是 MME-P，不能直接比。
- **VQAv2**：本文没有，本项目的主基准无法对照。
- **FastV**：不能比（见「预算与计算量口径」）。

### 与路由、K = 1、覆盖 16、DivPrune 底座的关系（AI，附理由）

1. **按题目句式路由（[IDEA-0003](../IDEA-0003_按题目句式路由.md)）**：SCOPE 不看题目、不路由，一种规则用于所有题，所以对“按题型路由”没有新颖性威胁（这一点的先例仍是 [CLASP](PAPER-0008_CLASP.md) 与 [VIP-Router](PAPER-0009_VIP-Router.md)）。但有一个需要注意的重叠：SCOPE 在 64 个时相对 VisionZip 提高最多的是 POPE（存在类是非题）、SEED 和 GQA；本项目的 DivPrune-64 增益也主要在 yes/no 和计数题（任务给出）；而本项目的路由恰好把这两类题交给题目查询。理由：同一类题上，覆盖 / 多样性底座已经补上了一部分本来要靠题目查询补的东西。推论：在覆盖 / 多样性底座上，路由的增量可能变小，甚至变负（未检验；EXP-0033 的 X3 正是这个检验）。SCOPE 没有按题型拆分，从论文里查不到（UN）。
2. **K = 1**：SCOPE 只在 LLM 之前剪，没有比较剪的位置，对 K = 1 既不支持也不反对。p.6 说接在编码器之后是为了“最大化信息保留”，这是主张，没有实验（SC）。本项目“多样性选择放到 K = 1 没有平均的帮助（VQAv2 −1.51、GQA +0.88）”与 DivPrune 自己的 Table 3（同样 TFLOPs 下在第 1 层后选比在 LLM 之前选差，查新补充）大体同向（VQAv2 上更差，GQA 上略好），与本文无关。
3. **覆盖 16（[IDEA-0005](../IDEA-0005_定位后覆盖.md)、[IDEA-0006](../IDEA-0006_强底座上的定位后覆盖.md)）**：两个“覆盖”意思相反。SCOPE 的覆盖是整张图、特征空间、与题目无关，目的是把 token 铺开；本项目的覆盖 16 是在题目定位的框里加密，目的是集中。理由：定义不同（p.4–5 对 F69 的配置）。所以 SCOPE 不构成威胁，但写作时要换个说法或每次加限定。若以后在覆盖 / 多样性底座上加“框内加密”，可以用 SCOPE 的 θ-覆盖量一下整体覆盖损失了多少。
4. **DivPrune 底座（DEC-0061）**：SCOPE 是 DivPrune 之后最近的一步：同属覆盖 / 多样性一类，多乘了一项显著性；它在 Qwen2-VL 上报告比 DivPrune 高 +0.7 / +3.7（保留 25% / 10%，p.16 Table 8），在 LLaVA-1.5 上没比。它与本项目 DivPrune 实现的两处不同：目标（各 token 最大相似度之和，对最大最小距离）和显著性加权。
   - 方向一致的地方：Table 4 中只看覆盖比只看 CLS 显著性高（GQA +3.1、POPE +5.3，LLM 之前、64 个）；本项目 K = 1 时多样性比 CLS 排序高约 2 分（F72～F74）。两边都说明，在很少的 token 下，“铺开”比“取 CLS 注意力最高的”更重要。
   - 对“底座加显著性”的提醒：SCOPE 中显著性只在覆盖之上加 +0.2～+1.8（单次）；本项目 [IDEA-0007](../IDEA-0007_一层全看再按编码器排序.md) 加编码器排序也没有可检出的作用（F66），K = 1 时两种排序留下的 token 只有约三分之一相同、准确率却没有差别（F70）。所以在 DivPrune 底座上乘 CLS 注意力，预期增益小；而且要先处理填充区（F63）。
   - 结构上的相似：本项目的强对照 A = 去位置先验的注意力（显著性）+ 6×6 分块轮流取（图上的空间覆盖），本身就是一种“显著性 + 覆盖”，只是覆盖在图上而不是在特征空间、显著性来自 LLM 第 2 层而不是 CLS。所以“显著性 + 覆盖”这个大思路不能算作本项目的新意。

### 新颖性威胁（AI）

| 本项目可能写的 | SCOPE 是否已做 | 判断 |
| --- | --- | --- |
| 按题目句式路由 | 没有，不看题目 | 不受 SCOPE 威胁 |
| K = 1 的放置 | 没有 | 不受 SCOPE 威胁；先例是 DivPrune Table 3、ShaRP（查新补充） |
| “重要性 × 多样性”的组合（EXP-0033 的 X2：在 CLS 注意力前 128 个里按 DivPrune 选 64 个） | 做了同一思路的另一种形式：覆盖增益乘 CLS 注意力^α（p.6），NeurIPS 2025 | **威胁高**：X2 只能作探索性组合；写成方法必须引 SCOPE，并与 SCOPE 在同一流程中比较 |
| “64 个时覆盖 / 多样性一类好于 VisionZip”这一现象 | 已报告（p.7 Table 1、p.9 Table 4），DivPrune、CoverPruner 等也有 | 只能写成本流程中的复现，不是新发现 |
| K = 1 时多样性好于 CLS 排序（F72～F74） | 只在 LLM 之前比过（Coverage-only 对 Saliency-only） | 方向相同、放置不同；算补充证据，不是新发现 |
| 题目相关 × 多样性（EXP-0033 的 X3） | 不用题目 | 不受 SCOPE 威胁；但 MODE_C-0002 的 C12 已记 CDPruner 做了“题目相关 + 多样性”，查新补充记 CoverPruner 做了按题目加权的覆盖（都只读过摘要或 HTML 摘录），威胁来自它们 |
| 覆盖 16（框内加密） | SCOPE 的“覆盖”是另一回事 | 不受威胁；术语要区分 |
| 用 θ-覆盖作诊断 | SCOPE 提出（p.4 Definition 1） | 借用要引 |

### 能借鉴什么（建议，未做；都免训练）

1. **把 SCOPE 当作同一流程里的直接对照**：按 Algorithm 1 自己实现（与 DivPrune-64 同样的投影层特征、同样在 LLM 之前），跑 α = 1（SCOPE）与 α = 0（只看覆盖）两个配置，先在开发集（VQAv2 dev2、GQA val400）上与 DivPrune-64、VisionZip-64 比。理由：它在 Qwen2-VL 上报告高于 DivPrune，又和下一轮的底座同类；论文主表很可能要放它。代价：2 个配置 × 2 个集，按 EXP-0032、EXP-0033 的每次约 13～14 分钟估，约 1 GPU 小时。官方代码要下载须经用户许可。照本项目的规矩，开发集上看过之后，要写成方法须在新抽的数据上预注册确认（P14）。
2. **同时跑排除填充区的版本**（A 已有排除填充区的实现），检查 CLS 加权是否被填充区带偏（F63）。
3. **离线算 θ-覆盖**（p.4 Definition 1）：在已保存的保留 token 记录上，比较 VisionZip-64、DivPrune-64 和路由的被路由题，看覆盖与逐题对错的关系，以及题目查询选法会让整体覆盖降多少。需要投影层特征；记录里有没有，未核对（UN）。
4. **把题目信号放进 SCOPE 的乘法形式**：用 Δ(v; S) × r_v^α 代替 CLS 显著性，r_v 取本项目的题目查询余弦（调成正数），作为 X3“先按查询取 round(64 × S) 个、再按多样性补满”的软版本。这只是想法；进 Mode C 与 discover-innovation 之前，要先读 CDPruner、CoverPruner、MMTok 的原文（PAPER-0011～0013 已下载、未写笔记）查新。
5. **写作上的旁证**：Fig. 1(b)“排在后面的 token 注意力几乎一样”可以和本项目 F70（K = 1 时两种排序留下的 47 个只有约三分之一相同，准确率没有差别）放在一起，作为“不必细排注意力”的旁证，注明设置不同（图中文字，只有 MME 上的一张图）。

### 对写作与实验的影响（AI）

- **相关工作**：把 SCOPE 写成“免训练、与题目无关、在 LLM 之前的显著性 × 覆盖”，与 DivPrune、CoverPruner 同属覆盖 / 多样性一类；NeurIPS 2025。
- **不要写“优于 SCOPE”**：在 GQA testdev 与 POPE 上，它报告的 64 档数字比本项目目前最好的配置高约 1 分（上一节）。若主表放它，最好在本流程中自己实现后比（同一评测、同一不剪分数、同样的区间）。
- **引用它的对照数字时注明**：FastV、SparseVLM、VisionZip 与 VisionZip 论文的表相同（SparseVLM 是 2024.10 版），PDrop 来自 SparseVLM 论文；FastV 的预算口径不明；没有 VQAv2。
- **下一轮的读法**：若 DivPrune 底座上的路由（X3）增量很小，SCOPE 的结果提示一个原因：覆盖类底座已经补上了是非、计数题的一部分；按题型拆开看（本项目已有这套分组）可以分辨。

## 交接 Deep Read

- 候选 P：只按显著性（CLS 注意力）挑出的 token 挤在少数区域，θ-覆盖偏低、文字说甚至比随机还差（图在 p.4 Fig. 2，文字在 p.5，SC；复核时改正：原写“θ-覆盖低于随机（p.5 Fig. 2）”）；按注意力排在后面的 token 几乎分不开（p.2 Fig. 1(b)）。但 θ-覆盖与覆盖目标在同一特征空间里算，不能证明覆盖是答对的原因（AI）。
- 候选 F：LLaVA-1.5-7B、64 个时，与题目无关的覆盖选择（只看覆盖）已比 VisionZip-64 高 GQA +3.0、POPE +5.1，再乘 CLS 显著性只多 +0.2～+1.8（p.9 Table 4 对 p.7 Table 1；单次、混了来源，AI）；提高集中在 POPE、SEED、GQA，MME、SQA、MMVet 几乎不变或更低（p.7 Table 1）。
- 候选连接：覆盖 / 多样性底座提高最多的题型（SCOPE 的 POPE 存在类是非题；本项目 DivPrune-64 的 yes/no 与计数题）与本项目路由的目标题型重合，路由在这类底座上的增量可能变小（AI，待 EXP-0033 的 X3 检验）。
- 机制候选见「方法与贡献」；适用条件与替代解释见「限制与替代解释」。没有自有实验，不向实验台账写入成功或失败结果。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-SCOPE-20261030
analysis_id: AP-SCOPE-20261030
created_at: '2026-10-30'
source_ids:
- PAPER-0018
- arXiv:2510.24214v1
- https://github.com/kinredon/SCOPE（代码未下载、未核验（需许可））
evidence_status: partial
evidence_locations:
- 'E-SC-1: 解析全文 p.5 Fig.3 图注；p.6 §3.3；p.7 §4.1；p.3 贡献三'
- 'E-SC-2: p.7 §4.1'
- 'E-SC-3: p.6 Algorithm 1；p.5 式(6) 前后'
- 'E-SC-4: 全文通读（UN）'
- 'E-SC-5: p.4–5 §3.2、Definition 1、Fig.2'
- 'E-SC-6: p.2 Fig.1(b)（图中文字）'
- 'E-SC-7: p.3；p.4 Fig.2 图注；p.5 Fig.3 图注；p.7 Table 1'
- 'E-SC-8: p.7 Table 1；本项目 EXP-0002-R11、R09、R07（推断）'
- 'E-SC-9: p.10 Table 5；p.7 §4.1'
- 'E-SC-10: p.7 Table 1；p.8 §4.2'
- 'E-SC-11: p.7 Table 1；p.16 Table 8；全文检索'
- 'E-SC-12: p.7 Table 1；p.8 Table 2；VisionZip 解析全文 p.5–6（推断）'
- 'E-SC-13: p.7 Table 1；SparseVLM 解析全文 p.6 Table 1；E-SV-C13（推断）'
- 'E-SC-14: p.9 Table 4'
- 'E-SC-15: p.9 Table 4；p.7 Table 1（推断）'
- 'E-SC-16: p.9 正文与 Table 4'
- 'E-SC-17: p.17 C.5、Fig.6'
- 'E-SC-18: p.9 Fig.4、§4.3'
- 'E-SC-19: p.8 Table 2；p.9'
- 'E-SC-20: p.9 Table 3'
- 'E-SC-21: p.15 C.1、Table 6'
- 'E-SC-22: p.15–16 C.2、p.16 Table 7'
- 'E-SC-23: p.16 C.3、Table 8'
- 'E-SC-24: p.16–17 C.4、p.17 Table 9'
- 'E-SC-25: p.22 检查表第 7 项'
- 'E-SC-26: p.14–15 附录 B；p.7 §4.1'
- 'E-SC-27: p.2；p.6；p.7 Table 1；p.8 §4.2；p.18（推断）'
- 'E-SC-28: p.18 附录 F'
- 'E-SC-29: p.1；p.22 检查表第 5 项；p.24 第 12 项'
- 'E-SC-30: p.7 Table 1、p.9 Table 4；本项目 EXP-0022、EXP-0027、EXP-0002、EXP-0015～0017（推断）'
missing_evidence:
- 没有看 PDF，只读 MinerU flash 档的解析文本；公式块与图中数值读不出
- 代码未下载、未核验（需许可）；许可证未知；代码是否已公开未知
- 相似度所用特征、多头合并、位置编号、填充区处理、LLaVA-NeXT、视频与 Qwen2-VL 上的具体做法未写
- FastV、SparseVLM、PDrop 的预算定义未写；对照数字多为引用
- 没有 VQAv2；LLaVA-1.5 上没有 DivPrune 对照；没有误差条；没有 FLOPs；时延的范围未定义
- 没有本项目独立运行 SCOPE 的结果
- 任务给出的 F72～F74、DEC-0061 在只读导出中没有，未核对原记录
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」
research_question: 显著性剪枝丢失语义完整、尾部注意力分不开；同时考虑显著性与覆盖的免训练选择，能否在很少的视觉 token 下保住性能？
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「预算与计算量口径」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「代码与许可」与「与当前项目的关系」
candidate_P: 只按 CLS 显著性挑出的 token 集中、θ-覆盖偏低、文字说甚至比随机还差（图在 p.4 Fig.2，文字 p.5，SC；复核时改正：原写“低于随机（p.5 Fig.2）”），尾部注意力几乎分不开（p.2 Fig.1(b)）；θ-覆盖与覆盖目标同空间，不能证明覆盖是答对的原因（推断）。
candidate_F: LLaVA-1.5-7B、64 个时，只看覆盖已比 VisionZip-64 高 GQA +3.0、POPE +5.1，再乘 CLS 显著性只多 +0.2～+1.8（p.9 Table 4 对 p.7 Table 1，单次，推断）；提高集中在 POPE、SEED、GQA。
evidence_index:
- E-SC-1
- E-SC-2
- E-SC-3
- E-SC-4
- E-SC-5
- E-SC-6
- E-SC-7
- E-SC-8
- E-SC-9
- E-SC-10
- E-SC-11
- E-SC-12
- E-SC-13
- E-SC-14
- E-SC-15
- E-SC-16
- E-SC-17
- E-SC-18
- E-SC-19
- E-SC-20
- E-SC-21
- E-SC-22
- E-SC-23
- E-SC-24
- E-SC-25
- E-SC-26
- E-SC-27
- E-SC-28
- E-SC-29
- E-SC-30
unresolved_questions:
- SCOPE 的余弦用编码器第 −2 层的输出还是投影层之后的输出；多头如何合并；留下的 token 的位置编号；会不会选到填充区
- 同一流程中 SCOPE-64、只看覆盖（α = 0）与 DivPrune-64、VisionZip-64 在 VQAv2、GQA 上差多少
- SCOPE 的增益按题型（是非、计数、其他）怎么分布；在覆盖类底座上，句式路由还剩多少增量
- 表中 FastV、SparseVLM、PDrop 的“保留 N 个”是怎么定义的
- Table 6 中 VisionZip 平均与复算不一致的原因；Table 5 的时延量的是什么
- 代码是否已公开、与论文是否一致、用什么许可证
```

## 主代理抽查（2026-10-31）

- 写笔记的工作流（2026-10-30）原定由另一个子代理逐条复核，复核因用量上限没有完成；2026-10-31 用户睡觉期间不开子代理（DEC-0063），第二人复核留到之后。本笔记目前只经过主代理的抽查：摘要的 96.0% 与 Table 4 的 Coverage-only 一行（GQA 58.1）在解析全文里找到。没有抽查到的数字以 PDF 为准。

## 第二人复核（2026-10-31）

复核人：另一个子代理，只读解析全文（D:/6CJNM007/tmp/read_repo/main_54bf807 的只读导出）；没有看 PDF，没有联网、下载或运行代码。本节之前的正文只改了下面列出的地方，每处改动在正文里带“（复核时改正：原写 …）”。上一节“主代理抽查”保留原文。

### 核对了多少

- 解析全文从头读到尾（p.1–26 全部）。约核对 300 处：
  - Table 1 全部：15 行 × 8 项原始分数逐格对笔记；120 个相对百分比与 15 个平均逐格用“原始分数 ÷ 不剪枝分数”复算。
  - Table 2、3、4、5、6、7、8、9 中笔记引用的数字；Table 6、7 的平均也复算过。
  - 附录 B 的划分与指标；检查表第 5、7、11、12、16 项；Fig. 1(b) 图中文字；Algorithm 1 的行号；页标记与章节出处。
  - 笔记里“方法与贡献”“研究设计”“预算与计算量口径”“结果”“证据与实现核对”“限制与替代解释”“与当前项目的关系”“交接 Deep Read”“ANALYSIS_PACKAGE”中关于这篇论文本身的说法（约 80 条）。
- 另外对照了同一导出里的 VisionZip 解析全文（PAPER-0006 p.5–6 的 Table 1、2）和 SparseVLM 解析全文（PAPER-0005 p.6 表格行）：E-SC-12 说 SCOPE 表里 FastV / SparseVLM / VisionZip / 不剪的数字与 VisionZip 论文逐格相同（含 VisionZip-192 的 MME 1782.6 写成 1783，FastV-192 的 MMVet 89.7%）——对上了；E-SC-13 与“限制”里 SparseVLM-64 的 GQA 53.8、POPE 77.5，以及 PDrop 的 SEED 90.7% 对 93.3%——对上了。
- 用只读命令重算了解析全文的 SHA-256，与笔记和 literature/processed/README 登记的前 16 位一致。PDF 本身没有读，其校验和与页数 26 只核对了登记表和页标记。
- 任务里提到的“本项目已引用说法的逐条核对”一节：这份笔记里没有这个标题的小节（全文检索“逐条核对”没有结果）。我核对的是上面列出的各节里关于这篇论文本身的说法。关于本项目自己实验的数字（EXP、F 号、DEC）按任务要求没有核对。

### 数字有没有错

- 没有发现数字错误。LLaVA-1.5-7B 的 Table 1 每一格、相对值和平均都与笔记一致；对照方法（FastV、SparseVLM、VisionZip、PDrop）的数字、预算口径的说法（进 LLM 之前剪、表头 Retain 64 Tokens ↓88.9%、FastV 等“保留 N 个”的定义全文未写、没有 FLOPs、只有 Table 5 一处时延）都与全文一致。
- 笔记自己标出的文字与表不一致（正文 +6.0% 对表 +10.0；192 档 POPE 100.2% 对表 100.6%；p.2 写 9 倍、p.6 写 8 倍；FastV-192 MMVet 89.7% 对复算 89.1%；Table 6 的 VisionZip 平均 97.0% 对复算 98.0%、93.7% 对复算 93.4%）我都独立复算过，说法成立。补充一点：正文说的 100.2% 在 Table 1 里出现在 128 档的 POPE，也出现在 192 档的 SEED。

### 改了哪些（改前 → 改后，依据）

1. 解析质量一节：“公式块都是图片引用，没有文字” → “多数公式块是图片引用”，补充式 (1)、(2) 在解析里是零散文字。依据：p.4 解析文本里有 L(M(S, T), M(V, T))、min S、QK⊤、√d 和式号。
2. “方法与贡献”的“在哪里剪”：“LLM 的每一层都只看留下的 token”原放在出处后面，像是原文直接写的 → 改为 AI（由“在送进 LLM 之前剪”推出）；E-SC-1 的层级 DE → DE / AI。依据：p.5 Fig. 3 图注、p.6 §3.3、p.7 §4.1 只写“送进 LLM 之前减少”“接在视觉编码器之后”，没有逐字写每层。
3. “预算与计算量口径”：“LLM 的 32 层每层都是 N（p.5、p.7）” → 去掉“32 层”并标 AI。依据：全文检索“32”只出现在分数和 Fig. 4 的 32 个 token 里，层数全文未见。
4. θ-覆盖（共 4 处：“结果及不确定性”的 θ-覆盖条、E-SC-5、交接里的候选 P、ANALYSIS_PACKAGE 的 candidate_P）：“只按显著性选的覆盖在各个 θ 下都低于随机，标 SC/DE” → “在不同 θ 下都偏低，文字说甚至比随机还差（SC）”，并说明曲线读不出、无法确认每个 θ 都低于随机。依据：p.5 §3.2 文字说覆盖在不同 θ 下一直偏低，并用了“even performing worse than the random selection baseline”这样的话，没有逐个 θ 说；图在 p.4，曲线数值读不出。
5. 同上几处里的页码：Fig. 2 写“p.5” → 图在 p.4、文字在 p.5。依据：p.4 页标记之后是 Fig. 2 及其图注。
6. Fig. 1(b) 的尾部统计：“后 64 个（第 65～128 名）” → 补注“图中文字写 64 tail tokens；第 65～128 名是按‘前 128 个’推的（AI）”。依据：图里只有“Statistics of 64 tail tokens”与数值，没有写名次。
7. “研究设计”的模型一行：“视频用 Video-LLaVA（p.7 §4.1；p.15–16）” → “（p.7 §4.1；p.9 Table 3）；附录的几个模型在 p.15–16”。依据：Video-LLaVA 在 p.7 和 p.9；p.15–16 讲的是 13B 与 Qwen2-VL。
8. E-SC-22（LLaVA-NeXT-13B）出处：“p.16 C.2、Table 7” → “p.15–16 C.2、p.16 Table 7”（ANALYSIS_PACKAGE 同改）。依据：C.2 小节标题在 p.15 末，表在 p.16。
9. E-SC-24 与“其他结果”的 OCR 类条（OCR 类）出处：“p.17 C.4” → “p.16–17 C.4、p.17 Table 9”（ANALYSIS_PACKAGE 同改）。依据：C.4 小节标题在 p.16 末，表在 p.17。

### 还有哪些读不清（没有改）

- Fig. 1(c) 的图中文字里有 97.8 和 97.2 两个数，名次顺序错乱；Table 1 的 SCOPE-64 里 MMVet 是 97.7%、TextVQA 是 97.3%。差 0.1，可能是图与表取整不同，解析读不出对应关系，笔记没有引用图里这两个数，所以没改。
- 公式 (3)、(5)～(9) 是图片，Algorithm 1 以外的式子只能按正文文字读。
- Fig. 2、4、6 的曲线读不出（笔记已写）。
- E-SC-13 中 SparseVLM 自己的论文表（PAPER-0005 p.6）的表号归属：解析里表注排在表格后面，我只核对了表格行的数字，没有逐一确认它属于哪个表号。
- PDF 本身、SparseVLM / VisionZip 两篇的 PDF 都没有看。

### 两条专门的问题

(a) 剪枝后留下的视觉 token 的位置编号（position ids / RoPE 位置）：全文未见。检索 position、RoPE、rotary、positional、order、index、indices 都没有相关内容（匹配到的只是 compositional 等词的一部分）。Algorithm 1 返回的是集合 S（p.6），没有写送进 LLM 时的顺序，也没有写保留原始编号、重新连续编号或插值。笔记 E-SC-4 与“方法与贡献”第 6 点的 UN 成立。

(b) 定位类任务（RefCOCO 一类、输出框坐标、visual grounding）：全文未见。检索 RefCOCO、grounding、bounding、bbox、detection、localiz、coordinate 都没有相关内容。图像任务只有 GQA、MMBench、MME、POPE、ScienceQA、TextVQA、SEED-Bench、MM-Vet（p.7 §4.1、p.14–15）；附录另有 MMMU（Table 2、7）、DocVQA、ChartQA、OCRBench（Table 9）；视频有 TGIF、MSVD、MSRVTT、ActivityNet（p.7、p.9）；Qwen2-VL 上只有 GQA、MMB、MME、POPE 四项（p.16 Table 8）。没有一项要求输出框坐标。
