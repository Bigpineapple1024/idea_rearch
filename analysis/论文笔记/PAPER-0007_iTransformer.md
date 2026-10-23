# iTransformer：研究分析笔记

> 源码核验与本项目实测见补充笔记 [PAPER-0007_iTransformer_源码核验](PAPER-0007_iTransformer_源码核验.md)（证据 ID 带 C 前缀）。

分析日期：2026-10-23。本笔记由实验室电脑上的 Claude Code 会话按 analyze-paper 工作流读 PDF 写成（交接文档第 8 步），与前六篇由用户在另一会话中写成的原文笔记来源不同。

## 来源与阅读范围

- 标题：iTransformer: Inverted Transformers Are Effective for Time Series Forecasting。
- 作者/机构：Yong Liu、Tengge Hu、Haoran Zhang（共同一作）、Haixu Wu、Shiyu Wang、Lintao Ma、Mingsheng Long；清华大学软件学院、蚂蚁集团。
- 版本：arXiv:2310.06625v4，2024-03-14；ICLR 2024（PDF 页眉为会议版）。
- [原文](https://arxiv.org/pdf/2310.06625v4)；[固定源码](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124)（分支 `main`，MIT）。
- PDF：`papers/iTransformer-arxiv-2310.06625.pdf`，25 页；SHA-256：`83e889795af52c672090255493266e3ea93e7b2a32c528f58074f2f9f757ffc5`。全文未提交 Git。
- 实际阅读：正文第 1～9 页、附录 A（实现与数据）、B（消融）、C（超参数敏感性）及 Table 1、3、4、5、10；未读附录 D～F 的全部图表与参考文献。提取工具 PyMuPDF，表格数字逐行核对过；不是 MinerU 解析。
- SC=作者主张，DE=直接查到的表格/代码内容，AI=分析推断，UN=未知。DE 不代表我们已复现实验。

## 研究问题

Transformer 预测器把同一时刻的多个变量嵌成一个 token，再在时间维上做注意力。这样做是否用错了 Transformer 的组件？改为把每个变量的整段序列当作一个 token，能否在不改任何组件的前提下得到更好的预测？

## 方法与贡献

- **倒置结构**（E-IT-1）：每个变量长度 T 的回看序列经 MLP 嵌成一个 D 维 token。自注意力在 N 个变量 token 之间建模多变量相关；FFN 对每个 token 独立编码序列表示；LayerNorm 作用在每个变量 token 的特征维上。仅编码器，没有位置编码，最后用 MLP 投影到 S 步。
- **作者主张的贡献**（SC，E-IT-7）：
  - 不修改组件、只换维度就能达到 SOTA；
  - 这一框架能提升 Transformer 及其高效变体（平均 MSE 降低 16.8%～38.9%）；
  - 能泛化到训练时没见过的变量；
  - 回看窗口加长时性能随之提高；
  - 注意力图可解释为变量相关。

## 研究设计

- **研究类型**：计算模型基准研究。
- **数据**：ECL、ETT 四个子集、Exchange、Traffic、Weather、Solar-Energy、PEMS 四个子集；附录另有 Market 六个子集。按时间顺序沿用 TimesNet 的划分，ETTh 的规模为 (8545, 2881, 2881)（E-IT-8）。
- **设置**：回看固定 96；预测长度 96/192/336/720，PEMS 为 12/24/36/48。评价 MSE、MAE。
- **实现**（E-IT-5）：单张 P100 16GB；Adam，学习率取自 {1e-3, 5e-4, 1e-4}；L2 损失；batch 32；固定 10 个 epoch。
  - 块数 L 取自 {2, 3, 4}，维度 D 取自 {256, 512}。
  - 超参数按什么标准选定（验证集还是测试集）正文未写。
- **重复**：附录 Table 5 报告 5 个随机种子下的标准差，只覆盖 ECL、ETTh2、Exchange、Solar、Traffic、Weather，**不含 ETTh1**（E-IT-6）。

## 结果及不确定性

- **Table 1（四个预测长度平均，回看 96）**（E-IT-2）：
  - iTransformer 在 ECL（0.178）、Traffic（0.428）、Solar（0.233）、PEMS（0.119）上 MSE 最低。
  - **ETT（四个子集平均）为 0.383，低于 RLinear 0.380 和 PatchTST 0.381，在自己的表中排第三**。
  - Weather 0.258，与 PatchTST、TimesNet、Crossformer 的 0.259 几乎相同。
- **Table 10（ETT 逐预测长度）**（E-IT-3）：

  | 数据集 | 96 | 192 | 336 | 720 | 平均 |
  | --- | --- | --- | --- | --- | --- |
  | ETTh1（MSE / MAE） | 0.386 / 0.405 | 0.441 / 0.436 | 0.487 / 0.458 | 0.503 / 0.491 | 0.454 / 0.447 |
  | ETTh2 | 0.297 / 0.349 | 0.380 / 0.400 | 0.428 / 0.432 | 0.427 / 0.445 | 0.383 / 0.407 |
  | ETTm1 | 0.334 / 0.368 | 0.377 / 0.391 | 0.426 / 0.420 | 0.491 / 0.459 | 0.407 / 0.410 |
  | ETTm2 | 0.180 / 0.264 | 0.250 / 0.309 | 0.311 / 0.348 | 0.412 / 0.407 | 0.288 / 0.332 |

- **Table 5 的种子标准差**：ETTh2 的 MSE 标准差为 0.001～0.004，均值与 Table 10 完全相同。因此主表很可能是 5 个种子的均值（AI，正文没有明说）。ETTh1 没有标准差。
- **Table 3 消融**（E-IT-4）：只在 ECL、Traffic、Weather、Solar 上做，没有 ETT。
  - 原始 Transformer 的排列（时间维注意力、变量维 FFN）在 Traffic 上最差（0.863 vs 0.428）。
  - 在 Weather 上，两个维度都用 FFN 反而更好（0.248 vs 0.258）。附录 B 也承认变量少的数据集上 FFN-FFN 表现不错。
- **其他图示**：泛化（Fig. 5）、加长回看（Fig. 6，只有 ECL、Traffic）、注意力可解释性（Fig. 7，Solar 个例）都没有给出不确定性。

## 证据与实现核对

| E ID | 层级 | 内容 | 位置 | 支持范围/限制 |
| --- | --- | --- | --- | --- |
| E-IT-1 | DE | 变量 token、变量间注意力、逐 token FFN、逐 token LayerNorm、无位置编码 | PDF p.4 §3.1 Eq.1、Fig.4；p.5 §3.2 Eq.2；p.13 Algorithm 1 | 结构定义；代码另有日历特征 token，正文未提（E-IT-C02） |
| E-IT-2 | DE | 回看 96 的平均结果：ETT 平均排第三；高维数据集最好 | PDF p.6 Table 1 | 作者结果 |
| E-IT-3 | DE | ETT 逐预测长度的 MSE/MAE | PDF p.23 Table 10 | EXP-0004 的复现对照值 |
| E-IT-4 | DE | 组件替换与去除消融（无 ETT） | PDF p.8 Table 3；p.16 Table 6 | ETT 上的组件作用未测 |
| E-IT-5 | DE | 硬件、优化器、学习率网格、batch、epoch、L/D 网格 | PDF p.14 A.2 | 选择标准未写 |
| E-IT-6 | DE | 5 个种子的标准差（不含 ETTh1） | PDF p.14 Table 5 | 公开代码种子写死为 2023（E-IT-C06） |
| E-IT-7 | SC | 框架提升各 Transformer 变体、泛化到新变量、受益于长回看、注意力可解释 | PDF p.6–9 Table 2、Fig.5–7 | 部分数据集、个例可视化 |
| E-IT-8 | DE | 按时间顺序的 TimesNet 划分；数据规模 | PDF p.13 A.1；p.14 Table 4 | Table 4 的规模口径与代码的点数不同（E-IT-C03） |

## 限制与替代解释

- **作者没有专门的局限性一节**，只有结论与伦理声明（PDF p.9–10）。
- **低维数据上优势有限**（E-IT-2、E-IT-4）：ETT 只有 7 个变量，变量间注意力可做的事很少。iTransformer 在 ETT 上不如两个更简单的方法，消融也没有覆盖 ETT。[AI] 它在 ETT 上的表现可能主要来自逐变量 MLP 嵌入和实例归一化，而不是变量注意力。
- **超参数选择口径不明**（E-IT-5）：学习率、L、D 是网格选定的，但没说用验证集还是测试集（UN）。官方脚本中 ETTh2/ETTm1/ETTm2 的 D=128，不在正文写的 {256, 512} 里（E-IT-C08）。
- **种子**：主表是否为多种子均值没有明说；公开代码固定种子 2023，复现多种子需要包装脚本。
- **机制解释**：可解释性来自个例可视化与 CKA 相关，不是因果证据。

## 与当前项目的关系

- **定位**：时序方向的主 baseline（DEC-0012）。
  - 选择理由：近三年、有源码、MIT；被后续工作广泛比较。Semantic Scholar 2026-10-23 计 2676 次引用（[citations-20261023.json](citations-20261023.json)）。
  - 在 ETTh1 上比 2025 年方法差一点：0.454 vs 约 0.44，均为报告值。
- **复现**：见 EXP-0004（3 个种子；判据为报告值落在均值 ± 2 倍标准差内）。ETTh1 没有作者的标准差，只能用本项目的种子波动判断。
- **可借鉴的原则**：逐变量嵌入 + 实例归一化 + 线性投影这一骨架很简单，便于做组件级诊断。
- **必须重新验证**：变量注意力在低维数据上的作用；日历特征 token 的贡献。

## 交接 Deep Read

- **候选 P**：
  - P-IT-1：低维数据（ETT）上变量注意力是否有用（E-IT-2、E-IT-4）；
  - P-IT-2：代码把日历特征当作额外 token 输入，正文未描述，贡献未知（E-IT-C02）。
- **候选 F**：
  - F-IT-1：iTransformer 在自己的 Table 1 中 ETT 平均排第三，高维数据集最好；
  - F-IT-2：变量少的数据集上 FFN-FFN 不差于甚至优于注意力（Weather，E-IT-4）。
- 机制候选见「方法与贡献」，适用条件见「限制与替代解释」。在 EXP-0004 完成前没有自有实验结果。

## ANALYSIS_PACKAGE

```yaml
package_type: ANALYSIS_PACKAGE
package_version: research-pipeline-contract v1.0
package_id: AP-iTransformer-20261023
analysis_id: AP-iTransformer-20261023
created_at: '2026-10-23'
source_ids:
- PAPER-0007
- https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124
evidence_status: partial
evidence_locations:
- 'E-IT-1: PDF p.4 §3.1 Eq.1, Fig.4; p.5 §3.2 Eq.2; p.13 Algorithm 1'
- 'E-IT-2: PDF p.6 Table 1'
- 'E-IT-3: PDF p.23 Table 10'
- 'E-IT-4: PDF p.8 Table 3; p.16 Table 6'
- 'E-IT-5: PDF p.14 A.2'
- 'E-IT-6: PDF p.14 Table 5'
- 'E-IT-7: PDF p.6-9 Table 2, Fig.5-7'
- 'E-IT-8: PDF p.13 A.1; p.14 Table 4'
missing_evidence:
- 附录 D～F 的全部图表未逐项阅读
- 超参数选择标准未写明
- 本项目复现（EXP-0004）尚未完成
next_skill: deep-read Mode C
source_identity_and_version: 见「来源与阅读范围」
research_question: 把每个变量的整段序列作为 token、在变量维做注意力，能否在不改组件的前提下改进 Transformer 预测器
study_type_and_context: 见「研究设计」
design_units_samples_or_materials: 见「研究设计」
intervention_input_or_procedure: 见「方法与贡献」
measurements_and_analysis: 见「研究设计」与「结果及不确定性」
results_with_uncertainty: 见「结果及不确定性」
claim_evidence_implementation_map: 见「证据与实现核对」
limitations_biases_and_ethics: 见「限制与替代解释」
reproducibility_and_transferability: 见「与当前项目的关系」与源码核验补充
candidate_P: [P-IT-1 低维数据上变量注意力的作用, P-IT-2 日历特征 token 的贡献]
candidate_F: [F-IT-1 ETT 平均在自己的表中排第三, F-IT-2 变量少时 FFN-FFN 不差]
evidence_index: [E-IT-1, E-IT-2, E-IT-3, E-IT-4, E-IT-5, E-IT-6, E-IT-7, E-IT-8]
unresolved_questions:
- 主表是否为多种子均值
- 超参数按验证集还是测试集选定
- 日历特征 token 去掉后 ETT 上的变化
```
