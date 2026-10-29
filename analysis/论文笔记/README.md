# 论文笔记

每篇论文使用稳定 paper_id，按 [ANALYSIS_PACKAGE](../templates/ANALYSIS_PACKAGE.md) 创建笔记，例如 PAPER-0004_<短标题>.md。

记录核心任务、方法架构、关键贡献、实验结果、与自身实验的关系、论文现象、优缺点、可借鉴内容与核心总结；同时核验设计、材料、测量、原文位置与结论边界。

| paper_id | 标题 / 年份 | 原文来源 | 原文笔记 | 源码核验补充 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| PAPER-0001 | DUET / KDD2025 | arXiv 2412.10859v3；代码 decisionintelligence/DUET@dcc6e678 | [AP-DUET](PAPER-0001_DUET.md) | [DUET 源码核验](PAPER-0001_DUET_源码核验.md) | 原文+源码；ETTh1 已在 CPU 复现（EXP-0003） |
| PAPER-0002 | TQNet / ICML2025 | arXiv 2505.12917v2；代码 ACAT-SCUT/TQNet@15e19cb2 | [AP-TQNet](PAPER-0002_TQNet.md) | [TQNet 源码核验](PAPER-0002_TQNet_源码核验.md) | 原文+源码；ETTh1 已在 CPU 复现（EXP-0003） |
| PAPER-0003 | WPMixer / AAAI2025 | arXiv 2412.17176v1；代码 Secure-and-Intelligent-Systems-Lab/WPMixer@74104c9d | [AP-WPMixer](PAPER-0003_WPMixer.md) | [WPMixer 源码核验](PAPER-0003_WPMixer_源码核验.md) | 原文+源码；ETTh1 已在 CPU 运行，drop_last 影响已量化（EXP-0003） |
| PAPER-0004 | FastV / ECCV2024 | arXiv 2403.06764v3；代码 pkunlp-icler/FastV@d1659729 | [AP-FastV](PAPER-0004_FastV.md) | [FastV 源码核验](PAPER-0004_FastV_源码核验.md) | 原文+源码；CPU 随机权重核验（EXP-0001）；GPU 复现已完成（EXP-0002；原写“GPU 复现待执行”，2026-10-25 跑偏检查补注） |
| PAPER-0005 | SparseVLM / ICML2025 | arXiv 2410.04417v4；代码 Gumpest/SparseVLMs v1.5@734d9003、main@a9e71427 | [AP-SparseVLM](PAPER-0005_SparseVLM.md) | [SparseVLM 源码核验](PAPER-0005_SparseVLM_源码核验.md) | 原文+源码（核验补充含 SparseVLM+）；GPU 复现已完成（EXP-0002；原写“GPU 复现待执行”，2026-10-25 跑偏检查补注） |
| PAPER-0006 | VisionZip / CVPR2025 | 正式论文+补充；代码 dvlab-research/VisionZip@8f86b55c | [AP-VisionZip](PAPER-0006_VisionZip.md) | [VisionZip 源码核验](PAPER-0006_VisionZip_源码核验.md) | 原文+源码；GPU 复现已完成（EXP-0002；原写“GPU 复现待执行”，2026-10-25 跑偏检查补注） |
| PAPER-0007 | iTransformer / ICLR2024 | arXiv 2310.06625v4；代码 thuml/iTransformer@c2426e68 | [AP-iTransformer](PAPER-0007_iTransformer.md) | [iTransformer 源码核验](PAPER-0007_iTransformer_源码核验.md) | 原文+源码（实验室会话按 analyze-paper 写成）；GPU 复现见 EXP-0004（2026-10-29 完整阅读补注：EXP-0004 已完成：16 个组合中 6 个落在 ±2 SD 内，记为偏差、继续使用，DEC-0018；时序方向自 DEC-0020 暂停） |
| PAPER-0008 | CLASP / arXiv 2026-04 | arXiv 2604.12767v1（papers/CLASP-arxiv-2604.12767.pdf，sha256 534da994…） | [CLASP](PAPER-0008_CLASP.md) | 无（代码未下载、未核验，需许可） | 原文（2026-10-25，按 PDF 逐页阅读，另一代理逐页核验；未用 MinerU。2026-10-26 补注：MinerU flash 档全文已解析，见 [literature/processed](../../literature/processed/README.md)，笔记未据此改动） |
| PAPER-0009 | VIP-Router / arXiv 2026-09 | arXiv 2609.10346v1（papers/VIP-Router-arxiv-2609.10346.pdf，sha256 0d598900…） | [VIP-Router](PAPER-0009_VIP-Router.md) | 无（论文没有给代码链接） | 原文（同上）（2026-10-26 补注：MinerU 全文已解析，见 literature/processed） |
| —（领域 B） | Li & Roth / COLING 2002；Moldovan 等 / ACL 2000 | ACL Anthology C02-1150、P00-1071（papers/，sha256 6133941c…、5a35e1c1…） | [领域 B 原文核对](domainB_原文核对_LiRoth_Moldovan.md) | — | 原文核对 IDEA-0003 的引用（2026-10-25）；Lally 等 2012 只有付费版，未读（2026-10-26 补注：MinerU 全文已解析，见 literature/processed） |

两类笔记的分工：

- **原文笔记**（`PAPER-xxxx_<名称>.md`）：PAPER-0001～0006 由用户在另一会话中读 PDF 原文写成；PAPER-0007 由实验室电脑上的会话按 analyze-paper 读 PDF 写成。（2026-10-29 完整阅读补注：PAPER-0008、PAPER-0009 与领域 B 原文核对（2026-10-25）各由一个工作流子代理读本地 PDF 写成，再由另一个子代理逐页核验并改正；三份都没有源码核验补充，原因见表中各行。）证据 ID 形如 E-FV-1。
- **源码核验补充**（`PAPER-xxxx_<名称>_源码核验.md`）：本项目读官方源码、做 CPU 探针和复现得到，证据 ID 带 C 前缀（如 E-FV-C01）。两份内容冲突时，以能追溯到原文页码或源码行号的一方为准，并在两份笔记中都注明。

来源、稳定链接和校验和见 [sources-20261023.json](sources-20261023.json)（2026-10-29 完整阅读补注：该文件只登记 PAPER-0001～0007，其 scope 字段写 six，实际有 7 条（PAPER-0007 于同日由实验室会话补入，该条 read_scope 已注明）；2026-10-25 经用户许可下载的四篇 PDF（PAPER-0008、PAPER-0009 与领域 B 的两篇）的下载地址、字节数、页数和完整 SHA-256 登记在本文件末尾“2026-10-25 下载的四篇 PDF 的登记”一节）；引用量见 [citations-20261023.json](citations-20261023.json)（Semantic Scholar，2026-10-23 查询）；综合入口见[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)和[视觉 Mode C](../视觉Token压缩_ModeC.md)（2026-10-29 完整阅读补注：该视觉 Mode C 文件自 2026-10-24 起由 [MODE_C-0002](../MODE_C-0002_视觉token剪枝.md) 取代，只作历史版本；时序方向（PAPER-0001～0003、PAPER-0007）的矩阵是 [MODE_C-0003](../MODE_C-0003_时序预测.md)）。原文笔记用 pdftotext/PyMuPDF 提取核对，非 MinerU；PDF 全文未提交。论文结果不进入自有实验台账。新增笔记后同步本表与 [index](../index.md)。

## 2026-10-25 下载的四篇 PDF 的登记（2026-10-29 完整阅读补注）

2026-10-25 经用户许可下载（“要下载读原文”，见[跑偏检查_2026-10-25](../跑偏检查_2026-10-25.md)），获取日期 2026-10-25。下载脚本 `D:\6CJNM007\tmp\papers_1025\download_papers.sh` 按下表字节数核对，并检查文件头为 %PDF-；原始目录 `D:\6CJNM007\data\downloads\papers_1025\`（含 SHA256SUMS）；PDF 复制到 `papers/`，不跟踪。[literature/processed/README](../../literature/processed/README.md) 末段说“完整 sha256 见论文笔记 README”，指本节。

| 对象 | 下载地址 | 字节数 | 页数 | SHA-256（完整） |
| --- | --- | --- | --- | --- |
| PAPER-0008 CLASP，arXiv:2604.12767v1；`papers/CLASP-arxiv-2604.12767.pdf` | https://arxiv.org/pdf/2604.12767 | 3900009 | 37 | 534da9941a5867087434d4ee86c3de8be2c52b87059ba69a17b49d3333141fe7 |
| PAPER-0009 VIP-Router，arXiv:2609.10346v1；`papers/VIP-Router-arxiv-2609.10346.pdf` | https://arxiv.org/pdf/2609.10346 | 2101912 | 26 | 0d59890089fd7fefafd87697db5329319f04916379bbe5ae7dcdf6e87b916d4d |
| Li & Roth，COLING 2002，ACL Anthology C02-1150；`papers/LiRoth-coling2002-C02-1150.pdf` | https://aclanthology.org/C02-1150.pdf | 106463 | 7 | 6133941caa29d0b6e918188b825811e80ddabe29d86ba6fe805dcc32e9dec3cb |
| Moldovan 等，ACL 2000，P00-1071；`papers/Moldovan-acl2000-P00-1071.pdf` | https://aclanthology.org/P00-1071.pdf | 286082 | 8 | 5a35e1c190a9b14466f0dbdad0cf81565b8b44c82096c7f474f8766727e3296d |

两篇 arXiv 的下载地址不带版本号，版本以 PDF 首页为准（v1）。
