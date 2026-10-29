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
| PAPER-0010 | DivPrune / arXiv 2025-03 | arXiv 2503.02175v2（papers/DivPrune-arxiv-2503.02175.pdf，sha256 9f93e2fb…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0011 | CoverPruner / arXiv 2026-09 | arXiv 2609.03158v1（papers/CoverPruner-arxiv-2609.03158.pdf，sha256 b2f40481…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0012 | MMTok / arXiv 2025-08 | arXiv 2508.18264v2；ICLR 2026（papers/MMTok-arxiv-2508.18264.pdf，sha256 706b0792…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0013 | CDPruner / arXiv 2025-06 | arXiv 2506.10967v2（papers/CDPruner-arxiv-2506.10967.pdf，sha256 2a46e686…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0014 | VScan / arXiv 2025-05 | arXiv 2505.22654v3；TMLR（01/2026）（papers/VScan-arxiv-2505.22654.pdf，sha256 da9c5824…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0015 | ShaRP / arXiv 2025-12 | arXiv 2512.05385v2（papers/ShaRP-arxiv-2512.05385.pdf，sha256 2a5cd4fc…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0016 | FSR / arXiv 2026-02 | arXiv 2602.05809v2（papers/FSR-arxiv-2602.05809.pdf，sha256 b034fff5…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0017 | VisionTrim / arXiv 2026-01 | arXiv 2601.22674v3；ICLR 2026（papers/VisionTrim-arxiv-2601.22674.pdf，sha256 794614de…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0018 | SCOPE / arXiv 2025-10 | arXiv 2510.24214v1（papers/SCOPE-arxiv-2510.24214.pdf，sha256 50a1f9a3…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0019 | HoloV / arXiv 2025-10 | arXiv 2510.02912v2（papers/HoloV-arxiv-2510.02912.pdf，sha256 3f3353a8…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
| PAPER-0020 | ZOO-Prune / arXiv 2025-09 | arXiv 2509.24837v4（papers/ZOO-Prune-arxiv-2509.24837.pdf，sha256 c77f17b8…） | 未写 | 无（代码未下载） | 只下载与解析（2026-10-29）；尚未阅读 |
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

## 2026-10-29 下载的十一篇 PDF 的登记

2026-10-29 经用户许可下载（“我同意下载所有你需要的论文原文pdf”；只下 PDF，不下代码）。用途：核对 MODE_C-0002 的 P13（留 64 个 token 时覆盖 / 多样性一类在文献中高于 VisionZip）与本项目 DivPrune 的写法（EXP-0032、EXP-0033）；原文笔记尚未写，写时按 analyze-paper 逐页读 PDF。下载脚本 `D:\6CJNM007\tmp\read_repo\download_pdfs.sh`：每篇从 `https://arxiv.org/pdf/<号>` 下到 `D:\6CJNM007\tmp\downloads\<号>\` 的新文件夹，检查文件头为 %PDF 后复制到 `papers/`（不跟踪）；HTTP 都是 200。下载地址不带版本号，版本取 PDF 第 1 页的 arXiv 标记（MinerU 解析的第 1 页）；页数取 MinerU 的页标记。[literature/processed](../../literature/processed/README.md) 有解析全文。

| 对象 | 下载地址 | 字节数 | 页数 | SHA-256（完整） | 获取时间 |
| --- | --- | --- | --- | --- | --- |
| PAPER-0010 DivPrune，arXiv:2503.02175v2（第 1 页标记，2025-04-01）；《DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models》，第一作者 Saeed Ranjbar Alvar；`papers/DivPrune-arxiv-2503.02175.pdf` | https://arxiv.org/pdf/2503.02175 | 12038847 | 17 | 9f93e2fb05a2b677ddda40f6b9da09092dfb783976dbcdbe490b51141f90a15a | 2026-10-29T11:50:49+09:00 |
| PAPER-0011 CoverPruner，arXiv:2609.03158v1（第 1 页标记）；《Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization》，第一作者 Qingchan Zhu；`papers/CoverPruner-arxiv-2609.03158.pdf` | https://arxiv.org/pdf/2609.03158 | 6927504 | 20 | b2f40481bc02b3ae6b97a22651027b09aaee6eae7b32759233c2798fffb724bf | 2026-10-29T11:50:52+09:00 |
| PAPER-0012 MMTok，arXiv:2508.18264v2（第 1 页标记，2026-03-03）；《MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs》，第一作者 Sixun Dong；第 1 页写 ICLR 2026；`papers/MMTok-arxiv-2508.18264.pdf` | https://arxiv.org/pdf/2508.18264 | 11995795 | 25 | 706b0792bfff8edce3e6b69e66fd4924baa840970110e7a4fcea6d347f2a34da | 2026-10-29T11:50:57+09:00 |
| PAPER-0013 CDPruner，arXiv:2506.10967v2（第 1 页标记，2025-07-01）；《Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs》，第一作者 Qizhe Zhang；`papers/CDPruner-arxiv-2506.10967.pdf` | https://arxiv.org/pdf/2506.10967 | 5408917 | 22 | 2a46e68695ee0a81ee6bef16d16628d3be3bb99cc0f39b17853a0795376c1937 | 2026-10-29T11:51:02+09:00 |
| PAPER-0014 VScan，arXiv:2505.22654v3（第 1 页标记，2026-01-30）；《VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models》，第一作者 Ce Zhang；第 1 页写 TMLR（01/2026）；`papers/VScan-arxiv-2505.22654.pdf` | https://arxiv.org/pdf/2505.22654 | 10737611 | 24 | da9c58244a7145c9e648d9dd101f35fae8a7d2846f1f083e10b7682635c42f50 | 2026-10-29T11:51:05+09:00 |
| PAPER-0015 ShaRP，arXiv:2512.05385v2（第 1 页标记，2026-03-14）；《ShaRP: SHAllow-LayeR Pruning for Efficient Video Large Language Models》，第一作者 Yingjie Xia；`papers/ShaRP-arxiv-2512.05385.pdf` | https://arxiv.org/pdf/2512.05385 | 20320247 | 19 | 2a5cd4fc5fe4c3f2d56e353810e14bfc7796a2791d4666b57fd3ac34716847a9 | 2026-10-29T11:51:14+09:00 |
| PAPER-0016 FSR，arXiv:2602.05809v2（第 1 页标记，2026-02-09）；《Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning》，第一作者 Enwei Tong；`papers/FSR-arxiv-2602.05809.pdf` | https://arxiv.org/pdf/2602.05809 | 1983195 | 16 | b034fff51d633918b41e8e53c7792a45a982fbe904251fda750b92367cc037ad | 2026-10-29T11:51:18+09:00 |
| PAPER-0017 VisionTrim，arXiv:2601.22674v3（第 1 页标记，2026-03-28）；《VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration》，第一作者 Hanxun Yu；第 1 页写 ICLR 2026；`papers/VisionTrim-arxiv-2601.22674.pdf` | https://arxiv.org/pdf/2601.22674 | 7588458 | 31 | 794614de3a8c1d290d7ac12d846cd8765dfd49af805753ef5f507fed03304552 | 2026-10-29T11:51:22+09:00 |
| PAPER-0018 SCOPE，arXiv:2510.24214v1（第 1 页标记，2025-10-28）；《SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs》，第一作者 Jinhong Deng；`papers/SCOPE-arxiv-2510.24214.pdf` | https://arxiv.org/pdf/2510.24214 | 2516112 | 26 | 50a1f9a32777dc0f7146ace8a1c49fe58a99e44c97018447a1aff06c81fde14e | 2026-10-29T11:51:25+09:00 |
| PAPER-0019 HoloV，arXiv:2510.02912v2（第 1 页标记，2025-10-10）；《Don’t Just Chase “Highlighted Tokens” in MLLMs: Revisiting Visual Holistic Context Retention》，第一作者 Xin Zou；`papers/HoloV-arxiv-2510.02912.pdf` | https://arxiv.org/pdf/2510.02912 | 3701109 | 34 | 3f3353a89d1e9da20f7b268c737622de780c4d1999b0a6733e0f66f24eb28836 | 2026-10-29T11:51:29+09:00 |
| PAPER-0020 ZOO-Prune，arXiv:2509.24837v4（第 1 页标记，2026-03-20）；《ZOO-Prune: Training-Free Token Pruning via Zeroth-Order Gradient Estimation in Vision-Language Models》，第一作者 Youngeun Kim；`papers/ZOO-Prune-arxiv-2509.24837.pdf` | https://arxiv.org/pdf/2509.24837 | 31981999 | 20 | c77f17b8b06308fd80f1a73042f68fddd6381530f3a1f1813b42da1b9279a93a | 2026-10-29T11:51:38+09:00 |
