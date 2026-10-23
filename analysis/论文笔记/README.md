# 论文笔记

每篇论文使用稳定 paper_id，按 [ANALYSIS_PACKAGE](../templates/ANALYSIS_PACKAGE.md) 创建笔记，例如 PAPER-0004_<短标题>.md。

记录核心任务、方法架构、关键贡献、实验结果、与自身实验的关系、论文现象、优缺点、可借鉴内容与核心总结；同时核验设计、材料、测量、原文位置与结论边界。

| paper_id | 标题 / 年份 | 原文来源 | 解析全文 | 分析包 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| paper_id | 标题 / 年份 | 原文来源 | 原文笔记 | 源码核验补充 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| PAPER-0001 | DUET / KDD2025 | arXiv 2412.10859v3；代码 decisionintelligence/DUET@dcc6e678 | [AP-DUET](PAPER-0001_DUET.md) | [DUET 源码核验](PAPER-0001_DUET_源码核验.md) | 原文+源码；ETTh1 已在 CPU 复现（EXP-0003） |
| PAPER-0002 | TQNet / ICML2025 | arXiv 2505.12917v2；代码 ACAT-SCUT/TQNet@15e19cb2 | [AP-TQNet](PAPER-0002_TQNet.md) | [TQNet 源码核验](PAPER-0002_TQNet_源码核验.md) | 原文+源码；ETTh1 已在 CPU 复现（EXP-0003） |
| PAPER-0003 | WPMixer / AAAI2025 | arXiv 2412.17176v1；代码 Secure-and-Intelligent-Systems-Lab/WPMixer@74104c9d | [AP-WPMixer](PAPER-0003_WPMixer.md) | [WPMixer 源码核验](PAPER-0003_WPMixer_源码核验.md) | 原文+源码；ETTh1 已在 CPU 运行，drop_last 影响已量化（EXP-0003） |
| PAPER-0004 | FastV / ECCV2024 | arXiv 2403.06764v3；代码 pkunlp-icler/FastV@d1659729 | [AP-FastV](PAPER-0004_FastV.md) | [FastV 源码核验](PAPER-0004_FastV_源码核验.md) | 原文+源码；CPU 随机权重核验（EXP-0001），GPU 复现待执行 |
| PAPER-0005 | SparseVLM / ICML2025 | arXiv 2410.04417v4；代码 Gumpest/SparseVLMs v1.5@734d9003、main@a9e71427 | [AP-SparseVLM](PAPER-0005_SparseVLM.md) | [SparseVLM 源码核验](PAPER-0005_SparseVLM_源码核验.md) | 原文+源码（核验补充含 SparseVLM+）；GPU 复现待执行 |
| PAPER-0006 | VisionZip / CVPR2025 | 正式论文+补充；代码 dvlab-research/VisionZip@8f86b55c | [AP-VisionZip](PAPER-0006_VisionZip.md) | [VisionZip 源码核验](PAPER-0006_VisionZip_源码核验.md) | 原文+源码；GPU 复现待执行 |

两类笔记的分工：

- **原文笔记**（`PAPER-xxxx_<名称>.md`）：用户在另一会话中读 PDF 原文写成，证据 ID 形如 E-FV-1。
- **源码核验补充**（`PAPER-xxxx_<名称>_源码核验.md`）：本项目读官方源码、做 CPU 探针和复现得到，证据 ID 带 C 前缀（如 E-FV-C01）。两份内容冲突时，以能追溯到原文页码或源码行号的一方为准，并在两份笔记中都注明。

来源、稳定链接和校验和见 [sources-20261023.json](sources-20261023.json)，综合入口见[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)和[视觉 Mode C](../视觉Token压缩_ModeC.md)。原文笔记用 pdftotext/PyMuPDF 提取核对，非 MinerU；PDF 全文未提交。论文结果不进入自有实验台账。新增笔记后同步本表与 [index](../index.md)。
