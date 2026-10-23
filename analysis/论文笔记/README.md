# 论文笔记

每篇论文使用稳定 paper_id，按 [ANALYSIS_PACKAGE](../templates/ANALYSIS_PACKAGE.md) 创建笔记，例如 PAPER-0001_<短标题>.md。

记录核心任务、方法架构、关键贡献、实验结果、与自身实验的关系、论文现象、优缺点、可借鉴内容与核心总结；同时核验设计、材料、测量、原文位置与结论边界。

| paper_id | 标题 / 年份 | 原文来源 | 解析全文 | 分析包 | 核验状态 |
| --- | --- | --- | --- | --- | --- |
| PAPER-0001 | DUET / KDD2025 | arXiv 2412.10859v3 | 本地 papers/ | [AP-DUET](PAPER-0001_DUET.md) | 原文+静态源码；未复现 |
| PAPER-0002 | TQNet / ICML2025 | arXiv 2505.12917v2 | 本地 papers/ | [AP-TQNet](PAPER-0002_TQNet.md) | 原文+静态源码；未复现 |
| PAPER-0003 | WPMixer / AAAI2025 | arXiv 2412.17176v1 | 本地 papers/ | [AP-WPMixer](PAPER-0003_WPMixer.md) | 原文+静态源码；未复现 |
| PAPER-0004 | FastV / ECCV2024 | arXiv 2403.06764v3 | 本地 papers/ | [AP-FastV](PAPER-0004_FastV.md) | 原文+关键实现；区分mask/inplace |
| PAPER-0005 | SparseVLM / ICML2025 | arXiv 2410.04417v4 | 本地 papers/ | [AP-SparseVLM](PAPER-0005_SparseVLM.md) | 原文+v1.5；非SparseVLM+ |
| PAPER-0006 | VisionZip / CVPR2025 | 正式论文+补充 | 本地 papers/ | [AP-VisionZip](PAPER-0006_VisionZip.md) | 原文+关键实现；未复现 |

来源、稳定链接和校验和见 [sources-20261023.json](sources-20261023.json)，综合入口见[六篇总结](../六篇论文总结与视觉Token压缩趋势.md)和[视觉 Mode C](../视觉Token压缩_ModeC.md)。使用 pdftotext/PyMuPDF 提取核对，非 MinerU；PDF 全文未提交。论文结果不进入自有实验台账。
