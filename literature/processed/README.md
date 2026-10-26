# MinerU 解析全文（只读）

解析日期：2026-10-24。工具：MinerU 4.0.12（conda 环境 `mineru`，`MINERU_HOME=D:/6CJNM007/cache/mineru`），在实验室电脑本地运行，未使用 `--remote`，遥测状态为 disabled（解析前后各查一次，`last_flush_at: never`）。

- **解析档位：flash（只有文本）**。本机的基本安装只有 flash 档：basic/standard/advanced 需要本地 parse-server 和另外下载的版面、表格、公式模型，没有安装。因此：
  - 表格被压成普通文本，行列关系可能错乱；公式没有结构化；
  - 文中的 `![Image block](doc:…)` 是 MinerU 文档库内部的引用，不是图片文件；
  - 每页开头有 `<!-- page N of M -->` 标记，可按页回查 PDF。
- **命令**：`mineru parse papers/<PDF> --tier flash --pages all --wait 3600 -o D:/6CJNM007/tmp/mineru_flash/<论文 ID>/<名称>.md`，再原样复制到本目录（脚本先 `mineru server start`，结束后 `mineru server stop`）。同日第一次尝试没有加 `--tier flash`，8 个文件都以“No basic, standard, or advanced engine available”退出，没有产生输出。
- **引用规则**：数字、表格和公式以 PDF 为准；本目录只用于检索和定位页码。

| 论文 ID | PDF（papers/，不跟踪） | PDF sha256（前 16 位） | 页数 | 解析文件 | 解析文件 sha256（前 16 位） |
| --- | --- | --- | --- | --- | --- |
| PAPER-0001 | DUET-arxiv-2412.10859.pdf | a26e3deb2ec3821c | 14 | PAPER-0001/DUET-arxiv-2412.10859.md | 7024049cecda1a8e |
| PAPER-0002 | TQNet-arxiv-2505.12917.pdf | 3a406d4a89215be1 | 18 | PAPER-0002/TQNet-arxiv-2505.12917.md | 3801dc943057dab8 |
| PAPER-0003 | WPMixer-arxiv-2412.17176.pdf | f95ac458aef1fbfc | 12 | PAPER-0003/WPMixer-arxiv-2412.17176.md | f0592ab3293e9cb5 |
| PAPER-0004 | FastV-arxiv-2403.06764.pdf | b4532650958db4f5 | 18 | PAPER-0004/FastV-arxiv-2403.06764.md | c3893af99198933d |
| PAPER-0005 | SparseVLM-arxiv-2410.04417.pdf | ccb0ff92d74d782f | 18 | PAPER-0005/SparseVLM-arxiv-2410.04417.md | 44ffc58163f5ef08 |
| PAPER-0006 | VisionZip-cvpr.pdf | d4e22f2136ba1b7e | 11 | PAPER-0006/VisionZip-cvpr.md | 94e7fa7d9739e4aa |
| PAPER-0006 | VisionZip-cvpr-supp.pdf | f47e9bd1b26bdc8b | 17 | PAPER-0006/VisionZip-cvpr-supp.md | 604925887f1064fb |
| PAPER-0007 | iTransformer-arxiv-2310.06625.pdf | 83e889795af52c67 | 25 | PAPER-0007/iTransformer-arxiv-2310.06625.md | 4b222f04464b30a7 |

**2026-10-26 补充解析**：2026-10-25 下载的四篇 PDF 用同样的方法解析（脚本 `D:\6CJNM007\tmp\mineru_parse_flash2.sh`，13:06:56～13:10:54，每个都 exit 0；未使用 `--remote`，远程 parse-server 指向本机的空地址，遥测解析前后都是 disabled、`last_flush_at: never`；页标记数与 PDF 页数相同）。原文笔记（PAPER-0008、PAPER-0009、领域 B 原文核对）写于解析之前，按 PDF 逐页阅读，没有据此改动。

| 论文 ID | PDF（papers/，不跟踪） | PDF sha256（前 16 位） | 页数 | 解析文件 | 解析文件 sha256（前 16 位） |
| --- | --- | --- | --- | --- | --- |
| PAPER-0008 | CLASP-arxiv-2604.12767.pdf | 534da9941a586708 | 37 | PAPER-0008/CLASP-arxiv-2604.12767.md | f06d38c7a515355b |
| PAPER-0009 | VIP-Router-arxiv-2609.10346.pdf | 0d59890089fd7fef | 26 | PAPER-0009/VIP-Router-arxiv-2609.10346.md | a88ce4e42d48e46a |
| —（领域 B） | LiRoth-coling2002-C02-1150.pdf | 6133941caa29d0b6 | 7 | domainB/LiRoth-coling2002-C02-1150.md | 424ec5ddcbae1cd8 |
| —（领域 B） | Moldovan-acl2000-P00-1071.pdf | 5a35e1c190a9b144 | 8 | domainB/Moldovan-acl2000-P00-1071.md | aab0ed66330f0c76 |

PDF 的完整 sha256 与版本见 [论文来源清单](../../analysis/论文笔记/sources-20261023.json)（PAPER-0008、0009 与领域 B 的两篇见 [论文笔记 README](../../analysis/论文笔记/README.md)）。
