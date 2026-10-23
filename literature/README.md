# 论文解析材料

- PDF 原文放 papers/，不跟踪。
- MinerU 解析全文放 processed/<论文ID>/，保留图片、表格和公式关联。
- 登记标题、作者、年份、DOI/URL、PDF 校验和、解析工具版本、参数与获取日期。
- 解析全文只读。OCR、公式和表格疑点记录到笔记并回查 PDF；重解析时保留版本，不覆盖已经引用的证据。
- 自维护笔记放在 [analysis/论文笔记/](../analysis/论文笔记/README.md)。
- 涉及受限材料时确认其存储与分享范围。

## MinerU（实验室电脑，2026-10-23）

- **版本与环境**：MinerU 4.0.12，conda 环境 `mineru`（Python 3.11）。基本安装用 ONNX Runtime（含 DirectML），不依赖 torch。
- **数据位置**：`MINERU_HOME=D:/6CJNM007/cache/mineru`；不设置时默认写到 C 盘的 `~/.mineru`（DEC-0013 不允许）。
- **遥测已关闭**：4.0.12 是预发布版，只要状态不是 disabled 就会发送使用统计：`doclib/telemetry/service.py` 第 100～102 行，“只在 enabled 时发送”的判断被注释掉了。安装后先 `mineru server start`、立即 `mineru telemetry disable`，`mineru telemetry status` 显示 `state: disabled`、`last_flush_at: never`。重装或换 `MINERU_HOME` 时必须重做。
- **只在本地解析**：不使用 `--remote`（会把文档上传到 mineru.net）。`mineru parse` 默认只解析前 10 页，全文需 `--pages all`。
- **用法**（每条命令先加载环境脚本并设置 `MINERU_HOME`）：

  ```bash
  source /d/6CJNM007/idea_rearch/code/env/windows_env.sh mineru && export MINERU_HOME=D:/6CJNM007/cache/mineru \
    && mineru server start && mineru parse papers/<PDF> --pages all -o literature/processed/<paper_id>/<名称>.md
  ```
