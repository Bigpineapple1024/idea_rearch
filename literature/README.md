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
- **本机只有 flash 档（2026-10-24 实测）**：basic/standard/advanced 需要本地 parse-server（`parse_server.local.mode`，默认 disabled）和另外下载的版面、表格、公式模型，没有安装；不加 `--tier flash` 时报“No basic, standard, or advanced engine available”。现有全文用 flash 档解析（只有文本），见 [processed/README](processed/README.md)。
- **远端健康探测已关闭**：服务器运行时会定期向 `parse_server.remote.url`（默认 `https://mineru.net/api`）发 `GET /v1/tiers` 探测（`doclib/background/parse_server_health.py`），不含文档内容；不加 `--remote` 时解析任务的隐私级别是 local，不会改走远端（`doclib/services/parse_svc.py` 的 `_resolve_api_target`）。2026-10-24 已执行 `mineru config set parse_server.remote.url http://127.0.0.1:9/api`，探测改指本机无效地址，不再联系 mineru.net。重装或换 `MINERU_HOME` 时与遥测一起重做。
- **用法**（每条命令先加载环境脚本并设置 `MINERU_HOME`）（2026-10-29 完整阅读补注：下面代码块里的 `mineru parse` 缺 `--tier flash`，本机会以“No basic, standard, or advanced engine available”退出（见上面“本机只有 flash 档”一条），而且直接写进 literature/processed/，与上面“重解析时保留版本，不覆盖已经引用的证据”冲突。实际用过的做法见 [processed/README](processed/README.md) 的“命令”一条：先 `mineru server start`，再 `mineru parse papers/<PDF> --tier flash --pages all --wait 3600 -o D:/6CJNM007/tmp/mineru_flash/<论文 ID>/<名称>.md`，核对后原样复制到 literature/processed/<论文 ID>/，最后 `mineru server stop`；解析前后各查一次 `mineru telemetry status`。已有且被引用的解析文件不覆盖，重解析另存版本并在 processed/README 登记 sha256。2026-10-25 下载的四篇 PDF 已按这个做法解析，见 processed/README 的“2026-10-26 补充解析”）：

  ```bash
  source /d/6CJNM007/idea_rearch/code/env/windows_env.sh mineru && export MINERU_HOME=D:/6CJNM007/cache/mineru \
    && mineru server start && mineru parse papers/<PDF> --pages all -o literature/processed/<paper_id>/<名称>.md
  ```
