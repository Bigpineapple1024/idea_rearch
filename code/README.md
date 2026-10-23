# 实验代码

| 目录 | 方向 | 实验分支 | 状态 |
| --- | --- | --- | --- |
| [token_pruning/](token_pruning/README.md) | 视觉 token 剪枝（主 baseline FastV） | code/token-pruning | CPU 随机权重核验完成（EXP-0001），GPU 复现待执行 |
| [ts_forecasting/](ts_forecasting/README.md) | 时序预测（主 baseline DUET） | code/ts-forecasting | ETTh1 已在 CPU 上真实复现（EXP-0003） |

外部仓库（LLaVA、FastV、SparseVLM、VisionZip、DUET、TQNet、WPMixer）不放进本仓库，按 [source/README](../source/README.md) 登记的 commit 克隆到 `source/` 下（只读）；本目录只放本项目自己的代码。

## 环境（实验室电脑）

- **系统**：WSL2（Ubuntu），不用原生 Windows。原因是 LLaVA 的 flash-attn、DUET 依赖的 ray 在原生 Windows 上容易出问题。仓库放在 WSL 的 Linux 文件系统里（如 `~/idea_rearch`），不放在 `/mnt/c` 或 `/mnt/d` 下，否则 IO 很慢。
- **环境隔离**：每个外部仓库一个 conda 环境，版本按官方推荐：
  - LLaVA / FastV / VisionZip 共用 `tp`：Python 3.10，torch 2.1.2，transformers 4.37.2；
  - SparseVLM 单独 `sparsevlm`：transformers 4.37.0，flash_attn 2.3.3；
  - DUET 单独 `duet`：Python 3.8，torch 2.4.1；
  - TQNet 单独 `tqnet`：Python 3.8；
  - WPMixer 单独 `wpmixer`：Python 3.10，torch 2.0.1。

  若官方版本与本机 CUDA 驱动不兼容，记录实际使用的版本和原因。
- **开始前先检查**：`nvidia-smi`（显卡、显存、驱动、CUDA 版本）；每个环境里运行 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`；然后跑各方向目录里的 CPU 自检脚本。

## 每次运行的留存（详见研究流程“运行产物、编号与备份”）

```text
code/output/<EXP-xxxx>/<EXP-xxxx-Rnn>/
├── cmd.txt          完整命令行
├── config.*         配置快照
├── git.txt          本仓库 commit + git diff；外部仓库 commit
├── seed.txt         随机种子
├── env.txt          conda env export 或 pip freeze
├── gpu.txt          nvidia-smi 输出
├── stdout.log       完整日志（含 stderr）
├── metrics.json     指标
└── SHA256SUMS       以上文件的 sha256
```

- `code/output/` 不进 git；每次运行结束后复制到 D 盘备份：`/mnt/d/idea_rearch_backup/<EXP>/<RUN>/`。
- 实验记录中写明备份路径和 `SHA256SUMS` 的校验和。
- 所有运行（包括失败、中断）登记到 [实验结果总表](../analysis/experiment-results.md)。

## 规则

- 每次运行绑定 IDEA ID（复现与探索性组合写 none）、实验 ID、代码 commit、配置、数据版本与划分、种子和环境。
- 修改外部方法时记录来源、许可证、上游 commit 与变更点；外部仓库只读，改动放在本目录。
- 事实记录与 squeeze 报告放在 [analysis/实验报告/](../analysis/实验报告/README.md)。
