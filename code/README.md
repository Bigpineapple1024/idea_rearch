# 实验代码

| 目录 | 方向 | 实验分支 | 状态 |
| --- | --- | --- | --- |
| [token_pruning/](token_pruning/README.md) | 视觉 token 剪枝（主 baseline FastV） | code/token-pruning | CPU 随机权重核验完成（EXP-0001），GPU 复现待执行 |
| [ts_forecasting/](ts_forecasting/README.md) | 时序预测（主 baseline iTransformer） | code/ts-forecasting | 对照 DUET/TQNet/WPMixer 已在 CPU 上运行 ETTh1（EXP-0003）；iTransformer 待复现 |

外部仓库（LLaVA、FastV、SparseVLM、VisionZip、iTransformer、DUET、TQNet、WPMixer）不放进本仓库，按 [source/README](../source/README.md) 登记的 commit 克隆到 `source/` 下（只读）；本目录只放本项目自己的代码。

## 环境（实验室电脑）

**所有文件只放在 `D:\6CJNM007`**（用户在实验室电脑上的专用文件夹，DEC-0013）：仓库、外部仓库、conda 或 WSL 发行版、模型与包缓存、数据、运行产物和备份都不写到其他位置。开始前先确认该文件夹所在盘的可用空间：LLaVA-1.5-7B 与 CLIP 权重约 15GB，COCO val2014 约 6GB，conda 环境每个数 GB。

### 方案 A（优先）：WSL2，发行版导入到 D:\6CJNM007\wsl

LLaVA 生态、SparseVLM 指定的 flash_attn 2.3.3、DUET 依赖的 ray 在 Linux 上最省事。WSL 默认把发行版装在 C 盘，所以用 `wsl --import` 把整个 Linux 文件系统放进该文件夹：

```powershell
wsl --status        # 确认本机已启用 WSL2；未启用且没有管理员权限时改用方案 B
# 从 cloud-images.ubuntu.com/wsl/ 下载 Ubuntu 22.04 的 WSL rootfs（.tar.gz）到 D:\6CJNM007\wsl\
wsl --import idea D:\6CJNM007\wsl\idea D:\6CJNM007\wsl\<下载的 rootfs 文件> --version 2
wsl -d idea
```

进入发行版后，仓库、conda（`~/miniconda3`）、缓存和数据都放在 Linux 主目录里；它们实际存放在 `D:\6CJNM007\wsl\idea\ext4.vhdx` 中，IO 速度正常，也满足只用该文件夹的要求。备份写到 `/mnt/d/6CJNM007/backup/`，在 Windows 里可以直接看到。

### 方案 B：原生 Windows

```text
D:\6CJNM007\
├── idea_rearch\      本仓库；外部仓库克隆到 idea_rearch\source\（不提交）
├── miniconda3\       conda 安装位置（安装时选 Just Me 并指定此路径），环境在 miniconda3\envs\
├── cache\            huggingface\、torch\、pip\、conda_pkgs\
├── data\             LLaVA eval、COCO val2014、MME、ETT 与 TFB 数据
├── tmp\              临时文件
└── backup\<EXP>\<RUN>\  运行产物备份
```

在用户环境变量（或每次会话开头）设置：

```powershell
setx HF_HOME D:\6CJNM007\cache\huggingface
setx TORCH_HOME D:\6CJNM007\cache\torch
setx PIP_CACHE_DIR D:\6CJNM007\cache\pip
setx CONDA_PKGS_DIRS D:\6CJNM007\cache\conda_pkgs
# 装包时编译和解压的临时文件：在当前终端里临时设置即可
$env:TMP="D:\6CJNM007\tmp"; $env:TEMP="D:\6CJNM007\tmp"
```

原生 Windows 的已知风险（遇到时记录实际做法，作为环境偏差写进实验记录）：

- flash-attn 没有官方 Windows wheel。LLaVA 推理评测不需要它；SparseVLM 指定了 flash_attn 2.3.3，先试不装它能否用 eager 注意力运行。
- `.sh` 脚本（如 `run_pope_mme.sh`、各仓库的 scripts/）需要用 Git Bash 运行，或改写为 PowerShell。
- DUET 依赖 ray；TFB 的串行后端是否在 Windows 上可用需要先试。

### 两种方案共同的环境隔离

每个外部仓库一个 conda 环境，版本按官方推荐：

- LLaVA / FastV / VisionZip 共用 `tp`：Python 3.10，torch 2.1.2，transformers 4.37.2；
- SparseVLM 单独 `sparsevlm`：transformers 4.37.0，flash_attn 2.3.3；
- iTransformer 单独 `itrans`：按其 requirements.txt；
- DUET 单独 `duet`：Python 3.8，torch 2.4.1；
- TQNet 单独 `tqnet`：Python 3.8；
- WPMixer 单独 `wpmixer`：Python 3.10，torch 2.0.1。

若官方版本与本机 CUDA 驱动不兼容，记录实际使用的版本和原因。

**开始前先检查**：`nvidia-smi`（显卡、显存、驱动、CUDA 版本）；每个环境里运行 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`；然后跑各方向目录里的 CPU 自检脚本。

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

- `code/output/` 不进 git；每次运行结束后复制到 `D:\6CJNM007\backup\<EXP>\<RUN>\`（WSL 下为 `/mnt/d/6CJNM007/backup/<EXP>/<RUN>/`）。
- 实验记录中写明备份路径和 `SHA256SUMS` 的校验和。
- 所有运行（包括失败、中断）登记到 [实验结果总表](../analysis/experiment-results.md)。

## 规则

- 每次运行绑定 IDEA ID（复现与探索性组合写 none）、实验 ID、代码 commit、配置、数据版本与划分、种子和环境。
- 修改外部方法时记录来源、许可证、上游 commit 与变更点；外部仓库只读，改动放在本目录。
- 事实记录与 squeeze 报告放在 [analysis/实验报告/](../analysis/实验报告/README.md)。
