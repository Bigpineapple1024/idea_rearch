# 实验代码

| 目录 | 方向 | 实验分支 | 状态 |
| --- | --- | --- | --- |
| [token_pruning/](token_pruning/README.md) | 视觉 token 剪枝（主 baseline FastV） | code/token-pruning | CPU 随机权重核验完成（EXP-0001），GPU 复现待执行 |
| [env/](env/windows_env.sh) | 实验室电脑（原生 Windows）的环境脚本 | 所有分支 | 2026-10-23 在实验室电脑上运行通过；六个环境已装好 CUDA 版 torch（“再装”一列待 source/ 克隆后进行） |
| [ts_forecasting/](ts_forecasting/README.md) | 时序预测（主 baseline iTransformer） | code/ts-forecasting | 对照 DUET/TQNet/WPMixer 已在 CPU 上运行 ETTh1（EXP-0003）；iTransformer 待复现 |

外部仓库（LLaVA、FastV、SparseVLM、VisionZip、iTransformer、DUET、TQNet、WPMixer）不放进本仓库，按 [source/README](../source/README.md) 登记的 commit 克隆到 `source/` 下（只读）；本目录只放本项目自己的代码。

## 环境（实验室电脑：原生 Windows，DEC-0014）

**所有文件只放在 `D:\6CJNM007`**（DEC-0013）。用户选择方案 B：原生 Windows，不用 WSL。Claude Code 在 Windows 上通过 Git Bash 执行命令。

```text
D:\6CJNM007\
├── idea_rearch\      本仓库；外部仓库克隆到 idea_rearch\source\（不提交）
├── miniforge3\       conda（Miniforge），各环境在 miniforge3\envs\
├── cache\            huggingface\、torch\、pip\、conda_pkgs\、matplotlib\
├── data\             LLaVA eval、COCO val2014、MME、ETT 与 TFB 数据
├── tmp\              临时文件
└── backup\<EXP>\<RUN>\  运行产物备份
```

开始前先确认 D 盘可用空间：LLaVA-1.5-7B 与 CLIP 权重约 15GB，COCO val2014 约 6GB，每个 conda 环境数 GB。

### 一次性安装

1. **conda 用 Miniforge**（github.com/conda-forge/miniforge 的 `Miniforge3-Windows-x86_64.exe`），不用 Miniconda：Anaconda 默认频道对 200 人以上的机构有付费条款，conda-forge 没有这一限制。在 cmd 中静默安装到该文件夹，不改 PATH、不注册为系统 Python：

   ```bat
   start /wait "" Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=D:\6CJNM007\miniforge3
   ```

   不要运行 `conda init`，它会修改用户目录（C 盘）里的 shell 配置；改用下面的环境脚本激活。
2. **克隆时关闭换行符转换**，否则 .sh 脚本会被转成 CRLF，在 Git Bash 里报 `$'\r': command not found`。本仓库已用 `.gitattributes` 固定 `*.sh` 为 LF，外部仓库没有，所以一律这样克隆：

   ```bash
   git clone -c core.autocrlf=false https://github.com/Bigpineapple1024/idea_rearch /d/6CJNM007/idea_rearch
   ```

   本仓库若已按默认设置克隆过，不必重新克隆：拉取后执行 `rm code/token_pruning/run_pope_mme.sh && git checkout -- code/token_pruning/run_pope_mme.sh`，让这个脚本按 `.gitattributes` 重新以 LF 检出。

### 每条命令先加载环境脚本

```bash
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp && python ...
```

[env/windows_env.sh](env/windows_env.sh) 把 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`CONDA_PKGS_DIRS`、`MPLCONFIGDIR`、`TMP/TEMP/TMPDIR` 指向 `D:\6CJNM007`，设置 `PYTHONUTF8=1`（否则 Python 按系统代码页读文件），然后激活参数指定的 conda 环境。Claude Code 不在两条命令之间保留 shell 状态，所以每条命令都要加载。

传给 Python 的路径一律写成 `D:/6CJNM007/...`：Git Bash 只对命令参数里的 `/d/...` 做自动转换，环境变量和配置文件里的路径不能指望它，而 Windows 上的 Python 不认识 `/d/...`。

### 各环境：先从 PyTorch 的 CUDA 源装 torch

Windows 上 PyPI 的 torch 只有 CPU 版。每个环境先从 PyTorch 的 CUDA 源装与官方要求**同版本号**的 torch，之后 `pip install -r requirements.txt` 或 `pip install -e` 会认为已满足，不再替换。

| 环境 | 用途 | Python | 先装（CUDA 源） | 再装 |
| --- | --- | --- | --- | --- |
| `tp` | LLaVA、FastV、VisionZip | 3.10 | torch==2.1.2 torchvision==0.16.2（cu121） | `pip install -e source/LLaVA` |
| `sparsevlm` | SparseVLM | 3.10 | torch==2.1.2 torchvision==0.16.2（cu121） | `pip install -e source/SparseVLMs` 和 transformers==4.37.0；不装 flash_attn |
| `itrans` | iTransformer | 3.10 | torch==2.0.0（cu118） | `pip install -r source/iTransformer/requirements.txt` |
| `duet` | DUET | 3.8 | torch==2.4.1（cu121） | `pip install -r source/DUET/requirements.txt` |
| `tqnet` | TQNet | 3.8 | torch==2.4.1（cu121） | `pip install -r source/TQNet/requirements.txt` |
| `wpmixer` | WPMixer | 3.10 | torch==2.0.1（cu118） | `pip install -r source/WPMixer/requirements.txt` |

以 `tp` 为例：

```bash
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh && conda create -n tp python=3.10 -y
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp && cd /d/6CJNM007/idea_rearch \
  && pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 \
  && pip install -e source/LLaVA \
  && python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 应输出 2.1.2+cu121 12.1 True
```

- cu121 需要 `nvidia-smi` 右上角的 CUDA Version ≥ 12.1；驱动较旧时一律改用 cu118（上表各 torch 版本都有 cu118 构建）。
- 每装完一批依赖都再查一次 torch 版本，确认没有被换成 CPU 版或别的版本；若被换掉，重装 CUDA 版并记录是哪个包引起的。
- 若官方版本与本机驱动不兼容，记录实际使用的版本和原因。
- **numpy**：从 CUDA 源装 torchvision 0.16.2 时会顺带装上 numpy 2.x，而 torch 2.1.2 按 NumPy 1.x 编译，`torch.from_numpy` 会报 “Numpy is not available”。`tp`、`sparsevlm` 装完 torch 后再执行 `pip install numpy==1.26.4`。

#### 实验室电脑上的安装结果（2026-10-23）

Miniforge 26.7.2-1 装在 `D:\6CJNM007\miniforge3`。它的 `.condarc` 加了 `register_envs: false` 和 `number_channel_notices: 0`，否则 conda 会往 `C:\Users\<用户>\.conda\environments.txt` 和 AppData 写文件。

| 环境 | Python | torch | CUDA | `torch.cuda.is_available()` | 备注 |
| --- | --- | --- | --- | --- | --- |
| `tp` | 3.10.22 | 2.1.2+cu121（torchvision 0.16.2+cu121） | 12.1 | True | numpy 改为 1.26.4 |
| `sparsevlm` | 3.10.22 | 2.1.2+cu121（torchvision 0.16.2+cu121） | 12.1 | True | numpy 改为 1.26.4 |
| `itrans` | 3.10.22 | 2.0.0+cu118 | 11.8 | True | numpy 由 requirements 安装 |
| `wpmixer` | 3.10.22 | 2.0.1+cu118 | 11.8 | True | numpy 由 requirements 安装 |
| `duet` | 3.8.20 | 2.4.1+cu121 | 12.1 | True | numpy 由 requirements 安装 |
| `tqnet` | 3.8.20 | 2.4.1+cu121 | 12.1 | True | numpy 由 requirements 安装 |

每个环境都在 RTX 3090 上做过一次矩阵乘法。“再装”一列需要先把外部仓库克隆到 `source/`，尚未进行。

### 原生 Windows 上的已知差异（来自代码核查，尚未在 Windows 上运行）

- **DataLoader 子进程**：Windows 用 spawn 启动 worker，子进程会重新导入主模块。
  - LLaVA 的 `model_vqa_loader` 在 worker 中读取只在 `__main__` 里定义的 `args`，worker 数大于 0 时会报 NameError。本项目的 `run_eval.py` 在 Windows 上已把 worker 设为 0；SparseVLM 自带的评测脚本没有这个处理，需要用包装脚本运行（候选做法：在 `sparsevlm` 环境里用 `run_eval.py --method vanilla`，待验证）。
  - TQNet 的 run.py 没有 `__main__` 保护，worker 数大于 0 时子进程会重新执行训练。时序方向的命令统一加 `--num_workers 0`（不影响结果，数据顺序由主进程决定）。
- **flash-attn** 没有官方 Windows 版本，不装。LLaVA 推理评测不需要它。SparseVLM（main 分支 a9e71427）只在 `llava/train/` 的训练补丁里导入 flash_attn；评测时 `use_flash_attn` 默认为 False，走 transformers 默认的 sdpa，而它的解码层本来就断言只支持 sdpa。v1.5 分支是否相同，克隆后先核对。
- **DUET**：requirements 里的 ray 在导入 TFB 时就会被加载，需确认 Windows 上能装上；评测用默认的 sequential 后端，它在本进程内运行，`--timeout` 参数会被忽略。
- **HF 缓存**：未开启 Windows 开发者模式时，Hugging Face 缓存用复制代替符号链接，占用空间会多一些。

### 开始前先检查

`nvidia-smi`（显卡、显存、驱动、CUDA 版本）；每个环境里运行上面的 torch 检查；然后跑各方向目录里的 CPU 自检脚本。

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

- `code/output/` 不进 git；每次运行结束后复制到 `D:\6CJNM007\backup\<EXP>\<RUN>\`（Git Bash 中为 `/d/6CJNM007/backup/<EXP>/<RUN>/`）。
- 实验记录中写明备份路径和 `SHA256SUMS` 的校验和。
- 所有运行（包括失败、中断）登记到 [实验结果总表](../analysis/experiment-results.md)。

## 规则

- 每次运行绑定 IDEA ID（复现与探索性组合写 none）、实验 ID、代码 commit、配置、数据版本与划分、种子和环境。
- 修改外部方法时记录来源、许可证、上游 commit 与变更点；外部仓库只读，改动放在本目录。
- 事实记录与 squeeze 报告放在 [analysis/实验报告/](../analysis/实验报告/README.md)。
