# 实验代码

| 目录 | 方向 | 实验分支 | 状态 |
| --- | --- | --- | --- |
| [token_pruning/](token_pruning/README.md) | 视觉 token 剪枝（主 baseline FastV） | code/token-pruning | CPU 随机权重核验完成（EXP-0001），GPU 复现待执行（2026-10-29 完整阅读补注：已过时。GPU 复现已在 EXP-0002 完成（2026-10-23～24，实验室 RTX 3090，49 次运行，含失败）；此后视觉方向的实验到 EXP-0029 都已完成，IDEA-0006 在 EXP-0029 结束（DEC-0054）。方向格的“主 baseline FastV”也不再只指 FastV：IDEA-0006 起在 VisionZip-64 上做（DEC-0044）。当前的视觉代码在 code/token-pruning 分支，本分支 code/ts-forecasting 上的 code/token_pruning/ 是 2026-10-23 的旧副本，其 README 的状态同样已过时。最新状态见 main 分支的 analysis/index.md） |
| [env/](env/windows_env.sh) | 实验室电脑（原生 Windows）的环境脚本 | 所有分支 | 2026-10-23 在实验室电脑上运行通过；六个环境（含“再装”一列）已装好并核对 |
| [ts_forecasting/](ts_forecasting/README.md) | 时序预测（主 baseline iTransformer） | code/ts-forecasting | 对照 DUET/TQNet/WPMixer 已在 CPU 上运行 ETTh1（EXP-0003）；iTransformer 待复现（2026-10-29 完整阅读补注：已过时。iTransformer 已在实验室 GPU 上复现（EXP-0004，记为偏差后继续使用，DEC-0018）；DUET、TQNet、WPMixer 已按统一协议在 GPU 上重跑（EXP-0005）；之后做了 EXP-0009、EXP-0012 两个诊断；时序方向在第 ⑦ 步后暂停（DEC-0020、DEC-0021），代码在 code/ts-forecasting 分支（撰写补注时为 099fead）） |

外部仓库（LLaVA、FastV、SparseVLM、VisionZip、iTransformer、DUET、TQNet、WPMixer）不放进本仓库，按 [source/README](../source/README.md) 登记的 commit 克隆到 `source/` 下（只读）；本目录只放本项目自己的代码。

## 环境（实验室电脑：原生 Windows，DEC-0014）

**所有文件只放在 `D:\6CJNM007`**（DEC-0013），软件、git 配置和 SSH 密钥也一样，不在 C 盘安装软件、不写 C 盘用户目录。用户选择方案 B：原生 Windows，不用 WSL。Claude Code 在 Windows 上通过 Git Bash 执行命令。

```text
D:\6CJNM007\
├── idea_rearch\      本仓库；外部仓库克隆到 idea_rearch\source\（不提交）
├── git\              Git for Windows 便携版（PortableGit），不在 PATH 上
├── .ssh\             GitHub deploy key 与 known_hosts（只有本人和 SYSTEM 可读）
├── miniforge3\       conda（Miniforge），各环境在 miniforge3\envs\
├── cache\            huggingface\、torch\、pip\、conda_pkgs\、matplotlib\
├── data\             LLaVA eval、COCO val2014、MME、ETT 与 TFB 数据
├── tmp\              临时文件
└── backup\<EXP>\<RUN>\  运行产物备份
```

开始前先确认 D 盘可用空间：LLaVA-1.5-7B 与 CLIP 权重约 15GB，COCO val2014 约 6GB，每个 conda 环境数 GB。

### 一次性安装

1. **Git 用便携版**（github.com/git-for-windows/git 的 `PortableGit-<版本>-64-bit.7z.exe`），不用安装版：安装版会写进 C 盘的用户目录、PATH 和右键菜单。解压到该文件夹，再运行一次 `post-install.bat`（只改 git 文件夹内部）：

   ```bat
   PortableGit-<版本>-64-bit.7z.exe -o"D:\6CJNM007\git" -y
   cmd /c D:\6CJNM007\git\post-install.bat
   ```

   Git Bash 是 `D:\6CJNM007\git\bin\bash.exe`，PowerShell 里用 `D:\6CJNM007\git\cmd\git.exe`。在终端里启动 `claude` 前先执行 `$env:CLAUDE_CODE_GIT_BASH_PATH = "D:\6CJNM007\git\bin\bash.exe"`。不要用 `git config --global`，它写 C 盘的 `~/.gitconfig`。
2. **conda 用 Miniforge**（github.com/conda-forge/miniforge 的 `Miniforge3-Windows-x86_64.exe`），不用 Miniconda：Anaconda 默认频道对 200 人以上的机构有付费条款，conda-forge 没有这一限制。在 cmd 中静默安装到该文件夹，不改 PATH、不注册为系统 Python：

   ```bat
   start /wait "" Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=D:\6CJNM007\miniforge3
   ```

   - 不要运行 `conda init`，它会修改用户目录（C 盘）里的 shell 配置；改用下面的环境脚本激活。
   - 在 `D:\6CJNM007\miniforge3\.condarc` 末尾加 `register_envs: false` 和 `number_channel_notices: 0`，否则 conda 会往 `C:\Users\<用户>\.conda\environments.txt` 和 AppData 写文件。
   - 安装程序会在 C 盘建开始菜单快捷方式、在 HKCU 写卸载登记，装完删掉；卸载时直接删除 `miniforge3` 文件夹。
3. **克隆时关闭换行符转换**，否则 .sh 脚本会被转成 CRLF，在 Git Bash 里报 `$'\r': command not found`。本仓库已用 `.gitattributes` 固定 `*.sh` 为 LF，外部仓库没有，所以一律这样克隆：

   ```bash
   git clone -c core.autocrlf=false https://github.com/Bigpineapple1024/idea_rearch /d/6CJNM007/idea_rearch
   ```

   本仓库若已按默认设置克隆过，不必重新克隆：拉取后执行 `rm code/token_pruning/run_pope_mme.sh && git checkout -- code/token_pruning/run_pope_mme.sh`，让这个脚本按 `.gitattributes` 重新以 LF 检出。
4. **推送用 SSH deploy key**（只能访问本仓库）。`D:\6CJNM007` 继承了 D 盘根目录的权限，本机所有登录用户都可读写，所以密钥文件夹要去掉继承权限。密钥不设口令，因为 Claude Code 推送时无法输入口令。在 PowerShell 中：

   ```powershell
   $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
   mkdir D:\6CJNM007\.ssh
   icacls D:\6CJNM007\.ssh /inheritance:r /grant:r "*${sid}:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F"
   D:\6CJNM007\git\usr\bin\ssh-keygen.exe -t ed25519 -N '""' -C idea_rearch-deploy-key -f D:/6CJNM007/.ssh/id_ed25519_idea_rearch
   ```

   - `known_hosts` 用 api.github.com/meta 里的 `ssh_keys`，每行前加 `github.com `，再用 `ssh-keygen -lf` 与 GitHub 公布的指纹核对。
   - 公钥 `id_ed25519_idea_rearch.pub` 加到仓库的 Settings → Deploy keys，勾选 Allow write access。
   - 实验室网络封了 22 端口，所以走 `ssh.github.com:443`。在仓库里设置（只写 `.git/config`）：

   ```bash
   git config user.name Bigpineapple1024
   git config user.email 102575093+Bigpineapple1024@users.noreply.github.com
   git config core.sshCommand "D:/6CJNM007/git/usr/bin/ssh.exe -F /dev/null -i D:/6CJNM007/.ssh/id_ed25519_idea_rearch -o IdentitiesOnly=yes -o UserKnownHostsFile=D:/6CJNM007/.ssh/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes -o HostName=ssh.github.com -o Port=443 -o HostKeyAlias=github.com"
   git remote set-url origin git@github.com:Bigpineapple1024/idea_rearch.git
   ```

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
- **装“再装”一列时用约束文件固定 torch**：LLaVA 的 peft、bitsandbytes 等不固定版本，新版可能要求更高的 torch，pip 就会从 PyPI 换成 CPU 版。每个环境用 `pip install -c <约束文件> ...`，约束文件写已装的 `torch==<版本>+cu<版本>`；`tp`、`sparsevlm` 再加 `torchvision==0.16.2+cu121` 和 `numpy==1.26.4`。
- **peft**：LLaVA 不固定 peft，最新版要用新版 accelerate 的 `clear_device_cache`，与 LLaVA 固定的 accelerate 0.21.0 不兼容，`import peft` 报 ImportError。`tp`、`sparsevlm` 装 `peft==0.10.0`（2024-03，与 LLaVA 固定的 commit 同期）。LLaVA 只在加载 LoRA 权重时用 peft，llava-v1.5-7b 本身不受影响。
- **setuptools**：`wpmixer` 环境没有 `pkg_resources`，WPMixer 内置的 `pytorch_wavelets/dtcwt` 需要它，装 `setuptools==69.5.1`。外部仓库里只有这一处用到 `pkg_resources`。
- **thop**（2026-10-24 补装）：WPMixer 的 `exp/exp_main.py` 在模块顶部导入 `thop`（只用于统计 GFLOPs），但它不在 requirements.txt 里；缺少时 `run_LTF.py` 一启动就报 ModuleNotFoundError（EXP-0005-R13～R24）。用约束文件固定 torch 后装 `thop`（0.1.1.post2209072238），torch 未变，`pip check` 无冲突。（2026-10-29 完整阅读补注：本条和下表 wpmixer 一行的 thop 从 main 上的同名文件补入，此前只在 main 的副本里有。）
- **sparsevlm 的 transformers**：按 SparseVLM 官方要求装 4.37.0 后，`pip check` 会提示 llava 要求 4.37.2，这是预期的。

#### 实验室电脑上的安装结果（2026-10-23）

Git 为 PortableGit 2.56.0.3（`D:\6CJNM007\git`），Miniforge 26.7.2-1 装在 `D:\6CJNM007\miniforge3`（`.condarc` 已按上文修改）。外部仓库按 [source/README](../source/README.md) 的 commit 克隆在 `source/`（不提交）。

| 环境 | Python | torch | CUDA | GPU 可用 | “再装”后的关键版本 |
| --- | --- | --- | --- | --- | --- |
| `tp` | 3.10.22 | 2.1.2+cu121（torchvision 0.16.2+cu121） | 12.1 | True | llava 1.2.2.post1（`source/LLaVA`，可编辑安装）、transformers 4.37.2、tokenizers 0.15.1、accelerate 0.21.0、timm 0.6.13、peft 0.10.0、bitsandbytes 0.45.5、numpy 1.26.4 |
| `sparsevlm` | 3.10.22 | 2.1.2+cu121（torchvision 0.16.2+cu121） | 12.1 | True | llava 1.2.2.post1（`source/SparseVLMs` v1.5，可编辑安装）、transformers 4.37.0、tokenizers 0.15.1、accelerate 0.21.0、peft 0.10.0、bitsandbytes 0.45.5、numpy 1.26.4 |
| `itrans` | 3.10.22 | 2.0.0+cu118 | 11.8 | True | numpy 1.23.5、pandas 1.5.3、scikit-learn 1.2.2、matplotlib 3.7.0、reformer-pytorch 1.4.4 |
| `wpmixer` | 3.10.22 | 2.0.1+cu118 | 11.8 | True | numpy 1.25.2、pandas 1.5.3、scikit-learn 1.3.0、optuna 3.6.1、PyWavelets 1.6.0、setuptools 69.5.1、thop 0.1.1（2026-10-24 补装） |
| `duet` | 3.8.20 | 2.4.1+cu121 | 12.1 | True | numpy 1.24.4、pandas 1.5.3、darts 0.25.0、salesforce-merlion 2.0.2、ray 2.10.0、pytorch-lightning 2.4.0、lightgbm 4.1.0 |
| `tqnet` | 3.8.20 | 2.4.1+cu121 | 12.1 | True | requirements 不固定版本，装到的是 numpy 1.24.4、pandas 2.0.3、scikit-learn 1.3.2、matplotlib 3.7.5 |

- 每个环境都在 RTX 3090 上做过一次矩阵乘法。
- 装完“再装”一列后又核对了三点：torch 没有被替换，关键包都能导入，`pip check` 没有冲突（sparsevlm 的 transformers 提示除外）。
- 各环境的完整 `pip freeze`、安装脚本和日志在 `D:\6CJNM007\backup\env_setup_2026-10-23\`。

### 原生 Windows 上的已知差异（视觉方向已在 EXP-0002 中实际运行；时序方向来自代码核查）

（2026-10-29 完整阅读补注：时序方向此后也在实验室电脑上实际运行了，EXP-0004、EXP-0005 都加 `--num_workers 0`（见两份记录的配置），所以下面时序方向的条目不再只是代码核查。）

- **DataLoader 子进程**：Windows 用 spawn 启动 worker，子进程会重新导入主模块。
  - LLaVA 的 `model_vqa_loader` 在 worker 中读取只在 `__main__` 里定义的 `args`，worker 数大于 0 时会报 NameError。本项目的 `run_eval.py` 换用自己的 DataLoader（数据集、collate_fn 和顺序与 LLaVA 相同），在每个 worker 启动时把 `args` 传进去，默认 4 个 worker（`--num-workers` 可改）。答案与 0 个 worker 时逐条相同（EXP-0002-R05 对 R02、R34 对 R04），每题快约 7%。
  - SparseVLM 不用它自带的评测脚本，而是在 `sparsevlm` 环境里运行 `run_eval.py --method sparsevlm`：它调用 SparseVLM 自己的 `model_vqa_loader`（剪枝在 SparseVLM 的模型代码里按 `retained_tokens` 执行），DataLoader 的处理同上（EXP-0002-R36～R41）。
  - TQNet 的 run.py 没有 `__main__` 保护，worker 数大于 0 时子进程会重新执行训练。时序方向的命令统一加 `--num_workers 0`（不影响结果，数据顺序由主进程决定）。
- **flash-attn** 没有官方 Windows 版本，不装。LLaVA 推理评测不需要它。
  - **SparseVLM 需要它（更正，2026-10-24）**：之前按“`llava/train/` 以外没有 `flash_attn` 导入”判断评测不需要，这是错的。
  - SparseVLM 的 `load_pretrained_model` 默认 `dynamic_sparse=True`，会给 32 层都挂上 `LlamaDynamicvitFlashAttention2`（`llava/model/builder.py` 第 114～121 行），每层注意力最终调用 transformers 的 `_flash_attention_forward`，也就是 `flash_attn_func`。导入发生在 transformers 内部，所以在 SparseVLM 源码里搜不到。
  - 解码层“只支持 sdpa”的断言只针对不用的 `self_attn`。没有 flash-attn 时会报 `NameError: name 'flash_attn_func' is not defined`（EXP-0002-R18～R23）。
  - 处理：`code/token_pruning/sparsevlm_sdpa.py` 在运行时把这一个函数换成 PyTorch 的 SDPA。注意力在数学上相同，只是内核不同，与显式 softmax 注意力的差在 2.1e-07 以内（`python sparsevlm_sdpa.py`）。SparseVLM 自己的 token 打分不受影响，仓库文件不改。`run_eval.py --method sparsevlm` 在 Windows 上自动启用，并在 meta.json 里记录。
- **DUET**：requirements 里的 ray 在导入 TFB 时就会被加载；ray 2.10.0 已在实验室电脑上安装并能导入（2026-10-23）。评测用默认的 sequential 后端，它在本进程内运行，`--timeout` 参数会被忽略。
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
