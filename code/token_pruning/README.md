# token_pruning：视觉 token 剪枝 baseline 复现与代码级核验

方向 T1（见 [毕业选题候选](../../analysis/毕业选题候选.md)）。三篇论文的分析见 analysis/论文笔记/PAPER-0004～0006，代码级核验记录见 [EXP-0001](../../analysis/实验报告/EXP-0001_record.md)，GPU 复现见 [EXP-0002](../../analysis/实验报告/EXP-0002_record.md) 与 [SQZ-EXP-0002](../../analysis/实验报告/EXP-0002_squeeze.md)。

**状态（2026-10-24）**：EXP-0002 已在实验室 RTX 3090 上用真实 LLaVA-1.5-7B 权重跑完 POPE、MME（vanilla、FastV、VisionZip、SparseVLM、同预算随机 × 192/128/64，共 49 次运行，含失败）。CPU 自检脚本仍可用随机小模型运行。

## 文件

| 文件 | 作用 | 已验证范围 |
| --- | --- | --- |
| fastv_llava.py | 用于官方 LLaVA 的 FastV（KV cache，K 与保留数可调）；给定 `seed` 时改为同数量的均匀随机选择 | 与 FastV 官方 lmms-eval 实现在随机小模型上逐步输出一致（相对误差 ≤3e-7），且不重复写入 KV cache；EXP-0002 |
| run_eval.py | 在 LLaVA 的 model_vqa_loader 评测里切换 vanilla / FastV / VisionZip / 随机 / SparseVLM；Windows 上的 DataLoader worker；`--layer-token-log` 记录每层输入的 token 数 | EXP-0002 |
| exp0002.py | EXP-0002 的运行器：每次运行写 cmd、config、git、seed、env、gpu、stdout、答案、metrics、status 和 SHA256SUMS，复制到备份并核对校验和，追加到 runs.tsv；拒绝复用已有的 run ID | EXP-0002 的 49 次运行 |
| sparsevlm_sdpa.py | Windows 上把 SparseVLM 的 FlashAttention-2 调用换成 PyTorch SDPA（运行时替换，不改仓库文件） | 自检：与显式 softmax 注意力的差 ≤2.1e-07；EXP-0002-R36～R41 |
| score_pope_mme.py | 用官方 eval_pope.py 与 MME 的 convert_answer_to_mme.py、calculation.py 打分，写 metrics.json | 给 LLaVA-1.5-13B 官方预测打分，与 MODEL_ZOO 一致 |
| bootstrap_ci.py | 一次运行的指标或两次运行之差的 bootstrap 95% 区间（POPE 按子集重采样题目，MME 按子任务重采样图像） | EXP-0002 |
| analyze_exp0002.py | 从运行目录生成 EXP-0002 的全部分析表（code/output/EXP-0002/analysis/） | EXP-0002 |
| squeeze_exp0002.py | SQZ-EXP-0002 的探索性检查：答案格式、FastV 与随机的逐题对照、按物体的召回率 | EXP-0002 |
| prepare_eval_data.sh | 下载并校验 POPE 标注、COCO val2014、MME 图像和 LLaVA eval.zip | 实验室电脑已运行 |
| run_pope_mme.sh | 早期的批量评测脚本 | 只检查过 DRY_RUN 输出；EXP-0002 改用 exp0002.py |
| smoke_test.py | 用随机小 LLaVA 走一遍官方 generate()，打印每层视觉 token 数 | CPU 已运行，可复现 |
| check_fastv_equivalence.py | fastv_llava 与 FastV 官方实现的等价性检查 | CPU 已运行 |
| probes/*.py | 三个仓库的代码级探针和 FLOPs 计算 | CPU 已运行，输出见 EXP-0001 |

SparseVLM 也通过 `run_eval.py` 运行：在 `sparsevlm` 环境里用 `--method sparsevlm --retained-tokens <预算>`。它调用 SparseVLM 自带的 model_vqa_loader，剪枝由 SparseVLM 的模型代码执行；`run_eval.py` 会检查加载的确实是 SparseVLM 的版本。

## 固定版本

| 组件 | 来源 | commit / 版本 |
| --- | --- | --- |
| LLaVA | github.com/haotian-liu/LLaVA | c121f0432da27facab705978f83c4ada465e46fd |
| VisionZip | github.com/dvlab-research/VisionZip | 8f86b55c6f000eb033e6912538af2dd7dcb30502 |
| FastV | github.com/pkunlp-icler/FastV | d1659729b5bf1be225e99ee15783deeea80f63b1 |
| SparseVLM | github.com/Gumpest/SparseVLMs | v1.5 分支 734d9003879689713a132b83297dfad060a5c9af（ICML 版）；main a9e71427220c4cacb6afb5a96d23cfb43142e0c3（含 SparseVLM+） |
| Python 依赖 | LLaVA pyproject.toml | torch 2.1.2、torchvision 0.16.2、transformers 4.37.2、tokenizers 0.15.1、accelerate 0.21.0、timm 0.6.13 |
| 权重 | Hugging Face | liuhaotian/llava-v1.5-7b@4481d270、openai/clip-vit-large-patch14-336@ce19dc91 |

## GPU 上的复现步骤（实验室电脑：原生 Windows + Git Bash）

环境约定见 [code/README](../README.md)：所有文件在 `D:\6CJNM007`，每条命令先加载 `code/env/windows_env.sh`。环境 `tp`、`sparsevlm` 的安装结果也在那里。下面的 `$E` 只是简写；Claude Code 不在命令之间保留 shell 变量，实际执行时每条命令都要写出完整路径或在同一条命令里定义它：

```bash
E="source /d/6CJNM007/idea_rearch/code/env/windows_env.sh"

# 1. 先跑 CPU 自检（验证环境）。传给 Python 的路径用 D:/ 形式
$E tp && cd /d/6CJNM007/idea_rearch/code/token_pruning && VISIONZIP_ROOT=D:/6CJNM007/idea_rearch/source/VisionZip python smoke_test.py

# 2. 权重与数据：按固定版本下载到 D:\6CJNM007（hf：权重与 MME 图像；files：其余下载；extract：解压 eval.zip 等）
#    校验和写入 D:\6CJNM007\data\downloads\SHA256SUMS
$E tp && bash /d/6CJNM007/idea_rearch/code/token_pruning/prepare_eval_data.sh hf
$E tp && bash /d/6CJNM007/idea_rearch/code/token_pruning/prepare_eval_data.sh files
$E tp && bash /d/6CJNM007/idea_rearch/code/token_pruning/prepare_eval_data.sh extract

# 3. 正式评测：每个参数是 <run ID>:<配置>:<benchmark>，按顺序执行；配置与 benchmark 的列表在 exp0002.py 开头
#    运行器在 tp 环境中启动，按配置自动切换到 tp 或 sparsevlm 环境
$E tp && cd /d/6CJNM007/idea_rearch/code/token_pruning && python exp0002.py EXP-0002-R50:fastv64:pope EXP-0002-R51:random64_s2021:mme

# 4. 分析表：从全部运行目录重新生成 code/output/EXP-0002/analysis/
$E tp && cd /d/6CJNM007/idea_rearch/code/token_pruning && python analyze_exp0002.py
```

权重下载到 `D:\6CJNM007\cache\huggingface`（约 15GB）；评测时 exp0002.py 设 `HF_HUB_OFFLINE=1`，只用已下载的固定版本。

Windows 上的 DataLoader：LLaVA 的 `CustomDataset` 在 worker 中读取只在 `__main__` 里定义的 `args`，Windows 用 spawn 启动的 worker 取不到它。`run_eval.py` 用自己的 DataLoader（数据集、collate_fn 和顺序与 LLaVA 相同），在每个 worker 启动时把 `args` 传进去，默认 4 个 worker。答案与 0 个 worker 时逐条相同（EXP-0002-R05 对 R02、R34 对 R04）。

每次运行在答案文件旁写 `<answers>.meta.json`（方法参数、LLaVA 参数、加载与评测时间、每题秒数、评测期间峰值显存）。所有运行（包括失败和中断）都要登记到 [实验结果总表](../../analysis/experiment-results.md)。

### 复现是否成功的判断

先看 vanilla 能否复现 LLaVA 官方 MODEL_ZOO 的 7B 数字：POPE 85.9、MME 1510.7（同时还有 GQA 62.0、TextVQA 58.2）。偏差超过约 1 个点时，先排查环境、数据、prompt 和解码设置，不要急着开始改方法。VisionZip、SparseVLM 和其中 FastV 行的对照值见 EXP-0002 记录的计划 v1.1。

EXP-0002 的结果：vanilla POPE 85.90（通过）；MME-P 1507.63，比官方低 3.07（只换注意力内核就移动 6.93，记为偏差）；VisionZip、SparseVLM 的 POPE 三档都与各自论文相差 ≤0.2。

### SparseVLM

```bash
cd /d/6CJNM007/idea_rearch/source && git clone -c core.autocrlf=false https://github.com/Gumpest/SparseVLMs && git -C SparseVLMs checkout v1.5
# sparsevlm 环境的安装见 code/README（torch 2.1.2+cu121、pip install -e source/SparseVLMs、transformers 4.37.0，不装 flash_attn）
```

注意：

- 只支持 batch size 1（batch 2 时第二个样本会被剪掉全部视觉 token，见 EXP-0001）；代码里写死了 `.cuda()`；它的“64/128/192”是预算表的名义层平均，不是输入 LLM 的 token 数。
- 每层注意力都调用 FlashAttention-2（`llava/model/builder.py` 第 114～121 行给 32 层挂上 `LlamaDynamicvitFlashAttention2`）。Windows 上没有 flash-attn，`run_eval.py --method sparsevlm` 会自动用 `sparsevlm_sdpa.py` 换成 SDPA，并在 meta.json 里记录。
- 它为每层新建一份注意力模块并复制权重，原模块保留，所以评测显存比其他方法多约 4 GiB（EXP-0002：18.1 GiB 对 14.0～14.5 GiB）。
- 官方 `scripts/v1_5/eval/*.sh` 的路径写死为作者集群路径，本项目不用这些脚本。

## 预算的口径

三篇论文的“保留 64 个 token”含义不同，比较时必须写明口径：

- VisionZip：在进入 LLM 之前压缩，所有 32 层都只看到 64 个视觉 token。
- SparseVLM：名义值是预算表的 32 层平均。实测（EXP-0002-R47～R49，50 道 POPE，每题相同）：前 3 层 576 个，之后 192 档为 309/204/113，128 档为 312/117/39，64 档为 82/32/18；层平均 206.5/145.4/82.25，比名义预算多 7.6%/13.6%/28.5%。原因是剪枝层本身用剪枝前的 token 计算，并在每次剪枝后追加回收 token。
- FastV：K 层以后保留 keep 个。本目录按“32 层平均值相等”换算，K=2 时 192/128/64 对应 keep=166/98/30；实测 64 档为前 2 层 576、之后 30 个（层平均 64.125）。

三者的理论 FLOPs 基本相同（probes/flops_budget.py），但每层看到的信息、显存峰值和延迟不同。论文中应同时报告 FLOPs、实测每层 token 数和实测延迟。
