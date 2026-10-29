# token_pruning：视觉 token 剪枝 baseline 复现与代码级核验

方向 T1（见 [毕业选题候选](../../analysis/毕业选题候选.md)）。三篇论文的分析见 analysis/论文笔记/PAPER-0004～0006，代码级核验记录见 [EXP-0001](../../analysis/实验报告/EXP-0001_record.md)。

**状态（2026-10-23）**：所有脚本都在 CPU 上用随机权重的小模型跑通，**还没有在真实 LLaVA-1.5-7B 权重和 benchmark 上运行**，因此没有任何准确率或速度结果。真实复现需要在 24GB GPU 上按下文执行。（2026-10-29 完整阅读补注：本分支的 code/token_pruning/ 是 2026-10-23 的旧副本，状态已过时：GPU 复现已在 EXP-0002 完成，视觉方向的实验已到 EXP-0029。视觉方向的代码与状态以 code/token-pruning 分支和 main 的 analysis/index.md 为准。）

## 文件

| 文件 | 作用 | 已验证范围 |
| --- | --- | --- |
| fastv_llava.py | 用于官方 LLaVA 的 FastV（KV cache，K 与保留数可调） | 与 FastV 官方 lmms-eval 实现在随机小模型上逐步输出一致（相对误差 ≤3e-7），且不重复写入 KV cache |
| run_eval.py | 在 LLaVA 的 model_vqa_loader 评测里切换 vanilla / FastV / VisionZip | 参数解析已测试；GPU 评测未运行 |
| run_pope_mme.sh | 三种方法 × 192/128/64 预算 × POPE、MME | DRY_RUN 输出已检查；GPU 评测未运行 |
| smoke_test.py | 用随机小 LLaVA 走一遍官方 generate()，打印每层视觉 token 数 | CPU 已运行，可复现 |
| check_fastv_equivalence.py | fastv_llava 与 FastV 官方实现的等价性检查 | CPU 已运行 |
| probes/*.py | 三个仓库的代码级探针和 FLOPs 计算 | CPU 已运行，输出见 EXP-0001 |

SparseVLM 不经过本目录，它自带一套 LLaVA 分叉，直接在它自己的仓库里运行（见下文）。

## 固定版本

| 组件 | 来源 | commit / 版本 |
| --- | --- | --- |
| LLaVA | github.com/haotian-liu/LLaVA | c121f0432da27facab705978f83c4ada465e46fd |
| VisionZip | github.com/dvlab-research/VisionZip | 8f86b55c6f000eb033e6912538af2dd7dcb30502 |
| FastV | github.com/pkunlp-icler/FastV | d1659729b5bf1be225e99ee15783deeea80f63b1 |
| SparseVLM | github.com/Gumpest/SparseVLMs | v1.5 分支 734d9003879689713a132b83297dfad060a5c9af（ICML 版）；main a9e71427220c4cacb6afb5a96d23cfb43142e0c3（含 SparseVLM+） |
| Python 依赖 | LLaVA pyproject.toml | torch 2.1.2、torchvision 0.16.2、transformers 4.37.2、tokenizers 0.15.1、accelerate 0.21.0、timm 0.6.13 |
| 权重 | Hugging Face | liuhaotian/llava-v1.5-7b、openai/clip-vit-large-patch14-336 |

## GPU 上的复现步骤（实验室电脑：原生 Windows + Git Bash）

环境约定见 [code/README](../README.md)：所有文件在 `D:\6CJNM007`，每条命令先加载 `code/env/windows_env.sh`。下面的 `$E`、`$SRC` 只是简写；Claude Code 不在命令之间保留 shell 变量，实际执行时每条命令都要写出完整路径或在同一条命令里定义它们：

```bash
E="source /d/6CJNM007/idea_rearch/code/env/windows_env.sh"
SRC=/d/6CJNM007/idea_rearch/source

# 1. 环境（Python 3.10，与 LLaVA 官方一致）。Windows 上先从 CUDA 源装 torch，再装 LLaVA
$E && conda create -n tp python=3.10 -y
cd $SRC && git clone -c core.autocrlf=false https://github.com/haotian-liu/LLaVA && git -C LLaVA checkout c121f0432da27facab705978f83c4ada465e46fd
cd $SRC && git clone -c core.autocrlf=false https://github.com/dvlab-research/VisionZip && git -C VisionZip checkout 8f86b55c6f000eb033e6912538af2dd7dcb30502
$E tp && pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 && pip install -e $SRC/LLaVA
$E tp && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.1.2+cu121 True
# 不要 pip install visionzip：PyPI 上 0.1.3 的 wheel 只有元数据，没有代码；用 VISIONZIP_ROOT 指向仓库

# 2. 权重：首次运行自动下载到 HF_HOME（D:\6CJNM007\cache\huggingface），约 15GB；国内可设 HF_ENDPOINT=https://hf-mirror.com

# 3. 数据：按 LLaVA docs/Evaluation.md 下载 eval.zip，解压到 LLaVA/playground/data/eval
#    POPE 需要 COCO val2014 图像；MME 需要 MME_Benchmark_release_version 与 eval_tool

# 4. 先跑 CPU 自检（可选，验证环境）。传给 Python 的路径用 D:/ 形式
$E tp && cd /d/6CJNM007/idea_rearch/code/token_pruning && VISIONZIP_ROOT=D:/6CJNM007/idea_rearch/source/VisionZip python smoke_test.py

# 5. 正式评测：vanilla、VisionZip、FastV × 192/128/64 × POPE、MME
$E tp && LLAVA_ROOT=D:/6CJNM007/idea_rearch/source/LLaVA VISIONZIP_ROOT=D:/6CJNM007/idea_rearch/source/VisionZip \
  bash /d/6CJNM007/idea_rearch/code/token_pruning/run_pope_mme.sh
```

在 Windows 上，`run_eval.py` 会把 LLaVA 评测的 DataLoader worker 设为 0：LLaVA 的 `CustomDataset` 在 worker 中读取只在 `__main__` 里定义的 `args`，Windows 用 spawn 启动的 worker 取不到它。这只影响数据加载速度，不影响结果。

每次运行在答案文件旁写 `<answers>.meta.json`（方法参数、LLaVA 参数、墙钟时间）。所有运行（包括失败和中断）都要登记到 [实验结果总表](../../analysis/experiment-results.md)。

### 复现是否成功的判断

先看 vanilla 能否复现 LLaVA 官方 MODEL_ZOO 的 7B 数字：POPE 85.9、MME 1510.7（同时还有 GQA 62.0、TextVQA 58.2）。偏差超过约 1 个点时，先排查环境、数据、prompt 和解码设置，不要急着开始改方法。VisionZip 和 FastV 的对照值请从各自论文的表格回查后填入实验记录，这里不凭记忆给出。

### SparseVLM（在它自己的仓库中运行）

```bash
cd $SRC && git clone -c core.autocrlf=false https://github.com/Gumpest/SparseVLMs && git -C SparseVLMs checkout v1.5
$E && conda create -n sparsevlm python=3.10 -y
$E sparsevlm && pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 \
  && pip install -e $SRC/SparseVLMs && pip install transformers==4.37.0
# 官方说明还要装 flash_attn==2.3.3；它没有 Windows 版本，评测路径也不需要（见 code/README），不装
# scripts/v1_5/eval/*.sh 里的路径写死为作者集群的 /mnt/bn/...，先改成本地路径
RETAIN_TOKN=64 bash scripts/v1_5/eval/pope.sh   # Linux 上的官方用法；Windows 上见下方说明
```

注意：

- 只支持 batch size 1（batch 2 时第二个样本会被剪掉全部视觉 token，见 EXP-0001）；代码里写死了 `.cuda()`；它的“64/128/192”是 32 层平均的视觉 token 数，不是输入 LLM 的 token 数。
- **Windows**：官方脚本调用 `python -m llava.eval.model_vqa_loader`，DataLoader 默认 4 个 worker，会遇到与 LLaVA 相同的 `args` 问题而报 NameError。不改它的仓库文件，用包装脚本把 worker 设为 0。候选做法是在 `sparsevlm` 环境里运行本目录的 `run_eval.py --method vanilla`（剪枝由 SparseVLM 自己的模型代码按环境变量执行），待验证它的参数与 SparseVLM 的 `model_vqa_loader` 一致。

## 预算的口径

三篇论文的“保留 64 个 token”含义不同，比较时必须写明口径：

- VisionZip：在进入 LLM 之前压缩，所有 32 层都只看到 64 个视觉 token。
- SparseVLM：32 层的平均值。前 2 层仍是 576 个，最后 17 层只有 17～20 个。
- FastV：K 层以后保留 keep 个。本目录按“32 层平均值相等”换算，K=2 时 192/128/64 对应 keep=166/98/30。

三者的理论 FLOPs 基本相同（probes/flops_budget.py），但 prefill 显存峰值、首 token 延迟和各层看到的信息不同。论文中应同时报告 FLOPs 和实测延迟。
