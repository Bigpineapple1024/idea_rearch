# token_pruning：视觉 token 剪枝 baseline 复现与代码级核验

方向 T1（见 [毕业选题候选](../../analysis/毕业选题候选.md)）。三篇论文的分析见 analysis/论文笔记/PAPER-0004～0006，代码级核验记录见 [EXP-0001](../../analysis/实验报告/EXP-0001_record.md)，GPU 复现见 [EXP-0002](../../analysis/实验报告/EXP-0002_record.md) 与 [SQZ-EXP-0002](../../analysis/实验报告/EXP-0002_squeeze.md)。

**状态（2026-10-25）**：视觉方向的全部 GPU 运行（EXP-0002～EXP-0021，含失败）都由 exp0002.py 执行；EXP-0022 在 GQA testdev-balanced 上检验按题目句式路由。CPU 自检脚本仍可用随机小模型运行。原状态行（2026-10-24）：EXP-0002 已在实验室 RTX 3090 上用真实 LLaVA-1.5-7B 权重跑完 POPE、MME（49 次运行，含失败）。

**校验和**：EXP-0021 及以前的运行目录中，SHA256SUMS 是 Windows 文本模式写的 CRLF 行尾，Git Bash 中要用 `tr -d '\r' < SHA256SUMS | sha256sum -c -` 核对（直接 `sha256sum -c` 会因文件名末尾的 \r 报“找不到文件”；2026-10-25 的跑偏检查全部核对通过）。EXP-0022 起改为 LF，并在每次运行后把 runs.tsv 复制到备份目录（此前的 runs.tsv 已于同日补备份）。

## 文件

| 文件 | 作用 | 已验证范围 |
| --- | --- | --- |
| fastv_llava.py | 用于官方 LLaVA 的 FastV（KV cache，K 与保留数可调）；给定 `seed` 时改为同数量的均匀随机选择。诊断选项：`schedule` 多段剪枝（EXP-0007）、`score_prior` 分数除以位置曲线（EXP-0008）、`stratify` 把排序分到 n×n 个图像块里轮流取（EXP-0010）、`log_text_rows` 记录打分层上图像之后各文本 token 的注意力行（EXP-0011）；`score_kind="object_cos"` 改用图像 token 的输入 embedding 与物体名 embedding 的余弦相似度打分（只适用于 POPE 式题目），`exclude_padding` 不保留完全落在填充区的 patch（EXP-0011 D3）；`score_kind="mix"` 把名额分成两份，`mix_question` 个按物体名余弦、其余按 FastV 分数，第二轮先填空块；`score_kind="contrast_max"` / `"contrast_mean"` 用物体名余弦减去同一 token 与其他类名余弦的最大值 / 均值（`contrast_ids` 给出词表，EXP-0013）；都不设置时计算与原来相同 | 与 FastV 官方 lmms-eval 实现在随机小模型上逐步输出一致（相对误差 ≤3e-7），且不重复写入 KV cache；EXP-0002 |
| run_eval.py | 在 LLaVA 的 model_vqa_loader 评测里切换 vanilla / FastV / VisionZip / 随机 / SparseVLM；Windows 上的 DataLoader worker；`--layer-token-log` 记录每层输入的 token 数；`--text-rows-log` 保存图像之后各文本 token 的注意力行（.npz），`--answer-logit-log` 记录第一个回答 token 为 Yes/No 的对数概率（EXP-0011）；`--layer-rows-log`（只用于 vanilla）在每次 prefill 记录几层的 head 平均行、第 0～1 层的逐 head 行和 hidden state 余弦相似度（EXP-0011 D2b；这次 prefill 改用 eager 注意力）；`--fastv-score object_cos`、`--fastv-exclude-padding`（EXP-0011 D3）；`--fastv-score mix / contrast_max / contrast_mean`、`--fastv-mix-question`（对比打分的词表是 80 个 COCO 类名，按 POPE 题目的写法取 token），以及 `--name-cos-log`（每张图记录图像 token 与 80 个 COCO 类名的输入 embedding 余弦，EXP-0013） | EXP-0002；EXP-0011、EXP-0013 的选项在 CPU 上自检（check_text_rows_log.py、check_layer_rows_log.py、check_object_cos.py、check_mix_names.py） |
| exp0002.py | 全部视觉实验的运行器（名字沿用 EXP-0002）：每次运行写 cmd、config、git、seed、env、gpu、stdout、答案、metrics、status 和 SHA256SUMS，复制到备份并核对校验和，追加到 runs.tsv（EXP-0022 起同时备份 runs.tsv）；拒绝复用已有的 run ID。配置与 benchmark（含 `gqa_testdev`，EXP-0022）的列表在文件开头 | EXP-0002～EXP-0021 的全部运行 |
| sparsevlm_sdpa.py | Windows 上把 SparseVLM 的 FlashAttention-2 调用换成 PyTorch SDPA（运行时替换，不改仓库文件） | 自检：与显式 softmax 注意力的差 ≤2.1e-07；EXP-0002-R36～R41 |
| score_pope_mme.py | 用官方 eval_pope.py 与 MME 的 convert_answer_to_mme.py、calculation.py 打分，写 metrics.json；VQAv2 子集（vqav2_test、vqav2_dev、vqav2_test2）用 LLaVA 的 m4c 评估器，逐题分数写 vqa_scores.jsonl；GQA testdev-balanced（`gqa_testdev`，EXP-0022）按 LLaVA 的 convert_gqa_for_eval.py 与官方 eval.py 的口径（去掉句末句点、小写、与标准答案完全匹配），逐题分数写 gqa_scores.jsonl | 给 LLaVA-1.5-13B 官方预测打分，与 MODEL_ZOO 一致 |
| bootstrap_ci.py | 一次运行的指标或两次运行之差的 bootstrap 95% 区间（POPE 按子集重采样题目，MME 按子任务重采样图像） | EXP-0002 |
| analyze_exp0002.py | 从运行目录生成 EXP-0002 的全部分析表（code/output/EXP-0002/analysis/） | EXP-0002 |
| squeeze_exp0002.py | SQZ-EXP-0002 的探索性检查：答案格式、FastV 与随机的逐题对照、按物体的召回率 | EXP-0002 |
| analyze_object_boxes.py、analyze_object_boxes_extra.py | 用 COCO val2014 物体框分析已有 POPE 运行（不跑模型；被漏掉的 POPE 正样本的大小、位置、填充区，C10；extra 为看过第一版结果后补的检查）：①按物体大小、竖直位置统计漏答率；② pope_img20 上保留的 patch 落在填充区与被问物体上的比例；③ COCO 中没有同名实例的正样本。图像按 LLaVA 的 `expand2square` 映射到 24×24 网格。在 Windows 控制台用 `python -I -X utf8` 运行 | 第 ⑧ 步的分析：code/output/EXP-0002/deepdive3（1A4B-0003） |
| analyze_repope.py | 用 RePOPE（YanNeu/RePOPE@bcf22e56）的修订标注重算全部已完成的 POPE 运行：三个 split 的 F1 均值、配对差的 bootstrap 区间；先用原始标注复现每次运行登记的主指标作自检 | code/output/EXP-0002/deepdive4（DEC-0017） |
| prepare_eval_data.sh | 下载并校验 POPE 标注、COCO val2014、MME 图像和 LLaVA eval.zip | 实验室电脑已运行 |
| run_pope_mme.sh | 早期的批量评测脚本 | 只检查过 DRY_RUN 输出；EXP-0002 改用 exp0002.py |
| smoke_test.py | 用随机小 LLaVA 走一遍官方 generate()，打印每层视觉 token 数 | CPU 已运行，可复现 |
| check_fastv_equivalence.py | fastv_llava 与 FastV 官方实现的等价性检查 | CPU 已运行 |
| check_text_rows_log.py | EXP-0011 记录选项的 CPU 自检：在随机小 LLaVA 上确认记录不改变生成结果与保留集合，行、列、token id 与打分行一致，写出与读回一致 | CPU 已运行，13 项通过（2026-10-24） |
| check_layer_rows_log.py | `--layer-rows-log` 的 CPU 自检：在随机小 LLaVA 上确认题目句与物体名的 token 区间正确，记录的各项与直接重算一致，不改变生成结果，写出与读回一致 | 见 EXP-0011 记录 |
| prompt_spans.py | 从 LLaVA prompt 中图像之后的 token 找出 POPE 式题目句和物体名的 token 区间（与 analyze_text_rows.py 的规则相同） | 随 check_layer_rows_log.py 自检 |
| check_object_cos.py | D3 选项的 CPU 自检：打分等于输入 embedding 的余弦，保留集合等于离线规则（排除填充区后的分块轮流取），填充区掩码与 analyze_object_boxes 相同，参数检查 | CPU 已运行，12 项通过（2026-10-24）；同时复跑了 check_text_rows_log、check_layer_rows_log、check_fastv_equivalence（相对差 ≤ 3.1e-7） |
| analyze_d3.py | EXP-0011 D3 的分析：POPE F1（原始标注与 RePOPE）、配对 bootstrap 差、保留集合在被问物体上的命中率、剂量关系、按大小五分位的新漏答率；先核对每个运行的重算 F1 与登记值一致 | code/output/EXP-0011/d3_pope |
| analyze_layer_rows.py | EXP-0011 D2b 的离线统计（第 2 版）：参与判定的算子（第 0 层注意力、逐 head、归一化后平均、输入与前两层输出的余弦）与更深层的描述项，在分块 + 排除填充区下的小物体命中率、P1、P2、物体 AUC；先用 R02 自检 | code/output/EXP-0011/d2b_coco_dev_v2（第 1 版的输出在 d2b_coco_dev，AUC 有缺陷） |
| analyze_text_rows.py | EXP-0011 D2 的离线统计：第 1 层三种打分（最后一个 token、题目句、物体名）的同图不同题 Jaccard（P1）、物体对准比值（P2）、24 种模拟选择的小物体命中率（P3）和剂量关系；先用模拟复现运行的保留集合作自检 | code/output/EXP-0011/d2_coco_dev（开发集）、d2_pope |
| make_coco_dev.py | EXP-0011 的开发集：COCO val2014 中 POPE 没用到的 300 张图，按 POPE 的方式出 1800 题（种子 0），在它上面做选择，POPE 只作测试集 | 已生成（data/llava_eval/pope/llava_coco_dev*.jsonl，校验和在 llava_coco_dev.SHA256SUMS） |
| make_vqav2_sets.py | EXP-0018：VQAv2 验证集在本地 COCO val2014 图像上的两个子集。test 为 POPE、coco_dev 都没用的图中按种子 0 抽 1000 张的全部 5246 题；dev 为 coco_dev 300 张图上的 1485 题。`test2`（EXP-0020）：同一图像池去掉 test 的图后按种子 1 再抽 1000 张，5415 题 | 已生成（data/llava_eval/vqav2_val/，test、dev 的校验和在 SHA256SUMS，test2 的在各运行的 config.json：c086f656…、4eb4e5c2…）；划分经多代理复核 |
| make_coco_dev3.py | EXP-0013 的开发集：coco_dev 的 1800 题不变，在同样 300 张图上按 POPE 论文的做法补 popular、adversarial 负样本各 900 题 | 已生成（llava_coco_dev3*.jsonl，校验和在 llava_coco_dev3.SHA256SUMS） |
| check_mix_names.py | EXP-0013 选项的 CPU 自检：`--fastv-score mix` 的保留集合等于离线两轮规则（混合 30 = object_cos，混合 0 = 强对照 + 排除填充区）；对比打分等于重算值、排除与被问物体同拼写的类名；`--name-cos-log` 的数值与去重；coco_dev3 全部 prompt 的物体名 id（真实 tokenizer）；参数检查 | CPU 已运行，38 项通过（2026-10-24）；同时复跑了其余 check（FastV 等价相对差 ≤ 3.13e-7）与 smoke_test |
| analyze_kept_tokens.py | EXP-0006：FastV 与随机选择保留了图像的哪些位置 | code/output/EXP-0006 |
| analyze_missed_objects.py | SQZ-EXP-0002 补充：各剪枝方法是否漏掉同一批 POPE 物体（F10） | code/output/EXP-0002 |
| analyze_score_prior.py | EXP-0008：FastV 打分在空白图与真实图上的位置曲线，写出校正用的位置先验 | code/output/EXP-0008/analysis |
| make_pope_subset.py、make_blank_questions.py | pope_img20 诊断子集（20 张图的全部 POPE 题）与同一批题配一张灰色空白图（EXP-0008） | 已生成 |
| check_layer_profile.py | 批量运行前检查 `_layers` 诊断运行的逐层视觉 token 数是否符合预期，不符合就停止 | EXP-0007 用过 |
| analyze_fp.py | EXP-0013：物体名余弦变体多出的误报从哪来，哪种同预算修正在 coco_dev3 上最好 | code/output/EXP-0013 |
| check_query_idf.py | EXP-0014 的 CPU 自检：`--fastv-score query_mean / query_idf / attn_idf` 的打分与保留集合等于离线重算，IDF 表的性质（每题都有的 token 权重为 0），参数检查 | CPU 已运行，17 项通过 |
| analyze_query.py | EXP-0014：A / A′ / A′+B / A+B 在 POPE、MME 上的比较 | code/output/EXP-0014/analysis |
| make_visibility.py | EXP-0015：只用 coco_dev 的图估计每个词表 id 的可视性（峰值的 95% 分位的百分位秩） | code/output/EXP-0015/visibility |
| check_query_vis.py | EXP-0015 的 CPU 自检：`--fastv-score query_vis` 的保留集合等于离线两轮规则，可视性 1 / 0 时分别退化为 query_idf / 强对照，`--vocab-visibility-log` 与 make_visibility.py | CPU 已运行，14 项通过；EXP-0020 的改动后复跑通过 |
| analyze_vis.py | EXP-0015：A′+B+V 与 A、A′+B 在 POPE、MME 上的比较 | code/output/EXP-0015/analysis |
| analyze_0016.py、analyze_0017.py | EXP-0016（固定切分对照、128 档）与 EXP-0017（192 档）的分析 | code/output/EXP-0016/analysis、EXP-0017/analysis |
| squeeze_0017.py | EXP-0017 的深挖（2026-10-25 跑偏检查时补做，探索性）：64 / 128 / 192 三档的 POPE 按子集的 F1、召回、yes 比例、误报，题目查询的名额，MME 感知子任务的差 | code/output/EXP-0017/squeeze |
| analyze_0018.py | EXP-0018：VQAv2 test 上的 VQA 准确率与按图像重抽的配对差（按答案类型） | code/output/EXP-0018/analysis（a909c868…） |
| squeeze_0018.py、squeeze_0018_boxes.py | SQZ-EXP-0018（事后探索）：按查询名额、题型、是否点名 COCO 物体分解差异，按题目首词路由的事后估计；查询选的 token 是否落在被点名物体的框内 | code/output/EXP-0018/squeeze、squeeze_boxes |
| check_route.py | EXP-0020 的 CPU 自检：`--fastv-route yesno_howmany` 下，路由的题与 query_vis、不路由的题与强对照的保留集合相同（两者交替时也一样），记录正确；真实 tokenizer 下从 prompt ids 得到的决定与题目文字上的规则逐题相同，POPE 与 MME 感知子任务全部被路由 | CPU 已运行，17 项通过（2026-10-25） |
| analyze_0019.py | EXP-0019：A+B 与无先验强对照 A₀ 的 POPE 比较（按图像重抽），最终方法、vanilla、A、A₀ 的 margin AUC 与误报 | code/output/EXP-0019/analysis（a17545fa…） |
| analyze_0020.py | EXP-0020：路由与 A、A′+B+V 在 test2 上的比较及实现核对；运行前已在 EXP-0018 的输出上空跑 | code/output/EXP-0020/analysis |
| squeeze_0020.py、squeeze_0020_hybrid.py | SQZ-EXP-0020（事后探索）：路由增益的分解、按开头词与题型的差（含 test1→test2 的复现）、与 VisionZip-64 的差距、逐题 oracle、yes/no 的答案偏向；追加的 do 开头子集与拼接组合（由已有运行拼出）。GPU 运行期间从 D:/6CJNM007/tmp 运行，提交时只把 CRLF 换成 LF | code/output/EXP-0020/squeeze、squeeze_hybrid |
| analyze_0021.py | EXP-0021：192 档路由与 A，128 档 VisionZip、SparseVLM 与 EXP-0020 的 128 档运行在 test2 上的比较（只作描述）；运行前用 EXP-0020 的输出空跑 | code/output/EXP-0021/analysis |
| make_main_table.py | 论文主表草稿：方法 × 预算 × benchmark（POPE 原始标注与 RePOPE、MME-P、VQAv2 test2），每格回链运行 ID，只读已封存的输出 | code/output/PAPER-20261025/main_table |
| make_gqa_set.py | EXP-0022：从官方 GQA 压缩包写出 testdev-balanced 的逐题标注（答案、结构类型），取出 LLaVA 题目文件用到的 398 张图并记录 sha256；可读本地 images.zip，也可用 HTTP 分段请求只读同一官方文件的目录与这 398 个成员（CRC 校验）；核对题号、图像名与题目文字和 LLaVA 的文件一致 | 已运行（data/llava_eval/gqa/、data/gqa/） |
| analyze_0022.py | EXP-0022：GQA 上路由与 A、A′+B+V 的比较（判据 Q1～Q3）、按答案类型与结构类型、二选一题、oracle、yes 比例与参照；运行前用合成的替身运行空跑 | code/output/EXP-0022/analysis |
| sealed_copies/ | 只封存在 code/output 的分析文件夹里、此前从未提交的一次性脚本与 shell 包装（2026-10-25 跑偏检查时收集，去掉 CR 后原样提交；MANIFEST.tsv 列出来源文件夹与封存时的 sha256）。其中两份是后来改过的脚本的第一版：EXP-0002/analysis_v1/analyze_exp0002.py、EXP-0011/d2b_coco_dev/analyze_layer_rows.py | 与封存副本逐字节相同（除 CR） |
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
