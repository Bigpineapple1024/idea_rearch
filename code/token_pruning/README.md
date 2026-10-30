# token_pruning：视觉 token 剪枝 baseline 复现与代码级核验

方向 T1（见 [毕业选题候选](../../analysis/毕业选题候选.md)）。三篇论文的分析见 analysis/论文笔记/PAPER-0004～0006，代码级核验记录见 [EXP-0001](../../analysis/实验报告/EXP-0001_record.md)，GPU 复现见 [EXP-0002](../../analysis/实验报告/EXP-0002_record.md) 与 [SQZ-EXP-0002](../../analysis/实验报告/EXP-0002_squeeze.md)。

**状态（2026-10-30，EXP-0033 之后）**：视觉方向的全部 GPU 运行（EXP-0002～EXP-0033，含失败）都由 exp0002.py 执行（EXP-0023 只在 CPU 上读已封存的输出，不经它）。IDEA-0004（预算内放大）以 EXP-0025 结束（DEC-0041）；IDEA-0005（FastV 上的定位后覆盖）的确认实验 EXP-0027 判读为增益主要来自 K = 1、覆盖不作为方法的贡献（DEC-0045）；IDEA-0006（VisionZip-64 上的定位后覆盖）以 EXP-0029 结束（DEC-0054）；IDEA-0007（一层全看后按视觉塔的 CLS 注意力排序）以 EXP-0031 结束（DEC-0059）；EXP-0032 在本流水线里按论文自写 DivPrune，DivPrune-64 明显强于 VisionZip-64，此后以它为底座（DEC-0061）；EXP-0033 是两种方法组合的探索性运行（第 ④ 步第二轮）。CPU 自检脚本仍可用随机小模型运行。最新状态见 main 分支的 analysis/index.md 与 analysis/experiment-results.md。原状态行（2026-10-25）：视觉方向的全部 GPU 运行（EXP-0002～EXP-0021，含失败）都由 exp0002.py 执行；EXP-0022 在 GQA testdev-balanced 上检验按题目句式路由。更早的状态行（2026-10-24）：EXP-0002 已在实验室 RTX 3090 上用真实 LLaVA-1.5-7B 权重跑完 POPE、MME（49 次运行，含失败）。

**校验和**：EXP-0021 及以前的运行目录中，SHA256SUMS 是 Windows 文本模式写的 CRLF 行尾，Git Bash 中要用 `tr -d '\r' < SHA256SUMS | sha256sum -c -` 核对（直接 `sha256sum -c` 会因文件名末尾的 \r 报“找不到文件”；2026-10-25 的跑偏检查全部核对通过）。EXP-0022 起改为 LF，并在每次运行后把 runs.tsv 复制到备份目录（此前的 runs.tsv 已于同日补备份）。

## 文件

| 文件 | 作用 | 已验证范围 |
| --- | --- | --- |
| fastv_llava.py | 用于官方 LLaVA 的 FastV（KV cache，K 与保留数可调）；给定 `seed` 时改为同数量的均匀随机选择。诊断选项：`schedule` 多段剪枝（EXP-0007）、`score_prior` 分数除以位置曲线（EXP-0008）、`stratify` 把排序分到 n×n 个图像块里轮流取（EXP-0010）、`log_text_rows` 记录打分层上图像之后各文本 token 的注意力行（EXP-0011）；`score_kind="object_cos"` 改用图像 token 的输入 embedding 与物体名 embedding 的余弦相似度打分（只适用于 POPE 式题目），`exclude_padding` 不保留完全落在填充区的 patch（EXP-0011 D3）；`score_kind="mix"` 把名额分成两份，`mix_question` 个按物体名余弦、其余按 FastV 分数，第二轮先填空块；`score_kind="contrast_max"` / `"contrast_mean"` 用物体名余弦减去同一 token 与其他类名余弦的最大值 / 均值（`contrast_ids` 给出词表，EXP-0013）；都不设置时计算与原来相同 | 与 FastV 官方 lmms-eval 实现在随机小模型上逐步输出一致（相对误差 ≤3.3e-7（原写 ≤3e-7；2026-10-29 完整阅读补注：低于实测，EXP-0001-R5 Linux 预填充 3.23e-07，R11 Windows 预填充 3.13e-07）），且不重复写入 KV cache；EXP-0002（2026-10-29 完整阅读补注，2026-10-30 补到 EXP-0033：此后又加了 score_kind 的 query_mean / query_idf / attn_idf（EXP-0014）与 query_vis（EXP-0015）、route（EXP-0020）、zoom（IDEA-0004，EXP-0024、EXP-0025）、cover（含居中框与 cover_top 不围框对照；在 EXP-0024 中是 IDEA-0004 的对照，作为方法属 IDEA-0005，EXP-0026、EXP-0027）、cls 与 cls_unrouted（IDEA-0007，EXP-0030、EXP-0031）、div（DivPrune 的选择放在第 K 层，EXP-0032）；核对见 check_query_idf.py、check_query_vis.py、check_route.py、check_zoom.py、check_k1cls.py、check_divprune.py。VisionZip 上的覆盖在 visionzip_cover.py（IDEA-0006，EXP-0028、EXP-0029），DivPrune 在 divprune.py，EXP-0033 的组合在 combo.py） |
| run_eval.py | 在 LLaVA 的 model_vqa_loader 评测里切换 vanilla / FastV / VisionZip / 随机 / SparseVLM；Windows 上的 DataLoader worker；`--layer-token-log` 记录每层输入的 token 数；`--text-rows-log` 保存图像之后各文本 token 的注意力行（.npz），`--answer-logit-log` 记录第一个回答 token 为 Yes/No 的对数概率（EXP-0011）；`--layer-rows-log`（只用于 vanilla）在每次 prefill 记录几层的 head 平均行、第 0～1 层的逐 head 行和 hidden state 余弦相似度（EXP-0011 D2b；这次 prefill 改用 eager 注意力）；`--fastv-score object_cos`、`--fastv-exclude-padding`（EXP-0011 D3）；`--fastv-score mix / contrast_max / contrast_mean`、`--fastv-mix-question`（对比打分的词表是 80 个 COCO 类名，按 POPE 题目的写法取 token），以及 `--name-cos-log`（每张图记录图像 token 与 80 个 COCO 类名的输入 embedding 余弦，EXP-0013） | EXP-0002；EXP-0011、EXP-0013 的选项在 CPU 上自检（check_text_rows_log.py、check_layer_rows_log.py、check_object_cos.py、check_mix_names.py）（2026-10-29 完整阅读补注，2026-10-30 补到 EXP-0033：此后又加了 `--fastv-score` 的 query_mean / query_idf / attn_idf / query_vis / cls / div、`--fastv-visibility`、`--vocab-visibility-log`（EXP-0014、EXP-0015），`--fastv-route`（EXP-0020），`--fastv-zoom` 及其 pick / source / centre / on / box / smin / log 选项、`--fastv-cover`、`--fastv-cover-top`（IDEA-0004、IDEA-0005），`--vz-cover` 及其 on / centre / box / top 选项、`--vz-visibility`、`--vz-log`（IDEA-0006，调用 visionzip_cover.py），`--fastv-cls-unrouted`、`--fastv-cls-log`（IDEA-0007），`--method divprune`、`--divprune-keep`、`--divprune-log`、`--fastv-div-log`（EXP-0032，调用 divprune.py），`--method vzcand_div / div_qroute / div_ctx` 与 `--combo-*`、`--vz-fastv`（EXP-0033，调用 combo.py）；核对另见 check_query_idf.py、check_query_vis.py、check_route.py、check_zoom.py、check_vzcover.py、check_vzcover_dtype.py、check_k1cls.py、check_divprune.py、check_combo.py） |
| exp0002.py | 全部视觉实验的运行器（名字沿用 EXP-0002）：每次运行写 cmd、config、git、seed、env、gpu、stdout、答案、metrics、status 和 SHA256SUMS，复制到备份并核对校验和，追加到 runs.tsv（EXP-0022 起同时备份 runs.tsv）；拒绝复用已有的 run ID。配置与 benchmark（含 `gqa_testdev`，EXP-0022）的列表在文件开头 | EXP-0002～EXP-0033 的全部 GPU 运行（原写 EXP-0002～EXP-0021；2026-10-29 完整阅读补注，2026-10-30 补到 EXP-0033：EXP-0022、EXP-0024～EXP-0033 也都由它执行，EXP-0023 只在 CPU 上读已封存的输出，不经它；benchmark 此后又登记了 vqav2_dev2、vqav2_test3、gqa_val400（make_dev_sets.py，IDEA-0004）、textvqa_noocr、textvqa_ocr、vstar（DEC-0036）、vqav2_test4、gqa_val400b（make_confirm_sets.py，EXP-0031），配置一直到 EXP-0033 的组合；config.json 不记 .npy 输入表的哈希，见“已知局限”） |
| sparsevlm_sdpa.py | Windows 上把 SparseVLM 的 FlashAttention-2 调用换成 PyTorch SDPA（运行时替换，不改仓库文件） | 自检：与显式 softmax 注意力的差 ≤2.1e-07；EXP-0002-R36～R41 |
| score_pope_mme.py | 用官方 eval_pope.py 与 MME 的 convert_answer_to_mme.py、calculation.py 打分，写 metrics.json；VQAv2 子集（vqav2_test、vqav2_dev、vqav2_test2）用 LLaVA 的 m4c 评估器，逐题分数写 vqa_scores.jsonl；GQA testdev-balanced（`gqa_testdev`，EXP-0022）按 LLaVA 的 convert_gqa_for_eval.py 与官方 eval.py 的口径（去掉句末句点、小写、与标准答案完全匹配），逐题分数写 gqa_scores.jsonl | 给 LLaVA-1.5-13B 官方预测打分，与 MODEL_ZOO 一致（2026-10-29 完整阅读补注，2026-10-30 补到 EXP-0033：左列的说明此后又补了：vqav2_dev2、vqav2_test3、vqav2_test4 走 VQAv2 分支；gqa_val400、gqa_val400b 与 testdev 同一 GQA 计分；textvqa_noocr / textvqa_ocr 用 m4c（同 eval_textvqa.py），逐题写 textvqa_scores.jsonl；vstar 按选项字母计分，逐题写 vstar_scores.jsonl。已验证范围同样补了：TextVQA 由 check_textvqa_scoring.py 核对（13B 官方预测 61.254，eval_textvqa.py 61.25）；V* 由 check_vstar_scoring.py 核对，12 / 12 通过（EXP-0025）。另见“已知局限”的计分一条） |
| bootstrap_ci.py | 一次运行的指标或两次运行之差的 bootstrap 95% 区间（POPE 按子集重采样题目，MME 按子任务重采样图像） | EXP-0002 |
| analyze_exp0002.py | 从运行目录生成 EXP-0002 的全部分析表（code/output/EXP-0002/analysis/）（2026-10-29 完整阅读补注：没有防覆盖检查，重跑会改写已封存的 summary.json 与 tables.md；不要重跑，见 analyze_score_prior.py 一行） | EXP-0002 |
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
| check_object_cos.py | D3 选项的 CPU 自检：打分等于输入 embedding 的余弦，保留集合等于离线规则（排除填充区后的分块轮流取），填充区掩码与 analyze_object_boxes 相同，参数检查 | CPU 已运行，12 项通过（2026-10-24）；同时复跑了 check_text_rows_log、check_layer_rows_log、check_fastv_equivalence（相对差 ≤ 3.13e-7（原写 ≤ 3.1e-7，低于实测的 3.13e-7；2026-10-29 完整阅读补注）） |
| analyze_d3.py | EXP-0011 D3 的分析：POPE F1（原始标注与 RePOPE）、配对 bootstrap 差、保留集合在被问物体上的命中率、剂量关系、按大小五分位的新漏答率；先核对每个运行的重算 F1 与登记值一致 | code/output/EXP-0011/d3_pope |
| analyze_layer_rows.py | EXP-0011 D2b 的离线统计（第 2 版）：参与判定的算子（第 0 层注意力、逐 head、归一化后平均、输入与前两层输出的余弦）与更深层的描述项，在分块 + 排除填充区下的小物体命中率、P1、P2、物体 AUC；先用 R02 自检 | code/output/EXP-0011/d2b_coco_dev_v2（第 1 版的输出在 d2b_coco_dev，AUC 有缺陷） |
| analyze_text_rows.py | EXP-0011 D2 的离线统计：第 1 层三种打分（最后一个 token、题目句、物体名）的同图不同题 Jaccard（P1）、物体对准比值（P2）、24 种模拟选择的小物体命中率（P3）和剂量关系；先用模拟复现运行的保留集合作自检 | code/output/EXP-0011/d2_coco_dev（开发集）、d2_pope |
| make_coco_dev.py | EXP-0011 的开发集：COCO val2014 中 POPE 没用到的 300 张图，按 POPE 的方式出 1800 题（种子 0），在它上面做选择，POPE 只作测试集 | 已生成（data/llava_eval/pope/llava_coco_dev*.jsonl，校验和在 llava_coco_dev.SHA256SUMS） |
| make_vqav2_sets.py | EXP-0018：VQAv2 验证集在本地 COCO val2014 图像上的两个子集。test 为 POPE、coco_dev 都没用的图中按种子 0 抽 1000 张的全部 5246 题；dev 为 coco_dev 300 张图上的 1485 题。`test2`（EXP-0020）：同一图像池去掉 test 的图后按种子 1 再抽 1000 张，5415 题 | 已生成（data/llava_eval/vqav2_val/，test、dev 的校验和在 SHA256SUMS，test2 的在各运行的 config.json：c086f656…、4eb4e5c2…）；划分经多代理复核 |
| make_coco_dev3.py | EXP-0013 的开发集：coco_dev 的 1800 题不变，在同样 300 张图上按 POPE 论文的做法补 popular、adversarial 负样本各 900 题 | 已生成（llava_coco_dev3*.jsonl，校验和在 llava_coco_dev3.SHA256SUMS） |
| check_mix_names.py | EXP-0013 选项的 CPU 自检：`--fastv-score mix` 的保留集合等于离线两轮规则（混合 30 = object_cos，混合 0 = 强对照 + 排除填充区）；对比打分等于重算值、排除与被问物体同拼写的类名；`--name-cos-log` 的数值与去重；coco_dev3 全部 prompt 的物体名 id（真实 tokenizer）；参数检查 | CPU 已运行，38 项通过（2026-10-24）；同时复跑了其余 check（FastV 等价相对差 ≤ 3.13e-7）与 smoke_test |
| analyze_kept_tokens.py | EXP-0006：FastV 与随机选择保留了图像的哪些位置（2026-10-29 完整阅读补注：没有防覆盖检查，重跑会改写已封存的 `code/output/<实验>/analysis`（kept_summary.json、kept_stats.md）；不要重跑，见 analyze_score_prior.py 一行） | code/output/EXP-0006 |
| analyze_missed_objects.py | SQZ-EXP-0002 补充：各剪枝方法是否漏掉同一批 POPE 物体（F10）（2026-10-29 完整阅读补注：没有防覆盖检查，重跑会改写已封存的 code/output/EXP-0002/deepdive；不要重跑，见 analyze_score_prior.py 一行） | code/output/EXP-0002 |
| analyze_score_prior.py | EXP-0008：FastV 打分在空白图与真实图上的位置曲线，写出校正用的位置先验（2026-10-29 完整阅读补注：本脚本没有“已存在就停止”的检查，重跑会改写已封存的 EXP-0008/analysis，包括 position_prior.npy。这个文件是 exp0002.py 里 SCORE_PRIOR 的来源：K = 2 的所有 prior 配置（`fastv<预算>_prior_strat_nopad` 即强对照 A，以及由它派生的 qvis、qroute、放大与覆盖配置）都读它，而运行的 config.json 只记路径、不记它的 sha256，所以不要重跑。现存文件完好：sha256 85c73a91… 与 EXP-0008/analysis/SHA256SUMS 一致。K = 1 的先验在 EXP-0024/prior_k1，由 make_prior_k1.py 写出，那个脚本拒绝覆盖。） | code/output/EXP-0008/analysis |
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
| check_route.py | EXP-0020 的 CPU 自检：`--fastv-route yesno_howmany` 下，路由的题与 query_vis、不路由的题与强对照的保留集合相同（两者交替时也一样），记录正确；真实 tokenizer 下从 prompt ids 得到的决定与题目文字上的规则逐题相同，POPE 与 MME 感知子任务全部被路由 | CPU 已运行，18 项通过（2026-10-25；原写 17 项，EXP-0020 计划 v1.1 加了 192 档配置核对后为 18 项，见 EXP-0020 记录；2026-10-29 完整阅读补注） |
| analyze_0019.py | EXP-0019：A+B 与无先验强对照 A₀ 的 POPE 比较（按图像重抽），最终方法、vanilla、A、A₀ 的 margin AUC 与误报 | code/output/EXP-0019/analysis（a17545fa…） |
| analyze_0020.py | EXP-0020：路由与 A、A′+B+V 在 test2 上的比较及实现核对；运行前已在 EXP-0018 的输出上空跑 | code/output/EXP-0020/analysis |
| squeeze_0020.py、squeeze_0020_hybrid.py | SQZ-EXP-0020（事后探索）：路由增益的分解、按开头词与题型的差（含 test1→test2 的复现）、与 VisionZip-64 的差距、逐题 oracle、yes/no 的答案偏向；追加的 do 开头子集与拼接组合（由已有运行拼出）。GPU 运行期间从 D:/6CJNM007/tmp 运行，提交时只把 CRLF 换成 LF | code/output/EXP-0020/squeeze、squeeze_hybrid |
| analyze_0021.py | EXP-0021：192 档路由与 A，128 档 VisionZip、SparseVLM 与 EXP-0020 的 128 档运行在 test2 上的比较（只作描述）；运行前用 EXP-0020 的输出空跑 | code/output/EXP-0021/analysis |
| make_main_table.py | 论文主表草稿：方法 × 预算 × benchmark（POPE 原始标注与 RePOPE、MME-P、VQAv2 test2），每格回链运行 ID，只读已封存的输出 | code/output/PAPER-20261025/main_table |
| make_gqa_set.py | EXP-0022：从官方 GQA 压缩包写出 testdev-balanced 的逐题标注（答案、结构类型），取出 LLaVA 题目文件用到的 398 张图并记录 sha256；可读本地 images.zip，也可用 HTTP 分段请求只读同一官方文件的目录与这 398 个成员（CRC 校验）；核对题号、图像名与题目文字和 LLaVA 的文件一致 | 已运行（data/llava_eval/gqa/、data/gqa/） |
| analyze_0022.py | EXP-0022：GQA 上路由与 A、A′+B+V 的比较（判据 Q1～Q3）、按答案类型与结构类型、二选一题、oracle、yes 比例与参照；运行前用合成的替身运行空跑 | code/output/EXP-0022/analysis |
| squeeze_0022.py | SQZ-EXP-0022（事后探索，看过 0022.md 之后写）：路由在 GQA 上的增益从哪来、二选一题、GQA 的语义与细分题型、yes/no 偏向、VisionZip-64 与 EXP-0020 S4 的事后拼接、128 档、与 VQAv2 test2（EXP-0020）的对照；逐题分数的算法与 analyze_0022.py 相同，并对照已封存的 0022.json，按图像重抽用同一组抽样 | code/output/EXP-0022/squeeze（SHA256SUMS e39d8d8f…）；aa1434d 提交，与封存副本相同 |
| diag_0023.py | EXP-0023（CPU 诊断，不调用模型）：开放题的答案落在被点名物体（topic）周围多远——COCO 框映射到 24×24 网格（框盖住一个 patch 的 ≥ 25% 才算），按到 topic patch 的 Chebyshev 距离分环；为两个候选模块（以物体为锚的邻域、预算内放大）提供依据。数据：VQAv2 test1（配 EXP-0018 的运行）与 dev（只用标注） | code/output/EXP-0023/analysis（81e93b38…）；bfff4bc 提交，与封存副本相同 |
| make_dev_sets.py | IDEA-0004 的数据（2026-10-25，任何开发运行之前抽一次）：VQAv2 dev2（种子 2，1000 张图，5192 题，调参用）、test3（种子 3，1000 张图，5316 题，留给预注册的确认）、GQA val400（种子 0，400 张图，4786 题，与 testdev 不重合）；拒绝覆盖 | 已生成（vqav2_val/SHA256SUMS_idea0004、gqa/SHA256SUMS_val400、gqa/images_val400_SHA256SUMS） |
| make_textvqa_vstar.py | IDEA-0004 的确认数据（DEC-0036，经用户同意于 2026-10-25 下载）：TextVQA v0.5.1 val（核对服务器给的大小与 md5，只解压 3,166 张 val 图并逐图记 sha256；LLaVA 的题目文件按（图像 id，题目）改用 TextVQA 自己的题号，带与不带 OCR 行各一份，逐题标注另写一份）与 V* Bench 的 191 题（版本见 vstar_bench.revision.json）；拒绝覆盖，新文件旁写校验和 | 已生成（data/llava_eval/textvqa/、vstar/，图像在 data/textvqa/train_images） |
| zoom_share.py | EXP-0024 运行前、只用题目文字的事实：开发集与 TextVQA、V* 上有多少题会被放大（不路由且 S ≥ 0.2），分词、路由、IDF 与可见性的口径与 run_eval 相同；不读答案、不读图、不调用模型（开头说明里“按 question_type 列出”一句与代码不符，见文件内补注） | EXP-0024、EXP-0025 记录（例如 TextVQA 路由 382 题、不路由 4618 题；V* 路由 83 题、不路由 108 题） |
| make_prior_k1.py | EXP-0024：K = 1 的位置先验，即 EXP-0024-R01（空白图上的 pope_img20_blank）每个 patch 第 0 层分数的平均，做法同 analyze_score_prior.py 对 K = 2（EXP-0008）；拒绝覆盖，写完复制到备份并核对 | code/output/EXP-0024/prior_k1（SHA256SUMS 10558482…；position_prior.npy 60a047d7…），exp0002.py 的 PRIOR_K1 读它 |
| check_zoom.py、kept_dump.py | IDEA-0004 的 `--fastv-zoom` 与 `--fastv-cover`（IDEA-0005 也用）的 CPU 自检（随机小 LLaVA，EXP-0020 的路由配置，640 × 480 的图）：不加新选项时保留集合与生成结果等于已提交的代码（kept_dump.py 在两个进程里分别跑新旧代码）；放大的各种取法与对照、覆盖（含居中框、K = 1、不围框对照）的离线规则，框的约束、预算、日志，参数与配置核对。要从新代码的副本目录运行（见文件内 2026-10-29 补注） | 56/56（EXP-0024）、64/64（EXP-0025）、71/71 与 77/77（EXP-0026）；此后每次改 FastV 一族的代码都从副本目录重跑 77/77（EXP-0028、EXP-0030、EXP-0032、EXP-0033） |
| analyze_0024.py | EXP-0024：IDEA-0004（预算内放大）在开发集 VQAv2 dev2、GQA val400 上的分析（64 档）：候选与同 K 路由比，各有“如实运行”与“只放大不路由题”（由已有答案拼接）两种；对照、选择规则、归因与各项分解都在运行前写定；逐题分数从 answers.jsonl 重算并与每次运行的分数文件核对 | code/output/EXP-0024/analysis（af76abaa…；判读“不像”，DEC-0039） |
| squeeze_0024.py | SQZ-EXP-0024（事后探索）：格点与池化的裁剪 token；覆盖 16 的增益与框是否命中被点名物体、是否偏离图像中心、框内原有 token 多少的关系，以及翻转；K = 1 的作用按答案类型，K = 1 保留了 K = 2 的多少；两者的增益是否落在同一批题 | code/output/EXP-0024/squeeze（81b7eb6f…）；a2a51a0 提交，与封存副本相同 |
| analyze_0025.py | EXP-0025：EXP-0024 选出的放大变体在看细节的题上的检验（TextVQA v0.5.1 val 不给 OCR 为主检验、给 OCR 为参照，V* Bench），判据 T1～T3 与“大”的门槛在运行前写定；对照为同 K 路由、全图 lattice、居中框 | code/output/EXP-0025/analysis（e5f2c6e3…；T1～T3 都不成立，IDEA-0004 结束，DEC-0041） |
| check_textvqa_scoring.py | TextVQA 计分路径的 CPU 核对：LLaVA 官方 LLaVA-1.5-13B 的预测按（图像 id，题目）映射到 TextVQA 的题号，用 score_pope_mme.score_textvqa 计分，与 LLaVA 的 eval_textvqa.py 对原文件的计分比 | 本项目 61.254，eval_textvqa.py 61.25（论文 61.3；EXP-0025）；不一致时只打印 MISMATCH、退出码仍为 0（见“已知局限”） |
| check_vstar_scoring.py | V* 计分路径的 CPU 核对：常见回答形式的选项字母解析；合成答案文件上全对（必须 100%）、全答 A（必须等于标签为 A 的比例）、全部无法解析（0%，都记为未解析） | 12 / 12 通过（EXP-0025） |
| analyze_0026.py | EXP-0026：IDEA-0005（FastV 上的定位后覆盖）在 dev2、val400 上的开发运行（64 档）：复用 EXP-0024 的 K = 2 路由、K = 1 路由与 K = 2 的覆盖 16；新加居中框覆盖、K = 1 的覆盖 16、覆盖 8，以及 v1.1 的逐 token 相关性对照（每题 query_vis、不围框）；选择规则、判读、归因与核对在运行前写定 | code/output/EXP-0026/analysis（6d7cc3db…；DEC-0042） |
| analyze_0027.py | EXP-0027：IDEA-0005 所选配置（K = 1、留 47、不路由题覆盖 16）在 VQAv2 test3 与 GQA testdev-balanced 上的确认（Q1 比 K = 1 路由，Q2 比 K = 2 路由，Q3 比 VisionZip-64 只描述）；GQA testdev 复用 EXP-0022 的运行 | code/output/EXP-0027/analysis（f0b9088e…；Q1 部分成立、Q2 成立，覆盖不作为方法的贡献，DEC-0045）；判读输出的一处缺口见“已知局限” |
| visionzip_cover.py | IDEA-0006：VisionZip 上的定位后覆盖（`--vz-cover`）。题目的 IDF × 可见性查询用 fastv_llava.zoom_box 在 24 × 24 网格上定位方框（与 IDEA-0005 对同一题、同一图定位的框相同），框内取 VisionZip 自己的主导 patch 数加 n 个、框外取其余（都按 CLS 注意力，完全落在填充里的 patch 不算框内），上下文合并照 VisionZip；只覆盖不路由的题（或每题），居中框与不围框两种对照；没有查询的预填充照跑 VisionZip 的原 forward | check_vzcover.py、check_vzcover_dtype.py；EXP-0028、EXP-0029。_forward 与 source/VisionZip（8f86b55c）的 CLIPVisionTower.forward 逐行对照（2026-10-30）：batch 张量分支的 30 行语句相同（其中 2 行只差空格），只多了覆盖的 5 行，list 分支改为报错 |
| check_vzcover.py | IDEA-0006 的 CPU 自检（随机小 LLaVA，VisionZip-64：主导 54、上下文 10，640 × 480 的图）：不加 `--vz-cover` 时就是 visionzip.visionzip 本身，复制的 forward 在没有查询时与 VisionZip 相同；路由题保持 VisionZip；不路由题的框、框内外的选法与 LLM 收到的 token（独立写出）；居中框、每题覆盖、覆盖 32、不围框对照（v1.1）、没有可见权重的题；参数与配置核对 | 40/40，v1.1 后 45/45（EXP-0028）；dtype 修正后重跑 45/45 |
| check_vzcover_dtype.py | EXP-0028 的 R03～R12 在 GPU 上第一个答案之前失败（visionzip_cover._cover 里 Float 与 Half 的矩阵乘）之后的修正核对：小 LLaVA 按真实布局设置（视觉塔 float32，投影层、LLM 与图像用 bfloat16 代替 float16），已提交的 15e75ed 在此布局上报与 GPU 相同的错，修正版能运行，路由题与 VisionZip 相同，框、主导 patch 与 LLM 收到的 token 按规则（上下文 token 在 bfloat16 的舍入内相同） | 11/11（EXP-0028） |
| analyze_0028.py | EXP-0028：IDEA-0006 在 dev2、val400 上的开发运行（VisionZip-64 上的覆盖 16、覆盖 32、每题覆盖 16，居中框与不围框对照）；v1.2 只改运行表（失败的 R03～R12 以 R13～R22 按原顺序重跑） | code/output/EXP-0028/analysis（2f6c5fb0…；判读“有希望”，DEC-0048） |
| analyze_0029.py | EXP-0029：IDEA-0006 所选配置（每题覆盖 16，从 EXP-0028 已封存的分析读）在 test3、testdev 上的确认（Q1 比 VisionZip-64，Q2 定位，Q3 区域对逐 token 相关性，Q4 TextVQA 无 OCR 只描述） | code/output/EXP-0029/analysis（5f7ae25a…；Q1～Q3 不成立，IDEA-0006 结束，DEC-0054） |
| check_k1cls.py | IDEA-0007 的 CPU 自检（`--fastv-score cls`、`--fastv-cls-unrouted`；K = 1、留 47，640 × 480 的图，填充不保留）：返回注意力的视觉塔 forward 给出普通 LLaVA 的特征，分数是第 -2 层 CLS 注意力的头和（独立重算）；保留集合；路由题的保留与答案与 K = 1 路由相同；覆盖 16 时的框内取法；日志；分数最大的 53 个就是 VisionZip 自己的主导 patch；参数与配置核对 | 32/32（EXP-0030，tmp 与仓库）；EXP-0032 回归 32/32 |
| analyze_0030.py | EXP-0030：IDEA-0007 在 dev2、val400 上的开发运行：一层全看（K = 1、留 47）后按视觉塔的 CLS 注意力排序，三个候选 k1cls、k1qcls、k1qcls_c16 与 VisionZip-64、IDEA-0006 的所选比；选择规则与判读在运行前写定 | code/output/EXP-0030/analysis（c6a48c42…；选中 k1qcls_c16，进入确认 EXP-0031） |
| make_confirm_sets.py | 新的确认数据（2026-10-26，在 EXP-0030 的任何结果与任何运行之前抽一次）：VQAv2 test4（种子 4，从图像池去掉 test、test2、dev2、test3 的图后抽 1000 张，5333 题）、GQA val400b（种子 1，从 val-balanced 去掉 val400 的图后抽 400 张，5165 题，与 val400、testdev 不重合）；拒绝覆盖，写出的每个文件都记进 SHA256SUMS | 已生成（vqav2_val/SHA256SUMS_test4、gqa/SHA256SUMS_val400b、gqa/images_val400b_SHA256SUMS）；EXP-0031 使用 |
| analyze_0031.py | EXP-0031：IDEA-0007 所选配置（从 EXP-0030 已封存的分析读）在新数据 test4、val400b 上的确认：Q1 比 VisionZip-64，Q2 比 IDEA-0006 的所选，Q3 比只换排序的 FastV 对应配置，Q4 TextVQA 无 OCR 只描述 | code/output/EXP-0031/analysis（47838ee0…；Q1、Q2 部分成立，Q3 不成立，编码器排序不作为方法的贡献，DEC-0059）；预注册的拼接分析脚本在 sealed_copies/EXP-0031/splice_analysis/ |
| divprune.py | DivPrune 式的选择（Alvar 等，CVPR 2025，arXiv 2503.02175），按论文的 Algorithm 1 自写，没有下载官方代码：余弦距离 d = 1 − cos 的最大最小多样性，第一个取最近邻最远的 token，之后每次取离已选集合最远的；贪心顺序与取多少无关。`apply_divprune` 在进入 LLM 之前保留每张图 keep 个投影输出（原顺序）；`maxmin_order` 也供 fastv_llava 的 score_kind "div"（K = 1：第 0 层看全部，第 1 层起按同一特征上的选择） | check_divprune.py；EXP-0032（DEC-0061）、EXP-0033 |
| check_divprune.py | EXP-0032 的 CPU 自检（随机小 LLaVA）：maxmin_order 等于直接按 Algorithm 1 写的参照（float64、在预先算好的距离矩阵上循环），前 47 个就是取 47 个的选择；`--method divprune` 时 LLM 看到系统提示、64 个图像 token 与文字，这 64 个就是普通 LLaVA 在所选 patch 上的投影输出（原顺序），每张图一条日志；`--fastv-score div`（K = 1）第 0 层看全部 576 个、第 1 层起留同一选择的前 47 个；参数与配置核对 | tmp 29/29（有两个 exp0002.py 变体），仓库 26/26（EXP-0032、EXP-0033 的排队脚本在提交后重跑） |
| analyze_0032.py | EXP-0032：DivPrune-64 在本流水线里的校准（Q1 比 VisionZip-64，只描述）与一层全看是否帮助多样性选择（Q2：k1div − div64）；同时描述 IDEA-0006 的所选、K = 1 路由、IDEA-0007 的 K = 1 CLS 排序与 EXP-0030 的所选 | code/output/EXP-0032/analysis（610b1536…；DivPrune-64 明显强于 VisionZip-64，一层全看没有帮助，DEC-0061） |
| combo.py | EXP-0033（第 ④ 步第二轮，探索性）：两种方法的组合，都在进入 LLM 之前（每层只看保留的图像 token）：X2 `apply_vzcand_div`（VisionZip 的 CLS 分数前 128 个作候选，在其投影输出上按最大最小顺序留 64）；X3 `apply_div_qroute`（DivPrune-64 为底，路由题先取与 IDF × 可见性查询最相似的 round(64 S) 个，再按最大最小顺序续选）；X4 `apply_div_ctx`（DivPrune 留 54，其余按 VisionZip 的上下文规则合并成 10 个；相似度用视觉塔特征的余弦代替 VisionZip 第 -2 层的 key，是近似）。X1（VisionZip-128 之后接原版 FastV）由 run_eval.py 的 `--vz-fastv` 实现 | check_combo.py；EXP-0033 |
| check_combo.py | EXP-0033 组合的 CPU 自检（随机小 LLaVA）：参照都在这里独立写出（Algorithm 1 用 float64 直读、VisionZip 的上下文规则写成循环、查询与 S 从替身表重算）：X2 的候选、保留与 LLM 收到的 token，X3 路由题的前 round(64 S) 个与续选、不路由题等于 DivPrune-64，X4 的 54 + 10，以及 `--vz-fastv` | tmp 37/37（有两个 exp0002.py 变体），仓库 32/32（EXP-0033 的排队脚本在提交后重跑） |
| analyze_0033.py | EXP-0033：探索性组合的分析（没有判据；计划固定报告的内容与三种“意外”：互补、抵消、按题型反转），四个组合 X1～X4 各与它的两个单独方法比；复用 VisionZip-64、K = 2 路由与 DivPrune-64 的已有运行；逐题分数从 answers.jsonl 重算并与每次运行的分数文件核对 | code/output/EXP-0033/analysis（d1059016…） |
| sealed_copies/ | 只封存在 code/output 的分析文件夹里、此前从未提交的一次性脚本与 shell 包装（2026-10-25 跑偏检查时收集，去掉 CR 后原样提交；MANIFEST.tsv 列出来源文件夹与封存时的 sha256）。其中两份是后来改过的脚本的第一版：EXP-0002/analysis_v1/analyze_exp0002.py、EXP-0011/d2b_coco_dev/analyze_layer_rows.py。（2026-10-29 完整阅读补注：EXP-0014 的 squeeze/、squeeze_v2/、squeeze_v3/ 三份 squeeze_query.py 相同，都是 v3，sha256 8b4fd191…，见 MANIFEST.tsv 第 34～36 行；v1、v2 的脚本没有保留，前两个目录的输出不能用目录里的脚本复现。EXP-0024 运行前审查的 44 个辅助脚本（25 个 .py、19 个 .sh，在 code/output/EXP-0024/pre_run_review/review/ 的 8 个子目录里）也只在 code/output 与备份里，没有收入本文件夹。）（2026-10-30 补，DEC-0058 的完整阅读：此前只在 D:/6CJNM007/tmp 的 12 个脚本已补封存到 code/output/<实验>/late_sealed_20261030/（EXP-0014 的 peek_mme.py；EXP-0018 的 verify.py、score_runs.py、scan_outputs.py；EXP-0022 的 route_share_gqa.py、verify_full_zip.py；EXP-0024 的 dryrun_0024.py、vstar_route_cat.py、check_0024.sh、cover_stats.py；EXP-0026 的两份 dryrun_0026.py；各有 notes.md 写明原路径与修改时间），连同只封存在 code/output 的 EXP-0027、EXP-0028 的 explore 脚本、SQZ-EXP-0029～0032 的脚本与 EXP-0031 的拼接分析脚本，共 24 个，收入本文件夹（MANIFEST.tsv 第 38～61 行）。仍只在 tmp 的：EXP-0022、EXP-0024 深挖用的 run_squeeze 系列与 seal_squeeze.sh，封存工具 seal_folder.sh 与各实验的 seal 系列脚本，排队脚本 chain_00xx*.sh。MANIFEST.tsv 只有 code/output 来源一列，放不下 tmp 来源。） | 与封存副本逐字节相同（除 CR）；2026-10-30 收入的 24 个同样核对 |
| probes/*.py | 三个仓库的代码级探针和 FLOPs 计算 | CPU 已运行，输出见 EXP-0001 |

SparseVLM 也通过 `run_eval.py` 运行：在 `sparsevlm` 环境里用 `--method sparsevlm --retained-tokens <预算>`。它调用 SparseVLM 自带的 model_vqa_loader，剪枝由 SparseVLM 的模型代码执行；`run_eval.py` 会检查加载的确实是 SparseVLM 的版本。

## 已知局限（2026-10-30 完整阅读，DEC-0058）

下面几条不改变已报告的结果（已核对过，或口径已写在相应记录里），但复用代码或重跑时要知道：

- **exp0002.py**：config.json 记题目文件与标注的 sha256，不记 method_args 里 .npy 输入表的哈希（EXP-0015 的 visibility.npy 0eb2258c…，EXP-0008 的 position_prior.npy 85c73a91…，EXP-0024 的 prior_k1/position_prior.npy 60a047d7…；现存文件都与封存的 SHA256SUMS 一致）；git.txt 里的 diff 不含未跟踪的文件；source 下的子仓库缺失时 sh() 不查返回码；备份目录已存在时 copytree 报错，这次运行在 runs.tsv 里没有行。下一次改 exp0002.py 时补记输入表的哈希。
- **不拒绝覆盖的分析脚本**：analyze_score_prior.py、analyze_kept_tokens.py、analyze_missed_objects.py、analyze_exp0002.py 重跑会改写已封存的输出（各行与文件内都有补注），不要重跑。
- **核对脚本的退出码**：check_fastv_equivalence.py 只打印相对误差，没有阈值；check_textvqa_scoring.py 不一致时打印 MISMATCH，退出码仍为 0。读这两个要看打印的数字。
- **计分**：score_pope_mme.py 的 POPE、MME 只核对子集与子任务的列表，不核对每题都有答案（metrics.json 记 n_scored）；vstar_letter 先看开头的字母，以独立的 “A” 或 “a” 开头的句子（如 “A man …”）会被读成选项 A。bootstrap_ci.py 的 POPE 按顺序配对，答案少于标注时静默截断（与 eval_pope.py 相同），且 POPE 以外的运行一律按 MME 读。analyze_repope.py 的 paired_boot 在每个 split 内按题重抽，不是按图像；按图像的区间见 main 分支的 analysis/审计记录_第12步_2026-10-25.md（AU-1）。
- **随机对照**：fastv_llava 的随机选择在整次运行里用同一个 Generator，抽到什么随题序而定；exp0002.py 的 random 配置不排除填充区，能选到完全落在填充里的 patch（原版 FastV 也不排除），早期“比随机”的结论按这个口径读。
- **数据文件的行尾**：make_vqav2_sets.py 等在 Windows 上用文本模式写，VQAv2 的六个子集（test、dev、test2、dev2、test3、test4，题目与标注）以及 coco_dev、coco_dev3、pope_img20 系列与 pope50 的题目文件是 CRLF；GQA、TextVQA、V*、POPE 原题是 LF（2026-10-30 逐个核对）。读 JSON 不受影响；记下的 sha256 对应现有的 CRLF 文件，不要转换行尾。
- **探索脚本的口径**：squeeze_0020_hybrid.py 第 48 行的 yes 比例不去句末句点，squeeze_0020.py 第 172 行去掉（rstrip(".")），两处 yes 比例的口径不同；squeeze_0018_boxes.py 不核对 n_prefills。
- **analyze_0027.py 第 179～185 行**：Q1 部分成立而 Q2 不成立时会落到“Q1、Q2 都不成立”一支（计划没规定这种组合）；EXP-0027 的实际结果是 Q1 部分成立、Q2 成立，没有走到这一支。
- **路径**：zoom_share.py 在本目录找不到 run_eval.py 时回退到 tmp 的副本（文件内有补注）；kept_dump.py 写死仓库的绝对路径；prepare_eval_data.sh 没有固定 eval.zip 的哈希；run_pope_mme.sh 用的是旧路径（只看过 DRY_RUN）。
- **make_main_table.py**：同一行的各列可能来自不同实验（例如强对照 A 的 64 档：POPE 为 EXP-0011-R07，MME 为 EXP-0014-R04，VQAv2 为 EXP-0020-R01），每格都回链运行 ID；路由在 POPE 与 MME-P 上每题都走查询，所以与 A′+B+V 共用运行。

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
# （2026-10-29 完整阅读补注：v1.5 是会移动的分支。要固定到“固定版本”表里的 commit，把上一行最后一条换成 git -C SparseVLMs checkout 734d9003879689713a132b83297dfad060a5c9af；本机克隆里 origin/v1.5 指向 734d900，本机的 source/SparseVLMs 就是按这个 commit 检出的（detached HEAD）。）
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
- FastV：K 层以后保留 keep 个。本目录按“32 层平均值相等”换算，K=2 时 192/128/64 对应 keep=166/98/30；实测 64 档为前 2 层 576、之后 30 个（层平均 64.125）。（2026-10-29 完整阅读补注：K = 1 时留 47 个：(576 + 31×47)/32 = 63.53（EXP-0024 起的 fastv64k1 开头的配置，IDEA-0005、IDEA-0007 用）。放大（IDEA-0004）在打分层之前多插 n 个 token、打分层之后图像 token 总数仍为 keep：K = 2、n = 16 时层平均 65.125（比 64.125 多 1.0），K = 1、n = 16 时 64.03（check_zoom.py 的预算核对）。VisionZip-64 上的覆盖（IDEA-0006）仍是每层 64 个。2026-10-30 补：DivPrune-64（EXP-0032）每层 64 个，K = 1 的多样性选择同 63.53；EXP-0033 的组合 X1（VisionZip-128 之后 FastV 第 2 层起留 60）为 (2×128 + 30×60)/32 = 64.25，其余组合每层 64 个。）

三者的理论 FLOPs 基本相同（probes/flops_budget.py），但每层看到的信息、显存峰值和延迟不同。论文中应同时报告 FLOPs、实测每层 token 数和实测延迟。（2026-10-29 完整阅读补注：上面“三者的理论 FLOPs 基本相同”按 EXP-0001 的名义分布计算（SparseVLM 前 2 层 576、之后 66/30/17），已被上一条的实测分布取代（F4）。按实测每层 token 数（SparseVLM-64：前 3 层 576，之后 82、32、18，各占 4、9、16 层，层平均 82.25；分段取自 exp0002.py 的 SVP），用 probes/flops_budget.py 的同一公式（系统提示 35、文本 20 token）重算，占 vanilla 的比例：FastV-64 为 18.61%，VisionZip-64 为 18.48%，SparseVLM-64 为 21.49%（名义分布下为 18.23%），实测比前两者高 15.5%～16.3%；文本 60 token 时为 23.34%、23.22%、26.05%（高 11.6%～12.2%）；128 档（文本 20 token）FastV-128 为 28.55%、VisionZip-128 为 28.49%、SparseVLM-128 实测为 31.39%（高 9.9%～10.2%）。所以“同预算下理论 FLOPs 基本相同”只对名义分布成立；按“64”并列比较 SparseVLM 时，FLOPs 应按实测分布报告。flops_budget.py 是 EXP-0001 的证据，保持原样。）
