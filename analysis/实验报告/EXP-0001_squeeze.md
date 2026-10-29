# SQUEEZE_REPORT：三个 token 剪枝仓库的代码级核验（EXP-0001）

状态：完成（2026-10-23，实验室电脑补做；交接文档第 7 步）。（2026-10-29 完整阅读补注：本报告里按预算表推算的 SparseVLM 层平均 188.2/123.9/61.7、前 2 层 576、比名义预算少 2～4%（交接包的 P-SQZ1-1 与 F4、深度分析“机制、消融和交互”一行），以及“学到什么”一节里的名义预算比层平均高 2～4%，已被 EXP-0002 的实测取代：SparseVLM 前 3 层为 576，层平均 206.5/145.4/82.25，比名义预算多 7.6%/13.6%/28.5%（EXP-0002-R47～R49）。见 SQZ-EXP-0002 的 F4 更新与已发现的实验现象 F4。）
报告 ID SQZ-EXP-0001 / 实验 ID EXP-0001 / 输入记录：analysis/实验报告/EXP-0001_record.md（2026-10-23 版）/ 日期 2026-10-23。

## 结构化交接包

```yaml
package_type: SQUEEZE_REPORT
package_version: research-pipeline-contract v1.0
package_id: SQZ-EXP-0001
squeeze_id: SQZ-EXP-0001
created_at: 2026-10-23
source_ids: [EXP-0001, PAPER-0004, PAPER-0005, PAPER-0006]
evidence_status: verified
evidence_locations:
- analysis/实验报告/EXP-0001_record.md（原始输出节，云端 Linux）
- code/output/EXP-0001/rerun-windows-2026-10-23/（实验室电脑复验 R1～R6 的日志与 SHA256SUMS；备份 D:\6CJNM007\backup\EXP-0001\rerun-windows-2026-10-23\）
- source/SparseVLMs@734d9003（v1.5）llava/model/language_model/score.py 第 7～14、31、35 行；modelling_sparse_llama.py 第 69、223 行
missing_evidence:
- 云端原始日志未保存，Linux 侧只有记录中抄录的文本
- SparseVLM v1.5 的 batch=2 行为未测（probe_sparsevlm.py 与 v1.5 的函数签名不兼容）
- 真实权重下 SparseVLM 每层实际视觉 token 数（含回收 token）未测
next_skill: deep-read Mode C
input_experiment_record_ids: [EXP-0001]
claim_scope: 三个仓库在随机权重小模型上的代码行为、预算口径与理论 FLOPs；不涉及任何真实准确率或速度
data_quality_and_provenance_gaps: 见“证据审计”
primary_result_with_uncertainty: 确定性代码检查；6 个运行在第二个平台上复现（见“深度分析”第 1、6 行）
process_dynamics: 不适用（无训练过程）
mechanism_and_intermediate_findings: 见“深度分析”第 5 行
errors_heterogeneity_and_boundary_conditions: 见“深度分析”第 3、4 行
robustness_sensitivity_and_alternative_explanations: 见“深度分析”第 6 行
new_P: [P-SQZ1-1, P-SQZ1-2]
new_F: none（F4 补入 v1.5 的预算表细节；跨平台复现记为证据 E-SQZ1-3，不作为现象）
new_M: none
evidence_index: 见“证据审计”
open_questions: [真实权重下 SparseVLM 每层实际 token 数是否等于预算表, SparseVLM v1.5 的 batch=2 行为]
next_routes: [EXP-0002 中用前向 hook 测 SparseVLM 每层视觉 token 数（少量 POPE 题即可）, 改写 probe_sparsevlm.py 兼容 v1.5 的 retained_tokens 参数, deep-read Mode C 回流]
mode_C_input_bundle:
  source_ids: [EXP-0001, SQZ-EXP-0001]
  P:
  - id: P-SQZ1-1
    text: "同一“预算”下三种方法每层看到的视觉 token 数不同，SparseVLM 的名义预算还比实际层平均高 2～4%；按名义预算并列比较是否公平（2026-10-29 完整阅读补注：2～4% 的方向已被 EXP-0002 实测推翻，SparseVLM 实际比名义预算多 7.6%/13.6%/28.5%，见本报告开头的补注）"
    evidence: [R2, R4, score.py@734d9003 第 7～14 行]
    status: DE（口径差异是代码事实；对准确率的影响待 EXP-0002）
  - id: P-SQZ1-2
    text: "SparseVLM 的回收 token 是否让真实前向中的每层 token 数偏离预算表"
    evidence: []
    status: UN
  F:
  - id: F4（细节更新，台账已有此 ID）
    text: "SparseVLM v1.5（ICML 版，EXP-0002 实际使用）的预算表与 main 分支 v1.0 相同：192/128/64 对应 (300,200,110)/(303,110,36)/(66,30,17)，在第 2、6、15 层前剪枝，层平均 188.2/123.9/61.7，比名义预算少 2.0%/3.2%/3.6%（2026-10-29 完整阅读补注：已被 EXP-0002 实测取代，实测层平均 206.5/145.4/82.25，比名义预算多 7.6%/13.6%/28.5%，见本报告开头的补注）"
    evidence: [score.py@734d9003 第 7～14 行, R2（main a9e71427，两个平台一致）]
    status: DE
  M: []
  E:
  - id: E-SQZ1-3
    text: "EXP-0001 的 6 个代码级结论在 Windows（Python 3.10.22）上复现：R2～R4、R6 的数值与 Linux 完全相同；R1 四种组合的成败与报错类型相同；R5 的相对误差同为 float32 舍入量级"
    location: code/output/EXP-0001/rerun-windows-2026-10-23/（SHA256SUMS）
  - id: E-SQZ1-1
    text: "R1 sdpa 无补丁时 IndexError 的越界下标：Linux 515，Windows 367（同为 size 347）"
    location: code/output/EXP-0001/rerun-windows-2026-10-23/R1_probe_fastv.log
  - id: E-SQZ1-2
    text: "R5 相对 L2 差：Linux 预填充 3.23e-07、解码 2.28～2.77e-07；Windows 预填充 3.13e-07、解码 2.60～3.08e-07"
    location: R5_check_fastv_equivalence.log；EXP-0001_record.md 第 93 行
  evidence_locations: [code/output/EXP-0001/rerun-windows-2026-10-23/SHA256SUMS]
```

## 证据审计

- **输入**：EXP-0001 记录中抄录的 6 个运行的全文输出（云端 Linux，Python 3.11，torch 2.1.2 CPU，transformers 4.37.2）。云端原始日志没有保存。
- **本次补充的复验**（看到记录之后、作为稳健性检查进行）：在实验室电脑（Windows 11、Python 3.10.22、torch 2.1.2+cu121 但在 CPU 上运行、transformers 4.37.2、accelerate 0.21.0、timm 0.6.13、tokenizers 0.15.1）上用同样的脚本重跑：
  - R1～R4：本次在 `code/token_pruning/probes/` 重跑；R2 用 `git archive` 导出的 SparseVLMs main@a9e71427（与 EXP-0001 相同版本），另用 v1.5@734d9003 试跑一次（R2b）。
  - R5、R6：取自交接第 3 步的自检日志。
  - 全部日志与 SHA256SUMS 在 `code/output/EXP-0001/rerun-windows-2026-10-23/`，并备份。
- **缺失与异常**：R2b 在 v1.5 上报 TypeError（`attn_postprocess_topk()` 需要 `retained_tokens` 参数），所以 v1.5 的 batch=2 行为未测；不是 v1.5 的运行失败，而是探针脚本按 main 分支写的。
- **预先计划与事后提出**：复验是看到记录后为检验可复现性而做；预算表在 v1.5 上的核对是为解释 EXP-0002 的口径而提出。

## 深度分析

| 角度 | 事实 / E ID | 解释 | 替代解释 | 证据限制 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 主结果和不确定性 | 6 个运行全部复现（E-SQZ1-3）：R2～R4 的数字、R6 每层 token 数与 Linux 完全相同；R5 的 KV 长度相同，相对误差 2.60～3.13e-07 与 Linux 的 2.28～3.23e-07 同一量级（E-SQZ1-2） | 代码级结论不依赖平台；R5 的误差是 float32 的累积舍入（单精度 eps 约 1.2e-07） | R5 的平台差异来自不同 BLAS 的求和顺序 | 两个平台用的 torch 版本相同，不能说明跨 torch 版本稳健 | 无需追加 |
| 训练 / 运行动态 | 不适用：没有训练或长时间运行 | — | — | — | — |
| 错误和失败模式 | FastV 官方 KV 版在 4 种组合中 3 种报错，只在 sdpa + rotary 补丁下运行，且第 K−1 层 KV 写入两次（1270 vs 635）；R1 的越界下标在两个平台不同（E-SQZ1-1） | 报错类型由代码路径决定；越界下标取决于随机注意力的 topk 顺序，数值实现不同就会变 | — | 随机权重 | 本项目用 fastv_llava.py，已与官方等价（R5） |
| 子组、切片和边界 | SparseVLM batch=2 时第二个样本保留 0 个视觉 token（main 上复现）；v1.5 未测 | batch 1 是 SparseVLM 的硬约束（EXP-0002 也按 batch 1 运行） | v1.5 可能修复了此问题 | v1.5 未测 | 改写探针以兼容 v1.5 的签名 |
| 机制、消融和交互 | 同一“预算”下每层分布差别很大（F4 更新，R4/R6）：VisionZip 从第 0 层起每层 64/128/192；FastV（K=2，keep 30/98/166）前 2 层 576、之后平坦，层平均 64.1/127.9/191.6；SparseVLM 前 2 层 576、最后 17 层只有 17/36/110，层平均 61.7/123.9/188.2。理论 FLOPs 在“64”档相差不到 1%（18.2%～18.7%） | 理论 FLOPs 相同不代表模型看到的信息相同；SparseVLM 在“同预算”下平均少看 2～4% 的 token（2026-10-29 完整阅读补注：实测相反，SparseVLM 比名义预算多看 7.6%～28.5%，前 3 层（不是 2 层）为 576，见本报告开头的补注） | SparseVLM 的回收 token 可能补回一部分（P-SQZ1-2） | 回收 token 的实际数量未测 | EXP-0002 中用 hook 测每层实际数 |
| 稳健性及跨数据集 | 跨平台复现见第 1 行（E-SQZ1-3）；预算表在 v1.5 与 main v1.0 一致（代码核对） | — | — | 只有两个平台、同一 torch 版本 | — |
| 局部收益、整体收益和代价 | R4：同一层平均预算下理论 FLOPs 与 KV 大小几乎相同；FastV keep 288（层平均 306）约为 vanilla 的 57～59% | 速度差异要看实测延迟（EXP-0002 记录每题秒数与峰值显存） | — | 理论值 | EXP-0002 的实测对照 |

## 学到什么

- **假设**：EXP-0001 没有预设方向；本次复验支持“代码级结论可复现”。
- **已排除与仍无法区分的解释**：排除了“EXP-0001 的结果是云端环境特有的”这一解释；仍无法区分 SparseVLM 名义预算与实际每层数之间的差距是否被回收 token 抵消。
- **保留 / 复验 / 回退 / 停止**：保留 EXP-0001 的全部结论。比较三种方法时，必须同时写明层平均预算和每层分布，并注明 SparseVLM 的名义预算比层平均高 2～4%。（2026-10-29 完整阅读补注：这条规则的数字已被 EXP-0002 取代，方向相反：SparseVLM 实际处理的视觉 token 比名义预算多 7.6%/13.6%/28.5%（EXP-0002-R47～R49 实测，层平均 206.5/145.4/82.25）。比较时应同时报告名义预算和实测层平均，见 SQZ-EXP-0002 的“保留 / 复验 / 回退 / 停止”。）
- **最小、最能区分的下一次实验**：在 EXP-0002 的环境里，对 SparseVLM 192/128/64 各跑约 50 道 POPE 题，用前向 hook 记录每层输入的视觉 token 数，与预算表比较。
- **回流 Deep Read**：P-SQZ1-1、P-SQZ1-2、F4 的更新与 E-SQZ1-3 交给 MODE_C 合并（见交接包）。

## 同步清单

- [ ] experiment-results.md：登记 6 个复验运行（EXP-0001-R7～R12）与本报告。
- [ ] 已发现的实验现象.md：F4 补入 v1.5 预算表与层平均差额，注明本报告。
- [x] 论文素材库.md：无新增（MAT-0001 已覆盖每层 token 分布图）。
- [ ] 最新实验分析报告.md：指向本报告。
- [x] version-evolution.md：无方向变化，不需要新条目。
- [ ] index.md：登记本报告。

（2026-10-29 完整阅读补注：上面未勾选的四项实际都已完成，只是没有勾选：experiment-results 的 EXP-0001 总表行（链接 SQZ-EXP-0001，含 Windows 复验 R7～R12）和运行表的 EXP-0001-R7～R12 行（含 R8b）；已发现的实验现象 F4（写入 v1.5 预算表与 SQZ-EXP-0001）；最新实验分析报告“之前的实验”一节的 EXP-0001 条；index 中的 EXP-0001_squeeze 登记行。）
