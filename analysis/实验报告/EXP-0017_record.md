# EXPERIMENT_RECORD：192 档复验（第 ⑫ 步第四轮）

状态：completed（2026-10-25）。计划 v1 在运行之前写定；4 次运行完成，核对全部通过。K1 成立（POPE +2.29），K2 不成立（MME −20.6，区间含 0）。阶段结论见 DEC-0029。深挖（squeeze）原来没有做，2026-10-25 跑偏检查时在记录内补做，见文末（2026-10-25 跑偏检查补注）。

- 实验 ID：EXP-0017。
- IDEA ID：IDEA-0002；DEC-0027、DEC-0028。
- 执行者：Claude Code（实验室电脑，分支 code/token-pruning）。

## 计划（v1，运行前写定）

### 问题

A′+B+V（IDF × 可视性，名额按可见占比分配）在 64 档上 POPE 显著高于 A、MME 点估计为正；128 档见 EXP-0016。到 192 档（保留 166 个，剪得更少），这种差别是否仍在？

### 运行

| 运行 | 配置 × 数据 | 组别 |
| --- | --- | --- |
| R01 | `fastv192_prior_strat_nopad_kl` × MME | A（192） |
| R02 | `fastv192_qvis_kl` × MME | A′+B+V（192） |
| R03 | `fastv192_prior_strat_nopad_kl` × POPE | A（192） |
| R04 | `fastv192_qvis_kl` × POPE | A′+B+V（192） |

可视性表与 EXP-0015 相同；名额 round(166 × S)。

### 判据

| 编号 | 判据 |
| --- | --- |
| K1 | 192 档 POPE：A′+B+V − A 的配对区间在 0 以上（原始标注与 RePOPE） |
| K2 | 192 档 MME-P：A′+B+V − A 的配对区间在 0 以上；点估计 > 0、区间含 0 时记为不变差 |

- 预测：预算越大，与题目无关的选择越容易保住物体，差别缩小（F25：随机选择保住物体的概率随保留数上升）。
- 同时报告 192 档的 VisionZip、SparseVLM、FastV（EXP-0002）。

### 实现核对

- R02、R04：按查询选的个数等于离线重算的 round(166 × S)（至少 99%）；主指标重算一致。

### 预算

- 4 次运行，约 1.0 GPU 小时；候选 1 累计 34 次（上限按 DEC-0028 调整为 60 次或 40 GPU 小时；2026-10-30 完整阅读补注：调整写在 EXP-0016 出结果之前，复盘在后，见审计 AU-18）。

### 代码

- `exp0002.py`：新增 `fastv192_qvis`；不改其他代码。

## 结果（R01～R04，2026-10-25）

分析 `analyze_0017.py`，输出 `code/output/EXP-0017/analysis/0017.{md,json}`。核对全部通过（按查询选的个数等于 round(166 × S)；主指标重算一致）。

| 配置（192 档） | POPE F1 原始 / RePOPE | MME-P |
| --- | --- | --- |
| A（强对照 + 排除填充区，R03、R01） | 83.12 / 88.08 | 1493.3 |
| **A′+B+V（R04、R02）** | **85.41 / 89.55** | **1472.6** |
| VisionZip-192 / SparseVLM-192 / FastV-192 | 85.49 / 85.29 / 75.76 | 1442.7 / 1448.0 / 1464.4 |
| vanilla | 85.90 / 90.64 | 1507.6 |

| 配对差 | POPE（原始 / RePOPE） | MME-P |
| --- | --- | --- |
| A′+B+V − A（K1、K2） | +2.29 [1.75, 2.88] / +1.47 [0.90, 2.09]（审计修正，DEC-0030：按图像重抽 [1.38, 3.22] / [0.53, 2.40]；表中为按题重抽的旧口径） | −20.6 [−55.8, +13.2] |
| A′+B+V − VisionZip-192 | −0.08 [−0.62, +0.47] / +0.44 | +29.9 [−14.3, +73.5] |
| A′+B+V − SparseVLM-192 | +0.12 [−0.42, +0.71] / +0.19 | +24.6 [−17.4, +68.2] |
| A′+B+V − vanilla | −0.49 [−0.97, −0.04] / −1.09（审计修正，DEC-0030：按图像重抽 [−1.20, +0.22]，原始标注下不再显著；RePOPE [−1.79, −0.38]） | −35.0 [−73.7, −1.5] |

- **判读**：K1 成立；K2 不成立。审计修正：按预设规则（点估计 > 0 才记为不变差）不能记为不变差，应为差异不显著，区间大部分在 0 以下。
- 全部运行：R01 `fastv192_prior_strat_nopad_kl` × MME（1493.3）、R02 `fastv192_qvis_kl` × MME（1472.6）、R03 `fastv192_prior_strat_nopad_kl` × POPE（83.12）、R04 `fastv192_qvis_kl` × POPE（85.41）；全部 completed，commit a560902，无未提交改动；校验和见 runs.tsv，备份 `D:\6CJNM007\backup\EXP-0017\`。
- 预算：4 次；候选 1 累计 34 次（上限 60，DEC-0028）。

## 深挖（squeeze，记录内；2026-10-25 跑偏检查时补做）

本实验记录时没有做深挖，2026-10-25 跑偏检查时补做。探索性，只读已封存的 metrics.json 与 kept_tokens.json，不调用模型。脚本 `squeeze_0017.py` 在 EXP-0021 占用 GPU 时从 `D:/6CJNM007/tmp/exp0017sq` 用 CPU 运行，之后提交到 code/token_pruning（commit 1d95df0，与封存在输出旁的副本相同）；输出 `code/output/EXP-0017/squeeze/squeeze_0017.{md,json}`（SHA256SUMS 的 sha256 9d283751…）。

- 对比的运行：POPE 64 档 EXP-0011-R07 / EXP-0015-R03，128 档 EXP-0016-R02 / R03，192 档本实验 R03 / R04；MME 64 档 EXP-0014-R04 / EXP-0015-R02，128 档 EXP-0016-R04 / R05，192 档本实验 R01 / R02；vanilla 为 EXP-0002-R04（POPE）、EXP-0002-R02（MME）。
- POPE 的数字是 0～1 的比例（原始标注）。子集与子任务上的差都没有区间，只作描述。

| 维度 | 发现 | 证据 | 新信息 |
| --- | --- | --- | --- |
| 预算 | POPE 上的提升随预算缩小，主要因为 A 的召回追了上来，A′+B+V 的召回变化不大。64 / 128 / 192 档的召回：A 0.511 / 0.680 / 0.740，A′+B+V 0.758 / 0.789 / 0.797（vanilla 0.791）；漏报：A 733 / 480 / 390，A′+B+V 363 / 316 / 305（vanilla 313）。三个子集共用同一批正样本，所以召回在三个子集上相同 | squeeze_0017.md 第 1 节；审计记录 AU-1 | 支持 F37 的解释（F25 的延伸） |
| 误报 | A′+B+V 的误报在三档上变化不大，都多于 vanilla：adversarial / popular / random 在 64 档 191 / 108 / 39，128 档 187 / 111 / 40，192 档 175 / 101 / 37，vanilla 132 / 69 / 31。A 的误报随预算增加（53 / 13 / 12、82 / 38 / 19、108 / 55 / 21），但都少于 vanilla。每一档、每个子集上，A′+B+V 的精确率都低于 A | squeeze_0017.md 第 1 节 | 更新 F30、F41：误报偏多在 128、192 档仍在 |
| 答案偏向 | 128、192 档上，A′+B+V 的 yes 比例在三个子集上都高于 vanilla（192 档 0.457 / 0.432 / 0.423，vanilla 0.440 / 0.419 / 0.419）；192 档的召回也略高于 vanilla（0.797 对 0.791）。64 档的 yes 比例（0.443 / 0.415 / 0.404）与 vanilla 接近 | squeeze_0017.md 第 1 节 | 新（描述，与 F41 同向） |
| 与不剪枝的差 | 192 档上 A′+B+V 的 F1 低于 vanilla 的是 adversarial（0.833 对 0.842）与 popular（0.855 对 0.861）；random 上 0.875 对 0.873。漏报比 vanilla 少（305 对 313），所以差距来自误报 | squeeze_0017.md 第 1 节 | 新（描述） |
| 名额 | 按题目查询选的名额占保留数的比例三档几乎相同：0.75、0.74、0.74（平均 n_q 22.4、72.9、123.3）。提升缩小不是因为查询的份额变小 | squeeze_0017.md 第 2 节 | 新（描述） |
| MME 子任务 | A′+B+V − A 在三档上都为正的只有 position（+10.0 / +10.0 / +5.0）与 color（+25.0 / +15.0 / +5.0）。192 档的总差主要来自 count −10.0 与 OCR −10.0。OCR（+27.5 / −5.0 / −10.0）与 existence（+10.0 / −10.0 / +0.0）在三档间符号不一致。四个名字类子任务（celebrity、landmark、posters、artwork）在 192 档只差 −4.1～−0.8；celebrity 上两种剪枝都低于 vanilla（A 126.5、A′+B+V 122.4、vanilla 136.2） | squeeze_0017.md 第 3 节 | 更新 F36：64 档的 OCR +27.5 在 128、192 档没有复现 |

```text
package_type: SQUEEZE_REPORT
squeeze_id: SQZ-EXP-0017（记录内）
input_experiment_record_ids: [EXP-0017, EXP-0016, EXP-0015, EXP-0014, EXP-0011, EXP-0002]
claim_scope: LLaVA-1.5-7B；层平均 64、128、192；POPE（原始标注）与 MME-P；探索性，只读已封存的 metrics.json 与 kept_tokens.json
primary_result_with_uncertainty: 192 档 A′+B+V − A：POPE +2.29（按图像重抽 [1.38, 3.22]，DEC-0030），MME-P −20.6 [−55.8, +13.2]（差异不显著）；A′+B+V − vanilla 的 POPE −0.49（按图像重抽 [−1.20, +0.22]，DEC-0030）；子集与子任务上的差没有区间
new_P: none
new_F: none（支持 F37 的解释；更新 F30、F41：误报偏多在 128、192 档仍在；更新 F36：OCR 的 +27.5 在 128、192 档没有复现）
new_M: none
evidence_index: code/output/EXP-0017/squeeze（SHA256SUMS 的 sha256 9d283751…）
next_routes: [F30、F36、F37、F41 补注本节的 128、192 档数字；写作时引用]
next_skill: deep-read Mode C
```
