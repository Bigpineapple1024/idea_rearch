# EXPERIMENT_RECORD：iTransformer 在 ETT 上的 GPU 复现（主 baseline，DEC-0012）

状态：计划 v1 已写定（2026-10-23，第一次运行之前；2026-10-24 与运行脚本一起提交，仍在第一次运行之前），执行中。
实验 ID：EXP-0004 / IDEA ID：none（baseline 复现）/ 日期：2026-10-23 起 / 执行者：Claude Code（实验室电脑，分支 code/ts-forecasting）。

## 计划（v1，2026-10-23，运行前写定）

- **要回答的问题**：官方 iTransformer（thuml/iTransformer@c2426e68）在实验室电脑上能否复现论文 Table 10 的 ETT 结果（回看 96）。
- **复现标准**（研究流程 ②）：每个“数据集 × 预测长度”，论文报告的 MSE 落在本项目 3 个种子（2021/2022/2023）的“均值 ± 2 倍标准差”之内；MAE 同样报告，作为次要判据。
- **对照值**（PAPER-0007 E-IT-3，PDF p.23 Table 10，MSE / MAE）：
  - ETTh1：96 0.386/0.405，192 0.441/0.436，336 0.487/0.458，720 0.503/0.491；
  - ETTh2：0.297/0.349，0.380/0.400，0.428/0.432，0.427/0.445；
  - ETTm1：0.334/0.368，0.377/0.391，0.426/0.420，0.491/0.459；
  - ETTm2：0.180/0.264，0.250/0.309，0.311/0.348，0.412/0.407。
- **包装脚本核验**（交接文档第 5 步）：
  - run.py 把种子写死为 2023，多种子通过 `code/ts_forecasting/itransformer_seed.py` 运行；它在内存中只替换 `fix_seed = 2023` 这一行，种子为 2023 时执行的代码与 run.py 字节相同。
  - 每个预测长度：直接运行官方 run.py 两次（种子 2023），再用包装脚本跑种子 2023 一次。
  - 判据：包装脚本的结果与两次直接运行之间的差，不超过两次直接运行彼此的差（GPU 上没有设置确定性，两次直接运行本身可能不同）。
- **阶段与顺序**：
  - 阶段 A：ETTh1 × 96/192/336/720，每档 5 次（直接 ×2、包装 2023/2021/2022），共 20 次。
  - 阶段 B（阶段 A 中至少 3 个预测长度达标后执行，交接文档“通过后扩展”）：ETTh2、ETTm1、ETTm2 × 4 个预测长度 × 种子 2021/2022/2023（包装脚本），共 36 次。
- **配置**：各数据集的官方脚本参数（PAPER-0007 E-IT-C07、C08），加 `--num_workers 0`（Windows）；数据为 ETDataset@1d16c8f4 的 CSV（sha256 记录在每次运行的 config.json）；环境 `itrans`（torch 2.0.0+cu118）。
- **指标**：标准化尺度上的 MSE（主）、MAE，全部测试窗口（batch 1，不丢窗口）。每次运行保存 pred.npy、true.npy，供逐窗口分析（SQZ-EXP-0003 的 P-SQZ3-1）。
- **预算与停止/回退**：
  - ETTh1 只有 1～2 个预测长度达标：按“环境 → 数据 → 配置 → 实现”排查，新的运行用新的 run ID。
  - 复现上限一周；仍不达标则按 DEC-0012 换备选 CycleNet，并记入 version-evolution。
- **留存**：每次运行在 `code/output/EXP-0004/<RUN>/`（run.py 的 checkpoints/、results/、test_results/ 也在其中），按研究流程保存全部文件和 SHA256SUMS，并复制到 `D:\6CJNM007\backup\EXP-0004\<RUN>\`。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0004-R01 | itransformer ETTh1-96，官方 run.py 直接运行，种子 2023 | 10-24 05:01 / 05:03 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.3866，MAE 0.4046 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R01/`，SHA256SUMS 的 sha256 1f2d7af5…；备份同名目录 |  |
| EXP-0004-R02 | itransformer ETTh1-96，官方 run.py 直接运行，种子 2023 | 10-24 05:04 / 05:06 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.3866，MAE 0.4046 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R02/`，SHA256SUMS 的 sha256 c9f9059d…；备份同名目录 |  |
| EXP-0004-R03 | itransformer ETTh1-96，包装脚本，种子 2023 | 10-24 05:06 / 05:08 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.3866，MAE 0.4046 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R03/`，SHA256SUMS 的 sha256 87ef8159…；备份同名目录 |  |
| EXP-0004-R04 | itransformer ETTh1-96，包装脚本，种子 2021 | 10-24 05:08 / 05:10 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.3855，MAE 0.4045 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R04/`，SHA256SUMS 的 sha256 3b955470…；备份同名目录 |  |
| EXP-0004-R05 | itransformer ETTh1-96，包装脚本，种子 2022 | 10-24 05:10 / 05:13 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.3852，MAE 0.4031 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R05/`，SHA256SUMS 的 sha256 cb82c8e2…；备份同名目录 |  |
| EXP-0004-R06 | itransformer ETTh1-192，官方 run.py 直接运行，种子 2023 | 10-24 05:13 / 05:16 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4413，MAE 0.4362 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R06/`，SHA256SUMS 的 sha256 ee58292d…；备份同名目录 |  |
| EXP-0004-R07 | itransformer ETTh1-192，官方 run.py 直接运行，种子 2023 | 10-24 05:16 / 05:18 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4413，MAE 0.4362 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R07/`，SHA256SUMS 的 sha256 2376e723…；备份同名目录 |  |
| EXP-0004-R08 | itransformer ETTh1-192，包装脚本，种子 2023 | 10-24 05:18 / 05:21 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4413，MAE 0.4362 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R08/`，SHA256SUMS 的 sha256 02d3d88d…；备份同名目录 |  |
| EXP-0004-R09 | itransformer ETTh1-192，包装脚本，种子 2021 | 10-24 05:21 / 05:23 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4401，MAE 0.4359 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R09/`，SHA256SUMS 的 sha256 c67e879d…；备份同名目录 |  |
| EXP-0004-R10 | itransformer ETTh1-192，包装脚本，种子 2022 | 10-24 05:23 / 05:25 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4463，MAE 0.4380 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R10/`，SHA256SUMS 的 sha256 e8235370…；备份同名目录 |  |
| EXP-0004-R11 | itransformer ETTh1-336，官方 run.py 直接运行，种子 2023 | 10-24 05:25 / 05:26 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4910，MAE 0.4622 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R11/`，SHA256SUMS 的 sha256 3fcac1b7…；备份同名目录 |  |
| EXP-0004-R12 | itransformer ETTh1-336，官方 run.py 直接运行，种子 2023 | 10-24 05:26 / 05:28 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4910，MAE 0.4622 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R12/`，SHA256SUMS 的 sha256 e5677447…；备份同名目录 |  |
| EXP-0004-R13 | itransformer ETTh1-336，包装脚本，种子 2023 | 10-24 05:28 / 05:29 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4910，MAE 0.4622 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R13/`，SHA256SUMS 的 sha256 dd7532ae…；备份同名目录 |  |
| EXP-0004-R14 | itransformer ETTh1-336，包装脚本，种子 2021 | 10-24 05:29 / 05:30 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4910，MAE 0.4606 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R14/`，SHA256SUMS 的 sha256 d326399f…；备份同名目录 |  |
| EXP-0004-R15 | itransformer ETTh1-336，包装脚本，种子 2022 | 10-24 05:30 / 05:32 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.4885，MAE 0.4594 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R15/`，SHA256SUMS 的 sha256 98bfa288…；备份同名目录 |  |
| EXP-0004-R16 | itransformer ETTh1-720，官方 run.py 直接运行，种子 2023 | 10-24 05:32 / 05:33 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.5093，MAE 0.4938 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R16/`，SHA256SUMS 的 sha256 f3670d7b…；备份同名目录 |  |
| EXP-0004-R17 | itransformer ETTh1-720，官方 run.py 直接运行，种子 2023 | 10-24 05:33 / 05:34 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.5093，MAE 0.4938 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R17/`，SHA256SUMS 的 sha256 800d5259…；备份同名目录 |  |
| EXP-0004-R18 | itransformer ETTh1-720，包装脚本，种子 2023 | 10-24 05:35 / 05:36 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.5093，MAE 0.4938 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R18/`，SHA256SUMS 的 sha256 e9b2161b…；备份同名目录 |  |
| EXP-0004-R19 | itransformer ETTh1-720，包装脚本，种子 2021 | 10-24 05:36 / 05:38 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.5174，MAE 0.4979 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R19/`，SHA256SUMS 的 sha256 b5adec30…；备份同名目录 |  |
| EXP-0004-R20 | itransformer ETTh1-720，包装脚本，种子 2022 | 10-24 05:38 / 05:39 | completed（退出码 0） | commit b140cf9，无未提交改动；`itrans` 环境；MSE 0.5092，MAE 0.4925 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R20/`，SHA256SUMS 的 sha256 556d0145…；备份同名目录 |  |

## 阶段 A 结果（ETTh1，2026-10-24）

由 `code/ts_forecasting/analyze_exp0004.py` 从 runs.tsv 生成（`code/output/EXP-0004/analysis/tables.md`）。均值与标准差取包装脚本的三个种子（样本标准差）。

| 数据集-预测长度 | 直接运行 run.py（种子 2023，两次） | 包装脚本 2023 与直接运行是否相同 | 包装脚本各种子 MSE（2021 / 2022 / 2023） | 均值 ± 标准差 | 论文 MSE | 是否在 ±2 SD 内（计划的标准） | 计入论文三位小数的舍入 | MAE 均值 ± 标准差 / 论文 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1-96 | 0.3866 / 0.3866 | 是（逐位相同） | 0.3855 / 0.3852 / 0.3866 | 0.3858 ± 0.0008 | 0.386 | 是 | 是 | 0.4041 ± 0.0008 / 0.405 |
| ETTh1-192 | 0.4413 / 0.4413 | 是 | 0.4401 / 0.4463 / 0.4413 | 0.4426 ± 0.0033 | 0.441 | 是 | 是 | 0.4367 ± 0.0011 / 0.436 |
| ETTh1-336 | 0.4910 / 0.4910 | 是 | 0.4910 / 0.4885 / 0.4910 | 0.4902 ± 0.0015 | 0.487 | **否**（下限 0.4872） | 是 | 0.4607 ± 0.0014 / 0.458 |
| ETTh1-720 | 0.5093 / 0.5093 | 是 | 0.5174 / 0.5092 / 0.5093 | 0.5120 ± 0.0047 | 0.503 | 是（下限 0.5025） | 是 | 0.4947 ± 0.0028 / 0.491 |

- **包装脚本核验通过**：四个预测长度上，包装脚本（种子 2023）与两次直接运行的 MSE、MAE 逐位相同；两次直接运行彼此也逐位相同，说明这台 GPU 上的训练是确定的。
- **复现判定**：按计划 v1 的标准，96、192、720 三档达标，336 档未达标（论文值 0.487 比 ±2 SD 的下限 0.4872 低 0.0002；若把论文值看作舍入到三位小数的区间 [0.4865, 0.4875)，则与 ±2 SD 重叠）。计划写的是“至少 3 个预测长度达标后执行阶段 B”，因此进入阶段 B。336 档按计划的字面标准记为未达标，不改写为通过；舍入的判断是看到结果后补充的，只作说明。
- **观察（待 squeeze）**：
  - 本项目的均值都高于论文值，差距随预测长度增大：+0.0002、+0.0016、+0.0032、+0.0090。
  - 种子间标准差也随预测长度增大：0.0008、0.0033、0.0015、0.0047。
  - MAE 的情况相同：336、720 档高于论文 0.0027、0.0037。
- **偏差**：无（阶段 A 按计划执行；分析脚本在阶段 A 结束后加入）。
