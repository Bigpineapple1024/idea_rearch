# EXPERIMENT_RECORD：iTransformer 在 ETT 上的 GPU 复现（主 baseline，DEC-0012）

状态：completed（2026-10-24）。计划 v1 写定于 2026-10-23，在第一次运行之前；2026-10-24 与运行脚本一起提交，仍在第一次运行之前。56 次运行全部完成。按计划的字面标准，16 个“数据集 × 预测长度”中 6 个达标；与论文 MSE 的差都在 0.0081 以内。用户决定记为偏差，继续以 iTransformer 为主 baseline（DEC-0018），见“阶段 B 结果与复现判定”。
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

## 阶段 B 的运行（2026-10-24，批处理 run_ts_batch2.sh）

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0004-R21 | itransformer ETTh2-96，包装脚本，种子 2021 | 10-24 09:50 / 09:52 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.2996，MAE 0.3498 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R21/`，SHA256SUMS 的 sha256 0b51705c…；备份同名目录 | |
| EXP-0004-R22 | itransformer ETTh2-96，包装脚本，种子 2022 | 10-24 09:52 / 09:54 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3001，MAE 0.3506 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R22/`，SHA256SUMS 的 sha256 53b09b90…；备份同名目录 | |
| EXP-0004-R23 | itransformer ETTh2-96，包装脚本，种子 2023 | 10-24 09:54 / 09:57 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3009，MAE 0.3499 | 2785 个测试窗口 | `code/output/EXP-0004/EXP-0004-R23/`，SHA256SUMS 的 sha256 9e2cb786…；备份同名目录 | |
| EXP-0004-R24 | itransformer ETTh2-192，包装脚本，种子 2021 | 10-24 09:57 / 09:59 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3815，MAE 0.3995 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R24/`，SHA256SUMS 的 sha256 25c53954…；备份同名目录 | |
| EXP-0004-R25 | itransformer ETTh2-192，包装脚本，种子 2022 | 10-24 09:59 / 10:01 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3796，MAE 0.3981 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R25/`，SHA256SUMS 的 sha256 30ed4ff4…；备份同名目录 | |
| EXP-0004-R26 | itransformer ETTh2-192，包装脚本，种子 2023 | 10-24 10:01 / 10:02 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3802，MAE 0.3988 | 2689 个测试窗口 | `code/output/EXP-0004/EXP-0004-R26/`，SHA256SUMS 的 sha256 41a149e1…；备份同名目录 | |
| EXP-0004-R27 | itransformer ETTh2-336，包装脚本，种子 2021 | 10-24 10:02 / 10:04 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4237，MAE 0.4326 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R27/`，SHA256SUMS 的 sha256 113c8976…；备份同名目录 | |
| EXP-0004-R28 | itransformer ETTh2-336，包装脚本，种子 2022 | 10-24 10:04 / 10:06 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4241，MAE 0.4344 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R28/`，SHA256SUMS 的 sha256 30aa4e50…；备份同名目录 | |
| EXP-0004-R29 | itransformer ETTh2-336，包装脚本，种子 2023 | 10-24 10:06 / 10:08 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4235，MAE 0.4322 | 2545 个测试窗口 | `code/output/EXP-0004/EXP-0004-R29/`，SHA256SUMS 的 sha256 26e6f7c3…；备份同名目录 | |
| EXP-0004-R30 | itransformer ETTh2-720，包装脚本，种子 2021 | 10-24 10:08 / 10:09 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4305，MAE 0.4461 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R30/`，SHA256SUMS 的 sha256 de594080…；备份同名目录 | |
| EXP-0004-R31 | itransformer ETTh2-720，包装脚本，种子 2022 | 10-24 10:09 / 10:11 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4288，MAE 0.4468 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R31/`，SHA256SUMS 的 sha256 127acb01…；备份同名目录 | |
| EXP-0004-R32 | itransformer ETTh2-720，包装脚本，种子 2023 | 10-24 10:11 / 10:12 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4304，MAE 0.4470 | 2161 个测试窗口 | `code/output/EXP-0004/EXP-0004-R32/`，SHA256SUMS 的 sha256 7c12acbd…；备份同名目录 | |
| EXP-0004-R33 | itransformer ETTm1-96，包装脚本，种子 2021 | 10-24 10:12 / 10:19 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3439，MAE 0.3782 | 11425 个测试窗口 | `code/output/EXP-0004/EXP-0004-R33/`，SHA256SUMS 的 sha256 3f8bbde6…；备份同名目录 | |
| EXP-0004-R34 | itransformer ETTm1-96，包装脚本，种子 2022 | 10-24 10:19 / 10:29 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3406，MAE 0.3758 | 11425 个测试窗口 | `code/output/EXP-0004/EXP-0004-R34/`，SHA256SUMS 的 sha256 c5a2f4de…；备份同名目录 | |
| EXP-0004-R35 | itransformer ETTm1-96，包装脚本，种子 2023 | 10-24 10:30 / 10:40 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3419，MAE 0.3767 | 11425 个测试窗口 | `code/output/EXP-0004/EXP-0004-R35/`，SHA256SUMS 的 sha256 4f63fc26…；备份同名目录 | |
| EXP-0004-R36 | itransformer ETTm1-192，包装脚本，种子 2021 | 10-24 10:40 / 10:48 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3810，MAE 0.3952 | 11329 个测试窗口 | `code/output/EXP-0004/EXP-0004-R36/`，SHA256SUMS 的 sha256 857af4d7…；备份同名目录 | |
| EXP-0004-R37 | itransformer ETTm1-192，包装脚本，种子 2022 | 10-24 10:48 / 10:55 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3844，MAE 0.3963 | 11329 个测试窗口 | `code/output/EXP-0004/EXP-0004-R37/`，SHA256SUMS 的 sha256 28d14d0b…；备份同名目录 | |
| EXP-0004-R38 | itransformer ETTm1-192，包装脚本，种子 2023 | 10-24 10:55 / 11:03 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3829，MAE 0.3957 | 11329 个测试窗口 | `code/output/EXP-0004/EXP-0004-R38/`，SHA256SUMS 的 sha256 86fcec63…；备份同名目录 | |
| EXP-0004-R39 | itransformer ETTm1-336，包装脚本，种子 2021 | 10-24 11:03 / 11:13 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4194，MAE 0.4192 | 11185 个测试窗口 | `code/output/EXP-0004/EXP-0004-R39/`，SHA256SUMS 的 sha256 2259f92d…；备份同名目录 | |
| EXP-0004-R40 | itransformer ETTm1-336，包装脚本，种子 2022 | 10-24 11:13 / 11:21 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4164，MAE 0.4168 | 11185 个测试窗口 | `code/output/EXP-0004/EXP-0004-R40/`，SHA256SUMS 的 sha256 4dff2f09…；备份同名目录 | |
| EXP-0004-R41 | itransformer ETTm1-336，包装脚本，种子 2023 | 10-24 11:21 / 11:31 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4182，MAE 0.4182 | 11185 个测试窗口 | `code/output/EXP-0004/EXP-0004-R41/`，SHA256SUMS 的 sha256 106382c6…；备份同名目录 | |
| EXP-0004-R42 | itransformer ETTm1-720，包装脚本，种子 2021 | 10-24 11:31 / 11:40 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4864，MAE 0.4557 | 10801 个测试窗口 | `code/output/EXP-0004/EXP-0004-R42/`，SHA256SUMS 的 sha256 f735aa25…；备份同名目录 | |
| EXP-0004-R43 | itransformer ETTm1-720，包装脚本，种子 2022 | 10-24 11:41 / 11:51 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4946，MAE 0.4604 | 10801 个测试窗口 | `code/output/EXP-0004/EXP-0004-R43/`，SHA256SUMS 的 sha256 8ea1c56d…；备份同名目录 | |
| EXP-0004-R44 | itransformer ETTm1-720，包装脚本，种子 2023 | 10-24 11:51 / 12:01 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4874，MAE 0.4567 | 10801 个测试窗口 | `code/output/EXP-0004/EXP-0004-R44/`，SHA256SUMS 的 sha256 bb9bc067…；备份同名目录 | |
| EXP-0004-R45 | itransformer ETTm2-96，包装脚本，种子 2021 | 10-24 12:01 / 12:08 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.1839，MAE 0.2681 | 11425 个测试窗口 | `code/output/EXP-0004/EXP-0004-R45/`，SHA256SUMS 的 sha256 00b7b9d0…；备份同名目录 | |
| EXP-0004-R46 | itransformer ETTm2-96，包装脚本，种子 2022 | 10-24 12:08 / 12:13 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.1859，MAE 0.2725 | 11425 个测试窗口 | `code/output/EXP-0004/EXP-0004-R46/`，SHA256SUMS 的 sha256 a982e71e…；备份同名目录 | |
| EXP-0004-R47 | itransformer ETTm2-96，包装脚本，种子 2023 | 10-24 12:13 / 12:19 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.1856，MAE 0.2723 | 11425 个测试窗口 | `code/output/EXP-0004/EXP-0004-R47/`，SHA256SUMS 的 sha256 40d195ae…；备份同名目录 | |
| EXP-0004-R48 | itransformer ETTm2-192，包装脚本，种子 2021 | 10-24 12:19 / 12:24 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.2530，MAE 0.3127 | 11329 个测试窗口 | `code/output/EXP-0004/EXP-0004-R48/`，SHA256SUMS 的 sha256 c8dfbfc4…；备份同名目录 | |
| EXP-0004-R49 | itransformer ETTm2-192，包装脚本，种子 2022 | 10-24 12:24 / 12:30 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.2520，MAE 0.3124 | 11329 个测试窗口 | `code/output/EXP-0004/EXP-0004-R49/`，SHA256SUMS 的 sha256 49e75408…；备份同名目录 | |
| EXP-0004-R50 | itransformer ETTm2-192，包装脚本，种子 2023 | 10-24 12:30 / 12:35 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.2539，MAE 0.3137 | 11329 个测试窗口 | `code/output/EXP-0004/EXP-0004-R50/`，SHA256SUMS 的 sha256 d10de658…；备份同名目录 | |
| EXP-0004-R51 | itransformer ETTm2-336，包装脚本，种子 2021 | 10-24 12:35 / 12:42 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3132，MAE 0.3504 | 11185 个测试窗口 | `code/output/EXP-0004/EXP-0004-R51/`，SHA256SUMS 的 sha256 e719d64c…；备份同名目录 | |
| EXP-0004-R52 | itransformer ETTm2-336，包装脚本，种子 2022 | 10-24 12:42 / 12:48 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3164，MAE 0.3535 | 11185 个测试窗口 | `code/output/EXP-0004/EXP-0004-R52/`，SHA256SUMS 的 sha256 769d7314…；备份同名目录 | |
| EXP-0004-R53 | itransformer ETTm2-336，包装脚本，种子 2023 | 10-24 12:48 / 12:54 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.3156，MAE 0.3506 | 11185 个测试窗口 | `code/output/EXP-0004/EXP-0004-R53/`，SHA256SUMS 的 sha256 15fe20a8…；备份同名目录 | |
| EXP-0004-R54 | itransformer ETTm2-720，包装脚本，种子 2021 | 10-24 12:54 / 13:00 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4120，MAE 0.4062 | 10801 个测试窗口 | `code/output/EXP-0004/EXP-0004-R54/`，SHA256SUMS 的 sha256 333f4476…；备份同名目录 | |
| EXP-0004-R55 | itransformer ETTm2-720，包装脚本，种子 2022 | 10-24 13:00 / 13:07 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4135，MAE 0.4068 | 10801 个测试窗口 | `code/output/EXP-0004/EXP-0004-R55/`，SHA256SUMS 的 sha256 edd0120a…；备份同名目录 | |
| EXP-0004-R56 | itransformer ETTm2-720，包装脚本，种子 2023 | 10-24 13:07 / 13:13 | completed（退出码 0） | commit e40bd90，无未提交改动；`itrans` 环境；MSE 0.4138，MAE 0.4068 | 10801 个测试窗口 | `code/output/EXP-0004/EXP-0004-R56/`，SHA256SUMS 的 sha256 926941d5…；备份同名目录 | |

- **配置**：36 次都在 commit e40bd90，没有未提交的改动。各数据集用官方脚本的参数（d_model = d_ff = 128，e_layers = 2，PAPER-0007 E-IT-C08），加 `--num_workers 0`。
- **耗时**：ETTh2 每次 1.5～2.5 分钟，ETTm1 7～10 分钟，ETTm2 约 5.5 分钟。
- **批处理日志**：`D:\6CJNM007\tmp\ts_batch2.log`。各阶段的退出码都是 0。

## 阶段 B 结果与复现判定

由 `analyze_exp0004.py` 生成（`code/output/EXP-0004/analysis/tables.md`）：

| 数据集-预测长度 | 包装脚本各种子 MSE（2021 / 2022 / 2023） | 均值 ± 标准差 | 论文 MSE | 均值 − 论文 | 是否在 ±2 SD 内（计划的标准） | 计入论文舍入 | MAE 均值 ± 标准差 / 论文 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2-96 | 0.2996 / 0.3001 / 0.3009 | 0.3002 ± 0.0006 | 0.297 | +0.0032 | 否 | 否 | 0.3501 ± 0.0004 / 0.349 |
| ETTh2-192 | 0.3815 / 0.3796 / 0.3802 | 0.3804 ± 0.0009 | 0.380 | +0.0004 | 是 | 是 | 0.3988 ± 0.0007 / 0.400 |
| ETTh2-336 | 0.4237 / 0.4241 / 0.4235 | 0.4238 ± 0.0003 | 0.428 | −0.0042 | 否 | 否 | 0.4330 ± 0.0012 / 0.432 |
| ETTh2-720 | 0.4305 / 0.4288 / 0.4304 | 0.4299 ± 0.0010 | 0.427 | +0.0029 | 否 | 否 | 0.4466 ± 0.0005 / 0.445 |
| ETTm1-96 | 0.3439 / 0.3406 / 0.3419 | 0.3421 ± 0.0017 | 0.334 | +0.0081 | 否 | 否 | 0.3769 ± 0.0012 / 0.368 |
| ETTm1-192 | 0.3810 / 0.3844 / 0.3829 | 0.3828 ± 0.0017 | 0.377 | +0.0058 | 否 | 否 | 0.3957 ± 0.0006 / 0.391 |
| ETTm1-336 | 0.4194 / 0.4164 / 0.4182 | 0.4180 ± 0.0015 | 0.426 | −0.0080 | 否 | 否 | 0.4181 ± 0.0012 / 0.420 |
| ETTm1-720 | 0.4864 / 0.4946 / 0.4874 | 0.4894 ± 0.0045 | 0.491 | −0.0016 | 是 | 是 | 0.4576 ± 0.0025 / 0.459 |
| ETTm2-96 | 0.1839 / 0.1859 / 0.1856 | 0.1851 ± 0.0011 | 0.180 | +0.0051 | 否 | 否 | 0.2710 ± 0.0025 / 0.264 |
| ETTm2-192 | 0.2530 / 0.2520 / 0.2539 | 0.2530 ± 0.0010 | 0.250 | +0.0030 | 否 | 否 | 0.3129 ± 0.0007 / 0.309 |
| ETTm2-336 | 0.3132 / 0.3164 / 0.3156 | 0.3151 ± 0.0016 | 0.311 | +0.0041 | 否 | 否 | 0.3515 ± 0.0017 / 0.348 |
| ETTm2-720 | 0.4120 / 0.4135 / 0.4138 | 0.4131 ± 0.0010 | 0.412 | +0.0011 | 是 | 是 | 0.4066 ± 0.0003 / 0.407 |

**判定（按计划 v1 的字面标准）**：
- 阶段 B 的 12 个组合中 3 个达标（ETTh2-192、ETTm1-720、ETTm2-720）。
- 加上阶段 A 的 3 个，16 个组合共 6 个达标。
- 计划没有给阶段 B 写单独的判读规则，只写了“复现不上先排查；一周内不行，按 DEC-0012 换备选 CycleNet”。

**用户决定**：记为偏差，继续以 iTransformer 为主 baseline（DEC-0018，2026-10-24，用户选 A）。理由：
- 论文与本项目的 MSE 差都在 0.0081 以内（相对差不超过 2.8%），而且有正有负（12 个中 3 个低于论文）。
- ±2 SD 只反映种子之间的差异（标准差 0.0003～0.0047），不包括环境差异。
- 官方脚本的超参数与论文所写的网格不一致（d_model = 128 不在 {256, 512} 内，学习率固定 1e-4，E-IT-C05、C08）。

以后的改进都与本项目自己的 iTransformer 运行比较；报告时写明“官方脚本、本项目环境、3 个种子”，并列出论文值。

**观察（squeeze 见 SQZ-EXP-0004）**：
- ETTh1 的差距随预测长度增大；阶段 B 的差距与预测长度没有一致的关系。
- MAE 与 MSE 偏离论文的方向，在 12 个组合中有 9 个相同。

**偏差**：阶段 B 的复现判定偏离研究流程第 ② 步的处理办法，由用户决定（DEC-0018）。
