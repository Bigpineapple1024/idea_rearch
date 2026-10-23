# EXPERIMENT_RECORD：DUET、TQNet、WPMixer 在统一协议下的 GPU 重跑（ETTh1）

状态：completed（2026-10-24）。计划 v1 写定于 2026-10-23（2026-10-24 提交，仍在第一次运行之前），v1.1 在第一次运行之前补入。39 次运行：成功 27 次；WPMixer 的 12 次因环境缺包失败，补装后用新 ID 重跑全部成功（见“偏差与限制”）。
实验 ID：EXP-0005 / IDEA ID：none（对照方法重评）/ 日期：2026-10-23 起 / 执行者：Claude Code（实验室电脑，分支 code/ts-forecasting）。

## 计划（v1，2026-10-23，运行前写定）

- **要回答的问题**：在与 EXP-0004 相同的统一协议下（回看 96、全部测试窗口计分、3 个种子、同一台 GPU），三个对照方法在 ETTh1 上的 MSE/MAE 是多少；EXP-0003（CPU、多数单种子）的结论是否保持。
- **统一协议**：
  - 回看 96，预测 96/192/336/720，按各仓库官方参数；
  - 测试集全部窗口计分（WPMixer 同时报告官方 drop_last 计分，便于与 EXP-0003 对照）；
  - 种子 2021/2022/2023，报告均值与标准差；
  - Windows 上统一 `--num_workers 0`。
- **组别**：
  - TQNet：官方 `scripts/TQNet/etth1.sh` 参数 × 4 档 × 3 个种子，共 12 次（测试加载本身不丢窗口）。
  - WPMixer：官方 `scripts/Unified/ETTh1_Unified_setup.sh` 的逐档参数 × 4 档 × 3 个种子，经 `code/ts_forecasting/wpmixer_droplast.py` 同时给出两种计分，共 12 次。注意其超参数是在测试集上搜索得到的（PAPER-0003 E-WP-C04）。
  - DUET：官方 `DUET_unified_seq_len_96.sh` 的 ETTh1-96 参数 × 3 个种子，经 `duet_test_stochasticity.py`，共 3 次；其他预测长度不在本实验范围（交接文档只要求补种子）。
- **数据**：
  - TQNet、WPMixer：ETDataset@1d16c8f4 的 ETTh1.csv（与 EXP-0004 相同）。
  - DUET：TFB 官方数据（DUET README 的 Google Drive，forecasting.zip，sha256 01794d00…）中的 ETTh1.csv。
  - 跨框架比较前先做数据核对：把 TFB 的 ETTh1 测试窗口与 iTransformer（EXP-0004）的测试窗口逐值比对（原始值与标准化值），记录是否一致及差异位置。
- **主指标**：MSE（全部窗口，标准化尺度）；MAE 次之。WPMixer 另报 drop_last 计分与两者之差。
- **补充分析**（来自 SQZ-EXP-0003 的 P-SQZ3-1）：保存逐窗口预测，比较测试期末尾窗口与其余窗口的误差。
- **预算与停止**：约 27 次运行，预计 2 GPU 小时以内；任一方法的运行失败时记录原因、修复后用新 run ID 重跑，不影响其他方法。
- **留存**：同 EXP-0004，`code/output/EXP-0005/<RUN>/`，备份到 `D:\6CJNM007\backup\EXP-0005\<RUN>\`。

## 计划 v1.1：运行入口与数据核对的具体做法（2026-10-24，第一次运行之前补入）

- **运行器**：`code/ts_forecasting/exp_ts.py` 新增 `tqnet`、`wpmixer`、`duet` 三个入口，参数与计划 v1 相同（官方脚本逐项抄录在 exp_ts.py 中）。
  - TQNet：经 `tqnet_save_pred.py` 运行官方 run.py。官方 test() 计算全部测试窗口的指标，但保存预测的 np.save 被注释掉；包装脚本在本进程内替换 `metric`，先保存 pred.npy、true.npy 再返回原指标，分数不变。metrics.json 同时记录官方 result.txt 中的 MSE/MAE 与由保存的数组重算的值。
  - WPMixer：照官方统一设置脚本加 `--use_amp`。它从 `./data/ETT/` 读数据并把 checkpoint 写进自己的目录，所以每次运行先把固定 commit 用 `git archive` 导出到运行目录中再运行，`source/WPMixer` 不被写入。`wpmixer_droplast.py` 新增保存全部窗口的预测（pred_all_windows.npy、true_all_windows.npy）。
  - DUET：同样从导出的副本运行，`dataset/forecasting/` 放 TFB 的 ETTh1.csv 与 FORECAST_META.csv；`duet_test_stochasticity.py` 新增 `--save-pred`，并只做 1 次 mask 重采样（EXP-0003 已测过 10 次）。种子用 run_benchmark.py 的 `--seed`。
- **TFB 数据**：用 `prepare_tfb_data.py` 从 forecasting.zip 只解压 `forecasting/ETTh1.csv`（sha256 9f245117…）与 `forecasting/FORECAST_META.csv`（deeb6377…）到 `D:\6CJNM007\data\tfb\`；每次运行的 config.json 记录所用数据文件的 sha256。
- **数据核对**（计划 v1 的“逐值比对”）：`compare_tfb_ett.py` 把 DUET 保存的 true.npy（TFB 测试窗口、训练集标准化）与 EXP-0004 中 iTransformer ETTh1-96 的 true.npy 逐值比较（窗口数、每个窗口的数值差、最大绝对差），并比较两份 ETTh1.csv 的原始值。
- **逐窗口分析**：`analyze_ts_windows.py` 对每次运行计算逐窗口 MSE、末 97/113 个窗口与其余窗口的 MSE、按十分位的 MSE，并在窗口数相同的方法之间逐窗口比较。

## 全部运行

| 运行 ID | 组别 / 种子 | 开始 / 结束 | 状态 / 退出码 | 实际配置 | 样本量 | 原始输出 / 校验和 / 备份 | 异常与偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0005-R01 | tqnet ETTh1-96，包装脚本，种子 2021 | 10-24 05:53 / 05:54 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.3730，MAE 0.3938 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R01/`，SHA256SUMS 的 sha256 98816392…；备份同名目录 |  |
| EXP-0005-R02 | tqnet ETTh1-96，包装脚本，种子 2022 | 10-24 05:54 / 05:54 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.3727，MAE 0.3931 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R02/`，SHA256SUMS 的 sha256 4360f579…；备份同名目录 |  |
| EXP-0005-R03 | tqnet ETTh1-96，包装脚本，种子 2023 | 10-24 05:54 / 05:54 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.3727，MAE 0.3941 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R03/`，SHA256SUMS 的 sha256 5958abb6…；备份同名目录 |  |
| EXP-0005-R04 | tqnet ETTh1-192，包装脚本，种子 2021 | 10-24 05:54 / 05:55 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4324，MAE 0.4265 | 2689 个测试窗口 | `code/output/EXP-0005/EXP-0005-R04/`，SHA256SUMS 的 sha256 9a916ab3…；备份同名目录 |  |
| EXP-0005-R05 | tqnet ETTh1-192，包装脚本，种子 2022 | 10-24 05:55 / 05:55 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4275，MAE 0.4227 | 2689 个测试窗口 | `code/output/EXP-0005/EXP-0005-R05/`，SHA256SUMS 的 sha256 e794c9bd…；备份同名目录 |  |
| EXP-0005-R06 | tqnet ETTh1-192，包装脚本，种子 2023 | 10-24 05:55 / 05:55 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4286，MAE 0.4260 | 2689 个测试窗口 | `code/output/EXP-0005/EXP-0005-R06/`，SHA256SUMS 的 sha256 b035e2b6…；备份同名目录 |  |
| EXP-0005-R07 | tqnet ETTh1-336，包装脚本，种子 2021 | 10-24 05:55 / 05:56 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4784，MAE 0.4489 | 2545 个测试窗口 | `code/output/EXP-0005/EXP-0005-R07/`，SHA256SUMS 的 sha256 08c58846…；备份同名目录 |  |
| EXP-0005-R08 | tqnet ETTh1-336，包装脚本，种子 2022 | 10-24 05:56 / 05:56 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4810，MAE 0.4489 | 2545 个测试窗口 | `code/output/EXP-0005/EXP-0005-R08/`，SHA256SUMS 的 sha256 80b923f7…；备份同名目录 |  |
| EXP-0005-R09 | tqnet ETTh1-336，包装脚本，种子 2023 | 10-24 05:56 / 05:56 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4795，MAE 0.4483 | 2545 个测试窗口 | `code/output/EXP-0005/EXP-0005-R09/`，SHA256SUMS 的 sha256 450594c1…；备份同名目录 |  |
| EXP-0005-R10 | tqnet ETTh1-720，包装脚本，种子 2021 | 10-24 05:56 / 05:57 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4881，MAE 0.4701 | 2161 个测试窗口 | `code/output/EXP-0005/EXP-0005-R10/`，SHA256SUMS 的 sha256 a9d872fd…；备份同名目录 |  |
| EXP-0005-R11 | tqnet ETTh1-720，包装脚本，种子 2022 | 10-24 05:57 / 05:57 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.5255，MAE 0.4914 | 2161 个测试窗口 | `code/output/EXP-0005/EXP-0005-R11/`，SHA256SUMS 的 sha256 1f0d8a0d…；备份同名目录 |  |
| EXP-0005-R12 | tqnet ETTh1-720，包装脚本，种子 2023 | 10-24 05:57 / 05:57 | completed（退出码 0） | commit 2a1b499，无未提交改动；`tqnet` 环境；MSE 0.4921，MAE 0.4716 | 2161 个测试窗口 | `code/output/EXP-0005/EXP-0005-R12/`，SHA256SUMS 的 sha256 3814f323…；备份同名目录 |  |
| EXP-0005-R13 | wpmixer ETTh1-96，包装脚本，种子 2021 | 10-24 05:57 / 05:58 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R13/`，SHA256SUMS 的 sha256 3da29767…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R28 |
| EXP-0005-R14 | wpmixer ETTh1-96，包装脚本，种子 2022 | 10-24 05:58 / 05:58 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R14/`，SHA256SUMS 的 sha256 446498a4…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R29 |
| EXP-0005-R15 | wpmixer ETTh1-96，包装脚本，种子 2023 | 10-24 05:58 / 05:58 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R15/`，SHA256SUMS 的 sha256 9b8c31ed…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R30 |
| EXP-0005-R16 | wpmixer ETTh1-192，包装脚本，种子 2021 | 10-24 05:58 / 05:58 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R16/`，SHA256SUMS 的 sha256 db9013df…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R31 |
| EXP-0005-R17 | wpmixer ETTh1-192，包装脚本，种子 2022 | 10-24 05:58 / 05:59 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R17/`，SHA256SUMS 的 sha256 c193c50a…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R32 |
| EXP-0005-R18 | wpmixer ETTh1-192，包装脚本，种子 2023 | 10-24 05:59 / 05:59 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R18/`，SHA256SUMS 的 sha256 c906464c…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R33 |
| EXP-0005-R19 | wpmixer ETTh1-336，包装脚本，种子 2021 | 10-24 05:59 / 05:59 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R19/`，SHA256SUMS 的 sha256 144870b6…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R34 |
| EXP-0005-R20 | wpmixer ETTh1-336，包装脚本，种子 2022 | 10-24 05:59 / 05:59 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R20/`，SHA256SUMS 的 sha256 4ed95a87…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R35 |
| EXP-0005-R21 | wpmixer ETTh1-336，包装脚本，种子 2023 | 10-24 05:59 / 05:59 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R21/`，SHA256SUMS 的 sha256 6e4b5d35…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R36 |
| EXP-0005-R22 | wpmixer ETTh1-720，包装脚本，种子 2021 | 10-24 05:59 / 05:59 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R22/`，SHA256SUMS 的 sha256 9abef462…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R37 |
| EXP-0005-R23 | wpmixer ETTh1-720，包装脚本，种子 2022 | 10-24 05:59 / 06:00 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R23/`，SHA256SUMS 的 sha256 71dbc29d…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R38 |
| EXP-0005-R24 | wpmixer ETTh1-720，包装脚本，种子 2023 | 10-24 06:00 / 06:00 | failed（退出码 1） | commit 2a1b499，无未提交改动；`wpmixer` 环境；NA | NA 个测试窗口 | `code/output/EXP-0005/EXP-0005-R24/`，SHA256SUMS 的 sha256 c47b033c…；备份同名目录 | ModuleNotFoundError: thop（WPMixer 的 exp_main.py 导入它，requirements.txt 未列出）；补装后重跑为 R39 |
| EXP-0005-R25 | duet ETTh1-96，包装脚本，种子 2021 | 10-24 06:00 / 06:02 | completed（退出码 0） | commit 2a1b499，无未提交改动；`duet` 环境；MSE 0.3775，MAE 0.3951 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R25/`，SHA256SUMS 的 sha256 eaa927d9…；备份同名目录 |  |
| EXP-0005-R26 | duet ETTh1-96，包装脚本，种子 2022 | 10-24 06:02 / 06:04 | completed（退出码 0） | commit 2a1b499，无未提交改动；`duet` 环境；MSE 0.3794，MAE 0.3949 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R26/`，SHA256SUMS 的 sha256 eee5ca7a…；备份同名目录 |  |
| EXP-0005-R27 | duet ETTh1-96，包装脚本，种子 2023 | 10-24 06:05 / 06:06 | completed（退出码 0） | commit 2a1b499，无未提交改动；`duet` 环境；MSE 0.3816，MAE 0.3974 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R27/`，SHA256SUMS 的 sha256 2d600dd3…；备份同名目录 |  |
| EXP-0005-R28 | wpmixer ETTh1-96，包装脚本，种子 2021 | 10-24 06:07 / 06:07 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.3776，MAE 0.3953 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R28/`，SHA256SUMS 的 sha256 51cee3cc…；备份同名目录 | R13 的重跑（补装 thop 后） |
| EXP-0005-R29 | wpmixer ETTh1-96，包装脚本，种子 2022 | 10-24 06:08 / 06:08 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.3774，MAE 0.3914 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R29/`，SHA256SUMS 的 sha256 2ad52e47…；备份同名目录 | R14 的重跑（补装 thop 后） |
| EXP-0005-R30 | wpmixer ETTh1-96，包装脚本，种子 2023 | 10-24 06:08 / 06:08 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.3785，MAE 0.3971 | 2785 个测试窗口 | `code/output/EXP-0005/EXP-0005-R30/`，SHA256SUMS 的 sha256 1fc77b84…；备份同名目录 | R15 的重跑（补装 thop 后） |
| EXP-0005-R31 | wpmixer ETTh1-192，包装脚本，种子 2021 | 10-24 06:09 / 06:09 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4358，MAE 0.4203 | 2689 个测试窗口 | `code/output/EXP-0005/EXP-0005-R31/`，SHA256SUMS 的 sha256 21836280…；备份同名目录 | R16 的重跑（补装 thop 后） |
| EXP-0005-R32 | wpmixer ETTh1-192，包装脚本，种子 2022 | 10-24 06:09 / 06:10 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4251，MAE 0.4172 | 2689 个测试窗口 | `code/output/EXP-0005/EXP-0005-R32/`，SHA256SUMS 的 sha256 57643fdb…；备份同名目录 | R17 的重跑（补装 thop 后） |
| EXP-0005-R33 | wpmixer ETTh1-192，包装脚本，种子 2023 | 10-24 06:10 / 06:11 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4345，MAE 0.4200 | 2689 个测试窗口 | `code/output/EXP-0005/EXP-0005-R33/`，SHA256SUMS 的 sha256 cd523cfb…；备份同名目录 | R18 的重跑（补装 thop 后） |
| EXP-0005-R34 | wpmixer ETTh1-336，包装脚本，种子 2021 | 10-24 06:11 / 06:11 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4674，MAE 0.4389 | 2545 个测试窗口 | `code/output/EXP-0005/EXP-0005-R34/`，SHA256SUMS 的 sha256 bf56f5d1…；备份同名目录 | R19 的重跑（补装 thop 后） |
| EXP-0005-R35 | wpmixer ETTh1-336，包装脚本，种子 2022 | 10-24 06:11 / 06:12 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4659，MAE 0.4381 | 2545 个测试窗口 | `code/output/EXP-0005/EXP-0005-R35/`，SHA256SUMS 的 sha256 fe100366…；备份同名目录 | R20 的重跑（补装 thop 后） |
| EXP-0005-R36 | wpmixer ETTh1-336，包装脚本，种子 2023 | 10-24 06:12 / 06:13 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4848，MAE 0.4444 | 2545 个测试窗口 | `code/output/EXP-0005/EXP-0005-R36/`，SHA256SUMS 的 sha256 6090d65a…；备份同名目录 | R21 的重跑（补装 thop 后） |
| EXP-0005-R37 | wpmixer ETTh1-720，包装脚本，种子 2021 | 10-24 06:13 / 06:13 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4708，MAE 0.4586 | 2161 个测试窗口 | `code/output/EXP-0005/EXP-0005-R37/`，SHA256SUMS 的 sha256 4b3622ea…；备份同名目录 | R22 的重跑（补装 thop 后） |
| EXP-0005-R38 | wpmixer ETTh1-720，包装脚本，种子 2022 | 10-24 06:13 / 06:14 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4598，MAE 0.4524 | 2161 个测试窗口 | `code/output/EXP-0005/EXP-0005-R38/`，SHA256SUMS 的 sha256 3dbf49b2…；备份同名目录 | R23 的重跑（补装 thop 后） |
| EXP-0005-R39 | wpmixer ETTh1-720，包装脚本，种子 2023 | 10-24 06:14 / 06:15 | completed（退出码 0） | commit 2a1b499，无未提交改动；`wpmixer` 环境；MSE 0.4659，MAE 0.4534 | 2161 个测试窗口 | `code/output/EXP-0005/EXP-0005-R39/`，SHA256SUMS 的 sha256 2400f420…；备份同名目录 | R24 的重跑（补装 thop 后） |

## 结果

由 `code/ts_forecasting/analyze_exp0005.py`（tables.md、summary.json）、`compare_tfb_ett.py`（compare_tfb_ett.json）与 `analyze_ts_windows.py`（windows_stats.md、windows_summary.json）生成，都在 `code/output/EXP-0005/analysis/`（SHA256SUMS 的 sha256 9440032b…，附三个脚本的副本，已备份）。MSE 为全部测试窗口、标准化尺度；均值与样本标准差取三个种子。iTransformer 一行取自 EXP-0004 阶段 A（包装脚本的三个种子），作为同一协议下的参照。

### 数据核对

- TFB 的 ETTh1.csv 有 17420 行，与 ETDataset 的 ETTh1.csv 日期完全相同，原始值最大绝对差 3.6e-15。
- DUET 保存的测试窗口（2785 × 96 × 7，训练集标准化）与 iTransformer ETTh1-96（EXP-0004-R03）的 true.npy 逐值相同：最大绝对差 2.2e-07，没有一个窗口的差超过 1e-4；变量顺序一致。
- 结论：DUET（TFB 框架）与 iTransformer、TQNet、WPMixer（TSLib 式框架）在 ETTh1-96 上计分的是同一批窗口、同一个标准化尺度，可以直接比较。

### 各方法的 MSE

| 方法 | 96 | 192 | 336 | 720 | 四档平均 | 作者报告 |
| --- | --- | --- | --- | --- | --- | --- |
| iTransformer（EXP-0004） | 0.3858 ± 0.0008 | 0.4426 ± 0.0033 | 0.4902 ± 0.0015 | 0.5120 ± 0.0047 | 0.4576 | 0.386/0.441/0.487/0.503，平均 0.454 |
| TQNet | 0.3728 ± 0.0002 | 0.4295 ± 0.0026 | 0.4796 ± 0.0013 | 0.5019 ± 0.0206 | 0.4460 | result.txt（种子 2024）0.3712/0.4284/0.4757/0.4874；论文平均 0.441 |
| WPMixer（全部窗口） | 0.3778 ± 0.0006 | 0.4318 ± 0.0058 | 0.4727 ± 0.0105 | 0.4655 ± 0.0055 | 0.4370 | 96 档 0.368（drop_last），论文平均 0.422 ± 0.001 |
| WPMixer（官方 drop_last 计分） | 0.3747 | 0.4317 | 0.4687 | 0.4572 | 0.4331 | 同上 |
| DUET | 0.3795 ± 0.0020 | — | — | — | — | 0.377 |

- 各种子：TQNet 720 档为 0.4881 / 0.5255 / 0.4921（种子 2022 明显偏高）；WPMixer 336 档为 0.4674 / 0.4659 / 0.4848；DUET 为 0.3775 / 0.3794 / 0.3816。
- 与 EXP-0003（CPU）相比：TQNet 各档的 GPU 均值比 CPU 单种子（2024）高 0.001～0.017；DUET 的 CPU 结果为 0.3898（种子 2021）、0.3790（种子 2022），GPU 上为 0.3775、0.3794。
- WPMixer 的 drop_last 计分比全部窗口低 0.0000～0.0084（与丢掉的窗口数一致：192 档只丢 1 个窗口），与 EXP-0003 的方向相同。

### 逐窗口误差

- **末尾窗口对所有方法都更难**：最后 97 个窗口的 MSE 与其余窗口之比，在 4 种方法、4 个预测长度上为 1.09～1.37（每种方法 × 长度取三个种子的平均），全部大于 1。F-TS2（EXP-0003 中只在 WPMixer 上由两种计分反推）在这里对四种方法都直接测到。
- **误差在测试期内的分布几乎与方法无关**：96 档按十分位的 MSE，四种方法都是约 0.31、0.22、0.28、0.32、0.48、0.38、0.61、0.36、0.41、0.41，最难的是第 7 个十分位；336、720 档最难的是最后一个十分位。
- **方法之间逐窗口高度相关**：ETTh1-96 上，TQNet（种子 2021）与 DUET、WPMixer 各次运行的逐窗口 MSE 相关系数为 0.971～0.980，与 TQNet 自己的另外两个种子为 0.994；TQNet 优于 DUET、WPMixer 的窗口比例为 54%～58%，与自己的另一个种子比为 49%～50%。完整表格见 windows_stats.md。

## 偏差与限制

- **失败与重跑**：WPMixer 的 R13～R24 在启动时报 `ModuleNotFoundError: No module named 'thop'`。thop 不在 WPMixer 的 requirements.txt 中，但它的 `exp/exp_main.py` 在模块顶部导入（只用于统计 GFLOPs）。用约束文件固定 torch 后在 `wpmixer` 环境中补装 thop 0.1.1，torch 未变、`pip check` 无冲突（code/README 已记录），随后用新 ID R28～R39 重跑，全部成功。失败的 12 个目录保留。
- WPMixer 按官方统一设置脚本使用 `--use_amp`（EXP-0003 在 CPU 上没有 AMP）。
- DUET 只做 1 次 mask 重采样（计划 v1.1）。
- 只有 ETTh1；DUET 只有 96 档。
- 三个种子；TQNet 720 档的标准差（0.0206）主要来自一个种子，结论对该档要谨慎。

## 结论边界与下一步

- **当前证据支持**：在同一协议、同一批窗口上，ETTh1 的四个预测长度中 TQNet 与 WPMixer 都好于 iTransformer；DUET 96 档与 WPMixer 接近。iTransformer 作为“效果稍弱”的 baseline 在 ETTh1 上成立。
- **当前证据不能支持**：方法之间的排序在其他数据集上是否相同；WPMixer 与论文的差距来自 GPU/AMP 还是来自测试集调参的超参数在新环境中失效。
- **最值得深挖的现象**：不同架构在同一批窗口上出错（逐窗口相关约 0.97），误差主要由测试期中的少数“难段”决定。下一步看这些难段的数据特征（例如与训练期的分布差异）。
- **后续路由**：squeeze-experiment（SQZ-EXP-0005）→ 时序方向的 Mode C（尚未建立）。
