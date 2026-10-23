# EXPERIMENT_RECORD：DUET、TQNet、WPMixer 在统一协议下的 GPU 重跑（ETTh1）

状态：计划 v1 已写定（2026-10-23，第一次运行之前；2026-10-24 提交，仍在第一次运行之前），执行中。三个方法的运行入口尚未加入 `exp_ts.py`，加入后另行提交。
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
