# ts_forecasting：多变量长期时序预测 baseline 复现与评测核验

方向：多变量长期预测。主 baseline 为 iTransformer（DEC-0012，备选 CycleNet）；DUET、TQNet、WPMixer 为强对照。三篇对照的论文分析见 analysis/论文笔记/PAPER-0001～0003，复现记录见 [EXP-0003](../../analysis/实验报告/EXP-0003_record.md)，三篇对比与 baseline 选择见 [时序三篇对比](../../analysis/专题分析/时序预测_DUET_TQNet_WPMixer对比.md)。

**状态（2026-10-23）**：在 CPU（4 核）上用真实 ETTh1 数据跑了 DUET、TQNet、WPMixer 三个官方仓库；这些是真实训练与测试结果，不是随机权重的代码检查。其他数据集尚未运行。iTransformer 尚未复现（云端 CPU 运行按用户要求中止，没有留下结果）。

## 文件

| 文件 | 作用 |
| --- | --- |
| exp_ts.py | 实验室 GPU 上的运行器（EXP-0004 起）：每次运行一个目录（仓库自己的 checkpoints/、results/ 也在其中），写 cmd、config、git、seed、env、gpu、stdout、metrics、status 和 SHA256SUMS，复制到备份并核对，追加到 runs.tsv；拒绝复用已有的 run ID |
| itransformer_seed.py | 用其他种子运行 iTransformer 官方 run.py：在内存中只替换 `fix_seed = 2023` 这一行后执行，不修改官方文件；种子 2023 时执行的代码与 run.py 字节相同 |
| itransformer_val.py | 不训练：官方 run.py 以 `--is_training 0` 载入 ./checkpoints/ 下的检查点，本脚本把交给 `exp.test()` 的数据换成验证集（加载方式与测试集相同：按顺序、batch 1），因此保存的 pred.npy、true.npy、metrics.npy 都是验证期的。exp_ts.py 的 `itransformer_val` 方法复制 EXP-0004 同数据集、同预测长度、同种子运行的检查点后调用它。CPU 冒烟测试：同一检查点走官方测试路径，预测与原运行保存的逐位相同 |
| tqnet_save_pred.py | 运行 TQNet 官方 run.py，在本进程内替换 `metric`，先保存测试集的 pred.npy、true.npy 再返回原指标（官方保存语句被注释掉了） |
| prepare_tfb_data.py | 从 TFB 的 forecasting.zip 只解压 ETTh1.csv 与 FORECAST_META.csv 到 `D:\6CJNM007\data\tfb\`，并写 SHA256SUMS |
| compare_tfb_ett.py | 核对 DUET（TFB 数据）与 iTransformer 等（ETDataset CSV）是否在同一批测试窗口上计分：原始值逐值比较，标准化后的测试窗口逐值比较 |
| analyze_exp0004.py | EXP-0004 的表格：各数据集 × 预测长度的直接运行、包装脚本各种子、均值 ± 标准差，以及与论文 Table 10 的复现判据 |
| analyze_ts_windows.py | 逐窗口误差：每个测试窗口的 MSE、末 97/113 个窗口与其余窗口的对比、按十分位的 MSE，以及窗口数相同的方法之间逐窗口比较 |
| analyze_exp0005.py | EXP-0005 的表格：DUET、TQNet、WPMixer 与 iTransformer 在同一协议下的各种子 MSE、均值 ± 标准差，WPMixer 另列官方 drop_last 计分 |
| analyze_hard_windows.py | SQZ-EXP-0005 补充分析：从 CSV 重建 ETTh1-96 测试窗口，逐窗口平均误差与水平差、目标段波动、输入偏离训练期的 Spearman 相关 |
| analyze_level_shape.py | F-TS8 的 1a–2b 分析：每个窗口的 MSE 精确拆成水平部分与形状部分；预测水平变化与真实水平变化的斜率；把预测的水平换成输入段均值等简单水平时的误差；各变量的有符号水平误差与各时期数据自身的平均漂移。可选数据集与验证期（EXP-0009 的 itransformer_val 运行） |
| analyze_exp0009_rules.py | EXP-0009 计划 v1、v1.1 的判读：从验证期与测试期的水平 / 形状拆分读出收缩系数表，逐个“数据集 × 预测长度”判断规则 1～3 与普遍性判据，写 code/output/EXP-0009/analysis/rules.md、rules.json |
| wpmixer_droplast.py | 运行 WPMixer 官方 run_LTF.py，同一个 checkpoint 在测试集上分别按官方 `drop_last=True` 和全部窗口打分；另把全部窗口的预测与真值存为 pred_all_windows.npy、true_all_windows.npy（存到启动时的目录） |
| duet_test_stochasticity.py | 运行 DUET 官方 TFB 入口，并在测试时用另外 `--resamples` 个随机种子重新采样通道 mask，测量分数波动；`--save-pred` 时保存官方预测与标准化后的真值 |
| probe_duet_mask.py | 代码级探针：随机权重下 DUET 在 eval 模式对同一输入两次前向输出是否相同 |
| to_tfb_format.py | 把 ETDataset 的宽表 CSV 转成 TFB/DUET 需要的长表与 FORECAST_META.csv（长度 14400 = 12/4/4 个月标准划分） |

三个仓库的文件都不被修改；包装脚本只在本进程内替换函数。

## 固定版本

| 组件 | 来源 | commit / 版本 |
| --- | --- | --- |
| iTransformer | github.com/thuml/iTransformer | c2426e68ca13f74aaec08045c5c724d8ad328124 |
| DUET | github.com/decisionintelligence/DUET | dcc6e6780a9138731b64b9b5398a94a1d97033f0 |
| TQNet | github.com/ACAT-SCUT/TQNet | 15e19cb23483ed52398566c4baa959168cfffa57 |
| WPMixer | github.com/Secure-and-Intelligent-Systems-Lab/WPMixer | 74104c9dddd54d279eb8323f48934b4fd75fcae7 |
| ETTh1 | github.com/zhouhaoyi/ETDataset ETT-small/ETTh1.csv | 仓库 HEAD 1d16c8f4，文件 sha256 f18de3ad…ee066，17420 行 |
| 本次运行环境 | CPU | Python 3.11、torch 2.1.2（CPU 运行）、pandas 1.5.3、scikit-learn 1.3.2、PyWavelets 1.6.0、optuna 3.6.1；DUET 另需 ray、dash、dash-bootstrap-components、reformer-pytorch |

各仓库官方推荐的环境不同（iTransformer：torch 2.0.0；DUET：Python 3.8 + torch 2.4.1；TQNet：Python 3.8；WPMixer：Python 3.10 + torch 2.0.1）。在实验室电脑（原生 Windows）上每个仓库用各自的环境，安装方法和每条命令前要加载的环境脚本见 [code/README](../README.md)，并记录与本次 CPU 结果的差异。

**Windows 上统一加 `--num_workers 0`**：TQNet 的 run.py 没有 `__main__` 保护，worker 数大于 0 时 Windows 会在子进程里重新执行训练；iTransformer 默认 10 个 worker，每个 epoch 都要重新启动子进程，很慢。数据顺序由主进程决定，worker 数不影响结果。WPMixer、DUET 的默认值已经是 0。下面命令中的 `/path/to/...` 换成 `D:/6CJNM007/idea_rearch/source/<仓库>`。

## 复现命令

```bash
# 数据
curl -o ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

# iTransformer（主 baseline；官方 scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh 的第 1 组参数）
# 预测长度 192 的 d_model/d_ff 为 256，336、720 为 512，以脚本为准
cd iTransformer && mkdir -p dataset/ETT-small && cp ../ETTh1.csv dataset/ETT-small/
python -u run.py --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_96_96 \
  --model iTransformer --data ETTh1 --features M --seq_len 96 --pred_len 96 --e_layers 2 --enc_in 7 --dec_in 7 --c_out 7 \
  --des Exp --d_model 256 --d_ff 256 --itr 1 --num_workers 0
# 注意：run.py 把随机种子写死为 2023（fix_seed = 2023）。其他种子用本目录的 itransformer_seed.py，
# 先用种子 2023 核对它与直接运行的结果一致（EXP-0004 计划）。实验室电脑上用 exp_ts.py 运行，例如
#   python exp_ts.py EXP-0004 EXP-0004-R01:itransformer:ETTh1:96:2023:direct EXP-0004-R04:itransformer:ETTh1:96:2021
# 测试集 batch_size=1，drop_last=True 不会丢窗口。

# TQNet（官方 scripts/TQNet/etth1.sh 的参数；CPU 约 1 分钟）
cd TQNet && mkdir -p dataset && cp ../ETTh1.csv dataset/
python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96_96 \
  --model TQNet --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --cycle 24 \
  --train_epochs 30 --patience 5 --dropout 0.5 --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed 2024 --num_workers 0

# WPMixer（scripts/Unified/ETTh1_Unified_setup.sh 第 1 组参数；GPU 上可加 --use_amp）
cd WPMixer && mkdir -p data/ETT && cp ../ETTh1.csv data/ETT/
python wpmixer_droplast.py /path/to/WPMixer -- --model WPMixer --task_name long_term_forecast --data ETTh1 \
  --seq_len 96 --pred_len 96 --d_model 256 --tfactor 3 --dfactor 3 --wavelet coif4 --level 1 --patch_len 16 \
  --stride 8 --batch_size 128 --learning_rate 0.002991949 --lradj type3 --dropout 0.1 --embedding_dropout 0.2 \
  --patience 5 --train_epochs 10

# DUET（scripts/multivariate_forecast/DUET_unified_seq_len_96.sh 的 ETTh1-96 参数）
python to_tfb_format.py ETTh1.csv /path/to/DUET
python duet_test_stochasticity.py /path/to/DUET --resamples 10 -- --config-path "rolling_forecast_config.json" \
  --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' --model-name "duet.DUET" \
  --model-hyper-params '{"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type3", "n_heads": 8, "norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 96}' \
  --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/DUET"
```

## 公平比较需要统一的地方

1. **测试集全部计分**：WPMixer 官方测试 `drop_last=True`，必须改为 False（或用 wpmixer_droplast.py 的 all_windows 分数）。
2. **超参数只在验证集上选**：WPMixer 的 Optuna 目标是测试集最小损失，其发布的超参数不能直接用于公平对比；DUET 主结果在多个回看窗口中“报告最好结果”，选择依据需回查论文。
3. **回看窗口固定**：主表统一用 96，另做 336/512 的补充实验。
4. **多种子**：至少 3～5 个种子，报告均值与标准差；TQNet 在 ETTh1-96 上的种子标准差约 0.003，与方法间差距同量级。
5. **DUET 的测试随机性**：通道 mask 在测试时也是随机采样。ETTh1-96 上实测只重采样 mask 时 MSE 标准差约 0.0003，影响很小，但报告时仍应注明种子。
