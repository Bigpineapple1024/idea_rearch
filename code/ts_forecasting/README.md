# ts_forecasting：多变量长期时序预测 baseline 复现与评测核验

方向：多变量长期预测（DUET、TQNet、WPMixer）。论文分析见 analysis/论文笔记/PAPER-0004～0006，复现记录见 [EXP-0003](../../analysis/实验报告/EXP-0003_record.md)，三篇对比与 baseline 选择见 [时序三篇对比](../../analysis/专题分析/时序预测_DUET_TQNet_WPMixer对比.md)。

**状态（2026-10-23）**：在 CPU（4 核）上用真实 ETTh1 数据跑了三个官方仓库；这些是真实训练与测试结果，不是随机权重的代码检查。其他数据集尚未运行。

## 文件

| 文件 | 作用 |
| --- | --- |
| wpmixer_droplast.py | 运行 WPMixer 官方 run_LTF.py，同一个 checkpoint 在测试集上分别按官方 `drop_last=True` 和全部窗口打分 |
| duet_test_stochasticity.py | 运行 DUET 官方 TFB 入口，并在测试时用另外 10 个随机种子重新采样通道 mask，测量分数波动 |
| probe_duet_mask.py | 代码级探针：随机权重下 DUET 在 eval 模式对同一输入两次前向输出是否相同 |
| to_tfb_format.py | 把 ETDataset 的宽表 CSV 转成 TFB/DUET 需要的长表与 FORECAST_META.csv（长度 14400 = 12/4/4 个月标准划分） |

三个仓库的文件都不被修改；包装脚本只在本进程内替换函数。

## 固定版本

| 组件 | 来源 | commit / 版本 |
| --- | --- | --- |
| DUET | github.com/decisionintelligence/DUET | dcc6e6780a9138731b64b9b5398a94a1d97033f0 |
| TQNet | github.com/ACAT-SCUT/TQNet | 15e19cb23483ed52398566c4baa959168cfffa57 |
| WPMixer | github.com/Secure-and-Intelligent-Systems-Lab/WPMixer | 74104c9dddd54d279eb8323f48934b4fd75fcae7 |
| ETTh1 | github.com/zhouhaoyi/ETDataset ETT-small/ETTh1.csv | 仓库 HEAD 1d16c8f4，文件 sha256 f18de3ad…ee066，17420 行 |
| 本次运行环境 | CPU | Python 3.11、torch 2.1.2（CPU 运行）、pandas 1.5.3、scikit-learn 1.3.2、PyWavelets 1.6.0、optuna 3.6.1；DUET 另需 ray、dash、dash-bootstrap-components、reformer-pytorch |

各仓库官方推荐的环境不同（DUET：Python 3.8 + torch 2.4.1；TQNet：Python 3.8；WPMixer：Python 3.10 + torch 2.0.1）。在 GPU 上正式实验时，建议每个仓库用各自的推荐环境，并记录与本次 CPU 结果的差异。

## 复现命令

```bash
# 数据
curl -o ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

# TQNet（官方 scripts/TQNet/etth1.sh 的参数；CPU 约 1 分钟）
cd TQNet && mkdir -p dataset && cp ../ETTh1.csv dataset/
python -u run.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_96_96 \
  --model TQNet --data ETTh1 --features M --seq_len 96 --pred_len 96 --enc_in 7 --cycle 24 \
  --train_epochs 30 --patience 5 --dropout 0.5 --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed 2024

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
