"""Which ETTh1 test windows are hard for every method? Relate per-window error to features of the data (SQZ-EXP-0005).

  python analyze_hard_windows.py      # writes code/output/EXP-0005/squeeze/{hard_windows.md,hard_windows.json}

Rebuilds the ETTh1-96 test windows from the CSV exactly as the TSLib-style loaders do (12/4/4-month split, test inputs
start 96 steps before the test border, standardised with the training split's mean and SD) and checks them against a
saved true.npy. Per window: the mean MSE over all ETTh1-96 runs of EXP-0004 (iTransformer) and EXP-0005 (TQNet,
WPMixer, DUET), and data features on the standardised scale: the level shift between the 96 target steps and the 96
input steps (mean over variables of |mean(target) - mean(input)|), the target's own spread, and how far the input's
mean lies from the training data. Reports Spearman correlations and the error by quintile of the level shift.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output")
CSV = "D:/6CJNM007/data/ETT-small/ETTh1.csv"
SEQ, PRED = 96, 96


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    df = pd.read_csv(CSV).drop(columns="date").to_numpy(dtype=np.float64)
    train_end = 12 * 30 * 24
    test_start, test_end = 12 * 30 * 24 + 4 * 30 * 24 - SEQ, 12 * 30 * 24 + 8 * 30 * 24
    mu, sd = df[:train_end].mean(axis=0), df[:train_end].std(axis=0)
    z = (df - mu) / sd
    seg = z[test_start:test_end]
    n = len(seg) - SEQ - PRED + 1
    inputs = np.stack([seg[i:i + SEQ] for i in range(n)])
    targets = np.stack([seg[i + SEQ:i + SEQ + PRED] for i in range(n)])
    ref = np.load(glob.glob(os.path.join(OUT, "EXP-0004", "EXP-0004-R03", "results", "*", "true.npy"))[0])
    rebuild_diff = float(np.abs(ref - targets).max())                # sklearn's StandardScaler uses the population SD
    runs = []
    for exp, pattern in (("EXP-0004", "results/*/pred.npy"), ("EXP-0005", "pred.npy"), ("EXP-0005", "pred_all_windows.npy")):
        for path in glob.glob(os.path.join(OUT, exp, f"{exp}-R*", pattern)):
            run_dir = path.split(os.sep + "results" + os.sep)[0] if "results" in path else os.path.dirname(path)
            cfg = json.load(open(os.path.join(run_dir, "config.json")))
            if cfg["dataset"] == "ETTh1" and cfg["pred_len"] == PRED:
                pred = np.load(path)
                if pred.shape == targets.shape:
                    runs.append((cfg["method"], os.path.basename(run_dir), ((pred - ref) ** 2).mean(axis=(1, 2))))
    err = np.mean([e for _, _, e in runs], axis=0)
    shift = np.abs(targets.mean(axis=1) - inputs.mean(axis=1)).mean(axis=1)
    spread = targets.std(axis=1).mean(axis=1)
    novelty = np.abs(inputs.mean(axis=1)).mean(axis=1)               # |input mean| in training-SD units
    q = pd.qcut(shift, 5, labels=False)
    by_quintile = [float(err[q == k].mean()) for k in range(5)]
    method_corr = {}
    for m in sorted({m for m, _, _ in runs}):
        es = [e for mm, _, e in runs if mm == m]
        method_corr[m] = spearman(np.mean(es, axis=0), shift)
    summary = {"n_windows": int(n), "rebuild_max_abs_diff_vs_saved_true": rebuild_diff, "n_runs": len(runs),
               "runs": [f"{m}:{r}" for m, r, _ in runs],
               "spearman_error_vs_level_shift": spearman(err, shift), "spearman_error_vs_target_spread": spearman(err, spread),
               "spearman_error_vs_input_novelty": spearman(err, novelty), "spearman_by_method_vs_level_shift": method_corr,
               "error_by_level_shift_quintile": by_quintile,
               "level_shift_quintile_edges": [float(x) for x in np.quantile(shift, [0, 0.2, 0.4, 0.6, 0.8, 1])]}
    os.makedirs(os.path.join(OUT, "EXP-0005", "squeeze"), exist_ok=True)
    with open(os.path.join(OUT, "EXP-0005", "squeeze", "hard_windows.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"重建的测试窗口与保存的 true.npy 最大差：{rebuild_diff:.2e}；参与平均的运行：{len(runs)} 次（ETTh1-96）。", "",
             "| 数据特征 | 与逐窗口平均 MSE 的 Spearman 相关 |", "| --- | --- |",
             f"| 目标段与输入段的水平差 | {summary['spearman_error_vs_level_shift']:.3f} |",
             f"| 目标段自身的波动 | {summary['spearman_error_vs_target_spread']:.3f} |",
             f"| 输入段均值偏离训练期的程度 | {summary['spearman_error_vs_input_novelty']:.3f} |", "",
             "| 方法 | 该方法逐窗口 MSE 与水平差的 Spearman 相关 |", "| --- | --- |"]
    lines += [f"| {m} | {c:.3f} |" for m, c in method_corr.items()]
    lines += ["", "| 水平差五分位（由小到大） | 1 | 2 | 3 | 4 | 5 |", "| --- | --- | --- | --- | --- | --- |",
              "| 平均 MSE | " + " | ".join(f"{x:.3f}" for x in by_quintile) + " |"]
    with open(os.path.join(OUT, "EXP-0005", "squeeze", "hard_windows.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
