"""Per-window test errors of the time-series runs (EXP-0004, EXP-0005): are the last test windows harder?

  python analyze_ts_windows.py <EXP>:<run_id> [<EXP>:<run_id> ...]    # writes code/output/<first EXP>/analysis/windows_*

Reads each run's saved predictions and targets (standardised scale; windows x horizon x variables):
results/*/pred.npy + true.npy (iTransformer), pred.npy + true.npy (TQNet, DUET), pred_all_windows.npy +
true_all_windows.npy (WPMixer). For every run: the MSE of each test window, the MSE of the last k windows against the
rest for k = 97 and 113 (what WPMixer's drop_last=True with batch 128 leaves out at horizon 96 and 336/720,
SQZ-EXP-0003), and the MSE by tenth of the test period. Runs with the same dataset and horizon are also compared
window by window (they score the same windows only when the window counts match; the script checks the counts).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))   # run from elsewhere: set the repo


def load(exp, run_id):
    d = os.path.join(REPO, "code", "output", exp, run_id)
    for pred_name, true_name in (("results/*/pred.npy", "results/*/true.npy"), ("pred.npy", "true.npy"),
                                 ("pred_all_windows.npy", "true_all_windows.npy")):
        preds, trues = glob.glob(os.path.join(d, pred_name)), glob.glob(os.path.join(d, true_name))
        if len(preds) == 1 and len(trues) == 1:
            with open(os.path.join(d, "config.json")) as f:
                cfg = json.load(f)
            return cfg, np.load(preds[0]), np.load(trues[0])
    raise SystemExit(f"{exp}/{run_id}: no saved predictions")


def main():
    specs = [s.split(":") for s in sys.argv[1:]]
    out = os.path.join(REPO, "code", "output", specs[0][0], "analysis")
    os.makedirs(out, exist_ok=True)
    rows, per_window = [], {}
    for exp, run_id in specs:
        cfg, pred, true = load(exp, run_id)
        mse = ((pred - true) ** 2).mean(axis=(1, 2))
        n = len(mse)
        tenths = [float(mse[int(n * i / 10):int(n * (i + 1) / 10)].mean()) for i in range(10)]
        row = {"run": f"{exp}/{run_id}", "method": cfg["method"], "dataset": cfg["dataset"], "pred_len": cfg["pred_len"],
               "seed": cfg["seed"], "n_windows": n, "mse": float(mse.mean()), "mse_by_tenth": tenths}
        for k in (97, 113):
            row[f"last{k}_mse"], row[f"rest{k}_mse"] = float(mse[-k:].mean()), float(mse[:-k].mean())
        rows.append(row)
        per_window[f"{exp}/{run_id}"] = mse
    with open(os.path.join(out, "windows_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    lines = ["| 运行 | 方法 | 数据集-预测长度 | 种子 | 窗口数 | 全部 MSE | 末 97 窗 / 其余 | 末 113 窗 / 其余 | 首个与末个十分位 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['method']} | {r['dataset']}-{r['pred_len']} | {r['seed']} | {r['n_windows']} | "
                     f"{r['mse']:.4f} | {r['last97_mse']:.4f} / {r['rest97_mse']:.4f} | "
                     f"{r['last113_mse']:.4f} / {r['rest113_mse']:.4f} | {r['mse_by_tenth'][0]:.4f} / {r['mse_by_tenth'][-1]:.4f} |")
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["pred_len"]), []).append(r["run"])
    lines += ["", "| 数据集-预测长度 | 运行 A | 运行 B | 窗口数 | 逐窗口 MSE 的相关系数 | A 优于 B 的窗口比例 |", "| --- | --- | --- | --- | --- | --- |"]
    for (ds, pl), runs in groups.items():
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = per_window[runs[i]], per_window[runs[j]]
                if len(a) != len(b):
                    lines.append(f"| {ds}-{pl} | {runs[i]} | {runs[j]} | {len(a)} vs {len(b)} | 窗口数不同，未比较 | — |")
                    continue
                lines.append(f"| {ds}-{pl} | {runs[i]} | {runs[j]} | {len(a)} | {np.corrcoef(a, b)[0, 1]:.3f} | "
                             f"{100 * float((a < b).mean()):.1f}% |")
    with open(os.path.join(out, "windows_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
