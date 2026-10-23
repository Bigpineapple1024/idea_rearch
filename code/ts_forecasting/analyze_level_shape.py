"""Split each ETT test (or validation) window's error into a level part and a shape part (1a-4b analysis of F-TS8).

  python analyze_level_shape.py [<dataset> <output EXP> [val]]   # default ETTh1 EXP-0005, test split
  # writes code/output/<output EXP>/deepdive/level_shape_<dataset>[_val].{md,json}
  # test: the runs of EXP-0004 and EXP-0005; val: the itransformer_val runs of EXP-0009 (validation predictions)

For a window with prediction P and target Y (H steps x 7 variables, training-standardised scale) the MSE splits exactly
into a level part, the mean over variables of (mean_t P - mean_t Y)^2, and a shape part, the MSE after removing each
series' own mean (the cross term is zero). With X the window's 96 input steps, the predicted level change is
mean P - mean X and the true one mean Y - mean X. Per method (one run per seed; runs of a method averaged, methods
weighted equally in "all"), at each horizon: MSE and level share overall and by quintile of the true level shift; slope
and correlation of the predicted on the true level change; the level error if the prediction's level were replaced by
the input's mean, the mean of its last 24 steps or its last value (shape kept); the level error with the predicted change
scaled by 0 to 1 (a description on the test set, not a tuned rule); the mean signed level error per variable; the mean
predicted level change per variable next to the data's own mean change over the train, validation and test windows; and
the level share per variable.
Test windows are rebuilt from the CSV as in analyze_hard_windows.py and checked against every run's saved targets.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output")
DATASET = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
OUT_EXP = sys.argv[2] if len(sys.argv) > 2 else "EXP-0005"
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "test"
CSV = f"D:/6CJNM007/data/ETT-small/{DATASET}.csv"
SEQ = 96
HORIZONS = (96, 192, 336, 720)
VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SOURCES = {"test": (("EXP-0004", "results/*/pred.npy", "true.npy"), ("EXP-0005", "pred.npy", "true.npy"),
                    ("EXP-0005", "pred_all_windows.npy", "true_all_windows.npy")),
           "val": (("EXP-0009", "results/*/pred.npy", "true.npy"),)}[SPLIT]
MONTH = 30 * 24 * (4 if DATASET.startswith("ETTm") else 1)          # TSLib's ETT borders; 15-minute data has 4x the steps
BORDERS = {"train": (0, 12 * MONTH), "val": (12 * MONTH - SEQ, 16 * MONTH), "test": (16 * MONTH - SEQ, 20 * MONTH)}


def rebuild(pred_len, split="test"):
    df = pd.read_csv(CSV).drop(columns="date").to_numpy(dtype=np.float64)
    train_end = 12 * MONTH
    mu, sd = df[:train_end].mean(axis=0), df[:train_end].std(axis=0)   # population SD, as sklearn's StandardScaler
    seg = ((df - mu) / sd)[slice(*BORDERS[split])]
    n = len(seg) - SEQ - pred_len + 1
    return (np.stack([seg[i:i + SEQ] for i in range(n)]),
            np.stack([seg[i + SEQ:i + SEQ + pred_len] for i in range(n)]))


def mean_change(split, pred_len):
    """Mean over the split's windows of (mean target - mean input) per variable: the data's own average drift."""
    x, y = rebuild(pred_len, split)
    return (y.mean(axis=1) - x.mean(axis=1)).mean(axis=0)


def find_runs(pred_len):
    """{(method, seed): (run_id, pred path, true path)}; the first run ID per seed is kept, later duplicates listed."""
    found = {}
    for exp, pattern, true_name in SOURCES:
        for path in sorted(glob.glob(os.path.join(OUT, exp, f"{exp}-R*", pattern))):
            path = os.path.normpath(path)
            marker = os.sep + "results" + os.sep
            run_dir = path.split(marker)[0] if marker in path else os.path.dirname(path)
            cfg = json.load(open(os.path.join(run_dir, "config.json")))
            if cfg["dataset"] != DATASET or cfg["pred_len"] != pred_len:
                continue
            found.setdefault((cfg["method"], cfg["seed"]), []).append(
                (os.path.basename(run_dir), path, os.path.join(os.path.dirname(path), true_name)))
    return found


SHRINK = (0.0, 0.25, 0.5, 0.75, 1.0)


def per_run(pred, x, y):
    p_mean, y_mean, x_mean = pred.mean(axis=1), y.mean(axis=1), x.mean(axis=1)            # (windows, variables)
    level = (p_mean - y_mean) ** 2
    shape = (((pred - p_mean[:, None]) - (y - y_mean[:, None])) ** 2).mean(axis=1)
    dp, dy = (p_mean - x_mean).ravel(), (y_mean - x_mean).ravel()
    slope = float(np.cov(dp, dy)[0, 1] / dy.var(ddof=1))
    # level error if the predicted change were scaled by a (a = 0: input mean, a = 1: the model as is); descriptive only
    shrink = [float(((x_mean + a * (p_mean - x_mean) - y_mean) ** 2).mean()) for a in SHRINK]
    stats = {"slope": slope, "corr": float(np.corrcoef(dp, dy)[0, 1]), "sd_pred_change": float(dp.std()),
             "sd_true_change": float(dy.std()), "shrink": shrink, "bias_by_variable": (p_mean - y_mean).mean(axis=0),
             "mean_pred_change": (p_mean - x_mean).mean(axis=0)}
    return level, shape, stats


def summarize(level, shape, quint):
    mse = level + shape
    row = {"mse": float(mse.mean()), "level": float(level.mean()), "shape": float(shape.mean()),
           "level_share": float(level.mean() / mse.mean())}
    row["by_quintile"] = [{"mse": float(mse[quint == k].mean()), "level_share": float(level[quint == k].mean() / mse[quint == k].mean())}
                          for k in range(5)]
    return row


def main():
    result = {}
    for h in HORIZONS:
        x, y = rebuild(h, SPLIT)
        found = find_runs(h)
        if not found:
            continue
        shift = np.abs(y.mean(axis=1) - x.mean(axis=1)).mean(axis=1)
        quint = pd.qcut(shift, 5, labels=False)
        methods, kept, duplicates, true_diff = {}, {}, {}, {}
        for (method, seed), runs in sorted(found.items()):
            run_id, pred_path, true_path = runs[0]
            pred = np.load(pred_path).astype(np.float64)
            if pred.shape != y.shape:
                continue
            true_diff[run_id] = float(np.abs(np.load(true_path) - y).max())
            for dup_id, dup_path, _ in runs[1:]:
                duplicates[dup_id] = {"same_seed_as": run_id, "max_abs_pred_diff": float(np.abs(np.load(dup_path) - pred).max())}
            methods.setdefault(method, []).append(per_run(pred, x, y))
            kept.setdefault(method, []).append(run_id)
        rows, mean_level, mean_shape = {}, {}, {}
        for method, items in methods.items():
            mean_level[method] = np.mean([it[0] for it in items], axis=0)
            mean_shape[method] = np.mean([it[1] for it in items], axis=0)
            rows[method] = summarize(mean_level[method].mean(axis=1), mean_shape[method].mean(axis=1), quint)
            stats = [it[2] for it in items]
            rows[method]["slope_pred_on_true_level_change"] = float(np.mean([s["slope"] for s in stats]))
            rows[method]["corr_pred_true_level_change"] = float(np.mean([s["corr"] for s in stats]))
            rows[method]["sd_pred_level_change"] = float(np.mean([s["sd_pred_change"] for s in stats]))
            rows[method]["sd_true_level_change"] = stats[0]["sd_true_change"]
            rows[method]["level_error_by_shrink_factor"] = dict(zip(map(str, SHRINK), np.mean([s["shrink"] for s in stats], axis=0).tolist()))
            rows[method]["mean_signed_level_error_by_variable"] = dict(zip(VARS, np.mean([s["bias_by_variable"] for s in stats], axis=0).tolist()))
            rows[method]["mean_predicted_change_by_variable"] = dict(zip(VARS, np.mean([s["mean_pred_change"] for s in stats], axis=0).tolist()))
            rows[method]["n_runs"] = len(items)
        all_level = np.mean(list(mean_level.values()), axis=0)                # (windows, variables), methods weighted equally
        all_shape = np.mean(list(mean_shape.values()), axis=0)
        rows["all"] = summarize(all_level.mean(axis=1), all_shape.mean(axis=1), quint)
        y_mean = y.mean(axis=1)
        naive = {"model_level_error_all_methods": float(all_level.mean()),
                 "input_mean": float(((x.mean(axis=1) - y_mean) ** 2).mean()),
                 "input_last_24_mean": float(((x[:, -24:].mean(axis=1) - y_mean) ** 2).mean()),
                 "input_last_value": float(((x[:, -1] - y_mean) ** 2).mean())}
        per_var = {v: {"mse": float((all_level[:, i] + all_shape[:, i]).mean()),
                       "level_share": float(all_level[:, i].mean() / (all_level[:, i] + all_shape[:, i]).mean())}
                   for i, v in enumerate(VARS)}
        result[str(h)] = {"n_windows": int(len(y)), "runs": kept, "same_seed_duplicates_left_out": duplicates,
                          "max_abs_diff_saved_true_vs_rebuilt": true_diff, "methods": rows, "naive_level_error": naive,
                          "input_mean_signed_level_error_by_variable": (x.mean(axis=1) - y_mean).mean(axis=0).tolist(),
                          "data_mean_change_by_split": {s: dict(zip(VARS, mean_change(s, h).tolist())) for s in BORDERS},
                          "per_variable_all_methods": per_var,
                          "level_shift_quintile_edges": [float(q) for q in np.quantile(shift, [0, 0.2, 0.4, 0.6, 0.8, 1])]}
    out_dir = os.path.join(OUT, OUT_EXP, "deepdive")
    name = f"level_shape_{DATASET}" + ("" if SPLIT == "test" else f"_{SPLIT}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    lines = [f"{DATASET}，{'测试期' if SPLIT == 'test' else '验证期'}：每个窗口的 MSE 精确拆成水平部分（预测段与目标段各变量均值之差的平方）"
             "和形状部分（各自减去均值后的 MSE）。每个方法每个种子取一次运行；all 为各方法等权平均。", ""]
    for h, r in result.items():
        diff = max(r["max_abs_diff_saved_true_vs_rebuilt"].values())
        lines += [f"### 预测长度 {h}（{r['n_windows']} 个窗口；保存的目标与重建窗口最大差 {diff:.1e}）", "",
                  "| 方法 | 运行数 | MSE | 水平部分 | 形状部分 | 水平占比 | 水平差最大五分位的水平占比 | 预测水平变化对真实水平变化的斜率 | 相关 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, row in r["methods"].items():
            extra = ("—", "—", "—") if m == "all" else (row["n_runs"], f"{row['slope_pred_on_true_level_change']:.3f}",
                                                        f"{row['corr_pred_true_level_change']:.3f}")
            lines.append(f"| {m} | {extra[0]} | {row['mse']:.4f} | {row['level']:.4f} | {row['shape']:.4f} | {row['level_share']:.1%} | "
                         f"{row['by_quintile'][4]['level_share']:.1%} | {extra[1]} | {extra[2]} |")
        q = r["methods"]["all"]["by_quintile"]
        lines += ["", "| 水平差五分位（all） | 1 | 2 | 3 | 4 | 5 |", "| --- | --- | --- | --- | --- | --- |",
                  "| MSE | " + " | ".join(f"{b['mse']:.3f}" for b in q) + " |",
                  "| 水平占比 | " + " | ".join(f"{b['level_share']:.1%}" for b in q) + " |", ""]
        nv = r["naive_level_error"]
        lines += ["| 水平部分的来源（形状部分不变） | 平均水平误差 |", "| --- | --- |",
                  f"| 各方法自己的预测（all） | {nv['model_level_error_all_methods']:.4f} |",
                  f"| 换成输入段均值 | {nv['input_mean']:.4f} |", f"| 换成输入段最后 24 步的均值 | {nv['input_last_24_mean']:.4f} |",
                  f"| 换成输入段最后一个值 | {nv['input_last_value']:.4f} |", "",
                  "| 方法 | 预测水平变化的标准差 | 真实水平变化的标准差 | 水平误差：变化 ×0 | ×0.25 | ×0.5 | ×0.75 | ×1（原样） |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, row in r["methods"].items():
            if m != "all":
                sh = row["level_error_by_shrink_factor"]
                lines.append(f"| {m} | {row['sd_pred_level_change']:.3f} | {row['sd_true_level_change']:.3f} | "
                             + " | ".join(f"{sh[str(a)]:.4f}" for a in SHRINK) + " |")
        lines += ["", "| 方法：各变量平均有符号水平误差（预测均值 − 目标均值） | " + " | ".join(VARS) + " |",
                  "| --- |" + " --- |" * len(VARS)]
        for m, row in r["methods"].items():
            if m != "all":
                lines.append(f"| {m} | " + " | ".join(f"{row['mean_signed_level_error_by_variable'][v]:+.3f}" for v in VARS) + " |")
        lines += [f"| 输入段均值 | " + " | ".join(f"{b:+.3f}" for b in r["input_mean_signed_level_error_by_variable"]) + " |", "",
                  "| 平均水平变化（目标段均值 − 输入段均值） | " + " | ".join(VARS) + " |", "| --- |" + " --- |" * len(VARS)]
        for s, ch in r["data_mean_change_by_split"].items():
            lines.append(f"| 数据：{s} 期全部窗口 | " + " | ".join(f"{ch[v]:+.3f}" for v in VARS) + " |")
        for m, row in r["methods"].items():
            if m != "all":
                lines.append(f"| 预测：{m}（{'测试期' if SPLIT == 'test' else '验证期'}） | "
                             + " | ".join(f"{row['mean_predicted_change_by_variable'][v]:+.3f}" for v in VARS) + " |")
        lines += ["",
                  "| 变量（all） | " + " | ".join(VARS) + " |", "| --- |" + " --- |" * len(VARS),
                  "| MSE | " + " | ".join(f"{r['per_variable_all_methods'][v]['mse']:.3f}" for v in VARS) + " |",
                  "| 水平占比 | " + " | ".join(f"{r['per_variable_all_methods'][v]['level_share']:.1%}" for v in VARS) + " |", ""]
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
