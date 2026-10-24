"""EXP-0012 squeeze (exploratory, after the plan-v1 verdict): where S1's gains and losses come from.

  python squeeze_level_predictor.py     # writes code/output/EXP-0012/squeeze/level_predictor_squeeze.{json,md}

Per dataset, horizon and period: S1's level error minus iTransformer's, per variable; S1's coefficient on the input mean
m, per variable; two ablations fit the same way on the training windows, S1 without m (intercept, last - m, last24 - m,
slope) and m alone (intercept, m); and the mean input level m per variable over each period's windows (the training
scale has mean 0 and SD 1). Post hoc: nothing here changes the plan-v1 verdict.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_level_predictor import (DATASETS, HORIZONS, OUT, SPLIT_JSON, features, load_pred, scaled)   # noqa: E402

VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
SETS = {"s1": [0, 1, 2, 3], "no_m": [1, 2, 3], "only_m": [0]}


def fit(f, cols):
    coef = []
    for v in range(f["dy"].shape[1]):
        a = np.column_stack([np.ones(len(f["dy"])), f["stats"][:, v][:, cols]])
        coef.append(np.linalg.lstsq(a, f["dy"][:, v], rcond=None)[0])
    return coef


def predict(coef, f, cols):
    return np.stack([c[0] + f["stats"][:, v][:, cols] @ c[1:] for v, c in enumerate(coef)], axis=1)


def main():
    out_dir = f"{OUT}/EXP-0012/squeeze"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sealed = {(ds, split): json.load(open(f"{OUT}/{SPLIT_JSON[split](ds)[0]}", encoding="utf-8"))
              for ds in DATASETS for split in ("val", "test")}
    rows = []
    for ds in DATASETS:
        z, borders = scaled(ds)
        for h in HORIZONS:
            f = {s: features(z[slice(*borders[s])], h) for s in borders}
            coefs = {k: fit(f["train"], cols) for k, cols in SETS.items()}
            row = {"dataset": ds, "horizon": h, "coef_on_m_s1": [float(c[1]) for c in coefs["s1"]],
                   "mean_m": {s: f[s]["m"].mean(axis=0).tolist() for s in borders},
                   "mean_dy": {s: f[s]["dy"].mean(axis=0).tolist() for s in borders}}
            for split in ("val", "test"):
                g = f[split]
                method = SPLIT_JSON[split](ds)[1]
                runs = sealed[(ds, split)][str(h)]["runs"][method]
                it_var = np.mean([((load_pred(r).mean(axis=1) - g["y_mean"]) ** 2).mean(axis=0) for r in runs], axis=0)
                res = {"itransformer_level_by_variable": it_var.tolist()}
                for k, cols in SETS.items():
                    err = ((predict(coefs[k], g, cols) - g["dy"]) ** 2).mean(axis=0)
                    res[f"delta_mse_{k}"] = float(err.mean() - it_var.mean())
                    res[f"delta_mse_{k}_by_variable"] = (err - it_var).tolist()
                row[split] = res
            rows.append(row)
            print(f"{ds}-{h}: test S1 {row['test']['delta_mse_s1']:+.4f} no_m {row['test']['delta_mse_no_m']:+.4f} "
                  f"only_m {row['test']['delta_mse_only_m']:+.4f}", flush=True)
    os.makedirs(out_dir)
    with open(f"{out_dir}/level_predictor_squeeze.json", "w") as fh:
        names = ["m", "last - m", "last24 - m", "slope"]
        json.dump({"rows": rows, "variables": VARS,
                   "features": {k: ["intercept"] + [names[c] for c in cols] for k, cols in SETS.items()}}, fh, indent=2)
    lines = ["EXP-0012 squeeze（探索性，在计划 v1 的判定之后做，不改变判定）：ΔMSE = 预测器的水平误差 − iTransformer 的水平误差"
             "（3 个种子平均）。S1 去掉 m：只用截距、最后值 − m、末 24 步均值 − m、斜率；只用 m：截距与 m。", "",
             "| 数据集-预测长度 | 测试 ΔMSE：S1 / 去掉 m / 只用 m | 验证 ΔMSE：S1 / 去掉 m / 只用 m | S1 对 m 的系数（7 个变量的范围） |",
             "| --- | --- | --- | --- |"]
    for r in rows:
        t, v = r["test"], r["val"]
        lines.append(f"| {r['dataset']}-{r['horizon']} | {t['delta_mse_s1']:+.4f} / {t['delta_mse_no_m']:+.4f} / {t['delta_mse_only_m']:+.4f} | "
                     f"{v['delta_mse_s1']:+.4f} / {v['delta_mse_no_m']:+.4f} / {v['delta_mse_only_m']:+.4f} | "
                     f"{min(r['coef_on_m_s1']):+.3f} ～ {max(r['coef_on_m_s1']):+.3f} |")
    lines += ["", "| 数据集-预测长度：测试 ΔMSE（S1）按变量 | " + " | ".join(VARS) + " |", "| --- |" + " --- |" * len(VARS)]
    for r in rows:
        lines.append(f"| {r['dataset']}-{r['horizon']} | " + " | ".join(f"{x:+.3f}" for x in r["test"]["delta_mse_s1_by_variable"]) + " |")
    lines += ["", "| 数据集：输入段均值 m 的平均（预测长度 96 的窗口；训练期标准化后） | 时期 | " + " | ".join(VARS) + " |",
              "| --- | --- |" + " --- |" * len(VARS)]
    for r in rows:
        if r["horizon"] == 96:
            for s in ("train", "val", "test"):
                lines.append(f"| {r['dataset']} | {s} | " + " | ".join(f"{x:+.2f}" for x in r["mean_m"][s]) + " |")
    lines += ["", "| 数据集：S1 对 m 的系数（预测长度 96 / 720） | " + " | ".join(VARS) + " |", "| --- |" + " --- |" * len(VARS)]
    for ds in DATASETS:
        a = next(r for r in rows if r["dataset"] == ds and r["horizon"] == 96)["coef_on_m_s1"]
        b = next(r for r in rows if r["dataset"] == ds and r["horizon"] == 720)["coef_on_m_s1"]
        lines.append(f"| {ds} | " + " | ".join(f"{x:+.2f} / {y:+.2f}" for x, y in zip(a, b)) + " |")
    with open(f"{out_dir}/level_predictor_squeeze.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
