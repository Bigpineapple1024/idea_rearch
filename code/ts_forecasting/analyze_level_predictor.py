"""EXP-0012 (plan v1): can a simple predictor fit on the training windows predict the future window's level better than
iTransformer's own level? The CPU check of time-series candidate 1 ("level path").

  python analyze_level_predictor.py      # writes code/output/EXP-0012/analysis/level_predictor.{json,md}

Windows, scaling and the level/shape split are those of analyze_level_shape.py (EXP-0004/0005/0009 deepdive): TSLib's
ETT borders, training mean and population SD, stride 1, 96 input steps. A window's level is each variable's mean over the
H target steps; the level error is the mean over windows and variables of (predicted level - true level)^2, and the MSE
is the level error plus the shape error exactly. Predictors of the change dy = mean Y - mean X, fit by least squares on
the training windows of each dataset and horizon:
  S1 (primary): per variable, on the same variable's input mean m, last value - m, mean of the last 24 steps - m and the
      least-squares slope over the 96 input steps, with an intercept;
  S2 (secondary): per target variable, the same four statistics of all 7 variables (28 features and an intercept);
  S0: no change (the input mean).
iTransformer's test predictions are the EXP-0004 runs the sealed level/shape splits used; its validation predictions are
the EXP-0009 itransformer_val runs, which load the same checkpoints. Replacing iTransformer's level by S1's (shape kept)
changes the MSE by S1's level error minus iTransformer's. The recomputed iTransformer level error, MSE and correlation and
S0's level error must equal the sealed JSONs (relative difference < 1e-6), or the script stops.
Decision (S1 only): a dataset-horizon is clearly better when the test change is below -0.005, the validation change is
below 0 and S1's correlation beats iTransformer's in both periods. High: clear cases in at least two datasets. Low: no
dataset-horizon has a test change below -0.005 together with a validation change below 0. Otherwise medium.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = "D:/6CJNM007/idea_rearch/code/output"
DATA = "D:/6CJNM007/data/ETT-small"
SEQ = 96
LAST = 24
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
HORIZONS = (96, 192, 336, 720)
WEIGHTS = [round(0.1 * k, 1) for k in range(11)]
THRESHOLD = 0.005
TOL = 1e-6
SLOPE_W = (np.arange(SEQ) - (SEQ - 1) / 2) / (SEQ * (SEQ ** 2 - 1) / 12)     # least-squares slope weights
SPLIT_JSON = {"test": lambda ds: (f"EXP-0005/deepdive/level_shape_{ds}.json" if ds == "ETTh1"
                                  else f"EXP-0004/deepdive/level_shape_{ds}.json", "itransformer"),
              "val": lambda ds: (f"EXP-0009/deepdive/level_shape_{ds}_val.json", "itransformer_val")}


def scaled(ds):
    df = pd.read_csv(f"{DATA}/{ds}.csv").drop(columns="date").to_numpy(dtype=np.float64)
    month = 30 * 24 * (4 if ds.startswith("ETTm") else 1)
    train_end = 12 * month
    mu, sd = df[:train_end].mean(axis=0), df[:train_end].std(axis=0)
    borders = {"train": (0, train_end), "val": (train_end - SEQ, 16 * month), "test": (16 * month - SEQ, 20 * month)}
    return (df - mu) / sd, borders


def features(seg, h):
    """Input statistics, true change and true level of every window of a segment (windows, variables)."""
    n = len(seg) - SEQ - h + 1
    c = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
    i = np.arange(n)
    m = (c[i + SEQ] - c[i]) / SEQ
    last = seg[i + SEQ - 1]
    last24 = (c[i + SEQ] - c[i + SEQ - LAST]) / LAST
    slope = np.lib.stride_tricks.sliding_window_view(seg, SEQ, axis=0)[:n] @ SLOPE_W
    y_mean = (c[i + SEQ + h] - c[i + SEQ]) / h
    return {"m": m, "stats": np.stack([m, last - m, last24 - m, slope], axis=2), "dy": y_mean - m, "y_mean": y_mean}


def targets(seg, h):
    n = len(seg) - SEQ - h + 1
    return np.lib.stride_tricks.sliding_window_view(seg[SEQ:], h, axis=0)[:n].transpose(0, 2, 1)   # (windows, h, vars)


def fit_s1(f):
    coef = []
    for v in range(f["dy"].shape[1]):
        a = np.column_stack([np.ones(len(f["dy"])), f["stats"][:, v]])
        coef.append(np.linalg.lstsq(a, f["dy"][:, v], rcond=None)[0])
    return np.array(coef)                                                               # (vars, 5)


def predict_s1(coef, f):
    return np.stack([coef[v, 0] + f["stats"][:, v] @ coef[v, 1:] for v in range(len(coef))], axis=1)


def design_s2(f):
    return np.column_stack([np.ones(len(f["dy"])), f["stats"].reshape(len(f["dy"]), -1)])


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def run_dir(run):
    exp = run.rsplit("-R", 1)[0]
    return f"{OUT}/{exp}/{run}"


def load_pred(run):
    paths = glob.glob(f"{run_dir(run)}/results/*/pred.npy")
    if len(paths) != 1:
        sys.exit(f"{run}: expected one pred.npy, found {paths}")
    return np.load(paths[0]).astype(np.float64)


def check(name, ours, sealed):
    if not np.isclose(ours, sealed, rtol=TOL, atol=0):
        sys.exit(f"stop: {name} recomputed {ours!r} differs from the sealed {sealed!r}")


def main():
    out_dir = f"{OUT}/EXP-0012/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sealed = {(ds, split): json.load(open(f"{OUT}/{SPLIT_JSON[split](ds)[0]}", encoding="utf-8"))
              for ds in DATASETS for split in ("val", "test")}
    rows = []
    for ds in DATASETS:
        z, borders = scaled(ds)
        for h in HORIZONS:
            f = {s: features(z[slice(*borders[s])], h) for s in borders}
            coef = fit_s1(f["train"])
            coef2 = np.linalg.lstsq(design_s2(f["train"]), f["train"]["dy"], rcond=None)[0]
            row = {"dataset": ds, "horizon": h, "n_windows": {s: int(len(f[s]["dy"])) for s in borders},
                   "s1_coef": coef.tolist(), "train": {
                       "level_s0": float((f["train"]["dy"] ** 2).mean()),
                       "level_s1": float(((predict_s1(coef, f["train"]) - f["train"]["dy"]) ** 2).mean()),
                       "level_s2": float(((design_s2(f["train"]) @ coef2 - f["train"]["dy"]) ** 2).mean())}}
            seeds = {}
            for split in ("val", "test"):
                g = f[split]
                s1 = predict_s1(coef, g)
                s2 = design_s2(g) @ coef2
                d = sealed[(ds, split)][str(h)]
                method = SPLIT_JSON[split](ds)[1]
                runs = d["runs"][method]
                if d["n_windows"] != len(g["dy"]):
                    sys.exit(f"stop: {ds}-{h} {split}: {len(g['dy'])} windows, sealed {d['n_windows']}")
                y = targets(z[slice(*borders[split])], h)
                per_seed = []
                for run in runs:
                    cfg = json.load(open(f"{run_dir(run)}/config.json"))
                    pred = load_pred(run)
                    if pred.shape != y.shape:
                        sys.exit(f"stop: {run} pred {pred.shape} vs windows {y.shape}")
                    p_mean = pred.mean(axis=1)
                    level = float(((p_mean - g["y_mean"]) ** 2).mean())
                    mse = float(((pred - y) ** 2).mean())
                    blend = [float(((p_mean + w * (g["m"] + s1 - p_mean) - g["y_mean"]) ** 2).mean()) for w in WEIGHTS]
                    item = {"run": run, "seed": cfg["seed"], "level": level, "mse": mse, "blend": blend,
                            "corr": corr(p_mean - g["m"], g["dy"])}
                    if split == "val":
                        item["source_run"] = open(f"{run_dir(run)}/source_run.txt").readline().strip()
                    per_seed.append(item)
                sd = sealed[(ds, split)][str(h)]["methods"][method]
                check(f"{ds}-{h} {split} iTransformer level", np.mean([p["level"] for p in per_seed]), sd["level"])
                check(f"{ds}-{h} {split} iTransformer MSE", np.mean([p["mse"] for p in per_seed]), sd["mse"])
                check(f"{ds}-{h} {split} iTransformer corr", np.mean([p["corr"] for p in per_seed]),
                      sd["corr_pred_true_level_change"])
                level_s0 = float((g["dy"] ** 2).mean())
                check(f"{ds}-{h} {split} S0 level", level_s0, d["naive_level_error"]["input_mean"])
                level_s1 = float(((s1 - g["dy"]) ** 2).mean())
                level_s2 = float(((s2 - g["dy"]) ** 2).mean())
                it_level = float(np.mean([p["level"] for p in per_seed]))
                it_mse = float(np.mean([p["mse"] for p in per_seed]))
                row[split] = {"itransformer_runs": runs, "seeds": [p["seed"] for p in per_seed],
                              "level_s0": level_s0, "level_s1": level_s1, "level_s2": level_s2,
                              "level_itransformer": it_level, "mse_itransformer": it_mse,
                              "corr_s1": corr(s1, g["dy"]), "corr_s2": corr(s2, g["dy"]),
                              "corr_itransformer": float(np.mean([p["corr"] for p in per_seed])),
                              "delta_mse_s1": level_s1 - it_level, "delta_mse_s2": level_s2 - it_level,
                              "delta_mse_s1_by_seed": [level_s1 - p["level"] for p in per_seed],
                              "delta_mse_upper_bound": -it_level,
                              "blend_level": np.mean([p["blend"] for p in per_seed], axis=0).tolist()}
                seeds[split] = per_seed
            for v, t in zip(seeds["val"], seeds["test"]):
                if v["seed"] != t["seed"] or v["source_run"] != t["run"]:
                    sys.exit(f"stop: {ds}-{h}: validation run {v['run']} loads {v['source_run']}, test run is {t['run']}")
            w_star = WEIGHTS[int(np.argmin(row["val"]["blend_level"]))]
            row["blend"] = {"w_chosen_on_val": w_star,
                            "test_delta_mse": row["test"]["blend_level"][WEIGHTS.index(w_star)] - row["test"]["level_itransformer"]}
            for s in ("s1", "s2"):
                v, t = row["val"], row["test"]
                row[f"clear_{s}"] = bool(t[f"delta_mse_{s}"] < -THRESHOLD and v[f"delta_mse_{s}"] < 0
                                         and t[f"corr_{s}"] > t["corr_itransformer"] and v[f"corr_{s}"] > v["corr_itransformer"])
                row[f"gain_{s}"] = bool(t[f"delta_mse_{s}"] < -THRESHOLD and v[f"delta_mse_{s}"] < 0)
            rows.append(row)
            print(f"{ds}-{h}: S1 test {row['test']['delta_mse_s1']:+.4f} val {row['val']['delta_mse_s1']:+.4f}; "
                  f"corr S1 {row['test']['corr_s1']:.3f} / iT {row['test']['corr_itransformer']:.3f}", flush=True)

    def verdict(s):
        clear = sorted({r["dataset"] for r in rows if r[f"clear_{s}"]})
        if len(clear) >= 2:
            return "高", clear
        return ("低" if not any(r[f"gain_{s}"] for r in rows) else "中"), clear

    summary = {"rows": rows, "verdict_s1": verdict("s1"), "verdict_s2": verdict("s2"), "threshold": THRESHOLD,
               "weights": WEIGHTS, "features_s1": ["intercept", "m", "last - m", "last24 - m", "slope"]}
    os.makedirs(out_dir)
    with open(f"{out_dir}/level_predictor.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    lines = ["EXP-0012：只在训练期拟合的简单预测器 vs iTransformer 自己的水平（3 个种子平均）。ΔMSE = 把 iTransformer 的水平换成"
             "该预测器的水平（形状不变）后 MSE 的变化，等于两者水平误差之差。“明显更好”：测试期 ΔMSE < −0.005、验证期 ΔMSE < 0，"
             "且两期 S1 的相关都高于 iTransformer。", "",
             "| 数据集-预测长度 | 相关 测试：iT / S1 / S2 | 相关 验证：iT / S1 / S2 | 测试水平误差 S0 / iT / S1 / S2 | "
             "ΔMSE S1 测试（占 MSE） | ΔMSE S1 验证 | ΔMSE S2 测试 / 验证 | 上限（测试） | 明显更好 S1 / S2 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        t, v = r["test"], r["val"]
        lines.append(f"| {r['dataset']}-{r['horizon']} | {t['corr_itransformer']:.3f} / {t['corr_s1']:.3f} / {t['corr_s2']:.3f} | "
                     f"{v['corr_itransformer']:.3f} / {v['corr_s1']:.3f} / {v['corr_s2']:.3f} | "
                     f"{t['level_s0']:.4f} / {t['level_itransformer']:.4f} / {t['level_s1']:.4f} / {t['level_s2']:.4f} | "
                     f"{t['delta_mse_s1']:+.4f}（{100 * t['delta_mse_s1'] / t['mse_itransformer']:+.1f}%） | "
                     f"{v['delta_mse_s1']:+.4f} | {t['delta_mse_s2']:+.4f} / {v['delta_mse_s2']:+.4f} | "
                     f"{t['delta_mse_upper_bound']:+.4f} | {'是' if r['clear_s1'] else '—'} / {'是' if r['clear_s2'] else '—'} |")
    lines += ["", "| 数据集-预测长度 | 训练期水平误差 S0 / S1 / S2 | 混合权重 w（验证期选） | 混合后测试 ΔMSE |", "| --- | --- | --- | --- |"]
    for r in rows:
        tr = r["train"]
        lines.append(f"| {r['dataset']}-{r['horizon']} | {tr['level_s0']:.4f} / {tr['level_s1']:.4f} / {tr['level_s2']:.4f} | "
                     f"{r['blend']['w_chosen_on_val']:g} | {r['blend']['test_delta_mse']:+.4f} |")
    for s in ("s1", "s2"):
        cat, clear = summary[f"verdict_{s}"]
        lines += ["", f"判定（{s.upper()}{'，主分析' if s == 's1' else '，只作描述'}）：{cat}；明显更好的数据集："
                  + ("、".join(clear) if clear else "无")]
    with open(f"{out_dir}/level_predictor.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
