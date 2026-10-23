"""How much of iTransformer's predicted level change is the projector's constant bias? (1A4B-0002, 3a item 3)

  python analyze_projector_bias.py [<dataset>]   # default ETTh1
  # writes code/output/EXP-0004/deepdive/projector_bias_<dataset>.{md,json}

iTransformer forecasts variable v as projector(h_v) * sigma_v + mu_v, with mu_v and sigma_v the input window's mean and
population SD (+1e-5 under the root), so the predicted level change in normalised units is
c_v = mean_t projector(h_v)_t = w_bar . h_v + b_bar, where b_bar, the mean of the projector bias over the pred_len
outputs, is the same for every variable and window. c_v is recovered exactly from the saved test predictions and the
rebuilt inputs as (mean_t P_v - mu_v) / sigma_v; b_bar is read from the checkpoint. Per horizon and seed (first EXP-0004
run of each seed): b_bar, the mean and SD of c, and the level error with c replaced by b_bar alone (the constant) or by
c - b_bar (the input-dependent part), next to the model as is and c = 0 (the input mean). The shape part is unchanged.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output")
sys.path.insert(0, HERE)
from analyze_level_shape import DATASET, HORIZONS, VARS, find_runs, rebuild  # noqa: E402  (reads sys.argv[1])


def main():
    result = {}
    for h in HORIZONS:
        x, y = rebuild(h, "test")
        mu = x.mean(axis=1)                                                       # (windows, variables)
        sigma = np.sqrt(((x - mu[:, None]) ** 2).mean(axis=1) + 1e-5)
        y_mean = y.mean(axis=1)
        rows = {}
        for (method, seed), runs in sorted(find_runs(h).items()):
            if method != "itransformer":
                continue
            run_id, pred_path, _ = runs[0]
            run_dir = pred_path.split(os.sep + "results" + os.sep)[0]
            ckpt = glob.glob(os.path.join(run_dir, "checkpoints", "*", "checkpoint.pth"))[0]
            b_bar = float(torch.load(ckpt, map_location="cpu")["projector.bias"].double().mean())
            pred = np.load(pred_path).astype(np.float64)
            c = (pred.mean(axis=1) - mu) / sigma
            level = {name: float(((mu + sigma * cc - y_mean) ** 2).mean()) for name, cc in
                     (("model", c), ("constant_only", np.full_like(c, b_bar)), ("input_dependent_only", c - b_bar),
                      ("no_change", np.zeros_like(c)))}
            rows[f"seed {seed} ({run_id})"] = {"b_bar": b_bar, "c_mean": float(c.mean()), "c_sd": float(c.std()),
                                               "c_mean_by_variable": dict(zip(VARS, c.mean(axis=0).tolist())),
                                               "level_error": level}
        result[str(h)] = rows
    out_dir = os.path.join(OUT, "EXP-0004", "deepdive")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"projector_bias_{DATASET}.json"), "w") as f:
        json.dump(result, f, indent=2)
    lines = [f"{DATASET} 测试期，iTransformer：预测的水平变化（归一化单位）c = 投影头输出在预测段上的均值 = w̄·h + b̄。", "",
             "| 预测长度 | 运行 | b̄（常数） | c 的均值 | c 的标准差 | 水平误差：原样 | 只留常数 b̄ | 只留依赖输入的部分 | c = 0（输入均值） |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for h, rows in result.items():
        for name, r in rows.items():
            e = r["level_error"]
            lines.append(f"| {h} | {name} | {r['b_bar']:+.4f} | {r['c_mean']:+.4f} | {r['c_sd']:.4f} | {e['model']:.4f} | "
                         f"{e['constant_only']:.4f} | {e['input_dependent_only']:.4f} | {e['no_change']:.4f} |")
    with open(os.path.join(out_dir, f"projector_bias_{DATASET}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
