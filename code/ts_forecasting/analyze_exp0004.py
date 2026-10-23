"""EXP-0004 tables: iTransformer against Table 10 of the paper, per dataset and prediction length.

  python analyze_exp0004.py     # writes code/output/EXP-0004/analysis/{tables.md,summary.json}

Reads runs.tsv. Per dataset x prediction length: the direct run.py runs (seed 2023) and the wrapper runs; whether the
wrapper with seed 2023 equals the direct runs; mean and sample SD over the wrapper seeds 2021/2022/2023; and the
reproduction criterion of the plan (the paper's MSE inside mean +- 2 SD). The paper rounds to three decimals, so
the table also says whether the rounding interval [value - 0.0005, value + 0.0005] overlaps mean +- 2 SD.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output", "EXP-0004")
# PAPER-0007 E-IT-3: iTransformer paper (arXiv 2310.06625v4) Table 10, look-back 96, MSE / MAE
PAPER = {"ETTh1": {96: (0.386, 0.405), 192: (0.441, 0.436), 336: (0.487, 0.458), 720: (0.503, 0.491)},
         "ETTh2": {96: (0.297, 0.349), 192: (0.380, 0.400), 336: (0.428, 0.432), 720: (0.427, 0.445)},
         "ETTm1": {96: (0.334, 0.368), 192: (0.377, 0.391), 336: (0.426, 0.420), 720: (0.491, 0.459)},
         "ETTm2": {96: (0.180, 0.264), 192: (0.250, 0.309), 336: (0.311, 0.348), 720: (0.412, 0.407)}}


def main():
    with open(os.path.join(OUT, "runs.tsv")) as f:
        runs = [r for r in csv.DictReader(f, delimiter="\t") if r["status"] == "completed"]
    summary = []
    for ds in PAPER:
        for pl in PAPER[ds]:
            group = [r for r in runs if r["dataset"] == ds and int(r["pred_len"]) == pl]
            if not group:
                continue
            direct = [float(r["mse"]) for r in group if r["entry"] == "direct"]
            wrapper = {int(r["seed"]): (float(r["mse"]), float(r["mae"])) for r in group if r["entry"] == "wrapper"}
            seeds = sorted(wrapper)
            mse = np.array([wrapper[s][0] for s in seeds])
            mae = np.array([wrapper[s][1] for s in seeds])
            row = {"dataset": ds, "pred_len": pl, "runs": [r["run_id"] for r in group], "direct_mse": direct,
                   "wrapper_seeds": seeds, "wrapper_mse": mse.tolist(), "wrapper_mae": mae.tolist(),
                   "paper_mse": PAPER[ds][pl][0], "paper_mae": PAPER[ds][pl][1]}
            if 2023 in wrapper and direct:
                row["wrapper_2023_equals_direct"] = all(abs(wrapper[2023][0] - d) == 0 for d in direct)
            if len(seeds) >= 2:
                m, sd = float(mse.mean()), float(mse.std(ddof=1))
                p = PAPER[ds][pl][0]
                row.update({"mse_mean": m, "mse_sd": sd, "mae_mean": float(mae.mean()), "mae_sd": float(mae.std(ddof=1)),
                            "pass": abs(p - m) <= 2 * sd, "pass_with_rounding": p - 0.0005 <= m + 2 * sd and p + 0.0005 >= m - 2 * sd})
            summary.append(row)
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    with open(os.path.join(OUT, "analysis", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["| 数据集-预测长度 | 直接运行 run.py（种子 2023） | 包装脚本 2023 是否相同 | 包装脚本各种子 MSE（种子） | 均值 ± 标准差 | 论文 MSE | "
             "是否在 ±2 SD 内 | 计入论文舍入 | MAE 均值 ± 标准差 / 论文 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary:
        fmt = lambda xs: " / ".join(f"{x:.4f}" for x in xs)
        lines.append(f"| {r['dataset']}-{r['pred_len']} | {fmt(r['direct_mse']) or 'NA'} | "
                     f"{'是' if r.get('wrapper_2023_equals_direct') else ('否' if 'wrapper_2023_equals_direct' in r else 'NA')} | "
                     f"{fmt(r['wrapper_mse'])}（{', '.join(map(str, r['wrapper_seeds']))}） | "
                     + (f"{r['mse_mean']:.4f} ± {r['mse_sd']:.4f} | {r['paper_mse']:.3f} | {'是' if r['pass'] else '否'} | "
                        f"{'是' if r['pass_with_rounding'] else '否'} | {r['mae_mean']:.4f} ± {r['mae_sd']:.4f} / {r['paper_mae']:.3f} |"
                        if "mse_mean" in r else f"NA | {r['paper_mse']:.3f} | NA | NA | NA |"))
    with open(os.path.join(OUT, "analysis", "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
