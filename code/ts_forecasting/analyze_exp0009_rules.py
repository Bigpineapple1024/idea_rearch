"""EXP-0009: apply the plan's interpretation rules (v1, v1.1) to the level/shape splits.

  python analyze_exp0009_rules.py      # writes code/output/EXP-0009/analysis/{rules.md,rules.json}

Validation period: code/output/EXP-0009/deepdive/level_shape_<ds>_val.json (method itransformer_val).
Test period: code/output/EXP-0005/deepdive/level_shape_ETTh1.json and code/output/EXP-0004/deepdive/level_shape_<ds>.json
(method itransformer). Both pool the three seeds. The shrink factor a scales the predicted level change; shape errors
do not depend on it, so the test MSE change at a equals the test level-error change. Rules (plan v1):
  1. validation 336/720 with a=0 below a=1: the learned level change is harmful beyond the test period (candidate);
  2. validation a=1 better but test a=0 better: harm comes from the test period's shift (phenomenon only);
  3. the validation-chosen a lowers test MSE by more than 0.005: worth step-7 ranking.
Generality (v1.1): at 336 or 720 on at least one dataset other than ETTh1, a=0 below a=1 on validation and test.
"""
import json
import os
import sys

OUT = "D:/6CJNM007/idea_rearch/code/output"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
HORIZONS = ("96", "192", "336", "720")
FACTORS = ("0.0", "0.25", "0.5", "0.75", "1.0")


def load(path, method):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {h: {"shrink": {a: d[h]["methods"][method]["level_error_by_shrink_factor"][a] for a in FACTORS},
                "mse": d[h]["methods"][method]["mse"], "level_share": d[h]["methods"][method]["level_share"],
                "n_runs": d[h]["methods"][method].get("n_runs")} for h in HORIZONS}


def main():
    out_dir = f"{OUT}/EXP-0009/analysis"
    if os.path.exists(f"{out_dir}/rules.json"):
        sys.exit("rules.json exists; results are never overwritten")
    rows, per = [], {}
    for ds in DATASETS:
        val = load(f"{OUT}/EXP-0009/deepdive/level_shape_{ds}_val.json", "itransformer_val")
        test_path = (f"{OUT}/EXP-0005/deepdive/level_shape_ETTh1.json" if ds == "ETTh1"
                     else f"{OUT}/EXP-0004/deepdive/level_shape_{ds}.json")
        test = load(test_path, "itransformer")
        for h in HORIZONS:
            v, t = val[h]["shrink"], test[h]["shrink"]
            a_val = min(FACTORS, key=lambda a: v[a])
            a_test = min(FACTORS, key=lambda a: t[a])
            delta = t[a_val] - t["1.0"]
            r = {"dataset": ds, "horizon": int(h), "val_a0": v["0.0"], "val_a1": v["1.0"], "test_a0": t["0.0"],
                 "test_a1": t["1.0"], "a_val": float(a_val), "a_test": float(a_test), "test_mse_change_at_a_val": delta,
                 "val_mse": val[h]["mse"], "test_mse": test[h]["mse"], "val_level_share": val[h]["level_share"],
                 "test_level_share": test[h]["level_share"], "val_runs": val[h]["n_runs"], "test_runs": test[h]["n_runs"],
                 "rule1": int(h) >= 336 and v["0.0"] < v["1.0"],
                 "rule2": v["1.0"] < v["0.0"] and t["0.0"] < t["1.0"],
                 "rule3": delta < -0.005,
                 "no_change_better_both": v["0.0"] < v["1.0"] and t["0.0"] < t["1.0"]}
            rows.append(r)
    general = [r for r in rows if r["dataset"] != "ETTh1" and r["horizon"] >= 336 and r["no_change_better_both"]]
    summary = {"rows": rows, "generality_cases": [(r["dataset"], r["horizon"]) for r in general],
               "generality_holds": bool(general), "test_sources": {"ETTh1": "EXP-0005/deepdive/level_shape_ETTh1.json",
                                                                   "others": "EXP-0004/deepdive/level_shape_<ds>.json"}}
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/rules.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["| 数据集-预测长度 | 验证期水平误差 a=0 / a=1 | 测试期水平误差 a=0 / a=1 | 验证期选的 a | 测试期最好的 a | "
             "测试 MSE 变化（用验证期的 a） | 规则 1 | 规则 2 | 规则 3 | a=0 在两期都更好 | 水平占比 验证 / 测试 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['dataset']}-{r['horizon']} | {r['val_a0']:.4f} / {r['val_a1']:.4f} | {r['test_a0']:.4f} / "
                     f"{r['test_a1']:.4f} | {r['a_val']:g} | {r['a_test']:g} | {r['test_mse_change_at_a_val']:+.4f} | "
                     f"{'是' if r['rule1'] else '—'} | {'是' if r['rule2'] else '—'} | {'是' if r['rule3'] else '—'} | "
                     f"{'是' if r['no_change_better_both'] else '—'} | {100 * r['val_level_share']:.0f}% / "
                     f"{100 * r['test_level_share']:.0f}% |")
    lines += ["", "普遍性（v1.1，ETTh1 以外、336 或 720、两期都是 a=0 更好）：" +
              ("成立：" + "、".join(f"{d}-{h}" for d, h in summary["generality_cases"]) if general else "不成立")]
    with open(f"{out_dir}/rules.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
