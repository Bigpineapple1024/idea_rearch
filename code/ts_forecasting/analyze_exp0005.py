"""EXP-0005 tables: DUET, TQNet and WPMixer on ETTh1 under one protocol, next to iTransformer (EXP-0004 phase A).

  python analyze_exp0005.py     # writes code/output/EXP-0005/analysis/{tables.md,summary.json}

Reads both runs.tsv files and the WPMixer metrics.json (drop_last scoring). Per method x prediction length: the MSE
of each seed (all test windows), mean and sample SD, MAE mean, and for WPMixer the repository's drop_last scoring with
the difference. Reference values are the authors' (official result.txt, README figures or paper tables) and this
project's CPU runs (EXP-0003); they are listed for orientation, not as pass/fail gates (EXP-0005 has none).
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))   # run from elsewhere: set the repo
OUT = os.path.join(REPO, "code", "output")
# authors' values, ETTh1, look-back 96, MSE
AUTHOR = {"tqnet": {96: 0.3712, 192: 0.4284, 336: 0.4757, 720: 0.4874},       # repository result.txt, seed 2024
          "wpmixer": {96: 0.368},                                             # unified setting, README Table 9 (drop_last)
          "duet": {96: 0.377},                                                # unified setting, README figure
          "itransformer": {96: 0.386, 192: 0.441, 336: 0.487, 720: 0.503}}    # paper Table 10
AUTHOR_AVG = {"tqnet": 0.441, "wpmixer": 0.422}                               # paper averages over the four lengths
# EXP-0003 (CPU): all-window MSE; TQNet seed 2024, WPMixer seed 42, DUET seeds 2021/2022
CPU = {"tqnet": {96: 0.3703, 192: 0.4282, 336: 0.4754, 720: 0.4849}, "wpmixer": {96: 0.3727, 336: 0.4735, 720: 0.4615},
       "duet": {96: "0.3898 / 0.3790"}}


def runs(exp):
    with open(os.path.join(OUT, exp, "runs.tsv")) as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["status"] == "completed"]


def main():
    rows = [r for r in runs("EXP-0004") if r["dataset"] == "ETTh1" and r["entry"] == "wrapper"] + runs("EXP-0005")
    summary, lines = [], ["| 方法 | 预测长度 | 各种子 MSE（2021 / 2022 / 2023，全部窗口） | 均值 ± 标准差 | MAE 均值 | 作者报告值 | 本项目 CPU（EXP-0003） | WPMixer drop_last 计分均值（差） |",
                          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for method in ("itransformer", "tqnet", "wpmixer", "duet"):
        for pl in (96, 192, 336, 720):
            group = sorted((r for r in rows if r["method"] == method and int(r["pred_len"]) == pl), key=lambda r: int(r["seed"]))
            if not group:
                continue
            mse = np.array([float(r["mse"]) for r in group])
            mae = np.array([float(r["mae"]) for r in group])
            entry = {"method": method, "pred_len": pl, "runs": [r["run_id"] for r in group], "seeds": [int(r["seed"]) for r in group],
                     "mse": mse.tolist(), "mse_mean": float(mse.mean()), "mse_sd": float(mse.std(ddof=1)) if len(mse) > 1 else None,
                     "mae_mean": float(mae.mean()), "author": AUTHOR.get(method, {}).get(pl), "cpu": CPU.get(method, {}).get(pl)}
            if method == "wpmixer":
                dl = []
                for r in group:
                    with open(os.path.join(OUT, "EXP-0005", r["run_id"], "metrics.json")) as f:
                        dl.append(json.load(f)["drop_last_mse"])
                entry["drop_last_mse"] = dl
                entry["drop_last_mean"] = float(np.mean(dl))
            summary.append(entry)
            sd = f" ± {entry['mse_sd']:.4f}" if entry["mse_sd"] is not None else ""
            dl_text = (f"{entry['drop_last_mean']:.4f}（{entry['drop_last_mean'] - entry['mse_mean']:+.4f}）"
                       if "drop_last_mean" in entry else "—")
            lines.append(f"| {method} | {pl} | {' / '.join(f'{x:.4f}' for x in mse)} | {entry['mse_mean']:.4f}{sd} | "
                         f"{entry['mae_mean']:.4f} | {entry['author'] if entry['author'] is not None else '—'} | "
                         f"{entry['cpu'] if entry['cpu'] is not None else '—'} | {dl_text} |")
    lines += ["", "| 方法 | 四个预测长度的平均 MSE（三种子均值） | 作者报告的平均 |", "| --- | --- | --- |"]
    for method in ("itransformer", "tqnet", "wpmixer"):
        means = [e["mse_mean"] for e in summary if e["method"] == method]
        if len(means) == 4:
            extra = ""
            if method == "wpmixer":
                extra = f"（drop_last 计分 {np.mean([e['drop_last_mean'] for e in summary if e['method'] == method]):.4f}）"
            lines.append(f"| {method} | {np.mean(means):.4f}{extra} | {AUTHOR_AVG.get(method, '论文 Table 10 均值 0.454')} |")
    os.makedirs(os.path.join(OUT, "EXP-0005", "analysis"), exist_ok=True)
    with open(os.path.join(OUT, "EXP-0005", "analysis", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "EXP-0005", "analysis", "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
