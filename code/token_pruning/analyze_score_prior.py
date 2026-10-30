"""FastV's score curve over patch positions on blank vs real images (EXP-0008), and the prior for the corrected runs.

  python analyze_score_prior.py <EXP> <blank run> <real run>   # writes code/output/<EXP>/analysis/{position_prior.npy,
                                                                #   score_curve.md, score_curve.json}

Each run folder holds scores.npy (prefills x 576: FastV's attention score of every image patch, from the last
prompt token at layer K-1, head-averaged, before pruning). The prior is the mean score per patch over the blank-image
prefills. Reported: the per-row mean of the blank and real curves, how much of the real per-patch mean the blank curve
explains (Pearson r over the 576 positions), and the share of the total score mass in the bottom four rows.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GRID = 24


def main():
    exp, blank_run, real_run = sys.argv[1:4]
    out = os.path.join(REPO, "code", "output", exp)
    blank = np.load(f"{out}/{blank_run}/scores.npy")
    real = np.load(f"{out}/{real_run}/scores.npy")
    prior = blank.mean(axis=0)
    # 2026-10-30 note: no overwrite check; rerunning rewrites the sealed EXP-0008/analysis, including position_prior.npy,
    # which every K = 2 prior config reads (README); do not rerun
    os.makedirs(f"{out}/analysis", exist_ok=True)
    np.save(f"{out}/analysis/position_prior.npy", prior.astype(np.float32))
    real_mean = real.mean(axis=0)
    per_question_r = [float(np.corrcoef(prior, s)[0, 1]) for s in real]
    summary = {
        "blank_run": blank_run, "real_run": real_run, "n_blank": int(blank.shape[0]), "n_real": int(real.shape[0]),
        "pearson_r_prior_vs_real_mean": float(np.corrcoef(prior, real_mean)[0, 1]),
        "pearson_r_prior_vs_real_per_question_median": float(np.median(per_question_r)),
        "bottom4_rows_share_blank": float(prior.reshape(GRID, GRID)[20:].sum() / prior.sum()),
        "bottom4_rows_share_real": float(real_mean.reshape(GRID, GRID)[20:].sum() / real_mean.sum()),
        "last_patch_share_blank": float(prior[-1] / prior.sum()),
        "last_patch_share_real": float(real_mean[-1] / real_mean.sum()),
        "row_mean_blank": prior.reshape(GRID, GRID).mean(axis=1).tolist(),
        "row_mean_real": real_mean.reshape(GRID, GRID).mean(axis=1).tolist(),
        "blank_total_image_attention": float(blank.sum(axis=1).mean()),
        "real_total_image_attention": float(real.sum(axis=1).mean()),
    }
    with open(f"{out}/analysis/score_curve.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"| 量 | 空白图像（{blank_run}） | 真实图像（{real_run}） |", "| --- | --- | --- |",
             f"| 预填充次数 | {summary['n_blank']} | {summary['n_real']} |",
             f"| 图像 token 得到的注意力总和（最后一个 token，head 平均） | {summary['blank_total_image_attention']:.4f} | "
             f"{summary['real_total_image_attention']:.4f} |",
             f"| 底部 4 行占图像注意力的比例（均匀 16.7%） | {100 * summary['bottom4_rows_share_blank']:.1f}% | "
             f"{100 * summary['bottom4_rows_share_real']:.1f}% |",
             f"| 最后一个 patch 的占比（均匀 0.17%） | {100 * summary['last_patch_share_blank']:.2f}% | "
             f"{100 * summary['last_patch_share_real']:.2f}% |",
             f"| 与空白曲线的 Pearson r（576 个位置） | — | 平均分数：{summary['pearson_r_prior_vs_real_mean']:.3f}；"
             f"逐题中位数：{summary['pearson_r_prior_vs_real_per_question_median']:.3f} |",
             "", "| 行 | " + " | ".join(str(i) for i in range(GRID)) + " |", "| --- | " + " | ".join("---" for _ in range(GRID)) + " |"]
    for name, row in (("空白（相对均匀）", summary["row_mean_blank"]), ("真实（相对均匀）", summary["row_mean_real"])):
        uniform = sum(row) / GRID
        lines.append(f"| {name} | " + " | ".join(f"{x / uniform:.2f}" for x in row) + " |")
    with open(f"{out}/analysis/score_curve.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
