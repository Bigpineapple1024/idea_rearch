"""Where on the image do FastV and random selection keep their tokens? (EXP-0006, kept-patch diagnostic)

  python analyze_kept_tokens.py <EXP> <run_id> [<run_id> ...]    # writes code/output/<EXP>/analysis/kept_{stats.md,summary.json}

Each run folder holds kept_tokens.json (one list of kept patch indices per prefill, row-major on LLaVA-1.5's 24x24
grid) and answers.jsonl, in the order of the pope_img20 question file; the labels file from make_pope_subset.py gives
image, split, label and object per question. Statistics per run:
- position: mean row and column of the kept patches (uniform: 11.5), share in the bottom quarter (rows 18-23, 25%);
- spread: share of the 36 blocks of 4x4 patches holding a kept patch, and the mean nearest-neighbour distance;
- fixed positions: how often each patch index is kept over all questions, the top indices and the normalised entropy;
- question dependence: Jaccard overlap of kept sets for two different questions on the same image vs two images;
- answers: accuracy and yes ratio on the subset, and the spread statistics split by correct / wrong positives.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from bootstrap_ci import pope_no  # noqa: E402

GRID = 24
LABELS = "D:/6CJNM007/data/llava_eval/pope/llava_pope_test_img20_labels.jsonl"


def spread(idx):
    rows, cols = np.divmod(np.asarray(idx), GRID)
    blocks = len(set(zip(rows // 4, cols // 4))) / 36
    pts = np.stack([rows, cols], 1).astype(float)
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return blocks, float(d.min(1).mean())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def analyse(out, run_id, info):
    with open(f"{out}/{run_id}/kept_tokens.json") as f:
        kept = json.load(f)
    with open(f"{out}/{run_id}/answers.jsonl") as f:
        answers = [json.loads(line) for line in f]
    if len(kept["kept"]) != len(info) or len(answers) != len(info):
        raise SystemExit(f"{run_id}: {len(kept['kept'])} kept sets, {len(answers)} answers, {len(info)} questions")
    sets = kept["kept"]
    allidx = np.concatenate([np.asarray(s) for s in sets])
    rows, cols = np.divmod(allidx, GRID)
    freq = np.bincount(allidx, minlength=GRID * GRID) / len(sets)
    p = freq / freq.sum()
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(GRID * GRID))
    top = np.argsort(-freq)[:10]
    spreads = np.array([spread(s) for s in sets])
    same, other = [], []
    for i, j in itertools.combinations(range(len(sets)), 2):
        if info[i]["image"] == info[j]["image"]:
            if answers[i]["prompt"] != answers[j]["prompt"]:        # positive questions repeat verbatim across splits
                same.append(jaccard(sets[i], sets[j]))
        else:
            other.append(jaccard(sets[i], sets[j]))
    said_yes = np.array([not pope_no(a["text"]) for a in answers])
    is_yes = np.array([q["label"] == "yes" for q in info])
    correct = said_yes == is_yes
    pos = is_yes
    return {
        "run_id": run_id, "keep": kept["keep"], "n_questions": len(sets),
        "mean_row": float(rows.mean()), "mean_col": float(cols.mean()), "bottom_quarter_share": float((rows >= 18).mean()),
        "block_coverage_mean": float(spreads[:, 0].mean()), "nn_distance_mean": float(spreads[:, 1].mean()),
        "entropy_normalised": entropy,
        "top_patches": [{"index": int(t), "row": int(t // GRID), "col": int(t % GRID), "share_of_questions": float(freq[t])}
                        for t in top],
        "jaccard_same_image_mean": float(np.mean(same)) if same else None, "n_same_image_pairs": len(same),
        "jaccard_other_image_mean": float(np.mean(other)), "n_other_image_pairs": len(other),
        "accuracy": float(correct.mean()), "yes_ratio": float(said_yes.mean()),
        "positives_correct": int((correct & pos).sum()), "positives_wrong": int((~correct & pos).sum()),
        "block_coverage_correct_positives": float(spreads[correct & pos, 0].mean()) if (correct & pos).any() else None,
        "block_coverage_wrong_positives": float(spreads[~correct & pos, 0].mean()) if (~correct & pos).any() else None,
        "row_histogram": np.bincount(rows, minlength=GRID).tolist(),
    }


def main():
    exp, run_ids = sys.argv[1], sys.argv[2:]
    out = os.path.join(REPO, "code", "output", exp)
    with open(LABELS) as f:
        info = [json.loads(line) for line in f]
    stats = [analyse(out, r, info) for r in run_ids]
    os.makedirs(f"{out}/analysis", exist_ok=True)
    with open(f"{out}/analysis/kept_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    lines = ["| 运行 | keep | 平均行 / 列（均匀 11.5） | 底部 1/4 占比（均匀 25%） | 4×4 块覆盖率 | 最近邻距离 | 归一化熵 | "
             "同图不同题 Jaccard | 不同图 Jaccard | 子集准确率 / yes 比例 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in stats:
        lines.append(f"| {s['run_id']} | {s['keep']} | {s['mean_row']:.2f} / {s['mean_col']:.2f} | "
                     f"{100 * s['bottom_quarter_share']:.1f}% | {100 * s['block_coverage_mean']:.1f}% | "
                     f"{s['nn_distance_mean']:.2f} | {s['entropy_normalised']:.3f} | "
                     f"{s['jaccard_same_image_mean']:.3f}（{s['n_same_image_pairs']} 对） | "
                     f"{s['jaccard_other_image_mean']:.3f} | {100 * s['accuracy']:.1f}% / {100 * s['yes_ratio']:.1f}% |")
    lines += ["", "| 运行 | 最常被保留的 patch（行, 列：被保留的题目比例） | 正样本答对 / 答错的 4×4 块覆盖率 |", "| --- | --- | --- |"]
    for s in stats:
        tops = "；".join(f"({t['row']}, {t['col']}): {100 * t['share_of_questions']:.0f}%" for t in s["top_patches"][:6])
        cov = (f"{100 * s['block_coverage_correct_positives']:.1f}% / {100 * s['block_coverage_wrong_positives']:.1f}%"
               if s["block_coverage_correct_positives"] is not None and s["block_coverage_wrong_positives"] is not None
               else "NA")
        lines.append(f"| {s['run_id']} | {tops} | {cov}（{s['positives_correct']} / {s['positives_wrong']} 题） |")
    with open(f"{out}/analysis/kept_stats.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
