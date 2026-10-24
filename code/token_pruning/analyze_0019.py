"""EXP-0019 (plan v1): the audit's two checks on POPE. Q1: A+B (IDF-weighted layer-1 attention rows, no prior) against
A0, the strong baseline without its position prior; Q2: margin AUC of A'+B+V, A and vanilla, with false positives.

  python analyze_0019.py      # writes code/output/EXP-0019/analysis/{0019.md,0019.json}

POPE F1 differences resample images (2000 draws, seed 0) and recompute each split's F1 on all questions of the drawn
images, as the audit's cluster_boot.py does (DEC-0030). Margin = log P(Yes) - log P(No) of the first answer token; its AUC
separates positives from negatives within each split, paired differences resample images the same way.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_fp import auc   # noqa: E402
from analyze_query import row   # noqa: E402
from analyze_repope import EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, pope_no, read_labels   # noqa: E402

RUNS = {"A+B": "EXP-0014/EXP-0014-R03", "A0": "EXP-0019/EXP-0019-R01", "A": "EXP-0019/EXP-0019-R02",
        "A_EXP-0011": "EXP-0011/EXP-0011-R07", "A'+B": "EXP-0014/EXP-0014-R02", "A'": "EXP-0014/EXP-0014-R01",
        "A'+B+V": "EXP-0015/EXP-0015-R03", "vanilla": "EXP-0019/EXP-0019-R03"}
F1_PAIRS = [("A+B", "A0"), ("A0", "A"), ("A+B", "A"), ("A'+B", "A'")]
AUC_PAIRS = [("A'+B+V", "vanilla"), ("A'+B+V", "A"), ("A", "vanilla")]
N_BOOT = 2000


def f1(yes, lab, w):
    tp = (w * (yes & lab)).sum()
    fp = (w * (yes & ~lab)).sum()
    fn = (w * (~yes & lab)).sum()
    return 2 * tp / (2 * tp + fp + fn)


def main():
    out_dir = f"{OUT}/EXP-0019/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    qs = [json.loads(l) for l in open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl")]
    keys = [(q["category"], q["image"], OBJECT.match(q["text"]).group(1)) for q in qs]
    labels = {"original": read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json"),
              "repope": read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json")}
    images = sorted({k[1] for k in keys})
    img = np.array([images.index(k[1]) for k in keys])
    split = np.array([k[0] for k in keys])
    answers, said = {}, {}
    checks = {}
    for name, run in RUNS.items():
        row(run)                                                    # stops unless the run is completed
        a = [json.loads(l) for l in open(f"{OUT}/{run}/answers.jsonl")]
        if [x["question_id"] for x in a] != [q["question_id"] for q in qs]:
            sys.exit(f"{name}: answers not in question order")
        answers[name] = [x["text"] for x in a]
        said[name] = np.array([not pope_no(t) for t in answers[name]])
    checks["R02 answers equal EXP-0011-R07 question by question"] = float(np.mean(
        [x == y for x, y in zip(answers["A"], answers["A_EXP-0011"])]))
    margin = {}
    for name in ("A'+B+V", "A", "vanilla", "A0"):
        lg = json.load(open(f"{OUT}/{RUNS[name]}/answer_logits.json"))
        lp = np.array([x["logprobs"] for x in lg["first_token"]])
        if len(lp) != len(qs):
            sys.exit(f"{name}: {len(lp)} log-probability rows for {len(qs)} questions")
        margin[name] = lp[:, 0] - lp[:, 1]                          # log P(▁Yes) - log P(▁No)
        checks[f"{name}: first answer token is Yes / No / yes / no"] = float(np.mean(
            [x["argmax"] in lg["ids"] for x in lg["first_token"]]))
        checks[f"{name}: margin sign equals the answer"] = float(np.mean((margin[name] > 0) == said[name]))
    if checks["R02 answers equal EXP-0011-R07 question by question"] < 1.0:
        print("STOP", checks)
        sys.exit(1)

    rng = np.random.default_rng(0)
    counts = np.stack([np.bincount(rng.integers(0, len(images), len(images)), minlength=len(images)) for _ in range(N_BOOT)])
    res = {"checks": checks, "f1": {}, "f1_pairs": {}, "auc": {}, "auc_pairs": {}, "fp": {}}
    for lab_name, lab in labels.items():
        keep = np.array([k in lab for k in keys])
        y = np.array([lab.get(k, False) for k in keys])
        for name in RUNS:
            res["f1"][f"{lab_name} | {name}"] = 100 * float(np.mean(
                [f1(said[name][(split == s) & keep], y[(split == s) & keep], 1) for s in SPLITS]))
        for a, b in F1_PAIRS:
            per_split, point = [], []
            for s in SPLITS:
                m = (split == s) & keep
                w = counts[:, img[m]]
                fa = np.array([f1(said[a][m], y[m], wi) for wi in w])
                fb = np.array([f1(said[b][m], y[m], wi) for wi in w])
                per_split.append(fa - fb)
                point.append(f1(said[a][m], y[m], 1) - f1(said[b][m], y[m], 1))
            d = 100 * np.mean(per_split, axis=0)
            res["f1_pairs"][f"{lab_name} | {a} - {b}"] = [100 * float(np.mean(point)), float(np.percentile(d, 2.5)),
                                                         float(np.percentile(d, 97.5))]
        for name in ("A'+B+V", "A", "vanilla", "A0"):
            for s in SPLITS:
                m = (split == s) & keep
                res["auc"][f"{lab_name} | {name} | {s}"] = auc(margin[name][m], y[m])
                res["fp"][f"{lab_name} | {name} | {s}"] = {"FP": int((said[name][m] & ~y[m]).sum()),
                                                          "accuracy": 100 * float((said[name][m] == y[m]).mean())}
        by_img = [np.flatnonzero(img == k) for k in range(len(images))]
        draws = [np.concatenate([by_img[k] for k in np.repeat(np.arange(len(images)), c)]) for c in counts]
        for a, b in AUC_PAIRS:
            for s in SPLITS:
                boots = []
                for idx in draws:
                    idx = idx[(split[idx] == s) & keep[idx]]
                    boots.append(auc(margin[a][idx], y[idx]) - auc(margin[b][idx], y[idx]))
                pt = res["auc"][f"{lab_name} | {a} | {s}"] - res["auc"][f"{lab_name} | {b} | {s}"]
                res["auc_pairs"][f"{lab_name} | {a} - {b} | {s}"] = [pt, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    q1 = res["f1_pairs"]["original | A+B - A0"]
    res["Q1"] = "above 0" if q1[1] > 0 else ("below 0" if q1[2] < 0 else "contains 0")
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0019.json", "w"), indent=2)
    L = ["# EXP-0019（计划 v1）：A+B 的无先验对照；最终方法的区分力（POPE）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", "## POPE F1（三个 split 的均值）", "", "| 配置 | 原始标注 | RePOPE |", "| --- | --- | --- |"]
    L += [f"| {n} | {res['f1'][f'original | {n}']:.2f} | {res['f1'][f'repope | {n}']:.2f} |" for n in RUNS]
    L += ["", "## F1 配对差（按图像重抽 2000 次）", "", "| 比较 | 原始标注 | RePOPE |", "| --- | --- | --- |"]
    for a, b in F1_PAIRS:
        o, r = res["f1_pairs"][f"original | {a} - {b}"], res["f1_pairs"][f"repope | {a} - {b}"]
        L.append(f"| {a} − {b} | {o[0]:+.2f} [{o[1]:+.2f}, {o[2]:+.2f}] | {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] |")
    L += ["", "## margin AUC、误报与准确率（原始标注）", "", "| 配置 | split | AUC | 误报 | 准确率 |", "| --- | --- | --- | --- | --- |"]
    for name in ("A'+B+V", "A", "A0", "vanilla"):
        for s in SPLITS:
            fp = res["fp"][f"original | {name} | {s}"]
            L.append(f"| {name} | {s} | {res['auc'][f'original | {name} | {s}']:.4f} | {fp['FP']} | {fp['accuracy']:.2f} |")
    L += ["", "## margin AUC 配对差（按图像重抽 2000 次）", "", "| 比较 | split | 原始标注 | RePOPE |", "| --- | --- | --- | --- |"]
    for a, b in AUC_PAIRS:
        for s in SPLITS:
            o, r = res["auc_pairs"][f"original | {a} - {b} | {s}"], res["auc_pairs"][f"repope | {a} - {b} | {s}"]
            L.append(f"| {a} − {b} | {s} | {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}] | {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] |")
    L += ["", f"**Q1**：A+B − A₀ 的区间（原始标注）{ {'above 0': '在 0 以上', 'below 0': '在 0 以下', 'contains 0': '含 0'}[res['Q1']] }。"]
    open(f"{out_dir}/0019.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
