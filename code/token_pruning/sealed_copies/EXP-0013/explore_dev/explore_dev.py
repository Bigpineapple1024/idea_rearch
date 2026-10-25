"""EXP-0013 pre-plan exploration on the dev set only (coco_dev, random negatives): where do the variant's false positives
come from? Uses existing runs only: EXP-0011-R04 (vanilla, eager prefill, cosines logged), R02 (strong baseline, answer
log-probabilities), R05 (C0_o variant, kept sets). No POPE data is read.

  python -I -X utf8 explore_dev.py <out folder>
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import COCO, EVAL_DATA, OUT, padding_patches   # noqa: E402
from analyze_repope import pope_no   # noqa: E402

OUT_DIR = sys.argv[1]
N_IMG = 576


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")                    # ranks with ties averaged
    sorted_v = allv[order]
    r = np.empty_like(allv)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    os.makedirs(OUT_DIR, exist_ok=False)
    with open(f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl") as f:
        qs = [json.loads(line) for line in f]
    n = len(qs)
    lab = np.array([q["label"] == "yes" for q in qs])
    said = {}
    for name, run in (("vanilla", "EXP-0011-R04"), ("strong", "EXP-0011-R02"), ("variant", "EXP-0011-R05")):
        with open(f"{OUT}/EXP-0011/{run}/answers.jsonl") as f:
            ans = [json.loads(line) for line in f]
        assert [a["question_id"] for a in ans] == [q["question_id"] for q in qs]
        said[name] = np.array([not pope_no(a["text"]) for a in ans])
    lines = ["# EXP-0013 pre-plan exploration (dev set, random negatives)", ""]
    lines.append("| run | recall | FP rate | precision | F1 | yes ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, s in said.items():
        tp, fp = int((s & lab).sum()), int((s & ~lab).sum())
        fn = int((~s & lab).sum())
        p, r = tp / (tp + fp), tp / (tp + fn)
        lines.append(f"| {name} | {100 * r:.1f}% | {100 * fp / (~lab).sum():.1f}% ({fp}) | {100 * p:.1f}% | "
                     f"{100 * 2 * p * r / (p + r):.2f} | {100 * s.mean():.1f}% |")
    neg = ~lab
    lines += ["", "FP overlap on negatives: variant FP that vanilla also says yes: "
              f"{int((said['variant'] & said['vanilla'] & neg).sum())} of {int((said['variant'] & neg).sum())}; "
              f"vanilla FP {int((said['vanilla'] & neg).sum())}; strong FP {int((said['strong'] & neg).sum())}"]

    z = np.load(f"{OUT}/EXP-0011/EXP-0011-R04/layer_rows.npz")
    cos = z["cos"][:, 0, 1].astype(np.float64)                  # hidden_states[0], object span: C0_o
    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["file_name"]: im for im in coco["images"]}
    nonpad = {}
    for q in qs:
        if q["image"] not in nonpad:
            nonpad[q["image"]] = np.setdiff1d(np.arange(N_IMG), padding_patches(images[q["image"]]))
    kept05 = json.load(open(f"{OUT}/EXP-0011/EXP-0011-R05/kept_tokens.json"))["kept"]
    stats = defaultdict(list)
    for i, q in enumerate(qs):
        c = cos[i, nonpad[q["image"]]]
        srt = np.sort(c)[::-1]
        stats["max"].append(srt[0])
        stats["top5_mean"].append(srt[:5].mean())
        stats["top30_mean"].append(srt[:30].mean())
        stats["mean"].append(c.mean())
        stats["max_minus_mean"].append(srt[0] - c.mean())
        stats["z_max"].append((srt[0] - c.mean()) / c.std())
        stats["kept_mean"].append(cos[i, kept05[i]].mean())
    # image-level normalisation: the same image's six questions; contrast to the other five object names
    by_img = defaultdict(list)
    for i, q in enumerate(qs):
        by_img[q["image"]].append(i)
    contrast = np.zeros(n)
    for img, idx in by_img.items():
        for i in idx:
            others = [j for j in idx if j != i]
            d = cos[i] - cos[others].mean(0)
            contrast[i] = np.sort(d[nonpad[img]])[::-1][:5].mean()
    stats["contrast_top5_vs_other_names"] = list(contrast)
    lines += ["", "## Presence statistics from C0_o (R04): AUC positives vs negatives", "", "| statistic | AUC |",
              "| --- | --- |"]
    for k, v in stats.items():
        v = np.array(v)
        lines.append(f"| {k} | {auc(v[lab], v[~lab]):.3f} |")

    # dose: FP rate of the variant on negatives by quintile of the kept tokens' mean cosine
    km = np.array(stats["kept_mean"])
    edges = np.quantile(km[neg], [0.2, 0.4, 0.6, 0.8])
    qbin = np.searchsorted(edges, km, side="right")
    lines += ["", "## Negatives: yes rate by quintile of the variant's kept-token mean C0_o", "",
              "| quintile | n | variant yes | vanilla yes | strong yes | mean kept cos |", "| --- | --- | --- | --- | --- | --- |"]
    for b in range(5):
        m = neg & (qbin == b)
        lines.append(f"| {b + 1} | {int(m.sum())} | {100 * said['variant'][m].mean():.1f}% | "
                     f"{100 * said['vanilla'][m].mean():.1f}% | {100 * said['strong'][m].mean():.1f}% | {km[m].mean():.4f} |")
    mx = np.array(stats["max"])
    edges = np.quantile(mx[neg], [0.2, 0.4, 0.6, 0.8])
    qbin = np.searchsorted(edges, mx, side="right")
    lines += ["", "## Negatives: yes rate by quintile of the max C0_o over the image", "",
              "| quintile | n | variant yes | vanilla yes | strong yes |", "| --- | --- | --- | --- | --- |"]
    for b in range(5):
        m = neg & (qbin == b)
        lines.append(f"| {b + 1} | {int(m.sum())} | {100 * said['variant'][m].mean():.1f}% | "
                     f"{100 * said['vanilla'][m].mean():.1f}% | {100 * said['strong'][m].mean():.1f}% |")
    # strong baseline log-probabilities: margin AUC
    lg = json.load(open(f"{OUT}/EXP-0011/EXP-0011-R02/answer_logits.json"))
    lp = np.array([r["logprobs"] for r in lg["first_token"]])
    margin = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
    lines += ["", f"Strong baseline margin (log P(Yes|yes) - log P(No|no)) AUC: {auc(margin[lab], margin[~lab]):.3f}"]
    # which objects produce the variant's false positives
    fp_obj = defaultdict(int)
    for i in np.flatnonzero(said["variant"] & neg):
        fp_obj[qs[i]["object"]] += 1
    lines += ["", "Variant FP objects (dev): " + ", ".join(f"{k} {v}" for k, v in sorted(fp_obj.items(), key=lambda kv: -kv[1]))]
    with open(f"{OUT_DIR}/explore_dev.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
