"""EXP-0013 (plan v1): where do the C0_o variant's extra false positives come from, and which budget-neutral fix is best on
the dev set coco_dev3?

  python analyze_fp.py      # writes code/output/EXP-0013/fp_dev3/{fp.md,fp.json}

Runs (all coco_dev3, 3600 questions): R01 vanilla (log-probabilities, name cosines), R02 strong baseline with padding
excluded (scores: FastV's attention before the prior), R03 the EXP-0011 variant (scores: C0_o), R04-R06 budget splits
mix20/15/10, R07/R08 contrast_max/mean (scores: the contrast score). Each split of coco_dev3 is its 900 positives plus
the 900 negatives of one kind (random, popular, adversarial); F1 (yes = positive) is averaged over the three splits.

Implementation checks (plan v1; the script stops when one fails): R03's kept sets equal EXP-0011-R05's on the first 1800
questions (>= 99%); R02's kept sets equal the block rotation of its scores / the position prior over non-padding patches
(>= 99%); R04-R06 equal the two-stage simulation from R03's cosines and R02's scores (>= 99%); R01's cosine to the asked
name equals R03's logged cosine within float16 error (>= 99% of values); R07/R08's logged scores equal the contrast
recomputed from R01's name cosines (>= 99% of values within 3e-3) and their kept sets the block rotation of the logged
scores (>= 99%). Reported only: R01's answers against EXP-0011-R04 (eager prefill) on the first 1800 questions.

Decision (plan v1): candidates mix20/15/10, contrast max/mean and the calibrated variant (yes iff log P(Yes) - log P(No)
> b, b from -4 to 4 in steps of 0.25, chosen on one fold of images and scored on the other, folds by sorted file name
parity); the best dev3 F1 goes to POPE if its paired difference to R03 (image bootstrap, 1000 resamples, seed 0) has a
95% interval above 0. Q1: margin AUC per split, variant - vanilla with image-bootstrap intervals (P1: within
[-0.01, 0.01] on popular and adversarial). Q2: false positives on popular + adversarial, mix10 < mix15 < mix20 < variant
(P2). Q3: on negatives, the share of kept patches on annotated objects of other categories and the share whose best
matching COCO name is the asked one, false positives against the rest (P3, first half descriptive); R07 against R03 on
popular + adversarial false positives and recall (P3, second half: fewer false positives, recall down by <= 3 points).
Descriptive: small-object hits on positives (area <= 1.35%) and the yes share by kept patches on the object.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import COCO, EVAL_DATA, OUT, object_record, padding_patches   # noqa: E402
from analyze_repope import pope_no   # noqa: E402

EXP = "EXP-0013"
RUNS = {"vanilla": "R01", "agnostic": "R02", "variant": "R03", "mix20": "R04", "mix15": "R05", "mix10": "R06",
        "contrast_max": "R07", "contrast_mean": "R08"}
KEPT_RUNS = [k for k in RUNS if k != "vanilla"]
SPLITS = ("random", "popular", "adversarial")
PRIOR = f"{OUT}/EXP-0008/analysis/position_prior.npy"
EDGES = json.load(open(f"{OUT}/EXP-0002/deepdive3/object_boxes.json"))["size_quintile_edges_area_frac"][:2]
KEEP, SIDE, BLOCKS, N_IMG, N_BOOT = 30, 24, 6, 576, 1000
GRID_B = np.arange(-4, 4.0001, 0.25)


def cell(p):
    return (p // SIDE) * BLOCKS // SIDE * BLOCKS + (p % SIDE) * BLOCKS // SIDE


def stratified(score, cand, k, taken=None):
    """fastv_llava._stratified_top on the candidate patches, blocks already holding kept patches waiting their turn."""
    s = score[cand].tolist()
    cells = [cell(int(p)) for p in cand]
    taken, rank = dict(taken or {}), [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cells[i], 0)
        taken[cells[i]] = rank[i] + 1
    return [int(cand[i]) for i in sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:k]]


def f1_counts(yes, lab):
    tp, fp, fn = int((yes & lab).sum()), int((yes & ~lab).sum()), int((~yes & lab).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0, p, r, fp


def auc(score, lab):
    score, lab = np.asarray(score, float), np.asarray(lab, bool)
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = lab.sum(), (~lab).sum()
    return float((ranks[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Dev3:
    def __init__(self):
        with open(f"{EVAL_DATA}/pope/llava_coco_dev3_labels.jsonl") as f:
            self.q = [json.loads(line) for line in f]
        self.n = len(self.q)
        self.kind = np.array([q["split"] for q in self.q])
        self.lab = np.array([q["label"] == "yes" for q in self.q])
        self.img = np.array([q["image"] for q in self.q])
        self.images = sorted(set(self.img))
        self.img_idx = np.array([self.images.index(i) for i in self.img])
        self.fold = np.array([self.images.index(i) % 2 for i in self.img])
        # members of split s: positives and the negatives of kind s
        self.member = {s: (self.kind == "positive") | (self.kind == s) for s in SPLITS}

    def f1(self, yes, mask=None):
        mask = np.ones(self.n, bool) if mask is None else mask
        per = {s: f1_counts(yes[self.member[s] & mask], self.lab[self.member[s] & mask]) for s in SPLITS}
        return 100 * float(np.mean([per[s][0] for s in SPLITS])), per

    def boot_diff(self, yes_a, yes_b, seed=0):
        """F1 mean over splits, a - b, resampling images (with replacement) 1000 times."""
        rng = np.random.default_rng(seed)
        n_img = len(self.images)
        by_img = [np.flatnonzero(self.img_idx == k) for k in range(n_img)]
        diffs = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = np.concatenate([by_img[k] for k in rng.integers(0, n_img, n_img)])
            fa = np.mean([f1_counts(yes_a[idx][self.member[s][idx]], self.lab[idx][self.member[s][idx]])[0] for s in SPLITS])
            fb = np.mean([f1_counts(yes_b[idx][self.member[s][idx]], self.lab[idx][self.member[s][idx]])[0] for s in SPLITS])
            diffs[b] = 100 * (fa - fb)
        point = self.f1(yes_a)[0] - self.f1(yes_b)[0]
        return [float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    def boot_mean_diff(self, va, vb, mask_a, mask_b, seed=0):
        """Difference of means of va over mask_a and vb over mask_b, resampling images."""
        rng = np.random.default_rng(seed)
        n_img = len(self.images)
        by_img = [np.flatnonzero(self.img_idx == k) for k in range(n_img)]
        diffs = []
        for b in range(N_BOOT):
            idx = np.concatenate([by_img[k] for k in rng.integers(0, n_img, n_img)])
            a, c = va[idx][mask_a[idx]], vb[idx][mask_b[idx]]
            if len(a) and len(c):
                diffs.append(a.mean() - c.mean())
        point = float(va[mask_a].mean() - vb[mask_b].mean())
        return [point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def load_run(name):
    run = f"{OUT}/{EXP}/{EXP}-{RUNS[name]}"
    with open(f"{run}/status.json") as f:
        if json.load(f)["status"] != "completed":
            sys.exit(f"{run} is not completed")
    r = {"dir": run}
    with open(f"{run}/answers.jsonl") as f:
        r["answers"] = [json.loads(line) for line in f]
    lg = json.load(open(f"{run}/answer_logits.json"))
    lp = np.array([x["logprobs"] for x in lg["first_token"]])
    r["margin"] = lp[:, 0] - lp[:, 1]                       # log P(▁Yes) - log P(▁No)
    r["argmax"] = np.array([x["argmax"] for x in lg["first_token"]])
    r["ids"] = lg["ids"]
    if os.path.exists(f"{run}/kept_tokens.json"):
        kt = json.load(open(f"{run}/kept_tokens.json"))
        r["kept"] = kt["kept"]
        r["kept_q"] = kt.get("kept_by_object_name")
    if os.path.exists(f"{run}/scores.npy"):
        r["scores"] = np.load(f"{run}/scores.npy").astype(np.float64)
    if os.path.exists(f"{run}/name_cos.npz"):
        r["names"] = np.load(f"{run}/name_cos.npz")
    return r


def main():
    out_dir = f"{OUT}/{EXP}/fp_dev3"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    d = Dev3()
    runs = {name: load_run(name) for name in RUNS}
    for name, r in runs.items():
        if [a["question_id"] for a in r["answers"]] != [q["question_id"] for q in d.q] or len(r["margin"]) != d.n:
            sys.exit(f"{name}: answers or log-probabilities not in question order")
    yes = {name: np.array([not pope_no(a["text"]) for a in r["answers"]]) for name, r in runs.items()}

    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["file_name"]: im for im in coco["images"]}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    inst = defaultdict(list)
    for a in coco["annotations"]:
        inst[a["image_id"]].append(a)
    pad = {img: padding_patches(images[img]) for img in d.images}
    nonpad = {img: np.setdiff1d(np.arange(N_IMG), p) for img, p in pad.items()}
    prior = np.load(PRIOR).astype(np.float64)
    checks = {}

    # --- implementation checks ---
    r05 = json.load(open(f"{OUT}/EXP-0011/EXP-0011-R05/kept_tokens.json"))["kept"]
    checks["R03 kept sets equal EXP-0011-R05 on coco_dev (>= 99%)"] = float(np.mean(
        [sorted(runs["variant"]["kept"][i]) == sorted(r05[i]) for i in range(1800)]))
    att = runs["agnostic"]["scores"]
    cosq = runs["variant"]["scores"]
    sim_ag = [stratified(att[i] / prior, nonpad[d.img[i]], KEEP) for i in range(d.n)]
    checks["R02 kept sets equal the rotation of its scores / prior (>= 99%)"] = float(np.mean(
        [sorted(runs["agnostic"]["kept"][i]) == sorted(sim_ag[i]) for i in range(d.n)]))
    for name, nq in (("mix20", 20), ("mix15", 15), ("mix10", 10)):
        same = 0
        for i in range(d.n):
            cand = nonpad[d.img[i]]
            first = stratified(cosq[i], cand, nq)
            filled = defaultdict(int)
            for p in first:
                filled[cell(p)] += 1
            second = stratified(att[i] / prior, np.setdiff1d(cand, first), KEEP - nq, filled)
            same += sorted(first + second) == sorted(runs[name]["kept"][i]) and sorted(first) == runs[name]["kept_q"][i]
        checks[f"{name} kept sets equal the two-stage simulation (>= 99%)"] = same / d.n
    nz = runs["vanilla"]["names"]
    names = nz["names"].tolist()
    ncos = nz["cos"].astype(np.float64)                     # (distinct images, 576, 80)
    nidx = nz["index"]
    asked = np.array([names.index(q["object"]) for q in d.q])
    cos_from_names = np.stack([ncos[nidx[i], :, asked[i]] for i in range(d.n)])
    checks["R01 asked-name cosine equals R03's logged cosine (>= 99% within 2e-3)"] = float(
        np.mean(np.abs(cos_from_names - cosq) <= 2e-3))
    for name, ref in (("contrast_max", "max"), ("contrast_mean", "mean")):
        rec = np.zeros((d.n, N_IMG))
        for i in range(d.n):
            alt = np.delete(ncos[nidx[i]], asked[i], axis=1)
            rec[i] = ncos[nidx[i], :, asked[i]] - (alt.max(1) if ref == "max" else alt.mean(1))
        checks[f"{name} logged score equals the recomputation from R01 (>= 99% within 3e-3)"] = float(
            np.mean(np.abs(rec - runs[name]["scores"]) <= 3e-3))
        checks[f"{name} kept sets equal the rotation of the logged score (>= 99%)"] = float(np.mean(
            [sorted(stratified(runs[name]["scores"][i], nonpad[d.img[i]], KEEP)) == sorted(runs[name]["kept"][i])
             for i in range(d.n)]))
    failed = [k for k, v in checks.items() if v < 0.99]
    r04 = [json.loads(line) for line in open(f"{OUT}/EXP-0011/EXP-0011-R04/answers.jsonl")]
    agree_r04 = float(np.mean([(not pope_no(a["text"])) == yes["vanilla"][i] for i, a in enumerate(r04)]))
    first_tok = {name: float(np.mean(np.isin(r["argmax"], r["ids"][:2]))) for name, r in runs.items()}
    margin_vs_answer = {name: float(np.mean((r["margin"] > 0) == yes[name])) for name, r in runs.items()}
    if failed:
        print("STOP: implementation checks failed:", failed, checks)
        sys.exit(1)

    # --- dev3 F1, false positives, recall ---
    table = {}
    for name in RUNS:
        f1, per = d.f1(yes[name])
        table[name] = {"f1": f1, "per_split": {s: {"f1": 100 * per[s][0], "precision": 100 * per[s][1],
                                                    "recall": 100 * per[s][2], "fp": per[s][3]} for s in SPLITS},
                       "recall": 100 * float(yes[name][d.kind == "positive"].mean()),
                       "fp_pop_adv": int((yes[name] & np.isin(d.kind, ["popular", "adversarial"])).sum())}

    # --- calibration of the variant (and vanilla, descriptive) ---
    def best_b(margin, mask):
        scores = [d.f1(margin > b, mask)[0] for b in GRID_B]
        return float(GRID_B[int(np.argmax(scores))]), float(max(scores))

    calib = {}
    for name in ("variant", "vanilla", "contrast_max", "contrast_mean", "mix15"):
        m = runs[name]["margin"]
        b_all, f_all = best_b(m, None)
        cross = np.zeros(d.n, bool)
        b_fold = {}
        for k in (0, 1):
            b_k, _ = best_b(m, d.fold == k)
            b_fold[k] = b_k
            other = d.fold == 1 - k
            cross[other] = m[other] > b_k
        calib[name] = {"b_in_sample": b_all, "f1_in_sample": f_all, "b_fold0": b_fold[0], "b_fold1": b_fold[1],
                       "f1_cross_fitted": d.f1(cross)[0], "cross_yes": cross}
    yes["calibrated_variant"] = calib["variant"]["cross_yes"]
    f1c, perc = d.f1(yes["calibrated_variant"])
    table["calibrated_variant"] = {"f1": f1c, "per_split": {s: {"f1": 100 * perc[s][0], "precision": 100 * perc[s][1],
                                                                "recall": 100 * perc[s][2], "fp": perc[s][3]} for s in SPLITS},
                                   "recall": 100 * float(yes["calibrated_variant"][d.kind == "positive"].mean()),
                                   "fp_pop_adv": int((yes["calibrated_variant"] & np.isin(d.kind, ["popular", "adversarial"])).sum())}

    # --- decision ---
    candidates = ["mix20", "mix15", "mix10", "contrast_max", "contrast_mean", "calibrated_variant"]
    diffs = {c: d.boot_diff(yes[c], yes["variant"]) for c in candidates}
    diffs.update({f"{c} - vanilla": d.boot_diff(yes[c], yes["vanilla"]) for c in ["variant"] + candidates})
    best = max(candidates, key=lambda c: table[c]["f1"])
    passed = diffs[best][1] > 0

    # --- Q1: margin AUC per split ---
    q1 = {}
    for name in ["vanilla", "variant", "agnostic", "contrast_max", "contrast_mean", "mix15"]:
        q1[name] = {s: auc(runs[name]["margin"][d.member[s]], d.lab[d.member[s]]) for s in SPLITS}
    q1_diff = {}
    rng = np.random.default_rng(0)
    by_img = [np.flatnonzero(d.img_idx == k) for k in range(len(d.images))]
    for s in SPLITS:
        boots = []
        for _ in range(N_BOOT):
            idx = np.concatenate([by_img[k] for k in rng.integers(0, len(d.images), len(d.images))])
            idx = idx[d.member[s][idx]]
            boots.append(auc(runs["variant"]["margin"][idx], d.lab[idx]) - auc(runs["vanilla"]["margin"][idx], d.lab[idx]))
        q1_diff[s] = [q1["variant"][s] - q1["vanilla"][s], float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    p1 = all(-0.01 <= q1_diff[s][1] and q1_diff[s][2] <= 0.01 for s in ("popular", "adversarial"))
    p1_loss = any(q1_diff[s][2] < -0.01 for s in ("popular", "adversarial"))

    # --- Q2 ---
    fpa = {n: table[n]["fp_pop_adv"] for n in ["mix10", "mix15", "mix20", "variant"]}
    p2 = fpa["mix10"] < fpa["mix15"] < fpa["mix20"] < fpa["variant"]

    # --- Q3 and object hits ---
    other_patches, obj_patches, area = {}, {}, {}
    for i, q in enumerate(d.q):
        im = images[q["image"]]
        anns = inst[im["id"]]
        if q["label"] == "no":
            rest = [a for a in anns if cat_name[a["category_id"]] != q["object"]]
            other_patches[i] = set(object_record(im, rest)["object_patches"]) if rest else set()
        else:
            own = [a for a in anns if cat_name[a["category_id"]] == q["object"]]
            rec = object_record(im, own)
            obj_patches[i], area[i] = set(rec["object_patches"]), rec["area_frac"]
    best_name = {i: ncos[nidx[i]].argmax(1) for i in range(d.n)}
    q3 = {}
    neg_pa = np.isin(d.kind, ["popular", "adversarial"])
    for name in ["variant", "contrast_max", "contrast_mean", "agnostic", "mix15"]:
        kept = runs[name]["kept"]
        on_other = np.array([len(set(kept[i]) & other_patches[i]) / KEEP if i in other_patches else np.nan
                             for i in range(d.n)])
        best_is_asked = np.array([float(np.mean(best_name[i][kept[i]] == asked[i])) for i in range(d.n)])
        row = {}
        for s in ("popular", "adversarial"):
            neg = d.kind == s
            fp_m, ok_m = neg & yes[name], neg & ~yes[name]
            row[s] = {"n_fp": int(fp_m.sum()),
                      "on_other_fp": float(np.nanmean(on_other[fp_m])) if fp_m.any() else float("nan"),
                      "on_other_rest": float(np.nanmean(on_other[ok_m])),
                      "on_other_diff": d.boot_mean_diff(on_other, on_other, fp_m, ok_m) if fp_m.sum() > 1 else None,
                      "best_is_asked_fp": float(best_is_asked[fp_m].mean()) if fp_m.any() else float("nan"),
                      "best_is_asked_rest": float(best_is_asked[ok_m].mean())}
        row["on_other_all_pa"] = float(np.nanmean(on_other[neg_pa]))
        row["best_is_asked_all_pa"] = float(best_is_asked[neg_pa].mean())
        q3[name] = row
    p3a = q3["variant"]["adversarial"]["on_other_fp"] > q3["variant"]["adversarial"]["on_other_rest"]
    p3b = (table["contrast_max"]["fp_pop_adv"] < table["variant"]["fp_pop_adv"]
           and table["contrast_max"]["recall"] >= table["variant"]["recall"] - 3)

    pos = [i for i in range(d.n) if d.lab[i]]
    small = [i for i in pos if area[i] <= EDGES[1]]
    hits, dose = {}, {}
    for name in KEPT_RUNS:
        kept = runs[name]["kept"]
        on = {i: len(set(kept[i]) & obj_patches[i]) for i in pos}
        hits[name] = {"small_hit": 100 * float(np.mean([on[i] > 0 for i in small])),
                      "all_hit": 100 * float(np.mean([on[i] > 0 for i in pos])),
                      "mean_on_object": float(np.mean([on[i] for i in pos]))}
        groups = [[i for i in pos if on[i] == 0], [i for i in pos if on[i] == 1], [i for i in pos if 2 <= on[i] <= 3],
                  [i for i in pos if on[i] >= 4]]
        dose[name] = [[100 * float(yes[name][g].mean()) if g else float("nan"), len(g)] for g in groups]

    summary = {"checks": checks, "r01_vs_exp0011_r04_answer_agreement": agree_r04, "first_token_yes_no_share": first_tok,
               "margin_sign_equals_answer": margin_vs_answer, "table": table,
               "calibration": {k: {kk: vv for kk, vv in v.items() if kk != "cross_yes"} for k, v in calib.items()},
               "paired_vs_variant": {c: diffs[c] for c in candidates},
               "paired_vs_vanilla": {k: v for k, v in diffs.items() if k.endswith("vanilla")},
               "decision": {"best": best, "passed": bool(passed)}, "q1_auc": q1, "q1_variant_minus_vanilla": q1_diff,
               "p1_threshold_shift": bool(p1), "p1_discrimination_loss": bool(p1_loss), "q2_fp_pop_adv": fpa,
               "p2": bool(p2), "q3": q3, "p3_first_half": bool(p3a), "p3_second_half": bool(p3b),
               "object_hits": hits, "dose": dose, "n_small_positives": len(small), "n_positives": len(pos)}
    os.makedirs(out_dir)
    with open(f"{out_dir}/fp.json", "w") as f:
        json.dump(summary, f, indent=2)

    L = ["# EXP-0013 开发集 coco_dev3：误报的来源与同预算修正（计划 v1）", "", "## 实现核对", ""]
    L += [f"- {k}: {100 * v:.2f}%" for k, v in checks.items()]
    L += [f"- （只报告）R01 与 EXP-0011-R04 在前 1800 题上答案相同：{100 * agree_r04:.2f}%",
          "- 第一个回答 token 是 Yes / No 的比例：" + "，".join(f"{k} {100 * v:.1f}%" for k, v in first_tok.items()),
          "- margin > 0 与实际答案一致的比例：" + "，".join(f"{k} {100 * v:.1f}%" for k, v in margin_vs_answer.items()), "",
          "## 开发集 F1（三个 split 的均值）、召回与误报", "",
          "| 配置 | F1 | random F1 | popular F1 | adversarial F1 | 召回 | 误报 random / popular / adversarial |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, t in table.items():
        ps = t["per_split"]
        L.append(f"| {name} | {t['f1']:.2f} | {ps['random']['f1']:.2f} | {ps['popular']['f1']:.2f} | "
                 f"{ps['adversarial']['f1']:.2f} | {t['recall']:.1f}% | {ps['random']['fp']} / {ps['popular']['fp']} / "
                 f"{ps['adversarial']['fp']} |")
    L += ["", "## 配对差（开发集 F1，百分点；按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间] |", "| --- | --- |"]
    for k, v in diffs.items():
        label = k if k.endswith("vanilla") else f"{k} - variant"
        L.append(f"| {label} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] |")
    L += ["", f"**挑选（计划 v1）**：开发集 F1 最高的是 {best}（{table[best]['f1']:.2f}），与变体的配对差 "
              f"{diffs[best][0]:+.2f} [{diffs[best][1]:+.2f}, {diffs[best][2]:+.2f}]：{'进入 POPE' if passed else '不进入 POPE'}。", "",
          "## 校准（答 yes 当且仅当 margin > b）", "", "| 模型 | 样本内 b | 样本内 F1 | 两折的 b | 两折交叉 F1 |",
          "| --- | --- | --- | --- | --- |"]
    for k, v in calib.items():
        L.append(f"| {k} | {v['b_in_sample']:+.2f} | {v['f1_in_sample']:.2f} | {v['b_fold0']:+.2f} / {v['b_fold1']:+.2f} | "
                 f"{v['f1_cross_fitted']:.2f} |")
    L += ["", "## Q1：margin 区分正负样本的 AUC", "", "| 模型 | random | popular | adversarial |", "| --- | --- | --- | --- |"]
    for k, v in q1.items():
        L.append(f"| {k} | {v['random']:.4f} | {v['popular']:.4f} | {v['adversarial']:.4f} |")
    L.append("| 变体 − vanilla | " + " | ".join(f"{q1_diff[s][0]:+.4f} [{q1_diff[s][1]:+.4f}, {q1_diff[s][2]:+.4f}]"
                                              for s in SPLITS) + " |")
    L += ["", f"P1（阈值偏移，popular 与 adversarial 的区间都在 [−0.01, 0.01] 内）：{'成立' if p1 else '不成立'}；"
              f"区分力下降（区间整体低于 −0.01）：{'是' if p1_loss else '否'}。", "",
          f"## Q2：popular + adversarial 的误报数", "", "，".join(f"{k} {v}" for k, v in fpa.items()) +
          f"。P2（mix10 < mix15 < mix20 < 变体）：{'成立' if p2 else '不成立'}。", "",
          "## Q3：负样本上保留的 patch", "",
          "| 配置 | split | 误报数 | 落在其他物体上的比例：误报 / 其余（差 [95% 区间]） | 最像被问类名的比例：误报 / 其余 |",
          "| --- | --- | --- | --- | --- |"]
    for name, row in q3.items():
        for s in ("popular", "adversarial"):
            r = row[s]
            dd = r["on_other_diff"]
            dtxt = f"{100 * dd[0]:+.1f} [{100 * dd[1]:+.1f}, {100 * dd[2]:+.1f}]" if dd else "—"
            L.append(f"| {name} | {s} | {r['n_fp']} | {100 * r['on_other_fp']:.1f}% / {100 * r['on_other_rest']:.1f}%（{dtxt}） | "
                     f"{100 * r['best_is_asked_fp']:.1f}% / {100 * r['best_is_asked_rest']:.1f}% |")
    L += ["", "全部 popular + adversarial 负样本：" + "；".join(
        f"{k} 其他物体上 {100 * v['on_other_all_pa']:.1f}%、最像被问类名 {100 * v['best_is_asked_all_pa']:.1f}%" for k, v in q3.items()),
          "", f"P3 前半（adversarial 误报题落在其他物体上的比例更高）：{'是' if p3a else '否'}；"
              f"后半（contrast_max 的 popular + adversarial 误报比变体少，召回降幅 ≤ 3 个百分点）：{'成立' if p3b else '不成立'}。", "",
          f"## 正样本上的物体命中（{len(small)} 个面积 ≤ 1.35% 的正样本 / 全部 {len(pos)} 个）", "",
          "| 配置 | 小物体命中 | 全部命中 | 物体上保留的 patch 数 | 答 yes 的比例：0 / 1 / 2～3 / ≥4 个（题数） |",
          "| --- | --- | --- | --- | --- |"]
    for name in KEPT_RUNS:
        h = hits[name]
        L.append(f"| {name} | {h['small_hit']:.1f}% | {h['all_hit']:.1f}% | {h['mean_on_object']:.2f} | " +
                 " / ".join(f"{v:.1f}%（{c}）" for v, c in dose[name]) + " |")
    with open(f"{out_dir}/fp.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
