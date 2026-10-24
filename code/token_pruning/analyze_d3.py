"""EXP-0011 D3 (plan v1.2): does ranking by the object name's input-embedding cosine (blocks, padding excluded) beat
the strong baseline on POPE, and how much of it is padding exclusion alone?

  python analyze_d3.py      # writes code/output/EXP-0011/d3_pope/{d3.md,d3.json}

Runs: the strong baseline EXP-0011-R03 (answers identical to EXP-0010-R05), the variant R06, the padding-only control
R07, and for reference vanilla, FastV-64, random-64 (seed 2021), VisionZip-64 and SparseVLM-64 from EXP-0002. POPE F1 is
the mean over the three splits, scored as llava/eval/eval_pope.py, with the original labels (each run's rescore must
equal its registered headline) and with RePOPE (analyze_repope.py's matching). Paired differences: bootstrap over
questions within each split (1000 resamples, seed 0). Plan v1.2's criterion: R06 - R03 above 0 with the interval above 0.
From the kept sets (R03, R06, R07): the share of positives with at least one kept patch on the asked object by the size
tiers of D2, the padding share of the kept patches, the yes share by kept patches on the object (0, 1, 2-3, >= 4), and
the new-miss rate on positives vanilla answers yes three times by the size quintiles of deepdive3 (comparable to F25).
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import COCO, object_record, padding_patches   # noqa: E402
from analyze_repope import (EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, paired_boot, pope_no, read_labels,   # noqa: E402
                            score)

RUNS = {"strong_baseline": "EXP-0011/EXP-0011-R03", "objcos_variant": "EXP-0011/EXP-0011-R06",
        "padding_only": "EXP-0011/EXP-0011-R07", "vanilla": "EXP-0002/EXP-0002-R04", "fastv64": "EXP-0002/EXP-0002-R11",
        "random64_s2021": "EXP-0002/EXP-0002-R29", "visionzip64": "EXP-0002/EXP-0002-R17",
        "sparsevlm64": "EXP-0002/EXP-0002-R41"}
PAIRS = [("objcos_variant", "strong_baseline"), ("objcos_variant", "padding_only"), ("padding_only", "strong_baseline"),
         ("objcos_variant", "random64_s2021"), ("objcos_variant", "visionzip64"), ("objcos_variant", "sparsevlm64")]
KEPT = ("strong_baseline", "objcos_variant", "padding_only")
EDGES = json.load(open(f"{OUT}/EXP-0002/deepdive3/object_boxes.json"))["size_quintile_edges_area_frac"]
N_BOOT = 1000


def headline(run):
    exp, run_id = run.split("/")
    with open(f"{OUT}/{exp}/runs.tsv") as f:
        row = {r["run_id"]: r for r in csv.DictReader(f, delimiter="\t")}[run_id]
    if row["status"] != "completed":
        sys.exit(f"{run} is not completed")
    return float(row["headline"])


def boot_images(values, groups, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan")] * 3
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values, minlength=len(uniq)), np.bincount(inv, minlength=len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    m = sums[idx].sum(1) / counts[idx].sum(1)
    return [float(values.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def fmt(ci, scale=1, digits=2):
    m, lo, hi = (scale * v for v in ci)
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def pct(ci):
    m, lo, hi = (100 * v for v in ci)
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    out_dir = f"{OUT}/EXP-0011/d3_pope"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    order = []
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            order.append((q["question_id"], (q["category"], q["image"], OBJECT.match(q["text"]).group(1))))
    key_of = dict(order)
    original = read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json")
    repope = {k: v for k, v in read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json").items() if k in original}

    said, answers_in_order, results = {}, {}, {}
    for name, run in RUNS.items():
        with open(f"{OUT}/{run}/answers.jsonl") as f:
            ans = [json.loads(line) for line in f]
        if [a["question_id"] for a in ans] != [qid for qid, _ in order]:
            sys.exit(f"{run}: answers are not in question order")
        s = {key_of[a["question_id"]]: not pope_no(a["text"]) for a in ans}
        per_o, f1_o = score(s, original)
        if abs(f1_o - headline(run)) > 1e-6:
            sys.exit(f"{run}: rescored F1 {f1_o} differs from the registered {headline(run)}")
        per_r, f1_r = score(s, repope)
        said[name], answers_in_order[name] = s, [not pope_no(a["text"]) for a in ans]
        results[name] = {"run": run, "f1_original": f1_o, "f1_repope": f1_r, "original": per_o, "repope": per_r}
    pairs = {f"{a} - {b}": {"original": paired_boot(said[a], said[b], original),
                            "repope": paired_boot(said[a], said[b], repope)} for a, b in PAIRS}
    main_pair = pairs["objcos_variant - strong_baseline"]
    passed = {lab: bool(main_pair[lab][0] > 0 and main_pair[lab][1] > 0) for lab in ("original", "repope")}

    # kept sets on the asked objects
    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["file_name"]: im for im in coco["images"]}
    category = {c["name"]: c["id"] for c in coco["categories"]}
    instances = defaultdict(list)
    for a in coco["annotations"]:
        instances[(a["image_id"], a["category_id"])].append(a)
    pad = {}
    obj_rec = {}
    for qid, (split, img, obj) in order:
        if img not in pad:
            pad[img] = set(padding_patches(images[img]).tolist())
        if original[(split, img, obj)] and (img, obj) not in obj_rec:
            anns = instances.get((images[img]["id"], category.get(obj)), [])
            obj_rec[(img, obj)] = object_record(images[img], anns) if anns else None
    kept = {name: json.load(open(f"{OUT}/{RUNS[name]}/kept_tokens.json"))["kept"] for name in KEPT}
    pos_idx = [i for i, (qid, k) in enumerate(order) if original[k] and obj_rec[(k[1], k[2])] is not None]
    pos_img = np.array([order[i][1][1] for i in pos_idx])
    area = np.array([obj_rec[(order[i][1][1], order[i][1][2])]["area_frac"] for i in pos_idx])
    tier = np.searchsorted(EDGES[:2], area, side="right")
    quint = np.searchsorted(EDGES, area, side="right")
    kept_stats = {}
    for name in KEPT:
        on = np.array([len(set(kept[name][i]) & set(obj_rec[(order[i][1][1], order[i][1][2])]["object_patches"]))
                       for i in pos_idx])
        hit = on > 0
        yes = np.array([answers_in_order[name][i] for i in pos_idx], dtype=float)
        dose = [0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3 for n in on]
        kept_stats[name] = {
            "hit_by_tier": [boot_images(hit[tier == t], pos_img[tier == t]) for t in range(3)],
            "hit_small": boot_images(hit[tier <= 1], pos_img[tier <= 1]), "hit_all": boot_images(hit, pos_img),
            "kept_on_object_mean": float(on.mean()),
            "padding_share": float(np.mean([len(set(s) & pad[order[i][1][1]]) / len(s) for i, s in enumerate(kept[name])])),
            "yes_by_dose": [boot_images(yes[np.array(dose) == g], pos_img[np.array(dose) == g]) + [int((np.array(dose) == g).sum())]
                            for g in range(4)]}
    # new misses on positives vanilla answers yes in all three splits, by deepdive3's size quintiles
    by_pos = defaultdict(list)
    for i in pos_idx:
        by_pos[(order[i][1][1], order[i][1][2])].append(i)
    keys = sorted(by_pos)
    vanilla_ok = np.array([all(answers_in_order["vanilla"][i] for i in by_pos[k]) for k in keys])
    q5 = np.searchsorted(EDGES, np.array([obj_rec[k]["area_frac"] for k in keys]), side="right")
    k_img = np.array([k[0] for k in keys])
    new_miss = {}
    for name in RUNS:
        if name == "vanilla":
            continue
        miss = np.array([np.mean([not answers_in_order[name][i] for i in by_pos[k]]) for k in keys])
        new_miss[name] = [boot_images(miss[vanilla_ok & (q5 == b)], k_img[vanilla_ok & (q5 == b)]) for b in range(5)]

    summary = {"runs": results, "pairs": pairs, "criterion_passed": passed, "kept_on_pope": kept_stats,
               "new_miss_by_size_quintile": new_miss, "n_positive_questions_matched": len(pos_idx),
               "n_positives_vanilla_ok": int(vanilla_ok.sum()), "size_edges": EDGES}
    os.makedirs(out_dir)
    with open(f"{out_dir}/d3.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["EXP-0011 D3（计划 v1.2）：POPE 三子集 F1 均值；原始标注与 RePOPE。每个运行按原始标注重算的 F1 都与登记的主指标一致。", "",
             "| 配置（运行） | F1 原始 | F1 RePOPE | 召回（原始，三子集平均） | 精确率 | yes 比例 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        o = r["original"]
        lines.append(f"| {name}（{r['run']}） | {r['f1_original']:.2f} | {r['f1_repope']:.2f} | "
                     f"{100 * np.mean([o[s]['recall'] for s in SPLITS]):.1f}% | {100 * np.mean([o[s]['precision'] for s in SPLITS]):.1f}% | "
                     f"{100 * np.mean([o[s]['yes_ratio'] for s in SPLITS]):.1f}% |")
    lines += ["", "## 配对差（F1，百分点；题目在各子集内重抽 1000 次）", "", "| 比较 | 原始标注 | RePOPE |", "| --- | --- | --- |"]
    for k, v in pairs.items():
        lines.append(f"| {k} | {fmt(v['original'])} | {fmt(v['repope'])} |")
    lines += ["", f"**判据（v1.2）**：变体 − 强对照 > 0，且区间在 0 以上。原始标注：{'成立' if passed['original'] else '不成立'}；"
              f"RePOPE：{'成立' if passed['repope'] else '不成立'}。", "",
              "## 保留集合与被问物体（POPE 正样本题，与 COCO 实例匹配的）", "",
              "| 配置 | ≤ 0.33% | 0.33～1.35% | > 1.35% | ≤ 1.35% | 全部 | 物体上保留的 patch 数 | 落在填充区的比例 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in kept_stats.items():
        lines.append(f"| {name} | " + " | ".join(pct(x) for x in s["hit_by_tier"]) + f" | {pct(s['hit_small'])} | "
                     f"{pct(s['hit_all'])} | {s['kept_on_object_mean']:.2f} | {100 * s['padding_share']:.1f}% |")
    lines += ["", "| 配置：正样本答 yes 的比例，按物体上保留的 patch 数 | 0 | 1 | 2～3 | ≥ 4 |", "| --- | --- | --- | --- | --- |"]
    for name, s in kept_stats.items():
        lines.append(f"| {name} | " + " | ".join(f"{pct(d[:3])}（{d[3]}）" for d in s["yes_by_dose"]) + " |")
    lines += ["", f"## 新漏答率（vanilla 三次都答对的 {int(vanilla_ok.sum())} 个正样本，按 deepdive3 的大小五分位，%）", "",
              "| 配置 | Q1 最小 | Q2 | Q3 | Q4 | Q5 最大 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, rows in new_miss.items():
        lines.append(f"| {name} | " + " | ".join(pct(x) for x in rows) + " |")
    with open(f"{out_dir}/d3.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
