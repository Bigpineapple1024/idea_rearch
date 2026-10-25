"""EXP-0011 D2b (plan v1.1): can another layer or another operator, available before FastV prunes at layer 2, keep small
asked objects 10 points more often than the strong baseline? Offline, from the vanilla logging run on the dev set.

  python analyze_layer_rows.py <R04 run> <R02 run>    # writes code/output/EXP-0011/d2b_coco_dev/

Inputs: R04 (vanilla_layerrows x coco_dev) layer_rows.npz; R02 (strong baseline x coco_dev) kept_tokens.json and
text_rows.npz. Self-check (plan v1.1, stop if violated): the ids after the image equal R02's; R04's layer-1 last-token
row differs from R02's by at most 1% of the row's maximum in every question; the strong baseline simulated from R04's
rows equals R02's kept set in at least 90% of the questions.
Operators eligible for the decision (each with the question sentence "q" and the object name "o"):
  A0  head-averaged row at layer 0;
  H0, H1  the single head with the highest mean object enrichment at layer 0 / 1, chosen on one fold of images (sorted
          file names, odd / even) and applied to the other fold;
  R0, R1  per-head rows normalised over the 576 image tokens, then averaged (heads weighted equally);
  C0, C1, C2  cosine similarity of each image token's state to the span's mean state at hidden_states 0 / 1 / 2.
Every operator selects 30 patches by the 6 x 6 block rotation (fastv_llava._stratified_top) with the padding excluded,
without the position prior. Decision: the best eligible operator's hit rate on positives with area <= 1.35% must be at
least 10 points above the strong baseline (R02's kept sets). Also reported: the difference to padding exclusion alone
(last row + prior + blocks + no padding, from R02's rows), P1 (same-image different-question Jaccard of the global top-30),
P2 (attention mass ratio, attention operators only) and the object AUC (object vs other image patches, padding left out).
Descriptive only: head-averaged rows at layers 2, 4, 8, 16 and 31. Object patches, padding and size tiers as in D2.
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import COCO, EVAL_DATA, OUT, object_record, padding_patches   # noqa: E402
from prompt_spans import piece_text   # noqa: E402

R04, R02 = sys.argv[1], sys.argv[2]
PRIOR = f"{OUT}/EXP-0008/analysis/position_prior.npy"
EDGES = json.load(open(f"{OUT}/EXP-0002/deepdive3/object_boxes.json"))["size_quintile_edges_area_frac"][:2]
KEEP, SIDE, BLOCKS, N_IMG, N_BOOT, THRESHOLD = 30, 24, 6, 576, 1000, 0.10
SPANS = {"q": 0, "o": 1}


def stratified(score, cand):
    """fastv_llava._stratified_top on the candidate patches (ascending patch order), as a sorted list of patch indices."""
    s = score[cand].tolist()
    cell = ((cand // SIDE) * BLOCKS // SIDE * BLOCKS + (cand % SIDE) * BLOCKS // SIDE).tolist()
    taken, rank = {}, [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cell[i], 0)
        taken[cell[i]] = rank[i] + 1
    return sorted(int(cand[i]) for i in sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:KEEP])


def global_top(score):
    return set(np.lexsort((np.arange(N_IMG), -score.astype(np.float64)))[:KEEP].tolist())


def boot_images(values, groups, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [float("nan")] * 3
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values, minlength=len(uniq)), np.bincount(inv, minlength=len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    m = sums[idx].sum(1) / counts[idx].sum(1)
    return [float(values.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def fmt(ci, scale=100, digits=1):
    m, lo, hi = (scale * v for v in ci)
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def auc(score, obj, keep):
    """P(object patch scores above a non-object patch) over the non-padding patches."""
    s = score[keep]
    is_obj = np.isin(keep, obj)
    ranks = np.argsort(np.argsort(s, kind="stable"), kind="stable") + 1.0
    n1, n0 = is_obj.sum(), (~is_obj).sum()
    return float((ranks[is_obj].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    out_dir = f"{OUT}/EXP-0011/d2b_coco_dev"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    z = np.load(f"{OUT}/EXP-0011/{R04}/layer_rows.npz")
    avg, head, cos = z["avg_rows"].astype(np.float32), z["head_rows"].astype(np.float32), z["cos"]
    layers, head_layers, levels = z["layers"].tolist(), z["head_layers"].tolist(), z["state_levels"].tolist()
    r02 = np.load(f"{OUT}/EXP-0011/{R02}/text_rows.npz")
    kept02 = json.load(open(f"{OUT}/EXP-0011/{R02}/kept_tokens.json"))["kept"]
    with open(f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl") as f:
        qs = [json.loads(line) for line in f]
    n = len(qs)
    if not (avg.shape[0] == n == len(kept02) == len(r02["offsets"]) - 1):
        sys.exit("R04, R02 and the question file differ in number")
    prior = np.load(PRIOR).astype(np.float32)

    # self-check against R02
    if not (np.array_equal(z["offsets"], r02["offsets"]) and np.array_equal(z["token_ids"], r02["token_ids"])):
        sys.exit("stop: the ids after the image differ from R02's")
    last02 = np.stack([r02["rows"][r02["offsets"][i + 1] - 1].astype(np.float32) for i in range(n)])
    last04 = avg[:, layers.index(1), 0]
    rel = np.abs(last04 - last02).max(1) / last02.max(1)
    sim_same = np.mean([stratified(last04[i] / prior, np.arange(N_IMG)) == sorted(kept02[i]) for i in range(n)])
    if rel.max() > 0.01 or sim_same < 0.90:
        sys.exit(f"stop: layer-1 last row differs by up to {rel.max():.4f} of its maximum; simulated kept sets equal "
                 f"R02's in {sim_same:.3f} of questions")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    for i in range(n):
        ids = z["token_ids"][z["offsets"][i]:z["offsets"][i + 1]].tolist()
        lo, hi = z["object_span"][i]
        name = "".join(piece_text(p) for p in tokenizer.convert_ids_to_tokens(ids[lo:hi])).strip()
        if name != qs[i]["object"]:
            sys.exit(f"stop: object span of question {qs[i]['question_id']} reads {name!r}, label {qs[i]['object']!r}")

    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["file_name"]: im for im in coco["images"]}
    category = {c["name"]: c["id"] for c in coco["categories"]}
    instances = defaultdict(list)
    for a in coco["annotations"]:
        instances[(a["image_id"], a["category_id"])].append(a)
    pad = {img: padding_patches(images[img]) for img in {q["image"] for q in qs}}
    nonpad = {img: np.setdiff1d(np.arange(N_IMG), p) for img, p in pad.items()}
    fold = {img: k % 2 for k, img in enumerate(sorted(pad))}
    obj_patches, area = {}, {}
    for i, q in enumerate(qs):
        if q["label"] == "yes":
            im = images[q["image"]]
            rec = object_record(im, instances[(im["id"], category[q["object"]])])
            obj_patches[i], area[i] = np.asarray(rec["object_patches"]), rec["area_frac"]
    pos = sorted(obj_patches)
    pos_img = np.array([qs[i]["image"] for i in pos])
    tier = np.searchsorted(EDGES, np.array([area[i] for i in pos]), side="right")
    small = tier <= 1

    def enrich(row, i):
        return float(row[obj_patches[i]].sum() / row.sum() / (len(obj_patches[i]) / N_IMG))

    # per-head choice by two folds of images
    chosen = {}
    for li, layer in enumerate(head_layers):
        for s, si in SPANS.items():
            e = np.array([[enrich(head[i, li, si, h], i) for h in range(head.shape[3])] for i in pos])   # (pos, heads)
            f = np.array([fold[qs[i]["image"]] for i in pos])
            chosen[(layer, s)] = {k: int(e[f == k].mean(0).argmax()) for k in (0, 1)}
            chosen[(layer, s, "mean_enrichment_by_head")] = e.mean(0).tolist()

    def op_rows(name):
        """(n, 576) scores of an operator."""
        kind, s = name[:-2], SPANS[name[-1]]
        if kind.startswith("A"):                     # head-averaged attention at a layer; its span axis is last, q, o
            return avg[:, layers.index(int(kind[1:])), s + 1]
        if kind.startswith("R"):
            li = head_layers.index(int(kind[1:]))
            h = head[:, li, s]
            return (h / h.sum(-1, keepdims=True)).mean(1)
        if kind.startswith("H"):
            layer = int(kind[1:])
            li = head_layers.index(layer)
            other = {0: chosen[(layer, name[-1])][1], 1: chosen[(layer, name[-1])][0]}   # chosen on the other fold
            return np.stack([head[i, li, s, other[fold[qs[i]["image"]]]] for i in range(n)])
        if kind.startswith("C"):
            return cos[:, levels.index(int(kind[1:])), s]
        raise ValueError(name)

    eligible = [f"{k}_{s}" for k in ("A0", "H0", "H1", "R0", "R1", "C0", "C1", "C2") for s in SPANS]
    descriptive = [f"A{layer}_{s}" for layer in (2, 4, 8, 16, 31) for s in SPANS] + [f"Alast{layer}" for layer in (2, 4, 8, 16, 31)]
    results, hits = {}, {}
    for name in eligible + descriptive:
        if name.startswith("Alast"):
            rows = avg[:, layers.index(int(name[5:])), 0]
        else:
            rows = op_rows(name)
        sel = [stratified(rows[i].astype(np.float32), nonpad[qs[i]["image"]]) for i in range(n)]
        h = np.array([np.isin(sel[i], obj_patches[i]).any() for i in pos])
        hits[name] = h
        by_img = defaultdict(dict)
        for i, q in enumerate(qs):
            by_img[q["image"]][q["object"]] = global_top(rows[i])
        jac = [len(o[a] & o[b]) / len(o[a] | o[b]) for o in by_img.values() for a, b in itertools.combinations(sorted(o), 2)]
        is_attention = not name.startswith("C")
        results[name] = {
            "hit_small": boot_images(h[small], pos_img[small]), "hit_all": boot_images(h, pos_img),
            "hit_by_tier": [boot_images(h[tier == k], pos_img[tier == k]) for k in range(3)],
            "p1_jaccard": float(np.mean(jac)),
            "p2_enrichment": float(np.mean([enrich(rows[i], i) for i in pos])) if is_attention else None,
            "object_auc": float(np.mean([auc(rows[i], obj_patches[i], nonpad[qs[i]["image"]]) for i in pos]))}
        print(f"{name}: small {100 * results[name]['hit_small'][0]:.1f}  all {100 * results[name]['hit_all'][0]:.1f}  "
              f"J {results[name]['p1_jaccard']:.3f}  AUC {results[name]['object_auc']:.3f}", flush=True)

    base = np.array([np.isin(kept02[i], obj_patches[i]).any() for i in pos])
    nopad_only = np.array([np.isin(stratified(last02[i] / prior, nonpad[qs[i]["image"]]), obj_patches[i]).any() for i in pos])
    best = max(eligible, key=lambda k: (results[k]["hit_small"][0], results[k]["hit_all"][0]))
    d_base = boot_images(hits[best][small].astype(float) - base[small], pos_img[small])
    d_nopad = boot_images(hits[best][small].astype(float) - nopad_only[small], pos_img[small])
    passed = bool(d_base[0] >= THRESHOLD)
    summary = {"r04": R04, "r02": R02, "n_questions": n, "n_positives": len(pos), "n_small": int(small.sum()),
               "self_check": {"max_rel_diff_layer1_last_row": float(rel.max()), "simulated_kept_equal_share": float(sim_same)},
               "heads_chosen": {f"layer{k[0]}_{k[1]}": v for k, v in chosen.items() if len(k) == 2},
               "mean_enrichment_by_head": {f"layer{k[0]}_{k[1]}": v for k, v in chosen.items() if len(k) == 3},
               "strong_baseline": {"hit_small": boot_images(base[small], pos_img[small]), "hit_all": boot_images(base, pos_img)},
               "padding_exclusion_only": {"hit_small": boot_images(nopad_only[small], pos_img[small]),
                                          "hit_all": boot_images(nopad_only, pos_img)},
               "operators": results, "eligible": eligible, "descriptive": descriptive, "best_eligible": best,
               "best_minus_baseline_small": d_base, "best_minus_padding_exclusion_small": d_nopad, "passed": passed}
    os.makedirs(out_dir)
    with open(f"{out_dir}/d2b.json", "w") as f:
        json.dump(summary, f, indent=2)
    sb, po = summary["strong_baseline"], summary["padding_exclusion_only"]
    lines = [f"EXP-0011 D2b，开发集（{R04} 的 vanilla 记录；对照 {R02}）：{n} 题，正样本 {len(pos)} 个，其中面积 ≤ 1.35% 的 "
             f"{int(small.sum())} 个。自检：第 1 层最后一个 token 行与 R02 的最大相对差 {rel.max():.4f}（上限 0.01）；"
             f"模拟的强对照与 R02 的保留集合相同的题占 {100 * sim_same:.1f}%（下限 90%）；物体名区间与标签逐题一致。", "",
             f"对照：强对照（R02 实际保留）≤ 1.35% 命中 {fmt(sb['hit_small'])}，全部 {fmt(sb['hit_all'])}；"
             f"只排除填充区 {fmt(po['hit_small'])}，全部 {fmt(po['hit_all'])}。", "",
             "所有算子都用 6×6 分块轮流取 30 个、排除填充区、不去位置。q = 题目句，o = 物体名。", "",
             "| 算子 | 参与判定 | ≤ 1.35% 命中（%） | ≤ 0.33% / 0.33～1.35% / 更大 | 全部 | P1 Jaccard | P2 比值 | 物体 AUC |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in eligible + descriptive:
        r = results[name]
        p2 = "—" if r["p2_enrichment"] is None else f"{r['p2_enrichment']:.2f}"
        lines.append(f"| {name} | {'是' if name in eligible else '描述'} | {fmt(r['hit_small'])} | "
                     + " / ".join(f"{100 * x[0]:.1f}" for x in r["hit_by_tier"]) + f" | {fmt(r['hit_all'])} | "
                     f"{r['p1_jaccard']:.3f} | {p2} | {r['object_auc']:.3f} |")
    lines += ["", "逐 head 选择（在一折上选、用到另一折）："]
    for k, v in summary["heads_chosen"].items():
        e = summary["mean_enrichment_by_head"][k]
        lines.append(f"- {k}：折 0 选 head {v[0]}，折 1 选 head {v[1]}；全部正样本上各 head 比值的最大值 {max(e):.2f}，"
                     f"中位数 {np.median(e):.2f}")
    lines += ["", "## 按计划 v1.1 的判定", "",
              f"- 参与判定的算子中最好的是 {best}：≤ 1.35% 命中 {fmt(results[best]['hit_small'])}。",
              f"- 比强对照 {fmt(d_base)} 个百分点（门槛 +10）；比只排除填充区 {fmt(d_nopad)}。",
              f"- 判定：{'通过，做 D3' if passed else '不通过：按 DEC-0017 判定候选 1 不行，换候选 2'}。"]
    with open(f"{out_dir}/d2b.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
