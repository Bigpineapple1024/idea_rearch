"""EXP-0011 D2 (plan v1): do the question tokens' attention rows at FastV's scoring layer depend on the question, fall on
the asked object, and keep small asked objects better than the strong baseline's selection? Offline, from a logging run.

  python analyze_text_rows.py <run> <bench>     # e.g. EXP-0011-R02 coco_dev; writes code/output/EXP-0011/d2_<bench>/

Rows are the head-averaged attention of every text token after the image over the 576 image tokens at layer 1 (the
layer before FastV's K = 2), logged by run_eval.py --text-rows-log. Three scores per question: the last token's row
(FastV as is; the template ":"), the mean of the rows of the question sentence "Is there a(n) X in the image?", and the
mean of the rows of the object name X. Token spans come from the LLaVA tokenizer's pieces of the logged ids.
P1: Jaccard of the global top-30 sets of two questions about different objects on the same image (raw rows, no prior).
P2: per positive, the share of the row's image mass on the object patches over the object's share of the 576 patches.
P3: simulated selections of 30 patches (global top-30 or the 6 x 6 block rotation of fastv_llava._stratified_top; each
with or without the position prior of EXP-0008 and with or without excluding the padding) and their hit rate on
positives (at least one kept patch on the object), by the size tiers of the POPE quintile edges (<= 0.33%, 0.33% to
1.35%, larger; largest instance's mask area over the image). Self-check: last row + prior + blocks must reproduce the
run's kept sets exactly. Dose: share of positives answered yes by kept patches on the object (0, 1, 2-3, >= 4) under the
run's own kept sets. Intervals: 95% bootstrap over images (1000 resamples, seed 0), paired where two scores are compared.
Object patches and padding as in analyze_object_boxes.py (EXP-0002 deepdive3).
"""
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import COCO, EVAL_DATA, OUT, SPLITS, object_record, padding_patches, pope_no   # noqa: E402

RUN, BENCH = sys.argv[1], sys.argv[2]
PRIOR = f"{OUT}/EXP-0008/analysis/position_prior.npy"
EDGES = json.load(open(f"{OUT}/EXP-0002/deepdive3/object_boxes.json"))["size_quintile_edges_area_frac"][:2]
QUESTION = re.compile(r"Is there an? (.+?) in the (?:image|imange)\?")
KEEP, SIDE, BLOCKS, N_IMG, N_BOOT = 30, 24, 6, 576, 1000
SCORES = ("last", "question", "object")
TIERS = ("<=0.33%", "0.33-1.35%", ">1.35%")
DOSE = ("0", "1", "2-3", ">=4")


def questions():
    """[(question_id, image, object, label)] in the order of the run's question file."""
    if BENCH == "coco_dev":
        with open(f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl") as f:
            return [(q["question_id"], q["image"], q["object"], q["label"] == "yes") for q in map(json.loads, f)]
    labels = {}
    for split in SPLITS:
        with open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json") as f:
            for q in map(json.loads, f):
                labels[(split, q["image"], QUESTION.match(q["text"]).group(1))] = q["label"] == "yes"
    out = []
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for q in map(json.loads, f):
            obj = QUESTION.match(q["text"]).group(1)
            out.append((q["question_id"], q["image"], obj, labels[(q["category"], q["image"], obj)]))
    return out


def piece_text(p):
    m = re.fullmatch(r"<0x([0-9A-F]{2})>", p)
    return chr(int(m.group(1), 16)) if m else p.replace("\u2581", " ")


def spans(tokenizer, ids):
    """Token indices of the question sentence and of the object name within the ids after the image."""
    pieces = [piece_text(p) for p in tokenizer.convert_ids_to_tokens(ids)]
    starts = np.cumsum([0] + [len(p) for p in pieces])
    text = "".join(pieces)
    m = QUESTION.search(text)
    if m is None:
        sys.exit(f"no question in {text!r}")

    def covering(lo, hi):        # tokens whose non-space characters fall inside [lo, hi)
        out = []
        for i, p in enumerate(pieces):
            a = starts[i] + (len(p) - len(p.lstrip(" ")))
            b = starts[i + 1]
            if a < b and a < hi and b > lo:
                out.append(i)
        return out

    return covering(m.start(), m.end()), covering(m.start(1), m.end(1)), m.group(1)


def stratified(score, cand):
    """fastv_llava._stratified_top on the candidate patches (ascending patch order), as a list of patch indices."""
    s = score[cand].tolist()
    cell = ((cand // SIDE) * BLOCKS // SIDE * BLOCKS + (cand % SIDE) * BLOCKS // SIDE).tolist()
    taken, rank = {}, [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cell[i], 0)
        taken[cell[i]] = rank[i] + 1
    return sorted(int(cand[i]) for i in sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:KEEP])


def global_top(score, cand):
    order = sorted(range(len(cand)), key=lambda i: (-float(score[cand[i]]), int(cand[i])))
    return sorted(int(cand[i]) for i in order[:KEEP])


def boot_images(values, groups, seed=0):
    """Mean and 95% interval of values (one per item), resampling the items' images."""
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


def main():
    out_dir = f"{OUT}/EXP-0011/d2_{BENCH}"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    run_dir = f"{OUT}/EXP-0011/{RUN}"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    qs = questions()
    with open(f"{run_dir}/answers.jsonl") as f:
        answers = [json.loads(line) for line in f]
    kept = json.load(open(f"{run_dir}/kept_tokens.json"))["kept"]
    z = np.load(f"{run_dir}/text_rows.npz")
    rows, offsets, token_ids = z["rows"], z["offsets"], z["token_ids"]
    if not (len(qs) == len(answers) == len(kept) == len(offsets) - 1):
        sys.exit("questions, answers, kept sets and text rows differ in number")
    if [a["question_id"] for a in answers] != [q[0] for q in qs]:
        sys.exit("answers are not in question-file order")
    prior = np.load(PRIOR).astype(np.float32)

    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["file_name"]: im for im in coco["images"]}
    category = {c["name"]: c["id"] for c in coco["categories"]}
    instances = defaultdict(list)
    for a in coco["annotations"]:
        instances[(a["image_id"], a["category_id"])].append(a)
    pad = {img: padding_patches(images[img]) for img in {q[1] for q in qs}}

    variants = [(s, mode, pr, px) for s in SCORES for mode in ("global", "blocks") for pr in (False, True) for px in (False, True)]
    name = {v: f"{v[0]}+{v[1]}" + ("+prior" if v[2] else "") + ("+nopad" if v[3] else "") for v in variants}
    baseline = ("last", "blocks", True, False)
    items, top30, mismatch, examples = [], defaultdict(dict), 0, []
    for i, (qid, img, obj, label) in enumerate(qs):
        r = rows[offsets[i]:offsets[i + 1]].astype(np.float32)
        ids = token_ids[offsets[i]:offsets[i + 1]].tolist()
        q_idx, o_idx, obj_text = spans(tokenizer, ids)
        if obj_text != obj or not q_idx or not o_idx:
            sys.exit(f"question {qid}: object {obj_text!r} in the prompt, {obj!r} in the labels, spans {q_idx} {o_idx}")
        if i < 3:
            p = tokenizer.convert_ids_to_tokens(ids)
            examples.append({"qid": qid, "pieces": p, "question": [p[j] for j in q_idx], "object": [p[j] for j in o_idx]})
        score = {"last": r[-1], "question": r[q_idx].mean(0), "object": r[o_idx].mean(0)}
        for s in SCORES:
            top30[img].setdefault(obj, {})[s] = set(global_top(score[s], np.arange(N_IMG)))
        sim = {}
        for v in variants:
            sc = score[v[0]] / prior if v[2] else score[v[0]]
            cand = np.setdiff1d(np.arange(N_IMG), pad[img]) if v[3] else np.arange(N_IMG)
            sim[v] = (stratified if v[1] == "blocks" else global_top)(sc.astype(np.float32), cand)
        if sim[baseline] != sorted(kept[i]):
            mismatch += 1
        item = {"qid": qid, "image": img, "object": obj, "label": label, "yes": not pope_no(answers[i]["text"]),
                "pad_share": {name[v]: float(np.isin(sim[v], pad[img]).mean()) for v in variants}}
        if label:
            im = images[img]
            anns = instances.get((im["id"], category.get(obj)), [])
            if not anns:
                item["unmatched"] = True
            else:
                rec = object_record(im, anns)
                patches = np.asarray(rec["object_patches"])
                item.update(area=rec["area_frac"], n_obj=rec["n_object_patches"],
                            on_obj={name[v]: int(np.isin(sim[v], patches).sum()) for v in variants},
                            on_obj_actual=int(np.isin(kept[i], patches).sum()),
                            enrich={s: float(score[s][patches].sum() / score[s].sum() / (len(patches) / N_IMG)) for s in SCORES})
        items.append(item)
    if mismatch:
        sys.exit(f"stop: the simulated strong baseline differs from the run's kept sets in {mismatch} of {len(qs)} prefills")

    # P1: same image, different objects
    jac = {s: [] for s in SCORES}
    jac_img = []
    for img, objs in top30.items():
        for a, b in itertools.combinations(sorted(objs), 2):
            for s in SCORES:
                jac[s].append(len(objs[a][s] & objs[b][s]) / len(objs[a][s] | objs[b][s]))
            jac_img.append(img)
    p1 = {s: boot_images(jac[s], jac_img) for s in SCORES}
    p1_diff = {s: boot_images(np.subtract(jac["last"], jac[s]), jac_img) for s in ("question", "object")}

    pos = [it for it in items if it["label"] and not it.get("unmatched")]
    area = np.array([it["area"] for it in pos])
    tier = np.searchsorted(EDGES, area, side="right")
    small = tier <= 1
    pimg = [it["image"] for it in pos]
    p2 = {s: {"mean": boot_images([it["enrich"][s] for it in pos], pimg),
              "median": float(np.median([it["enrich"][s] for it in pos])),
              "by_tier_mean": [float(np.mean([it["enrich"][s] for it, t in zip(pos, tier) if t == k])) for k in range(3)]}
          for s in SCORES}
    p2_diff = {s: boot_images([it["enrich"][s] - it["enrich"]["last"] for it in pos], pimg) for s in ("question", "object")}

    hits = {name[v]: np.array([it["on_obj"][name[v]] > 0 for it in pos]) for v in variants}
    sim_table = {}
    for v in variants:
        h = hits[name[v]]
        sim_table[name[v]] = {"hit_by_tier": [boot_images(h[tier == k], np.array(pimg)[tier == k]) for k in range(3)],
                              "hit_small": boot_images(h[small], np.array(pimg)[small]), "hit_all": boot_images(h, pimg),
                              "kept_on_object_mean": float(np.mean([it["on_obj"][name[v]] for it in pos])),
                              "padding_share": float(np.mean([it["pad_share"][name[v]] for it in items]))}
    question_blocks = [v for v in variants if v[0] != "last" and v[1] == "blocks"]
    best = max(question_blocks, key=lambda v: (sim_table[name[v]]["hit_small"][0], sim_table[name[v]]["hit_all"][0]))
    p3_diff = boot_images(hits[name[best]][small].astype(float) - hits[name[baseline]][small], np.array(pimg)[small])
    p3_all = {name[v]: boot_images(hits[name[v]][small].astype(float) - hits[name[baseline]][small], np.array(pimg)[small])
              for v in question_blocks}

    def dose_group(n):
        return 0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3
    groups = np.array([dose_group(it["on_obj_actual"]) for it in pos])
    yes = np.array([it["yes"] for it in pos], dtype=float)
    dose = {"all": [boot_images(yes[groups == g], np.array(pimg)[groups == g]) + [int((groups == g).sum())] for g in range(4)],
            "small": [boot_images(yes[small & (groups == g)], np.array(pimg)[small & (groups == g)])
                      + [int((small & (groups == g)).sum())] for g in range(4)]}

    verdict = {"P1": {s: bool(p1_diff[s][0] >= 0.1) for s in ("question", "object")},
               "P2": {s: bool(p2[s]["mean"][0] > 1 and p2_diff[s][0] > 0) for s in ("question", "object")},
               "P3": bool(p3_diff[0] >= 0.10), "P3_best_variant": name[best]}
    summary = {"run": RUN, "bench": BENCH, "n_questions": len(qs), "n_positives": int(sum(it["label"] for it in items)),
               "n_positives_matched": len(pos), "n_positives_unmatched": int(sum(1 for it in items if it.get("unmatched"))),
               "size_tier_edges": EDGES, "n_by_tier": [int((tier == k).sum()) for k in range(3)],
               "median_object_patches_by_tier": [float(np.median([it["n_obj"] for it, t in zip(pos, tier) if t == k]))
                                                 for k in range(3)],
               "self_check": "simulated last+blocks+prior equals the run's kept sets in every prefill",
               "token_span_examples": examples,
               "p1_jaccard": p1, "p1_last_minus": p1_diff, "n_p1_pairs": len(jac_img),
               "p2_enrichment": p2, "p2_minus_last": p2_diff, "simulation": sim_table,
               "p3_best_minus_baseline_small": p3_diff, "p3_each_question_blocks_minus_baseline_small": p3_all,
               "dose_yes_share_by_kept_on_object": dose, "verdict": verdict,
               "answered_yes_share_positives": float(yes.mean()),
               "answered_yes_share_negatives": float(np.mean([it["yes"] for it in items if not it["label"]]))}
    os.makedirs(out_dir)
    with open(f"{out_dir}/d2.json", "w") as f:
        json.dump(summary, f, indent=2)

    t = summary["n_by_tier"]
    lines = [f"EXP-0011 D2，{BENCH}（{RUN}）：{len(qs)} 题，正样本 {len(pos)} 个与 COCO 实例匹配"
             f"（未匹配 {summary['n_positives_unmatched']}）。大小档（最大实例面积占原图）：≤ {100 * EDGES[0]:.2f}% {t[0]} 个，"
             f"{100 * EDGES[0]:.2f}%～{100 * EDGES[1]:.2f}% {t[1]} 个，更大 {t[2]} 个；占 patch 中位数 "
             + " / ".join(f"{x:.0f}" for x in summary["median_object_patches_by_tier"])
             + "。自检：模拟的强对照（最后一个 token 行 + 去位置 + 6×6 分块）与运行的保留集合逐题相同。", "",
             "## P1：同图不同题的 top-30 Jaccard（原始行，不去位置；95% 区间按图像重抽）", "",
             f"| 打分 | Jaccard | 最后一个 token 行 − 该打分 |", "| --- | --- | --- |"]
    for s in SCORES:
        lines.append(f"| {s} | {fmt(p1[s], 1, 3)} | " + ("—" if s == "last" else fmt(p1_diff[s], 1, 3)) + " |")
    lines += [f"（{len(jac_img)} 对题目）", "", "## P2：物体 patch 上的注意力质量 ÷ 物体占 patch 的比例（正样本）", "",
              "| 打分 | 均值 | 中位数 | 按大小档的均值 | 该打分 − 最后一个 token 行 |", "| --- | --- | --- | --- | --- |"]
    for s in SCORES:
        lines.append(f"| {s} | {fmt(p2[s]['mean'], 1, 2)} | {p2[s]['median']:.2f} | "
                     + " / ".join(f"{x:.2f}" for x in p2[s]["by_tier_mean"]) + " | "
                     + ("—" if s == "last" else fmt(p2_diff[s], 1, 2)) + " |")
    lines += ["", "## P3：模拟选择 30 个 patch 时，被问物体至少保留一个 patch 的比例（正样本，%）", "",
              "| 变体 | " + " | ".join(TIERS) + " | ≤ 1.35% | 全部 | 物体上保留的 patch 数（均值） | 落在填充区的比例 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in variants:
        r = sim_table[name[v]]
        mark = "（强对照）" if v == baseline else "（最好的题目行 + 分块）" if v == best else ""
        lines.append(f"| {name[v]}{mark} | " + " | ".join(fmt(x) for x in r["hit_by_tier"]) + f" | {fmt(r['hit_small'])} | "
                     f"{fmt(r['hit_all'])} | {r['kept_on_object_mean']:.2f} | {100 * r['padding_share']:.1f}% |")
    lines += ["", "≤ 1.35% 的正样本上，各“题目行 + 分块”变体 − 强对照（百分点，配对）：", ""]
    lines += [f"- {k}：{fmt(d)}" for k, d in p3_all.items()]
    lines += ["", "## 剂量：运行实际保留的集合下，按物体上保留的 patch 数分组，正样本答 yes 的比例", "",
              "| 组 | " + " | ".join(DOSE) + " |", "| --- | --- | --- | --- | --- |",
              "| 全部正样本 | " + " | ".join(f"{fmt(d[:3])}（{d[3]}）" for d in dose["all"]) + " |",
              "| ≤ 1.35% 的正样本 | " + " | ".join(f"{fmt(d[:3])}（{d[3]}）" for d in dose["small"]) + " |", "",
              f"答 yes 的比例：正样本 {100 * summary['answered_yes_share_positives']:.1f}%，负样本 "
              f"{100 * summary['answered_yes_share_negatives']:.1f}%（COCO 标注会漏标物体，只作参考）。", "",
              "## 按计划 v1 的判读", "",
              f"- P1（Jaccard 比最后一个 token 行至少低 0.1）：question {'成立' if verdict['P1']['question'] else '不成立'}，"
              f"object {'成立' if verdict['P1']['object'] else '不成立'}",
              f"- P2（比值 > 1 且高于最后一个 token 行）：question {'成立' if verdict['P2']['question'] else '不成立'}，"
              f"object {'成立' if verdict['P2']['object'] else '不成立'}",
              f"- P3（≤ 1.35% 上最好的题目行 + 分块比强对照至少高 10 个百分点）：{'成立' if verdict['P3'] else '不成立'}；"
              f"最好的变体 {name[best]}，差 {fmt(p3_diff)}"]
    with open(f"{out_dir}/d2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
