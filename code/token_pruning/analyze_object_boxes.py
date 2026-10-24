"""Which POPE positives do the pruning methods miss? Object size, position and padding from COCO boxes (C10, P-SQZ7-4)

  python analyze_object_boxes.py      # writes code/output/EXP-0002/deepdive3/{object_boxes.md,object_boxes.json,positives.json}

Uses the COCO val2014 instance annotations (user-approved download, 2026-10-24). A POPE positive (image, object) is
matched to the instances of the COCO category with that name in that image; unmatched positives are counted and left
out. LLaVA-1.5 pads each image to a centred square with the mean colour (mm_utils.expand2square) and cuts it into
24 x 24 patches; boxes are mapped onto that grid. Per positive: size = mask area of the largest instance over the
image area; object patches = patches covered at least 1/4 by an instance box (at least one per instance); vertical
position = centre of the largest instance's box over the image height (0 top, 1 bottom).

Per full-POPE run: the miss probability of each positive (share of its three questions answered no; random-64 pools
its three seeds), miss rates by size quintile, new misses (positives vanilla answers yes three times) by size quintile
and by vertical third, paired SparseVLM-64 minus VisionZip-64 by size quintile, and the size of positives grouped by
how often random selection missed them. Bootstrap 95% intervals over positives (1000 resamples, seed 0).
Per pope_img20 run with kept_tokens.json: the share of kept patches lying wholly in the padding against the padding's
share of the grid, and for positive questions the kept patches on the asked object, split by the answer.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output")
EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
COCO = os.environ.get("COCO_INSTANCES", "D:/6CJNM007/data/coco/annotations/instances_val2014.json")
SPLITS = ("adversarial", "popular", "random")
OBJECT = re.compile(r"Is there an? (.+?) in the (?:image|imange)\?")     # POPE's "an" questions spell it "imange"
GRID = 24
N_BOOT = 1000
FULL = {                                                 # config -> run folders under code/output
    "vanilla": ["EXP-0002/EXP-0002-R04"],
    "fastv64": ["EXP-0002/EXP-0002-R11"],
    "fastv64_prior": ["EXP-0008/EXP-0008-R05"],
    "fastv64_strat": ["EXP-0010/EXP-0010-R03"],
    "fastv64_prior_strat": ["EXP-0010/EXP-0010-R05"],
    "random64": ["EXP-0002/EXP-0002-R29", "EXP-0002/EXP-0002-R31", "EXP-0002/EXP-0002-R33"],
    "visionzip64": ["EXP-0002/EXP-0002-R17"],
    "sparsevlm64": ["EXP-0002/EXP-0002-R41"],
    "fastv_svp64": ["EXP-0007/EXP-0007-R04"],
    "random_svp64": ["EXP-0007/EXP-0007-R03"],
    "fastv_flat64": ["EXP-0007/EXP-0007-R06"],
    "random_flat64": ["EXP-0007/EXP-0007-R05"],
    "fastv128": ["EXP-0002/EXP-0002-R09"],
    "random128": ["EXP-0002/EXP-0002-R27"],
    "visionzip128": ["EXP-0002/EXP-0002-R15"],
    "sparsevlm128": ["EXP-0002/EXP-0002-R39"],
    "fastv192": ["EXP-0002/EXP-0002-R07"],
    "random192": ["EXP-0002/EXP-0002-R25"],
    "visionzip192": ["EXP-0002/EXP-0002-R13"],
    "sparsevlm192": ["EXP-0002/EXP-0002-R37"],
}
PRIMARY = ["vanilla", "fastv64", "fastv64_prior", "fastv64_strat", "fastv64_prior_strat", "random64", "visionzip64",
           "sparsevlm64"]
KEPT = {"fastv64": "EXP-0006/EXP-0006-R01", "random64_s2021": "EXP-0006/EXP-0006-R02",
        "fastv64_prior": "EXP-0008/EXP-0008-R03", "fastv64_strat": "EXP-0010/EXP-0010-R01",
        "fastv64_prior_strat": "EXP-0010/EXP-0010-R02", "fastv192": "EXP-0006/EXP-0006-R03",
        "random192_s2021": "EXP-0006/EXP-0006-R04", "fastv192_prior": "EXP-0008/EXP-0008-R04"}
IMG20_LABELS = f"{EVAL_DATA}/pope/llava_pope_test_img20_labels.jsonl"


def pope_no(text):
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "No" in words or "not" in words or "no" in words


def check_completed(run):
    exp, run_id = run.split("/")
    with open(f"{OUT}/{exp}/runs.tsv") as f:
        status = {r["run_id"]: r["status"] for r in csv.DictReader(f, delimiter="\t")}
    if status.get(run_id) != "completed":
        sys.exit(f"{run} is not a completed run ({status.get(run_id)})")


def geometry(im):
    w, h = im["width"], im["height"]
    s = max(w, h)
    ox = (s - w) // 2 if h > w else 0
    oy = (s - h) // 2 if w > h else 0
    return w, h, s, ox, oy


def overlap(box, s, ox, oy):
    """Share of each patch (rows x cols) covered by box = (x, y, w, h) in original pixels, on the padded square."""
    p = s / GRID
    x0, y0 = box[0] + ox, box[1] + oy
    x1, y1 = x0 + box[2], y0 + box[3]
    edges = np.arange(GRID + 1) * p
    cx = np.clip(np.minimum(edges[1:], x1) - np.maximum(edges[:-1], x0), 0, None) / p
    cy = np.clip(np.minimum(edges[1:], y1) - np.maximum(edges[:-1], y0), 0, None) / p
    return np.outer(cy, cx)


def padding_patches(im):
    w, h, s, ox, oy = geometry(im)
    return np.flatnonzero(overlap((0, 0, w, h), s, ox, oy).ravel() <= 1e-9)


def object_record(im, anns):
    w, h, s, ox, oy = geometry(im)
    mask = np.zeros(GRID * GRID, dtype=bool)
    for a in anns:
        cover = overlap(a["bbox"], s, ox, oy).ravel()
        hit = cover >= 0.25
        if hit.any():
            mask |= hit
        elif cover.max() > 0:                            # an instance smaller than a quarter patch: its best patch
            mask |= cover == cover.max()
    big = max(anns, key=lambda a: a["area"])
    x, y, bw, bh = big["bbox"]
    return {"n_instances": len(anns), "area_frac": big["area"] / (w * h), "y_centre": (y + bh / 2) / h,
            "x_centre": (x + bw / 2) / w, "n_object_patches": int(mask.sum()),
            "object_patches": np.flatnonzero(mask).tolist(), "width": w, "height": h}


def load():
    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["file_name"]: im for im in coco["images"]}
    category = {c["name"]: c["id"] for c in coco["categories"]}
    instances = defaultdict(list)
    for a in coco["annotations"]:
        instances[(a["image_id"], a["category_id"])].append(a)
    questions = {}
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            questions[q["question_id"]] = (q["category"], q["image"], OBJECT.match(q["text"]).group(1))
    labels = {}
    for split in SPLITS:
        with open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json") as f:
            for line in f:
                q = json.loads(line)
                labels[(split, q["image"], OBJECT.match(q["text"]).group(1))] = q["label"] == "yes"
    return images, category, instances, questions, labels


def boot(x, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan")] * 3
    idx = np.random.default_rng(seed).integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_two(a, b, seed=0):
    """Difference of means of two independent groups, resampled within each group."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(1) - b[rng.integers(0, len(b), size=(N_BOOT, len(b)))].mean(1)
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def fmt(ci, scale=100, digits=1):
    m, lo, hi = (scale * v for v in ci)
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main():
    out_dir = f"{OUT}/EXP-0002/deepdive3"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} already exists; results are never overwritten")
    images, category, instances, questions, labels = load()
    positives = sorted({(img, obj) for (split, img, obj), y in labels.items() if y and split == "adversarial"})
    same = all({(i, o) for (s, i, o), y in labels.items() if y and s == sp} == set(positives) for sp in SPLITS)
    records, unmatched = [], []
    for img, obj in positives:
        im = images.get(img)
        anns = instances.get((im["id"], category[obj]), []) if im is not None and obj in category else []
        if not anns:
            unmatched.append([img, obj])
            continue
        records.append({"image": img, "object": obj, **object_record(im, anns)})
    index = {(r["image"], r["object"]): i for i, r in enumerate(records)}
    n = len(records)
    area = np.array([r["area_frac"] for r in records])
    ypos = np.array([r["y_centre"] for r in records])
    footprint = np.array([r["n_object_patches"] for r in records])

    miss = {}
    for config, runs in FULL.items():
        hits, counts = np.zeros(n), np.zeros(n)
        for run in runs:
            check_completed(run)
            with open(f"{OUT}/{run}/answers.jsonl") as f:
                for line in f:
                    a = json.loads(line)
                    split, img, obj = questions[a["question_id"]]
                    if labels[(split, img, obj)] and (img, obj) in index:
                        i = index[(img, obj)]
                        hits[i] += pope_no(a["text"])
                        counts[i] += 1
        if not (counts == 3 * len(runs)).all():
            sys.exit(f"{config}: expected {3 * len(runs)} answers per positive")
        miss[config] = hits / counts
    vanilla_ok = miss["vanilla"] == 0

    edges = np.quantile(area, [0.2, 0.4, 0.6, 0.8])
    size_bin = np.searchsorted(edges, area, side="right")
    third = np.minimum((ypos * 3).astype(int), 2)
    by_size = {c: [boot(m[size_bin == b]) for b in range(5)] + [boot(m)] for c, m in miss.items()}
    new_by_size = {c: [boot(m[vanilla_ok & (size_bin == b)]) for b in range(5)] + [boot(m[vanilla_ok])]
                   for c, m in miss.items() if c != "vanilla"}
    new_by_third = {c: [boot(m[vanilla_ok & (third == t)]) for t in range(3)]
                    + [boot_two(m[vanilla_ok & (third == 0)], m[vanilla_ok & (third == 2)])]
                    for c, m in miss.items() if c != "vanilla"}
    diff = miss["sparsevlm64"] - miss["visionzip64"]
    svz_by_size = [boot(diff[size_bin == b]) for b in range(5)] + [boot(diff)]
    groups = {}
    for config in ("random_svp64", "random_flat64"):
        k = np.rint(miss[config] * 3).astype(int)
        groups[config] = {int(j): {"n": int((k == j).sum()), "median_area_frac": float(np.median(area[k == j])),
                                   "median_object_patches": float(np.median(footprint[k == j])),
                                   "share_smallest_quintile": float((size_bin[k == j] == 0).mean())}
                          for j in range(4) if (k == j).any()}
    shared = {"vanilla_missed": ~vanilla_ok,
              "vanilla_ok_missed_by_fastv_visionzip_sparsevlm":
                  vanilla_ok & (miss["fastv64"] > 0) & (miss["visionzip64"] > 0) & (miss["sparsevlm64"] > 0),
              "vanilla_ok_missed_by_none": vanilla_ok & (miss["fastv64"] == 0) & (miss["visionzip64"] == 0)
                  & (miss["sparsevlm64"] == 0)}
    shared = {g: {"n": int(m.sum()), "median_area_frac": float(np.median(area[m])),
                  "median_object_patches": float(np.median(footprint[m]))} for g, m in shared.items()}

    with open(IMG20_LABELS) as f:
        info = [json.loads(line) for line in f]
    pad = {img: padding_patches(images[img]) for img in {q["image"] for q in info}}
    kept_stats = {}
    for config, run in KEPT.items():
        check_completed(run)
        with open(f"{OUT}/{run}/kept_tokens.json") as f:
            kept = json.load(f)
        with open(f"{OUT}/{run}/answers.jsonl") as f:
            answers = [json.loads(line) for line in f]
        if len(kept["kept"]) != len(info) or len(answers) != len(info):
            sys.exit(f"{run}: kept sets, answers and questions differ in number")
        pad_share, pad_expected, on_obj = [], [], {"yes": [], "no": []}
        for q, s, a in zip(info, kept["kept"], answers):
            s = np.asarray(s)
            pad_share.append(np.isin(s, pad[q["image"]]).mean())
            pad_expected.append(len(pad[q["image"]]) / GRID ** 2)
            if q["label"] == "yes" and (q["image"], q["object"]) in index:
                obj = records[index[(q["image"], q["object"])]]["object_patches"]
                on_obj["no" if pope_no(a["text"]) else "yes"].append(int(np.isin(s, obj).sum()))
        kept_stats[config] = {"run": run, "keep": kept["keep"], "padding_share": float(np.mean(pad_share)),
                              "padding_share_if_uniform": float(np.mean(pad_expected)),
                              "positives_answered": {k: len(v) for k, v in on_obj.items()},
                              "kept_on_object_mean": {k: float(np.mean(v)) if v else None for k, v in on_obj.items()},
                              "share_with_none_on_object": {k: float(np.mean(np.asarray(v) == 0)) if v else None
                                                            for k, v in on_obj.items()}}
    img20_records = [records[index[k]] for k in {(q["image"], q["object"]) for q in info if q["label"] == "yes"}
                     if k in index]
    orientation = defaultdict(int)
    for img in {r["image"] for r in records}:
        w, h = images[img]["width"], images[img]["height"]
        orientation["landscape" if w > h else "portrait" if h > w else "square"] += 1

    summary = {"n_positives": len(positives), "positives_identical_across_splits": same, "n_matched": n,
               "unmatched": unmatched, "image_orientation": dict(orientation),
               "size_quintile_edges_area_frac": edges.tolist(),
               "median_object_patches_by_size_quintile": [float(np.median(footprint[size_bin == b])) for b in range(5)],
               "n_by_size_quintile": [int((size_bin == b).sum()) for b in range(5)],
               "n_vanilla_ok": int(vanilla_ok.sum()),
               "n_vanilla_ok_by_size_quintile": [int((vanilla_ok & (size_bin == b)).sum()) for b in range(5)],
               "n_vanilla_ok_by_third": [int((vanilla_ok & (third == t)).sum()) for t in range(3)],
               "miss_by_size_quintile": by_size, "new_miss_by_size_quintile": new_by_size,
               "new_miss_by_third_and_top_minus_bottom": new_by_third,
               "sparsevlm64_minus_visionzip64_by_size_quintile": svz_by_size,
               "random_miss_count_groups": groups, "shared_miss_groups": shared, "kept_on_pope_img20": kept_stats,
               "img20_positives_median_object_patches": float(np.median([r["n_object_patches"] for r in img20_records]))}
    os.makedirs(out_dir)
    with open(f"{out_dir}/object_boxes.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{out_dir}/positives.json", "w") as f:
        json.dump(records, f)

    q = ["Q1 最小", "Q2", "Q3", "Q4", "Q5 最大"]
    e = [0] + [100 * v for v in edges] + [100]
    lines = [f"POPE 正样本 {len(positives)} 个（三个 split {'相同' if same else '不同'}），与 COCO 实例匹配 {n} 个，"
             f"未匹配 {len(unmatched)} 个。图像方向：{dict(orientation)}。vanilla 三次都答对的正样本 {int(vanilla_ok.sum())} 个。", "",
             "物体大小五分位（最大实例的掩码面积占原图的比例）：" + "；".join(
                 f"{q[b]} {e[b]:.2f}%～{e[b + 1]:.2f}%（{int((size_bin == b).sum())} 个，占 patch 中位数 "
                 f"{np.median(footprint[size_bin == b]):.0f}）" for b in range(5)), "",
             "## 漏答率（全部正样本，按物体大小）", "", "| 方法 | " + " | ".join(q) + " | 全部 |", "| --- |" + " --- |" * 6]
    for c in PRIMARY:
        lines.append(f"| {c} | " + " | ".join(f"{100 * by_size[c][b][0]:.1f}%" for b in range(6)) + " |")
    lines += ["", "## 新漏答率（vanilla 三次都答对的正样本中被漏掉的比例，95% 区间）", "",
              "| 方法 | " + " | ".join(q) + " | 全部 |", "| --- |" + " --- |" * 6]
    for c in [c for c in FULL if c != "vanilla"]:
        lines.append(f"| {c} | " + " | ".join(fmt(new_by_size[c][b]) for b in range(6)) + " |")
    lines += ["", "## 新漏答率按物体的竖直位置（最大实例框中心，原图的上、中、下三分之一）", "",
              "| 方法 | 上 | 中 | 下 | 上 − 下 |", "| --- | --- | --- | --- | --- |"]
    for c in [c for c in FULL if c != "vanilla"]:
        lines.append(f"| {c} | " + " | ".join(fmt(new_by_third[c][t]) for t in range(4)) + " |")
    lines += [f"（vanilla 答对的正样本在上、中、下的个数：{[int((vanilla_ok & (third == t)).sum()) for t in range(3)]}）", "",
              "## SparseVLM-64 − VisionZip-64 的漏答率（配对，按物体大小，百分点）", "",
              "| " + " | ".join(q) + " | 全部 |", "|" + " --- |" * 6,
              "| " + " | ".join(fmt(svz_by_size[b]) for b in range(6)) + " |", "",
              "## 随机选择下按漏答次数分组的正样本（三个 split 各问一次）", "",
              "| 运行 | 漏答次数 | 个数 | 面积占比中位数 | 占 patch 中位数 | 属于最小五分位的比例 |", "| --- | --- | --- | --- | --- | --- |"]
    for config, g in groups.items():
        for j, v in g.items():
            lines.append(f"| {config} | {j} | {v['n']} | {100 * v['median_area_frac']:.2f}% | {v['median_object_patches']:.0f} | "
                         f"{100 * v['share_smallest_quintile']:.1f}% |")
    lines += ["", "## 各方法共同漏掉的正样本", "", "| 组 | 个数 | 面积占比中位数 | 占 patch 中位数 |", "| --- | --- | --- | --- |"]
    for g, v in shared.items():
        lines.append(f"| {g} | {v['n']} | {100 * v['median_area_frac']:.2f}% | {v['median_object_patches']:.0f} |")
    lines += ["", "## pope_img20 上保留的 patch：填充区与被问物体", "",
              "| 运行 | keep | 落在填充区的比例 | 填充区占网格的比例（均匀抽取的期望） | 正样本答 yes / no 的题数 | "
              "物体上保留的 patch 数（答 yes / 答 no） | 物体上一个都没保留的比例（答 yes / 答 no） |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for config, s in kept_stats.items():
        ky, kn = s["kept_on_object_mean"]["yes"], s["kept_on_object_mean"]["no"]
        zy, zn = s["share_with_none_on_object"]["yes"], s["share_with_none_on_object"]["no"]
        lines.append(f"| {config}（{s['run']}） | {s['keep']} | {100 * s['padding_share']:.1f}% | "
                     f"{100 * s['padding_share_if_uniform']:.1f}% | {s['positives_answered']['yes']} / {s['positives_answered']['no']} | "
                     f"{'NA' if ky is None else f'{ky:.2f}'} / {'NA' if kn is None else f'{kn:.2f}'} | "
                     f"{'NA' if zy is None else f'{100 * zy:.0f}%'} / {'NA' if zn is None else f'{100 * zn:.0f}%'} |")
    lines.append(f"（pope_img20 正样本占 patch 数的中位数：{summary['img20_positives_median_object_patches']:.0f}）")
    with open(f"{out_dir}/object_boxes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
