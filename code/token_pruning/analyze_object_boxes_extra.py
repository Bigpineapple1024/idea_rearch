"""Follow-up checks on deepdive3 (analyze_object_boxes.py), written after its first results:

  python analyze_object_boxes_extra.py      # writes code/output/EXP-0002/deepdive3/{extra.md,extra.json}

1. POPE positives with no COCO instance of the same name in the image: how often each method answers no on them.
2. Vertical position with size held fixed: new misses (positives vanilla answers yes three times) by position third
   within size terciles, and the median size per third.
3. Padding: where the kept padding patches lie (above / below the image for landscape images, left / right for portrait).
4. Kept patches on the asked object, size held fixed: share of the object's patches kept, and P(yes) when at least one
   object patch was kept against when none was (pope_img20 positives).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_object_boxes import (FULL, GRID, IMG20_LABELS, KEPT, OUT, boot, boot_two, check_completed, fmt,  # noqa: E402
                                  geometry, load, overlap, pope_no)

MAIN = ["vanilla", "fastv64", "fastv64_prior", "fastv64_strat", "fastv64_prior_strat", "random64", "visionzip64",
        "sparsevlm64", "random_svp64"]


def main():
    out_dir = f"{OUT}/EXP-0002/deepdive3"
    if not os.path.isdir(out_dir) or os.path.exists(f"{out_dir}/extra.json"):
        sys.exit("needs deepdive3 from analyze_object_boxes.py and no earlier extra.json")
    with open(f"{out_dir}/positives.json") as f:
        records = json.load(f)
    with open(f"{out_dir}/object_boxes.json") as f:
        first = json.load(f)
    images, category, instances, questions, labels = load()
    matched = {(r["image"], r["object"]): i for i, r in enumerate(records)}
    unmatched = {tuple(u): i for i, u in enumerate(first["unmatched"])}

    miss, miss_u = {}, {}
    for config in MAIN:
        runs = FULL[config]
        h, c = np.zeros(len(records)), np.zeros(len(records))
        hu, cu = np.zeros(len(unmatched)), np.zeros(len(unmatched))
        for run in runs:
            check_completed(run)
            with open(f"{OUT}/{run}/answers.jsonl") as f:
                for line in f:
                    a = json.loads(line)
                    split, img, obj = questions[a["question_id"]]
                    if not labels[(split, img, obj)]:
                        continue
                    no = pope_no(a["text"])
                    if (img, obj) in matched:
                        h[matched[(img, obj)]] += no
                        c[matched[(img, obj)]] += 1
                    else:
                        hu[unmatched[(img, obj)]] += no
                        cu[unmatched[(img, obj)]] += 1
        if not ((c == 3 * len(runs)).all() and (cu == 3 * len(runs)).all()):
            sys.exit(f"{config}: answers missing")
        miss[config], miss_u[config] = h / c, hu / cu
    part1 = {cfg: {"matched": boot(miss[cfg]), "unmatched": boot(miss_u[cfg])} for cfg in MAIN}
    vanilla_misses = int((miss["vanilla"] > 0).sum() + (miss_u["vanilla"] > 0).sum())
    share_unmatched = float((miss_u["vanilla"] > 0).sum() / vanilla_misses)

    area = np.array([r["area_frac"] for r in records])
    third = np.minimum((np.array([r["y_centre"] for r in records]) * 3).astype(int), 2)
    tercile = np.searchsorted(np.quantile(area, [1 / 3, 2 / 3]), area, side="right")
    ok = miss["vanilla"] == 0
    part2 = {"median_area_by_third": [float(np.median(area[third == t])) for t in range(3)],
             "n_vanilla_ok": [[int((ok & (tercile == s) & (third == t)).sum()) for t in range(3)] for s in range(3)]}
    for cfg in MAIN[1:]:
        m = miss[cfg]
        part2[cfg] = [[boot(m[ok & (tercile == s) & (third == t)]) for t in range(3)]
                      + [boot_two(m[ok & (tercile == s) & (third == 0)], m[ok & (tercile == s) & (third == 2)])]
                      for s in range(3)]

    with open(IMG20_LABELS) as f:
        info = [json.loads(line) for line in f]
    part3, part4 = {}, {}
    for cfg, run in KEPT.items():
        check_completed(run)
        with open(f"{OUT}/{run}/kept_tokens.json") as f:
            kept = json.load(f)["kept"]
        with open(f"{OUT}/{run}/answers.jsonl") as f:
            answers = [json.loads(line) for line in f]
        side = {"above": 0, "below": 0, "left": 0, "right": 0, "image": 0}
        frac, yes_hit, yes_none = {"yes": [], "no": []}, [], []
        for q, s, a in zip(info, kept, answers):
            w, h, sq, ox, oy = geometry(images[q["image"]])
            inside = overlap((0, 0, w, h), sq, ox, oy).ravel() > 1e-9
            for p in s:
                r, col = divmod(p, GRID)
                if inside[p]:
                    side["image"] += 1
                elif w > h:
                    side["above" if r < GRID / 2 else "below"] += 1
                else:
                    side["left" if col < GRID / 2 else "right"] += 1
            if q["label"] == "yes" and (q["image"], q["object"]) in matched:
                rec = records[matched[(q["image"], q["object"])]]
                on = int(np.isin(np.asarray(s), rec["object_patches"]).sum())
                said_yes = not pope_no(a["text"])
                frac["yes" if said_yes else "no"].append(on / rec["n_object_patches"])
                (yes_hit if on > 0 else yes_none).append(said_yes)
        total = sum(side.values())
        part3[cfg] = {k: v / total for k, v in side.items()}
        part4[cfg] = {"object_share_kept": {k: float(np.mean(v)) if v else None for k, v in frac.items()},
                      "p_yes_if_object_patch_kept": [float(np.mean(yes_hit)) if yes_hit else None, len(yes_hit)],
                      "p_yes_if_none_kept": [float(np.mean(yes_none)) if yes_none else None, len(yes_none)]}
    orient = {"landscape": 0, "portrait": 0, "square": 0}
    for img in {q["image"] for q in info}:
        w, h = images[img]["width"], images[img]["height"]
        orient["landscape" if w > h else "portrait" if h > w else "square"] += 1

    summary = {"unmatched_miss": part1, "n_unmatched": len(unmatched), "vanilla_misses_all_positives": vanilla_misses,
               "share_of_vanilla_misses_on_unmatched": share_unmatched, "position_within_size": part2,
               "padding_side_share": part3, "img20_orientation": orient, "object_hit": part4}
    with open(f"{out_dir}/extra.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"## 1. COCO 中没有同名实例的 POPE 正样本（{len(unmatched)} 个）", "",
             f"vanilla 在全部 1500 个正样本上漏掉 {vanilla_misses} 个，其中 {100 * share_unmatched:.1f}% 是这类正样本。", "",
             "| 方法 | 漏答率：有实例的正样本 | 漏答率：没有实例的正样本 |", "| --- | --- | --- |"]
    for cfg in MAIN:
        lines.append(f"| {cfg} | {fmt(part1[cfg]['matched'])} | {fmt(part1[cfg]['unmatched'])} |")
    lines += ["", "## 2. 物体大小固定后的竖直位置（新漏答率，按面积三分位分层）", "",
              f"各位置物体面积占比的中位数（上 / 中 / 下）：" + " / ".join(f"{100 * v:.2f}%" for v in part2["median_area_by_third"]),
              f"；vanilla 答对的正样本个数（行：小、中、大；列：上、中、下）：{part2['n_vanilla_ok']}", "",
              "| 方法 | 大小 | 上 | 中 | 下 | 上 − 下 |", "| --- | --- | --- | --- | --- | --- |"]
    for cfg in MAIN[1:]:
        for s, name in enumerate(("小", "中", "大")):
            lines.append(f"| {cfg} | {name} | " + " | ".join(fmt(v) for v in part2[cfg][s]) + " |")
    lines += ["", f"## 3. 保留的填充区 patch 在哪一侧（pope_img20：{orient}）", "",
              "| 运行 | 图像内 | 填充区：上方 | 填充区：下方 | 填充区：左侧 | 填充区：右侧 |", "| --- | --- | --- | --- | --- | --- |"]
    for cfg, v in part3.items():
        lines.append(f"| {cfg} | {100 * v['image']:.1f}% | {100 * v['above']:.1f}% | {100 * v['below']:.1f}% | "
                     f"{100 * v['left']:.1f}% | {100 * v['right']:.1f}% |")
    lines += ["", "## 4. 被问物体上保留的 patch（pope_img20 正样本）", "",
              "| 运行 | 物体 patch 被保留的比例（答 yes / 答 no） | 至少保留 1 个物体 patch 时答 yes 的比例 | 一个都没保留时答 yes 的比例 |",
              "| --- | --- | --- | --- |"]
    for cfg, v in part4.items():
        sy, sn = v["object_share_kept"]["yes"], v["object_share_kept"]["no"]
        (ph, nh), (pn, nn) = v["p_yes_if_object_patch_kept"], v["p_yes_if_none_kept"]
        lines.append(f"| {cfg} | {'NA' if sy is None else f'{100 * sy:.1f}%'} / {'NA' if sn is None else f'{100 * sn:.1f}%'} | "
                     f"{'NA' if ph is None else f'{100 * ph:.0f}%'}（{nh} 题） | {'NA' if pn is None else f'{100 * pn:.0f}%'}（{nn} 题） |")
    with open(f"{out_dir}/extra.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
