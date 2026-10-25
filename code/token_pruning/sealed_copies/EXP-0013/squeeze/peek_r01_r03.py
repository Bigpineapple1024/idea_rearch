"""Early look at EXP-0013 R01-R03 (exploratory; the plan's decision waits for all eight runs and analyze_fp.py):
dev3 F1 / false positives per split, margin AUC per split, and where the variant's kept patches lie on negatives."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "D:/6CJNM007/tmp/exp0013")
sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_fp import Dev3, KEEP, SPLITS, auc, load_run   # noqa: E402
from analyze_object_boxes import COCO, object_record   # noqa: E402
from analyze_repope import pope_no   # noqa: E402

d = Dev3()
runs = {name: load_run(name) for name in ("vanilla", "agnostic", "variant")}
yes = {n: np.array([not pope_no(a["text"]) for a in r["answers"]]) for n, r in runs.items()}
for n in runs:
    f1, per = d.f1(yes[n])
    print(f"{n}: F1 {f1:.2f} | " + " | ".join(f"{s} F1 {100 * per[s][0]:.1f} FP {per[s][3]}" for s in SPLITS)
          + f" | recall {100 * yes[n][d.kind == 'positive'].mean():.1f}")
    print("   margin AUC: " + ", ".join(f"{s} {auc(r_m, d.lab[d.member[s]]):.4f}" for s, r_m in
                                         ((s, runs[n]['margin'][d.member[s]]) for s in SPLITS)))
coco = json.load(open(COCO, encoding="utf-8"))
images = {im["file_name"]: im for im in coco["images"]}
cat = {c["id"]: c["name"] for c in coco["categories"]}
inst = defaultdict(list)
for a in coco["annotations"]:
    inst[a["image_id"]].append(a)
names = runs["vanilla"]["names"]["names"].tolist()
ncos = runs["vanilla"]["names"]["cos"].astype(np.float64)
nidx = runs["vanilla"]["names"]["index"]
for s in ("popular", "adversarial"):
    for n in ("variant", "agnostic"):
        on, best = [], []
        for i in np.flatnonzero(d.kind == s):
            im = images[d.q[i]["image"]]
            rest = [a for a in inst[im["id"]] if cat[a["category_id"]] != d.q[i]["object"]]
            other = set(object_record(im, rest)["object_patches"])
            k = runs[n]["kept"][i]
            on.append(len(set(k) & other) / KEEP)
            best.append(float(np.mean(ncos[nidx[i]][k].argmax(1) == names.index(d.q[i]["object"]))))
        on, best = np.array(on), np.array(best)
        fp = yes[n][d.kind == s]
        print(f"{s} {n}: FP {fp.sum()} | kept on other objects: FP {100 * on[fp].mean():.1f}% vs rest "
              f"{100 * on[~fp].mean():.1f}% | best name = asked: FP {100 * best[fp].mean():.1f}% vs rest {100 * best[~fp].mean():.1f}%")
