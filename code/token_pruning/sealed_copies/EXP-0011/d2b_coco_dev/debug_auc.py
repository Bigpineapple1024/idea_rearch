"""Debug the NaN object AUC of analyze_layer_rows.py: which positives give n1 == 0 or n0 == 0, and why."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_object_boxes import COCO, EVAL_DATA, object_record, padding_patches   # noqa: E402

with open(f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl") as f:
    qs = [json.loads(line) for line in f]
with open(COCO, encoding="utf-8") as f:
    coco = json.load(f)
images = {im["file_name"]: im for im in coco["images"]}
category = {c["name"]: c["id"] for c in coco["categories"]}
instances = defaultdict(list)
for a in coco["annotations"]:
    instances[(a["image_id"], a["category_id"])].append(a)
bad = []
for i, q in enumerate(qs):
    if q["label"] != "yes":
        continue
    im = images[q["image"]]
    rec = object_record(im, instances[(im["id"], category[q["object"]])])
    obj = np.asarray(rec["object_patches"])
    keep = np.setdiff1d(np.arange(576), padding_patches(im))
    is_obj = np.isin(keep, obj)
    n1, n0 = int(is_obj.sum()), int((~is_obj).sum())
    if n1 == 0 or n0 == 0:
        bad.append((q["question_id"], q["object"], len(obj), n1, n0, len(keep), rec["area_frac"]))
print(len(bad), "positives with n1 == 0 or n0 == 0")
for b in bad[:20]:
    print(b)
