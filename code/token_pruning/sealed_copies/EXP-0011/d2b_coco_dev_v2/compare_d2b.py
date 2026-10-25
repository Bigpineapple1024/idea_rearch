"""Compare d2b_coco_dev (v1) with d2b_coco_dev_v2: every field except the object AUC must be identical."""
import json
import sys

OUT = "D:/6CJNM007/idea_rearch/code/output/EXP-0011"
a = json.load(open(f"{OUT}/d2b_coco_dev/d2b.json"))
b = json.load(open(f"{OUT}/d2b_coco_dev_v2/d2b.json"))
for r in list(a["operators"].values()) + list(b["operators"].values()):
    r.pop("object_auc", None)
    r.pop("object_auc_undefined", None)
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("fields that differ apart from the AUC:", diff or "none")
print("v2 object AUC:", {k: round(v["object_auc"], 3) for k, v in json.load(open(f"{OUT}/d2b_coco_dev_v2/d2b.json"))["operators"].items()})
sys.exit(1 if diff else 0)
