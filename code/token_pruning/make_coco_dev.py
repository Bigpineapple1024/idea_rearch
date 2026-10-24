"""Build a POPE-style selection set from COCO val2014 images that POPE does not use (EXP-0011).

  python make_coco_dev.py      # writes <LLAVA_EVAL_DATA>/pope/llava_coco_dev.jsonl and llava_coco_dev_labels.jsonl

Choices made in EXP-0011 (1A4B-0003, D2) are taken on this set, so POPE stays a test set (research process, rule 3).
Like POPE: images with at least three annotated categories; per image three positives (categories present) and three
negatives (categories absent), drawn uniformly at random (POPE's "random" negatives), seed 0, 300 images; the prompt
is POPE's question plus LLaVA's "Answer the question using a single word or phrase." Labels come from the COCO
instance annotations (user-approved download, 2026-10-24), which miss some objects (F27): good enough to choose
between scoring rules, not to report accuracy.
"""
import json
import os
import random
import sys
from collections import defaultdict

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
COCO = os.environ.get("COCO_INSTANCES", "D:/6CJNM007/data/coco/annotations/instances_val2014.json")
IMAGES = os.environ.get("COCO_VAL2014", "D:/6CJNM007/data/coco/val2014")
N_IMAGES, PER_KIND, SEED = 300, 3, 0
SUFFIX = "\nAnswer the question using a single word or phrase."


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def main():
    out_q = f"{EVAL_DATA}/pope/llava_coco_dev.jsonl"
    out_l = f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl"
    if os.path.exists(out_q) or os.path.exists(out_l):
        sys.exit("the selection set already exists; it is never rebuilt in place")
    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    pope_images = set()
    for split in ("adversarial", "popular", "random"):
        with open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json") as f:
            pope_images |= {json.loads(line)["image"] for line in f}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    present = defaultdict(set)
    for a in coco["annotations"]:
        present[a["image_id"]].add(names[a["category_id"]])
    pool = sorted((im for im in coco["images"] if im["file_name"] not in pope_images and len(present[im["id"]]) >= 3
                   and os.path.exists(f"{IMAGES}/{im['file_name']}")), key=lambda im: im["id"])
    rng = random.Random(SEED)
    chosen = rng.sample(pool, N_IMAGES)
    all_names = sorted(names.values())
    qid = 0
    with open(out_q, "w") as fq, open(out_l, "w") as fl:
        for im in chosen:
            pos = rng.sample(sorted(present[im["id"]]), PER_KIND)
            neg = rng.sample([n for n in all_names if n not in present[im["id"]]], PER_KIND)
            for obj, label in [(o, "yes") for o in pos] + [(o, "no") for o in neg]:
                qid += 1
                text = f"Is there {article(obj)} {obj} in the image?"
                fq.write(json.dumps({"question_id": qid, "image": im["file_name"], "text": text + SUFFIX,
                                     "category": "coco_dev"}) + "\n")
                fl.write(json.dumps({"question_id": qid, "image": im["file_name"], "split": "coco_dev",
                                     "label": label, "object": obj}) + "\n")
    print(f"{N_IMAGES} images (pool {len(pool)}, none in POPE), {qid} questions -> {out_q}")


if __name__ == "__main__":
    main()
