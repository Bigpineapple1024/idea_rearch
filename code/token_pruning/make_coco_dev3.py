"""Add POPE's popular and adversarial negatives to the EXP-0011 selection set (EXP-0013).

  python make_coco_dev3.py     # writes <LLAVA_EVAL_DATA>/pope/llava_coco_dev3.jsonl and llava_coco_dev3_labels.jsonl

The selection set coco_dev (make_coco_dev.py) has POPE's "random" negatives only, while the false positives of EXP-0011's
variant (F30) are largest on POPE's adversarial and popular splits. coco_dev3 keeps every coco_dev question (same ids,
texts, labels: 900 positives, 900 random negatives) and adds, per image, three popular and three adversarial negatives
(ids 1801-3600), built as the POPE paper describes them (Li et al., EMNLP 2023, arXiv 2305.10355, section 5.1): popular
= the k = 3 most frequent categories absent from the image; adversarial = the 3 absent categories with the most
co-occurrences with the categories present (here: the sum over the present categories of the number of images holding
both). Frequencies and co-occurrences count images among the COCO val2014 images POPE does not use; "present" means
annotated in COCO (instances_val2014.json), so a few negatives may be unannotated objects (F27). Ties go by name. Each
split of the set is its 900 positives plus the 900 negatives of that kind, as in POPE. Like coco_dev, the set is for
choosing between rules, not for reporting accuracy.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
COCO = os.environ.get("COCO_INSTANCES", "D:/6CJNM007/data/coco/annotations/instances_val2014.json")
PER_KIND = 3
SUFFIX = "\nAnswer the question using a single word or phrase."


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def main():
    dev_q = f"{EVAL_DATA}/pope/llava_coco_dev.jsonl"
    dev_l = f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl"
    out_q = f"{EVAL_DATA}/pope/llava_coco_dev3.jsonl"
    out_l = f"{EVAL_DATA}/pope/llava_coco_dev3_labels.jsonl"
    if os.path.exists(out_q) or os.path.exists(out_l):
        sys.exit("the selection set already exists; it is never rebuilt in place")
    with open(dev_q) as f:
        questions = [json.loads(line) for line in f]
    with open(dev_l) as f:
        labels = [json.loads(line) for line in f]
    if [q["question_id"] for q in questions] != [lab["question_id"] for lab in labels]:
        sys.exit("coco_dev questions and labels differ")
    with open(COCO, encoding="utf-8") as f:
        coco = json.load(f)
    pope_images = set()
    for split in ("adversarial", "popular", "random"):
        with open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json") as f:
            pope_images |= {json.loads(line)["image"] for line in f}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    file_of = {im["id"]: im["file_name"] for im in coco["images"]}
    present = defaultdict(set)
    for a in coco["annotations"]:
        present[file_of[a["image_id"]]].add(names[a["category_id"]])
    pool = {img: cats for img, cats in present.items() if img not in pope_images}
    freq = Counter(c for cats in pool.values() for c in cats)
    cooc = Counter()
    for cats in pool.values():
        for a, b in combinations(sorted(cats), 2):
            cooc[(a, b)] += 1
            cooc[(b, a)] += 1
    by_freq = sorted(names.values(), key=lambda c: (-freq[c], c))

    images = []
    for lab in labels:
        if lab["image"] not in images:
            images.append(lab["image"])
    new_q, new_l = [], []
    qid = max(q["question_id"] for q in questions)
    for img in images:
        here = present[img]
        absent = [c for c in names.values() if c not in here]
        popular = [c for c in by_freq if c not in here][:PER_KIND]
        adversarial = sorted(absent, key=lambda c: (-sum(cooc[(g, c)] for g in here), c))[:PER_KIND]
        for kind, objs in (("popular", popular), ("adversarial", adversarial)):
            for obj in objs:
                qid += 1
                text = f"Is there {article(obj)} {obj} in the image?"
                new_q.append({"question_id": qid, "image": img, "text": text + SUFFIX, "category": f"coco_dev3_{kind}"})
                new_l.append({"question_id": qid, "image": img, "split": kind, "label": "no", "object": obj})
    with open(out_q, "w") as fq, open(out_l, "w") as fl:
        for q in questions:
            fq.write(json.dumps(q) + "\n")
        for lab in labels:
            kind = "positive" if lab["label"] == "yes" else "random"
            fl.write(json.dumps({**lab, "split": kind}) + "\n")
        for q, lab in zip(new_q, new_l):
            fq.write(json.dumps(q) + "\n")
            fl.write(json.dumps(lab) + "\n")
    kinds = Counter(lab["split"] for lab in new_l)
    print(f"{len(images)} images, {len(questions)} coco_dev questions kept, {len(new_q)} added ({dict(kinds)}) -> {out_q}")
    print("most frequent categories in the pool:", by_freq[:8])
    print("popular negatives:", Counter(lab["object"] for lab in new_l if lab["split"] == "popular").most_common(8))
    print("adversarial negatives:", Counter(lab["object"] for lab in new_l if lab["split"] == "adversarial").most_common(8))


if __name__ == "__main__":
    main()
