"""VQAv2 val subsets on the local COCO val2014 images (EXP-0018): a general-question benchmark with many questions.

  python make_vqav2_sets.py     # writes <LLAVA_EVAL_DATA>/vqav2_val/{llava_vqav2_val_{test,dev}.jsonl, *_annotations.jsonl}

Source: VQAv2 val (v2_OpenEnded_mscoco_val2014_questions.json, v2_mscoco_val2014_annotations.json; user-approved
download 2026-10-25, checksums in data/vqav2/SHA256SUMS). test: every question on 1000 images drawn with seed 0 from the
val2014 images that neither POPE nor coco_dev uses (and that exist locally); dev: every question on the 300 coco_dev
images. The prompt is LLaVA's VQAv2 prompt (question + "\\nAnswer the question using a single word or phrase."); each
annotation keeps the ten human answers, the answer type and the question type. LLaVA-1.5 trained on VQAv2 train, not val.
"""
import json
import os
import random
import sys
from collections import defaultdict

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
VQA = os.environ.get("VQAV2_DIR", "D:/6CJNM007/data/vqav2")
IMAGES = os.environ.get("COCO_VAL2014", "D:/6CJNM007/data/coco/val2014")
N_TEST_IMAGES, SEED = 1000, 0
SUFFIX = "\nAnswer the question using a single word or phrase."


def main():
    out = f"{EVAL_DATA}/vqav2_val"
    if os.path.exists(out):
        sys.exit(f"{out} exists; the sets are never rebuilt in place")
    qs = json.load(open(f"{VQA}/v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
    anns = {a["question_id"]: a for a in json.load(open(f"{VQA}/v2_mscoco_val2014_annotations.json"))["annotations"]}
    name = lambda image_id: f"COCO_val2014_{image_id:012d}.jpg"   # noqa: E731
    pope = set()
    for split in ("adversarial", "popular", "random"):
        pope |= {json.loads(line)["image"] for line in open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json")}
    dev_images = {json.loads(line)["image"] for line in open(f"{EVAL_DATA}/pope/llava_coco_dev.jsonl")}
    by_image = defaultdict(list)
    for q in qs:
        by_image[q["image_id"]].append(q)
    pool = sorted(i for i in by_image if name(i) not in pope and name(i) not in dev_images and os.path.exists(f"{IMAGES}/{name(i)}"))
    test_images = set(random.Random(SEED).sample(pool, N_TEST_IMAGES))
    os.makedirs(out)
    counts = {}
    for split, keep in (("test", lambda i: i in test_images), ("dev", lambda i: name(i) in dev_images)):
        n = 0
        with open(f"{out}/llava_vqav2_val_{split}.jsonl", "w") as fq, open(f"{out}/llava_vqav2_val_{split}_annotations.jsonl", "w") as fa:
            for image_id in sorted(i for i in by_image if keep(i)):
                for q in sorted(by_image[image_id], key=lambda q: q["question_id"]):
                    a = anns[q["question_id"]]
                    fq.write(json.dumps({"question_id": q["question_id"], "image": name(image_id), "text": q["question"] + SUFFIX,
                                         "category": a["answer_type"]}) + "\n")
                    fa.write(json.dumps({"question_id": q["question_id"], "image": name(image_id), "answers": [x["answer"] for x in a["answers"]],
                                         "answer_type": a["answer_type"], "question_type": a["question_type"]}) + "\n")
                    n += 1
        counts[split] = n
    print(f"pool {len(pool)} images; test {N_TEST_IMAGES} images / {counts['test']} questions; dev {len(dev_images)} images / {counts['dev']} questions")


if __name__ == "__main__":
    main()
