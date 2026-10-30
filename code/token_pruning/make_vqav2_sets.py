"""VQAv2 val subsets on the local COCO val2014 images (EXP-0018): a general-question benchmark with many questions.

  python make_vqav2_sets.py         # writes <LLAVA_EVAL_DATA>/vqav2_val/{llava_vqav2_val_{test,dev}.jsonl, *_annotations.jsonl}
  python make_vqav2_sets.py test2   # EXP-0020: adds llava_vqav2_val_test2{,_annotations}.jsonl to the same folder

Source: VQAv2 val (v2_OpenEnded_mscoco_val2014_questions.json, v2_mscoco_val2014_annotations.json; user-approved
download 2026-10-25, checksums in data/vqav2/SHA256SUMS). test: every question on 1000 images drawn with seed 0 from the
val2014 images that neither POPE nor coco_dev uses (and that exist locally); dev: every question on the 300 coco_dev
images. test2 (EXP-0020): every question on 1000 further images drawn with seed 1 from the same pool minus test's images
(test's draw is redone and checked against the existing file first). The prompt is LLaVA's VQAv2 prompt (question +
"\\nAnswer the question using a single word or phrase."); each annotation keeps the ten human answers, the answer type
and the question type. LLaVA-1.5 trained on VQAv2 train, not on val's questions and answers; the val2014 images may still
appear in its COCO-train2017-based instruction data, as POPE's do (affects absolute accuracy, not comparisons on the same
model). The dev images all hold >= 3 COCO categories (they are coco_dev's), unlike the uniformly drawn test images.
"""
import json
import os
import random
import sys
from collections import defaultdict

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
VQA = os.environ.get("VQAV2_DIR", "D:/6CJNM007/data/vqav2")
IMAGES = os.environ.get("COCO_VAL2014", "D:/6CJNM007/data/coco/val2014")
N_TEST_IMAGES, SEED, SEED_TEST2 = 1000, 0, 1
SUFFIX = "\nAnswer the question using a single word or phrase."


def name(image_id):
    return f"COCO_val2014_{image_id:012d}.jpg"


def load():
    """The questions by image, their annotations, the coco_dev images and the pool test is drawn from."""
    qs = json.load(open(f"{VQA}/v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
    anns = {a["question_id"]: a for a in json.load(open(f"{VQA}/v2_mscoco_val2014_annotations.json"))["annotations"]}
    pope = set()
    for split in ("adversarial", "popular", "random"):
        pope |= {json.loads(line)["image"] for line in open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json")}
    dev_images = {json.loads(line)["image"] for line in open(f"{EVAL_DATA}/pope/llava_coco_dev.jsonl")}
    by_image = defaultdict(list)
    for q in qs:
        by_image[q["image_id"]].append(q)
    pool = sorted(i for i in by_image if name(i) not in pope and name(i) not in dev_images and os.path.exists(f"{IMAGES}/{name(i)}"))
    return by_image, anns, dev_images, pool


def write(out, split, image_ids, by_image, anns):
    n = 0
    # 2026-10-30 note: text mode, so on Windows these files are CRLF (all six VQAv2 subsets); their checksums are of those files
    with open(f"{out}/llava_vqav2_val_{split}.jsonl", "w") as fq, open(f"{out}/llava_vqav2_val_{split}_annotations.jsonl", "w") as fa:
        for image_id in sorted(image_ids):
            for q in sorted(by_image[image_id], key=lambda q: q["question_id"]):
                a = anns[q["question_id"]]
                fq.write(json.dumps({"question_id": q["question_id"], "image": name(image_id), "text": q["question"] + SUFFIX,
                                     "category": a["answer_type"]}) + "\n")
                fa.write(json.dumps({"question_id": q["question_id"], "image": name(image_id), "answers": [x["answer"] for x in a["answers"]],
                                     "answer_type": a["answer_type"], "question_type": a["question_type"]}) + "\n")
                n += 1
    return n


def main():
    out = f"{EVAL_DATA}/vqav2_val"
    if os.path.exists(out):
        sys.exit(f"{out} exists; the sets are never rebuilt in place")
    by_image, anns, dev_images, pool = load()
    test_images = set(random.Random(SEED).sample(pool, N_TEST_IMAGES))
    os.makedirs(out)
    counts = {"test": write(out, "test", test_images, by_image, anns),
              "dev": write(out, "dev", [i for i in by_image if name(i) in dev_images], by_image, anns)}
    print(f"pool {len(pool)} images; test {N_TEST_IMAGES} images / {counts['test']} questions; dev {len(dev_images)} images / {counts['dev']} questions")


def add_test2():
    out = f"{EVAL_DATA}/vqav2_val"
    if os.path.exists(f"{out}/llava_vqav2_val_test2.jsonl") or os.path.exists(f"{out}/llava_vqav2_val_test2_annotations.jsonl"):
        sys.exit("test2 exists; the sets are never rebuilt in place")
    by_image, anns, _, pool = load()
    test_images = set(random.Random(SEED).sample(pool, N_TEST_IMAGES))
    existing = {json.loads(line)["image"] for line in open(f"{out}/llava_vqav2_val_test.jsonl")}
    if existing != {name(i) for i in test_images}:
        sys.exit("redrawn test images differ from llava_vqav2_val_test.jsonl; stopping")
    rest = sorted(set(pool) - test_images)
    test2_images = set(random.Random(SEED_TEST2).sample(rest, N_TEST_IMAGES))
    n = write(out, "test2", test2_images, by_image, anns)
    print(f"pool {len(pool)} images, {len(rest)} after removing test; test2 {N_TEST_IMAGES} images / {n} questions")


if __name__ == "__main__":
    if sys.argv[1:] == ["test2"]:
        add_test2()
    else:
        main()
