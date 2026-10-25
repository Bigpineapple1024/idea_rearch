"""IDEA-0004 data (2026-10-25), drawn once before any development run:

  python make_dev_sets.py

- VQAv2 dev2: every question on 1000 val2014 images drawn with seed 2 from make_vqav2_sets.py's pool minus test's and
  test2's images (both redrawn and checked against the existing files first). For tuning the new module.
- VQAv2 test3: every question on 1000 further images drawn with seed 3 from the pool minus test, test2 and dev2. Kept for the
  pre-registered confirmation only; nothing is run on it before the confirmation plan is committed.
- GQA val400: every balanced question on 400 images drawn with seed 0 from GQA's val-balanced images (official
  questions1.2.zip, images from the official images.zip, both already downloaded); asserted disjoint from testdev's 398.
Never overwrites; writes the same formats as make_vqav2_sets.py and make_gqa_set.py.
"""
import hashlib
import json
import os
import random
import sys
import zipfile

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import make_vqav2_sets as mv   # noqa: E402

EV = "D:/6CJNM007/data/llava_eval"
QZIP = "D:/6CJNM007/data/downloads/gqa/questions1.2.zip"
IZIP = "D:/6CJNM007/data/downloads/gqa/images.zip"
GQA_IMG = "D:/6CJNM007/data/gqa/images_val400"
SEED_DEV2, SEED_TEST3, SEED_GQA = 2, 3, 0


def vqav2():
    out = f"{EV}/vqav2_val"
    for s in ("dev2", "test3"):
        for suffix in ("", "_annotations"):
            if os.path.exists(f"{out}/llava_vqav2_val_{s}{suffix}.jsonl"):
                sys.exit(f"{s} exists; never rebuilt")
    by_image, anns, _, pool = mv.load()
    test = set(random.Random(mv.SEED).sample(pool, mv.N_TEST_IMAGES))
    if {json.loads(l)["image"] for l in open(f"{out}/llava_vqav2_val_test.jsonl")} != {mv.name(i) for i in test}:
        sys.exit("redrawn test images differ from the existing file")
    rest = sorted(set(pool) - test)
    test2 = set(random.Random(mv.SEED_TEST2).sample(rest, mv.N_TEST_IMAGES))
    if {json.loads(l)["image"] for l in open(f"{out}/llava_vqav2_val_test2.jsonl")} != {mv.name(i) for i in test2}:
        sys.exit("redrawn test2 images differ from the existing file")
    rest2 = sorted(set(rest) - test2)
    dev2 = set(random.Random(SEED_DEV2).sample(rest2, mv.N_TEST_IMAGES))
    rest3 = sorted(set(rest2) - dev2)
    test3 = set(random.Random(SEED_TEST3).sample(rest3, mv.N_TEST_IMAGES))
    n2 = mv.write(out, "dev2", dev2, by_image, anns)
    n3 = mv.write(out, "test3", test3, by_image, anns)
    print(f"pool {len(pool)}; dev2 {len(dev2)} images / {n2} questions; test3 {len(test3)} images / {n3} questions")


def gqa():
    qf, af = f"{EV}/gqa/llava_gqa_val400.jsonl", f"{EV}/gqa/llava_gqa_val400_annotations.jsonl"
    for p in (qf, af, GQA_IMG):
        if os.path.exists(p):
            sys.exit(f"{p} exists; never rebuilt")
    with zipfile.ZipFile(QZIP) as z:
        name = [n for n in z.namelist() if n.endswith("val_balanced_questions.json")]
        assert len(name) == 1, name
        val = json.loads(z.read(name[0]))
    testdev_images = {json.loads(l)["image"] for l in open(f"{EV}/gqa/llava_gqa_testdev_balanced.jsonl", encoding="utf-8")}
    images = sorted({f"{q['imageId']}.jpg" for q in val.values()})
    pick = set(random.Random(SEED_GQA).sample(images, 400))
    assert not pick & testdev_images, "GQA val400 overlaps testdev"
    qids = sorted((i for i, q in val.items() if f"{q['imageId']}.jpg" in pick), key=lambda i: (val[i]["imageId"], i))
    os.makedirs(GQA_IMG)
    sums = []
    with zipfile.ZipFile(IZIP) as z:
        members = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".jpg")}
        for im in sorted(pick):
            data = z.read(members[im])
            open(f"{GQA_IMG}/{im}", "wb").write(data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {im}")
    open(f"{GQA_IMG}_SHA256SUMS", "w", newline="\n").write("\n".join(sums) + "\n")
    with open(qf, "w", encoding="utf-8", newline="\n") as fq, open(af, "w", encoding="utf-8", newline="\n") as fa:
        for i in qids:
            q = val[i]
            fq.write(json.dumps({"question_id": i, "image": f"{q['imageId']}.jpg", "text": q["question"] + mv.SUFFIX, "category": "default"}) + "\n")
            fa.write(json.dumps({"question_id": i, "image": f"{q['imageId']}.jpg", "answer": q["answer"], "structural": q["types"]["structural"],
                                 "semantic": q["types"]["semantic"], "detailed": q["types"]["detailed"]}) + "\n")
    print(f"GQA val-balanced {len(val)} questions on {len(images)} images; val400 {len(pick)} images / {len(qids)} questions")


if __name__ == "__main__":
    vqav2()
    gqa()
