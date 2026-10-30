"""Fresh confirmation data for IDEA-0008 (2026-10-31), drawn once while EXP-0034 was running, before any EXP-0034 result
was seen and before any run on it:

  python -B make_confirm_sets2.py

- VQAv2 test5: every question on 1000 val2014 images drawn with seed 5 from make_vqav2_sets.py's pool minus the images of
  test, test2, dev2, test3 and test4 (each redrawn and checked against its existing file first).
- GQA val400c: every balanced question on 400 images drawn with seed 2 from GQA's val-balanced images minus val400's and
  val400b's (the official questions1.2.zip and images.zip, both already downloaded); asserted disjoint from testdev.
Never overwrites; writes the same formats as make_confirm_sets.py and a SHA256SUMS file for each set.
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
GQA_IMG = "D:/6CJNM007/data/gqa/images_val400c"
SEED_DEV2, SEED_TEST3, SEED_TEST4, SEED_TEST5 = 2, 3, 4, 5      # as in make_dev_sets.py and make_confirm_sets.py
SEED_GQA_VAL400, SEED_GQA_VAL400B, SEED_GQA_VAL400C = 0, 1, 2


def images_of(path):
    return {json.loads(line)["image"] for line in open(path, encoding="utf-8")}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def vqav2():
    out = f"{EV}/vqav2_val"
    for p in (f"{out}/llava_vqav2_val_test5.jsonl", f"{out}/llava_vqav2_val_test5_annotations.jsonl", f"{out}/SHA256SUMS_test5"):
        if os.path.exists(p):
            sys.exit(f"{p} exists; never rebuilt")
    by_image, anns, _, pool = mv.load()
    rest = sorted(pool)
    for split, seed in (("test", mv.SEED), ("test2", mv.SEED_TEST2), ("dev2", SEED_DEV2), ("test3", SEED_TEST3), ("test4", SEED_TEST4)):
        drawn = set(random.Random(seed).sample(rest, mv.N_TEST_IMAGES))
        if images_of(f"{out}/llava_vqav2_val_{split}.jsonl") != {mv.name(i) for i in drawn}:
            sys.exit(f"redrawn {split} images differ from the existing file")
        rest = sorted(set(rest) - drawn)
    test5 = set(random.Random(SEED_TEST5).sample(rest, mv.N_TEST_IMAGES))
    used = set()
    for split in ("test", "dev", "test2", "dev2", "test3", "test4"):
        used |= images_of(f"{out}/llava_vqav2_val_{split}.jsonl")
    assert not {mv.name(i) for i in test5} & used, "test5 overlaps an earlier VQAv2 subset"
    n = mv.write(out, "test5", test5, by_image, anns)
    files = [f"{out}/llava_vqav2_val_test5.jsonl", f"{out}/llava_vqav2_val_test5_annotations.jsonl"]
    open(f"{out}/SHA256SUMS_test5", "w", newline="\n").write("".join(f"{sha(p)}  {os.path.basename(p)}\n" for p in files))
    print(f"pool {len(pool)}; {len(rest)} after removing test, test2, dev2, test3, test4; test5 {len(test5)} images / {n} questions")
    return files + [f"{out}/SHA256SUMS_test5"]


def gqa():
    qf, af = f"{EV}/gqa/llava_gqa_val400c.jsonl", f"{EV}/gqa/llava_gqa_val400c_annotations.jsonl"
    for p in (qf, af, GQA_IMG, f"{GQA_IMG}_SHA256SUMS", f"{EV}/gqa/SHA256SUMS_val400c"):
        if os.path.exists(p):
            sys.exit(f"{p} exists; never rebuilt")
    with zipfile.ZipFile(QZIP) as z:
        name = [n for n in z.namelist() if n.endswith("val_balanced_questions.json")]
        assert len(name) == 1, name
        val = json.loads(z.read(name[0]))
    images = sorted({f"{q['imageId']}.jpg" for q in val.values()})
    val400 = set(random.Random(SEED_GQA_VAL400).sample(images, 400))
    if images_of(f"{EV}/gqa/llava_gqa_val400.jsonl") != val400:
        sys.exit("redrawn val400 images differ from the existing file")
    rest = sorted(set(images) - val400)
    val400b = set(random.Random(SEED_GQA_VAL400B).sample(rest, 400))
    if images_of(f"{EV}/gqa/llava_gqa_val400b.jsonl") != val400b:
        sys.exit("redrawn val400b images differ from the existing file")
    rest = sorted(set(rest) - val400b)
    pick = set(random.Random(SEED_GQA_VAL400C).sample(rest, 400))
    assert not pick & (val400 | val400b), "val400c overlaps val400 or val400b"
    assert not pick & images_of(f"{EV}/gqa/llava_gqa_testdev_balanced.jsonl"), "val400c overlaps testdev"
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
    open(f"{EV}/gqa/SHA256SUMS_val400c", "w", newline="\n").write("".join(f"{sha(p)}  {os.path.basename(p)}\n" for p in (qf, af)))
    print(f"GQA val-balanced {len(val)} questions on {len(images)} images; val400c {len(pick)} images / {len(qids)} questions")
    return [qf, af, f"{GQA_IMG}_SHA256SUMS", f"{EV}/gqa/SHA256SUMS_val400c"]


if __name__ == "__main__":
    written = vqav2() + gqa()
    for p in written:
        print(f"{sha(p)}  {p}")
