"""Build the pope_img20 diagnostic subset: every POPE question (all three splits) about 20 images.

  python make_pope_subset.py      # writes llava_pope_test_img20.jsonl and _img20_labels.jsonl next to the test file

Images are the unique images of llava_pope_test.jsonl in file order, taking every 25th of the 500 (indices 0, 25, ...).
The question file keeps LLaVA's fields unchanged, so model_vqa_loader reads it as usual; the labels file adds the
split, the position of the question within its split, the label (eval_pope.py pairs answers and labels in order)
and the object named in the question, for the analysis. Prints the counts and both files' sha256.
"""
import hashlib
import json
import os
import re

POPE = "D:/6CJNM007/data/llava_eval/pope"
SPLITS = ("adversarial", "popular", "random")

with open(f"{POPE}/llava_pope_test.jsonl") as f:
    questions = [json.loads(line) for line in f]
labels = {}
for split in SPLITS:
    with open(f"{POPE}/coco/coco_pope_{split}.json") as f:
        labels[split] = [json.loads(line) for line in f]

images = list(dict.fromkeys(q["image"] for q in questions))
chosen = set(images[::25][:20])
position = {s: 0 for s in SPLITS}
subset, extra = [], []
for q in questions:
    split = q["category"]
    i = position[split]
    position[split] += 1
    if q["image"] not in chosen:
        continue
    lab = labels[split][i]
    if lab["image"] != q["image"]:
        raise SystemExit(f"label order mismatch at {split}[{i}]")
    subset.append(q)
    extra.append({"question_id": q["question_id"], "image": q["image"], "split": split, "index_in_split": i,
                  "label": lab["label"], "object": re.match(r"Is there an? (.+?) in the im", q["text"]).group(1)})

for name, rows in (("llava_pope_test_img20.jsonl", subset), ("llava_pope_test_img20_labels.jsonl", extra)):
    path = f"{POPE}/{name}"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(path, "rb") as f:
        print(hashlib.sha256(f.read()).hexdigest(), name)
print(len(chosen), "images,", len(subset), "questions,",
      {s: sum(e["split"] == s for e in extra) for s in SPLITS}, "per split,",
      sum(e["label"] == "yes" for e in extra), "positive")
