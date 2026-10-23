"""Blank-image control for the score-curve diagnostic (EXP-0008): the pope_img20 questions on one grey image.

  python make_blank_questions.py     # writes D:/6CJNM007/data/blank/blank_gray.png and llava_pope_test_img20_blank.jsonl

The image is 336x336, every pixel (128, 128, 128); the question file is llava_pope_test_img20.jsonl with only the
"image" field replaced, so the prompts (and their token positions) are the same as on the real images.
Prints both files' sha256.
"""
import hashlib
import json
import os

from PIL import Image

POPE = "D:/6CJNM007/data/llava_eval/pope"
BLANK = "D:/6CJNM007/data/blank"
image = f"{BLANK}/blank_gray.png"
questions = f"{POPE}/llava_pope_test_img20_blank.jsonl"
for path in (image, questions):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
os.makedirs(BLANK)
Image.new("RGB", (336, 336), (128, 128, 128)).save(image)
with open(f"{POPE}/llava_pope_test_img20.jsonl") as f, open(questions, "w") as out:
    for line in f:
        q = json.loads(line)
        q["image"] = "blank_gray.png"
        out.write(json.dumps(q) + "\n")
for path in (image, questions):
    with open(path, "rb") as f:
        print(hashlib.sha256(f.read()).hexdigest(), path)
