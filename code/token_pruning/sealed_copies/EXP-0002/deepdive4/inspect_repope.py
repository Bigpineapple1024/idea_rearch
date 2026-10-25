"""Which RePOPE questions have no LLaVA POPE counterpart, and how do the files differ?"""
import collections
import json
import re

OBJECT = re.compile(r"Is there an? (.+?) in the (?:image|imange)\?")
SPLITS = ("adversarial", "popular", "random")
E = "D:/6CJNM007/data/llava_eval/pope/coco"
R = "D:/6CJNM007/data/downloads/repope"


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return {(q["image"], OBJECT.match(q["text"]).group(1)): q for q in rows}, rows


for s in SPLITS:
    o, orows = load(f"{E}/coco_pope_{s}.json")
    r, rrows = load(f"{R}/coco_repope_{s}.json")
    extra = [k for k in r if k not in o]
    print(s, "original", len(orows), "unique", len(o), "| repope", len(rrows), "unique", len(r), "| unmatched", len(extra))
    print("  unmatched labels:", collections.Counter(r[k]["label"] for k in extra))
    print("  unmatched images also in original:", sum(1 for img, _ in extra if any(i == img for i, _ in o)))
    print("  question_id range repope:", min(q["question_id"] for q in rrows), max(q["question_id"] for q in rrows))
    for k in extra[:3]:
        img = k[0]
        print("  ", k, r[k], "| original objects for this image:",
              sorted((ob, o[(i, ob)]["label"]) for i, ob in o if i == img))
    gone = [k for k in o if k not in r]
    print("  original not in repope:", len(gone), collections.Counter(o[k]["label"] for k in gone))
