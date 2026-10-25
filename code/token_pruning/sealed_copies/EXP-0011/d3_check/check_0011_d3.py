"""EXP-0011 plan v1.2, R05 stop condition: the D3 variant's kept sets on the dev set must equal the offline rule applied
to R04's logged input-embedding cosines (C0_o, blocks, padding excluded) in at least 99% of the questions.

  python check_0011_d3.py      # reads code/output/EXP-0011/EXP-0011-R04/layer_rows.npz and EXP-0011-R05/kept_tokens.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_object_boxes import COCO, EVAL_DATA, padding_patches   # noqa: E402

OUT = "D:/6CJNM007/idea_rearch/code/output/EXP-0011"
SIDE, BLOCKS, KEEP = 24, 6, 30


def stratified(score, cand):
    s = score[cand].tolist()
    cell = ((cand // SIDE) * BLOCKS // SIDE * BLOCKS + (cand % SIDE) * BLOCKS // SIDE).tolist()
    taken, rank = {}, [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cell[i], 0)
        taken[cell[i]] = rank[i] + 1
    return sorted(int(cand[i]) for i in sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:KEEP])


def main():
    z = np.load(f"{OUT}/EXP-0011-R04/layer_rows.npz")
    levels = z["state_levels"].tolist()
    cos = z["cos"][:, levels.index(0), 1]                       # input embeddings, object-name span
    kept = json.load(open(f"{OUT}/EXP-0011-R05/kept_tokens.json"))
    with open(f"{EVAL_DATA}/pope/llava_coco_dev_labels.jsonl") as f:
        qs = [json.loads(line) for line in f]
    with open(COCO, encoding="utf-8") as f:
        images = {im["file_name"]: im for im in json.load(f)["images"]}
    if not (len(qs) == len(kept["kept"]) == kept["n_prefills"] == cos.shape[0]):
        sys.exit("questions, kept sets and R04 records differ in number")
    same, in_pad, full = 0, 0, 0
    for i, q in enumerate(qs):
        pad = padding_patches(images[q["image"]])
        sim = stratified(cos[i].astype(np.float32), np.setdiff1d(np.arange(SIDE * SIDE), pad))
        same += sim == sorted(kept["kept"][i])
        in_pad += bool(set(kept["kept"][i]) & set(pad.tolist()))
        full += len(kept["kept"][i]) == KEEP
    share = same / len(qs)
    print(f"R05 kept sets equal the offline rule on R04's cosines in {same} of {len(qs)} questions ({100 * share:.2f}%); "
          f"sets touching the padding: {in_pad}; sets with {KEEP} patches: {full}")
    sys.exit(0 if share >= 0.99 and in_pad == 0 else 1)


if __name__ == "__main__":
    main()
