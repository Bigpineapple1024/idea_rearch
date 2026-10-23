"""Exploratory checks behind the EXP-0002 squeeze report (SQZ-EXP-0002). CPU only; reads the run folders.

  python squeeze_exp0002.py > ../output/EXP-0002/squeeze/checks.txt

1. Answer format: does every answer start with yes/no, and how long are the answers.
2. FastV vs random selection on the same POPE questions, split by label.
3. Recall by object for the 15 most frequent positive objects at the 64 budget, with exact McNemar tests between
   methods (no multiple-comparison correction). POPE's positive questions are the same in all three splits, so one
   split ('random') is used.
"""
import collections
import json
import os
import re
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bootstrap_ci as bc  # noqa: E402

OUT = os.path.join(REPO, "code", "output", "EXP-0002")
POPE = {"vanilla": "R34", "fastv192": "R07", "fastv128": "R09", "fastv64": "R11", "random192": "R25", "random128": "R27",
        "random64_s2021": "R29", "random64_s2022": "R31", "random64_s2023": "R33", "visionzip192": "R13",
        "visionzip128": "R15", "visionzip64": "R17", "sparsevlm192": "R37", "sparsevlm128": "R39", "sparsevlm64": "R41"}
MME = {"vanilla": "R05", "fastv192": "R06", "fastv128": "R08", "fastv64": "R10", "random192": "R24", "random128": "R26",
       "random64_s2021": "R28", "visionzip192": "R12", "visionzip128": "R14", "visionzip64": "R35",
       "sparsevlm192": "R36", "sparsevlm128": "R38", "sparsevlm64": "R40"}


def run_dir(r):
    return f"{OUT}/EXP-0002-{r}"


def texts(r):
    with open(f"{run_dir(r)}/answers.jsonl") as f:
        return [json.loads(line)["text"].strip() for line in f]


def sign_test(b, c):
    """Exact two-sided McNemar test on the discordant counts b and c."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def answer_format():
    for bench, runs in (("POPE", POPE), ("MME", MME)):
        print(f"== {bench} answer format: config, answers not starting with yes/no, mean characters ==")
        for name, r in runs.items():
            t = texts(r)
            other = sum(1 for x in t if (x.split()[0].strip(".,").lower() if x else "") not in ("yes", "no"))
            print(f"{name}\t{other}\t{np.mean([len(x) for x in t]):.2f}")


def fastv_vs_random():
    print("== POPE, FastV vs random on the same questions (all splits) ==")
    for f_name, r_name in (("fastv192", "random192"), ("fastv128", "random128"), ("fastv64", "random64_s2021")):
        F, R = bc.load_pope(run_dir(POPE[f_name])), bc.load_pope(run_dir(POPE[r_name]))
        tab = collections.Counter()
        for split in F:
            fc, rc, label = F[split][:, 0] == F[split][:, 1], R[split][:, 0] == R[split][:, 1], F[split][:, 1]
            for lab, lab_name in ((True, "positive"), (False, "negative")):
                m = label == lab
                tab[f"{lab_name}: both right"] += int(np.sum(fc & rc & m))
                tab[f"{lab_name}: only FastV right"] += int(np.sum(fc & ~rc & m))
                tab[f"{lab_name}: only random right"] += int(np.sum(~fc & rc & m))
                tab[f"{lab_name}: both wrong"] += int(np.sum(~fc & ~rc & m))
        print(f_name, "vs", r_name, json.dumps(tab, ensure_ascii=False))


def recall_by_object():
    objs = []
    with open(f"{bc.EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if q["category"] == "random":
                objs.append(re.match(r"Is there an? (.+?) in the im", q["text"]).group(1))
    objs = np.array(objs)
    names = ["vanilla", "fastv64", "random64_s2021", "visionzip64", "sparsevlm64"]
    data = {k: bc.load_pope(run_dir(POPE[k]))["random"] for k in names}
    label = data["vanilla"][:, 1]
    top = [o for o, _ in collections.Counter(objs[label]).most_common(15)]
    print(f"== recall by object, 'random' split: {int(label.sum())} positives, "
          f"{len(set(objs[label]))} objects; columns {' '.join(names)} ==")
    for o in top:
        m = (objs == o) & label
        print(o, int(m.sum()), " ".join(f"{data[k][m, 0].mean():.2f}" for k in names))
    for a, b in (("visionzip64", "random64_s2021"), ("visionzip64", "sparsevlm64"), ("fastv64", "random64_s2021")):
        print(f"== {a} vs {b}: object, n, recall a, recall b, only a says yes, only b says yes, exact McNemar p ==")
        for o in top + ["(all except person)"]:
            m = ((objs == o) if o in top else (objs != "person")) & label
            ya, yb = data[a][m, 0], data[b][m, 0]
            only_a, only_b = int(np.sum(ya & ~yb)), int(np.sum(~ya & yb))
            print(o, int(m.sum()), f"{ya.mean():.3f}", f"{yb.mean():.3f}", only_a, only_b, f"{sign_test(only_a, only_b):.3g}")


if __name__ == "__main__":
    answer_format()
    fastv_vs_random()
    recall_by_object()
