"""SQZ-EXP-0013 exploratory checks on coco_dev3 (after the plan's analysis): accuracy as well as F1; which questions carry
the variant's extra false positives (overlap with vanilla, asked objects); the margin shift on negatives; whether
contrast_max's kept object patches are more 'convincing' (yes share by dose) and where contrast_max and the variant
disagree.

  python squeeze_fp.py <out folder>
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_fp import RUNS, SPLITS, Dev3, load_run   # noqa: E402
from analyze_repope import pope_no   # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=False)
d = Dev3()
runs = {n: load_run(n) for n in RUNS}
yes = {n: np.array([not pope_no(a["text"]) for a in r["answers"]]) for n, r in runs.items()}
L = ["# SQZ-EXP-0013 探索性检查（coco_dev3，计划之外）", ""]

L += ["## 准确率（三个 split 的均值）与按准确率选的阈值", "", "| 配置 | 准确率 | 按准确率最优的 b（样本内）| 该 b 下准确率 |", "| --- | --- | --- | --- |"]
grid = np.arange(-4, 4.0001, 0.25)


def acc(y):
    return 100 * float(np.mean([(y[d.member[s]] == d.lab[d.member[s]]).mean() for s in SPLITS]))


for n in RUNS:
    m = runs[n]["margin"]
    accs = [acc(m > b) for b in grid]
    L.append(f"| {n} | {acc(yes[n]):.2f} | {grid[int(np.argmax(accs))]:+.2f} | {max(accs):.2f} |")

neg = ~d.lab
L += ["", "## 变体与 vanilla 的误报重叠（负样本）", "", "| split | vanilla 误报 | 变体误报 | 两者都误报 | 只有变体 | 只有 vanilla |",
      "| --- | --- | --- | --- | --- | --- |"]
for s in SPLITS:
    k = d.kind == s
    a, b = yes["vanilla"] & k, yes["variant"] & k
    L.append(f"| {s} | {a.sum()} | {b.sum()} | {(a & b).sum()} | {(b & ~a).sum()} | {(a & ~b).sum()} |")
extra = yes["variant"] & ~yes["vanilla"] & neg
L += ["", "只有变体误报的被问物体：" + "，".join(f"{o} {c}" for o, c in Counter(d.q[i]["object"] for i in np.flatnonzero(extra)).most_common(12))]
fixed = ~yes["variant"] & yes["vanilla"] & neg
L += ["只有 vanilla 误报的被问物体：" + "，".join(f"{o} {c}" for o, c in Counter(d.q[i]["object"] for i in np.flatnonzero(fixed)).most_common(12))]

L += ["", "## 负样本上的 margin（log P(Yes) − log P(No)）", "", "| 配置 | random 均值 | popular 均值 | adversarial 均值 | 正样本均值 |", "| --- | --- | --- | --- | --- |"]
for n in RUNS:
    m = runs[n]["margin"]
    L.append(f"| {n} | " + " | ".join(f"{m[d.kind == s].mean():+.2f}" for s in SPLITS) + f" | {m[d.lab].mean():+.2f} |")

L += ["", "## contrast_max 与变体的逐题差异", ""]
for s in ("positive",) + SPLITS:
    k = d.kind == s
    a, b = yes["variant"] & k, yes["contrast_max"] & k
    L.append(f"- {s}: 变体 yes {a.sum()}，contrast_max yes {b.sum()}，只有变体 yes {(a & ~b).sum()}，只有 contrast_max yes {(b & ~a).sum()}")
jac = [len(set(runs["variant"]["kept"][i]) & set(runs["contrast_max"]["kept"][i])) / 30 for i in range(d.n)]
L.append(f"- 两者保留集合的重合（30 个中相同的比例）均值 {100 * np.mean(jac):.1f}%")
with open(f"{out}/squeeze_fp.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
