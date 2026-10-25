"""Step 9 pre-check (exploratory, dev set only): could a per-question choice between two existing selections help?
Each question's answer under a configuration is deterministic, so "use selection H when a presence statistic s is above
tau, else selection L" is evaluated exactly from the EXP-0013 runs. tau is chosen on one fold of images and scored on
the other (folds as in analyze_fp.py); the gated answers are compared with the variant (R03) by the image bootstrap.

  python gate_dev.py <out folder>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_fp import N_IMG, RUNS, Dev3, auc, load_run   # noqa: E402
from analyze_object_boxes import COCO, padding_patches   # noqa: E402
from analyze_repope import pope_no   # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=False)
d = Dev3()
runs = {n: load_run(n) for n in RUNS}
yes = {n: np.array([not pope_no(a["text"]) for a in r["answers"]]) for n, r in runs.items()}
coco = json.load(open(COCO, encoding="utf-8"))
images = {im["file_name"]: im for im in coco["images"]}
nonpad = {img: np.setdiff1d(np.arange(N_IMG), padding_patches(images[img])) for img in d.images}
cos = runs["variant"]["scores"]
con = runs["contrast_max"]["scores"]
nz = runs["vanilla"]["names"]
names = nz["names"].tolist()
ncos, nidx = nz["cos"].astype(np.float64), nz["index"]
asked = np.array([names.index(q["object"]) for q in d.q])
stats = {"max_cos": [], "max_minus_mean_cos": [], "top5_cos": [], "best_name_share": [], "max_contrast": [], "top5_contrast": []}
for i in range(d.n):
    c = cos[i, nonpad[d.img[i]]]
    srt = np.sort(c)[::-1]
    stats["max_cos"].append(srt[0])
    stats["max_minus_mean_cos"].append(srt[0] - c.mean())
    stats["top5_cos"].append(srt[:5].mean())
    stats["best_name_share"].append(float(np.mean(ncos[nidx[i]][nonpad[d.img[i]]].argmax(1) == asked[i])))
    k = np.sort(con[i, nonpad[d.img[i]]])[::-1]
    stats["max_contrast"].append(k[0])
    stats["top5_contrast"].append(k[:5].mean())
stats = {k: np.array(v) for k, v in stats.items()}
L = ["# 第 ⑨ 步前的离线诊断：按题在两种选择之间切换（开发集，探索性）", "",
     "## 物体在不在的信号：区分正负样本的 AUC（全部 3600 题）", "", "| 统计量 | AUC | random | popular | adversarial |", "| --- | --- | --- | --- | --- |"]
for k, v in stats.items():
    L.append(f"| {k} | {auc(v, d.lab):.3f} | " + " | ".join(f"{auc(v[d.member[s]], d.lab[d.member[s]]):.3f}" for s in ("random", "popular", "adversarial")) + " |")

pairs = [("variant", "agnostic"), ("variant", "mix10"), ("variant", "mix15"), ("contrast_max", "mix10"), ("contrast_max", "agnostic")]
L += ["", "## 按题切换：s > τ 用 H，否则用 L（τ 两折交叉选择；与变体的配对差按图像重抽）", "",
      "| H / L | 统计量 | F1 | 与变体的差 [95% 区间] | 用 H 的比例 | popular + adversarial 误报 | 召回 |", "| --- | --- | --- | --- | --- | --- | --- |"]
best = []
for hi, lo in pairs:
    for k, v in stats.items():
        cand = np.quantile(v, np.linspace(0, 1, 41))
        gated = np.zeros(d.n, bool)
        use_h = np.zeros(d.n, bool)
        for fold in (0, 1):
            m = d.fold == fold
            scores = [d.f1(np.where(v > t, yes[hi], yes[lo]), m)[0] for t in cand]
            t = cand[int(np.argmax(scores))]
            other = d.fold == 1 - fold
            gated[other] = np.where(v[other] > t, yes[hi][other], yes[lo][other])
            use_h[other] = v[other] > t
        f1 = d.f1(gated)[0]
        diff = d.boot_diff(gated, yes["variant"])
        fp = int((gated & np.isin(d.kind, ["popular", "adversarial"])).sum())
        rec = 100 * gated[d.kind == "positive"].mean()
        best.append((f1, hi, lo, k))
        L.append(f"| {hi} / {lo} | {k} | {f1:.2f} | {diff[0]:+.2f} [{diff[1]:+.2f}, {diff[2]:+.2f}] | {100 * use_h.mean():.0f}% | {fp} | {rec:.1f}% |")
L += ["", f"参照：变体 F1 {d.f1(yes['variant'])[0]:.2f}，vanilla {d.f1(yes['vanilla'])[0]:.2f}。",
      "说明：这是对已有运行答案的离线组合，τ 两折交叉选择；统计量都在第 2 层前可得（输入 embedding），不用答案或标签。"]
with open(f"{out}/gate_dev.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
