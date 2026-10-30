"""EXP-0028 记录内深挖（探索性，接 locate_dev2.py）：VQAv2 dev2 的不路由题，按“题目是否点名图中有的 COCO 物体”分组，
比较覆盖 16、居中框覆盖 16 与 VisionZip-64；并看 VisionZip 自己有几个主导 token 落在框里。
假设（看过 locate_dev2 之后提出）：VisionZip 的 CLS 注意力偏向显著物体，题目问的若是不显著的区域（天空、墙、招牌等），
它留下的证据少，定位才有用。只读已封存的运行，输出写到 D:/6CJNM007/tmp/exp0028/explore/locate_dev2b.md，不写仓库。
"""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0024 as a24  # noqa: E402
import analyze_0028 as an  # noqa: E402

bench = "vqav2_dev2"
b = an.load_bench(bench)
b["text"] = {str(q["question_id"]): q["text"] for q in map(json.loads, open(an.FILES[bench][0], encoding="utf-8"))}
n = len(b["order"])
sc = {}
for meth in ("vz", "c16", "c16ctr", "t16"):
    exp, rid = an.RUNS[(bench, meth)]
    a = {}
    for line in open(f"{an.OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
        x = json.loads(line)
        a[str(x["question_id"])] = x["text"]
    sc[meth] = an.recompute(bench, b, a)
recs = json.load(open(f"{an.OUT}/EXP-0028/EXP-0028-R13/vz.json"))["records"]
masks = a24.topic_masks(b)
cov = np.array([bool(x["cover"]) for x in recs])
named = np.array([m is not None for m in masks])
colour = b["groups"]["[colour]"]
base_in = np.array([x["baseline_in_box"] if x["cover"] else np.nan for x in recs], float)
side2 = np.array([x["side"] ** 2 if x["cover"] else np.nan for x in recs], float)
by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
rng = np.random.default_rng(0)
draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(an.N_BOOT)]


def paired(x, y, m):
    d = sc[x] - sc[y]
    boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
    return 100 * d[m].mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(m.sum())


fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）"   # noqa: E731
groups = {"点名图中 COCO 物体": cov & named, "没有点名": cov & ~named,
          "没有点名的颜色题": cov & ~named & colour, "没有点名的其他题": cov & ~named & ~colour,
          "点名的颜色题": cov & named & colour}
out = ["# VQAv2 dev2：覆盖的增益按“是否点名 COCO 物体”分组（EXP-0028，探索性）", "",
       "| 组（被覆盖的不路由题） | 题数 | VisionZip 框内主导 token（平均，共 53） | 框占网格 | vz 准确率 | c16 − vz | c16 − c16ctr | c16ctr − vz | t16 − vz |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for name, g in groups.items():
    out.append(f"| {name} | {int(g.sum())} | {np.nanmean(base_in[g]):.1f} | {np.nanmean(side2[g]) / 576:.2f} | {100 * sc['vz'][g].mean():.2f} | "
               f"{fmt(paired('c16', 'vz', g))} | {fmt(paired('c16', 'c16ctr', g))} | {fmt(paired('c16ctr', 'vz', g))} | {fmt(paired('t16', 'vz', g))} |")
words = {}
for i in np.flatnonzero(cov & ~named):
    q = b["text"][b["order"][i]].split("\n")[0].lower().rstrip("?")
    for w in q.split():
        words[w] = words.get(w, 0) + 1
common = [w for w, c in sorted(words.items(), key=lambda t: -t[1])[:40]]
out += ["", "没有点名的题里最常见的词（前 40）：" + "、".join(common)]
open("D:/6CJNM007/tmp/exp0028/explore/locate_dev2b.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
