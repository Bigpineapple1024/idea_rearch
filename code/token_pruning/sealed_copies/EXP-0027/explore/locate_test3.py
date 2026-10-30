"""EXP-0027 记录内深挖（看过分析之后，探索性，不是判据）：VQAv2 test3 上，定位框是否盖住题目点名的 COCO 物体，
覆盖的增益（K = 1 + 覆盖 16 − K = 1 路由）差多少。做法与 analyze_0026 的“定位分组”相同（analyze_0024.topic_masks：
题目点名、图中有的 COCO 物体的 patch；框盖住至少一半记为盖住），重抽同 analyze_0027（按图像 1000 次，种子 0）。
只读已封存的运行，输出写到 D:/6CJNM007/tmp/exp0027/explore/locate_test3.md，不写仓库。
"""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0024 as a24  # noqa: E402
import analyze_0027 as an  # noqa: E402

bench = "vqav2_test3"
b = an.load_bench(bench)
b["text"] = {str(q["question_id"]): q["text"] for q in map(json.loads, open(an.FILES[bench][0], encoding="utf-8"))}
n = len(b["order"])
sc = {}
for meth in ("sel", "router_k1", "visionzip", "router"):
    exp, rid = an.RUNS[(bench, meth)]
    a = {}
    for line in open(f"{an.OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
        x = json.loads(line)
        a[str(x["question_id"])] = x["text"]
    sc[meth] = an.recompute(bench, b, a)
recs = json.load(open(f"{an.OUT}/EXP-0027/EXP-0027-R01/zoom.json"))["records"]
masks = a24.topic_masks(b)
hit, has = np.zeros(n, bool), np.zeros(n, bool)
for i, (m, x) in enumerate(zip(masks, recs)):
    if m is not None and x["cover"]:
        has[i], hit[i] = True, (m & an.box_mask(x["grid_box"])).sum() / m.sum() >= 0.5
by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
rng = np.random.default_rng(0)
draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(an.N_BOOT)]


def paired(x, y, m):
    d = sc[x] - sc[y]
    boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
    return 100 * d[m].mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(m.sum())


out = ["# VQAv2 test3：定位框是否盖住话题物体（探索性）", "",
       f"- 不路由、题目点名图中 COCO 物体的题：{int(has.sum())}；定位框盖住话题物体至少一半：{hit[has].mean():.3f}", "",
       "| 比较 | 盖住 | 没盖住 |", "| --- | --- | --- |"]
fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）"   # noqa: E731
for x, y in (("sel", "router_k1"), ("sel", "visionzip"), ("router_k1", "visionzip"), ("router_k1", "router")):
    out.append(f"| {x} − {y} | {fmt(paired(x, y, has & hit))} | {fmt(paired(x, y, has & ~hit))} |")
open("D:/6CJNM007/tmp/exp0027/explore/locate_test3.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
