"""EXP-0028 记录内深挖（看过分析之后，探索性，不是判据）：VQAv2 dev2 上，定位框是否盖住题目点名的 COCO 物体，
VisionZip 上覆盖的增益差多少；VisionZip 自己在话题物体上留了几个主导 token，覆盖后又是几个。
做法与 analyze_0026 的定位分组、EXP-0027 的 locate_test3.py 相同（analyze_0024.topic_masks；框盖住物体 patch 的至少一半记为盖住），
重抽同 analyze_0028（按图像 1000 次，种子 0）。只读已封存的运行，输出写到 D:/6CJNM007/tmp/exp0028/explore/locate_dev2.md，不写仓库。
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
for meth in ("vz", "c16", "c16ctr", "t16", "c16all", "fk1c16"):
    exp, rid = an.RUNS[(bench, meth)]
    a = {}
    for line in open(f"{an.OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
        x = json.loads(line)
        a[str(x["question_id"])] = x["text"]
    sc[meth] = an.recompute(bench, b, a)
recs = json.load(open(f"{an.OUT}/EXP-0028/EXP-0028-R13/vz.json"))["records"]       # cover 16
masks = a24.topic_masks(b)
hit, has = np.zeros(n, bool), np.zeros(n, bool)
on_base, on_cov = np.zeros(n), np.zeros(n)          # dominant tokens on the named object: VisionZip's, after cover 16
for i, (m, x) in enumerate(zip(masks, recs)):
    if m is not None and x["cover"]:
        has[i], hit[i] = True, (m & an.box_mask(x["grid_box"])).sum() / m.sum() >= 0.5
        on_base[i], on_cov[i] = m[np.array(x["baseline"])].sum(), m[np.array(x["dominant"])].sum()
by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
rng = np.random.default_rng(0)
draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(an.N_BOOT)]


def paired(x, y, m):
    d = sc[x] - sc[y]
    boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
    return 100 * d[m].mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(m.sum())


fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）"   # noqa: E731
out = ["# VQAv2 dev2：VisionZip 上的定位框是否盖住话题物体（EXP-0028，探索性）", "",
       f"- 不路由、题目点名图中 COCO 物体的题：{int(has.sum())}；定位框盖住话题物体至少一半：{hit[has].mean():.3f}",
       f"- 话题物体上的主导 token（平均）：VisionZip {on_base[has].mean():.2f}，覆盖 16 后 {on_cov[has].mean():.2f}"
       f"（盖住的题 {on_base[has & hit].mean():.2f} → {on_cov[has & hit].mean():.2f}；没盖住 {on_base[has & ~hit].mean():.2f} → {on_cov[has & ~hit].mean():.2f}）", "",
       "| 比较 | 盖住 | 没盖住 |", "| --- | --- | --- |"]
for x, y in (("c16", "vz"), ("c16", "c16ctr"), ("c16ctr", "vz"), ("t16", "vz"), ("c16all", "vz"), ("fk1c16", "vz")):
    out.append(f"| {x} − {y} | {fmt(paired(x, y, has & hit))} | {fmt(paired(x, y, has & ~hit))} |")
out += ["", "按 VisionZip 自己在话题物体上的主导 token 个数分组（盖住的题）：", "", "| VisionZip 在物体上的个数 | 覆盖后（平均） | c16 − vz |", "| --- | --- | --- |"]
for lo, hi, name in ((0, 0, "0"), (1, 4, "1～4"), (5, 999, "≥ 5")):
    g = has & hit & (on_base >= lo) & (on_base <= hi)
    out.append(f"| {name} | {on_cov[g].mean():.2f} | {fmt(paired('c16', 'vz', g))} |")
open("D:/6CJNM007/tmp/exp0028/explore/locate_dev2.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
