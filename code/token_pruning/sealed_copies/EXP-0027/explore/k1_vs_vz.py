"""EXP-0027 记录内深挖（看过分析之后，探索性，不是判据）：K = 1 路由 − VisionZip-64 的配对差与区间。

预注册分析（analyze_0027.py）没有算这一对；这里用它的函数与同样的重抽（按图像 1000 次，种子 0），只读已封存的运行，
输出写到 D:/6CJNM007/tmp/exp0027/explore/k1_vs_vz.md，不写仓库。
"""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0027 as an  # noqa: E402

out = ["# K = 1 路由 − VisionZip-64（探索性）", ""]
for bench in an.BENCHES:
    b = an.load_bench(bench)
    n = len(b["order"])
    sc = {}
    for meth in ("router_k1", "visionzip", "router", "sel"):
        exp, rid = an.RUNS[(bench, meth)]
        d = f"{an.OUT}/{exp}/{exp}-{rid}"
        a = {}
        for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
            x = json.loads(line)
            a[str(x["question_id"])] = x["text"]
        sc[meth] = an.recompute(bench, b, a)
    by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(an.N_BOOT)]

    def paired(x, y, m=None):
        d = sc[x] - sc[y]
        m = np.ones(n, bool) if m is None else m
        boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return 100 * d[m].mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), int(m.sum())

    out.append(f"## {bench}")
    out.append("")
    out.append("| 比较 | 分组 | 差 [95% 区间]（题数） |")
    out.append("| --- | --- | --- |")
    for x, y in (("router_k1", "visionzip"),):
        for g, m in [("全部", None)] + list(b["groups"].items()):
            if m is not None and not m.any():
                continue
            v = paired(x, y, m)
            out.append(f"| {x} − {y} | {g} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}） |")
    out.append("")
open("D:/6CJNM007/tmp/exp0027/explore/k1_vs_vz.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
