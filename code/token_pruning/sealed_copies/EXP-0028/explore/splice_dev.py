"""EXP-0028 记录内深挖（看过分析之后，探索性，不是判据）：开发集上，按 IDEA-0003 的题目句式规则把两种底座拼起来——
不路由的题用 VisionZip-64 + 覆盖 16（EXP-0028-R13、R14），被路由的题（yes/no、how many）用 FastV K = 1 路由
（EXP-0026-R03、R04 的被路由题，覆盖不碰这些题，与 K = 1 路由逐题相同）——与 VisionZip-64、每题覆盖 16、FastV K = 1 + 覆盖 16 比较。
两个配置的层平均都是 64；拼接等于按题目文字分流的系统（贪心解码、batch 1，每题独立）。
只读已封存的运行，输出写到 D:/6CJNM007/tmp/exp0028/explore/splice_dev.md，不写仓库。
"""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as an  # noqa: E402

out = ["# 开发集：按题目句式拼接两种底座（EXP-0028，探索性）", "",
       "拼接 = 不路由的题取 VisionZip-64 + 覆盖 16，被路由的题取 FastV K = 1 路由。", "",
       "| 比较 | VQAv2 dev2 | GQA val400 |", "| --- | --- | --- |"]
rows = {}
for bench in an.BENCHES:
    b = an.load_bench(bench)
    sc = {}
    for meth in ("vz", "c16", "c16all", "fk1c16"):
        exp, rid = an.RUNS[(bench, meth)]
        a = {}
        for line in open(f"{an.OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
            x = json.loads(line)
            a[str(x["question_id"])] = x["text"]
        sc[meth] = an.recompute(bench, b, a)
    rt = b["routed"]
    sc["splice"] = np.where(rt, sc["fk1c16"], sc["c16"])
    by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(an.N_BOOT)]

    def paired(x, y, m=None):
        m = np.ones(len(b["order"]), bool) if m is None else m
        d = sc[x] - sc[y]
        boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return f"{100 * d[m].mean():+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]（{int(m.sum())}）"

    rows.setdefault("准确率：拼接", []).append(f"{100 * sc['splice'].mean():.2f}")
    for x, y in (("splice", "vz"), ("splice", "c16all"), ("splice", "fk1c16")):
        rows.setdefault(f"{x} − {y}", []).append(paired(x, y))
    rows.setdefault("fk1c16 − vz [routed]", []).append(paired("fk1c16", "vz", rt))
    rows.setdefault("c16 − fk1c16 [unrouted]", []).append(paired("c16", "fk1c16", ~rt))
    rows.setdefault("被路由的题占比", []).append(f"{rt.mean():.3f}")
for k, v in rows.items():
    out.append(f"| {k} | {v[0]} | {v[1]} |")
open("D:/6CJNM007/tmp/exp0028/explore/splice_dev.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
