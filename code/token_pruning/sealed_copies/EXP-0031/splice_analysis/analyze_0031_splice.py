"""EXP-0031 plan v1.1 (written 2026-10-29 while EXP-0031 ran, before any of its results was seen): a secondary,
pre-registered analysis of the two-base splice on the fresh sets. The splice came from exploring EXP-0028's dev runs
(routed questions take the K = 1 router, the others VisionZip-64 + cover 16; +1.96 / +1.17 over VisionZip-64 on VQAv2
dev2 / GQA val400 in analyze_0030) and was never pre-registered. EXP-0031's runs hold both halves on the fresh sets, so
no new run is needed: routed questions from R07 / R08 (K = 1 router + cover 16; the cover is only on unrouted questions,
so on routed ones it is the K = 1 router) and unrouted ones from R05 / R06 (VisionZip-64 + cover 16 on every question; on
unrouted questions the same computation as the cover on unrouted questions only).

  python analyze_0031_splice.py             # after EXP-0031's analysis is sealed: code/output/EXP-0031/splice_analysis/
  python analyze_0031_splice.py --dev DIR   # the same composition on the dev runs, written to DIR (in tmp); it must
                                            # reproduce analyze_0030's splice (accuracy and interval) exactly

S1 (secondary; EXP-0031's verdict does not change): splice - VisionZip-64 on VQAv2 test4 and GQA val400b, 95% interval
above 0 on both sets: holds; on one: partly; on neither: fails. S2: splice - the selected configuration (IDEA-0007),
read the same way. Reported: splice - VisionZip-64 + cover 16 on every question, splice - K = 1 router + cover 16, by
question group. Per-question scores are recomputed from answers.jsonl and checked against each run's own score file;
paired differences resample images with replacement (1000 draws, seed 0) within each set, as analyze_0030.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_0028 as a28  # noqa: E402
import analyze_0031 as a31  # noqa: E402

OUT = a28.OUT
N_BOOT = 1000
FRESH = {"sets": a31.GENERAL,
         "runs": {"vz": ("R03", "R04"), "c16all": ("R05", "R06"), "cp": ("R07", "R08"), "sel": ("R01", "R02")},
         "exp": {"vz": "EXP-0031", "c16all": "EXP-0031", "cp": "EXP-0031", "sel": "EXP-0031"}}
DEV = {"sets": a28.BENCHES,
       "runs": {"vz": ("R01", "R02"), "c16all": ("R19", "R20"), "cp": ("R03", "R04"), "sel": ("R05", "R06"),
                "router_k1": ("R06", "R07"), "c16": ("R13", "R14")},
       "exp": {"vz": "EXP-0028", "c16all": "EXP-0028", "cp": "EXP-0026", "sel": "EXP-0030", "router_k1": "EXP-0024",
               "c16": "EXP-0028"}}
CONFIG = {"vz": "visionzip64", "c16all": "visionzip64_c16all", "cp": "fastv64k1_qroute_c16_kl", "sel": "fastv64k1_qroute_cls_c16_kl",
          "router_k1": "fastv64k1_qroute_kl", "c16": "visionzip64_c16"}
NAMES = {"splice": "拼接（被路由的题 K = 1 路由，其余 VisionZip-64 + 覆盖 16）", "vz": "VisionZip-64",
         "c16all": "VisionZip-64 + 每题覆盖 16（IDEA-0006 的所选）", "cp": "K = 1 路由 + 覆盖 16（IDEA-0005 的所选）",
         "sel": "IDEA-0007 的所选（k1qcls_c16）"}


def run_rows(spec):
    rows = {}
    for exp in sorted(set(spec["exp"].values())):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    return rows


def main():
    dev = "--dev" in sys.argv
    spec = DEV if dev else FRESH
    out_dir = sys.argv[sys.argv.index("--dev") + 1] if dev else f"{OUT}/EXP-0031/splice_analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    if not dev:
        if not os.path.exists(f"{OUT}/EXP-0031/analysis/SHA256SUMS"):
            sys.exit("EXP-0031's analysis is not sealed yet (plan v1.1: the splice is analysed after it)")
        if json.load(open(f"{OUT}/EXP-0030/analysis/0030.json", encoding="utf-8"))["criteria"]["selected"] != "k1qcls_c16":
            sys.exit("EXP-0030's selection is not k1qcls_c16")
        a28.FILES.update(a31.FILES)
    rows = run_rows(spec)
    checks, res = {}, {"table": {}, "pairs": {}, "dev": dev}
    for i, bench in enumerate(spec["sets"]):
        b = a28.load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, d_of = {}, {}
        for meth, rids in spec["runs"].items():
            exp, rid = spec["exp"][meth], rids[i]
            r = rows.get(f"{exp}-{rid}", {})
            if r.get("status") != "completed" or r.get("config") != CONFIG[meth] or r.get("bench") != bench:
                sys.exit(f"{exp}-{rid}: {r.get('config')} x {r.get('bench')} ({r.get('status')}), planned {CONFIG[meth]} x {bench}")
            d = d_of[meth] = f"{OUT}/{exp}/{exp}-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            checks[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = a28.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {}
            for line in open(f"{d}/{logf}", encoding="utf-8"):
                x = json.loads(line)
                logged[str(x["question_id"])] = x["score"]
            checks[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            sc[meth] = s
        # the two halves: the counterpart covers no routed question, VisionZip-64 + cover 16 covers the unrouted ones
        zr = json.load(open(f"{d_of['cp']}/zoom.json"))["records"]
        vr = json.load(open(f"{d_of['c16all']}/vz.json"))["records"]
        checks[f"{bench} cp: one cover record per question; no routed question covered"] = float(
            len(zr) == n and not any(zr[k]["cover"] for k in range(n) if routed[k]))
        checks[f"{bench} c16all: one cover record per question; >= 99% of the unrouted questions covered"] = float(
            len(vr) == n and np.mean([vr[k]["cover"] for k in range(n) if not routed[k]]) >= 0.99)
        if dev:                      # the composition equals analyze_0030's splice of the K = 1 router and cover-16 runs
            checks[f"{bench}: routed questions: K = 1 router + cover 16 scores = K = 1 router's"] = float(
                np.array_equal(sc["cp"][routed], sc["router_k1"][routed]))
            checks[f"{bench}: unrouted questions: cover 16 on every question scores = cover 16 on unrouted ones'"] = float(
                np.array_equal(sc["c16all"][~routed], sc["c16"][~routed]))
        sc["splice"] = np.where(routed, sc["cp"], sc["c16all"])
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

        for y in ("vz", "sel", "c16all", "cp"):
            res["pairs"][f"{bench} splice - {y}"] = paired("splice", y)
            for g, m in b["groups"].items():
                if m.any():
                    res["pairs"][f"{bench} splice - {y} {g}"] = paired("splice", y, m)
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items() if v in NAMES}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    P = res["pairs"]
    sets = spec["sets"]

    def reading(y):
        up = sum(P[f"{bench} splice - {y}"][1] > 0 for bench in sets)
        return {2: "成立", 1: "部分成立", 0: "不成立"}[up]

    res["S1"], res["S2"], res["checks"] = reading("vz"), reading("sel"), checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/splice.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0031 计划 v1.1：拼接的预注册二级分析（{'开发集上的复算，不是结果' if dev else '新抽的数据'}：{'、'.join(sets)}）", "",
         "## 核对", ""] + [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    for bench in sets:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {NAMES[m]} | {x['accuracy']:.2f} | " + " | ".join(f"{x[g]:.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 写定的读法（二级，不改 EXP-0031 的判读）", "", f"| 比较 | {' | '.join(sets)} | 读法 |", "| --- |" + " --- |" * (len(sets) + 1)]
    for y, key in (("vz", "S1"), ("sel", "S2")):
        L.append(f"| 拼接 − {NAMES[y]} | " + " | ".join(fmt(P[f'{bench} splice - {y}']) for bench in sets) + f" | {key}：{res[key]} |")
    L += ["", "## 全部配对差（含分组；百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items()]
    open(f"{out_dir}/splice.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
