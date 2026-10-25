"""EXP-0027 (plan v1): confirmation of IDEA-0005's selected configuration (K = 1, keep 47, cover 16 on the unrouted
questions) on VQAv2 test3 and GQA testdev-balanced at the 64 budget.

  python analyze_0027.py      # writes code/output/EXP-0027/analysis/{0027.md,0027.json}

Q1 (primary): selected - K = 1 router, 95% interval above 0 on both sets; Q2: selected - K = 2 router (IDEA-0003), both
intervals above 0; Q3 (described only): selected - VisionZip-64. GQA testdev reuses EXP-0022's K = 2 router (R02),
vanilla (R06) and VisionZip-64 (R07). Per-question scores are recomputed from answers.jsonl (LLaVA's m4c evaluator on
the ten answers for VQAv2, exact match after rstrip('.').lower() for GQA) and checked against each run's own score
file. Paired differences resample images with replacement (1000 draws, seed 0) within each set.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from squeeze_0018 import EVAL_DATA, OUT   # noqa: E402

LLAVA = "D:/6CJNM007/idea_rearch/source/LLaVA"
GRID = 24
BENCHES = ("vqav2_test3", "gqa_testdev")
FILES = {"vqav2_test3": (f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test3.jsonl", f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test3_annotations.jsonl"),
         "gqa_testdev": (f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced.jsonl", f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced_annotations.jsonl")}
# (bench, method) -> (experiment, run ID)
RUNS = {("vqav2_test3", "sel"): ("EXP-0027", "R01"), ("vqav2_test3", "router_k1"): ("EXP-0027", "R02"),
        ("vqav2_test3", "router"): ("EXP-0027", "R03"), ("vqav2_test3", "visionzip"): ("EXP-0027", "R04"),
        ("gqa_testdev", "sel"): ("EXP-0027", "R05"), ("gqa_testdev", "router_k1"): ("EXP-0027", "R06"),
        ("gqa_testdev", "router"): ("EXP-0022", "R02"), ("gqa_testdev", "visionzip"): ("EXP-0022", "R07"),
        ("gqa_testdev", "vanilla"): ("EXP-0022", "R06")}
KEEP = {"sel": 47, "router_k1": 47, "router": 30}         # FastV-family runs with kept-token logs
COVER = 16
N_BOOT = 1000


def load_bench(bench):
    """Questions in file order with the fields the breakdowns need; ids as strings (as analyze_0024.load_bench)."""
    import run_eval
    qfile, afile = FILES[bench]
    qs = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    gts = {str(a["question_id"]): a for a in map(json.loads, open(afile, encoding="utf-8"))}
    order = [str(q["question_id"]) for q in qs]
    if sorted(order) != sorted(gts):
        sys.exit(f"{bench}: questions and annotations differ")
    b = {"order": order, "gts": gts, "routed": np.array([run_eval.routes_to_query(q["text"]) for q in qs])}
    images = sorted({gts[q]["image"] for q in order})
    idx = {im: k for k, im in enumerate(images)}
    b["images"], b["img_of"] = images, np.array([idx[gts[q]["image"]] for q in order])
    groups = {}
    if bench.startswith("vqav2"):
        kind = np.array([gts[q]["answer_type"] for q in order])
        groups["[colour]"] = np.array([gts[q]["question_type"].startswith("what color") for q in order])
    else:
        kind = np.array(["yes/no" if gts[q]["answer"] in ("yes", "no") else "other" for q in order])
        for s in sorted({gts[q]["structural"] for q in order}):
            groups[f"[structural {s}]"] = np.array([gts[q]["structural"] == s for q in order])
        for s in sorted({gts[q]["semantic"] for q in order}):
            groups[f"[semantic {s}]"] = np.array([gts[q]["semantic"] == s for q in order])
    b["groups"] = {**{f"[{k}]": kind == k for k in sorted(set(kind))}, **groups}
    b["groups"]["[unrouted]"] = ~b["routed"]
    return b


def recompute(bench, b, answers):
    if bench.startswith("vqav2"):
        sys.path.insert(0, LLAVA)
        from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
        ev = TextVQAAccuracyEvaluator()
        return np.array([ev._compute_answer_scores(b["gts"][q]["answers"]).get(ev.answer_processor(answers[q]), 0.0) for q in b["order"]])
    return np.array([float(answers[q].rstrip(".").lower() == b["gts"][q]["answer"]) for q in b["order"]])


def runs_rows():
    rows = {}
    for exp in ("EXP-0022", "EXP-0027"):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for (bench, meth), (exp, rid) in RUNS.items():
        if rows.get(f"{exp}-{rid}", {}).get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
    return rows


def box_mask(bx):
    m = np.zeros((GRID, GRID), bool)
    r0, c0, r1, c1 = bx
    m[r0:r1 + 1, c0:c1 + 1] = True
    return m.ravel()


def main():
    out_dir = f"{OUT}/EXP-0027/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    rows = runs_rows()
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "in_box": {}}
    for bench in BENCHES:
        b = load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, ans = {}, {}
        for (bb, meth), (exp, rid) in RUNS.items():
            if bb != bench:
                continue
            d = f"{OUT}/{exp}/{exp}-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            s = recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {}
            for line in open(f"{d}/{logf}", encoding="utf-8"):
                x = json.loads(line)
                logged[str(x["question_id"])] = x["score"]
            checks[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            r = rows[f"{exp}-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth], ans[meth] = s, [a[q] for q in b["order"]]
            if meth in KEEP:
                k = json.load(open(f"{d}/kept_tokens.json"))
                checks[f"{bench} {meth}: one kept record per question"] = float(k["n_prefills"] == n)
                checks[f"{bench} {meth}: {KEEP[meth]} image tokens kept in every prefill"] = float(np.mean([len(x) == KEEP[meth] for x in k["kept"]]))
                checks[f"{bench} {meth}: logged route decisions equal the rule on the question text"] = float(np.mean(np.array(k["routed"]) == routed))
                if meth == "sel":
                    z = json.load(open(f"{d}/zoom.json"))
                    recs = z["records"]
                    checks[f"{bench} sel: one cover record per question, in order"] = float(
                        z["n_prefills"] == n and [x["index"] for x in recs] == list(range(n)))
                    covered = np.array([bool(x["cover"]) for x in recs])
                    checks[f"{bench} sel: routed questions untouched, every unrouted one covered"] = float(
                        np.mean([(x.get("skip") == "routed") == bool(rt) for x, rt in zip(recs, routed)]) * ((covered == ~routed).mean() >= 0.99))
                    inb = [sum(box_mask(x["grid_box"])[p] for p in k["kept"][i]) == x["in_box"] and x["in_box"] <= min(x["baseline_in_box"] + COVER, 47)
                           for i, x in enumerate(recs) if x["cover"]]
                    checks[f"{bench} sel: kept in-box count = the logged count, at most the baseline's + 16"] = float(np.mean(inb)) if inb else 0.0
                    full = [x["in_box"] == min(x["baseline_in_box"] + COVER, 47) for x in recs if x["cover"]]
                    checks[f"{bench} sel: in-box count reaches the baseline's + 16 (capped) on >= 99% of covered questions"] = float(np.mean(full) >= 0.99)
                    res["in_box"][bench] = {"mean_in_box": float(np.mean([x["in_box"] for x in recs if x["cover"]])),
                                            "mean_changed": float(np.mean([x["changed"] for x in recs if x["cover"]])),
                                            "covered_share": float(covered.mean())}
        same = [ans["sel"][i] == ans["router_k1"][i] for i in range(n) if routed[i]]
        checks[f"{bench} sel: answers on the routed questions = router_k1's"] = float(np.mean(same)) if same else 1.0
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

        pairs = [("sel", "router_k1"), ("sel", "router"), ("sel", "visionzip"), ("router_k1", "router"), ("router", "visionzip")]
        if "vanilla" in sc:
            pairs += [("vanilla", "sel")]
        for x, y in pairs:
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items()}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    P = res["pairs"]
    q1 = {bench: P[f"{bench} sel - router_k1"] for bench in BENCHES}
    q2 = {bench: P[f"{bench} sel - router"] for bench in BENCHES}
    q3 = {bench: P[f"{bench} sel - visionzip"] for bench in BENCHES}
    n1 = sum(x[1] > 0 for x in q1.values())
    Q1 = "holds" if n1 == 2 else ("partly" if n1 == 1 else "fails")
    Q2 = all(x[1] > 0 for x in q2.values())
    above_vz = {bench: x[1] > 0 for bench, x in q3.items()}
    if Q1 == "holds":
        verdict = "Q1 成立：覆盖在新数据上成立；方案写成“K = 1 的路由 + 定位后覆盖”，K = 1 作为预算计法的事实报告"
    elif Q2:
        verdict = (("Q1 部分成立" if Q1 == "partly" else "Q1 不成立") + "、Q2 成立：增益主要来自 K = 1；覆盖不能作为方法的贡献"
                   "（Q1 部分成立时如实写出在哪个集上成立）")
    else:
        verdict = "Q1、Q2 都不成立：回到第 ⑨ 步"
    res["criteria"] = {"Q1": Q1, "Q1_pairs": q1, "Q2": Q2, "Q2_pairs": q2, "Q3_pairs": q3, "above_visionzip": above_vz, "verdict": verdict}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0027.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0027（计划 v1）：定位后覆盖的确认（VQAv2 test3、GQA testdev，64 档）", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {m} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "覆盖记录：" + "；".join(f"{k} {json.dumps(x)}" for k, x in res["in_box"].items()),
          "", "每题秒数：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    names = {"holds": "成立", "partly": "部分成立", "fails": "不成立"}
    L += ["", "## 按计划写定的判据", "",
          f"- **Q1**（所选 − K = 1 路由）：VQAv2 test3 {fmt(q1['vqav2_test3'])}，GQA testdev {fmt(q1['gqa_testdev'])} → {names[Q1]}",
          f"- **Q2**（所选 − K = 2 路由）：VQAv2 test3 {fmt(q2['vqav2_test3'])}，GQA testdev {fmt(q2['gqa_testdev'])} → {'成立' if Q2 else '不成立'}",
          f"- **Q3**（描述，所选 − VisionZip-64）：VQAv2 test3 {fmt(q3['vqav2_test3'])}，GQA testdev {fmt(q3['gqa_testdev'])}；"
          + "，".join(f"{k}{'高于' if v else '不能写高于'} VisionZip-64" for k, v in above_vz.items()),
          f"- **判读**：{verdict}"]
    open(f"{out_dir}/0027.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
