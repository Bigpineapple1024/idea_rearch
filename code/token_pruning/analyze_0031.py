"""EXP-0031 (plan v1, run only when EXP-0030's verdict is "promising"): confirmation of IDEA-0007 (one full LLM layer,
then VisionZip's ranking) on fresh data, VQAv2 test4 and GQA val400b (make_confirm_sets.py, drawn before any EXP-0030
result), with TextVQA (no OCR tokens) described, at the 64 budget (63.5 tokens per layer).

  python analyze_0031.py      # writes code/output/EXP-0031/analysis/{0031.md,0031.json}

The selected configuration is read from EXP-0030's sealed analysis (criteria.selected: k1cls, k1qcls or k1qcls_c16);
the plan was written before any EXP-0030 result. Q1 (primary): selected - VisionZip-64, 95% interval above 0 on both
fresh sets: holds; on one: partly; on neither: fails. Q2: selected - IDEA-0006's selected configuration (VisionZip-64 +
cover 16 on every question). Q3 (ranking): selected - its FastV counterpart at K = 1, which keeps the same routing and
cover and ranks by FastV's attention instead of the encoder's CLS attention (the K = 1 router for k1cls and k1qcls, the
K = 1 router + cover 16 for k1qcls_c16; for k1cls the routed questions differ too). Q2 and Q3 are read as Q1 (both,
one, neither). Q4 (described): selected - VisionZip-64 and - IDEA-0006's selected configuration on TextVQA without OCR
tokens (EXP-0029-R03, R04). Large (reported, not part of the verdict): the two-set mean of selected - VisionZip-64 is at
least +2.0. Per-question scores are recomputed from answers.jsonl and checked against each run's own score file; paired
differences resample images with replacement (1000 draws, seed 0) within each set.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_0025 as a25  # noqa: E402
import analyze_0028 as a28  # noqa: E402
from fastv_llava import _padding_mask  # noqa: E402
from squeeze_0018 import EVAL_DATA, OUT   # noqa: E402

SELECTIONS = {"k1cls": {"config": "fastv64k1_cls_kl", "cp": "router_k1", "cp_config": "fastv64k1_qroute_kl"},
              "k1qcls": {"config": "fastv64k1_qroute_cls_kl", "cp": "router_k1", "cp_config": "fastv64k1_qroute_kl"},
              "k1qcls_c16": {"config": "fastv64k1_qroute_cls_c16_kl", "cp": "fk1c16", "cp_config": "fastv64k1_qroute_c16_kl"}}
GENERAL = ("vqav2_test4", "gqa_val400b")
BENCHES = GENERAL + ("textvqa_noocr",)
FILES = {"vqav2_test4": (f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test4.jsonl", f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test4_annotations.jsonl",
                         "D:/6CJNM007/data/coco/val2014"),
         "gqa_val400b": (f"{EVAL_DATA}/gqa/llava_gqa_val400b.jsonl", f"{EVAL_DATA}/gqa/llava_gqa_val400b_annotations.jsonl",
                         "D:/6CJNM007/data/gqa/images_val400b")}
SCORES = {"vqav2_test4": "vqa_scores.jsonl", "gqa_val400b": "gqa_scores.jsonl", "textvqa_noocr": "textvqa_scores.jsonl"}
# (bench, method) -> (experiment, run ID); "cp" is the selected configuration's FastV counterpart
RUNS = {("vqav2_test4", "sel"): ("EXP-0031", "R01"), ("gqa_val400b", "sel"): ("EXP-0031", "R02"),
        ("vqav2_test4", "vz"): ("EXP-0031", "R03"), ("gqa_val400b", "vz"): ("EXP-0031", "R04"),
        ("vqav2_test4", "c16all"): ("EXP-0031", "R05"), ("gqa_val400b", "c16all"): ("EXP-0031", "R06"),
        ("vqav2_test4", "cp"): ("EXP-0031", "R07"), ("gqa_val400b", "cp"): ("EXP-0031", "R08"),
        ("textvqa_noocr", "sel"): ("EXP-0031", "R09"),
        ("textvqa_noocr", "vz"): ("EXP-0029", "R03"), ("textvqa_noocr", "c16all"): ("EXP-0029", "R04"),
        ("textvqa_noocr", "router"): ("EXP-0025", "R01"), ("textvqa_noocr", "vanilla"): ("EXP-0025", "R07")}
KEEP = 47
N_BOOT = 1000
LARGE = 2.0


def names(sel):
    cp = {"router_k1": "K = 1 路由（FastV 的注意力排序）", "fk1c16": "K = 1 路由 + 覆盖 16（FastV 的注意力排序）"}[SELECTIONS[sel]["cp"]]
    return {"sel": f"所选（IDEA-0007：{sel}）", "vz": "VisionZip-64", "c16all": "VisionZip-64 + 每题覆盖 16（IDEA-0006 的所选）",
            "cp": f"对应的 FastV 配置：{cp}", "router": "K = 2 路由（IDEA-0003）", "vanilla": "不剪枝"}


def selection():
    """EXP-0030's selected candidate; stops unless its verdict was promising (the plan's condition)."""
    c = json.load(open(f"{OUT}/EXP-0030/analysis/0030.json", encoding="utf-8"))["criteria"]
    if not c["verdict"].startswith("有希望"):
        sys.exit(f"EXP-0030's verdict is not promising: {c['verdict']}")
    return c["selected"]


def runs_rows(sel):
    rows = {}
    for exp in ("EXP-0025", "EXP-0029", "EXP-0031"):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for (bench, meth), (exp, rid) in RUNS.items():
        if rows.get(f"{exp}-{rid}", {}).get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
    S = SELECTIONS[sel]
    want = {"R01": S["config"], "R02": S["config"], "R09": S["config"], "R03": "visionzip64", "R04": "visionzip64",
            "R05": "visionzip64_c16all", "R06": "visionzip64_c16all", "R07": S["cp_config"], "R08": S["cp_config"]}
    want = {f"EXP-0031-{r}": c for r, c in want.items()}
    want.update({"EXP-0029-R03": "visionzip64", "EXP-0029-R04": "visionzip64_c16all"})
    bad = {r: rows[r]["config"] for r, c in want.items() if rows[r]["config"] != c}
    if bad:
        sys.exit(f"runs with another configuration than planned: {bad}")
    return rows


def load(bench):
    if bench in GENERAL:
        return a28.load_bench(bench)
    b = a25.load_bench(bench)
    sizes = a25.image_sizes(bench, b)
    b["size"] = [sizes[b["image"][q]] for q in b["order"]]
    return b


def recompute(bench, b, answers):
    return a28.recompute(bench, b, answers) if bench in GENERAL else a25.recompute(bench, b, answers)


def check_sel(checks, res, bench, sel, b, d, ref):
    """The implementation checks of the selected CLS run (plan v1, as analyze_0030.check_cls). ref: VisionZip's own
    dominant patches (from the VisionZip-64 + cover 16 run on the same set), the reference boxes and, on the fresh
    sets, the counterpart's kept tokens."""
    n, routed = len(b["order"]), b["routed"]
    cls = json.load(open(f"{d}/cls.json"))
    kept = json.load(open(f"{d}/kept_tokens.json"))
    recs, kk = cls["records"], kept["kept"]
    checks[f"{bench} sel: one CLS record and one kept list per prefill, in order"] = float(
        cls["n_prefills"] == n == len(recs) == len(kk) and [r["index"] for r in recs] == list(range(n)))
    used = np.array([bool(r["used"]) for r in recs])
    want = np.ones(n, bool) if sel == "k1cls" else ~routed
    checks[f"{bench} sel: ranked by the CLS score exactly where planned (rule on the question text)"] = float(np.mean(used == want))
    if sel != "k1cls":
        checks[f"{bench} sel: the kept log's routing decisions = the rule on the question text"] = float(
            np.mean(np.array(kept["routed"]) == routed))
    pad = [~_padding_mask(w, h, 24).numpy() for w, h in b["size"]]
    checks[f"{bench} sel: 47 kept patches per prefill, none wholly in the padding"] = float(np.mean(
        [len(k) == KEEP and len(set(k)) == KEEP and bool(pad[i][k].all()) for i, k in enumerate(kk)]))
    top = float(np.mean([r["top53"] == ref["vz_own"][i] for i, r in enumerate(recs)]))
    checks[f"{bench} sel: the 53 patches with the largest score = VisionZip's own dominant patches (its cover run) on >= 99%"] = float(top >= 0.99)
    res["cls"][f"{bench} sel top53 = VisionZip's"] = top
    if sel in ("k1cls", "k1qcls"):
        checks[f"{bench} sel: where ranked by the score, kept = its own choice"] = float(np.mean(
            [sorted(kk[i]) == recs[i]["own"] for i in range(n) if used[i]]))
    if sel != "k1cls" and "cp_kept" in ref:
        same = [sorted(kk[i]) == sorted(ref["cp_kept"][i]) for i in range(n) if routed[i]]
        checks[f"{bench} sel: routed questions keep what the counterpart keeps (>= 99%)"] = float(np.mean(same) >= 0.99)
        res["cls"][f"{bench} sel routed kept = counterpart's"] = float(np.mean(same))
    if sel == "k1qcls_c16":
        zr = json.load(open(f"{d}/zoom.json"))["records"]
        covered = [i for i in range(n) if zr[i]["cover"]]
        checks[f"{bench} sel: >= 99% of the unrouted questions covered, no routed one"] = float(
            all(not routed[i] for i in covered) and len(covered) >= 0.99 * (~routed).sum())
        both = [i for i in covered if ref["boxes"][i][0]]
        agree = float(np.mean([zr[i]["grid_box"] == ref["boxes"][i][1] for i in both])) if both else 0.0
        checks[f"{bench} sel: box = {ref['box_from']}'s box on >= 99%"] = float(len(both) >= 0.99 * len(covered) and agree >= 0.99)
        res["in_box"][f"{bench} sel box = {ref['box_from']}'s"] = agree
        ok = []
        for i in covered:
            box = a28.box_mask(zr[i]["grid_box"])
            k, own = np.array(kk[i]), np.array(recs[i]["own"])
            inside = int((box & pad[i]).sum())
            a_in, n_in = int(box[own].sum()), int(box[k].sum())
            ok.append(zr[i]["baseline_in_box"] == a_in and zr[i]["in_box"] == n_in == min(a_in + 16, KEEP, inside)
                      and set(own[box[own]]) <= set(k[box[k]]) and set(k[~box[k]]) <= set(own[~box[own]]))
        checks[f"{bench} sel: covered: in-box count = the CLS ranking's own + 16 (capped), more inside, a subset outside"] = float(np.mean(ok))
        rest = [i for i in range(n) if not routed[i] and not zr[i]["cover"]]
        checks[f"{bench} sel: unrouted but not covered: kept = the CLS ranking's own choice"] = float(
            np.mean([sorted(kk[i]) == recs[i]["own"] for i in rest])) if rest else 1.0
        c = [zr[i] for i in covered]
        res["in_box"][f"{bench} sel"] = {
            "covered_share": len(covered) / n, "mean_baseline_in_box": float(np.mean([x["baseline_in_box"] for x in c])),
            "mean_in_box": float(np.mean([x["in_box"] for x in c])), "mean_changed": float(np.mean([x["changed"] for x in c]))}


def main():
    out_dir = f"{OUT}/EXP-0031/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sel = selection()
    S = SELECTIONS[sel]
    NAMES = names(sel)
    a28.FILES.update(FILES)                               # analyze_0028's loader and checks, on the fresh sets
    rows = runs_rows(sel)
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "in_box": {}, "cls": {}}
    for bench in BENCHES:
        b = load(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, ans, d_of = {}, {}, {}
        for (bb, meth), (exp, rid) in RUNS.items():
            if bb != bench:
                continue
            d = d_of[meth] = f"{OUT}/{exp}/{exp}-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            checks[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = recompute(bench, b, a)
            logged = {}
            for line in open(f"{d}/{SCORES[bench]}", encoding="utf-8"):
                x = json.loads(line)
                logged[str(x["question_id"])] = x["score"]
            checks[f"{bench} {meth}: recomputed scores equal {SCORES[bench]}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            r = rows[f"{exp}-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth], ans[meth] = s, [a[q] for q in b["order"]]
        z = json.load(open(f"{d_of['c16all']}/vz.json"))
        checks[f"{bench} c16all: the log counts its prefills and covered ones"] = float(
            z["n_prefills"] == len(z["records"]) and z["n_covered"] == sum(x["cover"] for x in z["records"]))
        a28.check_vz_log(checks, res, bench, "c16all", b, z["records"])
        vzr = z["records"]
        ref = {"vz_own": [x["baseline"] if x["cover"] else x["dominant"] for x in vzr],
               "boxes": [(bool(x["cover"]), x.get("grid_box")) for x in vzr], "box_from": "the VisionZip-64 + cover 16 run"}
        if bench in GENERAL:
            ref["cp_kept"] = json.load(open(f"{d_of['cp']}/kept_tokens.json"))["kept"]
            if S["cp"] == "fk1c16":
                cz = json.load(open(f"{d_of['cp']}/zoom.json"))["records"]
                ref["boxes"], ref["box_from"] = [(bool(x["cover"]), x.get("grid_box")) for x in cz], "the counterpart (K = 1 + cover 16)"
                both = [i for i in range(n) if not routed[i] and vzr[i]["cover"] and cz[i]["cover"]]
                agree = float(np.mean([vzr[i]["grid_box"] == cz[i]["grid_box"] for i in both])) if both else 0.0
                checks[f"{bench} c16all: box = the counterpart's on >= 99% of the unrouted questions"] = float(
                    len(both) >= 0.99 * (~routed).sum() and agree >= 0.99)
                res["in_box"][f"{bench} c16all box = counterpart's"] = agree
            if sel != "k1cls":
                same = [ans["sel"][i] == ans["cp"][i] for i in range(n) if routed[i]]
                checks[f"{bench} sel: answers on the routed questions = the counterpart's (>= 99%)"] = float(np.mean(same) >= 0.99)
                res["cls"][f"{bench} sel routed answers = counterpart's"] = float(np.mean(same))
        check_sel(checks, res, bench, sel, b, d_of["sel"], ref)
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

        if bench in GENERAL:
            pairs = [("sel", "vz"), ("sel", "c16all"), ("sel", "cp"), ("c16all", "vz"), ("cp", "vz"), ("c16all", "cp")]
        else:
            pairs = [("sel", "vz"), ("sel", "c16all"), ("c16all", "vz"), ("sel", "router"), ("vanilla", "sel"), ("vanilla", "vz")]
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

    def count(x, y):
        pr = {bench: P[f"{bench} {x} - {y}"] for bench in GENERAL}
        return pr, sum(v[1] > 0 for v in pr.values())

    q1, n1 = count("sel", "vz")
    q2, n2 = count("sel", "c16all")
    q3, n3 = count("sel", "cp")
    q4 = {"vz": P["textvqa_noocr sel - vz"], "c16all": P["textvqa_noocr sel - c16all"]}
    mean1 = float(np.mean([q1[bench][0] for bench in GENERAL]))
    word = {2: "成立", 1: "部分成立", 0: "不成立"}
    Q1 = {2: "holds", 1: "partly", 0: "fails"}[n1]
    verdict = {"holds": "Q1 成立：IDEA-0007 在新数据上高于 VisionZip-64",
               "partly": "Q1 部分成立：如实写出在哪个集上成立",
               "fails": "Q1 不成立：IDEA-0007 在新数据上不作为方法的贡献"}[Q1]
    rank = "K = 1 路由" if sel == "k1cls" else "同样路由与覆盖下 FastV 的注意力排序"
    if Q1 != "fails":
        verdict += {2: "；Q2 成立：好于 IDEA-0006 的所选", 1: "；Q2 部分成立：好于 IDEA-0006 的所选只在一个集上可检出",
                    0: "；Q2 不成立：不能写好于 IDEA-0006 的所选"}[n2]
        verdict += {2: f"；Q3 成立：好于{rank}", 1: f"；Q3 部分成立：好于{rank}只在一个集上可检出",
                    0: f"；Q3 不成立：不能写好于{rank}"}[n3]
    res["criteria"] = {"selected": sel, "selection": S, "Q1": Q1, "Q1_pairs": q1, "Q1_mean": mean1, "Q2": word[n2], "Q2_pairs": q2,
                       "Q3": word[n3], "Q3_pairs": q3, "Q4_pairs": q4, "Q4_above_visionzip": bool(q4["vz"][1] > 0),
                       "large": mean1 >= LARGE, "verdict": verdict}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0031.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0031（计划 v1）：IDEA-0007 的确认（VQAv2 test4、GQA val400b，新抽的数据；TextVQA 描述；层平均 64 档）", "",
         f"所选（EXP-0030 的选择）：{sel} {json.dumps(S)}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    L += ["", "实测比例：" + "；".join(f"{k} {x:.4f}" for k, x in res["cls"].items())]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {NAMES[m]} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "覆盖记录：" + "；".join(f"{k} {json.dumps(x)}" for k, x in res["in_box"].items()),
          "", "每题秒数：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    L += ["", "## 按计划写定的判据", "",
          f"- **Q1**（所选 − VisionZip-64）：VQAv2 test4 {fmt(q1['vqav2_test4'])}，GQA val400b {fmt(q1['gqa_val400b'])} → {word[n1]}",
          f"- **Q2**（所选 − IDEA-0006 的所选）：VQAv2 test4 {fmt(q2['vqav2_test4'])}，GQA val400b {fmt(q2['gqa_val400b'])} → {word[n2]}",
          f"- **Q3**（所选 − {NAMES['cp']}）：VQAv2 test4 {fmt(q3['vqav2_test4'])}，GQA val400b {fmt(q3['gqa_val400b'])} → {word[n3]}",
          f"- **Q4**（描述，TextVQA 不给 OCR）：所选 − VisionZip-64 {fmt(q4['vz'])}" + ("，高于 VisionZip-64" if q4["vz"][1] > 0 else "，不能写高于 VisionZip-64")
          + f"；所选 − IDEA-0006 的所选 {fmt(q4['c16all'])}",
          f"- **大小（只报告）**：Q1 的两集平均 {mean1:+.2f}，{'达到' if mean1 >= LARGE else '没有达到'}写定的“提升大”（≥ +2.0）",
          f"- **判读**：{verdict}"]
    open(f"{out_dir}/0031.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
