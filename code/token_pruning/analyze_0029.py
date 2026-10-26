"""EXP-0029 (plan v1, run only when EXP-0028's verdict is "promising"): confirmation of IDEA-0006 (locate-then-cover
on VisionZip-64) on VQAv2 test3 and GQA testdev-balanced, with TextVQA (no OCR tokens) described, at the 64 budget.

  python analyze_0029.py      # writes code/output/EXP-0029/analysis/{0029.md,0029.json}

The selected configuration is read from EXP-0028's sealed analysis (criteria.selected: c16, c32 or c16all); the plan
was written before any EXP-0028 result. Q1 (primary): selected - VisionZip-64, 95% interval above 0 on both general
sets: holds; on one: partly; on neither: fails. Q2 (localisation): cover 16 - its centred-box control; Q3 (region or
per-token relevance): cover 16 - the no-box control; both read the same way (both intervals above 0, one, neither) and
use the cover-16 pair whatever was selected. Q4 (described): selected - VisionZip-64 on TextVQA without OCR tokens.
VisionZip-64 is EXP-0027-R04 on test3 and EXP-0022-R07 on testdev; FastV K = 1 + cover 16 (IDEA-0005) EXP-0027 R01,
R05; the K = 2 router EXP-0027-R03 / EXP-0022-R02 / EXP-0025-R01; vanilla EXP-0022-R06 / EXP-0025-R07. Per-question
scores are recomputed from answers.jsonl and checked against each run's own score file; paired differences resample
images with replacement (1000 draws, seed 0) within each set.
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
from squeeze_0018 import EVAL_DATA, OUT   # noqa: E402

SELECTIONS = {"c16": {"config": "visionzip64_c16", "cover": 16, "on": "unrouted"},
              "c32": {"config": "visionzip64_c32", "cover": 32, "on": "unrouted"},
              "c16all": {"config": "visionzip64_c16all", "cover": 16, "on": "all"}}
GENERAL = ("vqav2_test3", "gqa_testdev")
BENCHES = GENERAL + ("textvqa_noocr",)
FILES = {"vqav2_test3": (f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test3.jsonl", f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test3_annotations.jsonl",
                         "D:/6CJNM007/data/coco/val2014"),
         "gqa_testdev": (f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced.jsonl", f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced_annotations.jsonl",
                         "D:/6CJNM007/data/gqa/images")}
SCORES = {"vqav2_test3": "vqa_scores.jsonl", "gqa_testdev": "gqa_scores.jsonl", "textvqa_noocr": "textvqa_scores.jsonl"}
NAMES = {"sel": "所选（VisionZip-64 + 定位后覆盖）", "vz": "VisionZip-64", "c16": "覆盖 16", "c16ctr": "居中框覆盖 16",
         "t16": "不围框（与题目最相似的 16 个）", "fk1c16": "FastV K = 1 + 覆盖 16（IDEA-0005）", "router": "K = 2 路由（IDEA-0003）",
         "vanilla": "不剪枝"}
COVER = {"c16": 16, "c16ctr": 16, "t16": 16}
N_BOOT = 1000


def selection():
    """EXP-0028's selected candidate; stops unless its verdict was promising (the plan's condition)."""
    c = json.load(open(f"{OUT}/EXP-0028/analysis/0028.json", encoding="utf-8"))["criteria"]
    if not c["verdict"].startswith("有希望"):
        sys.exit(f"EXP-0028's verdict is not promising: {c['verdict']}")
    return c["selected"]


def run_table(sel):
    """(bench, method) -> (experiment, run ID); cover 16 is the selected run itself when cover 16 was selected."""
    r = {("vqav2_test3", "sel"): ("EXP-0029", "R01"), ("gqa_testdev", "sel"): ("EXP-0029", "R02"),
         ("textvqa_noocr", "vz"): ("EXP-0029", "R03"), ("textvqa_noocr", "sel"): ("EXP-0029", "R04"),
         ("vqav2_test3", "c16ctr"): ("EXP-0029", "R05"), ("gqa_testdev", "c16ctr"): ("EXP-0029", "R06"),
         ("vqav2_test3", "t16"): ("EXP-0029", "R07"), ("gqa_testdev", "t16"): ("EXP-0029", "R08"),
         ("vqav2_test3", "vz"): ("EXP-0027", "R04"), ("gqa_testdev", "vz"): ("EXP-0022", "R07"),
         ("vqav2_test3", "fk1c16"): ("EXP-0027", "R01"), ("gqa_testdev", "fk1c16"): ("EXP-0027", "R05"),
         ("vqav2_test3", "router"): ("EXP-0027", "R03"), ("gqa_testdev", "router"): ("EXP-0022", "R02"),
         ("textvqa_noocr", "router"): ("EXP-0025", "R01"),
         ("gqa_testdev", "vanilla"): ("EXP-0022", "R06"), ("textvqa_noocr", "vanilla"): ("EXP-0025", "R07")}
    if sel != "c16":
        r[("vqav2_test3", "c16")] = ("EXP-0029", "R09")
        r[("gqa_testdev", "c16")] = ("EXP-0029", "R10")
    return r


def runs_rows(runs):
    rows = {}
    for exp in ("EXP-0022", "EXP-0025", "EXP-0027", "EXP-0029"):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for (bench, meth), (exp, rid) in runs.items():
        if rows.get(f"{exp}-{rid}", {}).get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
    return rows


def load(bench):
    return a28.load_bench(bench) if bench in GENERAL else a25.load_bench(bench)


def recompute(bench, b, answers):
    return a28.recompute(bench, b, answers) if bench in GENERAL else a25.recompute(bench, b, answers)


def main():
    out_dir = f"{OUT}/EXP-0029/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sel = selection()
    S = SELECTIONS[sel]
    runs = run_table(sel)
    a28.FILES.update(FILES)                               # analyze_0028's loader and checks, on the confirmation sets
    a28.COVER.update(COVER)
    log_name = "c16all" if S["on"] == "all" else "sel"     # check_vz_log keys its routing rule on c16all
    a28.COVER[log_name] = S["cover"]
    rows = runs_rows(runs)
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "in_box": {}}
    for bench in BENCHES:
        b = load(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, ans, logs = {}, {}, {}
        for (bb, meth), (exp, rid) in runs.items():
            if bb != bench:
                continue
            d = f"{OUT}/{exp}/{exp}-{rid}"
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
            if meth in ("sel", "c16", "c16ctr", "t16"):
                z = json.load(open(f"{d}/vz.json"))
                checks[f"{bench} {meth}: the log counts its prefills and covered ones"] = float(
                    z["n_prefills"] == len(z["records"]) and z["n_covered"] == sum(x["cover"] for x in z["records"]))
                logs[meth] = z["records"]
                a28.check_vz_log(checks, res, bench, log_name if meth == "sel" else meth, b, logs[meth])
            if meth == "fk1c16":
                logs[meth] = json.load(open(f"{d}/zoom.json"))["records"]
        if bench in GENERAL and sel == "c16":
            sc["c16"], ans["c16"], logs["c16"] = sc["sel"], ans["sel"], logs["sel"]
        same_as_vz = [m for m in ("sel", "c16", "c16ctr", "t16") if m in ans and (m != "sel" or S["on"] == "unrouted")]
        for meth in same_as_vz:
            same = [ans[meth][i] == ans["vz"][i] for i in range(n) if routed[i]]
            checks[f"{bench} {meth}: answers on the routed questions = VisionZip-64's"] = float(np.mean(same)) if same else 1.0
        if bench in GENERAL:
            both = [i for i in range(n) if logs["sel"][i]["cover"] and logs["fk1c16"][i]["cover"]]
            agree = float(np.mean([logs["sel"][i]["grid_box"] == logs["fk1c16"][i]["grid_box"] for i in both])) if both else 0.0
            checks[f"{bench} sel: box = EXP-0027's K = 1 cover 16 box on >= 99% of the unrouted questions"] = float(
                len(both) >= 0.99 * (~routed).sum() and agree >= 0.99)
            res["in_box"][f"{bench} sel box = EXP-0027 box"] = agree
            for meth in ("c16", "t16"):
                same = [logs[meth][i]["grid_box"] == logs["sel"][i]["grid_box"] for i in range(n) if logs[meth][i]["cover"] and logs["sel"][i]["cover"]]
                checks[f"{bench} {meth}: box = the selected run's on the questions both cover"] = float(np.mean(same)) if same else 0.0
            ctr = [logs["c16ctr"][i]["side"] == logs["c16"][i]["side"]
                   and logs["c16ctr"][i]["grid_box"] == a28.centred_box(*b["size"][i], logs["c16"][i]["side"])
                   for i in range(n) if logs["c16ctr"][i]["cover"] and logs["c16"][i]["cover"]]
            checks[f"{bench} c16ctr: the centred box with cover 16's side"] = float(np.mean(ctr)) if ctr else 0.0
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
            pairs = [("sel", "vz"), ("c16", "c16ctr"), ("c16", "t16"), ("c16ctr", "vz"), ("t16", "vz"), ("sel", "fk1c16"),
                     ("sel", "router"), ("vz", "router"), ("fk1c16", "vz")] + ([("c16", "vz")] if sel != "c16" else [])
        else:
            pairs = [("sel", "vz"), ("sel", "router"), ("vz", "router"), ("vanilla", "vz")]
        if "vanilla" in sc:
            pairs += [("vanilla", "sel")]
        for x, y in pairs:
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items() if not (v == "c16" and sel == "c16")}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    P = res["pairs"]

    def count(x, y):
        pr = {bench: P[f"{bench} {x} - {y}"] for bench in GENERAL}
        return pr, sum(v[1] > 0 for v in pr.values())

    q1, n1 = count("sel", "vz")
    q2, n2 = count("c16", "c16ctr")
    q3, n3 = count("c16", "t16")
    q4 = P["textvqa_noocr sel - vz"]
    Q1 = {2: "holds", 1: "partly", 0: "fails"}[n1]
    word = {2: "成立", 1: "部分成立", 0: "不成立"}
    verdict = {"holds": "Q1 成立：定位后覆盖在 VisionZip-64 上、在新数据上成立",
               "partly": "Q1 部分成立：如实写出在哪个集上成立",
               "fails": "Q1 不成立：定位后覆盖在 VisionZip-64 上不作为方法的贡献"}[Q1]
    if Q1 != "fails":
        verdict += {2: "；Q2 成立：定位的作用在新数据上成立", 1: "；Q2 部分成立：定位的作用只在一个集上可检出",
                    0: "；Q2 不成立：不能写成定位的作用，只能写成把名额集中到一块"}[n2]
        verdict += {2: "；Q3 成立：按区域多留好于按逐 token 相似度多留", 1: "；Q3 部分成立：区域好于逐 token 相似度只在一个集上可检出",
                    0: "；Q3 不成立：不能写成区域好于逐 token 相似度"}[n3]
    res["criteria"] = {"selected": sel, "selection": S, "Q1": Q1, "Q1_pairs": q1, "Q2": word[n2], "Q2_pairs": q2,
                       "Q3": word[n3], "Q3_pairs": q3, "Q4_pair": q4, "Q4_above_visionzip": bool(q4[1] > 0),
                       "pair_note": None if sel == "c16" else f"所选为 {sel}，Q2、Q3 用覆盖 16 的这一对（R09、R10）",
                       "verdict": verdict}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0029.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0029（计划 v1）：VisionZip-64 上的定位后覆盖的确认（VQAv2 test3、GQA testdev；TextVQA 描述；64 档）", "",
         f"所选（EXP-0028 的选择）：{sel} {json.dumps(S)}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
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
    note = "" if sel == "c16" else f"（{res['criteria']['pair_note']}）"
    L += ["", "## 按计划写定的判据", "",
          f"- **Q1**（所选 − VisionZip-64）：VQAv2 test3 {fmt(q1['vqav2_test3'])}，GQA testdev {fmt(q1['gqa_testdev'])} → {word[n1]}",
          f"- **Q2**（定位：覆盖 16 − 居中框覆盖 16）：VQAv2 test3 {fmt(q2['vqav2_test3'])}，GQA testdev {fmt(q2['gqa_testdev'])} → {word[n2]}{note}",
          f"- **Q3**（区域：覆盖 16 − 不围框）：VQAv2 test3 {fmt(q3['vqav2_test3'])}，GQA testdev {fmt(q3['gqa_testdev'])} → {word[n3]}{note}",
          f"- **Q4**（描述，TextVQA 不给 OCR，所选 − VisionZip-64）：{fmt(q4)}" + ("，高于 VisionZip-64" if q4[1] > 0 else "，不能写高于 VisionZip-64"),
          f"- **判读**：{verdict}"]
    open(f"{out_dir}/0029.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
