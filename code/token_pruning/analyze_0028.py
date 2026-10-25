"""EXP-0028 (plan v1): IDEA-0006's development runs, locate-then-cover on VisionZip-64, on VQAv2 dev2 and GQA val400.

  python analyze_0028.py      # writes code/output/EXP-0028/analysis/{0028.md,0028.json}

Selection: of cover 16, cover 32 and cover 16 on every question, the candidate with the largest two-set mean of
candidate - VisionZip-64; within 0.1 of the largest, the first in that order. Verdict: promising when the selected one
is above VisionZip-64 on both sets and cover 16 is above its centred-box control (two-set mean); concentration, not
localisation, when only the second fails; not like when no candidate's two-set mean is above 0; otherwise
intermediate. Attribution (plan v1.1, not part of the selection or verdict): cover 16 - the no-box control (the 16
patches most similar to the question kept wherever they lie; R11, R12), two-set mean: above 0 means the region helps
more than per-token relevance. Described alongside: EXP-0026's selected configuration (FastV K = 1 + cover 16; R03, R04) and EXP-0024's
K = 2 router (IDEA-0003; R02, R03). EXP-0026's cover boxes are an implementation check of the box (it depends on the
input embeddings only, so VisionZip's cover must locate the same one). Per-question scores are recomputed from
answers.jsonl (LLaVA's m4c evaluator on the ten answers for VQAv2, exact match after rstrip('.').lower() for GQA) and
checked against each run's own score file. Paired differences resample images with replacement (1000 draws, seed 0)
within each set.
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
D = 53                                                    # VisionZip-64's dominant patches besides CLS
BENCHES = ("vqav2_dev2", "gqa_val400")
FILES = {"vqav2_dev2": (f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_dev2.jsonl", f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_dev2_annotations.jsonl",
                        "D:/6CJNM007/data/coco/val2014"),
         "gqa_val400": (f"{EVAL_DATA}/gqa/llava_gqa_val400.jsonl", f"{EVAL_DATA}/gqa/llava_gqa_val400_annotations.jsonl",
                        "D:/6CJNM007/data/gqa/images_val400")}
# (bench, method) -> (experiment, run ID)
RUNS = {("vqav2_dev2", "vz"): ("EXP-0028", "R01"), ("gqa_val400", "vz"): ("EXP-0028", "R02"),
        ("vqav2_dev2", "c16"): ("EXP-0028", "R03"), ("gqa_val400", "c16"): ("EXP-0028", "R04"),
        ("vqav2_dev2", "c16ctr"): ("EXP-0028", "R05"), ("gqa_val400", "c16ctr"): ("EXP-0028", "R06"),
        ("vqav2_dev2", "c32"): ("EXP-0028", "R07"), ("gqa_val400", "c32"): ("EXP-0028", "R08"),
        ("vqav2_dev2", "c16all"): ("EXP-0028", "R09"), ("gqa_val400", "c16all"): ("EXP-0028", "R10"),
        ("vqav2_dev2", "t16"): ("EXP-0028", "R11"), ("gqa_val400", "t16"): ("EXP-0028", "R12"),
        ("vqav2_dev2", "fk1c16"): ("EXP-0026", "R03"), ("gqa_val400", "fk1c16"): ("EXP-0026", "R04"),
        ("vqav2_dev2", "router"): ("EXP-0024", "R02"), ("gqa_val400", "router"): ("EXP-0024", "R03")}
COVER = {"c16": 16, "c16ctr": 16, "c32": 32, "c16all": 16, "t16": 16}       # VisionZip cover runs and their n
CANDIDATES = ("c16", "c32", "c16all")                             # in the tie order
NAMES = {"vz": "VisionZip-64", "c16": "覆盖 16", "c16ctr": "居中框覆盖 16", "c32": "覆盖 32", "c16all": "覆盖 16（每题）",
         "t16": "不围框（与题目最相似的 16 个）", "fk1c16": "FastV K = 1 + 覆盖 16（EXP-0026）", "router": "K = 2 路由（EXP-0024）"}
N_BOOT = 1000


def load_bench(bench):
    """Questions in file order with the fields the breakdowns need; ids as strings (as analyze_0027.load_bench)."""
    import run_eval
    from PIL import Image
    qfile, afile, folder = FILES[bench]
    qs = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    gts = {str(a["question_id"]): a for a in map(json.loads, open(afile, encoding="utf-8"))}
    order = [str(q["question_id"]) for q in qs]
    if sorted(order) != sorted(gts):
        sys.exit(f"{bench}: questions and annotations differ")
    b = {"order": order, "gts": gts, "routed": np.array([run_eval.routes_to_query(q["text"]) for q in qs])}
    images = sorted({gts[q]["image"] for q in order})
    idx = {im: k for k, im in enumerate(images)}
    b["images"], b["img_of"] = images, np.array([idx[gts[q]["image"]] for q in order])
    sizes = {}
    for q in qs:                                          # (width, height) of each question's image, header only
        if q["image"] not in sizes:
            with Image.open(os.path.join(folder, q["image"])) as im:
                sizes[q["image"]] = im.size
    b["size"] = [sizes[q["image"]] for q in qs]
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
    b["groups"]["[routed]"] = b["routed"]
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
    for exp in ("EXP-0024", "EXP-0026", "EXP-0028"):
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


def centred_box(width, height, side, min_side=8):
    """fastv_llava.zoom_box's centred box for a given side, written out: centred on the content band, kept inside it
    (inside the grid when the band is narrower than min_side)."""
    from fastv_llava import _content_band
    r_lo, r_hi, c_lo, c_hi = _content_band(width, height, GRID)
    band = min(r_hi - r_lo + 1, c_hi - c_lo + 1)
    if band >= min_side:
        lo_r, hi_r, lo_c, hi_c = r_lo, r_hi - side + 1, c_lo, c_hi - side + 1
    else:
        lo_r, hi_r, lo_c, hi_c = 0, GRID - side, 0, GRID - side
    r0 = int(min(max(round((r_lo + r_hi) / 2 - (side - 1) / 2), lo_r), hi_r))
    c0 = int(min(max(round((c_lo + c_hi) / 2 - (side - 1) / 2), lo_c), hi_c))
    return [r0, c0, r0 + side - 1, c0 + side - 1]


def check_vz_log(checks, res, bench, meth, b, recs):
    """The implementation checks of one VisionZip cover run's records (plan v1)."""
    n, routed, cov = len(b["order"]), b["routed"], COVER[meth]
    covered = np.array([bool(x["cover"]) for x in recs])
    checks[f"{bench} {meth}: one cover record per question, in order"] = float(
        len(recs) == n and [x["index"] for x in recs] == list(range(n)))
    if meth == "c16all":
        checks[f"{bench} {meth}: no question skipped as routed; >= 99% covered"] = float(
            not any(x.get("skip") == "routed" for x in recs)) * float(covered.mean() >= 0.99)
    else:
        checks[f"{bench} {meth}: routed questions untouched (rule on the question text)"] = float(
            np.mean([(x.get("skip") == "routed") == bool(rt) for x, rt in zip(recs, routed)]))
        checks[f"{bench} {meth}: >= 99% of the unrouted questions covered"] = float(covered[~routed].mean() >= 0.99)
    ok = []
    for x in recs:
        if not x["cover"]:
            ok.append(len(x["dominant"]) == D)
            continue
        m = box_mask(x["grid_box"])
        if meth == "t16":                                 # the region is the 16 patches most similar to the query
            m = np.zeros(GRID * GRID, bool)
            m[x["cover_top"]] = True
        dom, base = np.array(x["dominant"]), np.array(x["baseline"])
        ok.append(len(dom) == D and len(base) == D and len(set(dom)) == D
                  and (meth != "t16" or (len(x["cover_top"]) == cov and set(x["cover_top"]) <= set(x["dominant"])))
                  and int(m[dom].sum()) == x["in_box"] == min(x["baseline_in_box"] + cov, D, x["region"])
                  and int(m[base].sum()) >= x["baseline_in_box"] and x["changed"] == len(set(dom) - set(base))
                  and x["grid_box"][2] - x["grid_box"][0] == x["grid_box"][3] - x["grid_box"][1] == x["side"] - 1)
    checks[f"{bench} {meth}: 53 dominant patches; in-box count = VisionZip's + {cov} (capped), as logged"] = float(np.mean(ok))
    c = [x for x in recs if x["cover"]]
    res["in_box"][f"{bench} {meth}"] = {
        "covered_share": float(covered.mean()), "mean_baseline_in_box": float(np.mean([x["baseline_in_box"] for x in c])),
        "mean_in_box": float(np.mean([x["in_box"] for x in c])), "mean_changed": float(np.mean([x["changed"] for x in c])),
        "mean_box_share": float(np.mean([x["side"] ** 2 / GRID ** 2 for x in c])),
        "all_in_box_share": float(np.mean([x["in_box"] == D for x in c]))}


def main():
    out_dir = f"{OUT}/EXP-0028/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    rows = runs_rows()
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "in_box": {}}
    for bench in BENCHES:
        b = load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, ans, logs = {}, {}, {}
        for (bb, meth), (exp, rid) in RUNS.items():
            if bb != bench:
                continue
            d = f"{OUT}/{exp}/{exp}-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            checks[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
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
            if meth in COVER:
                z = json.load(open(f"{d}/vz.json"))
                checks[f"{bench} {meth}: the log counts its prefills and covered ones"] = float(
                    z["n_prefills"] == len(z["records"]) and z["n_covered"] == sum(x["cover"] for x in z["records"]))
                logs[meth] = z["records"]
                check_vz_log(checks, res, bench, meth, b, logs[meth])
            if meth == "fk1c16":
                logs[meth] = json.load(open(f"{d}/zoom.json"))["records"]
        for meth in ("c16", "c16ctr", "c32", "t16"):
            same = [ans[meth][i] == ans["vz"][i] for i in range(n) if routed[i]]
            checks[f"{bench} {meth}: answers on the routed questions = VisionZip-64's"] = float(np.mean(same)) if same else 1.0
        both = [i for i in range(n) if logs["c16"][i]["cover"] and logs["fk1c16"][i]["cover"]]
        checks[f"{bench} c16: box = EXP-0026's K = 1 cover 16 box (same query, same input embeddings) on >= 99%"] = float(
            len(both) >= 0.99 * (~routed).sum()
            and np.mean([logs["c16"][i]["grid_box"] == logs["fk1c16"][i]["grid_box"] for i in both]) >= 0.99)
        res["in_box"][f"{bench} c16 box = EXP-0026 box"] = float(np.mean([logs["c16"][i]["grid_box"] == logs["fk1c16"][i]["grid_box"] for i in both]))
        for meth in ("c32", "c16all", "t16"):
            same = [logs[meth][i]["grid_box"] == logs["c16"][i]["grid_box"] for i in range(n) if logs[meth][i]["cover"] and logs["c16"][i]["cover"]]
            checks[f"{bench} {meth}: box = cover 16's on the questions both cover"] = float(np.mean(same)) if same else 0.0
        ctr = [logs["c16ctr"][i]["side"] == logs["c16"][i]["side"]
               and logs["c16ctr"][i]["grid_box"] == centred_box(*b["size"][i], logs["c16"][i]["side"])
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

        pairs = [(c, "vz") for c in CANDIDATES] + [("c16", "c16ctr"), ("c16ctr", "vz"), ("c32", "c16"), ("c16all", "c16"),
                                                  ("c16", "t16"), ("t16", "vz"), ("vz", "router"), ("vz", "fk1c16")] + [
            (c, "fk1c16") for c in CANDIDATES]
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
    delta = {c: {bench: P[f"{bench} {c} - vz"] for bench in BENCHES} for c in CANDIDATES}
    mean = {c: float(np.mean([delta[c][bench][0] for bench in BENCHES])) for c in CANDIDATES}
    best = max(mean.values())
    sel = next(c for c in CANDIDATES if mean[c] >= best - 0.1)
    dc = delta[sel]
    dl = {bench: P[f"{bench} c16 - c16ctr"] for bench in BENCHES}
    dl_mean = float(np.mean([dl[bench][0] for bench in BENCHES]))
    up = all(dc[bench][0] > 0 for bench in BENCHES)
    if all(v <= 0 for v in mean.values()):
        verdict = "不像：三个候选比 VisionZip-64 的两集平均都 ≤ 0，定位后覆盖在 VisionZip 上不成立"
    elif up and dl_mean > 0:
        verdict = "有希望（进入确认）"
    elif up:
        verdict = "集中而非定位：增益不需要题目定位，交用户决定是否继续"
    else:
        verdict = "中间情况：交用户决定"
    strength = sum(dc[bench][1] > 0 for bench in BENCHES)
    dr = {bench: P[f"{bench} c16 - t16"] for bench in BENCHES}
    dr_mean = float(np.mean([dr[bench][0] for bench in BENCHES]))
    attribution = ("Δr > 0：按区域多留比按逐 token 相似度多留更好（区域的作用）" if dr_mean > 0
                   else "Δr ≤ 0：增益来自与题目相似的 token 本身，不需要框")
    res["criteria"] = {"delta": delta, "mean": mean, "selected": sel, "delta_c": dc, "delta_l": dl, "delta_l_mean": dl_mean,
                       "verdict": verdict, "intervals_above_0": strength,
                       "delta_l_note": None if sel == "c16" else f"所选为 {sel}，Δl 用覆盖 16 − 居中框覆盖 16",
                       "delta_r": dr, "delta_r_mean": dr_mean, "attribution": attribution}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0028.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0028（计划 v1）：VisionZip-64 上的定位后覆盖，开发集（VQAv2 dev2、GQA val400）", "", "## 核对", ""]
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
    L += ["", "## 按计划写定的选择与判读", "", "| 候选 − VisionZip-64 | VQAv2 dev2 | GQA val400 | Δ̄ |", "| --- | --- | --- | --- |"]
    L += [f"| {NAMES[c]} | {fmt(delta[c]['vqav2_dev2'])} | {fmt(delta[c]['gqa_val400'])} | {mean[c]:+.2f} |" for c in CANDIDATES]
    L += ["", f"- **选中**：{NAMES[sel]}",
          f"- **覆盖的增益** Δc（所选 − VisionZip-64）：VQAv2 {fmt(dc['vqav2_dev2'])}，GQA {fmt(dc['gqa_val400'])}；区间在 0 以上的集数 {strength}",
          f"- **定位的增益** Δl（覆盖 16 − 居中框覆盖 16）：VQAv2 {fmt(dl['vqav2_dev2'])}，GQA {fmt(dl['gqa_val400'])}，两集平均 {dl_mean:+.2f}"
          + ("" if sel == "c16" else f"（{res['criteria']['delta_l_note']}）"),
          f"- **判读**：{verdict}",
          "", "## 归因（计划 v1.1，不改选择与判读）", "",
          f"- **区域还是逐 token 相关性** Δr（覆盖 16 − 不围框）：VQAv2 {fmt(dr['vqav2_dev2'])}，GQA {fmt(dr['gqa_val400'])}，"
          f"两集平均 {dr_mean:+.2f} → {attribution}"]
    open(f"{out_dir}/0028.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
