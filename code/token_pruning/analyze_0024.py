"""EXP-0024 (plan v1): IDEA-0004's budgeted zoom on the development sets VQAv2 dev2 and GQA val400 at the 64 budget.

  python analyze_0024.py      # writes code/output/EXP-0024/analysis/{0024.md,0024.json}

Candidates (pooled crop and lattice crop at K = 2, pooled crop at K = 1) are compared with the router at the same K, each
in two variants: as run (every question zoomed, s_min 0) and unrouted-only, spliced from the run's answers on the
unrouted questions and the same-K router's answers on the routed ones (each prefill is independent; on the unrouted
questions an unrouted-only run computes the same thing). Controls: block means of the image's own tokens (p16full), the
centred box (z16pctr) and cover 16 (c16). The selection rule, the attribution rule and the breakdowns were written
before any run: answer kind, colour questions, GQA structural types, GQA position and attribute questions, the zoomed
questions, the localisation hit rate on COCO-labelled topic objects (VQAv2 dev2), box statistics, seconds per answer.
Per-question scores are recomputed from answers.jsonl (LLaVA's m4c evaluator for VQAv2, exact match after
rstrip('.').lower() for GQA) and checked against each run's own score file. Paired differences resample images with
replacement (1000 draws, seed 0) within each set.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_object_boxes import GRID, object_record   # noqa: E402
from squeeze_0018 import COCO_INSTANCES, EVAL_DATA, OUT, plural_forms   # noqa: E402

LLAVA = "D:/6CJNM007/idea_rearch/source/LLaVA"
BENCHES = ("vqav2_dev2", "gqa_val400")
# method -> run IDs of the plan's run table (R01 is the K = 1 prior's blank-image run, not analysed here)
RUNS = {"router": ("R02", "R03"), "z16p": ("R04", "R05"), "router_k1": ("R06", "R07"), "k1_z16p": ("R08", "R09"),
        "z16g": ("R10", "R11"), "p16full": ("R12", "R13"), "z16pctr": ("R14", "R15"), "c16": ("R16", "R17")}
# method -> same-K router, zoom tokens, keep, zoomed on every question
SPEC = {"router": (None, 0, 30, False), "router_k1": (None, 0, 47, False), "z16p": ("router", 16, 30, True),
        "k1_z16p": ("router_k1", 16, 47, True), "z16g": ("router", 16, 30, True), "p16full": ("router", 16, 30, True),
        "z16pctr": ("router", 16, 30, True), "c16": ("router", 0, 30, False)}
CANDIDATES = ("z16p", "z16g", "k1_z16p")
TIE = 0.1
OTHER_GAIN = 1.0
N_BOOT = 1000


def load_bench(bench):
    """Questions in file order with the fields the breakdowns need; ids as strings."""
    import run_eval
    if bench == "vqav2_dev2":
        qfile, afile = f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_dev2.jsonl", f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_dev2_annotations.jsonl"
    else:
        qfile, afile = f"{EVAL_DATA}/gqa/llava_gqa_val400.jsonl", f"{EVAL_DATA}/gqa/llava_gqa_val400_annotations.jsonl"
    qs = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    gts = {str(a["question_id"]): a for a in map(json.loads, open(afile, encoding="utf-8"))}
    order = [str(q["question_id"]) for q in qs]
    if sorted(order) != sorted(gts):
        sys.exit(f"{bench}: questions and annotations differ")
    b = {"order": order, "gts": gts, "text": {str(q["question_id"]): q["text"] for q in qs},
         "routed": np.array([run_eval.routes_to_query(q["text"]) for q in qs])}
    images = sorted({gts[q]["image"] for q in order})
    idx = {im: k for k, im in enumerate(images)}
    b["images"], b["img_of"] = images, np.array([idx[gts[q]["image"]] for q in order])
    groups = {}
    if bench == "vqav2_dev2":
        kind = np.array([gts[q]["answer_type"] for q in order])
        groups["[colour]"] = np.array([gts[q]["question_type"].startswith("what color") for q in order])
    else:
        kind = np.array(["yes/no" if gts[q]["answer"] in ("yes", "no") else "other" for q in order])
        for s in sorted({gts[q]["structural"] for q in order}):
            groups[f"[structural {s}]"] = np.array([gts[q]["structural"] == s for q in order])
        groups["[position]"] = np.array([gts[q]["detailed"].startswith("position") for q in order])
        groups["[attribute]"] = np.array([gts[q]["semantic"] == "attr" for q in order])
    b["groups"] = {**{f"[{k}]": kind == k for k in sorted(set(kind))}, **groups}
    b["groups"]["[unrouted]"] = ~b["routed"]
    return b


def recompute(bench, b, answers):
    if bench == "vqav2_dev2":
        sys.path.insert(0, LLAVA)
        from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
        ev = TextVQAAccuracyEvaluator()
        return np.array([ev._compute_answer_scores(b["gts"][q]["answers"]).get(ev.answer_processor(answers[q]), 0.0) for q in b["order"]])
    return np.array([float(answers[q].rstrip(".").lower() == b["gts"][q]["answer"]) for q in b["order"]])


def runs_rows():
    with open(f"{OUT}/EXP-0024/runs.tsv") as f:
        rows = {x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")}
    for meth, ids in RUNS.items():
        for rid in ids:
            if rows.get(f"EXP-0024-{rid}", {}).get("status") != "completed":
                sys.exit(f"EXP-0024-{rid} ({meth}) is not completed")
    return rows


def topic_masks(b):
    """VQAv2 dev2: for each question that names a COCO object present in its image (squeeze_0018's word matching), the
    object's patches on the 24 x 24 grid (analyze_object_boxes.object_record); others None."""
    coco = json.load(open(COCO_INSTANCES, encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    ims = {im["file_name"]: im for im in coco["images"]}
    inst = defaultdict(list)
    for a in coco["annotations"]:
        inst[(a["image_id"], names[a["category_id"]])].append(a)
    pats = {n: re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(plural_forms(n), key=len, reverse=True)) + r")\b")
            for n in names.values()}
    out = []
    for q in b["order"]:
        im = ims[b["gts"][q]["image"]]
        line = b["text"][q].split("\n")[0].lower()
        topic = [n for n, p in pats.items() if p.search(line) and inst.get((im["id"], n))]
        m = None
        if topic:
            m = np.zeros(GRID * GRID, bool)
            m[object_record(im, [a for n in topic for a in inst[(im["id"], n)]])["object_patches"]] = True
            m = m if m.any() else None
        out.append(m)
    return out


def box_mask(bx):
    m = np.zeros((GRID, GRID), bool)
    r0, c0, r1, c1 = bx
    m[r0:r1 + 1, c0:c1 + 1] = True
    return m.ravel()


def main():
    out_dir = f"{OUT}/EXP-0024/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    rows = runs_rows()
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "boxes": {}}
    delta = defaultdict(dict)                                   # variant -> bench -> [diff, lo, hi, n]
    for bi, bench in enumerate(BENCHES):
        b = load_bench(bench)
        n = len(b["order"])
        routed = b["routed"]
        sc, ans, kept, zoom = {}, {}, {}, {}
        for meth, ids in RUNS.items():
            rid = ids[bi]
            d = f"{OUT}/EXP-0024/EXP-0024-{rid}"
            a = {str(json.loads(l)["question_id"]): json.loads(l)["text"] for l in open(f"{d}/answers.jsonl")}
            s = recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench == "vqav2_dev2" else "gqa_scores.jsonl"
            logged = {str(json.loads(l)["question_id"]): json.loads(l)["score"] for l in open(f"{d}/{logf}")}
            checks[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            r = rows[f"EXP-0024-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth], ans[meth] = s, [a[q] for q in b["order"]]
            kept[meth] = json.load(open(f"{d}/kept_tokens.json"))
            checks[f"{bench} {meth}: one kept record per question"] = float(kept[meth]["n_prefills"] == n)
            checks[f"{bench} {meth}: logged route decisions equal the rule on the question text"] = float(np.mean(np.array(kept[meth]["routed"]) == routed))
            if os.path.exists(f"{d}/zoom.json"):
                z = json.load(open(f"{d}/zoom.json"))
                checks[f"{bench} {meth}: one zoom record per question, in order"] = float(
                    z["n_prefills"] == n and [x["index"] for x in z["records"]] == list(range(n)))
                zoom[meth] = z["records"]
        zoomed = {}
        for meth in RUNS:
            ref, nz, keep, every = SPEC[meth]
            k = kept[meth]["kept"]
            if ref is None:
                checks[f"{bench} {meth}: {keep} image tokens kept in every prefill"] = float(np.mean([len(x) == keep for x in k]))
                continue
            recs = zoom[meth]
            zoomed[meth] = np.array([bool(x["zoom"] or x["cover"]) for x in recs])
            if every:
                checks[f"{bench} {meth}: zoomed on at least 99% of the questions (s_min 0)"] = float(zoomed[meth].mean() >= 0.99)
            else:
                checks[f"{bench} {meth}: routed questions untouched, unrouted ones covered"] = float(np.mean(
                    [(x.get("skip") == "routed") == bool(rt) for x, rt in zip(recs, routed)]) * (zoomed[meth][~routed].mean() >= 0.99))
            want = [keep - nz if zoomed[meth][i] and nz else keep for i in range(n)]
            checks[f"{bench} {meth}: kept image tokens = keep - n when zoomed, keep otherwise"] = float(np.mean([len(x) == w for x, w in zip(k, want)]))
            if nz:
                checks[f"{bench} {meth}: kept log's zoom_tokens = n exactly on the zoomed prefills"] = float(np.mean(
                    [zt == (nz if zm else 0) for zt, zm in zip(kept[meth]["zoom_tokens"], zoomed[meth])]))
            same = [ans[meth][i] == ans[ref][i] for i in range(n) if not zoomed[meth][i]]
            checks[f"{bench} {meth}: answers of the prefills not zoomed = {ref}'s"] = float(np.mean(same)) if same else 1.0
            if meth == "c16":
                inb = [sum(box_mask(x["grid_box"])[p] for p in k[i]) == x["in_box"] for i, x in enumerate(recs) if x["cover"]]
                checks[f"{bench} c16: kept in-box count = the logged baseline count + 16 (capped)"] = float(np.mean(inb)) if inb else 0.0
                res["boxes"][f"{bench} c16 tokens changed (mean)"] = float(np.mean([x["changed"] for x in recs if x["cover"]]))
            if meth == "p16full":
                checks[f"{bench} p16full: every zoom record is the image's own pooled tokens"] = float(np.mean(
                    [x.get("source") == "full" for x in recs if x["zoom"]]))
        ref_box = [tuple(x["grid_box"]) if "grid_box" in x else None for x in zoom["z16p"]]
        for meth in ("z16g", "k1_z16p", "c16"):
            both = [(tuple(x["grid_box"]), y) for x, y in zip(zoom[meth], ref_box) if "grid_box" in x and y is not None]
            checks[f"{bench} {meth}: boxes equal z16p's (same locate step)"] = float(np.mean([x == y for x, y in both])) if both else 0.0
        both = [(x["side"], y["side"]) for x, y in zip(zoom["z16pctr"], zoom["z16p"]) if "side" in x and "side" in y]
        checks[f"{bench} z16pctr: box side equals z16p's"] = float(np.mean([x == y for x, y in both])) if both else 0.0
        located = [x for x in zoom["z16p"] if "side" in x]
        res["boxes"][bench] = {"mean_side": float(np.mean([x["side"] for x in located])),
                               "span_over_16": float(np.mean([x["span"] > 16 for x in located])),
                               "mean_top8_inside": float(np.mean([x["top_inside"] for x in located])),
                               "top1_inside": float(np.mean([x["top1_inside"] for x in located]))}
        for meth in RUNS:                                          # unrouted-only splices of the zoom-everything runs
            ref, nz, keep, every = SPEC[meth]
            if every:
                sc[f"{meth}|unrouted"] = np.where(routed, sc[ref], sc[meth])
                SPEC.setdefault(f"{meth}|unrouted", (ref, nz, keep, False))
                zoomed[f"{meth}|unrouted"] = zoomed[meth] & ~routed
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

        variants = [v for v in sc if SPEC[v][0] is not None]
        for v in variants:
            ref = SPEC[v][0]
            delta[v][bench] = paired(v, ref)
            res["pairs"][f"{bench} {v} - {ref}"] = delta[v][bench]
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} {v} - {ref} {g}"] = paired(v, ref, m)
            res["pairs"][f"{bench} {v} - {ref} [zoomed]"] = paired(v, ref, zoomed[v])
        for x, y in (("router_k1", "router"), ("k1_z16p", "router"), ("k1_z16p|unrouted", "router")):
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            delta[f"{x} vs {y}"][bench] = res["pairs"][f"{bench} {x} - {y}"]
        for v in ("", "|unrouted"):                               # attribution (K = 2, pooled crop)
            for ctrl in ("p16full", "z16pctr"):
                res["pairs"][f"{bench} z16p{v} - {ctrl}{v}"] = paired(f"z16p{v}", f"{ctrl}{v}")
                delta[f"z16p{v} vs {ctrl}{v}"][bench] = res["pairs"][f"{bench} z16p{v} - {ctrl}{v}"]
        res["pairs"][f"{bench} z16p|unrouted - c16"] = paired("z16p|unrouted", "c16")
        delta["z16p|unrouted vs c16"][bench] = res["pairs"][f"{bench} z16p|unrouted - c16"]
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items()}
        if bench == "vqav2_dev2":                                # localisation: located vs centred box on the topic object
            masks = topic_masks(b)
            loc = {}
            for meth in ("z16p", "z16pctr"):
                hit, has = np.zeros(n, bool), np.zeros(n, bool)
                for i, (m, x) in enumerate(zip(masks, zoom[meth])):
                    if m is not None and "grid_box" in x and not routed[i]:
                        has[i], hit[i] = True, (m & box_mask(x["grid_box"])).sum() / m.sum() >= 0.5
                loc[meth] = {"n": int(has.sum()), "hit_rate": float(hit[has].mean()) if has.any() else None}
                if meth == "z16p":
                    loc["z16p - router [hit]"] = paired("z16p", "router", has & hit)
                    loc["z16p - router [miss]"] = paired("z16p", "router", has & ~hit)
            res["localisation"] = loc
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    cand = [c + v for c in CANDIDATES for v in ("|unrouted", "")]
    mean = {c: (delta[c]["vqav2_dev2"][0] + delta[c]["gqa_val400"][0]) / 2 for c in cand}
    top = max(mean.values())
    best = [c for c in cand if mean[c] >= top - TIE][0]           # ties: unrouted before all, then the listed order
    other = res["pairs"][f"vqav2_dev2 {best} - {SPEC[best][0]} [other]"][0]
    promising = delta[best]["vqav2_dev2"][0] > 0 and delta[best]["gqa_val400"][0] > 0 and other > OTHER_GAIN
    unlikely = all(v <= 0 for v in mean.values())
    v = "|unrouted" if best.endswith("|unrouted") else ""
    attr = {k: (delta[k]["vqav2_dev2"][0] + delta[k]["gqa_val400"][0]) / 2 for k in (f"z16p{v} vs p16full{v}", f"z16p{v} vs z16pctr{v}")}
    res["selection"] = {"mean_delta": mean, "selected": best, "other_delta": other, "promising": promising, "unlikely": unlikely,
                        "verdict": "promising" if promising else ("unlikely" if unlikely else "intermediate"),
                        "attribution": attr, "zoom_attributed": all(x > 0 for x in attr.values()),
                        "k_lever": {bch: delta["router_k1 vs router"][bch] for bch in BENCHES}}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0024.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0024（计划 v1）：预算内放大，开发集（VQAv2 dev2、GQA val400，64 档）", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {m} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in res["pairs"].items() if x]
    loc = res["localisation"]
    L += ["", "## 定位（VQAv2 dev2，不路由且题目点名 COCO 物体的题）", "",
          f"- 框覆盖话题物体至少一半的比例：定位框 {loc['z16p']['hit_rate']:.3f}（{loc['z16p']['n']} 题），居中框 {loc['z16pctr']['hit_rate']:.3f}",
          f"- z16p − router，命中：{fmt(loc['z16p - router [hit]'])}；未命中：{fmt(loc['z16p - router [miss]'])}"]
    L += ["", "框的统计：" + "；".join(f"{k} {json.dumps(x)}" for k, x in res["boxes"].items())]
    L += ["", "每题秒数：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    s = res["selection"]
    L += ["", "## 按计划写定的选择规则", "", "| 候选 | Δ VQAv2 dev2 | Δ GQA val400 | 平均 Δ̄ |", "| --- | --- | --- | --- |"]
    L += [f"| {c} − {SPEC[c][0]} | {delta[c]['vqav2_dev2'][0]:+.2f} | {delta[c]['gqa_val400'][0]:+.2f} | {mean[c]:+.2f} |" for c in cand]
    L += ["", f"**选中**：{best}（VQAv2 other 上 {other:+.2f}）；判读："
              f"{ {'promising': '有希望（进入确认）', 'unlikely': '不像', 'intermediate': '中间情况（交用户决定）'}[s['verdict']] }。",
          f"**归因**（K = 2 池化放大，与选中的变体一致）：减去全图池化 {attr[f'z16p{v} vs p16full{v}']:+.2f}，减去居中框 "
          f"{attr[f'z16p{v} vs z16pctr{v}']:+.2f} → {'可以称为放大（定位后重新编码）' if s['zoom_attributed'] else '不能称为放大'}。",
          f"**预算杠杆**（K = 1 路由 − K = 2 路由）：VQAv2 {fmt(s['k_lever']['vqav2_dev2'])}；GQA {fmt(s['k_lever']['gqa_val400'])}。"]
    open(f"{out_dir}/0024.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
