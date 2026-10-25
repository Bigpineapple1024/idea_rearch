"""EXP-0026 (plan v1.1): IDEA-0005's locate-then-cover on the development sets VQAv2 dev2 and GQA val400 at the 64 budget.

  python analyze_0026.py      # writes code/output/EXP-0026/analysis/{0026.md,0026.json}

Runs of EXP-0024 are reused as references with their configs unchanged: the K = 2 router (R02, R03), the K = 1 router
(R06, R07) and cover 16 at K = 2 (R18, R17). New in EXP-0026: cover 16 with the centred box at K = 2 and at K = 1, cover
16 at K = 1, cover 8 at K = 2 (plan v1), and the per-token relevance controls of plan v1.1: query_vis on every question
(R09, R10) and the no-box cover, the 16 tokens most similar to the question kept anywhere (R11, R12). The selection rule,
the verdicts, the attribution rule and the checks were written before any EXP-0026 run; the v1.1 runs are optional
here (reported as not tested when absent). Per-question scores are recomputed from answers.jsonl as analyze_0024.py
does; paired differences resample images with replacement (1000 draws, seed 0) within each set.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_0024 as a24  # noqa: E402

BENCHES = a24.BENCHES
QFILES = {"vqav2_dev2": ("D:/6CJNM007/data/llava_eval/vqav2_val/llava_vqav2_val_dev2.jsonl", "D:/6CJNM007/data/coco/val2014"),
          "gqa_val400": ("D:/6CJNM007/data/llava_eval/gqa/llava_gqa_val400.jsonl", "D:/6CJNM007/data/gqa/images_val400")}
# method -> (experiment, run IDs on vqav2_dev2 and gqa_val400); k1_c16ctr on vqav2_dev2 is R13, the rerun of R05, which
# exited before its first answer (EXP-0026 record, main a776e95)
RUNS = {"router": ("EXP-0024", ("R02", "R03")), "router_k1": ("EXP-0024", ("R06", "R07")), "c16": ("EXP-0024", ("R18", "R17")),
        "c16ctr": ("EXP-0026", ("R01", "R02")), "k1_c16": ("EXP-0026", ("R03", "R04")), "k1_c16ctr": ("EXP-0026", ("R13", "R06")),
        "c8": ("EXP-0026", ("R07", "R08")), "qvis": ("EXP-0026", ("R09", "R10")), "t16": ("EXP-0026", ("R11", "R12"))}
OPTIONAL = ("qvis", "t16")                    # plan v1.1; reported as not tested when absent
# method -> (same-K router, keep, cover n, centred box); qvis has no cover record (query_vis on every question)
SPEC = {"router": (None, 30, 0, False), "router_k1": (None, 47, 0, False), "c16": ("router", 30, 16, False),
        "c16ctr": ("router", 30, 16, True), "k1_c16": ("router_k1", 47, 16, False), "k1_c16ctr": ("router_k1", 47, 16, True),
        "c8": ("router", 30, 8, False), "qvis": ("router", 30, 0, False), "t16": ("router", 30, 16, False)}
CANDIDATES = ("c16", "c8", "k1_c16")          # also the tie order
CENTRED = {"c16": ("c16", "c16ctr"), "c8": ("c16", "c16ctr"), "k1_c16": ("k1_c16", "k1_c16ctr")}   # c8 borrows K = 2's pair
TIE = 0.1
N_BOOT = 1000


def runs_rows():
    rows = {}
    for exp in ("EXP-0024", "EXP-0026"):
        with open(f"{a24.OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for meth, (exp, ids) in list(RUNS.items()):
        done = [rows.get(f"{exp}-{rid}", {}).get("status") == "completed" for rid in ids]
        if meth in OPTIONAL and not any(done):
            del RUNS[meth]                                     # not run (plan v1.1): reported as not tested
        elif not all(done):
            sys.exit(f"{exp}-{ids} ({meth}) not completed")
    return rows


def content_centres(bench, b):
    """(row, column) centre of each question's image content band on the 24 x 24 grid."""
    from PIL import Image
    from fastv_llava import _content_band
    qfile, folder = QFILES[bench]
    path = {str(q["question_id"]): q["image"] for q in map(json.loads, open(qfile, encoding="utf-8"))}
    cache, out = {}, []
    for q in b["order"]:
        p = path[q]
        if p not in cache:
            with Image.open(os.path.join(folder, p)) as im:
                r_lo, r_hi, c_lo, c_hi = _content_band(*im.size, a24.GRID)
            cache[p] = ((r_lo + r_hi) / 2, (c_lo + c_hi) / 2)
        out.append(cache[p])
    return out


def main():
    out_dir = f"{a24.OUT}/EXP-0026/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    rows = runs_rows()
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "in_box": {}}
    delta = {}
    for bi, bench in enumerate(BENCHES):
        b = a24.load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, ans, kept, zoom = {}, {}, {}, {}
        for meth, (exp, ids) in RUNS.items():
            rid = ids[bi]
            d = f"{a24.OUT}/{exp}/{exp}-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            s = a24.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench == "vqav2_dev2" else "gqa_scores.jsonl"
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
            k = json.load(open(f"{d}/kept_tokens.json"))
            kept[meth] = k["kept"]
            checks[f"{bench} {meth}: one kept record per question"] = float(k["n_prefills"] == n)
            checks[f"{bench} {meth}: logged route decisions equal the rule on the question text"] = float(np.mean(np.array(k["routed"]) == routed))
            if os.path.exists(f"{d}/zoom.json"):
                z = json.load(open(f"{d}/zoom.json"))
                checks[f"{bench} {meth}: one cover record per question, in order"] = float(
                    z["n_prefills"] == n and [x["index"] for x in z["records"]] == list(range(n)))
                zoom[meth] = z["records"]
        centres = content_centres(bench, b)
        for meth, (ref, keep, cov, ctr) in SPEC.items():
            if meth not in RUNS:
                continue
            checks[f"{bench} {meth}: {keep} image tokens kept in every prefill"] = float(np.mean([len(x) == keep for x in kept[meth]]))
            if ref is None:
                continue
            if meth == "qvis":                                 # no cover: only the routed questions are pinned
                same = [ans[meth][i] == ans[ref][i] for i in range(n) if routed[i]]
                checks[f"{bench} qvis: answers on the routed questions = router's"] = float(np.mean(same)) if same else 1.0
                continue
            recs = zoom[meth]
            if meth == "t16":                                  # the region is the logged set of 16 positions, not a box
                ok = [len(x["cover_top"]) == 16 and set(x["cover_top"]) <= set(kept[meth][i]) and x["in_box"] == 16
                      for i, x in enumerate(recs) if x["cover"]]
                checks[f"{bench} t16: the 16 logged most-similar tokens all kept, in-region count 16"] = float(np.mean(ok)) if ok else 0.0
                covered = np.array([bool(x["cover"]) for x in recs])
                checks[f"{bench} t16: routed questions untouched, every unrouted one covered"] = float(
                    np.mean([(x.get("skip") == "routed") == bool(rt) for x, rt in zip(recs, routed)]) * ((covered == ~routed).mean() >= 0.99))
                same = [ans[meth][i] == ans[ref][i] for i in range(n) if routed[i]]
                checks[f"{bench} t16: answers on the routed questions = router's"] = float(np.mean(same)) if same else 1.0
                res["in_box"][f"{bench} t16"] = {"mean_changed": float(np.mean([x["changed"] for x in recs if x["cover"]]))}
                continue
            covered = np.array([bool(x["cover"]) for x in recs])
            checks[f"{bench} {meth}: routed questions untouched, every unrouted one covered"] = float(
                np.mean([(x.get("skip") == "routed") == bool(rt) for x, rt in zip(recs, routed)]) * ((covered == ~routed).mean() >= 0.99))
            same = [ans[meth][i] == ans[ref][i] for i in range(n) if routed[i]]
            checks[f"{bench} {meth}: answers on the routed questions = {ref}'s"] = float(np.mean(same)) if same else 1.0
            inb = [sum(a24.box_mask(x["grid_box"])[p] for p in kept[meth][i]) == x["in_box"]
                   and x["in_box"] <= min(x["baseline_in_box"] + cov, keep) for i, x in enumerate(recs) if x["cover"]]
            checks[f"{bench} {meth}: kept in-box count = the logged count, at most the baseline's + {cov}"] = float(np.mean(inb)) if inb else 0.0
            full = [x["in_box"] == min(x["baseline_in_box"] + cov, keep) for x in recs if x["cover"]]
            checks[f"{bench} {meth}: in-box count reaches the baseline's + {cov} (capped by keep) on >= 99% of covered questions"] = float(
                np.mean(full) >= 0.99)
            res["in_box"][f"{bench} {meth}"] = {"mean_in_box": float(np.mean([x["in_box"] for x in recs if x["cover"]])),
                                                "all_kept_in_box": float(np.mean([x["in_box"] == keep for x in recs if x["cover"]])),
                                                "mean_changed": float(np.mean([x["changed"] for x in recs if x["cover"]]))}
            if ctr:
                ok = []
                for i, x in enumerate(recs):
                    if x["cover"]:
                        r0, c0, r1, c1 = x["grid_box"]
                        cr, cc = centres[i]
                        ok.append(abs((r0 + r1) / 2 - cr) <= 0.5 and abs((c0 + c1) / 2 - cc) <= 0.5)
                checks[f"{bench} {meth}: box centred on the image's content band"] = float(np.mean(ok)) if ok else 0.0
        for loc, cen in (("c16", "c16ctr"), ("k1_c16", "k1_c16ctr")):
            both = [(x["side"], y["side"]) for x, y in zip(zoom[loc], zoom[cen]) if x["cover"] and y["cover"]]
            checks[f"{bench} {cen}: box side equals {loc}'s"] = float(np.mean([p == q for p, q in both])) if both else 0.0
        for meth in ("k1_c16", "c8"):
            both = [(x["grid_box"], y["grid_box"]) for x, y in zip(zoom[meth], zoom["c16"]) if x["cover"] and y["cover"]]
            checks[f"{bench} {meth}: boxes equal EXP-0024's cover 16 (same locate step)"] = float(np.mean([p == q for p, q in both])) if both else 0.0
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

        pairs = [(c, "router") for c in CANDIDATES] + [("c16", "router"), ("c8", "router"), ("k1_c16", "router_k1"),
                                                         ("c16", "c16ctr"), ("k1_c16", "k1_c16ctr"), ("c16ctr", "router"),
                                                         ("k1_c16ctr", "router_k1"), ("k1_c16", "c16"), ("router_k1", "router"),
                                                         ("c16", "c8")]
        pairs += [p for p in (("c16", "t16"), ("t16", "router"), ("qvis", "router"), ("c16", "qvis"))
                  if p[0] in RUNS and p[1] in RUNS]
        for x, y in dict.fromkeys(pairs):
            key = f"{x} - {y}"
            res["pairs"][f"{bench} {key}"] = paired(x, y)
            delta.setdefault(key, {})[bench] = res["pairs"][f"{bench} {key}"]
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} {key} {g}"] = paired(x, y, m)
        if bench == "vqav2_dev2":                                  # localisation: the located box on the named COCO object
            masks = a24.topic_masks(b)
            hit, has = np.zeros(n, bool), np.zeros(n, bool)
            for i, (m, x) in enumerate(zip(masks, zoom["c16"])):
                if m is not None and x["cover"]:
                    has[i], hit[i] = True, (m & a24.box_mask(x["grid_box"])).sum() / m.sum() >= 0.5
            res["localisation"] = {"n": int(has.sum()), "hit_rate": float(hit[has].mean()) if has.any() else None}
            for x, y in (("c16", "router"), ("c16ctr", "router"), ("c16", "c16ctr"), ("k1_c16", "router_k1"), ("k1_c16", "k1_c16ctr")):
                res["localisation"][f"{x} - {y} [hit]"] = paired(x, y, has & hit)
                res["localisation"][f"{x} - {y} [miss]"] = paired(x, y, has & ~hit)
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items()}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)

    def mean2(key):
        return (delta[key]["vqav2_dev2"][0] + delta[key]["gqa_val400"][0]) / 2

    total = {c: mean2(f"{c} - router") for c in CANDIDATES}
    top = max(total.values())
    best = [c for c in CANDIDATES if total[c] >= top - TIE][0]
    ref = SPEC[best][0]
    cover_d = {bench: delta[f"{best} - {ref}"][bench][0] for bench in BENCHES}
    loc_pair = CENTRED[best]
    loc_mean = mean2(f"{loc_pair[0]} - {loc_pair[1]}")
    any_cover = any(mean2(f"{c} - {SPEC[c][0]}") > 0 for c in CANDIDATES)
    if all(v > 0 for v in cover_d.values()) and loc_mean > 0:
        verdict = "promising"
    elif all(v > 0 for v in cover_d.values()):
        verdict = "concentration"
    elif not any_cover:
        verdict = "unlikely"
    else:
        verdict = "intermediate"
    region = mean2("c16 - t16") if "c16 - t16" in delta else None       # plan v1.1's attribution rule
    res["selection"] = {"total_mean_delta": total, "selected": best, "cover_delta": cover_d, "localisation_pair": list(loc_pair),
                        "localisation_mean_delta": loc_mean, "cover_mean_delta": {c: mean2(f"{c} - {SPEC[c][0]}") for c in CANDIDATES},
                        "verdict": verdict, "region_vs_relevance_mean_delta": region,
                        "region_attributed": None if region is None else region > 0}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0026.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0026（计划 v1.1）：定位后覆盖，开发集（VQAv2 dev2、GQA val400，64 档）", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {m} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in res["pairs"].items() if x]
    loc = res["localisation"]
    L += ["", "## 定位（VQAv2 dev2，不路由且题目点名 COCO 物体的题；按定位框是否盖住话题物体分组）", "",
          f"- 定位框盖住话题物体至少一半的比例：{loc['hit_rate']:.3f}（{loc['n']} 题）"]
    L += [f"- {k}：{fmt(x)}" for k, x in loc.items() if k not in ("n", "hit_rate")]
    L += ["", "框内保留：" + "；".join(f"{k} {json.dumps(x)}" for k, x in res["in_box"].items()),
          "", "每题秒数：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    s = res["selection"]
    L += ["", "## 按计划写定的选择规则与判读", "", "| 候选 | − K = 2 路由：VQAv2 dev2 | GQA val400 | 平均 Δ̄ | − 同 K 路由（两集平均） |",
          "| --- | --- | --- | --- | --- |"]
    L += [f"| {c} | {delta[f'{c} - router']['vqav2_dev2'][0]:+.2f} | {delta[f'{c} - router']['gqa_val400'][0]:+.2f} | {total[c]:+.2f} "
          f"| {s['cover_mean_delta'][c]:+.2f} |" for c in CANDIDATES]
    names = {"promising": "有希望（进入确认）", "concentration": "集中而非定位（交用户决定）", "unlikely": "不像", "intermediate": "中间情况（交用户决定）"}
    L += ["", f"**选中**：{best}；覆盖的增益（− {ref}）：VQAv2 {cover_d['vqav2_dev2']:+.2f}，GQA {cover_d['gqa_val400']:+.2f}；"
              f"定位的增益（{loc_pair[0]} − {loc_pair[1]}，两集平均）{loc_mean:+.2f}；判读：{names[verdict]}。"]
    if region is None:
        L += ["", "**归因（计划 v1.1）**：区域还是逐 token 相关性：未检验（不围框的运行 R11、R12 没有运行）。"]
    else:
        L += ["", f"**归因（计划 v1.1）**：覆盖 16 − 不围框（两集平均）{region:+.2f} → "
                  f"{'按区域多留比按 token 相似度多留更好（区域的作用）' if region > 0 else '增益来自题目相关的 token 本身，不需要框'}。"]
    open(f"{out_dir}/0026.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
