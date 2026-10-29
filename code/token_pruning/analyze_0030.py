"""EXP-0030 (plan v1): IDEA-0007's development runs, one full LLM layer and then VisionZip's ranking, on VQAv2 dev2 and
GQA val400.

  python analyze_0030.py      # writes code/output/EXP-0030/analysis/{0030.md,0030.json}

K = 1, keep 47 (63.5 tokens per layer, as the K = 1 router): layer 0 sees all 576 patches; from layer 1 on 47 stay,
ranked by the vision tower's CLS attention at layer -2 (VisionZip's dominant score; padding never kept). Candidates, in
the tie order: k1cls (every question by that ranking; R01, R02), k1qcls (the router's routed questions by query_vis, as
the K = 1 router, the others by that ranking; R03, R04), k1qcls_c16 (k1qcls with cover 16 on the unrouted questions;
R05, R06). Selection: the candidate with the largest two-set mean of candidate - VisionZip-64 (EXP-0028-R01, R02);
within 0.1 of the largest, the first in that order. Verdict: promising when the selected one is above VisionZip-64 on
both sets and above EXP-0028's selected configuration (VisionZip-64 + cover 16 on every question, EXP-0028-R19, R20) in
the two-set mean; not better than IDEA-0006 when only the second fails; not like when no candidate's two-set mean is
above 0; otherwise intermediate. Large (reported, not part of the verdict): the selected one's two-set mean is at least
+2.0. Attribution (described, with pre-specified readings): the full first layer (k1cls - VisionZip-64), the ranking at
K = 1 on the unrouted questions (k1qcls - the K = 1 router, EXP-0024-R06, R07), routing (k1qcls - k1cls on the routed
questions), cover (k1qcls_c16 - k1qcls). Described alongside: IDEA-0006's runs (EXP-0028-R13, R14, R19, R20),
IDEA-0005's selected configuration (EXP-0026-R03, R04), the K = 2 router (EXP-0024-R02, R03) and the two-base splice
looked at after EXP-0028's analysis (exploratory: the K = 1 router on the routed questions, VisionZip-64 + cover 16 on
the others). Per-question scores are recomputed from answers.jsonl and checked against each run's own score file, and
paired differences resample images with replacement (1000 draws, seed 0) within each set, as in analyze_0028.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_0028 as a28  # noqa: E402
from fastv_llava import _padding_mask  # noqa: E402

OUT = a28.OUT
BENCHES = a28.BENCHES
KEEP = 47
# (bench, method) -> (experiment, run ID)
RUNS = {("vqav2_dev2", "k1cls"): ("EXP-0030", "R01"), ("gqa_val400", "k1cls"): ("EXP-0030", "R02"),
        ("vqav2_dev2", "k1qcls"): ("EXP-0030", "R03"), ("gqa_val400", "k1qcls"): ("EXP-0030", "R04"),
        ("vqav2_dev2", "k1qcls_c16"): ("EXP-0030", "R05"), ("gqa_val400", "k1qcls_c16"): ("EXP-0030", "R06"),
        ("vqav2_dev2", "vz"): ("EXP-0028", "R01"), ("gqa_val400", "vz"): ("EXP-0028", "R02"),
        ("vqav2_dev2", "c16"): ("EXP-0028", "R13"), ("gqa_val400", "c16"): ("EXP-0028", "R14"),
        ("vqav2_dev2", "c16all"): ("EXP-0028", "R19"), ("gqa_val400", "c16all"): ("EXP-0028", "R20"),
        ("vqav2_dev2", "router_k1"): ("EXP-0024", "R06"), ("gqa_val400", "router_k1"): ("EXP-0024", "R07"),
        ("vqav2_dev2", "fk1c16"): ("EXP-0026", "R03"), ("gqa_val400", "fk1c16"): ("EXP-0026", "R04"),
        ("vqav2_dev2", "router"): ("EXP-0024", "R02"), ("gqa_val400", "router"): ("EXP-0024", "R03")}
CANDIDATES = ("k1cls", "k1qcls", "k1qcls_c16")                    # in the tie order
NAMES = {"k1cls": "一层全看 + CLS 排序（每题）", "k1qcls": "一层全看 + CLS 排序（不路由的题）+ 路由",
         "k1qcls_c16": "上一行 + 覆盖 16", "vz": "VisionZip-64", "c16": "VisionZip-64 + 覆盖 16（EXP-0028）",
         "c16all": "VisionZip-64 + 每题覆盖 16（EXP-0028 所选）", "router_k1": "K = 1 路由（EXP-0024）",
         "fk1c16": "K = 1 路由 + 覆盖 16（EXP-0026 所选）", "router": "K = 2 路由（EXP-0024）",
         "splice": "拼接：被路由的题 K = 1 路由，其余 VisionZip-64 + 覆盖 16（探索性）"}
N_BOOT = 1000


def runs_rows():
    rows = {}
    for exp in ("EXP-0024", "EXP-0026", "EXP-0028", "EXP-0030"):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for (bench, meth), (exp, rid) in RUNS.items():
        if rows.get(f"{exp}-{rid}", {}).get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
    return rows


def run_dir(bench, meth):
    exp, rid = RUNS[(bench, meth)]
    return f"{OUT}/{exp}/{exp}-{rid}"


def check_cls(checks, res, bench, meth, b, d, ref):
    """The implementation checks of one CLS run (plan v1). ref: the reused runs' kept tokens, answers, boxes and
    VisionZip's own dominant patches."""
    n, routed = len(b["order"]), b["routed"]
    cls = json.load(open(f"{d}/cls.json"))
    kept = json.load(open(f"{d}/kept_tokens.json"))
    recs, kk = cls["records"], kept["kept"]
    checks[f"{bench} {meth}: one CLS record and one kept list per prefill, in order"] = float(
        cls["n_prefills"] == n == len(recs) == len(kk) and [r["index"] for r in recs] == list(range(n)))
    used = np.array([bool(r["used"]) for r in recs])
    want = np.ones(n, bool) if meth == "k1cls" else ~routed
    checks[f"{bench} {meth}: ranked by the CLS score exactly where planned (rule on the question text)"] = float(np.mean(used == want))
    if meth != "k1cls":
        checks[f"{bench} {meth}: the kept log's routing decisions = the rule on the question text"] = float(
            np.mean(np.array(kept["routed"]) == routed))
    pad = [~_padding_mask(w, h, 24).numpy() for w, h in b["size"]]
    checks[f"{bench} {meth}: 47 kept patches per prefill, none wholly in the padding"] = float(np.mean(
        [len(k) == KEEP and len(set(k)) == KEEP and bool(pad[i][k].all()) for i, k in enumerate(kk)]))
    checks[f"{bench} {meth}: the 53 patches with the largest score = VisionZip's own dominant patches (EXP-0028) on >= 99%"] = float(
        np.mean([r["top53"] == ref["vz_own"][i] for i, r in enumerate(recs)]) >= 0.99)
    res["cls"][f"{bench} {meth} top53 = VisionZip's"] = float(np.mean([r["top53"] == ref["vz_own"][i] for i, r in enumerate(recs)]))
    if meth in ("k1cls", "k1qcls"):
        checks[f"{bench} {meth}: where ranked by the score, kept = its own choice"] = float(np.mean(
            [sorted(kk[i]) == recs[i]["own"] for i in range(n) if used[i]]))
    if meth == "k1qcls":
        same = [sorted(kk[i]) == sorted(ref["router_k1_kept"][i]) for i in range(n) if routed[i]]
        checks[f"{bench} {meth}: routed questions keep what the K = 1 router keeps (>= 99%)"] = float(np.mean(same) >= 0.99)
        res["cls"][f"{bench} {meth} routed kept = K = 1 router's"] = float(np.mean(same))
    if meth == "k1qcls_c16":
        same = [sorted(kk[i]) == sorted(ref["fk1c16_kept"][i]) for i in range(n) if routed[i]]
        checks[f"{bench} {meth}: routed questions keep what the K = 1 router + cover 16 keeps (>= 99%)"] = float(np.mean(same) >= 0.99)
        res["cls"][f"{bench} {meth} routed kept = K = 1 router + cover 16's"] = float(np.mean(same))
        zr = json.load(open(f"{d}/zoom.json"))["records"]
        covered = [i for i in range(n) if zr[i]["cover"]]
        checks[f"{bench} {meth}: >= 99% of the unrouted questions covered, no routed one"] = float(
            all(not routed[i] for i in covered) and len(covered) >= 0.99 * (~routed).sum())
        both = [i for i in covered if ref["fk1c16_zoom"][i]["cover"]]
        checks[f"{bench} {meth}: box = EXP-0026's K = 1 cover 16 box on >= 99%"] = float(
            len(both) >= 0.99 * len(covered) and np.mean([zr[i]["grid_box"] == ref["fk1c16_zoom"][i]["grid_box"] for i in both]) >= 0.99)
        ok = []
        for i in covered:
            box = a28.box_mask(zr[i]["grid_box"])
            k, own = np.array(kk[i]), np.array(recs[i]["own"])
            inside = int((box & pad[i]).sum())
            a_in, n_in = int(box[own].sum()), int(box[k].sum())
            ok.append(zr[i]["baseline_in_box"] == a_in and zr[i]["in_box"] == n_in == min(a_in + 16, KEEP, inside)
                      and set(own[box[own]]) <= set(k[box[k]]) and set(k[~box[k]]) <= set(own[~box[own]]))
        checks[f"{bench} {meth}: covered: in-box count = the CLS ranking's own + 16 (capped), more inside, a subset outside"] = float(np.mean(ok))
        rest = [i for i in range(n) if not routed[i] and not zr[i]["cover"]]
        checks[f"{bench} {meth}: unrouted but not covered: kept = the CLS ranking's own choice"] = float(
            np.mean([sorted(kk[i]) == recs[i]["own"] for i in rest])) if rest else 1.0
        c = [zr[i] for i in covered]
        res["in_box"][f"{bench} {meth}"] = {
            "covered_share": len(covered) / n, "mean_baseline_in_box": float(np.mean([x["baseline_in_box"] for x in c])),
            "mean_in_box": float(np.mean([x["in_box"] for x in c])), "mean_changed": float(np.mean([x["changed"] for x in c]))}


def main():
    out_dir = f"{OUT}/EXP-0030/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    rows = runs_rows()
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "in_box": {}, "cls": {}}
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, ans = {}, {}
        for (bb, meth), (exp, rid) in RUNS.items():
            if bb != bench:
                continue
            d = run_dir(bench, meth)
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
            r = rows[f"{exp}-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth], ans[meth] = s, [a[q] for q in b["order"]]
        vzr = json.load(open(f"{run_dir(bench, 'c16')}/vz.json"))["records"]
        ref = {"vz_own": [x["baseline"] if x["cover"] else x["dominant"] for x in vzr],
               "router_k1_kept": json.load(open(f"{run_dir(bench, 'router_k1')}/kept_tokens.json"))["kept"],
               "fk1c16_kept": json.load(open(f"{run_dir(bench, 'fk1c16')}/kept_tokens.json"))["kept"],
               "fk1c16_zoom": json.load(open(f"{run_dir(bench, 'fk1c16')}/zoom.json"))["records"]}
        for meth in CANDIDATES:
            check_cls(checks, res, bench, meth, b, run_dir(bench, meth), ref)
        for meth, base in (("k1qcls", "router_k1"), ("k1qcls_c16", "fk1c16")):
            same = [ans[meth][i] == ans[base][i] for i in range(n) if routed[i]]
            checks[f"{bench} {meth}: answers on the routed questions = {NAMES[base]}'s (>= 99%)"] = float(np.mean(same) >= 0.99)
            res["cls"][f"{bench} {meth} routed answers = {base}'s"] = float(np.mean(same))
        sc["splice"] = np.where(routed, sc["router_k1"], sc["c16"])
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

        pairs = [(c, y) for c in CANDIDATES for y in ("vz", "c16all", "c16", "splice", "fk1c16", "router_k1", "router")] + [
            ("k1qcls", "k1cls"), ("k1qcls_c16", "k1qcls"), ("vz", "router_k1"), ("splice", "vz")]
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
    up = all(delta[sel][bench][0] > 0 for bench in BENCHES)
    d6 = {bench: P[f"{bench} {sel} - c16all"] for bench in BENCHES}
    d6_mean = float(np.mean([d6[bench][0] for bench in BENCHES]))
    if all(v <= 0 for v in mean.values()):
        verdict = "不像：三个候选比 VisionZip-64 的两集平均都 ≤ 0，一层全看再按 VisionZip 排序在本设置下不成立"
    elif up and d6_mean > 0:
        verdict = "有希望（进入确认）"
    elif up:
        verdict = "不比 IDEA-0006 好：高于 VisionZip-64，但不高于 EXP-0028 的所选，交用户决定"
    else:
        verdict = "中间情况：交用户决定"
    strength = sum(delta[sel][bench][1] > 0 for bench in BENCHES)
    large = mean[sel] >= 2.0

    def reading(key, pos, neg):
        v = {bench: P[f"{bench} {key}"] for bench in BENCHES}
        m = float(np.mean([v[bench][0] for bench in BENCHES]))
        return {"pairs": v, "mean": m, "reading": pos if m > 0 else neg}

    attribution = {
        "full_layer": reading("k1cls - vz", "一层全看 + CLS 排序（47 个）好于 VisionZip-64（64 个，含合并的 token）",
                              "一层全看 + CLS 排序不比 VisionZip-64 好"),
        "ranking": reading("k1qcls - router_k1 [unrouted]", "K = 1 时，不路由的题上 CLS 排序好于 FastV 的注意力排序",
                           "K = 1 时，不路由的题上 CLS 排序不比 FastV 的注意力排序好"),
        "routing": reading("k1qcls - k1cls [routed]", "被路由的题上 query_vis 好于 CLS 排序", "被路由的题上 query_vis 不比 CLS 排序好"),
        "cover": reading("k1qcls_c16 - k1qcls", "一层全看之后覆盖 16 仍有增益", "一层全看之后覆盖 16 没有增益")}
    res["criteria"] = {"delta": delta, "mean": mean, "selected": sel, "verdict": verdict, "intervals_above_0": strength,
                       "minus_c16all": d6, "minus_c16all_mean": d6_mean, "large": large, "attribution": attribution}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0030.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0030（计划 v1）：一层全看，再按 VisionZip 的排序（IDEA-0007），开发集（VQAv2 dev2、GQA val400）", "", "## 核对", ""]
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
    L += ["", "## 按计划写定的选择与判读", "", "| 候选 − VisionZip-64 | VQAv2 dev2 | GQA val400 | Δ̄ |", "| --- | --- | --- | --- |"]
    L += [f"| {NAMES[c]} | {fmt(delta[c]['vqav2_dev2'])} | {fmt(delta[c]['gqa_val400'])} | {mean[c]:+.2f} |" for c in CANDIDATES]
    L += ["", f"- **选中**：{NAMES[sel]}",
          f"- **比 VisionZip-64**：VQAv2 {fmt(delta[sel]['vqav2_dev2'])}，GQA {fmt(delta[sel]['gqa_val400'])}；区间在 0 以上的集数 {strength}",
          f"- **比 EXP-0028 的所选（VisionZip-64 + 每题覆盖 16）**：VQAv2 {fmt(d6['vqav2_dev2'])}，GQA {fmt(d6['gqa_val400'])}，两集平均 {d6_mean:+.2f}",
          f"- **判读**：{verdict}",
          f"- **大小（只报告）**：两集平均 {mean[sel]:+.2f}，{'达到' if large else '没有达到'}写定的“提升大”（≥ +2.0）",
          "", "## 归因（写定的读法，不改选择与判读）", ""]
    for key, label in (("full_layer", "一层全看（一层全看 + CLS 排序 − VisionZip-64）"),
                       ("ranking", "排序（K = 1，不路由的题：CLS 排序 − FastV 的注意力排序）"),
                       ("routing", "路由（被路由的题：query_vis − CLS 排序）"), ("cover", "覆盖（+ 覆盖 16 − 不加）")):
        v = attribution[key]
        L.append(f"- **{label}**：VQAv2 {fmt(v['pairs']['vqav2_dev2'])}，GQA {fmt(v['pairs']['gqa_val400'])}，"
                 f"两集平均 {v['mean']:+.2f} → {v['reading']}")
    open(f"{out_dir}/0030.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
