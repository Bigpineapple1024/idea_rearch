"""EXP-0032 (plan v1): DivPrune in our pipeline and whether one full LLM layer helps a diversity selection, on VQAv2 dev2
and GQA val400.

  python analyze_0032.py      # writes code/output/EXP-0032/analysis/{0032.md,0032.json}

div64 (R01, R02): DivPrune's max-min diversity (divprune.py, written from the paper), 64 of the 576 projected patch
tokens kept before the LLM. k1div (R03, R04): layer 0 sees all 576, from layer 1 on the first 47 of the same greedy
selection on the same features stay (63.5 tokens per layer), so only where the selection is applied differs. Q1
(calibration, described): div64 - VisionZip-64 (EXP-0028-R01, R02). Q2 (placement): k1div - div64, both 95% intervals
above 0: helps on both sets; one: partly; neither with a positive two-set mean: positive, not detected; otherwise: does
not help. Described alongside: IDEA-0006's selected configuration (EXP-0028-R19, R20), the K = 1 router (EXP-0024-R06,
R07), IDEA-0007's CLS ranking at K = 1 (EXP-0030-R01, R02) and EXP-0030's selected configuration (read from its sealed
analysis). Per-question scores are recomputed from answers.jsonl and checked against each run's own score file, and
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

OUT = a28.OUT
BENCHES = a28.BENCHES
KEEP_PRE, KEEP_K1 = 64, 47
I7_RUNS = {"k1cls": ("R01", "R02"), "k1qcls": ("R03", "R04"), "k1qcls_c16": ("R05", "R06")}
NAMES = {"div64": "DivPrune-64（在 LLM 之前，本实现）", "k1div": "一层全看 + 同一选择的前 47 个（K = 1）", "vz": "VisionZip-64",
         "c16all": "VisionZip-64 + 每题覆盖 16（IDEA-0006 的所选）", "router_k1": "K = 1 路由（EXP-0024）",
         "k1cls": "一层全看 + CLS 排序（EXP-0030）", "i7": "EXP-0030 的所选（IDEA-0007）"}
N_BOOT = 1000


def i7_selected():
    """EXP-0030's selected configuration (whatever its verdict), from its sealed analysis."""
    return json.load(open(f"{OUT}/EXP-0030/analysis/0030.json", encoding="utf-8"))["criteria"]["selected"]


def run_table(i7):
    r = {("vqav2_dev2", "div64"): ("EXP-0032", "R01"), ("gqa_val400", "div64"): ("EXP-0032", "R02"),
         ("vqav2_dev2", "k1div"): ("EXP-0032", "R03"), ("gqa_val400", "k1div"): ("EXP-0032", "R04"),
         ("vqav2_dev2", "vz"): ("EXP-0028", "R01"), ("gqa_val400", "vz"): ("EXP-0028", "R02"),
         ("vqav2_dev2", "c16all"): ("EXP-0028", "R19"), ("gqa_val400", "c16all"): ("EXP-0028", "R20"),
         ("vqav2_dev2", "router_k1"): ("EXP-0024", "R06"), ("gqa_val400", "router_k1"): ("EXP-0024", "R07"),
         ("vqav2_dev2", "k1cls"): ("EXP-0030", "R01"), ("gqa_val400", "k1cls"): ("EXP-0030", "R02")}
    a, b = I7_RUNS[i7]
    r[("vqav2_dev2", "i7")], r[("gqa_val400", "i7")] = ("EXP-0030", a), ("EXP-0030", b)
    return r


def runs_rows(runs):
    rows = {}
    for exp in ("EXP-0024", "EXP-0028", "EXP-0030", "EXP-0032"):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for (bench, meth), (exp, rid) in runs.items():
        if rows.get(f"{exp}-{rid}", {}).get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
    want = {"EXP-0032-R01": "divprune64", "EXP-0032-R02": "divprune64", "EXP-0032-R03": "fastv64k1_div_kl",
            "EXP-0032-R04": "fastv64k1_div_kl", "EXP-0028-R01": "visionzip64", "EXP-0028-R02": "visionzip64"}
    bad = {r: rows[r]["config"] for r, c in want.items() if rows[r]["config"] != c}
    if bad:
        sys.exit(f"runs with another configuration than planned: {bad}")
    return rows


def main():
    out_dir = f"{OUT}/EXP-0032/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    i7 = i7_selected()
    runs = run_table(i7)
    rows = runs_rows(runs)
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "div": {}, "i7_selected": i7}
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, d_of = {}, {}
        for (bb, meth), (exp, rid) in runs.items():
            if bb != bench:
                continue
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
            r = rows[f"{exp}-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth] = s
        pre = json.load(open(f"{d_of['div64']}/div.json"))
        recs = pre["records"]
        checks[f"{bench} div64: one selection record per question, in order, keep 64"] = float(
            pre["n_images"] == n == len(recs) and [x["index"] for x in recs] == list(range(n)) and pre["keep"] == KEEP_PRE)
        checks[f"{bench} div64: each keeps 64 distinct patches, fed in their original order"] = float(np.mean(
            [len(x["order"]) == KEEP_PRE == len(set(x["order"])) and x["kept"] == sorted(x["order"])
             and all(0 <= p < 576 for p in x["order"]) for x in recs]))
        k1 = json.load(open(f"{d_of['k1div']}/div.json"))
        kept = json.load(open(f"{d_of['k1div']}/kept_tokens.json"))["kept"]
        k1r = k1["records"]
        checks[f"{bench} k1div: one selection record and one kept list per prefill, in order, keep 47"] = float(
            k1["n_prefills"] == n == len(k1r) == len(kept) and [x["index"] for x in k1r] == list(range(n)) and k1["keep"] == KEEP_K1)
        checks[f"{bench} k1div: kept = the 47 logged, in their original order"] = float(np.mean(
            [len(x["order"]) == KEEP_K1 == len(set(x["order"])) and sorted(x["order"]) == sorted(k) for x, k in zip(k1r, kept)]))
        same = float(np.mean([x["order"] == y["order"][:KEEP_K1] for x, y in zip(k1r, recs)]))
        checks[f"{bench} k1div: its selection = the first 47 of div64's on the same features (>= 99%)"] = float(same >= 0.99)
        res["div"][f"{bench} k1div order = div64's first 47"] = same
        res["div"][f"{bench} div64 padding patches kept (mean)"] = float(np.mean([
            sum(not bool(m) for m in a28_allowed(b["size"][i])[x["order"]]) for i, x in enumerate(recs)]))
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

        pairs = [("div64", "vz"), ("k1div", "div64"), ("k1div", "vz"), ("div64", "c16all"), ("k1div", "c16all"),
                 ("div64", "i7"), ("k1div", "i7"), ("k1div", "k1cls"), ("k1div", "router_k1"), ("div64", "router_k1")]
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

    def two(x, y):
        v = {bench: P[f"{bench} {x} - {y}"] for bench in BENCHES}
        return v, float(np.mean([v[bench][0] for bench in BENCHES])), sum(v[bench][1] > 0 for bench in BENCHES)

    q1, m1, _ = two("div64", "vz")
    q2, m2, n2 = two("k1div", "div64")
    calib = ("明显强于 VisionZip-64（两集平均 ≥ +1.0，与文献的方向一致）" if m1 >= 1.0 else
             "略强于 VisionZip-64（两集平均在 0 与 +1.0 之间）" if m1 > 0 else "不强于 VisionZip-64（与文献不一致，先查实现）")
    if n2 == 2:
        place = "一层全看对多样性选择有帮助（两个集的区间都在 0 以上）"
    elif n2 == 1:
        place = "一层全看对多样性选择部分有帮助（只有一个集的区间在 0 以上）"
    elif m2 > 0:
        place = "方向为正，但两个集的区间都含 0"
    else:
        place = "一层全看对多样性选择没有帮助（两集平均 ≤ 0，与 DivPrune 的 Table 3 一致）"
    res["criteria"] = {"Q1_pairs": q1, "Q1_mean": m1, "Q1_reading": calib, "Q2_pairs": q2, "Q2_mean": m2,
                       "Q2_intervals_above_0": n2, "Q2_reading": place, "k1div_minus_vz_mean": two("k1div", "vz")[1],
                       "base_switch_suggested": m1 >= 1.0, "placement_candidate": m2 > 0 and n2 >= 1}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0032.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0032（计划 v1）：本流程中的 DivPrune-64，以及一层全看对多样性选择的作用（VQAv2 dev2、GQA val400）", "",
         f"EXP-0030 的所选（只作描述）：{i7}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    L += ["", "实测比例：" + "；".join(f"{k} {x:.4f}" for k, x in res["div"].items())]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {NAMES[m]} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "每题秒数：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    L += ["", "## 按计划写定的读法", "",
          f"- **Q1**（校准，描述：DivPrune-64 − VisionZip-64）：VQAv2 dev2 {fmt(q1['vqav2_dev2'])}，GQA val400 {fmt(q1['gqa_val400'])}，"
          f"两集平均 {m1:+.2f} → {calib}",
          f"- **Q2**（放置：一层全看 + 前 47 个 − DivPrune-64）：VQAv2 dev2 {fmt(q2['vqav2_dev2'])}，GQA val400 {fmt(q2['gqa_val400'])}，"
          f"两集平均 {m2:+.2f} → {place}",
          f"- **交用户决定的两点**：{'建议以 DivPrune-64 为下一轮的底座' if m1 >= 1.0 else '不建议仅因本实验换底座'}；"
          f"{'一层全看 + 多样性可作为候选方法，写新数据上的确认计划后再运行' if m2 > 0 and n2 >= 1 else '一层全看 + 多样性不作为候选方法'}"]
    open(f"{out_dir}/0032.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def a28_allowed(size):
    """Non-padding patches of LLaVA's square image (fastv_llava._padding_mask), as a numpy mask."""
    from fastv_llava import _padding_mask
    return ~_padding_mask(*size, 24).numpy()


if __name__ == "__main__":
    main()
