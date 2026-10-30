"""EXP-0033 (plan v1, v1.1): step 4, round 2 - exploratory combinations of two methods on VQAv2 dev2 and GQA val400 at a
layer average of 64. Exploratory: no criteria; the plan fixes what is reported and three kinds of surprise.

  python analyze_0033.py      # writes code/output/EXP-0033/analysis/{0033.md,0033.json}

Combinations (A, B, A+B): X1 VisionZip-64, FastV-64 (plain, K = 2, keep 30), VisionZip-128 then plain FastV keeping 60
from layer 2; X2 VisionZip-64, DivPrune-64, VisionZip's CLS top-128 candidates then max-min 64; X3 DivPrune-64, the K = 2
router (IDEA-0003), DivPrune-64 with the router's IDF x visibility query taking round(64 S) first on routed questions;
X4 DivPrune-64, VisionZip-64, DivPrune 54 + VisionZip's contextual merge into 10. Surprises (on either set):
complementary - A+B minus A and minus B both above 0, at least one 95% interval above 0; cancelling - both below 0, at
least one interval below 0; reversal by question type - a group of >= 300 questions where A+B minus the better single
method of that group has an interval excluding 0 and the opposite sign to the same difference on the whole set.
Reused runs: VisionZip-64 EXP-0028-R01, R02; the K = 2 router EXP-0024-R02, R03; DivPrune-64 EXP-0032-R01, R02 when both
completed, else EXP-0033-R13, R14. Per-question scores are recomputed from answers.jsonl and checked against each run's
own score file; paired differences resample images with replacement (1000 draws, seed 0) within each set, as in
analyze_0028.
Implementation checks on the runs' logs (stop below 99%): X1 128 image tokens at the LLM input, 60 kept, the tokens
entering each layer (layers 0-1 the same, 2-31 that minus 68); X2 the kept are the candidates at the logged max-min picks;
X3 the routing rule on the question text, S and round(64 S) recomputed offline, the query's picks first, unrouted
questions with DivPrune-64's order; X4 the first 54 of DivPrune-64's order and 10 merge targets outside them. What needs
the model's internals (X1's 128 equal VisionZip-128 alone, X2's candidates and max-min recomputed, X4's merge recomputed)
is check_combo.py's (CPU, small random model), rerun in the repository before the runs.
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
N_BOOT = 1000
MIN_GROUP = 300
COMBOS = {"X1": ("vz", "fastv64", "x1"), "X2": ("vz", "div64", "x2"), "X3": ("div64", "router", "x3"), "X4": ("div64", "vz", "x4")}
VIS = f"{OUT}/EXP-0015/visibility/visibility.npy"          # X3's visibility table (div64_qroute's --combo-visibility)
CONFIG = {"fastv64": "fastv64", "vanilla": "vanilla", "x1": "vz128_fastv60_kll", "x2": "vzcand128_div64", "x3": "div64_qroute",
          "x4": "div54_ctx10", "div64": "divprune64", "vz": "visionzip64", "router": "fastv64_qroute_kl"}
R33 = {"fastv64": ("R01", "R02"), "vanilla": ("R03", "R04"), "x1": ("R05", "R06"), "x2": ("R07", "R08"), "x3": ("R09", "R10"),
       "x4": ("R11", "R12")}
NAMES = {"vz": "VisionZip-64", "fastv64": "FastV-64（原版，K = 2，留 30）", "div64": "DivPrune-64", "router": "K = 2 路由（IDEA-0003）",
         "vanilla": "不剪枝", "x1": "X1：VisionZip-128 + FastV 留 60", "x2": "X2：CLS 前 128 候选 → DivPrune 64",
         "x3": "X3：DivPrune-64 + 句式路由的题目查询", "x4": "X4：DivPrune 54 + 上下文合并 10"}


def offline_share(bench):
    """S per question (file order) from the question file, the IDF over it and the visibility table (IDEA-0002's formulas),
    as analyze_0016.counts_check; the run computes it from the prompt ids of each prefill."""
    from transformers import AutoTokenizer
    import run_eval
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    vis = np.load(VIS)
    qfile = a28.FILES[bench][0]
    idf = run_eval.question_idf(tok, qfile, "vicuna_v1")
    s = []
    for line in open(qfile, encoding="utf-8"):
        ids = run_eval.prompt_ids_after_image(tok, json.loads(line)["text"], "vicuna_v1")
        w = np.array([idf["idf"].get(t, idf["unseen"]) for t in ids])
        s.append(float((w * vis[ids]).sum() / w.sum()) if w.sum() > 0 else 0.0)
    return np.array(s)


def all_rows():
    rows = {}
    for exp in ("EXP-0024", "EXP-0028", "EXP-0032", "EXP-0033"):
        p = f"{OUT}/{exp}/runs.tsv"
        if os.path.exists(p):
            with open(p) as f:
                rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    return rows


def run_table(rows):
    r = {}
    for meth, (a, b) in R33.items():
        r[("vqav2_dev2", meth)], r[("gqa_val400", meth)] = ("EXP-0033", a), ("EXP-0033", b)
    r[("vqav2_dev2", "vz")], r[("gqa_val400", "vz")] = ("EXP-0028", "R01"), ("EXP-0028", "R02")
    r[("vqav2_dev2", "router")], r[("gqa_val400", "router")] = ("EXP-0024", "R02"), ("EXP-0024", "R03")
    own = not all(rows.get(f"EXP-0032-{x}", {}).get("status") == "completed" for x in ("R01", "R02"))
    r[("vqav2_dev2", "div64")], r[("gqa_val400", "div64")] = (("EXP-0033", "R13"), ("EXP-0033", "R14")) if own else (
        ("EXP-0032", "R01"), ("EXP-0032", "R02"))
    for (bench, meth), (exp, rid) in r.items():
        x = rows.get(f"{exp}-{rid}", {})
        if x.get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
        if x.get("config") != CONFIG[meth] or x.get("bench") != bench:
            sys.exit(f"{exp}-{rid} has {x.get('config')} x {x.get('bench')}, planned {CONFIG[meth]} x {bench}")
    return r, own


def main():
    out_dir = f"{OUT}/EXP-0033/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    rows = all_rows()
    runs, own_div = run_table(rows)
    checks = {}
    res = {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "jaccard": {}, "flips": {}, "surprises": {},
           "div64_runs": "EXP-0033-R13, R14" if own_div else "EXP-0032-R01, R02"}
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
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
        div = json.load(open(f"{d_of['div64']}/div.json"))["records"]
        checks[f"{bench} div64: one selection record per question, keep 64"] = float(len(div) == n and all(len(x["order"]) == 64 for x in div))
        # X1: plain FastV keeps 60 of VisionZip-128's tokens from layer 2
        k = json.load(open(f"{d_of['x1']}/kept_tokens.json"))
        checks[f"{bench} x1: one kept list per prefill, 128 image tokens at the LLM input, 60 kept"] = float(
            k["n_prefills"] == n == len(k["kept"]) and k["n_image_tokens"] == [128] and k["keep"] == 60)
        checks[f"{bench} x1: each kept list has 60 distinct positions among the 128"] = float(np.mean(
            [len(x) == 60 == len(set(x)) and all(0 <= p < 128 for p in x) for x in k["kept"]]))
        lt = json.load(open(f"{d_of['x1']}/layer_tokens.json"))
        checks[f"{bench} x1: one complete row of 32 layer lengths per prefill"] = float(
            lt["n_prefills"] == n == lt["n_complete"] and lt["n_layers"] == 32)
        checks[f"{bench} x1: layers 0 and 1 see the same length, layers 2 to 31 that minus 68 (128 -> 60 image tokens)"] = float(np.mean(
            [len(t) == 32 and t[1] == t[0] and all(x == t[0] - 68 for x in t[2:]) for t in lt["tokens_per_layer"]]))
        # X2: CLS candidates, then max-min
        c2 = json.load(open(f"{d_of['x2']}/combo.json"))
        r2 = c2["records"]
        checks[f"{bench} x2: one record per question, cand 128, keep 64"] = float(
            c2["n_images"] == n == len(r2) and [x["index"] for x in r2] == list(range(n)) and c2["cand"] == 128 and c2["keep"] == 64)
        checks[f"{bench} x2: 128 distinct sorted candidates; the kept are the candidates at the 64 picks, sorted"] = float(np.mean(
            [len(x["cand"]) == 128 == len(set(x["cand"])) and x["cand"] == sorted(x["cand"]) and len(x["order"]) == 64 == len(set(x["order"]))
             and x["kept"] == sorted(x["cand"][o] for o in x["order"]) for x in r2]))
        # X3: routed questions take round(64 S) by the query first; the others keep DivPrune-64's order
        c3 = json.load(open(f"{d_of['x3']}/combo.json"))
        r3 = c3["records"]
        checks[f"{bench} x3: one record per question, keep 64"] = float(c3["n_images"] == n == len(r3) and c3["keep"] == 64)
        checks[f"{bench} x3: routing = the rule on the question text"] = float(np.mean([x["routed"] == bool(rt) for x, rt in zip(r3, routed)]))
        rr = [x for x in r3 if x["routed"]]
        s_off = offline_share(bench)
        checks[f"{bench} x3: routed: S and round(64 S) as recomputed offline from the IDF and visibility tables"] = float(np.mean(
            [abs(x["share"] - s_off[x["index"]]) < 1e-4 and x["n_query"] == round(64 * s_off[x["index"]]) for x in rr])) if rr else 0.0
        checks[f"{bench} x3: routed: round(64 S) first by the query, the max-min order continues"] = float(np.mean(
            [x["n_query"] == round(64 * x["share"]) and x["order"][:x["n_query"]] == x["first"] and len(x["order"]) == 64 == len(set(x["order"]))
             for x in rr])) if rr else 0.0
        checks[f"{bench} x3: unrouted: DivPrune-64's own order"] = float(np.mean([x["order"] == y["order"] for x, y in zip(r3, div)
                                                                                  if not x["routed"]]))
        # X4: 54 by max-min, 10 merged
        c4 = json.load(open(f"{d_of['x4']}/combo.json"))
        r4 = c4["records"]
        checks[f"{bench} x4: one record per question, keep 54, contextual 10"] = float(
            c4["n_images"] == n == len(r4) and c4["keep"] == 54 and c4["contextual"] == 10)
        checks[f"{bench} x4: 54 distinct kept, 10 distinct merge targets outside them"] = float(np.mean(
            [len(x["kept"]) == 54 == len(set(x["kept"])) and x["kept"] == sorted(x["order"]) and len(set(x["targets"])) == 10
             and not set(x["targets"]) & set(x["kept"]) for x in r4]))
        checks[f"{bench} x4: its 54 = the first 54 of DivPrune-64's order"] = float(np.mean([x["order"] == y["order"][:54]
                                                                                            for x, y in zip(r4, div)]))
        res["jaccard"][f"{bench} x2 vs div64"] = float(np.mean([len(set(x["kept"]) & set(y["kept"])) / len(set(x["kept"]) | set(y["kept"]))
                                                                 for x, y in zip(r2, div)]))
        res["jaccard"][f"{bench} x3 vs div64 (routed)"] = float(np.mean(
            [len(set(x["kept"]) & set(y["kept"])) / len(set(x["kept"]) | set(y["kept"])) for x, y in zip(r3, div) if x["routed"]])) if rr else None
        res["jaccard"][f"{bench} x3 routed share, mean S, mean query count"] = [float(np.mean([x["routed"] for x in r3])),
                                                                                 float(np.mean([x["share"] for x in rr])) if rr else None,
                                                                                 float(np.mean([x["n_query"] for x in rr])) if rr else None]
        by_img = [np.flatnonzero(b["img_of"] == k_) for k_ in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k_] for k_ in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

        pairs = []
        for name, (A, B, X) in COMBOS.items():
            pairs += [(X, A), (X, B), (B, A)]
            res["flips"][f"{bench} {name}"] = {f"{X}-{Y}": [int(((sc[X] - sc[Y]) != 0).sum()), int(((sc[X] - sc[Y]) > 0).sum()),
                                                            int(((sc[X] - sc[Y]) < 0).sum())] for Y in (A, B)}
        pairs += [("vanilla", m_) for m_ in ("vz", "fastv64", "div64", "router", "x1", "x2", "x3", "x4")]
        for x, y in dict.fromkeys(pairs):
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        for name, (A, B, X) in COMBOS.items():
            xa, xb = res["pairs"][f"{bench} {X} - {A}"], res["pairs"][f"{bench} {X} - {B}"]
            kinds = []
            if xa[0] > 0 and xb[0] > 0 and (xa[1] > 0 or xb[1] > 0):
                kinds.append("互补（比两者都好）")
            if xa[0] < 0 and xb[0] < 0 and (xa[2] < 0 or xb[2] < 0):
                kinds.append("互相抵消（比两者都差）")
            best = A if sc[A].mean() >= sc[B].mean() else B
            whole = paired(X, best)
            for g, m in b["groups"].items():
                if m.sum() < MIN_GROUP:
                    continue
                gb = A if sc[A][m].mean() >= sc[B][m].mean() else B
                v = paired(X, gb, m)
                if v and (v[1] > 0 or v[2] < 0) and whole[0] != 0 and np.sign(v[0]) != np.sign(whole[0]):
                    kinds.append(f"题型反转：{g}（{len(m.nonzero()[0])} 题，A+B − {NAMES[gb]} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]，"
                                 f"整体 {whole[0]:+.2f}）")
            res["surprises"][f"{bench} {name}"] = kinds or ["没有意外"]
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items()}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k_: v for k_, v in checks.items() if v < 0.99})
        sys.exit(1)
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0033.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    P = res["pairs"]
    L = ["# EXP-0033（计划 v1）：第 ④ 步第二轮，探索性组合（VQAv2 dev2、GQA val400；层平均 64）", "",
         f"DivPrune-64 用的运行：{res['div64_runs']}", "", "## 核对", ""]
    L += [f"- {k_}: {100 * x:.1f}%" for k_, x in checks.items()]
    L += ["", "选择记录：" + "；".join(f"{k_} {json.dumps(x)}" for k_, x in res["jaccard"].items())]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {NAMES[m]} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 每个组合（A+B − A、A+B − B、B − A；百分点，按图像重抽 1000 次）", ""]
    for name, (A, B, X) in COMBOS.items():
        L += [f"### {name}：A = {NAMES[A]}，B = {NAMES[B]}，A+B = {NAMES[X]}", "", "| 比较 | VQAv2 dev2 | GQA val400 |", "| --- | --- | --- |"]
        for x, y in ((X, A), (X, B), (B, A)):
            L.append(f"| {NAMES[x]} − {NAMES[y]} | {fmt(P[f'vqav2_dev2 {x} - {y}'])} | {fmt(P[f'gqa_val400 {x} - {y}'])} |")
        L += ["", "- 按计划写定的“意外”：" + "；".join(f"{bench} " + "、".join(res["surprises"][f"{bench} {name}"]) for bench in BENCHES),
              "- 逐题翻转（得分不同的题、变好、变差）：" + "；".join(f"{bench} {json.dumps(res['flips'][f'{bench} {name}'])}" for bench in BENCHES), ""]
    L += ["## 全部配对差（含分组）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k_} | {fmt(x)} |" for k_, x in P.items() if x]
    L += ["", "每题秒数：" + "，".join(f"{k_} {x:.4f}" for k_, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k_} {x:.2f}" for k_, x in res["peak_gib"].items())]
    open(f"{out_dir}/0033.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
