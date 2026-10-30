"""EXP-0037 (plan v1, exploratory budget check of IDEA-0008 on the development sets, written 2026-10-31 before any EXP-0036 or
EXP-0037 result): does the layer-0 peek gain more over DivPrune when far fewer tokens are kept (16 instead of 64)?

  python analyze_0037.py          # writes code/output/EXP-0037/analysis/{0037.md,0037.json}
  python analyze_0037.py --dry    # the same code on EXP-0034 / EXP-0032 runs as stand-ins (budget 64, keep 61) -> tmp; not results

Runs (VQAv2 dev2 / GQA val400): DivPrune-16 R01 / R02, peek keep 13 R03 / R04. peek: K = 1, DivPrune's first 13 chosen
before layer 0, layer 0 lets every token see all patches as keys and values, layers 1 to 31 keep 13; with the dropped
patches' keys and values only at layer 0 its prefill FLOPs are just under DivPrune-16's (the same rule gave 61 for 64).
Reused: EXP-0034-R01 / R02 (peek 61), EXP-0032-R01 / R02 (DivPrune-64), EXP-0033-R03 / R04 (unpruned). Only budget 16 (not
32 too): IDEA-0008's limit is 30 runs and 25 were used (EXP-0034 to EXP-0036).
Reading per budget (as EXP-0034's Q1): peek - DivPrune-B, 95% interval above 0 on both sets -> "promising"; on one ->
"partly"; neither but two-set mean > 0 -> "positive, not detected"; else "unlike". Two-set mean >= +1.5 is noted "large".
Trend (written rule): two-set mean at 16 > at 64 (64: EXP-0034's +1.26, recomputed here) -> "the gain is larger at the
smaller budget"; otherwise "it is not larger". Described: by question type, against the unpruned run, FLOPs.
Bootstrap: resample images within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring, as analyze_0034).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402
import analyze_0034 as a34  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
DRY = "--dry" in sys.argv
BENCHES = ("vqav2_dev2", "gqa_val400")
N_IMG, N_BOOT = 576, 1000
if DRY:
    DST = "D:/6CJNM007/tmp/exp0037_prep/dry_out"
    BUDGETS = {64: 61}
    RUNS = {(b, "div", 64): ("EXP-0032", r, "divprune64") for b, r in zip(BENCHES, ("R01", "R02"))}
    RUNS.update({(b, "peek", 64): ("EXP-0034", r, "fastv61k1_div_peek_kll") for b, r in zip(BENCHES, ("R01", "R02"))})
else:
    DST = f"{OUT}/EXP-0037/analysis"
    BUDGETS = {16: 13}
    RUNS = {(b, "div", 16): ("EXP-0037", r, "divprune16") for b, r in zip(BENCHES, ("R01", "R02"))}
    RUNS.update({(b, "peek", 16): ("EXP-0037", r, "fastv13k1_div_peek_kll") for b, r in zip(BENCHES, ("R03", "R04"))})
    RUNS.update({(b, "div", 64): ("EXP-0032", r, "divprune64") for b, r in zip(BENCHES, ("R01", "R02"))})
    RUNS.update({(b, "peek", 64): ("EXP-0034", r, "fastv61k1_div_peek_kll") for b, r in zip(BENCHES, ("R01", "R02"))})
RUNS.update({(b, "vanilla", 0): ("EXP-0033", r, "vanilla") for b, r in zip(BENCHES, ("R03", "R04"))})
KEEP_OF = {**BUDGETS, 64: 61}
LARGE = 1.5
CHECKS = {}


def macs(n_other, kept, mode, budget):
    """Prefill multiply-adds (analyze_0034.macs, generalised to k kept and to DivPrune-B)."""
    def layer(n_full, n_kv, pairs):
        return n_full * a34.FULL + n_kv * a34.KV + pairs * 2 * a34.D
    causal = lambda n: n * (n + 1) // 2   # noqa: E731
    if mode in ("div", "vanilla"):
        n = n_other + (budget if mode == "div" else N_IMG)
        return 32 * layer(n, 0, causal(n))
    k = len(kept)
    n_small = n_other + k
    later = 31 * layer(n_small, 0, causal(n_small))
    q_pos = list(range(35)) + [35 + p for p in kept] + list(range(35 + N_IMG, n_other + N_IMG))
    return layer(n_small, N_IMG - k, sum(p + 1 for p in q_pos)) + later


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    rows = {}
    for exp in sorted({e for e, _, _ in RUNS.values()}):
        rows.update({x["run_id"]: x for x in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    for (bench, meth, budget), (exp, rid, cfg) in RUNS.items():
        x = rows.get(f"{exp}-{rid}", {})
        CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench}"] = float(x.get("status") == "completed" and x.get("config") == cfg and x.get("bench") == bench)
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {[k for k, v in CHECKS.items() if v < 1]}")
    res = {"pairs": {}, "table": {}, "flops": {}, "seconds_per_answer": {}, "means": {}}
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc = {}
        for (bb, meth, budget), (exp, rid, cfg) in RUNS.items():
            if bb != bench:
                continue
            d = f"{OUT}/{exp}/{exp}-{rid}"
            a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}{budget}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = a28.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {str(x["question_id"]): x["score"] for x in (json.loads(l) for l in open(f"{d}/{logf}", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}{budget}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            CHECKS[f"{bench} {meth}{budget}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(rows[f"{exp}-{rid}"]["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}{budget or ''}"] = float(rows[f"{exp}-{rid}"]["s_per_answer"])
            sc[(meth, budget)] = s
        res["flops"][bench] = {}
        for budget, keep in KEEP_OF.items():
            dd = f"{OUT}/{RUNS[(bench, 'div', budget)][0]}/{RUNS[(bench, 'div', budget)][0]}-{RUNS[(bench, 'div', budget)][1]}"
            dp = f"{OUT}/{RUNS[(bench, 'peek', budget)][0]}/{RUNS[(bench, 'peek', budget)][0]}-{RUNS[(bench, 'peek', budget)][1]}"
            pre = json.load(open(f"{dd}/div.json"))["records"]
            rec = json.load(open(f"{dp}/div.json"))["records"]
            kept = json.load(open(f"{dp}/kept_tokens.json"))["kept"]
            lt = json.load(open(f"{dp}/layer_tokens.json"))
            CHECKS[f"{bench} div{budget}: one selection record per question, keep {budget}"] = float(len(pre) == n and all(len(x["order"]) == budget for x in pre))
            CHECKS[f"{bench} peek{budget}: one record (mode prefill), one kept list, one row of layer lengths per prefill, keep {keep}"] = float(
                len(rec) == n == len(kept) == lt["n_prefills"] == lt["n_complete"] and all(x.get("peek") == "prefill" for x in rec)
                and all(len(x["order"]) == keep == len(set(x["order"])) and sorted(x["order"]) == sorted(k) for x, k in zip(rec, kept)))
            CHECKS[f"{bench} peek{budget}: its selection = the first {keep} of DivPrune-{budget}'s (>= 99%)"] = float(
                np.mean([x["order"] == y["order"][:keep] for x, y in zip(rec, pre)]) >= 0.99)
            CHECKS[f"{bench} peek{budget}: layer 0 sees the whole prompt, layers 1 to 31 that minus {N_IMG - keep}"] = float(np.mean(
                [len(t) == 32 and all(v == t[0] - (N_IMG - keep) for v in t[1:]) for t in lt["tokens_per_layer"]]))
            n_other = [t[0] - N_IMG for t in lt["tokens_per_layer"]]
            tot = {m: float(np.mean([macs(o, sorted(x["order"]), m, budget) for o, x in zip(n_other, rec)])) for m in ("div", "peek_kv")}
            res["flops"][bench][str(budget)] = {"peek_kv_relative_to_div": tot["peek_kv"] / tot["div"], "keep": keep}
        ims = b["img_of"]
        by_img = [np.flatnonzero(ims == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100 * d[i][m[i]].mean() for i in draws if m[i].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]
        groups = b.get("groups", {})
        for budget in KEEP_OF:
            for x, y in ((("peek", budget), ("div", budget)), (("vanilla", 0), ("peek", budget)), (("vanilla", 0), ("div", budget))):
                key = f"{bench} {x[0]}{x[1] or ''} - {y[0]}{y[1]}"
                res["pairs"][key] = paired(x, y)
                if x[0] == "peek":
                    for g, m in groups.items():
                        res["pairs"][f"{key} {g}"] = paired(x, y, m)
        res["table"][bench] = {f"{m}{bg or ''}": {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[mm].mean()) for g, mm in groups.items() if mm.any()}}
                               for (m, bg), s in sc.items()}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]
    readings = {}
    for budget in KEEP_OF:
        v = [P[f"{bench} peek{budget} - div{budget}"] for bench in BENCHES]
        mean = float(np.mean([x[0] for x in v]))
        n_pos = sum(x[1] > 0 for x in v)
        word = "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike"
        readings[str(budget)] = {"reading": word, "mean": mean, "large": mean >= LARGE}
    res["readings"] = readings
    if not DRY:
        res["trend"] = "the gain is larger at the smaller budget" if readings["16"]["mean"] > readings["64"]["mean"] else "it is not larger"
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0037.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0037（计划 v1）：IDEA-0008 的预算扫描（开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {v['accuracy']:.2f}" for k, v in res["table"][bench].items()),
              "", "计算量（peek 被剪的只算 K、V，相对 DivPrune-B）：" + "；".join(f"B = {k}：留 {v['keep']}，{v['peek_kv_relative_to_div']:.4f}" for k, v in res["flops"][bench].items())]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "每题秒数：" + "，".join(f"{k} {v:.4f}" for k, v in res["seconds_per_answer"].items())]
    L += ["", "## 按计划写定的读法", ""]
    for budget in KEEP_OF:
        rd = readings[str(budget)]
        L.append(f"- 预算 {budget}（peek 留 {KEEP_OF[budget]}）：" + "，".join(f"{bench} {fmt(P[f'{bench} peek{budget} - div{budget}'])}" for bench in BENCHES)
                 + f"，两集平均 {rd['mean']:+.2f} → {rd['reading']}" + ("；提升大" if rd["large"] else ""))
    if "trend" in res:
        L.append(f"- 趋势：{res['trend']}")
    open(f"{DST}/0037.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
