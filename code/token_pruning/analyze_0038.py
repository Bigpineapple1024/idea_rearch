"""EXP-0038 (plan v1, IDEA-0009 on the development sets): does the layer-0 K/V-only peek add on top of standard prune-then-feed
pruning when the numbering is the same (contiguous from layer 1, as DivPrune before the LLM)?

  python analyze_0038.py          # writes code/output/EXP-0038/analysis/{0038.md,0038.json}
  python analyze_0038.py --dry    # the same code on EXP-0034 / EXP-0032 runs as stand-ins -> tmp; not results

Runs (VQAv2 dev2 / GQA val400): peekc R01 / R07 (K = 1, DivPrune's 61 chosen before layer 0, layer 0 lets every token see all
patches at the prompt's own positions, from layer 1 on the kept tokens and the text numbered contiguously); nopeekc R03 / R04
(the same, layer 0 sees none of the dropped patches); DivPrune-61 R05 / R06 (before the LLM). Reused: DivPrune-64
(EXP-0032-R01 / R02), peek and nopeek with the original numbering (EXP-0034-R01 to R04), unpruned (EXP-0033-R03 / R04).
v1.1 (plan v1.1, written before R07 ran): peekc on GQA val400 is R07, the rerun of R02, which exited with code 127 about 30 s
after it started, before its first answer and with no Python error (as EXP-0002-R16, EXP-0018-R01, EXP-0022-R03, EXP-0024-R16
and EXP-0026-R05 did); R02 stays a failed run. Nothing else changed.
Q1 (at about equal FLOPs): peekc - DivPrune-64; 95% interval above 0 on both sets -> "promising", on one -> "partly", neither
but two-set mean > 0 -> "positive, not detected", else "unlike"; two-set mean >= +1.5 noted "large". Q2 (the peek alone, the
same numbering at every layer): peekc - nopeekc, read the same way. Described: nopeekc - DivPrune-61, DivPrune-61 - DivPrune-64,
peekc - peek, nopeekc - nopeek, unpruned - peekc, by question type; FLOPs (analyze_0034.macs).
Bootstrap: resample images within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402
import analyze_0034 as a34  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
DRY = "--dry" in sys.argv
BENCHES = ("vqav2_dev2", "gqa_val400")
N_IMG, KEEP, N_BOOT, LARGE = 576, 61, 1000, 1.5
if DRY:
    DST = "D:/6CJNM007/tmp/exp0038/dry_out"
    NEW = {"peekc": ("EXP-0034", ("R01", "R02"), "fastv61k1_div_peek_kll", "prefill"),
           "nopeekc": ("EXP-0034", ("R03", "R04"), "fastv61k1_div_nopeek_kll", "none"),
           "div61": ("EXP-0032", ("R01", "R02"), "divprune64", None)}
    DIV61_KEEP = 64
else:
    DST = f"{OUT}/EXP-0038/analysis"
    NEW = {"peekc": ("EXP-0038", ("R01", "R07"), "fastv61k1_div_peekc_kll", "prefill"),
           "nopeekc": ("EXP-0038", ("R03", "R04"), "fastv61k1_div_nopeekc_kll", "none"),
           "div61": ("EXP-0038", ("R05", "R06"), "divprune61", None)}
    DIV61_KEEP = 61
OLD = {"div64": ("EXP-0032", ("R01", "R02"), "divprune64", None), "peek": ("EXP-0034", ("R01", "R02"), "fastv61k1_div_peek_kll", "prefill"),
       "nopeek": ("EXP-0034", ("R03", "R04"), "fastv61k1_div_nopeek_kll", "none"), "vanilla": ("EXP-0033", ("R03", "R04"), "vanilla", None)}
RUNS = {**NEW, **OLD}
CHECKS = {}


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    rows = {}
    for exp in sorted({v[0] for v in RUNS.values()}):
        rows.update({x["run_id"]: x for x in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    for meth, (exp, rids, cfg, _) in RUNS.items():
        for bench, rid in zip(BENCHES, rids):
            x = rows.get(f"{exp}-{rid}", {})
            CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench} ({meth})"] = float(x.get("status") == "completed" and x.get("config") == cfg and x.get("bench") == bench)
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {[k for k, v in CHECKS.items() if v < 1]}")
    res = {"pairs": {}, "table": {}, "flops": {}, "seconds_per_answer": {}}
    for bi, bench in enumerate(BENCHES):
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, d_of = {}, {}
        for meth, (exp, rids, cfg, _) in RUNS.items():
            d = d_of[meth] = f"{OUT}/{exp}/{exp}-{rids[bi]}"
            a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = a28.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {str(x["question_id"]): x["score"] for x in (json.loads(l) for l in open(f"{d}/{logf}", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            CHECKS[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(rows[f"{exp}-{rids[bi]}"]["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(rows[f"{exp}-{rids[bi]}"]["s_per_answer"])
            sc[meth] = s
        pre = json.load(open(f"{d_of['div64']}/div.json"))["records"]
        d61 = json.load(open(f"{d_of['div61']}/div.json"))
        CHECKS[f"{bench} div61: one selection record per question, keep {DIV61_KEEP}, = the first {DIV61_KEEP} of DivPrune-64's order (>= 99%)"] = float(
            len(d61["records"]) == n and all(len(x["order"]) == DIV61_KEEP for x in d61["records"])
            and np.mean([x["order"] == y["order"][:DIV61_KEEP] for x, y in zip(d61["records"], pre)]) >= 0.99)
        others = []
        for meth in ("peekc", "nopeekc"):
            mode = RUNS[meth][3]
            rec = json.load(open(f"{d_of[meth]}/div.json"))["records"]
            kept = json.load(open(f"{d_of[meth]}/kept_tokens.json"))["kept"]
            lt = json.load(open(f"{d_of[meth]}/layer_tokens.json"))
            CHECKS[f"{bench} {meth}: one record (mode {mode}), one kept list, one row of layer lengths per prefill, 61 kept = logged"] = float(
                len(rec) == n == len(kept) == lt["n_prefills"] == lt["n_complete"] and all(x.get("peek") == mode for x in rec)
                and all(len(x["order"]) == KEEP == len(set(x["order"])) and sorted(x["order"]) == sorted(k) for x, k in zip(rec, kept)))
            CHECKS[f"{bench} {meth}: its selection = the first 61 of DivPrune-64's (>= 99%)"] = float(np.mean([x["order"] == y["order"][:KEEP] for x, y in zip(rec, pre)]) >= 0.99)
            CHECKS[f"{bench} {meth}: layer 0 sees the whole prompt, layers 1 to 31 that minus 515"] = float(np.mean(
                [len(t) == 32 and all(v == t[0] - (N_IMG - KEEP) for v in t[1:]) for t in lt["tokens_per_layer"]]))
            others.append([t[0] - N_IMG for t in lt["tokens_per_layer"]])
        CHECKS[f"{bench}: peekc and nopeekc see the same prompts"] = float(others[0] == others[1])
        kept_p = [x["order"] for x in json.load(open(f"{d_of['peekc']}/div.json"))["records"]]
        tot = {m: float(np.mean([a34.macs(o, sorted(k), m) for o, k in zip(others[0], kept_p)])) for m in ("div64", "vanilla", "nopeek", "peek_kv", "peek_impl")}
        res["flops"][bench] = {"mean_non_image_tokens": float(np.mean(others[0])),
                               "relative_to_div64": {{"nopeek": "div61_or_nopeekc", "peek_kv": "peekc_kv"}.get(m, m): v / tot["div64"] for m, v in tot.items()}}
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
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
        for x, y in (("peekc", "div64"), ("peekc", "nopeekc"), ("nopeekc", "div61"), ("div61", "div64"), ("peekc", "peek"), ("nopeekc", "nopeek"), ("vanilla", "peekc")):
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            for g, m in groups.items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        res["table"][bench] = {m: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[mm].mean()) for g, mm in groups.items() if mm.any()}} for m, s in sc.items()}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]

    def reading(x, y):
        v = [P[f"{bench} {x} - {y}"] for bench in BENCHES]
        mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
        return {"reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
                "mean": mean, "large": mean >= LARGE}
    res["criteria"] = {"Q1": reading("peekc", "div64"), "Q2": reading("peekc", "nopeekc")}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0038.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0038（计划 v1）：IDEA-0009：编号相同时，第 0 层借看能否加在先剪后送之上（开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {v['accuracy']:.2f}" for k, v in res["table"][bench].items()),
              "", "计算量（相对 DivPrune-64，按结构算）：" + "；".join(f"{k} {v:.4f}" for k, v in res["flops"][bench]["relative_to_div64"].items())]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "每题秒数：" + "，".join(f"{k} {v:.4f}" for k, v in res["seconds_per_answer"].items()), "", "## 按计划写定的读法", ""]
    for q, (x, y) in (("Q1", ("peekc", "div64")), ("Q2", ("peekc", "nopeekc"))):
        c = res["criteria"][q]
        L.append(f"- **{q}**（{x} − {y}）：" + "，".join(f"{bench} {fmt(P[f'{bench} {x} - {y}'])}" for bench in BENCHES)
                 + f"，两集平均 {c['mean']:+.2f} → {c['reading']}" + ("；提升大" if c["large"] else ""))
    open(f"{DST}/0038.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
