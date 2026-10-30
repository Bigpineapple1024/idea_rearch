"""EXP-0034 (plan v1): IDEA-0008's development runs on VQAv2 dev2 and GQA val400: K = 1 with DivPrune's selection of 61
made before layer 0, and what layer 0 lets see the 515 patches to be dropped.

  python analyze_0034.py      # writes code/output/EXP-0034/analysis/{0034.md,0034.json}

peek (R01, R02): layer 0 lets every token see all 576 patches (as K = 1); nopeek (R03, R04): no token sees the dropped
ones (the kept patches and the text see what DivPrune would hand them, at their original positions); peektext (R05, R06):
only the text after the image sees them. In all three decoding never sees the dropped patches, and from layer 1 on 61
patches stay. If layer 0 computed only the keys and values of the dropped patches, peek and peektext would cost about the
FLOPs of 64 tokens in every layer (computed here, not timed). Q1 (primary, at about equal FLOPs): peek - DivPrune-64
(EXP-0032-R01, R02), both 95% intervals above 0: promising on both sets; one: partly; neither with a positive two-set mean:
positive, not detected; otherwise: not like. Q2 (mechanism, same kept tokens and positions): peek - nopeek, read the same
way. Described: nopeek - DivPrune-64 (61 against 64 tokens and original against packed positions), peektext against
nopeek and peek (who has to see the dropped patches), peek - the K = 1 selection of 47 (EXP-0032-R03, R04), against
VisionZip-64 (EXP-0028-R01, R02) and unpruned (EXP-0033-R03, R04), by question type. Per-question scores are recomputed
from answers.jsonl and checked against each run's own score file; paired differences resample images with replacement
(1000 draws, seed 0) within each set, as in analyze_0028.
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
KEEP, N_IMG, N_BOOT = 61, 576, 1000
MODES = {"peek": "prefill", "nopeek": "none", "peektext": "text"}
NEW = {"peek": ("R01", "R02"), "nopeek": ("R03", "R04"), "peektext": ("R05", "R06")}
OLD = {"div64": ("EXP-0032", ("R01", "R02"), "divprune64"), "k1div": ("EXP-0032", ("R03", "R04"), "fastv64k1_div_kl"),
       "vz": ("EXP-0028", ("R01", "R02"), "visionzip64"), "vanilla": ("EXP-0033", ("R03", "R04"), "vanilla")}
NAMES = {"peek": "K = 1、留 61、第 0 层都看（peek）", "nopeek": "同样的 61 个、第 0 层看不到被剪的（nopeek）",
         "peektext": "同样的 61 个、只有题目看被剪的（peektext）", "div64": "DivPrune-64", "k1div": "K = 1、留 47（EXP-0032）",
         "vz": "VisionZip-64", "vanilla": "不剪枝"}
D, M_FF = 4096, 11008                        # LLaMA-7B (LLaVA-1.5-7B's LLM): hidden size, MLP size; 32 layers
FULL, KV = 4 * D * D + 3 * D * M_FF, 2 * D * D   # multiply-adds per token and layer: all projections + MLP; keys and values only


def run_table():
    r = {}
    for meth, (a, b) in NEW.items():
        r[("vqav2_dev2", meth)], r[("gqa_val400", meth)] = ("EXP-0034", a, f"fastv61k1_div_{meth}_kll"), ("EXP-0034", b, f"fastv61k1_div_{meth}_kll")
    for meth, (exp, (a, b), cfg) in OLD.items():
        r[("vqav2_dev2", meth)], r[("gqa_val400", meth)] = (exp, a, cfg), (exp, b, cfg)
    rows = {}
    for exp in ("EXP-0028", "EXP-0032", "EXP-0033", "EXP-0034"):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    for (bench, meth), (exp, rid, cfg) in r.items():
        x = rows.get(f"{exp}-{rid}", {})
        if x.get("status") != "completed":
            sys.exit(f"{exp}-{rid} ({meth}, {bench}) is not completed")
        if x.get("config") != cfg or x.get("bench") != bench:
            sys.exit(f"{exp}-{rid} has {x.get('config')} x {x.get('bench')}, planned {cfg} x {bench}")
    return r, rows


def macs(n_other, kept, mode):
    """Prefill multiply-adds of the LLM for one prompt with n_other non-image tokens (system prompt and text; the image
    sits after the first SYS tokens, which only the causal sums need), counting the attention pairs causally. mode:
    'div64' (64 image tokens before the LLM), 'vanilla' (576), 'peek_kv' (layer 0: the 61 kept and the text fully, the 515
    dropped as keys and values only; every token attends to all earlier keys), 'nopeek' (61 tokens throughout),
    'peek_impl' (as run here: layer 0 computes all 576 fully)."""
    def layer(n_full, n_kv, pairs):
        return n_full * FULL + n_kv * KV + pairs * 2 * D
    causal = lambda n: n * (n + 1) // 2   # noqa: E731
    if mode in ("div64", "vanilla", "nopeek"):
        n = n_other + {"div64": 64, "vanilla": N_IMG, "nopeek": KEEP}[mode]
        return 32 * layer(n, 0, causal(n))
    n_small = n_other + KEEP
    later = 31 * layer(n_small, 0, causal(n_small))
    n_all = n_other + N_IMG
    if mode == "peek_impl":
        return layer(n_all, 0, causal(n_all)) + later
    # peek_kv: queries are the system prompt, the kept patches and the text, at their original positions in the full prompt
    sys_len = 35
    q_pos = list(range(sys_len)) + [sys_len + p for p in kept] + list(range(sys_len + N_IMG, n_all))
    return layer(n_small, N_IMG - KEEP, sum(p + 1 for p in q_pos)) + later


def main():
    out_dir = f"{OUT}/EXP-0034/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    runs, rows = run_table()
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "flops": {}, "logs": {}}
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, d_of = {}, {}
        for (bb, meth), (exp, rid, cfg) in runs.items():
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
        pre = json.load(open(f"{d_of['div64']}/div.json"))["records"]
        others = []
        for meth, mode in MODES.items():
            dl = json.load(open(f"{d_of[meth]}/div.json"))
            rec = dl["records"]
            kept = json.load(open(f"{d_of[meth]}/kept_tokens.json"))["kept"]
            lt = json.load(open(f"{d_of[meth]}/layer_tokens.json"))
            checks[f"{bench} {meth}: one selection record, one kept list and one row of layer lengths per prefill, keep 61"] = float(
                dl["n_prefills"] == n == len(rec) == len(kept) and [x["index"] for x in rec] == list(range(n)) and dl["keep"] == KEEP
                and lt["n_prefills"] == n == lt["n_complete"] and lt["n_layers"] == 32)
            checks[f"{bench} {meth}: every record carries the mode '{mode}'"] = float(np.mean([x.get("peek") == mode for x in rec]))
            checks[f"{bench} {meth}: kept = the 61 logged, 61 distinct patches"] = float(np.mean(
                [len(x["order"]) == KEEP == len(set(x["order"])) and sorted(x["order"]) == sorted(k) for x, k in zip(rec, kept)]))
            same = float(np.mean([x["order"] == y["order"][:KEEP] for x, y in zip(rec, pre)]))
            checks[f"{bench} {meth}: its selection = the first 61 of DivPrune-64's on the same features (>= 99%)"] = float(same >= 0.99)
            res["logs"][f"{bench} {meth} order = DivPrune-64's first 61"] = same
            checks[f"{bench} {meth}: layer 0 sees the whole prompt, layers 1 to 31 that minus 515 (61 patches)"] = float(np.mean(
                [len(t) == 32 and all(x == t[0] - (N_IMG - KEEP) for x in t[1:]) for t in lt["tokens_per_layer"]]))
            others.append([t[0] - N_IMG for t in lt["tokens_per_layer"]])
        checks[f"{bench}: the three runs see the same prompts (equal layer-0 lengths)"] = float(others[0] == others[1] == others[2])
        # FLOPs, from the prompts' lengths and the kept patches (not timed)
        n_other = others[0]
        kept_p = [x["order"] for x in json.load(open(f"{d_of['peek']}/div.json"))["records"]]
        tot = {m: float(np.mean([macs(o, sorted(k), m) for o, k in zip(n_other, kept_p)])) for m in ("div64", "vanilla", "nopeek", "peek_kv", "peek_impl")}
        res["flops"][bench] = {"mean_non_image_tokens": float(np.mean(n_other)), "macs": tot,
                               "relative_to_div64": {m: tot[m] / tot["div64"] for m in tot},
                               "token_layers_per_layer": {"div64": 64.0, "nopeek": float(KEEP), "peek_as_run": (N_IMG + 31 * KEEP) / 32,
                                                          "peek_kv_only": (32 * KEEP + (N_IMG - KEEP) * KV / FULL) / 32}}
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

        pairs = [("peek", "div64"), ("peek", "nopeek"), ("nopeek", "div64"), ("peektext", "nopeek"), ("peek", "peektext"),
                 ("peektext", "div64"), ("peek", "k1div"), ("peek", "vz"), ("vanilla", "peek"), ("vanilla", "div64")]
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

    def read(n_above, mean, yes2, yes1, pos, no):
        return yes2 if n_above == 2 else yes1 if n_above == 1 else pos if mean > 0 else no

    q1, m1, n1 = two("peek", "div64")
    q2, m2, n2 = two("peek", "nopeek")
    r1 = read(n1, m1, "有希望：两个集上都高于 DivPrune-64（区间都在 0 以上）", "部分：只有一个集的区间在 0 以上",
              "方向为正，但两个集的区间都含 0", "不像：两集平均 ≤ 0")
    r2 = read(n2, m2, "第 0 层看过被剪的 patch 有作用（两个集）", "第 0 层看过被剪的 patch 部分有作用（一个集）",
              "方向为正，但两个集的区间都含 0", "没有作用（两集平均 ≤ 0）")
    go = n1 == 2 or (n1 == 1 and m1 >= 0.5)
    res["criteria"] = {"Q1_pairs": q1, "Q1_mean": m1, "Q1_intervals_above_0": n1, "Q1_reading": r1, "Q1_large": m1 >= 1.5,
                       "Q2_pairs": q2, "Q2_mean": m2, "Q2_intervals_above_0": n2, "Q2_reading": r2,
                       "confirmation_next": go}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0034.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0034（计划 v1）：IDEA-0008 的开发运行（VQAv2 dev2、GQA val400）：K = 1、按 DivPrune 的选择留 61 个，第 0 层能否看到被剪的 patch", "",
         "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    L += ["", "选择记录：" + "；".join(f"{k} {x:.4f}" for k, x in res["logs"].items())]
    for bench in BENCHES:
        f = res["flops"][bench]
        L += ["", f"## 计算量（{bench}，按 LLaMA-7B 的结构计算，不是计时；平均每题非图像 token {f['mean_non_image_tokens']:.1f} 个）", "",
              "| 配置 | 预填充乘加（相对 DivPrune-64） |", "| --- | --- |"]
        L += [f"| {k} | {v:.4f} |" for k, v in f["relative_to_div64"].items()]
        L += ["", "每层的 token 当量：" + "；".join(f"{k} {v:.2f}" for k, v in f["token_layers_per_layer"].items())]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {NAMES[m]} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "每题秒数（第 0 层按全部 576 个计算的实现）：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    L += ["", "## 按计划写定的读法", "",
          f"- **Q1**（主要，计算量约相等：peek − DivPrune-64）：VQAv2 dev2 {fmt(q1['vqav2_dev2'])}，GQA val400 {fmt(q1['gqa_val400'])}，"
          f"两集平均 {m1:+.2f} → {r1}{'；两集平均 ≥ +1.5，记为“提升大”' if m1 >= 1.5 else ''}",
          f"- **Q2**（机制，同样的 61 个与位置：peek − nopeek）：VQAv2 dev2 {fmt(q2['vqav2_dev2'])}，GQA val400 {fmt(q2['gqa_val400'])}，"
          f"两集平均 {m2:+.2f} → {r2}",
          f"- **下一步**：{'写新数据上的确认计划（先抽新的数据、提交计划，再运行）' if go else '不进入确认；记录结果，交 Mode C 与用户'}"]
    open(f"{out_dir}/0034.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
