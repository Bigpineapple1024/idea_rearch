"""EXP-0041 (plan v1, IDEA-0011, development sets): CDPruner-64 with the pruned image given back as a layout (vfill.py).

  python analyze_0041.py          # writes code/output/EXP-0041/analysis/{0041.md,0041.json}
  python analyze_0041.py --dry    # the same code on stand-in runs (and the GPU smoke test's logs) -> tmp; not results

Runs (VQAv2 dev2 / GQA val400): fill R01 / R02 (every pruned slot holds its most similar kept token's key and value at its
own position, in every layer), shuffle R03 / R04 (the same with the pruned slots' representatives permuted), orig R05 /
R06 (the kept tokens at their original positions, gaps left), mass R07 / R08 (contiguous, each kept token's attention
logit + log of the slots it represents). Reused: b64 = CDPruner-64 (EXP-0039-R01 / R02), unpruned (EXP-0033-R03 / R04).
Q1 (about equal FLOPs): fill - b64. 95% interval above 0 on both sets -> "promising", on one -> "partly", neither but the
two-set mean > 0 -> "positive, not detected", else "unlike"; two-set mean >= +1.5 noted "large".
Q2 (does the content of the copies matter): fill - shuffle; Q3 (does filling the gaps matter beyond the original
positions): fill - orig; both read on Q1's scale.
Described: orig - b64, mass - b64, shuffle - b64, unpruned - fill; by answer type, GQA structural and semantic type,
routed or not; GQA position questions (detailed types starting with 'position'); yes/no questions by ground truth: the
share answered yes; the first answer token's yes / no log-odds (the four new runs log them); the assignment (mean cosine
of the pruned slots to their representative, the largest count, the kept tokens that represent no pruned slot); FLOPs per
prompt from the logged prompt lengths and kept patches (the slots add attention pairs, causal by position, and a rotation);
the image part of the KV cache as run; seconds per answer and peak memory.
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring).
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the
score file and the registered headline; the selection log has one record per question with 64 kept, the same kept
patches as CDPruner-64's (>= 99%); the layout log has the run's mode and one record per question with 64 image tokens
after the 35-token system prompt, 576 slots, the expected positions, the layout used in all 32 layers (none in orig), the
kept slots mapped to themselves, the same assignment in fill, orig and mass (>= 99%), and shuffle's a permutation of
fill's over the pruned slots (every question) that differs from it (>= 99%).
"""
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
DRY = "--dry" in sys.argv
BENCHES = ("vqav2_dev2", "gqa_val400")
SYS_LEN, N_LAYERS, N_SLOTS, KEEP, N_BOOT, LARGE = 35, 32, 576, 64, 1000, 1.5
D, M_FF = 4096, 11008                # LLaMA-7B: hidden size, MLP size
FULL = 4 * D * D + 3 * D * M_FF      # multiply-adds per token and layer: all projections + MLP
MODES = ("fill", "shuffle", "orig", "mass")
SMOKE = "D:/6CJNM007/tmp/exp0041/smoke_gpu"
if DRY:
    DST = "D:/6CJNM007/tmp/exp0041/dry_out"
    NEW = {"fill": ("EXP-0040", ("R05", "R06"), "cdpruner64_f19_logits"), "shuffle": ("EXP-0040", ("R01", "R02"), "cdpruner127_f19_logits"),
           "orig": ("EXP-0040", ("R03", "R04"), "cdpruner127_logits"), "mass": ("EXP-0038", ("R03", "R04"), "fastv61k1_div_nopeekc_kll")}
else:
    DST = f"{OUT}/EXP-0041/analysis"
    NEW = {m: ("EXP-0041", (f"R0{2 * i + 1}", f"R0{2 * i + 2}"), f"cdpruner64_{m}_logits") for i, m in enumerate(MODES)}
OLD = {"b64": ("EXP-0039", ("R01", "R02"), "cdpruner64"), "vanilla": ("EXP-0033", ("R03", "R04"), "vanilla")}
RUNS = {**NEW, **OLD}
PAIRS = (("fill", "b64"), ("fill", "shuffle"), ("fill", "orig"), ("orig", "b64"), ("mass", "b64"), ("shuffle", "b64"), ("vanilla", "fill"))
LO_PAIRS = (("fill", "orig"), ("fill", "shuffle"), ("fill", "mass"))
CHECKS = {}


def macs(n_other, kept, mode):
    """Prefill multiply-adds of the LLM for one prompt (n_other non-image tokens, the image after the system prompt, the
    kept patch indices `kept`), counting attention pairs causally (QK and AV, 2 D per pair); in fill and shuffle a kept
    image token at patch p sees the slots 0..p and the text sees all 576, plus a rotation of the 576 slot keys per layer."""
    n_img = len(kept) if mode != "vanilla" else N_SLOTS
    n = n_other + n_img
    if mode in ("fill", "shuffle"):
        n_text = n_other - SYS_LEN
        pairs = SYS_LEN * (SYS_LEN + 1) // 2 + sum(SYS_LEN + p + 1 for p in kept) + sum(SYS_LEN + N_SLOTS + t + 1 for t in range(n_text))
        extra = 2 * N_SLOTS * D
    else:
        pairs, extra = n * (n + 1) // 2, 0
    return N_LAYERS * (n * FULL + pairs * 2 * D + extra)


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def check_logs(tag, sel, lay, b64_kept, n):
    """The selection and layout logs of the four new runs (in the dry run, the smoke test's)."""
    kept_of = {}
    for m in MODES:
        recs = sel[m]["records"]
        kept_of[m] = [r["kept"] for r in recs]
        CHECKS[f"{tag} {m}: one selection record per question, {KEEP} kept"] = float(
            len(recs) == n and all(len(r["kept"]) == KEEP == len(set(r["kept"])) for r in recs))
        if b64_kept is not None:
            CHECKS[f"{tag} {m}: the same kept patches as CDPruner-64 (>= 99%)"] = float(np.mean([a == b for a, b in zip(kept_of[m], b64_kept)]) >= 0.99)
        lg = lay[m]
        pre = lg["prefills"]
        want_layers = 0 if m == "orig" else N_LAYERS
        CHECKS[f"{tag} {m}: the layout log has mode {m}, one record per question: {KEEP} image tokens after {SYS_LEN}, {N_SLOTS} slots, "
               f"the expected positions, the layout in {want_layers} layers"] = float(
            lg["mode"] == m and lg["n_prefills"] == n == len(pre)
            and all(r["n"] == KEEP and r["slots"] == N_SLOTS and r["start"] == SYS_LEN and r["positions_ok"] and r["layers"] == want_layers for r in pre))
        CHECKS[f"{tag} {m}: the kept slots map to themselves"] = float(all(
            all(r["rep"][p] == i for i, p in enumerate(k)) for r, k in zip(pre, kept_of[m])))
    rep = {m: [r["rep"] for r in lay[m]["prefills"]] for m in MODES}
    for m in ("orig", "mass"):
        CHECKS[f"{tag} {m}: the same assignment as fill (>= 99%)"] = float(np.mean([a == b for a, b in zip(rep[m], rep["fill"])]) >= 0.99)
    perm, differs = [], []
    for a, b, k in zip(rep["shuffle"], rep["fill"], kept_of["fill"]):
        ks = set(k)
        pa = sorted(x for j, x in enumerate(a) if j not in ks)
        pb = sorted(x for j, x in enumerate(b) if j not in ks)
        perm.append(pa == pb)
        differs.append(a != b)
    CHECKS[f"{tag} shuffle: a permutation of fill's assignment over the pruned slots (every question)"] = float(all(perm))
    CHECKS[f"{tag} shuffle: differs from fill's (>= 99%)"] = float(np.mean(differs) >= 0.99)
    stats = {"sim_mean": float(np.mean([r["sim_mean"] for r in lay["fill"]["prefills"]])),
             "count_max": float(np.mean([r["count_max"] for r in lay["fill"]["prefills"]])),
             "count_ones": float(np.mean([r["count_ones"] for r in lay["fill"]["prefills"]])),
             "decode_steps": {m: float(np.mean([r["decode_steps"] for r in lay[m]["prefills"]])) for m in MODES}}
    others = {m: [r["len"] - r["n"] for r in lay[m]["prefills"]] for m in MODES}
    return kept_of, others, stats


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    rows = {}
    for exp in sorted({v[0] for v in RUNS.values()}):
        rows.update({x["run_id"]: x for x in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    for meth, (exp, rids, cfg) in RUNS.items():
        for bench, rid in zip(BENCHES, rids):
            x = rows.get(f"{exp}-{rid}", {})
            CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench} ({meth})"] = float(x.get("status") == "completed" and x.get("config") == cfg and x.get("bench") == bench)
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {[k for k, v in CHECKS.items() if v < 1]}")
    res = {"pairs": {}, "table": {}, "position": {}, "yes_no": {}, "log_odds": {}, "assignment": {}, "flops": {}, "kv": {},
           "seconds_per_answer": {}, "peak_gib": {}}
    if DRY:
        sel = {m: json.load(open(f"{SMOKE}/{m}/cdp.json", encoding="utf-8")) for m in MODES}
        lay = {m: json.load(open(f"{SMOKE}/{m}/layout.json", encoding="utf-8")) for m in MODES}
        kept_s, others_s, stats_s = check_logs("smoke test", sel, lay, None, len(sel["fill"]["records"]))
        res["assignment"]["smoke test"] = stats_s
        base = [macs(o, k, "b64") for o, k in zip(others_s["fill"], kept_s["fill"])]
        res["flops"]["smoke test"] = {m: float(np.mean([macs(o, k, m) / b0 for o, k, b0 in zip(others_s["fill"], kept_s["fill"], base)]))
                                      for m in ("fill", "shuffle", "orig", "mass", "vanilla")}
    for bi, bench in enumerate(BENCHES):
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, said, d_of = {}, {}, {}
        for meth, (exp, rids, cfg) in RUNS.items():
            d = d_of[meth] = f"{OUT}/{exp}/{exp}-{rids[bi]}"
            a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = a28.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {str(x["question_id"]): x["score"] for x in (json.loads(l) for l in open(f"{d}/{logf}", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            CHECKS[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(rows[f"{exp}-{rids[bi]}"]["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(rows[f"{exp}-{rids[bi]}"]["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(rows[f"{exp}-{rids[bi]}"]["peak_gib"])
            sc[meth] = s
            said[meth] = [yes_no(a[q]) for q in b["order"]]
        lo = {}
        for m in MODES:
            x = json.load(open(f"{d_of[m]}/answer_logits.json", encoding="utf-8"))
            CHECKS[f"{bench} {m}: one first-token record per question"] = float(x["n_prefills"] == n == len(x["first_token"]))
            lp = np.array([r["logprobs"] for r in x["first_token"]])
            lo[m] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        if not DRY:
            sel = {m: json.load(open(f"{d_of[m]}/cdp.json", encoding="utf-8")) for m in MODES}
            lay = {m: json.load(open(f"{d_of[m]}/layout.json", encoding="utf-8")) for m in MODES}
            b64_kept = [r["kept"] for r in json.load(open(f"{d_of['b64']}/cdp.json", encoding="utf-8"))["records"]]
            kept_of, others, stats = check_logs(bench, sel, lay, b64_kept, n)
            CHECKS[f"{bench}: the four runs see the same prompts"] = float(all(others[m] == others["fill"] for m in MODES))
            res["assignment"][bench] = stats
            base = [macs(o, k, "b64") for o, k in zip(others["fill"], kept_of["fill"])]
            res["flops"][bench] = {"mean_non_image_tokens": float(np.mean(others["fill"])),
                                   "relative_to_b64": {m: float(np.mean([macs(o, k, m) / b0 for o, k, b0 in zip(others["fill"], kept_of["fill"], base)]))
                                                       for m in ("fill", "shuffle", "orig", "mass", "vanilla")}}
            res["kv"][bench] = {"fill": N_SLOTS / KEEP, "shuffle": N_SLOTS / KEEP, "orig": 1.0, "mass": 1.0, "b64": 1.0, "vanilla": N_SLOTS / KEEP}
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None, values=None):
            d, k = (sc[x] - sc[y], 100.0) if values is None else (values[x] - values[y], 1.0)
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [k * d[i][m[i]].mean() for i in draws if m[i].any()]
            return [k * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]
        groups = dict(b.get("groups", {}))
        if bench.startswith("gqa"):
            groups["[position]"] = np.array([b["gts"][q]["detailed"].startswith("position") for q in b["order"]])
        for x, y in PAIRS:
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            for g, m in groups.items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        res["table"][bench] = {m: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[mm].mean()) for g, mm in groups.items() if mm.any()}} for m, s in sc.items()}
        if bench.startswith("gqa"):
            pos = groups["[position]"]
            res["position"][bench] = {"n": int(pos.sum()), "types": dict(Counter(b["gts"][q]["detailed"] for q, p in zip(b["order"], pos) if p))}
            gt = np.array([b["gts"][q]["answer"] if b["gts"][q]["answer"] in ("yes", "no") else None for q in b["order"]], dtype=object)
        else:
            gt = np.array([(lambda top: top if top in ("yes", "no") else None)(Counter(b["gts"][q]["answers"]).most_common(1)[0][0])
                           if b["gts"][q]["answer_type"] == "yes/no" else None for q in b["order"]], dtype=object)
        res["yes_no"][bench] = {f"gt {g}": {"n": int((gt == g).sum()),
                                            "said_yes": {meth: float(np.mean([s == "yes" for s, gg in zip(said[meth], gt) if gg == g])) for meth in RUNS},
                                            "accuracy": {meth: 100 * float(sc[meth][gt == g].mean()) for meth in RUNS},
                                            "fill - b64": paired("fill", "b64", gt == g), "fill - orig": paired("fill", "orig", gt == g),
                                            "orig - b64": paired("orig", "b64", gt == g)} for g in ("yes", "no")}
        res["log_odds"][bench] = {f"gt {g}": {"mean": {m: float(v[gt == g].mean()) for m, v in lo.items()},
                                              **{f"{x} - {y}": paired(x, y, gt == g, lo) for x, y in LO_PAIRS}} for g in ("yes", "no")}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]

    def reading(x, y):
        v = [P[f"{bench} {x} - {y}"] for bench in BENCHES]
        mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
        return {"reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
                "mean": mean, "large": mean >= LARGE}
    res["criteria"] = {"Q1": reading("fill", "b64"), "Q2": reading("fill", "shuffle"), "Q3": reading("fill", "orig")}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0041.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    fmt3 = lambda x: "—" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0041（计划 v1）：IDEA-0011：把剪掉的图还给 LLM 一个布局（CDPruner-64，开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    for key in (["smoke test"] if DRY else []) + list(BENCHES):
        if key in res["assignment"]:
            s = res["assignment"][key]
            L += ["", f"## {key}：填补的分配", "", f"剪掉的 patch 与其代表的平均余弦 {s['sim_mean']:.3f}；一个保留 token 最多代表的槽数（平均）{s['count_max']:.1f}；"
                  f"不代表任何剪掉的 patch 的保留 token（平均）{s['count_ones']:.1f} 个；每题解码步数（平均）" + "；".join(f"{m} {v:.2f}" for m, v in s["decode_steps"].items())]
        if key in res["flops"] and key == "smoke test":
            L += ["", "计算量（相对 b64，按结构算，冒烟测试的提示）：" + "；".join(f"{m} {v:.4f}" for m, v in res["flops"][key].items())]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {x['accuracy']:.2f}" for k, x in res["table"][bench].items())]
        if bench in res["flops"]:
            L += ["", "计算量（相对 b64，按结构算，逐题平均）：" + "；".join(f"{k} {v:.4f}" for k, v in res["flops"][bench]["relative_to_b64"].items())
                  + f"（非图像 token 平均 {res['flops'][bench]['mean_non_image_tokens']:.1f} 个）",
                  "图像部分的 KV 缓存（相对 b64，按本流程的实现：fill、shuffle 存下 576 个槽）：" + "；".join(f"{k} {v:.3f}" for k, v in res["kv"][bench].items())]
        L += ["", "每题秒数：" + "；".join(f"{m} {res['seconds_per_answer'][f'{bench} {m}']:.4f}" for m in RUNS),
              "峰值显存（GiB）：" + "；".join(f"{m} {res['peak_gib'][f'{bench} {m}']:.2f}" for m in RUNS)]
        if bench in res["position"]:
            L += ["", f"位置题：{res['position'][bench]['n']} 题，" + "、".join(f"{t} {c}" for t, c in sorted(res["position"][bench]["types"].items()))]
        L += ["", "是非题（gt = 标准答案）：", ""]
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {x['said_yes'][m]:.3f}" for m in RUNS)
                     + "；准确率：" + "；".join(f"{m} {x['accuracy'][m]:.2f}" for m in RUNS)
                     + f"；fill − b64 {fmt(x['fill - b64'])}；fill − orig {fmt(x['fill - orig'])}；orig − b64 {fmt(x['orig - b64'])}")
        L += ["", "第一个答案 token 的对数几率 log(P(Yes)+P(yes)) − log(P(No)+P(no))：", ""]
        for g, x in res["log_odds"][bench].items():
            L.append(f"- {g}：" + "；".join(f"{m} {v:+.3f}" for m, v in x["mean"].items()) + "；"
                     + "；".join(f"{a} − {c} {fmt3(x[f'{a} - {c}'])}" for a, c in LO_PAIRS))
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    c = res["criteria"]
    names = {"Q1": ("fill", "b64", "FLOPs 约相等"), "Q2": ("fill", "shuffle", "复制的内容是否要紧"), "Q3": ("fill", "orig", "填上缺口是否要紧")}
    L += ["", "## 按计划写定的读法", ""]
    for q, (x, y, what) in names.items():
        L.append(f"- **{q}**（{x} − {y}，{what}）：" + "，".join(f"{bench} {fmt(P[f'{bench} {x} - {y}'])}" for bench in BENCHES)
                 + f"，两集平均 {c[q]['mean']:+.2f} → {c[q]['reading']}" + ("；提升大" if c[q]["large"] else ""))
    open(f"{DST}/0041.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
