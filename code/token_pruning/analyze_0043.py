"""EXP-0043 (plan v1, IDEA-0011's generalisation, development sets): the virtual fill (vfill.py) on another base, DivPrune-64,
and on CDPruner at budgets 128 and 192.

  python analyze_0043.py          # writes code/output/EXP-0043/analysis/{0043.md,0043.json}
  python analyze_0043.py --dry    # the same code on stand-in runs (and the GPU smoke test's logs) -> tmp; not results

Runs (VQAv2 dev2 / GQA val400): DivPrune-64 + fill R01 / R02; CDPruner-128 R03 / R04 and + fill R05 / R06; CDPruner-192
R07 / R08 and + fill R09 / R10. Reused: DivPrune-64 (EXP-0032-R01 / R02), unpruned (EXP-0033-R03 / R04); described beside
them: CDPruner-64 (EXP-0039-R01 / R02) and its fill (EXP-0041-R01 / R02).
Q1 (another base): div64 fill - div64. Q2 (budget 128): cdp128 fill - cdp128. Q3 (budget 192): cdp192 fill - cdp192.
   Each: 95% interval above 0 on both sets -> "promising", on one -> "partly", neither but the two-set mean > 0 ->
   "positive, not detected", else "unlike"; two-set mean >= +1.5 noted "large".
Described: unpruned - fill; by answer type, GQA structural and semantic type, routed or not; GQA position questions
(detailed types starting with 'position'); yes/no questions by ground truth (share answered yes); agreement with the
unpruned run's answer text (stripped, lower case, final full stop removed); the first answer token's yes / no log-odds
(the fills and CDPruner-128 / -192 log them): the AUC of gt yes against gt no on the yes/no questions; the assignment;
FLOPs per prompt (analyze_0041's count with n kept); the image part of the KV cache as run (576 / n for a fill); seconds
per answer and peak memory; the same for CDPruner-64 and its fill (64 beside 128 and 192).
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring).
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the
score file and the registered headline; the selection log has one record per question with n kept; each fill keeps
the same patches as its base (>= 99%); the layout log has mode fill and one record per question with n image tokens after
the 35-token system prompt, 576 slots, the expected positions, the layout in all 32 layers and the kept slots mapped to
themselves; a fill and its base were given the same prompts (answers.jsonl); one first-token record per question in each
run that logs them.
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
SYS_LEN, N_LAYERS, N_SLOTS, N_BOOT, LARGE = 35, 32, 576, 1000, 1.5
D, M_FF = 4096, 11008                # LLaMA-7B: hidden size, MLP size
FULL = 4 * D * D + 3 * D * M_FF      # multiply-adds per token and layer: all projections + MLP
SMOKE = "D:/6CJNM007/tmp/exp0043/smoke_gpu"
# setting -> (n kept, selection log, base run, fill run); a run is (experiment, (dev2 run, val400 run), config, logs logits)
SETTINGS = {
    "div64": (64, "div.json", ("EXP-0032", ("R01", "R02"), "divprune64", False), ("EXP-0043", ("R01", "R02"), "divprune64_fill_logits", True)),
    "cdp128": (128, "cdp.json", ("EXP-0043", ("R03", "R04"), "cdpruner128_logits", True), ("EXP-0043", ("R05", "R06"), "cdpruner128_fill_logits", True)),
    "cdp192": (192, "cdp.json", ("EXP-0043", ("R07", "R08"), "cdpruner192_logits", True), ("EXP-0043", ("R09", "R10"), "cdpruner192_fill_logits", True)),
    "cdp64": (64, "cdp.json", ("EXP-0039", ("R01", "R02"), "cdpruner64", False), ("EXP-0041", ("R01", "R02"), "cdpruner64_fill_logits", True)),
}
QUESTIONS = {"Q1": "div64", "Q2": "cdp128", "Q3": "cdp192"}
if DRY:
    DST = "D:/6CJNM007/tmp/exp0043/dry_out"
    STAND_IN = {"div64": (("EXP-0032", ("R01", "R02"), "divprune64", False), ("EXP-0041", ("R03", "R04"), "cdpruner64_shuffle_logits", True)),
                "cdp128": (("EXP-0040", ("R03", "R04"), "cdpruner127_logits", True), ("EXP-0041", ("R05", "R06"), "cdpruner64_orig_logits", True)),
                "cdp192": (("EXP-0040", ("R05", "R06"), "cdpruner64_f19_logits", True), ("EXP-0041", ("R07", "R08"), "cdpruner64_mass_logits", True))}
    SETTINGS = {s: (v[0], v[1], *STAND_IN[s]) if s in STAND_IN else v for s, v in SETTINGS.items()}
    SMOKE_CFG = {"div64 base": None, "div64 fill": "divprune64_fill", "cdp128 base": "cdpruner128", "cdp128 fill": "cdpruner128_fill",
                 "cdp192 base": "cdpruner192", "cdp192 fill": "cdpruner192_fill"}
else:
    DST = f"{OUT}/EXP-0043/analysis"
RUNS = {f"{s} {k}": spec for s, (_, _, base, fill) in SETTINGS.items() for k, spec in (("base", base), ("fill", fill))}
RUNS["vanilla"] = ("EXP-0033", ("R03", "R04"), "vanilla", False)
CHECKS = {}


def macs(n_other, kept, fill, vanilla=False):
    """Prefill multiply-adds of the LLM for one prompt (analyze_0041.macs with n kept): attention pairs counted causally
    (QK and AV, 2 D per pair); in a fill a kept image token at patch p sees the slots 0..p and the text sees all 576, plus a
    rotation of the 576 slot keys per layer."""
    n_img = N_SLOTS if vanilla else len(kept)
    n = n_other + n_img
    if fill:
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


def norm(text):
    t = text.strip().lower()
    return t[:-1] if t.endswith(".") else t


def auc(pos, neg):
    """Probability that a gt-yes question has the higher log-odds than a gt-no one (ties count half)."""
    v = np.concatenate([pos, neg])
    order = v.argsort(kind="mergesort")
    ranks = np.empty(len(v))
    s = v[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def check_fill_logs(tag, n_keep, kept_base, sel_fill, lay, n):
    """A fill's selection and layout logs against its base's kept patches (None: not compared)."""
    recs = sel_fill["records"]
    kept = [r["kept"] for r in recs]
    CHECKS[f"{tag}: the fill's selection log has one record per question, {n_keep} kept"] = float(
        len(recs) == n and all(len(k) == n_keep == len(set(k)) for k in kept))
    if kept_base is not None:
        CHECKS[f"{tag}: the fill keeps the same patches as its base (>= 99%)"] = float(
            len(kept_base) == n and np.mean([a == b for a, b in zip(kept, kept_base)]) >= 0.99)
    pre = lay["prefills"]
    CHECKS[f"{tag}: the layout log has mode fill, one record per question: {n_keep} image tokens after {SYS_LEN}, {N_SLOTS} slots, "
           f"the expected positions, the layout in {N_LAYERS} layers"] = float(
        lay["mode"] == "fill" and lay["n_prefills"] == n == len(pre)
        and all(r["n"] == n_keep and r["slots"] == N_SLOTS and r["start"] == SYS_LEN and r["positions_ok"] and r["layers"] == N_LAYERS for r in pre))
    CHECKS[f"{tag}: the kept slots map to themselves"] = float(all(all(r["rep"][p] == i for i, p in enumerate(k)) for r, k in zip(pre, kept)))
    stats = {"sim_mean": float(np.mean([r["sim_mean"] for r in pre])), "count_max": float(np.mean([r["count_max"] for r in pre])),
             "count_ones": float(np.mean([r["count_ones"] for r in pre])), "decode_steps": float(np.mean([r["decode_steps"] for r in pre]))}
    return kept, [r["len"] - r["n"] for r in pre], stats


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
    res = {"pairs": {}, "table": {}, "position": {}, "yes_no": {}, "agreement": {}, "auc": {}, "assignment": {}, "flops": {}, "kv": {},
           "seconds_per_answer": {}, "peak_gib": {}}
    if DRY:                                           # the smoke test's logs, through the same checks
        n8 = 8
        for s in ("div64", "cdp128", "cdp192"):
            n_keep, logf = SETTINGS[s][0], SETTINGS[s][1]
            sf = json.load(open(f"{SMOKE}/{SMOKE_CFG[s + ' fill']}/{logf}", encoding="utf-8"))
            if SMOKE_CFG[s + " base"]:
                kb = [r["kept"] for r in json.load(open(f"{SMOKE}/{SMOKE_CFG[s + ' base']}/{logf}", encoding="utf-8"))["records"]]
            else:                                     # DivPrune-64: EXP-0032-R01's first 8 questions
                kb = [r["kept"] for r in json.load(open(f"{OUT}/EXP-0032/EXP-0032-R01/div.json", encoding="utf-8"))["records"][:n8]]
            lay = json.load(open(f"{SMOKE}/{SMOKE_CFG[s + ' fill']}/layout.json", encoding="utf-8"))
            kept_s, others_s, stats_s = check_fill_logs(f"smoke test {s}", n_keep, kb, sf, lay, n8)
            res["assignment"][f"smoke test {s}"] = stats_s
            res["flops"][f"smoke test {s}"] = float(np.mean([macs(o, k, True) / macs(o, k, False) for o, k in zip(others_s, kept_s)]))
    for bi, bench in enumerate(BENCHES):
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, said, txt, prompts, d_of = {}, {}, {}, {}, {}
        for meth, (exp, rids, cfg, _) in RUNS.items():
            d = d_of[meth] = f"{OUT}/{exp}/{exp}-{rids[bi]}"
            rec = {str(x["question_id"]): x for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
            a = {q: x["text"] for q, x in rec.items()}
            CHECKS[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = a28.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {str(x["question_id"]): x["score"] for x in (json.loads(l) for l in open(f"{d}/{logf}", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            CHECKS[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(rows[f"{exp}-{rids[bi]}"]["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(rows[f"{exp}-{rids[bi]}"]["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(rows[f"{exp}-{rids[bi]}"]["peak_gib"])
            sc[meth] = s
            said[meth] = np.array([yes_no(a[q]) for q in b["order"]], dtype=object)
            txt[meth] = np.array([norm(a[q]) for q in b["order"]], dtype=object)
            prompts[meth] = [rec[q].get("prompt") for q in b["order"]]
        lo = {}
        for meth, (exp, rids, cfg, logits) in RUNS.items():
            if logits:
                x = json.load(open(f"{d_of[meth]}/answer_logits.json", encoding="utf-8"))
                CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == n == len(x["first_token"]))
                lp = np.array([r["logprobs"] for r in x["first_token"]])
                lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        res["assignment"][bench], res["flops"][bench], res["kv"][bench] = {}, {}, {}
        for s, (n_keep, logf, _, _) in SETTINGS.items():
            CHECKS[f"{bench} {s}: the fill and its base were given the same prompts"] = float(prompts[f"{s} fill"] == prompts[f"{s} base"])
            if DRY:
                continue
            sb = json.load(open(f"{d_of[s + ' base']}/{logf}", encoding="utf-8"))["records"]
            CHECKS[f"{bench} {s}: the base's selection log has one record per question, {n_keep} kept"] = float(
                len(sb) == n and all(len(r["kept"]) == n_keep for r in sb))
            sf = json.load(open(f"{d_of[s + ' fill']}/{logf}", encoding="utf-8"))
            lay = json.load(open(f"{d_of[s + ' fill']}/layout.json", encoding="utf-8"))
            kept, others, stats = check_fill_logs(f"{bench} {s}", n_keep, [r["kept"] for r in sb], sf, lay, n)
            res["assignment"][bench][s] = stats
            res["flops"][bench][s] = {"fill / base": float(np.mean([macs(o, k, True) / macs(o, k, False) for o, k in zip(others, kept)])),
                                      "vanilla / base": float(np.mean([macs(o, k, False, True) / macs(o, k, False) for o, k in zip(others, kept)]))}
            res["kv"][bench][s] = {"fill / base": N_SLOTS / n_keep}
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None, values=None):
            d, k = (sc[x] - sc[y], 100.0) if values is None else (values[x] - values[y], 100.0)
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [k * d[i][m[i]].mean() for i in draws if m[i].any()]
            return [k * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]
        groups = dict(b.get("groups", {}))
        if bench.startswith("gqa"):
            groups["[position]"] = np.array([b["gts"][q]["detailed"].startswith("position") for q in b["order"]])
            gt = np.array([b["gts"][q]["answer"] if b["gts"][q]["answer"] in ("yes", "no") else None for q in b["order"]], dtype=object)
        else:
            gt = np.array([(lambda top: top if top in ("yes", "no") else None)(Counter(b["gts"][q]["answers"]).most_common(1)[0][0])
                           if b["gts"][q]["answer_type"] == "yes/no" else None for q in b["order"]], dtype=object)
        groups["gt yes"], groups["gt no"] = gt == "yes", gt == "no"
        agree = {m: (txt[m] == txt["vanilla"]).astype(float) for m in RUNS if m != "vanilla"}
        for s in SETTINGS:
            f, base = f"{s} fill", f"{s} base"
            for x, y in ((f, base), ("vanilla", f)):
                res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
                for g, m in groups.items():
                    res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
            res["agreement"][f"{bench} {s}"] = {"fill": float(agree[f].mean()), "base": float(agree[base].mean()),
                                                "fill - base": paired(f, base, None, agree)}
        res["table"][bench] = {m: {"accuracy": 100 * float(v.mean()), **{g: 100 * float(v[mm].mean()) for g, mm in groups.items() if mm.any()}}
                               for m, v in sc.items()}
        if bench.startswith("gqa"):
            pos = groups["[position]"]
            res["position"][bench] = {"n": int(pos.sum()), "types": dict(Counter(b["gts"][q]["detailed"] for q, p in zip(b["order"], pos) if p))}
        res["yes_no"][bench] = {f"gt {g}": {"n": int((gt == g).sum()), "said_yes": {m: float(np.mean(said[m][gt == g] == "yes")) for m in RUNS}}
                                for g in ("yes", "no")}
        yn = (gt == "yes") | (gt == "no")
        res["auc"][bench] = {m: auc(v[gt == "yes"], v[gt == "no"]) for m, v in lo.items()}
        res["auc"][bench]["n"] = int(yn.sum())
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]

    def reading(s):
        v = [P[f"{bench} {s} fill - {s} base"] for bench in BENCHES]
        mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
        return {"setting": s, "reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
                "mean": mean, "large": mean >= LARGE}
    res["criteria"] = {q: reading(s) for q, s in QUESTIONS.items()}
    res["gain_by_budget"] = {s: float(np.mean([P[f"{bench} {s} fill - {s} base"][0] for bench in BENCHES])) for s in ("cdp64", "cdp128", "cdp192")}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0043.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0043（计划 v1）：IDEA-0011 的推广：DivPrune-64 加 fill，CDPruner-128、-192 加与不加 fill（开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}",
         "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    if DRY:
        L += ["", "## 冒烟测试的日志", ""] + [f"- {k}：{v}" for k, v in res["assignment"].items() if k.startswith("smoke")]
        L += [f"- FLOPs（fill / 底座，{k}）：{v:.4f}" for k, v in res["flops"].items() if k.startswith("smoke")]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {x['accuracy']:.2f}" for k, x in res["table"][bench].items()),
              "每题秒数：" + "；".join(f"{m} {res['seconds_per_answer'][f'{bench} {m}']:.4f}" for m in RUNS),
              "峰值显存（GiB）：" + "；".join(f"{m} {res['peak_gib'][f'{bench} {m}']:.2f}" for m in RUNS)]
        for s, st in res["assignment"].get(bench, {}).items():
            L.append(f"- {s} 填补的分配：平均余弦 {st['sim_mean']:.3f}；最多代表的槽数（平均）{st['count_max']:.1f}；不代表剪掉的 patch 的保留 token（平均）{st['count_ones']:.1f}；"
                     f"每题解码步数 {st['decode_steps']:.2f}；FLOPs fill / 底座 {res['flops'][bench][s]['fill / base']:.4f}、不剪枝 / 底座 {res['flops'][bench][s]['vanilla / base']:.4f}；"
                     f"图像 KV 缓存 fill / 底座 {res['kv'][bench][s]['fill / base']:.3f}")
        if bench in res["position"]:
            L.append(f"位置题：{res['position'][bench]['n']} 题，" + "、".join(f"{t} {c}" for t, c in sorted(res["position"][bench]["types"].items())))
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {v:.3f}" for m, v in x["said_yes"].items()))
        L.append(f"- 是非题（{res['auc'][bench]['n']} 题）上第一个答案 token 的对数几率的 AUC：" + "；".join(f"{m} {v:.4f}" for m, v in res["auc"][bench].items() if m != "n"))
        for s in SETTINGS:
            x = res["agreement"][f"{bench} {s}"]
            L.append(f"- {s} 与不剪枝的回答相同的比例：fill {x['fill']:.3f}；底座 {x['base']:.3f}；fill − 底座 {fmt(x['fill - base'])}")
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    c = res["criteria"]
    names = {"Q1": "换底座：DivPrune-64", "Q2": "预算 128：CDPruner-128", "Q3": "预算 192：CDPruner-192"}
    L += ["", "## 按计划写定的读法", ""]
    for q, what in names.items():
        s = QUESTIONS[q]
        L.append(f"- **{q}**（{what}，fill − 底座）：" + "，".join(f"{bench} {fmt(P[f'{bench} {s} fill - {s} base'])}" for bench in BENCHES)
                 + f"，两集平均 {c[q]['mean']:+.2f} → {c[q]['reading']}" + ("；提升大" if c[q]["large"] else ""))
    L.append("- 增益随预算（描述，两集平均）：" + "；".join(f"{s} {v:+.2f}" for s, v in res["gain_by_budget"].items()))
    open(f"{DST}/0043.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
