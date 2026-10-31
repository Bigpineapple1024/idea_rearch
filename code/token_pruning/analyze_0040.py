"""EXP-0040 (plan v1, IDEA-0010, development sets): at about the FLOPs of the base with 64 tokens, keep 127 tokens before
the LLM and freeze them (ShortV: keys and values only, no update) in ShortV's 19 layers for LLaVA-1.5-7B.

  python analyze_0040.py          # writes code/output/EXP-0040/analysis/{0040.md,0040.json}
  python analyze_0040.py --dry    # the same code on stand-in runs (and the GPU smoke test's logs) -> tmp; not results

Runs (VQAv2 dev2 / GQA val400), BASE = DivPrune or CDPruner (set before any EXP-0040 run, plan v1): f127 R01 / R02
(127 kept, frozen in layers 0, 2, 3, 12, 17-31), b127 R03 / R04 (127 kept, not frozen), f64 R05 / R06 (64 kept, frozen).
Reused: b64 (the base with 64), unpruned (EXP-0033-R03 / R04).
Q1 (about equal FLOPs): f127 - b64. 95% interval above 0 on both sets -> "promising", on one -> "partly", neither but the
two-set mean > 0 -> "positive, not detected", else "unlike"; two-set mean >= +1.5 noted "large".
Q2 (the cost of freezing at 127 tokens): f127 - b127; two-set mean >= -0.5 -> "cheap", else "costly".
Described: b127 - b64 (127 against 64, both unfrozen), f64 - b64 (freezing at 64), unpruned - f127; by answer type, GQA
structural and semantic type, routed or not; GQA position questions (detailed types starting with 'position'); yes/no
questions by ground truth: the share answered yes; the first answer token's yes / no log-odds (f127, b127, f64 log them);
FLOPs per prompt from the logged prompt lengths (a frozen image token costs its K and V projections, the other tokens'
attention to it is counted); the image tokens' KV cache; seconds per answer (the implementation computes frozen layers
fully and restores the image rows, so no speed-up is expected here).
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring).
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the
score file and the registered headline; the base's selection log has one record per question with 127 (or 64) kept and
the first 64 of the 127-token order equal the 64-token run's order (>= 99%); the freeze log lists ShortV's 19 layers and
has one record per question with the image after the 35-token system prompt, 127 (or 64) image tokens and 19 restorations.
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
BASE = "cdpruner"                    # plan v1: EXP-0039's Q1 "stronger" (DEC-0076), set before any EXP-0040 run
BENCHES = ("vqav2_dev2", "gqa_val400")
SHORTV = (0, 2, 3, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
SYS_LEN, N_LAYERS, N_BOOT, LARGE, CHEAP = 35, 32, 1000, 1.5, -0.5
D, M_FF = 4096, 11008                # LLaMA-7B: hidden size, MLP size
FULL, KV = 4 * D * D + 3 * D * M_FF, 2 * D * D   # multiply-adds per token and layer: all projections + MLP; keys and values only
SUFFIX = "\nAnswer the question using a single word or phrase."
B64 = {"divprune": ("EXP-0032", ("R01", "R02"), "divprune64", "div.json"), "cdpruner": ("EXP-0039", ("R01", "R02"), "cdpruner64", "cdp.json")}[BASE]
LOGF = B64[3]
if DRY:
    DST = "D:/6CJNM007/tmp/exp0040/dry_out"
    NEW = {"f127": ("EXP-0038", ("R05", "R06"), "divprune61"), "b127": ("EXP-0038", ("R03", "R04"), "fastv61k1_div_nopeekc_kll"),
           "f64": ("EXP-0034", ("R03", "R04"), "fastv61k1_div_nopeek_kll")}
else:
    DST = f"{OUT}/EXP-0040/analysis"
    NEW = {"f127": ("EXP-0040", ("R01", "R02"), f"{BASE}127_f19_logits"), "b127": ("EXP-0040", ("R03", "R04"), f"{BASE}127_logits"),
           "f64": ("EXP-0040", ("R05", "R06"), f"{BASE}64_f19_logits")}
OLD = {"b64": B64[:3], "vanilla": ("EXP-0033", ("R03", "R04"), "vanilla")}
RUNS = {**NEW, **OLD}
N_KEPT = {"f127": 127, "b127": 127, "f64": 64, "b64": 64}
FROZEN = {"f127": SHORTV, "f64": SHORTV, "b127": (), "b64": ()}
PAIRS = (("f127", "b64"), ("f127", "b127"), ("b127", "b64"), ("f64", "b64"), ("vanilla", "f127"))
CHECKS = {}


def macs(n_other, n_img, frozen):
    """Prefill multiply-adds of the LLM for one prompt (n_other non-image tokens, the image after the system prompt),
    counting attention pairs causally; in a frozen layer the image tokens are keys and values only and only the other
    tokens are queries."""
    n, total = n_other + n_img, 0
    q_pos = list(range(SYS_LEN)) + list(range(SYS_LEN + n_img, n))
    pairs_frozen = sum(p + 1 for p in q_pos)
    for layer in range(N_LAYERS):
        if layer in frozen:
            total += n_other * FULL + n_img * KV + pairs_frozen * 2 * D
        else:
            total += n * FULL + n * (n + 1) // 2 * 2 * D
    return total


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def check_logs(bench, d_of, n, smoke=None):
    """The selection and freeze logs (or, in the dry run, the smoke test's)."""
    if smoke is not None:
        sel = {m: json.load(open(f"{smoke}/{m}/{LOGF}", encoding="utf-8")) for m in ("f127", "f64")}
        frz = {m: json.load(open(f"{smoke}/{m}/freeze.json", encoding="utf-8")) for m in ("f127", "f64")}
        n = sel["f127"]["n_images"] if "n_images" in sel["f127"] else len(sel["f127"]["records"])
        b64 = None
    else:
        sel = {m: json.load(open(f"{d_of[m]}/{LOGF}", encoding="utf-8")) for m in ("f127", "b127", "f64", "b64")}
        frz = {m: json.load(open(f"{d_of[m]}/freeze.json", encoding="utf-8")) for m in ("f127", "f64")}
        b64 = [r["order"] for r in sel["b64"]["records"]]
    tag = "smoke test" if smoke is not None else bench
    for m, s in sel.items():
        recs = s["records"]
        CHECKS[f"{tag} {m}: one selection record per question, {N_KEPT[m]} kept"] = float(len(recs) == n and all(len(r["kept"]) == N_KEPT[m] == len(set(r["kept"])) for r in recs))
        if b64 is not None and m != "b64":
            CHECKS[f"{tag} {m}: the first 64 of its order = the 64-token run's order (>= 99%)"] = float(np.mean([r["order"][:64] == o for r, o in zip(recs, b64)]) >= 0.99)
    for m, f in frz.items():
        CHECKS[f"{tag} {m}: the freeze log lists ShortV's 19 layers, one record per question, image after {SYS_LEN} tokens, "
               f"{N_KEPT[m]} image tokens, 19 restorations"] = float(
            f["layers"] == list(SHORTV) and f["n_prefills"] == n == len(f["prefills"])
            and all(r["start"] == SYS_LEN and r["n_img"] == N_KEPT[m] and r["restored"] == len(SHORTV) for r in f["prefills"]))
    return {m: [r["len"] - r["n_img"] for r in f["prefills"]] for m, f in frz.items()}


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
    if DRY:
        check_logs(None, None, None, smoke="D:/6CJNM007/tmp/exp0040/smoke_gpu")
    res = {"pairs": {}, "table": {}, "position": {}, "yes_no": {}, "log_odds": {}, "flops": {}, "kv": {}, "seconds_per_answer": {}, "peak_gib": {}}
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
        if not DRY:
            others = check_logs(bench, d_of, n)
            n_other = others["f127"]
            CHECKS[f"{bench}: f127 and f64 see the same prompts"] = float(others["f127"] == others["f64"])
            base = np.array([macs(o, 64, ()) for o in n_other], dtype=float)
            res["flops"][bench] = {"mean_non_image_tokens": float(np.mean(n_other)),
                                   "relative_to_b64": {m: float(np.mean([macs(o, N_KEPT[m], FROZEN[m]) for o in n_other] / base)) for m in ("f127", "b127", "f64", "b64")}
                                   | {"vanilla": float(np.mean([macs(o, 576, ()) for o in n_other] / base))}}
            res["kv"][bench] = {m: N_KEPT[m] / 64 for m in ("f127", "b127", "f64", "b64")} | {"vanilla": 576 / 64}
            lo = {}
            for m in ("f127", "b127", "f64"):
                x = json.load(open(f"{d_of[m]}/answer_logits.json", encoding="utf-8"))
                CHECKS[f"{bench} {m}: one first-token record per question"] = float(x["n_prefills"] == n == len(x["first_token"]))
                lp = np.array([r["logprobs"] for r in x["first_token"]])
                lo[m] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
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
                                            "f127 - b64": paired("f127", "b64", gt == g), "f127 - b127": paired("f127", "b127", gt == g)} for g in ("yes", "no")}
        if not DRY:
            res["log_odds"][bench] = {f"gt {g}": {"mean": {m: float(v[gt == g].mean()) for m, v in lo.items()},
                                                  "f127 - b127": paired("f127", "b127", gt == g, lo), "f64 - b127": paired("f64", "b127", gt == g, lo)} for g in ("yes", "no")}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]

    def reading(x, y):
        v = [P[f"{bench} {x} - {y}"] for bench in BENCHES]
        mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
        return {"reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
                "mean": mean, "large": mean >= LARGE}
    q2 = float(np.mean([P[f"{bench} f127 - b127"][0] for bench in BENCHES]))
    res["criteria"] = {"Q1": reading("f127", "b64"), "Q2": {"reading": "cheap" if q2 >= CHEAP else "costly", "mean": q2}}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0040.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    fmt3 = lambda x: "—" if x is None else f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0040（计划 v1）：IDEA-0010：多留、冻住（{BASE}，开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {x['accuracy']:.2f}" for k, x in res["table"][bench].items())]
        if bench in res["flops"]:
            L += ["", "计算量（相对 b64，按结构算，逐题平均）：" + "；".join(f"{k} {v:.4f}" for k, v in res["flops"][bench]["relative_to_b64"].items())
                  + f"（非图像 token 平均 {res['flops'][bench]['mean_non_image_tokens']:.1f} 个）",
                  "图像 token 的 KV 缓存（相对 b64）：" + "；".join(f"{k} {v:.3f}" for k, v in res["kv"][bench].items())]
        L += ["", "每题秒数：" + "；".join(f"{m} {res['seconds_per_answer'][f'{bench} {m}']:.4f}" for m in RUNS),
              "峰值显存（GiB）：" + "；".join(f"{m} {res['peak_gib'][f'{bench} {m}']:.2f}" for m in RUNS)]
        if bench in res["position"]:
            L += ["", f"位置题：{res['position'][bench]['n']} 题，" + "、".join(f"{t} {c}" for t, c in sorted(res["position"][bench]["types"].items()))]
        L += ["", "是非题（gt = 标准答案）：", ""]
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {x['said_yes'][m]:.3f}" for m in RUNS)
                     + "；准确率：" + "；".join(f"{m} {x['accuracy'][m]:.2f}" for m in RUNS)
                     + f"；f127 − b64 {fmt(x['f127 - b64'])}；f127 − b127 {fmt(x['f127 - b127'])}")
        if bench in res["log_odds"]:
            L += ["", "第一个答案 token 的对数几率 log(P(Yes)+P(yes)) − log(P(No)+P(no))：", ""]
            for g, x in res["log_odds"][bench].items():
                L.append(f"- {g}：" + "；".join(f"{m} {v:+.3f}" for m, v in x["mean"].items()) + f"；f127 − b127 {fmt3(x['f127 - b127'])}；f64 − b127 {fmt3(x['f64 - b127'])}")
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    c1, c2 = res["criteria"]["Q1"], res["criteria"]["Q2"]
    L += ["", "## 按计划写定的读法", "",
          "- **Q1**（f127 − b64，FLOPs 约相等）：" + "，".join(f"{bench} {fmt(P[f'{bench} f127 - b64'])}" for bench in BENCHES)
          + f"，两集平均 {c1['mean']:+.2f} → {c1['reading']}" + ("；提升大" if c1["large"] else ""),
          "- **Q2**（f127 − b127，127 个时冻住的代价）：" + "，".join(f"{bench} {fmt(P[f'{bench} f127 - b127'])}" for bench in BENCHES)
          + f"，两集平均 {c2['mean']:+.2f} → {c2['reading']}"]
    open(f"{DST}/0040.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-2:]))


if __name__ == "__main__":
    main()
