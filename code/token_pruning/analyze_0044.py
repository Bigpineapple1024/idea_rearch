"""EXP-0044 (plan v1, IDEA-0011, development sets): the virtual fill on VisionZip-64 (vzfill.py), and the compact cache
(vfill.py compact=True) against the plain implementation.
Plan v1.2 (2026-11-01, DEC-0091, before any run): Q1 is dropped, the VisionZip fill failing on the GPU (vzfill.py feeds
float32 states to the half-precision projector); Q1 = False below keeps R01-R04 and everything about VisionZip out, and
Q2, Q3 and the other descriptions are unchanged.

  python analyze_0044.py          # writes code/output/EXP-0044/analysis/{0044.md,0044.json}
  python analyze_0044.py --dry    # the same code on stand-in runs (and the GPU smoke test's logs) -> tmp; not results

Runs (VQAv2 dev2 / GQA val400): VisionZip-64 R01 / R02 (rerun at this code, with the first-token log-probabilities), its
fill R03 / R04; CDPruner-64's fill with the compact cache R05 / R06; DivPrune-64 R07 / R08 (plan v1.1, DEC-0086: the base
EXP-0043 reused, rerun at this code). Reused: the plain fill EXP-0041-R01 / R02, CDPruner-64 EXP-0039-R01 / R02, unpruned
EXP-0033-R03 / R04; VisionZip-64 of EXP-0028-R01 / R02 described beside R01 / R02; DivPrune-64 of EXP-0032-R01 / R02 and
its fill EXP-0043-R01 / R02 beside R07 / R08.
Q1 (another base): vz64 fill - vz64. The 95% interval above 0 on both sets -> "promising", on one -> "partly", neither
   but the two-set mean > 0 -> "positive, not detected", else "unlike"; two-set mean >= +1.5 noted "large".
Q2 (the compact cache): the compact run's answers and first-token log-probabilities against the plain fill's (EXP-0041):
   identical on every question of both sets -> "identical"; answers identical on >= 99.5% of the questions of each set ->
   "equivalent"; otherwise "not equivalent".
Q3 (plan v1.1, the DivPrune-64 rerun): R07 / R08's answers against EXP-0032-R01 / R02's, read as Q2; described beside it:
   the rerun's kept patches against EXP-0032's (every question), rerun - EXP-0032, and EXP-0043's fill - the rerun (the
   cross-check of EXP-0043's Q1).
Described: unpruned - vz64 fill; by answer type, GQA structural and semantic type, routed or not; GQA position questions;
yes/no questions by ground truth (share answered yes); agreement with the unpruned run's answer text; the AUC of the
first-token log-odds (gt yes against gt no, yes/no questions) of VisionZip-64 and its fill; VisionZip's layout (mean
cosine of the free patch slots to their representative, the largest count, tokens that represent no free slot); FLOPs
(analyze_0041's count with 577 slots); the image part of the KV cache as run (plain fill: the 64 kept tokens in the cache
and 576 slots beside it; compact: the 64); seconds per answer and peak memory of the compact run, the plain fill and
CDPruner-64; the VisionZip-64 rerun against EXP-0028's (identical answers).
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring).
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the
score file and the registered headline; the fill's layout log has mode fill and one record per question with 64 image
tokens after the 35-token system prompt, 577 slots, the expected positions, the layout in all 32 layers, slot 0 for CLS
and each token's own slot mapped to itself; the compact run's layout log is marked compact with the plain fill's records
(64 tokens, 576 slots, 32 layers); the compact run and the plain fill keep the same patches (CDPruner's logs); runs
compared were given the same prompts; one first-token record per question in each run that logs them.
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
Q1 = False      # plan v1.2 (DEC-0091): Q1 (VisionZip-64 and its fill, R01-R04) dropped before any run
BENCHES = ("vqav2_dev2", "gqa_val400")
SYS_LEN, N_LAYERS, N_PATCH, KEEP, N_BOOT, LARGE = 35, 32, 576, 64, 1000, 1.5
D, M_FF = 4096, 11008
FULL = 4 * D * D + 3 * D * M_FF
SMOKE = "D:/6CJNM007/tmp/exp0044/smoke_gpu"
# name -> (experiment, (dev2 run, val400 run), config, logs logits)
if DRY:
    DST = "D:/6CJNM007/tmp/exp0044/dry_out"
    NEW = {"vz64": ("EXP-0040", ("R03", "R04"), "cdpruner127_logits", True), "vz64 fill": ("EXP-0040", ("R05", "R06"), "cdpruner64_f19_logits", True),
           "fillc": ("EXP-0041", ("R03", "R04"), "cdpruner64_shuffle_logits", True), "div64": ("EXP-0032", ("R01", "R02"), "divprune64", False),
           "div64 fill (EXP-0043)": ("EXP-0041", ("R05", "R06"), "cdpruner64_orig_logits", True)}      # stand-ins, not EXP-0043's runs
else:
    DST = f"{OUT}/EXP-0044/analysis"
    NEW = {"vz64": ("EXP-0044", ("R01", "R02"), "visionzip64_logits", True), "vz64 fill": ("EXP-0044", ("R03", "R04"), "visionzip64_fill_logits", True),
           "fillc": ("EXP-0044", ("R05", "R06"), "cdpruner64_fillc_logits", True), "div64": ("EXP-0044", ("R07", "R08"), "divprune64", False),
           "div64 fill (EXP-0043)": ("EXP-0043", ("R01", "R02"), "divprune64_fill_logits", True)}
OLD = {"fill": ("EXP-0041", ("R01", "R02"), "cdpruner64_fill_logits", True), "b64": ("EXP-0039", ("R01", "R02"), "cdpruner64", False),
       "vanilla": ("EXP-0033", ("R03", "R04"), "vanilla", False), "vz64 (EXP-0028)": ("EXP-0028", ("R01", "R02"), "visionzip64", False),
       "div64 (EXP-0032)": ("EXP-0032", ("R01", "R02"), "divprune64", False)}
if not Q1:
    for _k in ("vz64", "vz64 fill"):
        NEW.pop(_k)
    OLD.pop("vz64 (EXP-0028)")
RUNS = {**NEW, **OLD}
CHECKS = {}


def macs(n_other, kept, n_slots, fill):
    """Prefill multiply-adds (analyze_0041.macs with n_slots slots; `kept` are the image tokens' slots)."""
    n = n_other + len(kept)
    if fill:
        n_text = n_other - SYS_LEN
        pairs = SYS_LEN * (SYS_LEN + 1) // 2 + sum(SYS_LEN + s + 1 for s in kept) + sum(SYS_LEN + n_slots + t + 1 for t in range(n_text))
        extra = 2 * n_slots * D
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


def check_vz_log(tag, lay, n):
    pre = lay["prefills"]
    CHECKS[f"{tag}: the layout log has mode fill, one record per question: {KEEP} image tokens after {SYS_LEN}, {1 + N_PATCH} slots, "
           f"the expected positions, the layout in {N_LAYERS} layers"] = float(
        lay["mode"] == "fill" and not lay.get("compact") and lay["n_prefills"] == n == len(pre)
        and all(r["n"] == KEEP and r["slots"] == 1 + N_PATCH and r["start"] == SYS_LEN and r["positions_ok"] and r["layers"] == N_LAYERS for r in pre))
    CHECKS[f"{tag}: slot 0 is CLS, 53 dominant and 10 target slots, each token's own slot maps to itself"] = float(all(
        r["kept"][0] == 0 and len(r["dominant"]) == 53 and len(r["targets"]) == 10 and r["kept"] == [0] + [1 + p for p in r["dominant"]] + [1 + t for t in r["targets"]]
        and len(set(r["kept"])) == KEEP and all(r["rep"][s] == i for i, s in enumerate(r["kept"])) for r in pre))
    free = [[s for s in range(1 + N_PATCH) if s not in set(r["kept"])] for r in pre]
    stats = {"sim_mean": float(np.mean([r["sim_mean"] for r in pre])), "count_max": float(np.mean([r["count_max"] for r in pre])),
             "count_ones": float(np.mean([r["count_ones"] for r in pre])), "decode_steps": float(np.mean([r["decode_steps"] for r in pre])),
             "cls_represents": float(np.mean([sum(r["rep"][s] == 0 for s in f) for r, f in zip(pre, free)]))}
    return [r["kept"] for r in pre], [r["len"] - r["n"] for r in pre], stats


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
           "compact": {}, "rerun": {}, "seconds_per_answer": {}, "peak_gib": {}}
    if DRY:
        if Q1:
            lay = json.load(open(f"{SMOKE}/visionzip64_fill/layout.json", encoding="utf-8"))
            kept_s, others_s, stats_s = check_vz_log("smoke test vz64 fill", lay, len(lay["prefills"]))
            res["assignment"]["smoke test"] = stats_s
            res["flops"]["smoke test"] = float(np.mean([macs(o, k, 1 + N_PATCH, True) / macs(o, k, 1 + N_PATCH, False) for o, k in zip(others_s, kept_s)]))
        lc = json.load(open(f"{SMOKE}/cdpruner64_fillc/layout.json", encoding="utf-8"))
        CHECKS["smoke test fillc: the layout log is marked compact, fill, 64 tokens, 576 slots, 32 layers"] = float(
            lc["mode"] == "fill" and lc["compact"] is True and all(r["n"] == KEEP and r["slots"] == N_PATCH and r["layers"] == N_LAYERS and r["positions_ok"] for r in lc["prefills"]))
    for bi, bench in enumerate(BENCHES):
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, said, txt, prompts, raw, d_of = {}, {}, {}, {}, {}, {}
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
            raw[meth] = [a[q] for q in b["order"]]
            said[meth] = np.array([yes_no(a[q]) for q in b["order"]], dtype=object)
            txt[meth] = np.array([norm(a[q]) for q in b["order"]], dtype=object)
            prompts[meth] = [rec[q].get("prompt") for q in b["order"]]
        lp_of, lo = {}, {}
        for meth, (exp, rids, cfg, logits) in RUNS.items():
            if logits:
                x = json.load(open(f"{d_of[meth]}/answer_logits.json", encoding="utf-8"))
                CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == n == len(x["first_token"]))
                lp = np.array([r["logprobs"] for r in x["first_token"]])
                lp_of[meth] = lp
                lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        for x, y in (("vz64 fill", "vz64"), ("fillc", "fill"), ("vz64", "vz64 (EXP-0028)"), ("fill", "b64"), ("div64", "div64 (EXP-0032)"),
                     ("div64 fill (EXP-0043)", "div64")):
            if x in RUNS and y in RUNS:
                CHECKS[f"{bench} {x} and {y} were given the same prompts"] = float(prompts[x] == prompts[y])
        kd = [[r["kept"] for r in json.load(open(f"{d_of[m]}/div.json", encoding="utf-8"))["records"]] for m in ("div64", "div64 (EXP-0032)")]
        CHECKS[f"{bench} div64: the rerun keeps EXP-0032's patches (every question)"] = float(len(kd[0]) == len(kd[1]) == n and kd[0] == kd[1])
        # Q2: the compact run against the plain fill
        res["compact"][bench] = {"answers identical": float(np.mean([p == q for p, q in zip(raw["fillc"], raw["fill"])])),
                                 "log-probabilities identical": float(np.mean(np.all(lp_of["fillc"] == lp_of["fill"], axis=1))),
                                 "largest log-probability difference": float(np.abs(lp_of["fillc"] - lp_of["fill"]).max()),
                                 "accuracy difference": 100 * float(sc["fillc"].mean() - sc["fill"].mean())}
        res["rerun"][bench] = {m: {"answers identical": float(np.mean([p == q for p, q in zip(raw[m], raw[old])])),
                                   "accuracy difference": 100 * float(sc[m].mean() - sc[old].mean())}
                               for m, old in (("vz64", "vz64 (EXP-0028)"), ("div64", "div64 (EXP-0032)")) if m in RUNS}
        if not DRY:
            if Q1:
                lay = json.load(open(f"{d_of['vz64 fill']}/layout.json", encoding="utf-8"))
                kept, others, stats = check_vz_log(f"{bench} vz64 fill", lay, n)
                res["assignment"][bench] = stats
                res["flops"][bench] = {"vz64 fill / vz64": float(np.mean([macs(o, k, 1 + N_PATCH, True) / macs(o, k, 1 + N_PATCH, False) for o, k in zip(others, kept)]))}
            lc = json.load(open(f"{d_of['fillc']}/layout.json", encoding="utf-8"))
            lf = json.load(open(f"{d_of['fill']}/layout.json", encoding="utf-8"))
            CHECKS[f"{bench} fillc: the layout log is marked compact, mode fill, one record per question: 64 tokens, 576 slots, 32 layers, the plain fill's assignment"] = float(
                lc["mode"] == "fill" and lc["compact"] is True and lc["n_prefills"] == n == len(lc["prefills"])
                and all(r["n"] == KEEP and r["slots"] == N_PATCH and r["layers"] == N_LAYERS and r["positions_ok"] for r in lc["prefills"])
                and all(r["rep"] == s["rep"] for r, s in zip(lc["prefills"], lf["prefills"])))
            kc = [r["kept"] for r in json.load(open(f"{d_of['fillc']}/cdp.json", encoding="utf-8"))["records"]]
            kf = [r["kept"] for r in json.load(open(f"{d_of['fill']}/cdp.json", encoding="utf-8"))["records"]]
            CHECKS[f"{bench} fillc: the same kept patches as the plain fill (every question)"] = float(len(kc) == len(kf) == n and kc == kf)
            res["kv"][bench] = {"plain fill / b64": (KEEP + N_PATCH) / KEEP, "compact fill / b64": 1.0,
                                **({"vz64 fill (plain) / vz64": (KEEP + 1 + N_PATCH) / KEEP} if Q1 else {})}
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None, values=None):
            dd = (sc[x] - sc[y]) if values is None else (values[x] - values[y])
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100.0 * dd[i][m[i]].mean() for i in draws if m[i].any()]
            return [100.0 * float(dd[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]
        groups = dict(b.get("groups", {}))
        if bench.startswith("gqa"):
            groups["[position]"] = np.array([b["gts"][q]["detailed"].startswith("position") for q in b["order"]])
            gt = np.array([b["gts"][q]["answer"] if b["gts"][q]["answer"] in ("yes", "no") else None for q in b["order"]], dtype=object)
        else:
            gt = np.array([(lambda top: top if top in ("yes", "no") else None)(Counter(b["gts"][q]["answers"]).most_common(1)[0][0])
                           if b["gts"][q]["answer_type"] == "yes/no" else None for q in b["order"]], dtype=object)
        groups["gt yes"], groups["gt no"] = gt == "yes", gt == "no"
        agree = {m: (txt[m] == txt["vanilla"]).astype(float) for m in RUNS if m != "vanilla"}
        for x, y in (("vz64 fill", "vz64"), ("vanilla", "vz64 fill"), ("fillc", "fill"), ("vz64", "vz64 (EXP-0028)"), ("div64", "div64 (EXP-0032)"),
                     ("div64 fill (EXP-0043)", "div64")):
            if x not in RUNS or y not in RUNS:
                continue
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            if (x, y) in (("vz64 fill", "vz64"), ("vanilla", "vz64 fill")):
                for g, m in groups.items():
                    res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        if Q1:
            res["agreement"][bench] = {"vz64 fill": float(agree["vz64 fill"].mean()), "vz64": float(agree["vz64"].mean()),
                                       "vz64 fill - vz64": paired("vz64 fill", "vz64", None, agree)}
        res["table"][bench] = {m: {"accuracy": 100 * float(v.mean()), **{g: 100 * float(v[mm].mean()) for g, mm in groups.items() if mm.any()}}
                               for m, v in sc.items()}
        if bench.startswith("gqa"):
            pos = groups["[position]"]
            res["position"][bench] = {"n": int(pos.sum()), "types": dict(Counter(b["gts"][q]["detailed"] for q, p in zip(b["order"], pos) if p))}
        res["yes_no"][bench] = {f"gt {g}": {"n": int((gt == g).sum()), "said_yes": {m: float(np.mean(said[m][gt == g] == "yes")) for m in RUNS}}
                                for g in ("yes", "no")}
        res["auc"][bench] = {m: auc(lo[m][gt == "yes"], lo[m][gt == "no"]) for m in ("vz64", "vz64 fill") if m in RUNS}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]
    if Q1:
        v = [P[f"{bench} vz64 fill - vz64"] for bench in BENCHES]
        mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
        q1 = {"reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
              "mean": mean, "large": mean >= LARGE}
    else:
        q1 = {"reading": "not run (plan v1.2, DEC-0091)"}
    cp_ = res["compact"]
    q2 = ("identical" if all(cp_[bn]["answers identical"] == 1.0 and cp_[bn]["log-probabilities identical"] == 1.0 for bn in BENCHES)
          else "equivalent" if all(cp_[bn]["answers identical"] >= 0.995 for bn in BENCHES) else "not equivalent")
    rr = res["rerun"]
    q3 = ("identical" if all(rr[bn]["div64"]["answers identical"] == 1.0 for bn in BENCHES)
          else "equivalent" if all(rr[bn]["div64"]["answers identical"] >= 0.995 for bn in BENCHES) else "not equivalent")
    res["criteria"] = {"Q1": q1, "Q2": q2, "Q3": q3}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0044.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0044（计划 v1～v1.2）：紧凑的 KV 缓存；DivPrune-64 的重跑（开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    if DRY and Q1:
        L += ["", f"冒烟测试的 VisionZip 布局：{res['assignment']['smoke test']}；FLOPs fill / 底座 {res['flops']['smoke test']:.4f}"]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {x['accuracy']:.2f}" for k, x in res["table"][bench].items()),
              "每题秒数：" + "；".join(f"{m} {res['seconds_per_answer'][f'{bench} {m}']:.4f}" for m in RUNS),
              "峰值显存（GiB）：" + "；".join(f"{m} {res['peak_gib'][f'{bench} {m}']:.2f}" for m in RUNS)]
        if bench in res["assignment"]:
            st = res["assignment"][bench]
            L.append(f"- VisionZip 的布局：空槽与代表的平均余弦 {st['sim_mean']:.3f}；最多代表的槽数（平均）{st['count_max']:.1f}；不代表空槽的 token（平均）{st['count_ones']:.1f}；"
                     f"每题解码步数 {st['decode_steps']:.2f}；FLOPs fill / 底座 {res['flops'][bench]['vz64 fill / vz64']:.4f}；图像 KV 缓存（相对底座）" + "；".join(f"{k} {x:.3f}" for k, x in res["kv"][bench].items()))
        elif bench in res["kv"]:
            L.append("- 图像 KV 缓存（相对 CDPruner-64，按本流程的实现）：" + "；".join(f"{k} {x:.3f}" for k, x in res["kv"][bench].items()))
        if bench in res["position"]:
            L.append(f"位置题：{res['position'][bench]['n']} 题")
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {y:.3f}" for m, y in x["said_yes"].items()))
        if res["auc"][bench]:
            L.append("- 是非题上第一个答案 token 的对数几率的 AUC：" + "；".join(f"{m} {y:.4f}" for m, y in res["auc"][bench].items()))
        if bench in res["agreement"]:
            x = res["agreement"][bench]
            L.append(f"- 与不剪枝的回答相同的比例：vz64 fill {x['vz64 fill']:.3f}；vz64 {x['vz64']:.3f}；差 {fmt(x['vz64 fill - vz64'])}")
        c = res["compact"][bench]
        L.append(f"- 紧凑缓存 对 普通实现（EXP-0041）：答案相同 {c['answers identical']:.4f}；对数概率相同 {c['log-probabilities identical']:.4f}；"
                 f"最大差 {c['largest log-probability difference']:.3g}；准确率差 {c['accuracy difference']:+.3f}")
        for m, old in (("vz64", "EXP-0028"), ("div64", "EXP-0032")):
            if m not in res["rerun"][bench]:
                continue
            r = res["rerun"][bench][m]
            L.append(f"- {m} 在现在的代码上重跑 对 {old}：答案相同 {r['answers identical']:.4f}；准确率差 {r['accuracy difference']:+.3f}")
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "## 按计划写定的读法", "",
          ("- **Q1**（VisionZip-64，fill − 底座）：" + "，".join(f"{bench} {fmt(P[f'{bench} vz64 fill - vz64'])}" for bench in BENCHES)
           + f"，两集平均 {q1['mean']:+.2f} → {q1['reading']}" + ("；提升大" if q1["large"] else "")) if Q1
          else "- **Q1**（VisionZip-64，fill − 底座）：计划 v1.2 拿掉，没有运行（VisionZip 加 fill 在 GPU 上跑不通，DEC-0091）",
          f"- **Q2**（紧凑缓存 对 普通实现）：{q2}",
          f"- **Q3**（DivPrune-64 在现在的代码上重跑 对 EXP-0032）：{q3}；EXP-0043 的 fill − 重跑的 DivPrune-64（描述）："
          + "，".join(f"{bench} {fmt(P[f'{bench} div64 fill (EXP-0043) - div64'])}" for bench in BENCHES)]
    open(f"{DST}/0044.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
