"""EXP-0045 (plans v1 / v1.1, conditional): the fresh-data confirmation of the fill on the settings of EXP-0043 and EXP-0044
that qualified: A DivPrune-64 (also MME and POPE), B CDPruner-128, C CDPruner-192, D VisionZip-64.

  python analyze_0045.py          # writes code/output/EXP-0045/analysis/{0045.md,0045.json}
  python analyze_0045.py --dry    # the same code on stand-in runs -> tmp; not results

Data: VQAv2 test7 (1000 val2014 images, 5229 questions) and GQA val400e (400 GQA val images, 5037 questions), drawn once
by make_confirm_sets4.py on 2026-10-31 at 23:15 while EXP-0043's R01 ran, before any EXP-0043 result; MME and POPE for A.
A part ran if its run IDs are in runs.tsv; then all of them must be there and completed, and so must the unpruned R15 / R16.
A1, B, C, D (fresh general data): fill - base per set, per-question accuracy; 95% interval above 0 on both sets ->
   "promising", on one -> "partly", neither but the two-set mean > 0 -> "positive, not detected", else "unlike"; mean >=
   +1.5 noted "large". A2: fill - DivPrune-64 on MME and on POPE, per-question accuracy: the interval wholly below 0 ->
   "worse", else "not worse"; secondary (pre-registered, outside the conclusion): POPE fill - unpruned, wholly below 0 ->
   "worse than unpruned".
Conclusions: A: A1 "promising" and A2 "not worse" on both -> "confirmed"; A1 "partly" and A2 "not worse" on both ->
   "partly confirmed"; otherwise "not confirmed". B, C, D: "promising" -> "confirmed (general questions)", "partly" ->
   "partly confirmed", otherwise "not confirmed".
Described: unpruned - fill on the fresh sets; by answer type, GQA structural and semantic type, routed or not; GQA position
questions; yes/no by ground truth (share answered yes); agreement with the unpruned answer text; AUC of the first-token
log-odds on the yes/no questions for each run that logs them; MME subtask scores and POPE splits (accuracy, F1, yes
ratio) for A; the fill's assignment; seconds per answer and peak memory.
Bootstrap: resample images with replacement within each benchmark, 1000 draws, seed 0 (fresh sets: analyze_0028's image
order; MME, POPE: analyze_0036's).
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the
score files and headlines (fresh sets), metrics.json (MME subtasks, POPE splits); a fill keeps its base's patches
(selection logs; none for VisionZip, whose fill log is checked instead); the fill's layout log has one record per question
with n image tokens after 35, 576 slots (577 on VisionZip), the expected positions, the layout in 32 layers; a fill and
its base were given the same prompts; one first-token record per question where logged.
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
import analyze_0036 as a36  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
EV = "D:/6CJNM007/data/llava_eval"
DRY = "--dry" in sys.argv
N_BOOT, LARGE, SYS_LEN, N_LAYERS = 1000, 1.5, 35, 32
FRESH = {"vqav2_test7": (f"{EV}/vqav2_val/llava_vqav2_val_test7.jsonl", f"{EV}/vqav2_val/llava_vqav2_val_test7_annotations.jsonl",
                         "D:/6CJNM007/data/coco/val2014"),
         "gqa_val400e": (f"{EV}/gqa/llava_gqa_val400e.jsonl", f"{EV}/gqa/llava_gqa_val400e_annotations.jsonl",
                         "D:/6CJNM007/data/gqa/images_val400e")}
# part -> setting, n kept, slots, selection log, {(bench, role): (experiment, run, config, logs logits)}
if DRY:
    DST = "D:/6CJNM007/tmp/exp0045/dry_out"
    GENERAL = ("vqav2_dev2", "gqa_val400")
    E = "EXP-0041"
    PARTS = {
        "A": ("div64", 64, 576, None, {("vqav2_dev2", "fill"): (E, "R01", "cdpruner64_fill_logits", True), ("vqav2_dev2", "base"): ("EXP-0032", "R01", "divprune64", False),
                                       ("gqa_val400", "fill"): (E, "R02", "cdpruner64_fill_logits", True), ("gqa_val400", "base"): ("EXP-0032", "R02", "divprune64", False),
                                       ("mme", "fill"): ("EXP-0042", "R05", "cdpruner64_fill_logits", True), ("mme", "base"): ("EXP-0036", "R02", "divprune64", False),
                                       ("pope", "fill"): ("EXP-0042", "R07", "cdpruner64_fill_logits", True), ("pope", "base"): ("EXP-0036", "R08", "divprune64", False)}),
        "B": ("cdp128", 64, 576, None, {("vqav2_dev2", "fill"): (E, "R03", "cdpruner64_shuffle_logits", True), ("vqav2_dev2", "base"): ("EXP-0040", "R03", "cdpruner127_logits", True),
                                        ("gqa_val400", "fill"): (E, "R04", "cdpruner64_shuffle_logits", True), ("gqa_val400", "base"): ("EXP-0040", "R04", "cdpruner127_logits", True)}),
        "D": ("vz64", 64, 576, None, {("vqav2_dev2", "fill"): (E, "R05", "cdpruner64_orig_logits", True), ("vqav2_dev2", "base"): ("EXP-0028", "R01", "visionzip64", False),
                                      ("gqa_val400", "fill"): (E, "R06", "cdpruner64_orig_logits", True), ("gqa_val400", "base"): ("EXP-0028", "R02", "visionzip64", False)}),
    }
    VANILLA = {"vqav2_dev2": ("EXP-0033", "R03", "vanilla", False), "gqa_val400": ("EXP-0033", "R04", "vanilla", False)}
else:
    DST = f"{OUT}/EXP-0045/analysis"
    GENERAL = ("vqav2_test7", "gqa_val400e")
    E = "EXP-0045"

    def gen_runs(fill, base, rids, logits_base):
        return {(b, role): (E, rids[2 * i + j], cfg, lg) for i, b in enumerate(GENERAL)
                for j, (role, cfg, lg) in enumerate((("fill", fill, True), ("base", base, logits_base)))}
    PARTS = {
        "A": ("div64", 64, 576, "div.json", {**gen_runs("divprune64_fill_logits", "divprune64", ("R01", "R02", "R03", "R04"), False),
                                             ("mme", "fill"): (E, "R05", "divprune64_fill_logits", True), ("mme", "base"): ("EXP-0036", "R02", "divprune64", False),
                                             ("pope", "fill"): (E, "R06", "divprune64_fill_logits", True), ("pope", "base"): ("EXP-0036", "R08", "divprune64", False)}),
        "B": ("cdp128", 128, 576, "cdp.json", gen_runs("cdpruner128_fill_logits", "cdpruner128_logits", ("R07", "R08", "R09", "R10"), True)),
        "C": ("cdp192", 192, 576, "cdp.json", gen_runs("cdpruner192_fill_logits", "cdpruner192_logits", ("R11", "R12", "R13", "R14"), True)),
        "D": ("vz64", 64, 577, None, gen_runs("visionzip64_fill_logits", "visionzip64_logits", ("R17", "R18", "R19", "R20"), True)),
    }
    VANILLA = {"vqav2_test7": (E, "R15", "vanilla", False), "gqa_val400e": (E, "R16", "vanilla", False)}
VANILLA.update({"mme": ("EXP-0002", "R05", "vanilla", False), "pope": ("EXP-0002", "R34", "vanilla", False)})
CHECKS = a36.CHECKS


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


def run_dir(spec):
    exp, rid = spec[0], spec[1]
    return f"{OUT}/{exp}/{exp}-{rid}"


def reading(v):
    mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
    return {"reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
            "mean": mean, "large": mean >= LARGE}


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    a28.FILES.update(FRESH)
    rows = {}
    for exp in sorted({s[0] for _, _, _, _, r in PARTS.values() for s in r.values()} | {s[0] for s in VANILLA.values()}):
        if os.path.exists(f"{OUT}/{exp}/runs.tsv"):
            rows.update({x["run_id"]: x for x in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    ran = {p: any(f"{s[0]}-{s[1]}" in rows for (b, role), s in r.items() if s[0] == E) for p, (_, _, _, _, r) in PARTS.items()}
    if not any(ran.values()):
        sys.exit("STOP no part of EXP-0045 ran")
    for p, (_, _, _, _, r) in PARTS.items():
        if not ran[p]:
            continue
        for (bench, role), (exp, rid, cfg, _) in r.items():
            x = rows.get(f"{exp}-{rid}", {})
            CHECKS[f"{p}: {exp}-{rid} is a completed {cfg} x {bench} ({role})"] = float(x.get("status") == "completed" and x.get("config") == cfg and x.get("bench") == bench)
    for bench, (exp, rid, cfg, _) in VANILLA.items():
        if bench in GENERAL or ran["A"]:
            x = rows.get(f"{exp}-{rid}", {})
            CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench} (unpruned)"] = float(x.get("status") == "completed" and x.get("config") == cfg and x.get("bench") == bench)
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {[k for k, v in CHECKS.items() if v < 1]}")
    res = {"ran": ran, "pairs": {}, "table": {}, "yes_no": {}, "agreement": {}, "auc": {}, "assignment": {}, "seconds_per_answer": {},
           "peak_gib": {}, "mme": {}, "pope": {}, "criteria": {}}
    for bench in GENERAL:
        b = a28.load_bench(bench)
        n = len(b["order"])
        roles = {"vanilla": VANILLA[bench]}
        for p, (setting, _, _, _, r) in PARTS.items():
            if ran[p]:
                roles[f"{setting} fill"], roles[f"{setting} base"] = r[(bench, "fill")], r[(bench, "base")]
        sc, said, txt, prompts, lo = {}, {}, {}, {}, {}
        for meth, spec in roles.items():
            d = run_dir(spec)
            rec = {str(x["question_id"]): x for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
            a = {q: x["text"] for q, x in rec.items()}
            CHECKS[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = a28.recompute(bench, b, a)
            logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
            logged = {str(x["question_id"]): x["score"] for x in (json.loads(l) for l in open(f"{d}/{logf}", encoding="utf-8"))}
            CHECKS[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            CHECKS[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(rows[f"{spec[0]}-{spec[1]}"]["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(rows[f"{spec[0]}-{spec[1]}"]["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(rows[f"{spec[0]}-{spec[1]}"]["peak_gib"])
            sc[meth] = s
            said[meth] = np.array([yes_no(a[q]) for q in b["order"]], dtype=object)
            txt[meth] = np.array([norm(a[q]) for q in b["order"]], dtype=object)
            prompts[meth] = [rec[q].get("prompt") for q in b["order"]]
            if spec[3]:
                x = json.load(open(f"{d}/answer_logits.json", encoding="utf-8"))
                CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == n == len(x["first_token"]))
                lp = np.array([r_["logprobs"] for r_ in x["first_token"]])
                lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        res["assignment"][bench] = {}
        for p, (setting, n_keep, n_slots, sel, r) in PARTS.items():
            if not ran[p]:
                continue
            f, base = f"{setting} fill", f"{setting} base"
            CHECKS[f"{bench} {setting}: the fill and its base were given the same prompts"] = float(prompts[f] == prompts[base])
            if DRY:
                continue
            lay = json.load(open(f"{run_dir(r[(bench, 'fill')])}/layout.json", encoding="utf-8"))
            pre = lay["prefills"]
            CHECKS[f"{bench} {setting}: the layout log has mode fill, one record per question: {n_keep} image tokens after {SYS_LEN}, {n_slots} slots, "
                   f"the expected positions, the layout in {N_LAYERS} layers"] = float(
                lay["mode"] == "fill" and lay["n_prefills"] == n == len(pre)
                and all(x["n"] == n_keep and x["slots"] == n_slots and x["start"] == SYS_LEN and x["positions_ok"] and x["layers"] == N_LAYERS for x in pre))
            if sel:
                kf = [x["kept"] for x in json.load(open(f"{run_dir(r[(bench, 'fill')])}/{sel}", encoding="utf-8"))["records"]]
                kb = [x["kept"] for x in json.load(open(f"{run_dir(r[(bench, 'base')])}/{sel}", encoding="utf-8"))["records"]]
                CHECKS[f"{bench} {setting}: the fill keeps its base's patches (>= 99%)"] = float(len(kf) == len(kb) == n and np.mean([x == y for x, y in zip(kf, kb)]) >= 0.99)
            res["assignment"][bench][setting] = {"sim_mean": float(np.mean([x["sim_mean"] for x in pre])), "count_max": float(np.mean([x["count_max"] for x in pre])),
                                                 "count_ones": float(np.mean([x["count_ones"] for x in pre]))}
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
        agree = {m: (txt[m] == txt["vanilla"]).astype(float) for m in roles if m != "vanilla"}
        for p, (setting, _, _, _, _) in PARTS.items():
            if not ran[p]:
                continue
            f, base = f"{setting} fill", f"{setting} base"
            for x, y in ((f, base), ("vanilla", f)):
                res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
                for g, m in groups.items():
                    res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
            res["agreement"][f"{bench} {setting}"] = {"fill": float(agree[f].mean()), "base": float(agree[base].mean()), "fill - base": paired(f, base, None, agree)}
        res["table"][bench] = {m: {"accuracy": 100 * float(v.mean()), **{g: 100 * float(v[mm].mean()) for g, mm in groups.items() if mm.any()}} for m, v in sc.items()}
        res["yes_no"][bench] = {f"gt {g}": {"n": int((gt == g).sum()), "said_yes": {m: float(np.mean(said[m][gt == g] == "yes")) for m in roles}} for g in ("yes", "no")}
        res["auc"][bench] = {m: auc(v[gt == "yes"], v[gt == "no"]) for m, v in lo.items()}
    if ran["A"]:                                          # A2: MME and POPE (analyze_0036's loaders, as analyze_0042)
        r = PARTS["A"][4]
        a36.RUNS.clear()
        for bench in ("mme", "pope"):
            a36.RUNS.update({(bench, "fill"): r[(bench, "fill")][:3], (bench, "b64"): r[(bench, "base")][:3], (bench, "vanilla"): VANILLA[bench][:3]})
        a36.METHODS = ("fill", "b64", "vanilla")
        for bench, loader in (("mme", a36.load_mme), ("pope", a36.load_pope)):
            bb = loader()
            bb["draws"] = a36.draws_for(bb)
            bb["group"]["gt yes"], bb["group"]["gt no"] = bb["gt"] == "yes", bb["gt"] == "no"
            for x, y in (("fill", "b64"), ("vanilla", "fill")):
                res["pairs"][f"{bench} div64 {x} - {y}"] = a36.paired(bb, x, y)
                for g, m in bb["group"].items():
                    res["pairs"][f"{bench} div64 {x} - {y} {g}"] = a36.paired(bb, x, y, m)
            res["table"][bench] = {m: {"accuracy": 100 * float(bb["sc"][m].mean())} for m in a36.METHODS}
            res["yes_no"][bench] = {g: {"n": int(bb["group"][g].sum()), "said_yes": {m: float(np.mean(bb["said"][m][bb["group"][g]] == "yes")) for m in a36.METHODS}}
                                    for g in ("gt yes", "gt no")}
            for m in a36.METHODS:
                spec = {"fill": r[(bench, "fill")], "b64": r[(bench, "base")], "vanilla": VANILLA[bench]}[m]
                mt = json.load(open(f"{run_dir(spec)}/metrics.json", encoding="utf-8"))
                if bench == "mme":
                    res["mme"][m] = {k: mt[k] for k in ("perception", "cognition")}
                else:
                    res["pope"][m] = {s: {k: mt["splits"][s][k] for k in ("accuracy", "f1", "yes_ratio")} for s in ("adversarial", "popular", "random")}
            if not DRY:
                lay = json.load(open(f"{run_dir(r[(bench, 'fill')])}/layout.json", encoding="utf-8"))
                CHECKS[f"{bench} div64: the layout log has mode fill, one record per question, 64 image tokens, 576 slots, the expected positions, 32 layers"] = float(
                    lay["mode"] == "fill" and lay["n_prefills"] == len(bb["order"]) == len(lay["prefills"])
                    and all(x["n"] == 64 and x["slots"] == 576 and x["positions_ok"] and x["layers"] == N_LAYERS for x in lay["prefills"]))
                kf = [x["kept"] for x in json.load(open(f"{run_dir(r[(bench, 'fill')])}/div.json", encoding="utf-8"))["records"]]
                kb = [x["kept"] for x in json.load(open(f"{run_dir(r[(bench, 'base')])}/div.json", encoding="utf-8"))["records"]]
                CHECKS[f"{bench} div64: the fill keeps DivPrune-64's patches (>= 99%)"] = float(len(kf) == len(kb) and np.mean([x == y for x, y in zip(kf, kb)]) >= 0.99)
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]
    for p, (setting, _, _, _, _) in PARTS.items():
        if not ran[p]:
            continue
        q = reading([P[f"{bench} {setting} fill - {setting} base"] for bench in GENERAL])
        if p == "A":
            a2 = {bench: ("worse" if P[f"{bench} div64 fill - b64"][2] < 0 else "not worse") for bench in ("mme", "pope")}
            second = "worse than unpruned" if P["pope div64 vanilla - fill"][1] > 0 else "not worse than unpruned"
            ok2 = all(x == "not worse" for x in a2.values())
            concl = "confirmed" if q["reading"] == "promising" and ok2 else "partly confirmed" if q["reading"] == "partly" and ok2 else "not confirmed"
            res["criteria"]["A"] = {"A1": q, "A2": a2, "secondary (POPE against unpruned)": second, "conclusion": concl}
        else:
            concl = {"promising": "confirmed (general questions)", "partly": "partly confirmed"}.get(q["reading"], "not confirmed")
            res["criteria"][p] = {"Q": q, "conclusion": concl}
    res["checks"] = dict(CHECKS)
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0045.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0045（计划 v1、v1.1）：EXP-0043、EXP-0044 的确认（新数据）{'（替身数据的空跑，不是结果）' if DRY else ''}", "",
         "运行的部分：" + "、".join(f"{p}（{PARTS[p][0]}）" for p in PARTS if ran[p]), "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    for bench in GENERAL:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {x['accuracy']:.2f}" for k, x in res["table"][bench].items()),
              "每题秒数：" + "；".join(f"{k.split(' ', 1)[1]} {v:.4f}" for k, v in res["seconds_per_answer"].items() if k.startswith(bench)),
              "峰值显存（GiB）：" + "；".join(f"{k.split(' ', 1)[1]} {v:.2f}" for k, v in res["peak_gib"].items() if k.startswith(bench))]
        for s, st in res["assignment"].get(bench, {}).items():
            L.append(f"- {s} 填补的分配：平均余弦 {st['sim_mean']:.3f}；最多代表的槽数（平均）{st['count_max']:.1f}；不代表剪掉的 patch 的 token（平均）{st['count_ones']:.1f}")
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {v:.3f}" for m, v in x["said_yes"].items()))
        L.append("- 是非题上第一个答案 token 的对数几率的 AUC：" + "；".join(f"{m} {v:.4f}" for m, v in res["auc"][bench].items()))
        for k, x in res["agreement"].items():
            if k.startswith(bench):
                L.append(f"- {k.split(' ', 1)[1]} 与不剪枝的回答相同的比例：fill {x['fill']:.3f}；底座 {x['base']:.3f}；fill − 底座 {fmt(x['fill - base'])}")
    if ran["A"]:
        L += ["", "## MME、POPE（A：DivPrune-64）", "", "逐题准确率：" + "；".join(f"{b_} " + "、".join(f"{m} {x['accuracy']:.2f}" for m, x in res["table"][b_].items()) for b_ in ("mme", "pope")),
              "MME：" + "；".join(f"{m} 感知 {x['perception']:.1f}、认知 {x['cognition']:.1f}" for m, x in res["mme"].items())]
        for m, x in res["pope"].items():
            L.append(f"POPE {m}：" + "；".join(f"{s} 准确率 {100 * y['accuracy']:.2f}、F1 {100 * y['f1']:.2f}、答“是” {y['yes_ratio']:.3f}" for s, y in x.items()))
        for b_ in ("mme", "pope"):
            for g, x in res["yes_no"][b_].items():
                L.append(f"- {b_} {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {v:.3f}" for m, v in x["said_yes"].items()))
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "## 按计划写定的读法", ""]
    for p, c in res["criteria"].items():
        s = PARTS[p][0]
        q = c["A1"] if p == "A" else c["Q"]
        line = (f"- **{p}**（{s}，fill − 底座，新数据）：" + "，".join(f"{bench} {fmt(P[f'{bench} {s} fill - {s} base'])}" for bench in GENERAL)
                + f"，两集平均 {q['mean']:+.2f} → {q['reading']}" + ("；提升大" if q["large"] else ""))
        if p == "A":
            line += ("；MME " + fmt(P["mme div64 fill - b64"]) + f" → {c['A2']['mme']}，POPE " + fmt(P["pope div64 fill - b64"]) + f" → {c['A2']['pope']}"
                     + f"；次要读法：POPE 不剪枝 − fill {fmt(P['pope div64 vanilla - fill'])} → {c['secondary (POPE against unpruned)']}")
        L.append(line + f"；**结论**：{c['conclusion']}")
    open(f"{DST}/0045.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(res["criteria"]) + 1):]))


if __name__ == "__main__":
    main()
