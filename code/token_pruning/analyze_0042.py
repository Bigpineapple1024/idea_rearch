"""EXP-0042 (plan v1, IDEA-0011's confirmation, run only if EXP-0041's Q1 is "promising", or "partly" with a two-set mean
>= +0.5): fill (vfill.py) against CDPruner-64 on fresh data and on two other benchmarks.

  python analyze_0042.py          # writes code/output/EXP-0042/analysis/{0042.md,0042.json}
  python analyze_0042.py --dry    # the same code on stand-in runs -> tmp; not results

Data: VQAv2 test6 (1000 val2014 images, 5057 questions) and GQA val400d (400 val images, 5397 balanced questions), drawn
once by make_confirm_sets3.py on 2026-10-31 at 12:07, before any EXP-0040 result, and never used before EXP-0042; MME
(2374 questions) and POPE (8910). Runs: fill R01 / R03 / R05 / R07 and CDPruner-64 R02 / R04 / R06 / R08 on vqav2_test6 /
gqa_val400d / mme / pope; unpruned R09 / R10 on the two fresh sets (described); unpruned on MME / POPE reused
(EXP-0002-R05, EXP-0002-R34).
Q1 (fresh general data): fill - CDPruner-64 on vqav2_test6 and gqa_val400d, per-question accuracy; 95% interval above 0
   on both -> "promising", on one -> "partly", neither but the two-set mean > 0 -> "positive, not detected", else "unlike";
   two-set mean >= +1.5 noted "large".
Q2 (generalisation, IDEA-0008's lesson): fill - CDPruner-64 per-question accuracy on MME and on POPE: the 95% interval
   wholly below 0 -> "worse", else "not worse".
Conclusion: Q1 "promising" and Q2 "not worse" on both -> "confirmed"; Q1 "partly" and Q2 "not worse" on both -> "partly
   confirmed"; otherwise "not confirmed".
Described: by answer type and GQA type (as analyze_0028's groups), GQA position questions; yes/no questions by ground
truth (share answered yes) on the fresh sets and POPE; MME subtask scores; POPE accuracy / F1 / yes ratio per split;
unpruned - fill; the layout logs' assignment statistics; seconds per answer and peak memory.
Bootstrap: resample images with replacement within each benchmark, 1000 draws, seed 0.
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the
score files (fresh sets) and metrics.json (MME subtasks, POPE splits); fill and CDPruner-64 keep the same patches (>= 99%);
fill's layout log has one record per question with 64 image tokens, 576 slots, the expected positions and the layout
in 32 layers.
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
N_BOOT, LARGE, KEEP, N_SLOTS = 1000, 1.5, 64, 576
FRESH = {"vqav2_test6": (f"{EV}/vqav2_val/llava_vqav2_val_test6.jsonl", f"{EV}/vqav2_val/llava_vqav2_val_test6_annotations.jsonl",
                         "D:/6CJNM007/data/coco/val2014"),
         "gqa_val400d": (f"{EV}/gqa/llava_gqa_val400d.jsonl", f"{EV}/gqa/llava_gqa_val400d_annotations.jsonl",
                         "D:/6CJNM007/data/gqa/images_val400d")}
if DRY:
    DST = "D:/6CJNM007/tmp/exp0042/dry_out"
    GENERAL = ("vqav2_dev2", "gqa_val400")       # stand-ins: the development sets, older runs
    RUNS = {("vqav2_dev2", "fill"): ("EXP-0040", "R05", "cdpruner64_f19_logits"), ("vqav2_dev2", "b64"): ("EXP-0039", "R01", "cdpruner64"),
            ("gqa_val400", "fill"): ("EXP-0040", "R06", "cdpruner64_f19_logits"), ("gqa_val400", "b64"): ("EXP-0039", "R02", "cdpruner64"),
            ("mme", "fill"): ("EXP-0036", "R01", "fastv61k1_div_peek_kll"), ("mme", "b64"): ("EXP-0036", "R02", "divprune64"),
            ("pope", "fill"): ("EXP-0036", "R07", "fastv61k1_div_peek_kll"), ("pope", "b64"): ("EXP-0036", "R08", "divprune64"),
            ("vqav2_dev2", "vanilla"): ("EXP-0033", "R03", "vanilla"), ("gqa_val400", "vanilla"): ("EXP-0033", "R04", "vanilla")}
else:
    DST = f"{OUT}/EXP-0042/analysis"
    GENERAL = ("vqav2_test6", "gqa_val400d")
    RUNS = {("vqav2_test6", "fill"): ("EXP-0042", "R01", "cdpruner64_fill_logits"), ("vqav2_test6", "b64"): ("EXP-0042", "R02", "cdpruner64"),
            ("gqa_val400d", "fill"): ("EXP-0042", "R03", "cdpruner64_fill_logits"), ("gqa_val400d", "b64"): ("EXP-0042", "R04", "cdpruner64"),
            ("mme", "fill"): ("EXP-0042", "R05", "cdpruner64_fill_logits"), ("mme", "b64"): ("EXP-0042", "R06", "cdpruner64"),
            ("pope", "fill"): ("EXP-0042", "R07", "cdpruner64_fill_logits"), ("pope", "b64"): ("EXP-0042", "R08", "cdpruner64"),
            ("vqav2_test6", "vanilla"): ("EXP-0042", "R09", "vanilla"), ("gqa_val400d", "vanilla"): ("EXP-0042", "R10", "vanilla")}
RUNS.update({("mme", "vanilla"): ("EXP-0002", "R05", "vanilla"), ("pope", "vanilla"): ("EXP-0002", "R34", "vanilla")})
METHODS = ("fill", "b64", "vanilla")
CHECKS = a36.CHECKS                               # analyze_0036's loaders write their checks here too


def run_dir(bench, meth):
    exp, rid, _ = RUNS[(bench, meth)]
    return f"{OUT}/{exp}/{exp}-{rid}"


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def load_general(bench, rows):
    b = a28.load_bench(bench)
    # zero-padded image indices: analyze_0036.draws_for sorts them, so the draws are analyze_0028 / analyze_0040's
    out = {"order": b["order"], "image": [f"{int(b['img_of'][i]):05d}" for i in range(len(b["order"]))], "sc": {}, "said": {},
           "group": dict(b.get("groups", {}))}
    if bench.startswith("gqa"):
        out["group"]["[position]"] = np.array([b["gts"][q]["detailed"].startswith("position") for q in b["order"]])
        gt = np.array([b["gts"][q]["answer"] if b["gts"][q]["answer"] in ("yes", "no") else None for q in b["order"]], dtype=object)
    else:
        gt = np.array([(lambda top: top if top in ("yes", "no") else None)(Counter(b["gts"][q]["answers"]).most_common(1)[0][0])
                       if b["gts"][q]["answer_type"] == "yes/no" else None for q in b["order"]], dtype=object)
    out["group"]["gt yes"], out["group"]["gt no"] = gt == "yes", gt == "no"
    for meth in METHODS:
        exp, rid, _ = RUNS[(bench, meth)]
        d = run_dir(bench, meth)
        a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
        CHECKS[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
        s = a28.recompute(bench, b, a)
        logf = "vqa_scores.jsonl" if bench.startswith("vqav2") else "gqa_scores.jsonl"
        logged = {str(x["question_id"]): x["score"] for x in (json.loads(l) for l in open(f"{d}/{logf}", encoding="utf-8"))}
        CHECKS[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
        CHECKS[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(rows[f"{exp}-{rid}"]["headline"])) < 1e-6)
        out["sc"][meth] = s
        out["said"][meth] = np.array([yes_no(a[q]) for q in b["order"]], dtype=object)
    return out


def check_selection(bench, n):
    if DRY:
        return None
    sel = {m: json.load(open(f"{run_dir(bench, m)}/cdp.json", encoding="utf-8"))["records"] for m in ("fill", "b64")}
    CHECKS[f"{bench}: fill and CDPruner-64 keep the same 64 patches (>= 99%)"] = float(
        len(sel["fill"]) == len(sel["b64"]) == n and np.mean([a["kept"] == b["kept"] for a, b in zip(sel["fill"], sel["b64"])]) >= 0.99)
    lay = json.load(open(f"{run_dir(bench, 'fill')}/layout.json", encoding="utf-8"))
    pre = lay["prefills"]
    CHECKS[f"{bench} fill: the layout log has one record per question: {KEEP} image tokens, {N_SLOTS} slots, the expected positions, 32 layers"] = float(
        lay["mode"] == "fill" and lay["n_prefills"] == n == len(pre)
        and all(r["n"] == KEEP and r["slots"] == N_SLOTS and r["positions_ok"] and r["layers"] == 32 for r in pre))
    return {"sim_mean": float(np.mean([r["sim_mean"] for r in pre])), "count_max": float(np.mean([r["count_max"] for r in pre])),
            "count_ones": float(np.mean([r["count_ones"] for r in pre]))}


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    a28.FILES.update(FRESH)
    a36.RUNS.clear()
    a36.RUNS.update(RUNS)
    a36.METHODS = METHODS
    rows = {}
    for exp in sorted({e for e, _, _ in RUNS.values()}):
        rows.update({r["run_id"]: r for r in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    for (bench, meth), (exp, rid, cfg) in RUNS.items():
        r = rows.get(f"{exp}-{rid}", {})
        CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench} ({meth})"] = float(r.get("status") == "completed" and r.get("config") == cfg and r.get("bench") == bench)
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {[k for k, v in CHECKS.items() if v < 1]}")
    data = {bench: load_general(bench, rows) for bench in GENERAL}
    data["mme"], data["pope"] = a36.load_mme(), a36.load_pope()
    for b in ("mme", "pope"):
        data[b]["group"]["gt yes"] = data[b]["gt"] == "yes"
        data[b]["group"]["gt no"] = data[b]["gt"] == "no"
    res = {"pairs": {}, "table": {}, "yes_no": {}, "assignment": {}, "seconds_per_answer": {}, "peak_gib": {}}
    for bench, b in data.items():
        b["draws"] = a36.draws_for(b)
        stats = check_selection(bench, len(b["order"]))
        if stats:
            res["assignment"][bench] = stats
        for x, y in (("fill", "b64"), ("vanilla", "fill")):
            res["pairs"][f"{bench} {x} - {y}"] = a36.paired(b, x, y)
            for g, m in b["group"].items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = a36.paired(b, x, y, m)
        res["table"][bench] = {m: {"accuracy": 100 * float(b["sc"][m].mean()), **{g: 100 * float(b["sc"][m][mm].mean()) for g, mm in b["group"].items() if mm.any()}}
                               for m in METHODS}
        said = b["said"]
        res["yes_no"][bench] = {g: {"n": int(b["group"][g].sum()), "said_yes": {m: float(np.mean(said[m][b["group"][g]] == "yes")) for m in METHODS}}
                                for g in ("gt yes", "gt no")}
        for meth in METHODS:
            exp, rid, _ = RUNS[(bench, meth)]
            res["seconds_per_answer"][f"{bench} {meth}"] = float(rows[f"{exp}-{rid}"]["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(rows[f"{exp}-{rid}"]["peak_gib"])
    for meth in METHODS:                              # MME scores and POPE F1 per split, from metrics.json
        res["table"]["mme"][meth]["metrics"] = {k: v for k, v in json.load(open(f"{run_dir('mme', meth)}/metrics.json", encoding="utf-8")).items()
                                                if k in ("perception", "cognition")}
        pm = json.load(open(f"{run_dir('pope', meth)}/metrics.json", encoding="utf-8"))
        res["table"]["pope"][meth]["metrics"] = {s: {k: pm["splits"][s][k] for k in ("accuracy", "f1", "yes_ratio")} for s in ("adversarial", "popular", "random")}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]
    v = [P[f"{bench} fill - b64"] for bench in GENERAL]
    mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
    q1 = {"reading": "promising" if n_pos == 2 else "partly" if n_pos == 1 else "positive, not detected" if mean > 0 else "unlike",
          "mean": mean, "large": mean >= LARGE}
    q2 = {bench: ("worse" if P[f"{bench} fill - b64"][2] < 0 else "not worse") for bench in ("mme", "pope")}
    ok2 = all(x == "not worse" for x in q2.values())
    conclusion = ("confirmed" if q1["reading"] == "promising" and ok2 else "partly confirmed" if q1["reading"] == "partly" and ok2
                  else "not confirmed")
    res["criteria"] = {"Q1": q1, "Q2": q2, "conclusion": conclusion}
    res["checks"] = dict(CHECKS)
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0042.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0042（计划 v1）：IDEA-0011 的确认（fill 对 CDPruner-64）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    for bench in data:
        L += ["", f"## {bench}", "", "准确率（逐题）：" + "；".join(f"{m} {res['table'][bench][m]['accuracy']:.2f}" for m in METHODS),
              "每题秒数：" + "；".join(f"{m} {res['seconds_per_answer'][f'{bench} {m}']:.4f}" for m in METHODS),
              "峰值显存（GiB）：" + "；".join(f"{m} {res['peak_gib'][f'{bench} {m}']:.2f}" for m in METHODS)]
        if bench in res["assignment"]:
            s = res["assignment"][bench]
            L.append(f"填补的分配：平均余弦 {s['sim_mean']:.3f}；最多代表的槽数（平均）{s['count_max']:.1f}；不代表剪掉的 patch 的保留 token（平均）{s['count_ones']:.1f}")
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {x['said_yes'][m]:.3f}" for m in METHODS))
        if bench == "mme":
            L.append("MME：" + "；".join(f"{m} 感知 {res['table']['mme'][m]['metrics'].get('perception', float('nan')):.1f}、认知 {res['table']['mme'][m]['metrics'].get('cognition', float('nan')):.1f}" for m in METHODS))
        if bench == "pope":
            for m in METHODS:
                L.append(f"POPE {m}：" + "；".join(f"{s} 准确率 {100 * x['accuracy']:.2f}、F1 {100 * x['f1']:.2f}、答“是” {x['yes_ratio']:.3f}"
                                                   for s, x in res["table"]["pope"][m]["metrics"].items()))
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "## 按计划写定的读法", "",
          "- **Q1**（新抽的数据，fill − CDPruner-64）：" + "，".join(f"{bench} {fmt(P[f'{bench} fill - b64'])}" for bench in GENERAL)
          + f"，两集平均 {q1['mean']:+.2f} → {q1['reading']}" + ("；提升大" if q1["large"] else ""),
          "- **Q2**（推广，fill − CDPruner-64）：" + "，".join(f"{bench} {fmt(P[f'{bench} fill - b64'])} → {q2[bench]}" for bench in ("mme", "pope")),
          f"- **结论**：{conclusion}"]
    open(f"{DST}/0042.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
