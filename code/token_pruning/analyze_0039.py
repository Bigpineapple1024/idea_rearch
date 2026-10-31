"""EXP-0039 (plan v1): CDPruner-64 (ported from its official code, cdpruner.py) in this pipeline against DivPrune-64, on the
development sets; it decides the base of the next IDEA.

  python analyze_0039.py          # writes code/output/EXP-0039/analysis/{0039.md,0039.json}
  python analyze_0039.py --dry    # the same code with VisionZip-64 (EXP-0028-R01 / R02) standing in for CDPruner-64 and the
                                  # smoke test's 40-question log for the log checks -> tmp; not results

Runs (VQAv2 dev2 / GQA val400): CDPruner-64 R01 / R02. Reused: DivPrune-64 (EXP-0032-R01 / R02), VisionZip-64 (EXP-0028-R01 /
R02), unpruned (EXP-0033-R03 / R04).
Q1: CDPruner-64 - DivPrune-64. 95% interval above 0 on both sets -> "stronger" (the next IDEA builds on CDPruner-64); on one ->
"partly" (the next IDEA picks its base by the question types it targets and records why); else -> "not stronger" (DivPrune-64
stays the base: no text tower). Two-set mean >= +1.5 is noted "large".
Described: CDPruner-64 - VisionZip-64, DivPrune-64 - VisionZip-64, unpruned - CDPruner-64; by answer type, GQA structural and
semantic type, routed or not; GQA position questions (detailed types starting with 'position'); yes/no questions by ground
truth (GQA: the answer; VQAv2: answer type 'yes/no' and the most common of the ten answers is yes or no): the share answered yes.
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (analyze_0028's loader and scoring).
Checks: each run is the planned configuration x data and completed; one answer per question; recomputed scores equal the score
file and the registered headline; CDPruner's log has one record per question, each 64 distinct sorted patches in 0..575 whose
selection order is the same set, and its text is the question without the answer-format suffix.
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
KEEP, N_IMG, N_BOOT, LARGE = 64, 576, 1000, 1.5
SUFFIX = "\nAnswer the question using a single word or phrase."
if DRY:
    DST = "D:/6CJNM007/tmp/exp0039/dry_out"
    NEW = {"cdp": ("EXP-0028", ("R01", "R02"), "visionzip64")}
else:
    DST = f"{OUT}/EXP-0039/analysis"
    NEW = {"cdp": ("EXP-0039", ("R01", "R02"), "cdpruner64")}
OLD = {"div64": ("EXP-0032", ("R01", "R02"), "divprune64"), "vz64": ("EXP-0028", ("R01", "R02"), "visionzip64"),
       "vanilla": ("EXP-0033", ("R03", "R04"), "vanilla")}
RUNS = {**NEW, **OLD}
PAIRS = (("cdp", "div64"), ("cdp", "vz64"), ("div64", "vz64"), ("vanilla", "cdp"))
CHECKS = {}


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def check_log(name, log, questions):
    """CDPruner's log against the question file (records in question order: one image per question)."""
    recs = log["records"]
    CHECKS[f"{name}: one record per question, keep {KEEP}"] = float(log["keep"] == KEEP and log["n_images"] == len(recs) == len(questions))
    CHECKS[f"{name}: every record keeps {KEEP} distinct sorted patches in 0..{N_IMG - 1}, its order the same set (>= 99%)"] = float(np.mean(
        [len(r["kept"]) == KEEP and r["kept"] == sorted(set(r["kept"])) and 0 <= r["kept"][0] and r["kept"][-1] < N_IMG
         and sorted(r["order"]) == r["kept"] for r in recs]) >= 0.99) if recs else 0.0
    CHECKS[f"{name}: the record's text is the question without the answer-format suffix (>= 99%)"] = float(np.mean(
        [r["text"] == q["text"].replace(SUFFIX, "") for r, q in zip(recs, questions)]) >= 0.99) if recs else 0.0


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
    if DRY:      # the log checks on the smoke test's 40 questions (VQAv2 dev2's first 40)
        S = "D:/6CJNM007/tmp/cdpruner_port/smoke/"
        check_log("smoke test (40 questions)", json.load(open(S + "cdp.json", encoding="utf-8")), [json.loads(l) for l in open(S + "q40.jsonl", encoding="utf-8")])
    res = {"pairs": {}, "table": {}, "position": {}, "yes_no": {}, "seconds_per_answer": {}, "peak_gib": {}}
    for bi, bench in enumerate(BENCHES):
        b = a28.load_bench(bench)
        n = len(b["order"])
        qfile = a28.FILES[bench][0]
        questions = [json.loads(l) for l in open(qfile, encoding="utf-8")]
        assert [str(q["question_id"]) for q in questions] == b["order"]
        sc, said = {}, {}
        for meth, (exp, rids, cfg) in RUNS.items():
            d = f"{OUT}/{exp}/{exp}-{rids[bi]}"
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
            if meth == "cdp" and not DRY:
                check_log(f"{bench} cdp", json.load(open(f"{d}/cdp.json", encoding="utf-8")), questions)
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
        if bench.startswith("gqa"):
            gt = np.array([b["gts"][q]["answer"] if b["gts"][q]["answer"] in ("yes", "no") else None for q in b["order"]], dtype=object)
        else:
            gt = np.array([(lambda top: top if top in ("yes", "no") else None)(Counter(b["gts"][q]["answers"]).most_common(1)[0][0])
                           if b["gts"][q]["answer_type"] == "yes/no" else None for q in b["order"]], dtype=object)
        res["yes_no"][bench] = {f"gt {g}": {"n": int((gt == g).sum()),
                                            "said_yes": {meth: float(np.mean([s == "yes" for s, gg in zip(said[meth], gt) if gg == g])) for meth in RUNS},
                                            "accuracy": {meth: 100 * float(sc[meth][gt == g].mean()) for meth in RUNS},
                                            "cdp - div64": paired("cdp", "div64", gt == g)} for g in ("yes", "no")}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]
    v = [P[f"{bench} cdp - div64"] for bench in BENCHES]
    mean, n_pos = float(np.mean([t[0] for t in v])), sum(t[1] > 0 for t in v)
    res["criteria"] = {"Q1": {"reading": "stronger" if n_pos == 2 else "partly" if n_pos == 1 else "not stronger", "mean": mean, "large": mean >= LARGE}}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0039.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0039（计划 v1）：本流程中的 CDPruner-64 与 DivPrune-64（开发集）{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.0f}%" for k, x in CHECKS.items()]
    for bench in BENCHES:
        L += ["", f"## {bench}", "", "准确率：" + "；".join(f"{k} {x['accuracy']:.2f}" for k, x in res["table"][bench].items())]
        L += ["", "每题秒数：" + "；".join(f"{m} {res['seconds_per_answer'][f'{bench} {m}']:.4f}" for m in RUNS),
              "峰值显存（GiB）：" + "；".join(f"{m} {res['peak_gib'][f'{bench} {m}']:.2f}" for m in RUNS)]
        if bench in res["position"]:
            L += ["", f"位置题：{res['position'][bench]['n']} 题，" + "、".join(f"{t} {c}" for t, c in sorted(res["position"][bench]["types"].items()))]
        L += ["", "是非题（gt = 标准答案）：", ""]
        for g, x in res["yes_no"][bench].items():
            L.append(f"- {g}（{x['n']} 题）答“是”的比例：" + "；".join(f"{m} {x['said_yes'][m]:.3f}" for m in RUNS)
                     + "；准确率：" + "；".join(f"{m} {x['accuracy'][m]:.2f}" for m in RUNS) + f"；cdp − div64 {fmt(x['cdp - div64'])}")
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    c = res["criteria"]["Q1"]
    L += ["", "## 按计划写定的读法", "",
          "- **Q1**（cdp − div64）：" + "，".join(f"{bench} {fmt(P[f'{bench} cdp - div64'])}" for bench in BENCHES)
          + f"，两集平均 {c['mean']:+.2f} → {c['reading']}" + ("；提升大" if c["large"] else "")]
    open(f"{DST}/0039.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-1:]))


if __name__ == "__main__":
    main()
