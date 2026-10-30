"""EXP-0035 plan v1.1's secondary analysis (written 2026-10-31 around 03:20, while EXP-0035 ran and before any EXP-0035
result; SQZ-EXP-0034 on the development sets suggested both questions): on the fresh sets, does (S1) the original
position layout without the layer-0 peek lean towards "yes", and does the peek remove it; (S2) do the original position
ids help GQA's position questions?

  python -B analyze_0035_secondary.py          # EXP-0035's runs -> code/output/EXP-0035/secondary/{secondary_0035.md,json}
  python -B analyze_0035_secondary.py --dev    # the same code on the development runs (EXP-0032 / 0033 / 0034) -> tmp

S1 yes/no direction: questions whose ground truth is yes or no (GQA: the answer; VQAv2: answer type 'yes/no' and the most
   common of the ten answers is yes or no). Per config the share answered yes and the accuracy on gt-yes / gt-no; paired
   nopeek - div64 and peek - nopeek within gt-no, peek - nopeek within gt-yes; first-token log-odds of peek and nopeek
   (their answer_logits.json) by gt. Readings (plan v1.1): nopeek - div64 on gt-no with the 95% interval below 0 on both
   fresh sets / one / neither -> "the lean towards yes reproduces (both / one) / does not"; peek - nopeek on gt-no above 0
   -> "the peek removes it (both / one) / does not"; peek - nopeek on gt-yes below 0 -> "the peek's cost on gt-yes
   reproduces (both / one) / does not".
S2 GQA position questions: detailed types starting with 'position'. Paired nopeek - div64, peek - div64, vanilla - div64,
   pooled and per type. Reading: nopeek - div64 pooled above 0 -> "the original position ids help position questions
   (reproduces)", else "does not reproduce".
S3 described: GQA detailed types and VQAv2 question types with at least 60 questions, the three pairs.
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0. Nothing here changes EXP-0035's
primary readings (Q1 to Q3).
"""
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
EV = "D:/6CJNM007/data/llava_eval"
DEV = "--dev" in sys.argv
if DEV:
    DST = "D:/6CJNM007/tmp/exp0035/secondary/dev_out"
    BENCH = {"vqav2_dev2": (f"{EV}/vqav2_val/llava_vqav2_val_dev2.jsonl", f"{EV}/vqav2_val/llava_vqav2_val_dev2_annotations.jsonl", "vqa_scores.jsonl"),
             "gqa_val400": (f"{EV}/gqa/llava_gqa_val400.jsonl", f"{EV}/gqa/llava_gqa_val400_annotations.jsonl", "gqa_scores.jsonl")}
    V, Q = "vqav2_dev2", "gqa_val400"
    RUNS = {"peek": {V: ("EXP-0034", "R01", "fastv61k1_div_peek_kll"), Q: ("EXP-0034", "R02", "fastv61k1_div_peek_kll")},
            "div64": {V: ("EXP-0032", "R01", "divprune64"), Q: ("EXP-0032", "R02", "divprune64")},
            "nopeek": {V: ("EXP-0034", "R03", "fastv61k1_div_nopeek_kll"), Q: ("EXP-0034", "R04", "fastv61k1_div_nopeek_kll")},
            "vanilla": {V: ("EXP-0033", "R03", "vanilla"), Q: ("EXP-0033", "R04", "vanilla")}}
else:
    DST = f"{OUT}/EXP-0035/secondary"
    BENCH = {"vqav2_test5": (f"{EV}/vqav2_val/llava_vqav2_val_test5.jsonl", f"{EV}/vqav2_val/llava_vqav2_val_test5_annotations.jsonl", "vqa_scores.jsonl"),
             "gqa_val400c": (f"{EV}/gqa/llava_gqa_val400c.jsonl", f"{EV}/gqa/llava_gqa_val400c_annotations.jsonl", "gqa_scores.jsonl")}
    V, Q = "vqav2_test5", "gqa_val400c"
    RUNS = {"peek": {V: ("EXP-0035", "R01", "fastv61k1_div_peek_kll"), Q: ("EXP-0035", "R02", "fastv61k1_div_peek_kll")},
            "div64": {V: ("EXP-0035", "R03", "divprune64"), Q: ("EXP-0035", "R04", "divprune64")},
            "nopeek": {V: ("EXP-0035", "R05", "fastv61k1_div_nopeek_kll"), Q: ("EXP-0035", "R06", "fastv61k1_div_nopeek_kll")},
            "vanilla": {V: ("EXP-0035", "R09", "vanilla"), Q: ("EXP-0035", "R10", "vanilla")}}
PAIRS = [("peek", "nopeek"), ("nopeek", "div64"), ("peek", "div64")]
N_BOOT, MIN_GROUP = 1000, 60
CHECKS = {}
WORD = {2: "both", 1: "one", 0: "neither"}


def run_dir(meth, bench):
    exp, rid, _ = RUNS[meth][bench]
    return f"{OUT}/{exp}/{exp}-{rid}"


def check_runs():
    for meth, by in RUNS.items():
        for bench, (exp, rid, cfg) in by.items():
            rows = {r["run_id"]: r for r in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")}
            r = rows.get(f"{exp}-{rid}", {})
            CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench}"] = float(r.get("status") == "completed" and r.get("config") == cfg and r.get("bench") == bench)


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def load(bench):
    qf, af, sf = BENCH[bench]
    qs = [json.loads(l) for l in open(qf, encoding="utf-8")]
    ann = {str(a["question_id"]): a for a in (json.loads(l) for l in open(af, encoding="utf-8"))}
    order = [str(q["question_id"]) for q in qs]
    image = {str(q["question_id"]): q["image"] for q in qs}
    ims = sorted(set(image.values()))
    idx = {im: k for k, im in enumerate(ims)}
    img_of = np.array([idx[image[q]] for q in order])
    if bench.startswith("gqa"):
        gt = [ann[q]["answer"] if ann[q]["answer"] in ("yes", "no") else None for q in order]
        dtype = [ann[q]["detailed"] for q in order]
    else:
        gt = []
        for q in order:
            top = Counter(ann[q]["answers"]).most_common(1)[0][0] if ann[q]["answer_type"] == "yes/no" else None
            gt.append(top if top in ("yes", "no") else None)
        dtype = [ann[q]["question_type"] for q in order]
    sc, said = {}, {}
    for meth in RUNS:
        d = run_dir(meth, bench)
        s = {str(x["question_id"]): float(x["score"]) for x in (json.loads(l) for l in open(f"{d}/{sf}", encoding="utf-8"))}
        a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
        CHECKS[f"{bench} {meth}: one score and one answer per question"] = float(sorted(s) == sorted(order) == sorted(a))
        sc[meth] = np.array([s[q] for q in order])
        said[meth] = [yes_no(a[q]) for q in order]
    by_img = [np.flatnonzero(img_of == k) for k in range(len(ims))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]
    return {"order": order, "gt": np.array(gt, dtype=object), "dtype": dtype, "sc": sc, "said": said, "draws": draws, "n": len(order)}


def paired(b, x, y, mask=None, values=None):
    if values is None:
        d, k = b["sc"][x] - b["sc"][y], 100.0
    else:
        d, k = values[x] - values[y], 1.0
    m = np.ones(b["n"], bool) if mask is None else mask
    if m.sum() == 0:
        return None
    boot = [k * d[i][m[i]].mean() for i in b["draws"] if m[i].any()]
    return [k * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]


def fmt(x, digits=2):
    return "—" if x is None else f"{x[0]:+.{digits}f} [{x[1]:+.{digits}f}, {x[2]:+.{digits}f}]（{x[3]}）"


def log_odds(b, bench):
    lo = {}
    for meth in ("peek", "nopeek"):
        x = json.load(open(f"{run_dir(meth, bench)}/answer_logits.json", encoding="utf-8"))
        CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == b["n"] == len(x["first_token"]))
        assert x["tokens"] == ["▁Yes", "▁No", "▁yes", "▁no"], x["tokens"]
        lp = np.array([r["logprobs"] for r in x["first_token"]])
        lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        yn = {x["ids"][0]: "yes", x["ids"][2]: "yes", x["ids"][1]: "no", x["ids"][3]: "no"}
        agree = [yn[r["argmax"]] == s for r, s in zip(x["first_token"], b["said"][meth]) if r["argmax"] in yn]
        CHECKS[f"{bench} {meth}: answers agree with the logged first-token argmax (>= 99%)"] = float(np.mean(agree) >= 0.99)
    return lo


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    check_runs()
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {CHECKS}")
    res = {"S1": {}, "S2": {}, "S3": {}}
    for bench in BENCH:
        b = load(bench)
        lo = log_odds(b, bench)
        r1 = {}
        for g in ("yes", "no"):
            m = b["gt"] == g
            r1[f"gt {g}"] = {"n": int(m.sum()),
                             "said_yes": {meth: float(np.mean([s == "yes" for s, mm in zip(b["said"][meth], m) if mm])) for meth in RUNS},
                             "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS},
                             "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in PAIRS},
                             "log_odds": {meth: float(v[m].mean()) for meth, v in lo.items()},
                             "log_odds peek - nopeek": paired(b, "peek", "nopeek", m, lo)}
        res["S1"][bench] = r1
        if bench == Q:
            pos = np.array([t.startswith("position") for t in b["dtype"]])
            r2 = {"pooled": {"n": int(pos.sum()), "accuracy": {meth: 100 * float(b["sc"][meth][pos].mean()) for meth in RUNS},
                             "pairs": {f"{x} - {y}": paired(b, x, y, pos) for x, y in [("nopeek", "div64"), ("peek", "div64"), ("vanilla", "div64")]}}}
            for t in sorted({t for t in b["dtype"] if t.startswith("position")}):
                m = np.array([d == t for d in b["dtype"]])
                r2[t] = {"n": int(m.sum()), "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS},
                         "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in [("nopeek", "div64"), ("peek", "div64"), ("vanilla", "div64")]}}
            res["S2"] = r2
        r3 = {}
        for t, c in sorted(Counter(b["dtype"]).items(), key=lambda kv: -kv[1]):
            if c >= MIN_GROUP:
                m = np.array([d == t for d in b["dtype"]])
                r3[t] = {"n": c, "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS},
                         "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in PAIRS}}
        res["S3"][bench] = r3
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    S1 = res["S1"]
    lean = sum(S1[bb]["gt no"]["pairs"]["nopeek - div64"][2] < 0 for bb in BENCH)
    removes = sum(S1[bb]["gt no"]["pairs"]["peek - nopeek"][1] > 0 for bb in BENCH)
    cost = sum(S1[bb]["gt yes"]["pairs"]["peek - nopeek"][2] < 0 for bb in BENCH)
    p = res["S2"]["pooled"]["pairs"]["nopeek - div64"]
    res["readings"] = {"S1 lean towards yes (nopeek - div64 on gt-no below 0)": WORD[lean],
                       "S1 the peek removes it (peek - nopeek on gt-no above 0)": WORD[removes],
                       "S1 the peek's cost on gt-yes (peek - nopeek on gt-yes below 0)": WORD[cost],
                       "S2 original position ids help position questions (nopeek - div64 above 0)": "reproduces" if p[1] > 0 else "does not reproduce"}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/secondary_0035.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = [f"# EXP-0035 计划 v1.1 的二级分析{'（开发集上的空跑，不是 EXP-0035 的结果）' if DEV else ''}：偏“是”与位置题", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    L += ["", "## S1 是非题的方向", ""]
    for bench in BENCH:
        L += [f"### {bench}", "", "| gt | 题数 | 答“是”的比例 peek / DivPrune-64 / nopeek / 不剪枝 | 准确率 同上 | nopeek − DivPrune-64 | peek − nopeek | 对数几率 peek − nopeek |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for g, x in S1[bench].items():
            L.append(f"| {g} | {x['n']} | " + " / ".join(f"{x['said_yes'][m]:.3f}" for m in RUNS) + " | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in RUNS)
                     + f" | {fmt(x['pairs']['nopeek - div64'])} | {fmt(x['pairs']['peek - nopeek'])} | {fmt(x['log_odds peek - nopeek'], 3)} |")
        L.append("")
    L += [f"## S2 GQA 的位置题（{Q}）", "", "| 题型 | 题数 | 准确率 peek / DivPrune-64 / nopeek / 不剪枝 | nopeek − DivPrune-64 | peek − DivPrune-64 | 不剪枝 − DivPrune-64 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for t, x in res["S2"].items():
        L.append(f"| {t} | {x['n']} | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in RUNS) + " | "
                 + " | ".join(fmt(x["pairs"][k]) for k in ("nopeek - div64", "peek - div64", "vanilla - div64")) + " |")
    L += ["", "## 按计划 v1.1 写定的读法", ""] + [f"- {k}: {v}" for k, v in res["readings"].items()]
    for bench in BENCH:
        L += ["", f"## S3（描述）{bench}：至少 {MIN_GROUP} 题的细分题型", "", "| 题型 | 题数 | 准确率 peek / DivPrune-64 / nopeek / 不剪枝 | peek − nopeek | nopeek − DivPrune-64 | peek − DivPrune-64 |",
              "| --- | --- | --- | --- | --- | --- |"]
        for t, x in res["S3"][bench].items():
            L.append(f"| {t} | {x['n']} | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in RUNS) + " | " + " | ".join(fmt(x["pairs"][f"{p} - {q}"]) for p, q in PAIRS) + " |")
    open(f"{DST}/secondary_0035.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    print("written", DST)


if __name__ == "__main__":
    main()
