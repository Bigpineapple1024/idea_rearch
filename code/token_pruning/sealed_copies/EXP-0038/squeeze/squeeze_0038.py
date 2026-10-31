"""SQZ-EXP-0038 (written 2026-10-31 around 11:05, before running it; EXP-0038's sealed analysis had been read): is the peek's
gain in EXP-0038 (peekc - nopeekc) more than winning back what nopeekc's layer 0 at the original, sparse positions costs?

  python -B squeeze_0038.py      # writes code/output/EXP-0038/squeeze/{squeeze_0038.md,squeeze_0038.json}

Runs (all completed and sealed): EXP-0038-R01 / R07 (peekc; R07 reran R02), R03 / R04 (nopeekc), R05 / R06 (DivPrune-61);
EXP-0032-R01 / R02 (DivPrune-64); EXP-0034-R01 to R04 (peek, nopeek: original numbering at every layer); EXP-0033-R03 / R04
(unpruned).

S1 yes/no direction (as SQZ-EXP-0034 S1). Questions whose ground truth is yes or no (GQA: the answer; VQAv2: answer type
   'yes/no' and the most common of the ten answers is yes or no). Per config: the share answered yes and the accuracy on
   gt-yes and gt-no questions; paired differences within gt-yes and gt-no.
S2 first-token log-odds log(P(Yes) + P(yes)) - log(P(No) + P(no)) (peekc, nopeekc, peek, nopeek log them): the mean by gt;
   paired peekc - nopeekc and peek - nopeek.
S3 GQA position questions (detailed types starting with 'position', as analyze_0035_secondary S2): all configs, the pairs,
   and peek - peekc, nopeek - nopeekc (original numbering from layer 1 on against contiguous).
S4 by detailed type (at least 60 questions): peekc - DivPrune-61 and nopeekc - DivPrune-61.
The pairs: peekc - DivPrune-61 (layer 0 vanilla-like: the whole image at its positions, against layer 0 with only the 61
kept, contiguous), nopeekc - DivPrune-61, peekc - nopeekc, peekc - DivPrune-64.
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (as analyze_0038). Exploratory, many
groups, no multiple-comparison correction; nothing here changes EXP-0038's readings.
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
DST = f"{OUT}/EXP-0038/squeeze"
BENCH = {"vqav2_dev2": (f"{EV}/vqav2_val/llava_vqav2_val_dev2.jsonl", f"{EV}/vqav2_val/llava_vqav2_val_dev2_annotations.jsonl", "vqa_scores.jsonl"),
         "gqa_val400": (f"{EV}/gqa/llava_gqa_val400.jsonl", f"{EV}/gqa/llava_gqa_val400_annotations.jsonl", "gqa_scores.jsonl")}
RUNS = {"peekc": ("EXP-0038", {"vqav2_dev2": "R01", "gqa_val400": "R07"}, "fastv61k1_div_peekc_kll"),
        "nopeekc": ("EXP-0038", {"vqav2_dev2": "R03", "gqa_val400": "R04"}, "fastv61k1_div_nopeekc_kll"),
        "div61": ("EXP-0038", {"vqav2_dev2": "R05", "gqa_val400": "R06"}, "divprune61"),
        "div64": ("EXP-0032", {"vqav2_dev2": "R01", "gqa_val400": "R02"}, "divprune64"),
        "peek": ("EXP-0034", {"vqav2_dev2": "R01", "gqa_val400": "R02"}, "fastv61k1_div_peek_kll"),
        "nopeek": ("EXP-0034", {"vqav2_dev2": "R03", "gqa_val400": "R04"}, "fastv61k1_div_nopeek_kll"),
        "vanilla": ("EXP-0033", {"vqav2_dev2": "R03", "gqa_val400": "R04"}, "vanilla")}
PAIRS = [("peekc", "div61"), ("nopeekc", "div61"), ("peekc", "nopeekc"), ("peekc", "div64")]
SEALED = [("peekc", "nopeekc"), ("nopeekc", "div61"), ("peekc", "div64"), ("vanilla", "peekc"), ("peekc", "peek"), ("nopeekc", "nopeek")]
N_BOOT, MIN_GROUP = 1000, 60
CHECKS = {}


def run_dir(meth, bench):
    exp, ids, _ = RUNS[meth]
    return f"{OUT}/{exp}/{exp}-{ids[bench]}"


def check_runs():
    for meth, (exp, ids, cfg) in RUNS.items():
        rows = {r["run_id"]: r for r in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")}
        for bench, rid in ids.items():
            r = rows[f"{exp}-{rid}"]
            CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench}"] = float(r["status"] == "completed" and r["config"] == cfg and r["bench"] == bench)


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
    if bench == "gqa_val400":
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
    """Mean difference x - y (in points for scores, raw for values) over mask, with the 95% bootstrap interval."""
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


def s1(b):
    out = {}
    for g in ("yes", "no"):
        m = b["gt"] == g
        out[f"gt {g}"] = {"n": int(m.sum()),
                          "said_yes": {meth: float(np.mean([s == "yes" for s, mm in zip(b["said"][meth], m) if mm])) for meth in RUNS},
                          "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS},
                          "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in PAIRS}}
    return out


def s2(b, bench):
    lo = {}
    for meth in ("peekc", "nopeekc", "peek", "nopeek"):
        x = json.load(open(f"{run_dir(meth, bench)}/answer_logits.json", encoding="utf-8"))
        CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == b["n"] == len(x["first_token"]))
        assert x["tokens"] == ["▁Yes", "▁No", "▁yes", "▁no"], x["tokens"]
        lp = np.array([r["logprobs"] for r in x["first_token"]])
        lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        top = [r["argmax"] for r in x["first_token"]]
        yn = {x["ids"][0]: "yes", x["ids"][2]: "yes", x["ids"][1]: "no", x["ids"][3]: "no"}
        agree = [yn[t] == s for t, s in zip(top, b["said"][meth]) if t in yn]
        CHECKS[f"{bench} {meth}: answers agree with the logged first-token argmax (alignment, >= 99%)"] = float(np.mean(agree) >= 0.99)
    out = {}
    for g in ("yes", "no"):
        m = b["gt"] == g
        out[f"gt {g}"] = {"mean_log_odds": {meth: float(v[m].mean()) for meth, v in lo.items()},
                          "peekc - nopeekc": paired(b, "peekc", "nopeekc", m, lo), "peek - nopeek": paired(b, "peek", "nopeek", m, lo)}
    return out


def s3(b):
    pos = np.array([t.startswith("position") for t in b["dtype"]])
    pairs = PAIRS + [("peek", "peekc"), ("nopeek", "nopeekc"), ("vanilla", "peekc")]
    out = {"all position": {"n": int(pos.sum()), "accuracy": {meth: 100 * float(b["sc"][meth][pos].mean()) for meth in RUNS},
                            "pairs": {f"{x} - {y}": paired(b, x, y, pos) for x, y in pairs}}}
    for t in sorted({t for t in b["dtype"] if t.startswith("position")}):
        m = np.array([d == t for d in b["dtype"]])
        out[t] = {"n": int(m.sum()), "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS},
                  "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in pairs}}
    return out


def s4(b):
    out = {}
    for t, c in sorted(Counter(b["dtype"]).items(), key=lambda kv: -kv[1]):
        if c < MIN_GROUP:
            continue
        m = np.array([d == t for d in b["dtype"]])
        out[t] = {"n": c, "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in ("peekc", "nopeekc", "div61", "div64")},
                  "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in (("peekc", "div61"), ("nopeekc", "div61"))}}
    return out


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    check_runs()
    sealed = json.load(open(f"{OUT}/EXP-0038/analysis/0038.json", encoding="utf-8"))["pairs"]
    res = {}
    for bench in BENCH:
        b = load(bench)
        for x, y in SEALED:
            CHECKS[f"{bench} {x} - {y}: the overall mean difference equals the sealed 0038.json"] = float(
                abs(100 * float((b["sc"][x] - b["sc"][y]).mean()) - sealed[f"{bench} {x} - {y}"][0]) < 1e-6)
        res[bench] = {"overall": {f"{x} - {y}": paired(b, x, y) for x, y in PAIRS}, "S1": s1(b), "S2": s2(b, bench), "S4": s4(b)}
        if bench == "gqa_val400":
            res[bench]["S3"] = s3(b)
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/squeeze_0038.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = ["# SQZ-EXP-0038：借看的增益是否只是补回 nopeekc 第 0 层稀疏编号的损失", "",
         "事后、探索性；分组多，没有多重比较校正；不改 EXP-0038 的读法。区间：每个集内按图像重抽 1000 次（种子 0）。", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    for bench in BENCH:
        r = res[bench]
        L += ["", f"## {bench}", "", "### 整体", ""] + [f"- {k}: {fmt(x)}" for k, x in r["overall"].items()]
        L += ["", "### S1 是非题的方向（gt = 标准答案）", "",
              "| gt | 题数 | 答“是”的比例：" + " / ".join(RUNS) + " | 准确率：同上 |", "| --- | --- | --- | --- |"]
        for g, x in r["S1"].items():
            L.append(f"| {g} | {x['n']} | " + " / ".join(f"{x['said_yes'][m]:.3f}" for m in RUNS) + " | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in RUNS) + " |")
        L += ["", "| gt | " + " | ".join(f"{p} − {q}" for p, q in PAIRS) + " |", "| --- |" + " --- |" * len(PAIRS)]
        for g, x in r["S1"].items():
            L.append(f"| {g} | " + " | ".join(fmt(x["pairs"][f"{p} - {q}"]) for p, q in PAIRS) + " |")
        L += ["", "### S2 第一个答案 token 的对数几率 log(P(Yes)+P(yes)) − log(P(No)+P(no))", "",
              "| gt | 平均：peekc / nopeekc / peek / nopeek | peekc − nopeekc | peek − nopeek |", "| --- | --- | --- | --- |"]
        for g, x in r["S2"].items():
            L.append(f"| {g} | " + " / ".join(f"{x['mean_log_odds'][m]:+.3f}" for m in ("peekc", "nopeekc", "peek", "nopeek"))
                     + f" | {fmt(x['peekc - nopeekc'], 3)} | {fmt(x['peek - nopeek'], 3)} |")
        if "S3" in r:
            L += ["", "### S3 GQA 位置题（细分题型以 position 开头）", ""]
            for t, x in r["S3"].items():
                L += [f"- **{t}**（{x['n']} 题）准确率：" + "；".join(f"{m} {v:.2f}" for m, v in x["accuracy"].items())]
                L += [f"  - {k}: {fmt(v)}" for k, v in x["pairs"].items()]
        L += ["", f"### S4 按细分题型（至少 {MIN_GROUP} 题）", "",
              "| 题型 | 题数 | 准确率 peekc / nopeekc / DivPrune-61 / DivPrune-64 | peekc − DivPrune-61 | nopeekc − DivPrune-61 |", "| --- | --- | --- | --- | --- |"]
        for t, x in r["S4"].items():
            L.append(f"| {t} | {x['n']} | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in ("peekc", "nopeekc", "div61", "div64")) + " | "
                     + " | ".join(fmt(x["pairs"][k]) for k in ("peekc - div61", "nopeekc - div61")) + " |")
    open(f"{DST}/squeeze_0038.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", DST)


if __name__ == "__main__":
    main()
