"""SQZ-EXP-0042 (post-hoc, exploratory; written after reading EXP-0042's sealed 0042.md, before any number computed here):
why the virtual fill, which gains on the fresh VQAv2 / GQA sets, loses to CDPruner-64 on POPE.

  python -B squeeze_0042.py      # writes code/output/EXP-0042/squeeze/{squeeze_0042.md,squeeze_0042.json}

Runs (sealed, read only): EXP-0042-R01..R10, unpruned MME / POPE EXP-0002-R05 / R34, loaded with the committed
analyze_0042.py / analyze_0036.py loaders; every accuracy must equal analyze_0042's sealed table. Bootstrap as analyze_0042
(images resampled within each benchmark, 1000 draws, seed 0).
S1 (does fill give the unpruned model's answers back?): per question, does a run's answer equal the unpruned run's answer?
   POPE and MME: the scored yes / no label; VQAv2 test6 and GQA val400d: the answer text (stripped, lower case, final
   full stop removed), on all questions and on the yes / no questions. fill - b64 of that agreement (paired, by image).
S2 (is CDPruner-64 above the unpruned model on POPE?): b64 - vanilla and fill - vanilla on POPE, per split and by ground
   truth; b64 - vanilla on MME and on the two fresh sets.
S3 (a shift of the yes / no threshold, or less discrimination?): fill's first-token log-odds on POPE (yes + Yes against
   no + No, as analyze_0036): the share of questions where its sign gives fill's scored label; the AUC of gt yes against gt
   no, per split; "matched": fill's accuracy if, in each split, it said "yes" on as many questions as CDPruner-64 did (the
   threshold is the matching quantile of fill's log-odds, chosen on the same questions: descriptive, not a method), against
   b64 (the threshold stays fixed in the bootstrap).
S4 (MME): every subtask's score (acc + acc+) for fill, b64 and unpruned, from metrics.json; fill - b64 per subtask.
S5 (dev against fresh): fill - b64 by answer and question type on the development sets (EXP-0041's sealed 0041.json) and
   on the fresh sets (EXP-0042's sealed 0042.json), side by side; nothing recomputed.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402
import analyze_0036 as a36  # noqa: E402
import analyze_0042 as a42  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
DST = f"{OUT}/EXP-0042/squeeze"
GENERAL = ("vqav2_test6", "gqa_val400d")
SPLITS = ("adversarial", "popular", "random")
CHECKS = a36.CHECKS


def norm(text):
    t = text.strip().lower()
    return t[:-1] if t.endswith(".") else t


def auc(pos, neg):
    """Probability that a gt-yes question has higher log-odds than a gt-no one (ties count half)."""
    v = np.concatenate([pos, neg])
    order = v.argsort(kind="mergesort")
    ranks = np.empty(len(v))
    s = v[order]
    i = 0
    while i < len(s):                                  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    sealed = json.load(open(f"{OUT}/EXP-0042/analysis/0042.json", encoding="utf-8"))
    dev = json.load(open(f"{OUT}/EXP-0041/analysis/0041.json", encoding="utf-8"))
    a28.FILES.update(a42.FRESH)
    a36.RUNS.clear()
    a36.RUNS.update(a42.RUNS)
    a36.METHODS = a42.METHODS
    rows = {}
    for exp in sorted({e for e, _, _ in a42.RUNS.values()}):
        rows.update({r["run_id"]: r for r in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    data = {bench: a42.load_general(bench, rows) for bench in GENERAL}
    data["mme"], data["pope"] = a36.load_mme(), a36.load_pope()
    texts = {}
    for bench in GENERAL:
        b = a28.load_bench(bench)
        for meth in a42.METHODS:
            a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{a42.run_dir(bench, meth)}/answers.jsonl", encoding="utf-8"))}
            texts[(bench, meth)] = np.array([norm(a[q]) for q in b["order"]], dtype=object)
    for bench, b in data.items():
        b["draws"] = a36.draws_for(b)
        for meth in a42.METHODS:
            CHECKS[f"{bench} {meth}: accuracy equals analyze_0042's sealed table"] = float(
                abs(100 * b["sc"][meth].mean() - sealed["table"][bench][meth]["accuracy"]) < 1e-9)
    res = {"S1": {}, "S2": {}, "S3": {}, "S4": {}, "S5": {}}
    # S1: agreement with the unpruned run
    for bench, b in data.items():
        if bench in GENERAL:
            yn = b["group"]["gt yes"] | b["group"]["gt no"]
            agree = {m: (texts[(bench, m)] == texts[(bench, "vanilla")]).astype(float) for m in ("fill", "b64")}
            masks = {"all": None, "yes/no questions": yn}
        else:
            agree = {m: (b["said"][m] == b["said"]["vanilla"]).astype(float) for m in ("fill", "b64")}
            masks = {"all": None, "gt yes": b["gt"] == "yes", "gt no": b["gt"] == "no"}
            if bench == "pope":
                masks.update({s: np.array([x == s for x in b["split"]]) for s in SPLITS})
        res["S1"][bench] = {}
        for k, m in masks.items():
            mm = np.ones(len(b["order"]), bool) if m is None else m
            res["S1"][bench][k] = {"n": int(mm.sum()), "fill": float(agree["fill"][mm].mean()), "b64": float(agree["b64"][mm].mean()),
                                   "fill - b64": a36.paired(b, "fill", "b64", m, {x: 100 * v for x, v in agree.items()})}
    # S2: against the unpruned run
    p = data["pope"]
    groups = {"all": None, "gt yes": p["gt"] == "yes", "gt no": p["gt"] == "no"}
    for s in SPLITS:
        k = np.array([x == s for x in p["split"]])
        groups.update({s: k, f"{s} gt yes": k & (p["gt"] == "yes"), f"{s} gt no": k & (p["gt"] == "no")})
    res["S2"]["pope"] = {g: {"b64 - vanilla": a36.paired(p, "b64", "vanilla", m), "fill - vanilla": a36.paired(p, "fill", "vanilla", m),
                             "fill - b64": a36.paired(p, "fill", "b64", m)} for g, m in groups.items()}
    for bench in ("mme",) + GENERAL:
        res["S2"][bench] = {"b64 - vanilla": a36.paired(data[bench], "b64", "vanilla")}
    # S3: fill's first-token log-odds on POPE
    x = json.load(open(f"{a42.run_dir('pope', 'fill')}/answer_logits.json", encoding="utf-8"))
    CHECKS["pope fill: one first-token record per question"] = float(x["n_prefills"] == len(p["order"]) == len(x["first_token"]))
    CHECKS["pope fill: the logged tokens are Yes, No, yes, no"] = float(x["tokens"] == ["\u2581Yes", "\u2581No", "\u2581yes", "\u2581no"])
    lp = np.array([r["logprobs"] for r in x["first_token"]])
    lo = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
    sign = np.where(lo > 0, "yes", "no")
    CHECKS["pope fill: the log-odds' sign gives the scored label (>= 99%)"] = float(np.mean(sign == p["said"]["fill"]) >= 0.99)
    res["S3"]["sign = scored label"] = float(np.mean(sign == p["said"]["fill"]))
    matched = np.empty(len(p["order"]), dtype=object)
    covered = np.zeros(len(p["order"]), int)
    for s in SPLITS + ("all",):
        k = np.ones(len(p["order"]), bool) if s == "all" else np.array([x == s for x in p["split"]])
        gt = p["gt"][k]
        r = {"n": int(k.sum()), "auc": auc(lo[k][gt == "yes"], lo[k][gt == "no"]),
             "yes ratio": {m: float(np.mean(p["said"][m][k] == "yes")) for m in a42.METHODS},
             "accuracy": {m: 100 * float(p["sc"][m][k].mean()) for m in a42.METHODS}}
        if s != "all":                                # "matched": fill saying yes on as many questions as b64 did in this split
            n_yes = int((p["said"]["b64"][k] == "yes").sum())
            t = np.sort(lo[k])[::-1][n_yes - 1] if n_yes else np.inf     # the n_yes highest log-odds say yes
            said = np.where(lo[k] >= t, "yes", "no")
            r.update({"threshold": float(t), "matched: yes ratio": float(np.mean(said == "yes")),
                      "matched: accuracy": 100 * float(np.mean(said == gt))})
            matched[k] = said
            covered += k
        res["S3"][s] = r
    CHECKS["pope: the three splits cover every question once"] = float(np.all(covered == 1))
    p["sc"]["matched"] = (matched == p["gt"]).astype(float)
    res["S3"]["matched - b64"] = a36.paired(p, "matched", "b64")
    res["S3"]["matched - b64 gt yes"] = a36.paired(p, "matched", "b64", p["gt"] == "yes")
    res["S3"]["matched - b64 gt no"] = a36.paired(p, "matched", "b64", p["gt"] == "no")
    for s in SPLITS:
        res["S3"][s]["matched - b64"] = a36.paired(p, "matched", "b64", np.array([x == s for x in p["split"]]))
    # S4: MME subtasks
    for meth in a42.METHODS:
        m = json.load(open(f"{a42.run_dir('mme', meth)}/metrics.json", encoding="utf-8"))
        res["S4"][meth] = {**m["perception_tasks"], **m["cognition_tasks"]}
    res["S4"]["fill - b64"] = {t: res["S4"]["fill"][t] - res["S4"]["b64"][t] for t in res["S4"]["fill"]}
    # S5: development against fresh sets
    for (db, fb) in (("vqav2_dev2", "vqav2_test6"), ("gqa_val400", "gqa_val400d")):
        for kd, v in dev["pairs"].items():
            if kd.startswith(f"{db} fill - b64"):
                kf = kd.replace(db, fb, 1)
                res["S5"][kd.replace(db, fb.split("_")[0], 1)] = {"dev": v, "fresh": sealed["pairs"].get(kf)}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    res["checks"] = dict(CHECKS)
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/squeeze_0042.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    fmt = lambda x, k=2: "—" if x is None else f"{x[0]:+.{k}f} [{x[1]:+.{k}f}, {x[2]:+.{k}f}]（{x[3]}）"   # noqa: E731
    L = ["# SQZ-EXP-0042（事后探索，不是预注册）：虚拟填补为什么在 POPE 上比 CDPruner-64 差", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    L += ["", "## S1：与不剪枝的回答相同的比例（POPE、MME：计分的是 / 否；新数据：答案文字）", ""]
    for bench, x in res["S1"].items():
        for k, y in x.items():
            L.append(f"- {bench} {k}（{y['n']} 题）：fill {y['fill']:.3f}；b64 {y['b64']:.3f}；fill − b64（百分点）{fmt(y['fill - b64'])}")
    L += ["", "## S2：对不剪枝的差（百分点）", ""]
    for g, y in res["S2"]["pope"].items():
        L.append(f"- pope {g}：b64 − 不剪枝 {fmt(y['b64 - vanilla'])}；fill − 不剪枝 {fmt(y['fill - vanilla'])}；fill − b64 {fmt(y['fill - b64'])}")
    for bench in ("mme",) + GENERAL:
        L.append(f"- {bench}：b64 − 不剪枝 {fmt(res['S2'][bench]['b64 - vanilla'])}")
    s3 = res["S3"]
    L += ["", "## S3：fill 在 POPE 上第一个答案 token 的对数几率", "", f"- 对数几率的正负与计分的是 / 否相同：{s3['sign = scored label']:.4f}"]
    for s in SPLITS + ("all",):
        y = s3[s]
        line = (f"- {s}（{y['n']} 题）：AUC {y['auc']:.4f}；答“是”的比例 " + "；".join(f"{m} {v:.3f}" for m, v in y["yes ratio"].items())
                + "；准确率 " + "；".join(f"{m} {v:.2f}" for m, v in y["accuracy"].items()))
        if s != "all":
            line += (f"；fill 答“是”的题数与 b64 相同时（阈值 {y['threshold']:+.3f}）：答“是” {y['matched: yes ratio']:.3f}、"
                     f"准确率 {y['matched: accuracy']:.2f}，减 b64 {fmt(y['matched - b64'])}")
        L.append(line)
    L += [f"- 三个分集合起来：fill（答“是”的题数与 b64 相同）− b64 {fmt(s3['matched - b64'])}；"
          f"gt yes {fmt(s3['matched - b64 gt yes'])}；gt no {fmt(s3['matched - b64 gt no'])}"]
    L += ["", "## S4：MME 各子任务的分数（acc + acc+）", "", "| 子任务 | fill | b64 | 不剪枝 | fill − b64 |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {t} | {res['S4']['fill'][t]:.2f} | {res['S4']['b64'][t]:.2f} | {res['S4']['vanilla'][t]:.2f} | {res['S4']['fill - b64'][t]:+.2f} |" for t in res["S4"]["fill"]]
    L += ["", "## S5：fill − b64，开发集（EXP-0041）与新数据（EXP-0042）", "", "| 比较 | 开发集 | 新数据 |", "| --- | --- | --- |"]
    L += [f"| {k} | {fmt(v['dev'])} | {fmt(v['fresh'])} |" for k, v in res["S5"].items()]
    open(f"{DST}/squeeze_0042.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {DST}; {len(CHECKS)} checks")


if __name__ == "__main__":
    main()
