"""EXP-0020 squeeze (exploratory, after the pre-registered analysis): where the router's gain comes from on test2,
whether EXP-0018's per-question-type pattern replicates from test1 to test2, and where the gap to VisionZip-64 lies.

  python squeeze_0020.py      # writes code/output/EXP-0020/squeeze/{squeeze_0020.md,squeeze_0020.json}

Reads only sealed run outputs (answers, vqa_scores, kept_tokens, runs.tsv) and the subset annotations; no model calls.
Paired differences are resampled over images (1000 draws, seed 0), as in analyze_0020.py. Everything here is post hoc:
a pattern found here is a lead for later work, not a result of EXP-0020.
"""
import json
import os
import sys

import numpy as np

REPO = "D:/6CJNM007/idea_rearch"
OUT = f"{REPO}/code/output"
VQA = "D:/6CJNM007/data/llava_eval/vqav2_val"
RUNS = {"A_64": "R01", "router_64": "R02", "AV_64": "R03", "A_128": "R04", "router_128": "R05", "AV_128": "R06",
        "vanilla": "R07", "vz64": "R08", "sv64": "R09"}
RUNS_TEST1 = {"A_64": "R02", "AV_64": "R03", "vz64": "R05", "A_128": "R07", "AV_128": "R08", "vanilla": "R09"}   # EXP-0018
ROUTE_START = ("is ", "are ", "does ", "do ", "did ", "can ", "could ", "was ", "were ", "has ", "have ", "had ", "will ",
               "would ", "should ", "shall ", "may ", "might ", "must ", "how many ")      # = run_eval.ROUTE_START
TYPES = ("yes/no", "number", "other")
N_BOOT = 1000


def load_set(exp, runs, ann, qfile):
    gts = {a["question_id"]: a for a in map(json.loads, open(f"{VQA}/{ann}"))}
    qtext = {q["question_id"]: q["text"].split("\n")[0].strip().lower() for q in map(json.loads, open(f"{VQA}/{qfile}"))}
    qids = sorted(gts)
    images = sorted({g["image"] for g in gts.values()})
    img_idx = {im: k for k, im in enumerate(images)}
    s = {"qids": qids, "gts": gts, "qtext": qtext, "img_of": np.array([img_idx[gts[q]["image"]] for q in qids]),
         "n_images": len(images), "atype": np.array([gts[q]["answer_type"] for q in qids]),
         "qtype": np.array([gts[q]["question_type"] for q in qids]), "scores": {}, "answers": {}, "kept": {}}
    for name, rid in runs.items():
        d = f"{OUT}/{exp}/{exp}-{rid}"
        sc = {r["question_id"]: r["score"] for r in map(json.loads, open(f"{d}/vqa_scores.jsonl"))}
        rows = [json.loads(l) for l in open(f"{d}/answers.jsonl")]
        an = {r["question_id"]: r["text"] for r in rows}
        s["scores"][name] = np.array([sc[q] for q in qids])
        s["answers"][name] = [an[q] for q in qids]
        if os.path.exists(f"{d}/kept_tokens.json"):
            k = json.load(open(f"{d}/kept_tokens.json"))
            pos = {r["question_id"]: i for i, r in enumerate(rows)}
            s["kept"][name] = {"n_prefills": k["n_prefills"], **{f: [k[f][pos[q]] for q in qids] for f in ("kept", "routed") if f in k}}
    by_img = [np.flatnonzero(s["img_of"] == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    s["draws"] = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]
    return s


def paired(s, a, b, mask=None):
    d = s["scores"][a] - s["scores"][b]
    m = np.ones(len(d), bool) if mask is None else mask
    if m.sum() == 0:
        return None
    boot = [100 * d[idx][m[idx]].mean() for idx in s["draws"] if m[idx].any()]
    return {"n": int(m.sum()), "diff": 100 * float(d[m].mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def fmt(r):
    return f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]" if r else "—"


def majority(answers):
    vals, counts = np.unique(answers, return_counts=True)
    return vals[np.argmax(counts)]


def main():
    out_dir = f"{OUT}/EXP-0020/squeeze"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    t2 = load_set("EXP-0020", RUNS, "llava_vqav2_val_test2_annotations.jsonl", "llava_vqav2_val_test2.jsonl")
    t1 = load_set("EXP-0018", RUNS_TEST1, "llava_vqav2_val_test_annotations.jsonl", "llava_vqav2_val_test.jsonl")
    n = len(t2["qids"])
    atype, qtype = t2["atype"], t2["qtype"]
    route = np.array([t2["qtext"][q].startswith(ROUTE_START) for q in t2["qids"]])
    checks = {f"test2 {k}: one prefill per question": float(v["n_prefills"] == n) for k, v in t2["kept"].items()}
    checks.update({f"test1 {k}: one prefill per question": float(v["n_prefills"] == len(t1["qids"])) for k, v in t1["kept"].items()})
    for b in (64, 128):
        checks[f"router_{b}: logged decision = rule on the question text"] = float(np.mean(np.array(t2["kept"][f"router_{b}"]["routed"]) == route))
        checks[f"router_{b}: score = AV's when routed, A's otherwise"] = float(np.allclose(
            t2["scores"][f"router_{b}"], np.where(route, t2["scores"][f"AV_{b}"], t2["scores"][f"A_{b}"])))
    res = {"checks": checks, "n_questions": n}

    # 1. decomposition: router - A = (routed share) x (AV - A on the routed questions); what the rule leaves out
    dec = {}
    for b in (64, 128):
        rows = []
        for label, m in (("routed", route), ("not routed", ~route)):
            for t in ("all",) + TYPES:
                mm = m if t == "all" else m & (atype == t)
                r = paired(t2, f"AV_{b}", f"A_{b}", mm)
                if r:
                    rows.append({"subset": label, "type": t, **r, "share_of_all": float(mm.mean()),
                                 "contribution": float(mm.mean()) * r["diff"]})
        dec[str(b)] = rows
    res["decomposition"] = dec

    # 2. routed questions by their first word (or "how many")
    starts = []
    for p in ROUTE_START:
        m = np.array([t2["qtext"][q].startswith(p) for q in t2["qids"]])
        if m.sum() >= 20:
            starts.append({"start": p.strip(), "n": int(m.sum()), "yesno_share": float((atype[m] == "yes/no").mean()),
                           "64": paired(t2, "AV_64", "A_64", m), "128": paired(t2, "AV_128", "A_128", m)})
    res["by_start"] = starts

    # 3. question types with >= 40 questions on test2; replication of EXP-0018's per-type AV - A from test1
    qt = []
    for t in sorted(set(qtype)):
        m = qtype == t
        if m.sum() < 40:
            continue
        m1 = t1["qtype"] == t
        d1 = 100 * float((t1["scores"]["AV_64"] - t1["scores"]["A_64"])[m1].mean()) if m1.sum() >= 40 else None
        qt.append({"question_type": t, "answer_type": max(set(atype[m]), key=list(atype[m]).count), "n": int(m.sum()),
                   "routed_share": float(route[m].mean()), "AV_minus_A": paired(t2, "AV_64", "A_64", m),
                   "router_minus_A": paired(t2, "router_64", "A_64", m), "vz_minus_A": paired(t2, "vz64", "A_64", m),
                   "router_minus_vz": paired(t2, "router_64", "vz64", m), "A_acc": 100 * float(t2["scores"]["A_64"][m].mean()),
                   "vanilla_acc": 100 * float(t2["scores"]["vanilla"][m].mean()), "test1_n": int(m1.sum()), "test1_AV_minus_A": d1})
    res["by_question_type_64"] = sorted(qt, key=lambda r: r["AV_minus_A"]["diff"])
    both = [r for r in qt if r["test1_AV_minus_A"] is not None]
    x1 = np.array([r["test1_AV_minus_A"] for r in both])
    x2 = np.array([r["AV_minus_A"]["diff"] for r in both])
    big = np.abs(x1) >= 3
    res["replication_test1_test2"] = {
        "n_types": len(both), "pearson": float(np.corrcoef(x1, x2)[0, 1]), "spearman": float(np.corrcoef(ranks(x1), ranks(x2))[0, 1]),
        "same_sign": float(np.mean(np.sign(x1) == np.sign(x2))), "n_types_test1_abs_ge_3": int(big.sum()),
        "same_sign_when_test1_abs_ge_3": float(np.mean(np.sign(x1[big]) == np.sign(x2[big]))) if big.any() else None}
    res["test1_vs_test2_answer_type"] = {
        t: {"test1": paired(t1, "AV_64", "A_64", t1["atype"] == t), "test2": paired(t2, "AV_64", "A_64", atype == t)} for t in TYPES}

    # 4. the gap to VisionZip-64: share of the total gap by answer type; the question types that lose most
    gap = paired(t2, "router_64", "vz64")
    res["gap_to_vz64"] = {"total": gap, "by_type": {t: {**paired(t2, "router_64", "vz64", atype == t),
                                                        "contribution": float((atype == t).mean()) * paired(t2, "router_64", "vz64", atype == t)["diff"]}
                                                    for t in TYPES},
                          "A_minus_vz_other": paired(t2, "A_64", "vz64", atype == "other"),
                          "vanilla_minus_vz_other": paired(t2, "vanilla", "vz64", atype == "other")}

    # 5. the per-question oracle of two methods (an upper bound); compared with the same bound for other pairs
    orc = {}
    for x, y in (("A_64", "AV_64"), ("A_64", "vz64"), ("A_64", "sv64"), ("vz64", "sv64"), ("A_128", "AV_128"), ("A_64", "A_128")):
        sx, sy = t2["scores"][x], t2["scores"][y]
        orc[f"{x} | {y}"] = {"oracle_minus_first": 100 * float((np.maximum(sx, sy) - sx).mean()),
                             "differ": float((sx != sy).mean()), "second_better": float((sy > sx).mean()), "first_better": float((sx > sy).mean())}
    for b in (64, 128):
        sa, sav = t2["scores"][f"A_{b}"], t2["scores"][f"AV_{b}"]
        for label, m in (("routed", route), ("not routed", ~route)):
            orc[f"A_{b} | AV_{b}, {label}"] = {"oracle_minus_first": 100 * float(((np.maximum(sa, sav) - sa) * m).sum() / n),
                                             "differ": float((sa != sav)[m].mean()), "second_better": float((sav > sa)[m].mean()),
                                             "first_better": float((sa > sav)[m].mean())}
    res["oracle"] = orc

    # 6. yes/no questions: how often each method answers "yes", and accuracy split by the majority human answer
    yn = atype == "yes/no"
    maj = np.array([majority(t2["gts"][q]["answers"]) for q in t2["qids"]])
    gy, gn = yn & (maj == "yes"), yn & (maj == "no")
    bias = {"n_majority_yes": int(gy.sum()), "n_majority_no": int(gn.sum())}
    for name in ("vanilla", "A_64", "AV_64", "router_64", "vz64", "sv64", "A_128", "router_128"):
        ans = np.array([a.strip().lower().rstrip(".") for a in t2["answers"][name]])
        s = t2["scores"][name]
        bias[name] = {"yes_rate": float((ans[yn] == "yes").mean()), "acc_majority_yes": 100 * float(s[gy].mean()),
                      "acc_majority_no": 100 * float(s[gn].mean())}
    bias["router_64 - A_64 on majority-no"] = paired(t2, "router_64", "A_64", gn)
    bias["router_64 - A_64 on majority-yes"] = paired(t2, "router_64", "A_64", gy)
    res["yesno_bias"] = bias

    # 7. how many answers the router changes, and the cost of each run (runs.tsv)
    same = np.array([a == b for a, b in zip(t2["answers"]["router_64"], t2["answers"]["A_64"])])
    res["answer_changed_64"] = {"all": float(1 - same.mean()), "routed": float(1 - same[route].mean()), "not routed": float(1 - same[~route].mean())}
    tsv = [l.rstrip("\n").split("\t") for l in open(f"{OUT}/EXP-0020/runs.tsv")]
    head = tsv[0]
    res["cost"] = {r[0]: {"config": r[1], "s_per_answer": float(r[head.index("s_per_answer")]), "peak_gib": float(r[head.index("peak_gib")])}
                   for r in tsv[1:]}

    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/squeeze_0020.json", "w"), indent=2)
    L = [f"# EXP-0020 squeeze（探索性，事后分析；test2 {n} 题）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    for b in (64, 128):
        L += ["", f"## 1. 分解（{b} 档）：路由 − A = 路由题上的 A′+B+V − A × 路由比例", "",
              "| 子集 | 类型 | 题数 | 占全部 | A′+B+V − A [95%] | 对总差的贡献 |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r['subset']} | {r['type']} | {r['n']} | {r['share_of_all']:.3f} | {fmt(r)} | {r['contribution']:+.2f} |" for r in dec[str(b)]]
    L += ["", "## 2. 被路由的题按开头词（≥20 题）", "", "| 开头 | 题数 | yes/no 占比 | 64：A′+B+V − A | 128：A′+B+V − A |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {r['start']} | {r['n']} | {r['yesno_share']:.2f} | {fmt(r['64'])} | {fmt(r['128'])} |" for r in starts]
    rep = res["replication_test1_test2"]
    L += ["", "## 3. 按题型（test2 ≥40 题，64 档，按 A′+B+V − A 排序）", "",
          f"test1→test2 复验（两边都 ≥40 题的 {rep['n_types']} 个题型，A′+B+V − A）：Pearson {rep['pearson']:.2f}，Spearman {rep['spearman']:.2f}，"
          f"同号 {rep['same_sign']:.2f}；test1 上 |差| ≥3 的 {rep['n_types_test1_abs_ge_3']} 个题型同号 {rep['same_sign_when_test1_abs_ge_3']:.2f}。", "",
          "| 题型 | 答案类型 | 题数 | 路由比例 | A′+B+V − A | test1 的 A′+B+V − A | 路由 − A | VisionZip − A | 路由 − VisionZip | A | vanilla |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["by_question_type_64"]:
        t1d = f"{r['test1_AV_minus_A']:+.2f}" if r["test1_AV_minus_A"] is not None else "—"
        L.append(f"| {r['question_type']} | {r['answer_type']} | {r['n']} | {r['routed_share']:.2f} | {fmt(r['AV_minus_A'])} | {t1d} | "
                 f"{r['router_minus_A']['diff']:+.2f} | {r['vz_minus_A']['diff']:+.2f} | {r['router_minus_vz']['diff']:+.2f} | {r['A_acc']:.1f} | {r['vanilla_acc']:.1f} |")
    L += ["", "按答案类型的 test1 / test2（64 档 A′+B+V − A）：" + "；".join(
        f"{t} {fmt(v['test1'])} / {fmt(v['test2'])}" for t, v in res["test1_vs_test2_answer_type"].items())]
    g = res["gap_to_vz64"]
    L += ["", "## 4. 与 VisionZip-64 的差距", "", f"- 路由-64 − VisionZip-64：{fmt(g['total'])}"]
    L += [f"- {t}：{fmt(v)}，对总差的贡献 {v['contribution']:+.2f}" for t, v in g["by_type"].items()]
    L += [f"- other 上 A-64 − VisionZip-64：{fmt(g['A_minus_vz_other'])}；vanilla − VisionZip-64：{fmt(g['vanilla_minus_vz_other'])}"]
    L += ["", "## 5. 逐题 oracle（两种方法每题取较好者）", "", "| 两种方法 | oracle − 前者 | 得分不同 | 后者更好 | 前者更好 |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {k} | {v['oracle_minus_first']:+.2f} | {v['differ']:.3f} | {v['second_better']:.3f} | {v['first_better']:.3f} |" for k, v in orc.items()]
    L += ["", "（“routed / not routed” 两行的 oracle 列是该子集对全部题平均的贡献，两行相加等于整体。）"]
    L += ["", f"## 6. yes/no 题（多数人答 yes {bias['n_majority_yes']} 题，答 no {bias['n_majority_no']} 题）", "",
          "| 方法 | 答 yes 的比例 | 多数答 yes 的题上准确率 | 多数答 no 的题上准确率 |", "| --- | --- | --- | --- |"]
    L += [f"| {k} | {v['yes_rate']:.3f} | {v['acc_majority_yes']:.2f} | {v['acc_majority_no']:.2f} |" for k, v in bias.items() if isinstance(v, dict) and "yes_rate" in v]
    L += [f"- 路由-64 − A-64，多数答 no 的题：{fmt(bias['router_64 - A_64 on majority-no'])}；多数答 yes 的题：{fmt(bias['router_64 - A_64 on majority-yes'])}"]
    a = res["answer_changed_64"]
    L += ["", "## 7. 答案改变与成本", "", f"- 64 档路由与 A 答案不同的比例：全部 {a['all']:.3f}，路由题 {a['routed']:.3f}，不路由题 {a['not routed']:.3f}", "",
          "| 运行 | 配置 | 秒/题 | 峰值显存 GiB |", "| --- | --- | --- | --- |"]
    L += [f"| {k} | {v['config']} | {v['s_per_answer']:.4f} | {v['peak_gib']:.2f} |" for k, v in res["cost"].items()]
    open(f"{out_dir}/squeeze_0020.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
