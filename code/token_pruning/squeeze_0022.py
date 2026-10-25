"""SQZ-EXP-0022 (exploratory, written after the preregistered analysis 0022.md was seen): where the router's GQA gain
comes from, the either/or questions, GQA's semantic and detailed types, the yes/no bias, VisionZip-64 and the post-hoc
splice of EXP-0020 S4, the 128 budget, and the comparison with VQAv2 test2 (EXP-0020). Nothing here is a criterion.

  python -B squeeze_0022.py      # writes code/output/EXP-0022/squeeze/{squeeze_0022.md,squeeze_0022.json}

Per-question scores are recomputed from each run's answers.jsonl exactly as in analyze_0022.py and checked against the
sealed analysis (0022.json). Paired differences: images resampled with replacement 1000 times (seed 0), the same draws as
analyze_0022.py, so a pair computed in both has the same interval.
"""
import collections
import json
import os
import re
import sys

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_repope import EVAL_DATA, OUT   # noqa: E402

RUNS = {"A_64": "R01", "router_64": "R02", "A'+B+V_64": "R09", "A_128": "R04", "router_128": "R05",
        "vanilla": "R06", "visionzip64": "R07", "sparsevlm64": "R08"}
SEMANTIC = ("obj", "attr", "rel", "cat", "global")
STRUCT = ("verify", "logical", "choose", "compare", "query")
N_BOOT = 1000
MIN_DETAILED = 150


def fmt(p):
    return f"{p[0]:+.2f} [{p[1]:+.2f}, {p[2]:+.2f}]" if p else "—"


def main():
    out_dir = f"{OUT}/EXP-0022/squeeze"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sys.path.insert(0, "D:/6CJNM007/idea_rearch/source/LLaVA")
    import run_eval
    sealed = json.load(open(f"{OUT}/EXP-0022/analysis/0022.json"))
    qfile = f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced.jsonl"
    questions = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    gts = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced_annotations.jsonl", encoding="utf-8")}
    text = {str(q["question_id"]): q["text"] for q in questions}
    qids = sorted(gts)
    images = sorted({g["image"] for g in gts.values()})
    img_idx = {im: k for k, im in enumerate(images)}
    img_of = np.array([img_idx[gts[q]["image"]] for q in qids])
    gold = np.array([gts[q]["answer"] for q in qids])
    yn = np.isin(gold, ("yes", "no"))
    struct = np.array([gts[q]["structural"] for q in qids])
    sem = np.array([gts[q]["semantic"] for q in qids])
    det = np.array([gts[q]["detailed"] for q in qids])
    first = [text[q].split("\n")[0].lower() for q in qids]
    either_or = np.array([bool(re.search(r"\bor\b", f)) for f in first])
    routed = np.array([run_eval.routes_to_query(text[q]) for q in qids])
    head = np.array([f.split()[0] if f.split() else "" for f in first])
    scores, answers = {}, {}
    for name, rid in RUNS.items():
        run = f"{OUT}/EXP-0022/EXP-0022-{rid}"
        ans = {str(json.loads(l)["question_id"]): json.loads(l)["text"] for l in open(f"{run}/answers.jsonl")}
        answers[name] = np.array([ans[q].rstrip(".").lower() for q in qids])
        scores[name] = (answers[name] == gold).astype(float)
    checks = {f"{n}: accuracy equals the sealed analysis": float(abs(100 * scores[n].mean() - sealed["table"][n]["accuracy"]) < 1e-9)
              for n in RUNS}
    checks["routed share equals the sealed analysis"] = float(abs(routed.mean() - sealed["routed_share"]) < 1e-12)
    if any(v < 1 for v in checks.values()):
        print("STOP", checks)
        sys.exit(1)

    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]

    def paired(d, mask=None):
        m = np.ones(len(d), bool) if mask is None else mask
        if m.sum() == 0:
            return None
        boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

    def diff(x, y):
        return scores[x] - scores[y]

    check_q1 = paired(diff("router_64", "A_64"))
    checks["router_64 - A_64 interval equals the sealed analysis"] = float(np.allclose(check_q1[:3], sealed["pairs"]["router_64 - A_64"][:3]))
    if checks["router_64 - A_64 interval equals the sealed analysis"] < 1:
        print("STOP", checks)
        sys.exit(1)
    res = {"checks": checks}
    L = ["# SQZ-EXP-0022（事后探索，不是判据）：GQA testdev-balanced", "",
         "所有区间：按图像重抽 1000 次（种子 0），与 analyze_0022.py 同一组重抽。", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in checks.items()]

    # §1 decomposition of router_64 - A_64 = sum over groups of share x (A'+B+V - A) on the routed groups
    d_av = diff("A'+B+V_64", "A_64")
    groups = {"routed, gold yes/no, either/or": routed & yn & either_or,
              "routed, gold yes/no, not either/or": routed & yn & ~either_or,
              "routed, gold other, either/or": routed & ~yn & either_or,
              "routed, gold other, not either/or": routed & ~yn & ~either_or,
              "unrouted (router = A)": ~routed}
    dec = {}
    for g, m in groups.items():
        p = paired(d_av, m) if m.any() else None
        share = float(m.mean())
        dec[g] = {"n": int(m.sum()), "share": share, "A'+B+V - A": p,
                  "contribution_to_router-A": (share * p[0] if (p and g.startswith("routed")) else 0.0)}
    dec["routed (all)"] = {"n": int(routed.sum()), "share": float(routed.mean()), "A'+B+V - A": paired(d_av, routed)}
    res["decomposition_64"] = dec
    L += ["", "## 1. 路由-64 − A-64 的分解（= 各组比例 × 该组上 A′+B+V − A；不路由的组上路由 = A，贡献为 0，"
              "那一行的差是“若不路由也用 A′+B+V”的损失）", "",
          "| 组 | 题数 | 比例 | A′+B+V − A [95% 区间] | 对总差的贡献 |", "| --- | --- | --- | --- | --- |"]
    for g, v in dec.items():
        c = v.get("contribution_to_router-A")
        L.append(f"| {g} | {v['n']} | {v['share']:.3f} | " + fmt(v["A'+B+V - A"]) + " | "
                 + (f"{c:+.2f}" if c is not None else "") + " |")

    # §2 either/or questions: by semantic type and by detailed type
    eo = routed & either_or
    tab2 = {}
    for s in SEMANTIC:
        m = eo & (sem == s)
        if m.sum() >= 30:
            tab2[f"semantic {s}"] = paired(d_av, m)
    for t, c in collections.Counter(det[eo]).most_common():
        if c >= 50:
            tab2[f"detailed {t}"] = paired(d_av, eo & (det == t))
    res["either_or"] = tab2
    L += ["", f"## 2. 被路由的二选一题（{int(eo.sum())} 题）：A′+B+V − A 按语义类型与细类（≥ 30 / 50 题）", "",
          "| 组 | 差 [95% 区间] | 题数 |", "| --- | --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] | {v[3]} |" for k, v in tab2.items()]

    # §3 by semantic type (all questions)
    tab3 = {}
    for s in SEMANTIC:
        m = sem == s
        tab3[s] = {"n": int(m.sum()), "routed_share": float(routed[m].mean()), "yes/no_share": float(yn[m].mean()),
                   "A_64": 100 * float(scores["A_64"][m].mean()), "router_64": 100 * float(scores["router_64"][m].mean()),
                   "visionzip64": 100 * float(scores["visionzip64"][m].mean()), "vanilla": 100 * float(scores["vanilla"][m].mean()),
                   "router - A": paired(diff("router_64", "A_64"), m), "A'+B+V - A": paired(d_av, m),
                   "A'+B+V - A [unrouted]": paired(d_av, m & ~routed), "router - visionzip64": paired(diff("router_64", "visionzip64"), m)}
    res["semantic"] = tab3
    L += ["", "## 3. 按语义类型（全部题）", "",
          "| 类型 | 题数 | 路由比例 | A-64 | 路由-64 | VisionZip-64 | vanilla | 路由 − A | A′+B+V − A | A′+B+V − A（不路由题） | 路由 − VisionZip |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s, v in tab3.items():
        cells = [fmt(v[k]) for k in ("router - A", "A'+B+V - A", "A'+B+V - A [unrouted]", "router - visionzip64")]
        L.append(f"| {s} | {v['n']} | {v['routed_share']:.2f} | {v['A_64']:.2f} | {v['router_64']:.2f} | {v['visionzip64']:.2f} | "
                 f"{v['vanilla']:.2f} | " + " | ".join(cells) + " |")

    # §3b detailed types with at least MIN_DETAILED questions
    tab3b = {}
    for t, c in collections.Counter(det).most_common():
        if c < MIN_DETAILED:
            continue
        m = det == t
        tab3b[t] = {"n": int(c), "routed_share": float(routed[m].mean()), "A_64": 100 * float(scores["A_64"][m].mean()),
                    "router - A": paired(diff("router_64", "A_64"), m), "A'+B+V - A": paired(d_av, m),
                    "router - visionzip64": paired(diff("router_64", "visionzip64"), m),
                    "vanilla - A": paired(diff("vanilla", "A_64"), m)}
    res["detailed"] = tab3b
    L += ["", f"## 3b. 细类（≥ {MIN_DETAILED} 题；事后分组，多重比较未校正）", "",
          "| 细类 | 题数 | 路由比例 | A-64 | 路由 − A | A′+B+V − A | 路由 − VisionZip | vanilla − A |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t, v in tab3b.items():
        cells = [fmt(v[k]) for k in ("router - A", "A'+B+V - A", "router - visionzip64", "vanilla - A")]
        L.append(f"| {t} | {v['n']} | {v['routed_share']:.2f} | {v['A_64']:.2f} | " + " | ".join(cells) + " |")

    # §4 the yes/no bias: accuracy on gold-yes and gold-no questions
    gy, gn = gold == "yes", gold == "no"
    tab4 = {n: {"yes_rate": float(np.mean(answers[n][yn] == "yes")), "acc_gold_yes": 100 * float(scores[n][gy].mean()),
                "acc_gold_no": 100 * float(scores[n][gn].mean())} for n in RUNS}
    tab4["router_64 - A_64 [gold yes]"] = paired(diff("router_64", "A_64"), gy)
    tab4["router_64 - A_64 [gold no]"] = paired(diff("router_64", "A_64"), gn)
    tab4["router_128 - A_128 [gold yes]"] = paired(diff("router_128", "A_128"), gy)
    tab4["router_128 - A_128 [gold no]"] = paired(diff("router_128", "A_128"), gn)
    tab4["n_gold_yes"], tab4["n_gold_no"] = int(gy.sum()), int(gn.sum())
    res["yes_no_bias"] = tab4
    L += ["", f"## 4. yes/no 题的答案偏向（标准答案 yes {int(gy.sum())} 题、no {int(gn.sum())} 题）", "",
          "| 配置 | 答 yes 的比例 | 标准答案 yes 的题上准确率 | 标准答案 no 的题上准确率 |", "| --- | --- | --- | --- |"]
    L += [f"| {n} | {tab4[n]['yes_rate']:.3f} | {tab4[n]['acc_gold_yes']:.2f} | {tab4[n]['acc_gold_no']:.2f} |" for n in RUNS]
    for k in ("router_64 - A_64 [gold yes]", "router_64 - A_64 [gold no]", "router_128 - A_128 [gold yes]", "router_128 - A_128 [gold no]"):
        v = tab4[k]
        L.append(f"\n- {k}: {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]} 题）")

    # §5 VisionZip-64 and the post-hoc splice (routed questions -> A'+B+V-64, the rest -> VisionZip-64), as EXP-0020 S4
    splice = np.where(routed, scores["A'+B+V_64"], scores["visionzip64"])
    scores["splice(A'+B+V | VisionZip)"] = splice
    tab5 = {"splice accuracy": 100 * float(splice.mean())}
    for k, m in (("all", None), ("yes/no", yn), ("other", ~yn), ("routed", routed), ("unrouted", ~routed)):
        tab5[f"splice - visionzip64 [{k}]"] = paired(splice - scores["visionzip64"], m)
        tab5[f"router_64 - visionzip64 [{k}]"] = paired(diff("router_64", "visionzip64"), m)
    tab5["splice - router_64"] = paired(splice - scores["router_64"])
    for s in STRUCT:
        tab5[f"router_64 - visionzip64 [structural {s}]"] = paired(diff("router_64", "visionzip64"), struct == s)
    res["visionzip"] = tab5
    L += ["", "## 5. 与 VisionZip-64 的比较，以及事后拼接（被路由的题用 A′+B+V-64，其余用 VisionZip-64；不是运行过的方法）", "",
          f"拼接的准确率 {tab5['splice accuracy']:.2f}", "", "| 比较 | 差 [95% 区间] | 题数 |", "| --- | --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] | {v[3]} |" for k, v in tab5.items() if isinstance(v, list)]

    # §6 the 128 budget: router_128 - A_128 by group (unrouted is 0 by construction)
    d128 = diff("router_128", "A_128")
    tab6 = {"routed": paired(d128, routed), "routed, gold yes/no": paired(d128, routed & yn),
            "routed, gold other": paired(d128, routed & ~yn), "routed, either/or": paired(d128, routed & either_or),
            "unrouted (must be 0)": paired(d128, ~routed)}
    res["budget_128"] = tab6
    L += ["", "## 6. 128 档：路由 − A 按组（不路由的题按构造为 0）", "", "| 组 | 差 [95% 区间] | 题数 |", "| --- | --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] | {v[3]} |" for k, v in tab6.items()]

    # §7 how often the answers differ
    tab7 = {"router_64 vs A_64 answers differ [routed]": float(np.mean(answers["router_64"][routed] != answers["A_64"][routed])),
            "router_64 vs A_64 answers differ [unrouted]": float(np.mean(answers["router_64"][~routed] != answers["A_64"][~routed])),
            "router_128 vs A_128 answers differ [routed]": float(np.mean(answers["router_128"][routed] != answers["A_128"][routed])),
            "A'+B+V_64 vs A_64 answers differ [unrouted]": float(np.mean(answers["A'+B+V_64"][~routed] != answers["A_64"][~routed])),
            "routed heads": dict(collections.Counter(head[routed]).most_common(8))}
    res["agreement"] = tab7
    L += ["", "## 7. 答案不同的比例", ""] + [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in tab7.items()]

    # §8 GQA vs VQAv2 test2 (EXP-0020, sealed analysis)
    v20 = json.load(open(f"{OUT}/EXP-0020/analysis/0020.json"))
    keys = ["router_64 - A_64", "router_128 - A_128", "A'+B+V_64 - A_64", "A'+B+V_64 - A_64 [yes/no]", "A'+B+V_64 - A_64 [other]",
            "router_64 - visionzip64", "router_64 - sparsevlm64", "router_64 - vanilla", "A_64 - visionzip64"]
    tab8 = {k: {"VQAv2 test2": v20["pairs"].get(k), "GQA testdev": sealed["pairs"].get(k)} for k in keys}
    tab8["routed share"] = {"VQAv2 test2": v20["routed_share"], "GQA testdev": sealed["routed_share"]}
    res["vs_vqav2_test2"] = tab8
    L += ["", "## 8. 与 VQAv2 test2（EXP-0020）对照（两者的 other 不同：VQAv2 另有 number 类）", "",
          "| 比较 | VQAv2 test2 | GQA testdev |", "| --- | --- | --- |"]
    for k, v in tab8.items():
        if k == "routed share":
            L.append(f"| {k} | {v['VQAv2 test2']:.3f} | {v['GQA testdev']:.3f} |")
        else:
            L.append(f"| {k} | {fmt(v['VQAv2 test2'])} | {fmt(v['GQA testdev'])} |")
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/squeeze_0022.json", "w"), indent=2)
    open(f"{out_dir}/squeeze_0022.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
