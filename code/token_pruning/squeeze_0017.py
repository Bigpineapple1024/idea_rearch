"""EXP-0017 squeeze (written late, in the drift check of 2026-10-25; exploratory): what changes between A and A'+B+V at the
192 budget, next to 64 (EXP-0015 R03 vs EXP-0011 R07 / EXP-0014 R04) and 128 (EXP-0016). Reads only metrics.json and
kept_tokens.json of sealed runs; no model calls.

  python squeeze_0017.py      # writes code/output/EXP-0017/squeeze/{squeeze_0017.md,squeeze_0017.json}
"""
import json
import os
import sys

import numpy as np

OUT = "D:/6CJNM007/idea_rearch/code/output"
POPE = {64: ("EXP-0011/EXP-0011-R07", "EXP-0015/EXP-0015-R03"), 128: ("EXP-0016/EXP-0016-R02", "EXP-0016/EXP-0016-R03"),
        192: ("EXP-0017/EXP-0017-R03", "EXP-0017/EXP-0017-R04")}
MME = {64: ("EXP-0014/EXP-0014-R04", "EXP-0015/EXP-0015-R02"), 128: ("EXP-0016/EXP-0016-R04", "EXP-0016/EXP-0016-R05"),
       192: ("EXP-0017/EXP-0017-R01", "EXP-0017/EXP-0017-R02")}
VANILLA = {"pope": "EXP-0002/EXP-0002-R04", "mme": "EXP-0002/EXP-0002-R02"}
KEEP = {64: 30, 128: 98, 192: 166}


def m(run):
    return json.load(open(f"{OUT}/{run}/metrics.json"))


def main():
    out_dir = f"{OUT}/EXP-0017/squeeze"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    res = {"pope": {}, "mme": {}, "query_share": {}}
    v = m(VANILLA["pope"])["splits"]
    res["pope"]["vanilla"] = {s: {k: v[s][k] for k in ("f1", "recall", "precision", "yes_ratio", "FP", "FN")} for s in v}
    for b, (a, q) in POPE.items():
        ma, mq = m(a)["splits"], m(q)["splits"]
        res["pope"][b] = {s: {k: {"A": ma[s][k], "AV": mq[s][k]} for k in ("f1", "recall", "precision", "yes_ratio", "FP", "FN")} for s in ma}
        k = json.load(open(f"{OUT}/{q}/kept_tokens.json"))
        nq = np.array([len(x) for x in k["kept_by_object_name"]])
        res["query_share"][b] = {"mean_n_q": float(nq.mean()), "share_of_keep": float(nq.mean() / KEEP[b])}
    vm = m(VANILLA["mme"])["perception_tasks"]
    for b, (a, q) in MME.items():
        ta, tq = m(a)["perception_tasks"], m(q)["perception_tasks"]
        res["mme"][b] = {t: {"A": ta[t], "AV": tq[t], "diff": tq[t] - ta[t], "vanilla": vm[t]} for t in ta}
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/squeeze_0017.json", "w"), indent=2)
    L = ["# EXP-0017 squeeze（跑偏检查时补做，2026-10-25；探索性，只读已封存的 metrics.json 与 kept_tokens.json）", "",
         "## 1. POPE：按子集的 F1、召回、yes 比例、误报（A → A′+B+V）", "",
         "| 预算 | 子集 | F1 | 召回 | 精确率 | yes 比例 | 误报 FP | 漏报 FN |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for b in (64, 128, 192):
        for s, r in res["pope"][b].items():
            f = lambda key, d=3: f"{r[key]['A']:.{d}f} → {r[key]['AV']:.{d}f}"   # noqa: E731
            L.append(f"| {b} | {s} | {f('f1')} | {f('recall')} | {f('precision')} | {f('yes_ratio')} | {r['FP']['A']} → {r['FP']['AV']} | {r['FN']['A']} → {r['FN']['AV']} |")
    L += [""] + [f"- vanilla {s}：F1 {r['f1']:.3f}，召回 {r['recall']:.3f}，yes 比例 {r['yes_ratio']:.3f}，FP {r['FP']}，FN {r['FN']}" for s, r in res["pope"]["vanilla"].items()]
    L += ["", "## 2. 题目查询的名额（POPE，A′+B+V）", ""]
    L += [f"- {b} 档：平均 n_q {r['mean_n_q']:.1f}，占保留数 {r['share_of_keep']:.2f}" for b, r in res["query_share"].items()]
    L += ["", "## 3. MME 感知子任务：A′+B+V − A（三档）与 vanilla", "", "| 子任务 | 64 | 128 | 192 | 192 档 A / A′+B+V | vanilla |", "| --- | --- | --- | --- | --- | --- |"]
    for t in res["mme"][192]:
        L.append(f"| {t} | {res['mme'][64][t]['diff']:+.1f} | {res['mme'][128][t]['diff']:+.1f} | {res['mme'][192][t]['diff']:+.1f} | "
                 f"{res['mme'][192][t]['A']:.1f} / {res['mme'][192][t]['AV']:.1f} | {res['mme'][192][t]['vanilla']:.1f} |")
    open(f"{out_dir}/squeeze_0017.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
