"""SQZ-EXP-0033 补充（探索性，不是判据）：64 档的各种选法在不同题型上谁占优，按题目文字分题型再选方法能多得多少。
脚本写于 2026-10-30 晚、EXP-0033 运行中、看到任何 EXP-0033 结果之前（sha256 记在 squeeze/notes.md）；在预注册分析封存之后运行。
只读已封存的运行，CPU。

  python -B squeeze_0033_types.py     # 写 code/output/EXP-0033/squeeze_types/{types_0033.md,types_0033.json}；目录已存在就停

T1 各配置在每个题型上的准确率（VQAv2 dev2、GQA val400；分组同 analyze_0028.load_bench，另加只看题目文字的分组）。
T2 几对“铺开”与“聚焦”的配对差按题型：DivPrune-64 − VisionZip-64、K = 1 路由 + 覆盖 16 − VisionZip-64、DivPrune-64 − K = 1 路由 + 覆盖 16、
   X3 − DivPrune-64、每题覆盖 16 − VisionZip-64（区间：每个集内按图像重抽 1000 次，种子 0）。
T3 按题目文字分题型选方法的上限与交叉估计：文字分组 = 是非形式（被路由、不是 how many）、how many、what color、其余；候选是 64 档的 11 个配置。
   样本内：每组取最好的配置（乐观）。交叉：按图像随机分两半（20 次，种子 0～19），在一半上为每组、也为整体选最好的配置，在另一半上算
   “分组选” − “整体选一个”的差，取两个方向的平均；报告 20 次的均值与 2.5% / 97.5% 分位，以及各组最常被选中的配置。
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402
from squeeze_0018 import OUT  # noqa: E402

OUT_DIR = f"{OUT}/EXP-0033/squeeze_types"
N_BOOT = 1000
N_SPLIT = 20
BENCHES = ("vqav2_dev2", "gqa_val400")
RUNS = {"vz": ("EXP-0028", ("R01", "R02")), "c16all": ("EXP-0028", ("R19", "R20")), "router": ("EXP-0024", ("R02", "R03")),
        "router_k1": ("EXP-0024", ("R06", "R07")), "k1c16": ("EXP-0026", ("R03", "R04")), "div64": ("EXP-0032", ("R01", "R02")),
        "k1div": ("EXP-0032", ("R03", "R04")), "fastv64": ("EXP-0033", ("R01", "R02")), "vanilla": ("EXP-0033", ("R03", "R04")),
        "x1": ("EXP-0033", ("R05", "R06")), "x2": ("EXP-0033", ("R07", "R08")), "x3": ("EXP-0033", ("R09", "R10")),
        "x4": ("EXP-0033", ("R11", "R12"))}
CONFIG = {"vz": "visionzip64", "c16all": "visionzip64_c16all", "router": "fastv64_qroute_kl", "router_k1": "fastv64k1_qroute_kl",
          "k1c16": "fastv64k1_qroute_c16_kl", "div64": "divprune64", "k1div": "fastv64k1_div_kl", "fastv64": "fastv64",
          "vanilla": "vanilla", "x1": "vz128_fastv60_kll", "x2": "vzcand128_div64", "x3": "div64_qroute", "x4": "div54_ctx10"}
NAMES = {"vz": "VisionZip-64", "c16all": "VZ-64 每题覆盖 16", "router": "K = 2 路由", "router_k1": "K = 1 路由", "k1c16": "K = 1 路由 + 覆盖 16",
         "div64": "DivPrune-64", "k1div": "K = 1 多样性", "fastv64": "FastV-64", "vanilla": "不剪枝", "x1": "X1", "x2": "X2", "x3": "X3", "x4": "X4"}
CANDIDATES = [m for m in RUNS if m != "vanilla"]
PAIRS = [("div64", "vz"), ("k1c16", "vz"), ("div64", "k1c16"), ("x3", "div64"), ("c16all", "vz")]
fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）" if v else "—"   # noqa: E731


def text_groups(bench, b):
    """Groups that a selector could use at test time: from the question text only (the router's rule and the first words)."""
    qs = [json.loads(line) for line in open(a28.FILES[bench][0], encoding="utf-8")]
    assert [str(q["question_id"]) for q in qs] == b["order"]
    first = [q["text"].strip().lower() for q in qs]
    hm = np.array([t.startswith("how many") for t in first])
    col = np.array([t.startswith("what color") or t.startswith("what colour") for t in first])
    routed = b["routed"]
    return {"是非形式": routed & ~hm, "how many": hm, "what color": col & ~routed, "其余": ~routed & ~col & ~hm}


def main():
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} exists; results are never overwritten")
    if not os.path.exists(f"{OUT}/EXP-0033/analysis/SHA256SUMS"):
        sys.exit("the EXP-0033 analysis is not sealed yet")
    rows = {}
    for exp in sorted({e for e, _ in RUNS.values()}):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    checks, res = {}, {"T1": {}, "T2": {}, "T3": {}}
    L = ["# SQZ-EXP-0033 补充：64 档的各种选法在不同题型上谁占优（探索性）", "",
         "脚本写于看到任何 EXP-0033 结果之前，在预注册分析封存之后运行；全部是开发集上的事后计算。T3 的“分组选方法”是在开发集上挑的，"
         "样本内的上限偏乐观，交叉估计也只说明开发集内部的稳定性，不能代替新数据上的确认（P14）。", ""]
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc = {}
        for meth, (exp, rids) in RUNS.items():
            rid = rids[BENCHES.index(bench)]
            x = rows[f"{exp}-{rid}"]
            checks[f"{exp}-{rid} ({meth}, {bench}) completed with the expected configuration"] = float(
                x["status"] == "completed" and x["bench"] == bench and x["config"] == CONFIG[meth])
            a = {}
            for line in open(f"{OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
                y = json.loads(line)
                a[str(y["question_id"])] = y["text"]
            checks[f"{exp}-{rid}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            sc[meth] = a28.recompute(bench, b, a)
            checks[f"{exp}-{rid}: recomputed accuracy equals the registered headline"] = float(abs(100 * sc[meth].mean() - float(x["headline"])) < 1e-6)
        tg = text_groups(bench, b)
        checks[f"{bench}: the text groups partition the questions"] = float(np.all(sum(m.astype(int) for m in tg.values()) == 1))
        groups = {**b["groups"], **{f"[文字 {k}]": v for k, v in tg.items()}}
        # T1
        res["T1"][bench] = {g: {"n": int(m.sum()), **{k: 100 * float(sc[k][m].mean()) for k in RUNS}} for g, m in {"[全部]": np.ones(n, bool), **groups}.items() if m.any()}
        L += [f"## {bench}", "", "### T1 各配置在每个题型上的准确率", "",
              "| 组 | 题数 | " + " | ".join(NAMES[k] for k in RUNS) + " |", "| --- | --- |" + " --- |" * len(RUNS)]
        for g, r in res["T1"][bench].items():
            best = max(CANDIDATES, key=lambda k: r[k])
            L.append(f"| {g} | {r['n']} | " + " | ".join((f"**{r[k]:.2f}**" if k == best else f"{r[k]:.2f}") for k in RUNS) + " |")
        L.append("")
        # T2
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]

        def paired(x, y, m=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if m is None else m
            if not m.any():
                return None
            bs = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(m.sum())]

        res["T2"][bench] = {}
        L += ["### T2 “铺开”与“聚焦”的配对差按题型（百分点）", "", "| 组 | " + " | ".join(f"{NAMES[x]} − {NAMES[y]}" for x, y in PAIRS) + " |",
              "| --- |" + " --- |" * len(PAIRS)]
        for g, m in {"[全部]": None, **groups}.items():
            if m is not None and m.sum() < 100:
                continue
            r = {f"{x} - {y}": paired(x, y, m) for x, y in PAIRS}
            res["T2"][bench][g] = r
            L.append(f"| {g} | " + " | ".join(fmt(v) for v in r.values()) + " |")
        L.append("")
        # T3
        tg_list = list(tg.items())
        insample = sum(max(sc[k][m].sum() for k in CANDIDATES) for _, m in tg_list) / n
        best_single = max(CANDIDATES, key=lambda k: sc[k].mean())
        gains, picks = [], {g: {} for g, _ in tg_list}
        img_ids = np.arange(len(b["images"]))
        for s in range(N_SPLIT):
            half = np.random.default_rng(s).permutation(img_ids)[: len(img_ids) // 2]
            inA = np.isin(b["img_of"], half)
            g_fold = []
            for train, test in ((inA, ~inA), (~inA, inA)):
                single = max(CANDIDATES, key=lambda k: sc[k][train].mean())
                chosen = np.zeros(n)
                for g, m in tg_list:
                    k = max(CANDIDATES, key=lambda k_: sc[k_][train & m].mean() if (train & m).any() else -1)
                    picks[g][k] = picks[g].get(k, 0) + 1
                    chosen[m] = sc[k][m]
                g_fold.append(100 * float(chosen[test].mean() - sc[single][test].mean()))
            gains.append(float(np.mean(g_fold)))
        res["T3"][bench] = {"best_single": best_single, "best_single_acc": 100 * float(sc[best_single].mean()),
                            "in_sample_by_text_group": 100 * float(insample), "cross_fit_gain_mean": float(np.mean(gains)),
                            "cross_fit_gain_q": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
                            "picks": picks, "group_sizes": {g: int(m.sum()) for g, m in tg_list},
                            "group_best_in_sample": {g: max(CANDIDATES, key=lambda k: sc[k][m].mean()) for g, m in tg_list}}
        t3 = res["T3"][bench]
        L += ["### T3 按题目文字分组选方法", "",
              f"- 整体最好的单个配置：{NAMES[best_single]} {t3['best_single_acc']:.2f}。",
              f"- 样本内（每组取最好）：{t3['in_sample_by_text_group']:.2f}（高 {t3['in_sample_by_text_group'] - t3['best_single_acc']:+.2f}，乐观）。",
              f"- 交叉估计（{N_SPLIT} 次按图像对半分）：分组选 − 整体选一个 = {t3['cross_fit_gain_mean']:+.2f}（2.5% / 97.5% 分位 "
              f"{t3['cross_fit_gain_q'][0]:+.2f} / {t3['cross_fit_gain_q'][1]:+.2f}）。",
              "- 各组题数与样本内最好的配置：" + "；".join(f"{g} {t3['group_sizes'][g]} 题 → {NAMES[t3['group_best_in_sample'][g]]}" for g, _ in tg_list) + "。",
              "- 交叉估计中各组被选中的次数：" + "；".join(f"{g}：" + "、".join(f"{NAMES[k]} {v}" for k, v in sorted(p.items(), key=lambda kv: -kv[1]))
                                           for g, p in picks.items()) + "。", ""]
    if any(x < 1 for x in checks.values()):
        print("STOP", {k: x for k, x in checks.items() if x < 1})
        sys.exit(1)
    L += ["## 核对", ""] + [f"- {k}: {100 * x:.0f}%" for k, x in checks.items()]
    res["checks"] = checks
    os.makedirs(OUT_DIR)
    json.dump(res, open(f"{OUT_DIR}/types_0033.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(f"{OUT_DIR}/types_0033.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
