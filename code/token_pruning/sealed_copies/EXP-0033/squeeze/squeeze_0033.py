"""SQZ-EXP-0033（探索性，不是判据）：EXP-0033 的四个组合留下了什么、增减落在哪些题上。脚本写于 2026-10-30 晚、EXP-0033 运行中、
看到任何 EXP-0033 结果之前（sha256 记在 squeeze/notes.md）；在预注册分析封存之后运行。只读已封存的运行，CPU；重抽同 analyze_0033
（每个集内按图像 1000 次，种子 0）。

  python -B squeeze_0033.py     # 写 code/output/EXP-0033/squeeze/{squeeze_0033.md,squeeze_0033.json}；目录已存在就停

S1 选的形状：DivPrune-64、X2（CLS 前 128 个候选里做多样性）、X3、X4 的 54 个与 VisionZip 的 53 个主导 patch（EXP-0030 的 cls.json 记的同一
   CLS 分数的前 53 个）各覆盖 6 × 6 个块中的几个、留几个整块落在填充里的 patch、与 DivPrune-64 重合多少；X2 的 128 个候选覆盖几个块、有几个在填充里。
S2 铺开的剂量（F75 的“DivPrune 的优势来自铺开”）：按 X2 比 DivPrune-64 少覆盖的块数三等分，X2 − DivPrune-64；按两者的 Jaccard 三等分同样。
S3 X3 被路由的题：按答案类型与 S 的三等分，X3 − DivPrune-64、X3 − K = 2 路由；被路由的题上 X3 与 DivPrune-64 重合多少。
S4 X1 里 FastV 在 VisionZip-128 的 128 个 token 中留了哪些：CLS、107 个主导 patch、20 个上下文 token 各占多少（与按比例均匀相比），
   按留下的上下文 token 数分组的 X1 − VisionZip-64。
S5 两两互补的上限：逐题取两者中较好的一个的平均（oracle）比两者中较好的整体高多少；不剪枝答对而剪枝答错的题的比例。
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402
from fastv_llava import _padding_mask  # noqa: E402
from squeeze_0018 import OUT  # noqa: E402

OUT_DIR = f"{OUT}/EXP-0033/squeeze"
N_BOOT = 1000
BENCHES = ("vqav2_dev2", "gqa_val400")
RUNS = {"fastv64": ("EXP-0033", ("R01", "R02")), "vanilla": ("EXP-0033", ("R03", "R04")), "x1": ("EXP-0033", ("R05", "R06")),
        "x2": ("EXP-0033", ("R07", "R08")), "x3": ("EXP-0033", ("R09", "R10")), "x4": ("EXP-0033", ("R11", "R12")),
        "div64": ("EXP-0032", ("R01", "R02")), "vz": ("EXP-0028", ("R01", "R02")), "router": ("EXP-0024", ("R02", "R03")),
        "k1cls": ("EXP-0030", ("R01", "R02"))}
CONFIG = {"fastv64": "fastv64", "vanilla": "vanilla", "x1": "vz128_fastv60_kll", "x2": "vzcand128_div64", "x3": "div64_qroute",
          "x4": "div54_ctx10", "div64": "divprune64", "vz": "visionzip64", "router": "fastv64_qroute_kl", "k1cls": "fastv64k1_cls_kl"}
NAMES = {"fastv64": "FastV-64", "vanilla": "不剪枝", "x1": "X1", "x2": "X2", "x3": "X3", "x4": "X4", "div64": "DivPrune-64",
         "vz": "VisionZip-64", "router": "K = 2 路由", "k1cls": "K = 1 CLS 排序"}
SEALED = [("x1", "vz"), ("x1", "fastv64"), ("x2", "vz"), ("x2", "div64"), ("x3", "div64"), ("x3", "router"), ("x4", "div64"),
          ("x4", "vz"), ("vanilla", "div64")]
fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）" if v else "—"   # noqa: E731


def blocks(pos):
    return len({(p // 24 // 4, p % 24 // 4) for p in pos})


def jac(p, q):
    return len(set(p) & set(q)) / len(set(p) | set(q))


def thirds(v):
    q = np.quantile(v, [1 / 3, 2 / 3])
    return q, (("低三分之一", v <= q[0]), ("中间", (v > q[0]) & (v <= q[1])), ("高三分之一", v > q[1]))


def main():
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} exists; results are never overwritten")
    if not os.path.exists(f"{OUT}/EXP-0033/analysis/SHA256SUMS"):
        sys.exit("the EXP-0033 analysis is not sealed yet")
    rows = {}
    for exp in sorted({e for e, _ in RUNS.values()}):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    sealed = json.load(open(f"{OUT}/EXP-0033/analysis/0033.json", encoding="utf-8"))["pairs"]
    checks, res = {}, {"S1": {}, "S2": {}, "S3": {}, "S4": {}, "S5": {}}
    L = ["# SQZ-EXP-0033：四个组合留下了什么、增减落在哪些题上（探索性）", "",
         "脚本写于看到任何 EXP-0033 结果之前，在预注册分析封存之后运行；全部是事后计算，不改 EXP-0033 的读法。"
         "区间：每个集内按图像重抽 1000 次（种子 0）。", ""]
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, d_of = {}, {}
        for meth, (exp, rids) in RUNS.items():
            rid = rids[BENCHES.index(bench)]
            x = rows[f"{exp}-{rid}"]
            checks[f"{exp}-{rid} ({meth}, {bench}) completed with the expected configuration"] = float(
                x["status"] == "completed" and x["bench"] == bench and x["config"] == CONFIG[meth])
            d_of[meth] = f"{OUT}/{exp}/{exp}-{rid}"
            a = {}
            for line in open(f"{d_of[meth]}/answers.jsonl", encoding="utf-8"):
                y = json.loads(line)
                a[str(y["question_id"])] = y["text"]
            sc[meth] = a28.recompute(bench, b, a)
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

        for x, y in SEALED:
            key = f"{bench} {x} - {y}"
            checks[f"{key} equals the sealed EXP-0033 analysis"] = float(key in sealed and np.allclose(paired(x, y), sealed[key]))
        div = json.load(open(f"{d_of['div64']}/div.json"))["records"]
        r2 = json.load(open(f"{d_of['x2']}/combo.json"))["records"]
        r3 = json.load(open(f"{d_of['x3']}/combo.json"))["records"]
        r4 = json.load(open(f"{d_of['x4']}/combo.json"))["records"]
        cls = json.load(open(f"{d_of['k1cls']}/cls.json"))["records"]
        k1 = json.load(open(f"{d_of['x1']}/kept_tokens.json"))
        checks[f"{bench}: one record per question in every log"] = float(len(div) == len(r2) == len(r3) == len(r4) == len(cls)
                                                                        == len(k1["kept"]) == n)
        checks[f"{bench}: x3 unrouted = DivPrune-64 (as the sealed check)"] = float(np.mean(
            [x["kept"] == y["kept"] for x, y in zip(r3, div) if not x["routed"]]))
        pad = [_padding_mask(*b["size"][i], 24).numpy().astype(bool) for i in range(n)]
        checks[f"{bench}: padding masks are flat over the 576 patches"] = float(all(p.shape == (576,) for p in pad))
        sets = {"div64": [d["kept"] for d in div], "x2": [x["kept"] for x in r2], "x3": [x["kept"] for x in r3],
                "x4 54": [x["kept"] for x in r4], "vz 53": [c["top53"] for c in cls]}
        # S1
        r1 = {"blocks": {k: float(np.mean([blocks(s) for s in v])) for k, v in sets.items()},
              "padding": {k: float(np.mean([pad[i][s].sum() for i, s in enumerate(v)])) for k, v in sets.items()},
              "jaccard_with_div64": {k: float(np.mean([jac(s, d) for s, d in zip(v, sets["div64"])])) for k, v in sets.items() if k != "div64"},
              "x2_candidates_blocks": float(np.mean([blocks(x["cand"]) for x in r2])),
              "x2_candidates_padding": float(np.mean([pad[i][x["cand"]].sum() for i, x in enumerate(r2)])),
              "x2_candidates_hold_div64_share": float(np.mean([len(set(x["cand"]) & set(d)) / 64 for x, d in zip(r2, sets["div64"])])),
              "blocks_with_image": float(np.mean([blocks(np.flatnonzero(~p)) for p in pad]))}
        res["S1"][bench] = r1
        L += [f"## {bench}", "", "### S1 选的形状", "",
              f"- 有图像内容的块平均 {r1['blocks_with_image']:.1f} 个（6 × 6 个块，每块 4 × 4 个 patch）。覆盖的块：" +
              "；".join(f"{k} {v:.1f}" for k, v in r1["blocks"].items()) + "。",
              "- 整块落在填充里的 patch：" + "；".join(f"{k} {v:.2f}" for k, v in r1["padding"].items()) + "。",
              "- 与 DivPrune-64 的 Jaccard：" + "；".join(f"{k} {v:.3f}" for k, v in r1["jaccard_with_div64"].items()) + "。",
              f"- X2 的 128 个候选：覆盖 {r1['x2_candidates_blocks']:.1f} 个块，{r1['x2_candidates_padding']:.2f} 个在填充里，"
              f"含 DivPrune-64 所选的 {r1['x2_candidates_hold_div64_share']:.1%}。", ""]
        # S2
        dblk = np.array([blocks(d) - blocks(x) for d, x in zip(sets["div64"], sets["x2"])], float)
        jx2 = np.array([jac(x, d) for x, d in zip(sets["x2"], sets["div64"])])
        res["S2"][bench] = {}
        L += ["### S2 X2 少覆盖的块数、与 DivPrune-64 的 Jaccard（各三等分）", "", "| 分组 | 组 | 题数 | 组内均值 | X2 − DivPrune-64 |", "| --- | --- | --- | --- | --- |"]
        for label, v in (("少覆盖的块数", dblk), ("Jaccard", jx2)):
            cuts, parts = thirds(v)
            res["S2"][bench][label] = {"cuts": [float(c) for c in cuts]}
            for name, m in parts:
                r = {"n": int(m.sum()), "mean": float(v[m].mean()) if m.any() else None, "x2_div64": paired("x2", "div64", m)}
                res["S2"][bench][label][name] = r
                L.append(f"| {label} | {name} | {r['n']} | {r['mean']:.2f} | {fmt(r['x2_div64'])} |")
        L.append("")
        # S3
        routed = np.array([x["routed"] for x in r3])
        share = np.array([x["share"] if x["routed"] else np.nan for x in r3])
        kinds = {g: m for g, m in b["groups"].items() if g in ("[yes/no]", "[number]", "[other]")}
        res["S3"][bench] = {"routed": int(routed.sum()), "mean_n_query": float(np.mean([x["n_query"] for x in r3 if x["routed"]])),
                            "jaccard_x3_div64_routed": float(np.mean([jac(x["kept"], d) for x, d in zip(r3, sets["div64"]) if x["routed"]]))}
        L += ["### S3 X3 被路由的题", "",
              f"- 被路由 {res['S3'][bench]['routed']} 题，平均按查询取 {res['S3'][bench]['mean_n_query']:.1f} 个，"
              f"与 DivPrune-64 的 Jaccard {res['S3'][bench]['jaccard_x3_div64_routed']:.3f}。", "",
              "| 组 | 题数 | X3 − DivPrune-64 | X3 − K = 2 路由 |", "| --- | --- | --- | --- |"]
        groups = [(f"被路由 {g}", routed & m) for g, m in kinds.items()]
        if routed.any():
            cuts, parts = thirds(share[routed])
            for name, m in parts:
                mm = np.zeros(n, bool)
                mm[np.flatnonzero(routed)[m]] = True
                groups.append((f"被路由、S {name}", mm))
            res["S3"][bench]["S_cuts"] = [float(c) for c in cuts]
        for name, m in groups:
            r = {"n": int(m.sum()), "x3_div64": paired("x3", "div64", m), "x3_router": paired("x3", "router", m)}
            res["S3"][bench][name] = r
            L.append(f"| {name} | {r['n']} | {fmt(r['x3_div64'])} | {fmt(r['x3_router'])} |")
        L.append("")
        # S4
        kept = k1["kept"]
        cls_k = np.array([sum(p == 0 for p in x) for x in kept], float)
        dom_k = np.array([sum(1 <= p <= 107 for p in x) for x in kept], float)
        ctx_k = np.array([sum(p >= 108 for p in x) for x in kept], float)
        res["S4"][bench] = {"cls": float(cls_k.mean()), "dominant": float(dom_k.mean()), "contextual": float(ctx_k.mean()),
                            "uniform": {"cls": 60 / 128, "dominant": 60 * 107 / 128, "contextual": 60 * 20 / 128}}
        L += ["### S4 X1 中 FastV 在 VisionZip-128 的 128 个 token 里留了哪些", "",
              f"- 平均留 CLS {cls_k.mean():.2f} 个、主导 patch {dom_k.mean():.1f} 个、上下文 token {ctx_k.mean():.1f} 个"
              f"（按比例均匀时 {60 / 128:.2f} / {60 * 107 / 128:.1f} / {60 * 20 / 128:.1f}）。", "",
              "| 留下的上下文 token | 题数 | X1 − VisionZip-64 |", "| --- | --- | --- |"]
        cuts, parts = thirds(ctx_k)
        res["S4"][bench]["contextual_cuts"] = [float(c) for c in cuts]
        for name, m in parts:
            r = {"n": int(m.sum()), "x1_vz": paired("x1", "vz", m)}
            res["S4"][bench][name] = r
            L.append(f"| {name} | {r['n']} | {fmt(r['x1_vz'])} |")
        L.append("")
        # S5
        L += ["### S5 两两互补的上限与不剪枝的差", "", "| 两者 | 较好者的整体 | 逐题取较好（oracle） | 上限高出 |", "| --- | --- | --- | --- |"]
        res["S5"][bench] = {}
        for x, y in (("x1", "vz"), ("x1", "fastv64"), ("x2", "vz"), ("x2", "div64"), ("x3", "div64"), ("x3", "router"), ("x4", "div64"),
                     ("x4", "vz"), ("div64", "vz"), ("div64", "router")):
            best = 100 * max(sc[x].mean(), sc[y].mean())
            orc = 100 * float(np.maximum(sc[x], sc[y]).mean())
            res["S5"][bench][f"{x} | {y}"] = {"best": best, "oracle": orc}
            L.append(f"| {NAMES[x]} | {NAMES[y]} | {best:.2f} | {orc:.2f} | {orc - best:+.2f} |")
        lost = {m: float(((sc["vanilla"] > 0) & (sc[m] < sc["vanilla"])).mean()) for m in ("fastv64", "vz", "div64", "x1", "x2", "x3", "x4")}
        res["S5"][bench]["vanilla_better_share"] = lost
        L += ["", "- 不剪枝得分更高的题的比例：" + "；".join(f"{NAMES[m]} {v:.1%}" for m, v in lost.items()) + "。", ""]
    if any(x < 1 for x in checks.values()):
        print("STOP", {k: x for k, x in checks.items() if x < 1})
        sys.exit(1)
    L += ["## 核对", ""] + [f"- {k}: {100 * x:.0f}%" for k, x in checks.items()]
    res["checks"] = checks
    os.makedirs(OUT_DIR)
    json.dump(res, open(f"{OUT_DIR}/squeeze_0033.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(f"{OUT_DIR}/squeeze_0033.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
