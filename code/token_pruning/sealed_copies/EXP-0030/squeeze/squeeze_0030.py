"""SQZ-EXP-0030 (exploratory, after EXP-0030's analysis was sealed): where the one-full-layer + CLS ranking loses to
VisionZip-64, and where IDEA-0007's selection loses to the splice. CPU; reads the sealed run folders only.

  python squeeze_0030.py      # writes code/output/EXP-0030/squeeze/{squeeze_0030.md,squeeze_0030.json}

A. Overlap of k1cls's 47 kept patches with VisionZip's own 53 dominant ones (the same CLS score; k1cls never keeps a patch
   wholly in the padding), and how many of the 53 lie wholly in the padding.
B. k1cls - VisionZip-64 split by that overlap, by the padding share of the 53, and by question type (VQAv2's question_type
   with >= 150 questions; GQA's structural / semantic types).
C. On the unrouted questions, the selection (k1qcls_c16) - the splice's half (VisionZip-64 + cover 16), by question type.
Paired differences resample images with replacement (1000 draws, seed 0), as analyze_0030. Exploratory: many splits.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402
from fastv_llava import _padding_mask  # noqa: E402

OUT = a28.OUT
RUNS = {"k1cls": ("EXP-0030", "R01", "R02"), "sel": ("EXP-0030", "R05", "R06"), "vz": ("EXP-0028", "R01", "R02"),
        "c16": ("EXP-0028", "R13", "R14")}
MIN_TYPE = 150
N_BOOT = 1000


def main():
    out_dir = f"{OUT}/EXP-0030/squeeze"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    res = {}
    L = ["# SQZ-EXP-0030 的计算（探索性，EXP-0030 分析封存之后；只读已封存的运行）", ""]
    for k, bench in enumerate(a28.BENCHES):
        b = a28.load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc = {}
        for m, (exp, r1, r2) in RUNS.items():
            d = f"{OUT}/{exp}/{exp}-{(r1, r2)[k]}"
            a = {str(json.loads(l)["question_id"]): json.loads(l)["text"] for l in open(f"{d}/answers.jsonl", encoding="utf-8")}
            sc[m] = a28.recompute(bench, b, a)
        d1 = f"{OUT}/EXP-0030/EXP-0030-{RUNS['k1cls'][1 + k]}"
        top = [set(x["top53"]) for x in json.load(open(f"{d1}/cls.json"))["records"]]
        kept = [set(x) for x in json.load(open(f"{d1}/kept_tokens.json"))["kept"]]
        pad = [_padding_mask(w, h, 24).numpy() for w, h in b["size"]]           # True: wholly in the padding
        overlap = np.array([len(kp & t) for kp, t in zip(kept, top)])
        top_pad = np.array([int(p[sorted(t)].sum()) for p, t in zip(pad, top)])
        by_img = [np.flatnonzero(b["img_of"] == j) for j in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[j] for j in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]

        def paired(x, y, mask):
            dd = sc[x] - sc[y]
            if mask.sum() == 0:
                return None
            boot = [100 * dd[idx][mask[idx]].mean() for idx in draws if mask[idx].any()]
            return [100 * float(dd[mask].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(mask.sum())]

        r = res[bench] = {"overlap": {"mean": float(overlap.mean()), "all47": float((overlap == 47).mean()),
                                      "hist": {int(v): int((overlap == v).sum()) for v in np.unique(overlap)}},
                          "top53_in_padding": {"mean": float(top_pad.mean()), "any": float((top_pad > 0).mean())},
                          "A_k1cls_vz": {}, "C_sel_c16_unrouted": {}}
        allq = np.ones(n, bool)
        splits = {"全部": allq, "47 个都在 VisionZip 的 53 个里": overlap == 47, "不都在": overlap < 47,
                  "53 个里没有落在填充里的": top_pad == 0, "53 个里有落在填充里的": top_pad > 0,
                  "被路由": routed, "不路由": ~routed}
        if bench.startswith("vqav2"):
            qt = np.array([b["gts"][q]["question_type"] for q in b["order"]])
            for t in sorted(set(qt), key=lambda t: -(qt == t).sum()):
                if (qt == t).sum() >= MIN_TYPE:
                    splits[f"question_type: {t}"] = qt == t
        else:
            splits.update({g: m for g, m in b["groups"].items() if g.startswith(("[structural", "[semantic")) and m.sum() >= MIN_TYPE})
        for name, m in splits.items():
            r["A_k1cls_vz"][name] = paired("k1cls", "vz", m)
            r["C_sel_c16_unrouted"][name] = paired("sel", "c16", m & ~routed) if name not in ("被路由",) else None
        fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
        L += [f"## {bench}", "",
              f"- A. k1cls 留下的 47 个与 VisionZip 自己的 53 个主导 patch 的重合：平均 {r['overlap']['mean']:.2f} 个，"
              f"47 个都在其中的题占 {100 * r['overlap']['all47']:.1f}%；分布 {json.dumps(r['overlap']['hist'])}",
              f"- VisionZip 的 53 个里整块落在填充里的：平均 {r['top53_in_padding']['mean']:.2f} 个，至少一个的题占 {100 * r['top53_in_padding']['any']:.1f}%",
              "", "| 分组 | B. k1cls − VisionZip-64 | C. 不路由的题：所选 − VisionZip-64 + 覆盖 16 |", "| --- | --- | --- |"]
        L += [f"| {name} | {fmt(r['A_k1cls_vz'][name])} | {fmt(r['C_sel_c16_unrouted'][name])} |" for name in splits]
        L.append("")
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/squeeze_0030.json", "w"), indent=2, ensure_ascii=False)
    open(f"{out_dir}/squeeze_0030.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
