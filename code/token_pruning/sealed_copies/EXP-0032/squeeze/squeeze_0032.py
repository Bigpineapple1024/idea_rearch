"""SQZ-EXP-0032（看过预注册分析之后，探索性，不是判据）：本流程中的 DivPrune-64 为什么强于 VisionZip-64，同样的选择放到 K = 1 为什么
在 VQAv2 上变差，K = 1 时多样性为什么好于 CLS 排序。只读已封存的运行，CPU；重抽同 analyze_0032（每个集内按图像 1000 次，种子 0）。

  python -B squeeze_0032.py     # 写 code/output/EXP-0032/squeeze/{squeeze_0032.md,squeeze_0032.json}；目录已存在就停

S1 选了什么：DivPrune 的前 53 个与 VisionZip 的 53 个主导 patch（EXP-0030 的 cls.json 记录的同一 CLS 分数的前 53 个，与 VisionZip 自己的逐题相同）重合多少；
   K = 1 时多样性的 47 个与 CLS 排序的 47 个重合多少；各自覆盖 24 × 24 网格的 6 × 6 个块（每块 4 × 4 个 patch）中的几个；各自留了几个整块落在填充里的 patch。
S2 填充：按 DivPrune-64 留下的填充 patch 数分组，DivPrune-64 − VisionZip-64、K = 1 多样性 − DivPrune-64。
S3 逐题翻转：DivPrune-64 对 VisionZip-64、K = 1 路由、每题覆盖 16；K = 1 多样性对 DivPrune-64、对 K = 1 CLS 排序。
S4 两种 K = 1 选法分歧的大小：按两者的 Jaccard 分组，多样性 − CLS 排序的差。
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

OUT_DIR = f"{OUT}/EXP-0032/squeeze"
N_BOOT = 1000
BENCHES = ("vqav2_dev2", "gqa_val400")
RUNS = {"div64": ("EXP-0032", ("R01", "R02")), "k1div": ("EXP-0032", ("R03", "R04")), "vz": ("EXP-0028", ("R01", "R02")),
        "c16all": ("EXP-0028", ("R19", "R20")), "router_k1": ("EXP-0024", ("R06", "R07")), "k1cls": ("EXP-0030", ("R01", "R02"))}
CONFIG = {"div64": "divprune64", "k1div": "fastv64k1_div_kl", "vz": "visionzip64", "c16all": "visionzip64_c16all",
          "router_k1": "fastv64k1_qroute_kl", "k1cls": "fastv64k1_cls_kl"}
NAMES = {"div64": "DivPrune-64", "k1div": "K = 1 多样性（前 47 个）", "vz": "VisionZip-64", "c16all": "每题覆盖 16", "router_k1": "K = 1 路由",
         "k1cls": "K = 1 CLS 排序"}
SEALED = {("div64", "vz"), ("k1div", "div64"), ("k1div", "k1cls"), ("div64", "router_k1"), ("div64", "c16all"), ("k1div", "vz")}
fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）" if v else "—"   # noqa: E731


def run_dir(meth, bench):
    exp, rids = RUNS[meth]
    return exp, rids[BENCHES.index(bench)]


def blocks(pos):
    return len({(p // 24 // 4, p % 24 // 4) for p in pos})


def main():
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} exists; results are never overwritten")
    rows = {}
    for exp in sorted({e for e, _ in RUNS.values()}):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    sealed = json.load(open(f"{OUT}/EXP-0032/analysis/0032.json", encoding="utf-8"))["pairs"]
    checks, res = {}, {"S1": {}, "S2": {}, "S3": {}, "S4": {}}
    L = ["# SQZ-EXP-0032：DivPrune-64 选了什么、填充、逐题翻转与两种 K = 1 选法的分歧（探索性）", "",
         "全部是看过预注册分析之后的事后计算，不改 EXP-0032 的读法。区间：每个集内按图像重抽 1000 次（种子 0）。", ""]
    for bench in BENCHES:
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc = {}
        for meth in RUNS:
            exp, rid = run_dir(meth, bench)
            x = rows[f"{exp}-{rid}"]
            checks[f"{exp}-{rid} ({meth}, {bench}) completed with the expected configuration"] = float(
                x["status"] == "completed" and x["bench"] == bench and x["config"] == CONFIG[meth])
            a = {}
            for line in open(f"{OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
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
            checks[f"{key} equals the sealed EXP-0032 analysis"] = float(key in sealed and np.allclose(paired(x, y), sealed[key]))
        div = json.load(open(f"{OUT}/EXP-0032/EXP-0032-{run_dir('div64', bench)[1]}/div.json"))["records"]
        k1d = json.load(open(f"{OUT}/EXP-0032/EXP-0032-{run_dir('k1div', bench)[1]}/kept_tokens.json"))["kept"]
        cls = json.load(open(f"{OUT}/EXP-0030/EXP-0030-{run_dir('k1cls', bench)[1]}/cls.json"))["records"]
        k1c = json.load(open(f"{OUT}/EXP-0030/EXP-0030-{run_dir('k1cls', bench)[1]}/kept_tokens.json"))["kept"]
        checks[f"{bench}: one record per question in every log"] = float(len(div) == len(k1d) == len(cls) == len(k1c) == n)
        checks[f"{bench}: k1cls kept = its logged own 47"] = float(np.mean([sorted(c["own"]) == sorted(k) for c, k in zip(cls, k1c)]))
        checks[f"{bench}: k1div kept = DivPrune's first 47"] = float(np.mean([sorted(d["order"][:47]) == sorted(k) for d, k in zip(div, k1d)]))
        pad = [_padding_mask(*b["size"][i], 24).numpy().astype(bool) for i in range(n)]   # True where the patch lies wholly in padding
        checks[f"{bench}: padding masks are flat over the 576 patches"] = float(all(p.shape == (576,) for p in pad))
        npad = {"div64": np.array([int(pad[i][d["kept"]].sum()) for i, d in enumerate(div)]),
                "vz53": np.array([int(pad[i][c["top53"]].sum()) for i, c in enumerate(cls)]),
                "k1div": np.array([int(pad[i][k].sum()) for i, k in enumerate(k1d)]),
                "k1cls": np.array([int(pad[i][k].sum()) for i, k in enumerate(k1c)])}
        jac = lambda p, q: len(set(p) & set(q)) / len(set(p) | set(q))   # noqa: E731
        ov53 = np.array([len(set(d["order"][:53]) & set(c["top53"])) for d, c in zip(div, cls)])
        jk = np.array([jac(k, c) for k, c in zip(k1d, k1c)])
        ik = np.array([len(set(k) & set(c)) for k, c in zip(k1d, k1c)])
        bl = {"div64 first 53": np.array([blocks(d["order"][:53]) for d in div]), "vz 53": np.array([blocks(c["top53"]) for c in cls]),
              "div64 64": np.array([blocks(d["kept"]) for d in div]), "k1div 47": np.array([blocks(k) for k in k1d]),
              "k1cls 47": np.array([blocks(k) for k in k1c])}
        nonpad_blocks = np.array([blocks(np.flatnonzero(~pad[i])) for i in range(n)])
        r1 = {"overlap_div53_vz53": float(ov53.mean()), "jaccard_div53_vz53": float(np.mean([o / (106 - o) for o in ov53])),
              "jaccard_k1div_k1cls": float(jk.mean()), "common_k1div_k1cls": float(ik.mean()),
              "blocks_mean": {k: float(v.mean()) for k, v in bl.items()}, "blocks_with_image_mean": float(nonpad_blocks.mean()),
              "padding_mean": {k: float(v.mean()) for k, v in npad.items()}, "share_div64_any_padding": float((npad["div64"] > 0).mean())}
        res["S1"][bench] = r1
        L += [f"## {bench}", "", "### S1 选了什么", "",
              f"- DivPrune 的前 53 个与 VisionZip 的 53 个主导 patch 平均相同 {r1['overlap_div53_vz53']:.1f} 个（Jaccard {r1['jaccard_div53_vz53']:.3f}）。",
              f"- K = 1 时多样性的 47 个与 CLS 排序的 47 个平均相同 {r1['common_k1div_k1cls']:.1f} 个（Jaccard {r1['jaccard_k1div_k1cls']:.3f}）。",
              "- 覆盖 6 × 6 个块中的几个（每块 4 × 4 个 patch；有图像内容的块平均 " + f"{r1['blocks_with_image_mean']:.1f} 个）：" +
              "；".join(f"{k} {v:.1f}" for k, v in r1["blocks_mean"].items()) + "。",
              "- 整块落在填充里的 patch：" + "；".join(f"{k} {v:.2f}" for k, v in r1["padding_mean"].items()) +
              f"；DivPrune-64 至少留一个的题占 {r1['share_div64_any_padding']:.1%}。", ""]
        # S2 padding groups
        L += ["### S2 按 DivPrune-64 留下的填充 patch 数分组", "", "| 组 | 题数 | DivPrune-64 − VisionZip-64 | K = 1 多样性 − DivPrune-64 |", "| --- | --- | --- | --- |"]
        res["S2"][bench] = {}
        for name, m in (("0 个", npad["div64"] == 0), ("1～4 个", (npad["div64"] >= 1) & (npad["div64"] <= 4)),
                        ("5～9 个", (npad["div64"] >= 5) & (npad["div64"] <= 9)), ("10 个及以上", npad["div64"] >= 10)):
            r = {"n": int(m.sum()), "div64_vz": paired("div64", "vz", m), "k1div_div64": paired("k1div", "div64", m)}
            res["S2"][bench][name] = r
            L.append(f"| {name} | {r['n']} | {fmt(r['div64_vz'])} | {fmt(r['k1div_div64'])} |")
        L.append("")
        # S3 flips
        L += ["### S3 逐题翻转", "", "| 比较 | 得分不同的题 | 变好 | 变差 | 净差 |", "| --- | --- | --- | --- | --- |"]
        res["S3"][bench] = {}
        for x, y in (("div64", "vz"), ("div64", "router_k1"), ("div64", "c16all"), ("k1div", "div64"), ("k1div", "k1cls")):
            d = sc[x] - sc[y]
            r = {"changed": int((d != 0).sum()), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "net": 100 * float(d.mean())}
            res["S3"][bench][f"{x} - {y}"] = r
            L.append(f"| {NAMES[x]} − {NAMES[y]} | {r['changed']}（{r['changed'] / n:.1%}） | {r['better']} | {r['worse']} | {r['net']:+.2f} |")
        L.append("")
        # S4 divergence of the two K = 1 selections
        q = np.quantile(jk, [1 / 3, 2 / 3])
        L += ["### S4 两种 K = 1 选法分歧的大小（按 Jaccard 三等分）", "", "| 组 | 题数 | 平均 Jaccard | 多样性 − CLS 排序 |", "| --- | --- | --- | --- |"]
        res["S4"][bench] = {"cuts": [float(c) for c in q]}
        for name, m in (("分歧最大的三分之一", jk <= q[0]), ("中间", (jk > q[0]) & (jk <= q[1])), ("分歧最小的三分之一", jk > q[1])):
            r = {"n": int(m.sum()), "jaccard": float(jk[m].mean()), "k1div_k1cls": paired("k1div", "k1cls", m)}
            res["S4"][bench][name] = r
            L.append(f"| {name} | {r['n']} | {r['jaccard']:.3f} | {fmt(r['k1div_k1cls'])} |")
        L.append("")
    if any(x < 1 for x in checks.values()):
        print("STOP", {k: x for k, x in checks.items() if x < 1})
        sys.exit(1)
    L += ["## 核对", ""] + [f"- {k}: {100 * x:.0f}%" for k, x in checks.items()]
    res["checks"] = checks
    os.makedirs(OUT_DIR)
    json.dump(res, open(f"{OUT_DIR}/squeeze_0032.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(f"{OUT_DIR}/squeeze_0032.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
