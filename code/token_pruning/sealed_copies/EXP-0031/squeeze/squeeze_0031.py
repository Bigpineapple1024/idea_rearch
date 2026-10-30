"""SQZ-EXP-0031（看过预注册分析之后，探索性，不是判据）：IDEA-0007 的确认为什么只在 GQA 上成立，新加的编码器排序为什么没有作用；
几组新数据合起来，小的差别是什么样子。只读已封存的运行（EXP-0031 与开发、确认中复用的运行），CPU；重抽同 analyze_0031
（每个集内按图像 1000 次，种子 0）。

  python -B squeeze_0031.py     # 写 code/output/EXP-0031/squeeze/{squeeze_0031.md,squeeze_0031.json}；目录已存在就停

S1 几组数据合起来（事后）：每题覆盖 16 − VisionZip-64、K = 1 路由 + 覆盖 16 − VisionZip-64 在开发集、第一组确认集（test3 / testdev）、
   第二组确认集（test4 / val400b）上的差，两组确认集按题数合并的估计（每个集内按图像重抽、按抽样序号拼起来），以及两组确认集之差的区间。
S2 逐题翻转：所选、每题覆盖 16、对应的 FastV 配置两两之间得分不同的题，变好与变差各多少（新数据与 TextVQA）。
S3 排序：不路由的题上所选（CLS 排序 + 覆盖 16）与对应的 FastV 配置（FastV 的注意力排序 + 覆盖 16）留下的 47 个有多少相同
   （kept_tokens.json，开发集与新数据），以及这些题上两者的差。
S4 TextVQA（不给 OCR）：读字题与其余题（同 SQZ-EXP-0029 的分组），所选 − VisionZip-64、所选 − 每题覆盖 16、不剪枝 − 所选。
S5 分组在开发集与新数据上的方向：所选 − VisionZip-64、所选 − 每题覆盖 16，VQAv2 按答案类型与颜色题，GQA 按结构与语义类型；
   两集同号的组数（F65 的复验）。
"""
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0025 as a25  # noqa: E402
import analyze_0028 as a28  # noqa: E402
import analyze_0029 as a29  # noqa: E402
import analyze_0031 as a31  # noqa: E402
from squeeze_0018 import EVAL_DATA, OUT  # noqa: E402

OUT_DIR = f"{OUT}/EXP-0031/squeeze"
a28.FILES.update(a29.FILES)
a28.FILES.update(a31.FILES)
N_BOOT = 1000
SETS = {"vqav2": ("vqav2_dev2", "vqav2_test3", "vqav2_test4"), "gqa": ("gqa_val400", "gqa_testdev", "gqa_val400b")}
RUNS = {("vqav2_dev2", "vz"): ("EXP-0028", "R01"), ("gqa_val400", "vz"): ("EXP-0028", "R02"),
        ("vqav2_dev2", "c16all"): ("EXP-0028", "R19"), ("gqa_val400", "c16all"): ("EXP-0028", "R20"),
        ("vqav2_dev2", "cp"): ("EXP-0026", "R03"), ("gqa_val400", "cp"): ("EXP-0026", "R04"),
        ("vqav2_dev2", "sel"): ("EXP-0030", "R05"), ("gqa_val400", "sel"): ("EXP-0030", "R06"),
        ("vqav2_test3", "vz"): ("EXP-0027", "R04"), ("gqa_testdev", "vz"): ("EXP-0022", "R07"),
        ("vqav2_test3", "c16all"): ("EXP-0029", "R01"), ("gqa_testdev", "c16all"): ("EXP-0029", "R02"),
        ("vqav2_test3", "cp"): ("EXP-0027", "R01"), ("gqa_testdev", "cp"): ("EXP-0027", "R05")}
RUNS.update(a31.RUNS)
CONFIG = {"vz": "visionzip64", "c16all": "visionzip64_c16all", "cp": "fastv64k1_qroute_c16_kl", "sel": "fastv64k1_qroute_cls_c16_kl",
          "router": "fastv64_qroute_kl", "vanilla": "vanilla"}
NAMES = {"sel": "所选（IDEA-0007，k1qcls_c16）", "vz": "VisionZip-64", "c16all": "每题覆盖 16（IDEA-0006 的所选）",
         "cp": "K = 1 路由 + 覆盖 16（IDEA-0005 的所选，对应的 FastV 配置）", "router": "K = 2 路由", "vanilla": "不剪枝"}
# the sealed analyses' overall pairs that the recomputation must reproduce: (bench, x, y) -> (analysis json, key)
SEALED = {}
for bench in ("vqav2_test4", "gqa_val400b"):
    for x, y in (("sel", "vz"), ("sel", "c16all"), ("sel", "cp"), ("c16all", "vz"), ("cp", "vz")):
        SEALED[(bench, x, y)] = ("EXP-0031", f"{bench} {x} - {y}")
for bench in ("vqav2_test3", "gqa_testdev"):
    SEALED[(bench, "c16all", "vz")] = ("EXP-0029", f"{bench} sel - vz")
    SEALED[(bench, "cp", "vz")] = ("EXP-0029", f"{bench} fk1c16 - vz")
for bench in ("vqav2_dev2", "gqa_val400"):
    for y, k in (("vz", "vz"), ("c16all", "c16all"), ("cp", "fk1c16")):
        SEALED[(bench, "sel", y)] = ("EXP-0030", f"{bench} k1qcls_c16 - {k}")
    SEALED[(bench, "c16all", "vz")] = ("EXP-0028", f"{bench} c16all - vz")
for x, y in (("sel", "vz"), ("sel", "c16all"), ("vanilla", "sel")):
    SEALED[("textvqa_noocr", x, y)] = ("EXP-0031", f"textvqa_noocr {x} - {y}")

fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）" if v else "—"   # noqa: E731


def rows_of():
    rows = {}
    for exp in sorted({e for e, _ in RUNS.values()}):
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            rows.update({x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")})
    return rows


def load(bench):
    return a28.load_bench(bench) if bench != "textvqa_noocr" else a25.load_bench(bench)


def score(bench, b, meth):
    exp, rid = RUNS[(bench, meth)]
    a = {}
    for line in open(f"{OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
        x = json.loads(line)
        a[str(x["question_id"])] = x["text"]
    return a28.recompute(bench, b, a) if bench != "textvqa_noocr" else a25.recompute(bench, b, a)


def draws_of(b):
    by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
    rng = np.random.default_rng(0)
    return [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]


def paired(d, draws, m=None):
    m = np.ones(len(d), bool) if m is None else m
    if not m.any():
        return None
    bs = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
    return [100 * float(d[m].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(m.sum())]


def boot_means(d, draws):
    return np.array([100 * d[idx].mean() for idx in draws])


def main():
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} exists; results are never overwritten")
    rows = rows_of()
    checks = {}
    for (bench, meth), (exp, rid) in RUNS.items():
        x = rows[f"{exp}-{rid}"]
        checks[f"{exp}-{rid} ({meth}, {bench}) completed with the expected configuration"] = float(
            x["status"] == "completed" and x["bench"] == bench and x["config"] == CONFIG[meth])
    sealed = {e: json.load(open(f"{OUT}/{e}/analysis/{e[4:]}.json", encoding="utf-8"))["pairs"]
              for e in ("EXP-0028", "EXP-0029", "EXP-0030", "EXP-0031")}
    B, SC, DR = {}, {}, {}
    for bench in SETS["vqav2"] + SETS["gqa"] + ("textvqa_noocr",):
        B[bench] = b = load(bench)
        meths = [m for (bb, m) in RUNS if bb == bench]
        SC[bench] = {m: score(bench, b, m) for m in meths}
        DR[bench] = draws_of(b)
    for (bench, x, y), (e, key) in SEALED.items():
        v = paired(SC[bench][x] - SC[bench][y], DR[bench])
        checks[f"{bench} {x} - {y} equals the sealed {e} analysis ({key})"] = float(key in sealed[e] and np.allclose(v, sealed[e][key]))
    res = {"S1": {}, "S2": {}, "S3": {}, "S4": {}, "S5": {}}
    L = ["# SQZ-EXP-0031：IDEA-0007 的确认只在 GQA 上成立、编码器排序没有作用；几组新数据合起来的小差别（探索性）", "",
         "全部是看过预注册分析之后的事后计算，不改 EXP-0029、EXP-0031 的判读。区间：每个集内按图像重抽 1000 次（种子 0）；合并的估计按抽样序号把各集拼起来。", ""]

    # S1 pooled across the fresh sets
    L += ["## S1 两组确认集合起来（每题覆盖 16、K = 1 路由 + 覆盖 16，对 VisionZip-64）", "",
          "| 比较 | 数据 | 开发集 | 第一组确认集 | 第二组确认集 | 两组确认集合并 | 第二组 − 第一组 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for x in ("c16all", "cp"):
        for fam, (dev, c1, c2) in SETS.items():
            per = {s: paired(SC[s][x] - SC[s]["vz"], DR[s]) for s in (dev, c1, c2)}
            bm = {s: boot_means(SC[s][x] - SC[s]["vz"], DR[s]) for s in (c1, c2)}
            n1, n2 = len(SC[c1]["vz"]), len(SC[c2]["vz"])
            point = (per[c1][0] * n1 + per[c2][0] * n2) / (n1 + n2)
            pooled_bs = (bm[c1] * n1 + bm[c2] * n2) / (n1 + n2)
            pooled = [point, float(np.percentile(pooled_bs, 2.5)), float(np.percentile(pooled_bs, 97.5)), n1 + n2]
            gap_bs = bm[c2] - bm[c1]
            gap = [per[c2][0] - per[c1][0], float(np.percentile(gap_bs, 2.5)), float(np.percentile(gap_bs, 97.5)), n1 + n2]
            res["S1"][f"{fam} {x} - vz"] = {"dev": per[dev], "conf1": per[c1], "conf2": per[c2], "pooled": pooled, "conf2_minus_conf1": gap}
            L.append(f"| {NAMES[x]} − VisionZip-64 | {fam}（{dev} / {c1} / {c2}） | {fmt(per[dev])} | {fmt(per[c1])} | {fmt(per[c2])} | "
                     f"{fmt(pooled)} | {fmt(gap)} |")
    L += ["", "开发集上的配置是在开发集上选出的（每题覆盖 16 由 EXP-0028 的规则、K = 1 路由 + 覆盖 16 由 EXP-0026 的规则），不并入合并的估计。", ""]

    # S2 flips
    L += ["## S2 逐题翻转", "", "| 集 | 比较 | 题数 | 得分不同的题 | 变好 | 变差 | 净差（百分点） |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for bench in ("vqav2_test4", "gqa_val400b", "textvqa_noocr"):
        pairs = [("sel", "vz"), ("sel", "c16all"), ("c16all", "vz")] + ([("sel", "cp"), ("cp", "vz")] if bench != "textvqa_noocr" else [])
        for x, y in pairs:
            d = SC[bench][x] - SC[bench][y]
            r = {"n": int(len(d)), "changed": int((d != 0).sum()), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "net": 100 * float(d.mean())}
            res["S2"][f"{bench} {x} - {y}"] = r
            L.append(f"| {bench} | {NAMES[x]} − {NAMES[y]} | {r['n']} | {r['changed']}（{r['changed'] / r['n']:.1%}） | {r['better']} | {r['worse']} | {r['net']:+.2f} |")
    L.append("")

    # S3 the ranking: kept tokens of sel vs cp on unrouted questions
    L += ["## S3 不路由的题上，CLS 排序与 FastV 的注意力排序（都加覆盖 16）留下的 47 个", "",
          "| 集 | 不路由的题 | 两者留下的完全相同 | 平均 Jaccard | 平均相同个数（共 47） | 所选 − 对应的 FastV 配置（不路由的题） |", "| --- | --- | --- | --- | --- | --- |"]
    for bench in ("vqav2_dev2", "gqa_val400", "vqav2_test4", "gqa_val400b"):
        b = B[bench]
        kept = {}
        for m in ("sel", "cp"):
            exp, rid = RUNS[(bench, m)]
            k = json.load(open(f"{OUT}/{exp}/{exp}-{rid}/kept_tokens.json"))
            checks[f"{bench} {m}: one kept list of 47 per question"] = float(k["n_prefills"] == len(b["order"]) == len(k["kept"])
                                                                              and all(len(x) == 47 for x in k["kept"]))
            kept[m] = k["kept"]
        unr = ~np.asarray(b["routed"], bool)
        same_routed = np.mean([set(p) == set(q) for p, q, r in zip(kept["sel"], kept["cp"], b["routed"]) if r])
        checks[f"{bench}: routed questions keep the same 47 in both"] = float(same_routed)
        jac = np.array([len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(kept["sel"], kept["cp"])])
        inter = np.array([len(set(p) & set(q)) for p, q in zip(kept["sel"], kept["cp"])])
        ident = np.array([set(p) == set(q) for p, q in zip(kept["sel"], kept["cp"])])
        v = paired(SC[bench]["sel"] - SC[bench]["cp"], DR[bench], unr)
        r = {"n_unrouted": int(unr.sum()), "identical": float(ident[unr].mean()), "jaccard": float(jac[unr].mean()),
             "common": float(inter[unr].mean()), "sel_minus_cp_unrouted": v}
        res["S3"][bench] = r
        L.append(f"| {bench} | {r['n_unrouted']} | {r['identical']:.1%} | {r['jaccard']:.3f} | {r['common']:.1f} | {fmt(v)} |")
    L.append("")

    # S4 TextVQA reading questions
    tv = B["textvqa_noocr"]
    ocr = {}
    for line in open(f"{EVAL_DATA}/textvqa/llava_textvqa_val_ocr_qid.jsonl", encoding="utf-8"):
        x = json.loads(line)
        mt = re.search(r"Reference OCR token: (.*)\n", x["text"])
        ocr[str(x["question_id"])] = mt.group(1) if mt else ""
    reading, has_ocr = [], []
    for q in tv["order"]:
        major = Counter(a.lower().strip() for a in tv["gts"][q]["answers"]).most_common(1)[0][0]
        words = re.findall(r"[a-z0-9]+", major)
        toks = set(re.findall(r"[a-z0-9]+", ocr[q].lower()))
        reading.append(bool(words) and all(w in toks for w in words))
        has_ocr.append(bool(toks))
    reading, has_ocr = np.array(reading), np.array(has_ocr)
    groups = {"全部": np.ones(len(reading), bool), "多数答案都在 OCR token 里（读字题）": reading,
              "有 OCR token，答案不在里面": has_ocr & ~reading, "没有 OCR token": ~has_ocr}
    T, D = SC["textvqa_noocr"], DR["textvqa_noocr"]
    L += ["## S4 TextVQA（不给 OCR）：读字题与其余题", "", "分组同 SQZ-EXP-0029 的 S4。", "",
          "| 组 | 题数 | 所选的准确率 | 所选 − VisionZip-64 | 所选 − 每题覆盖 16 | 不剪枝 − 所选 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, g in groups.items():
        r = {"n": int(g.sum()), "sel_acc": 100 * float(T["sel"][g].mean()), "sel_vz": paired(T["sel"] - T["vz"], D, g),
             "sel_c16all": paired(T["sel"] - T["c16all"], D, g), "vanilla_sel": paired(T["vanilla"] - T["sel"], D, g)}
        res["S4"][name] = r
        L.append(f"| {name} | {r['n']} | {r['sel_acc']:.2f} | {fmt(r['sel_vz'])} | {fmt(r['sel_c16all'])} | {fmt(r['vanilla_sel'])} |")
    L.append("")

    # S5 group directions, development vs fresh
    L += ["## S5 分组的方向：开发集与新数据", "", "| 比较 | 组 | 开发集 | 新数据 | 同号 |", "| --- | --- | --- | --- | --- |"]
    for fam, dev, fresh in (("vqav2", "vqav2_dev2", "vqav2_test4"), ("gqa", "gqa_val400", "gqa_val400b")):
        gs = [g for g in B[fresh]["groups"] if g in B[dev]["groups"] and g not in ("[routed]", "[unrouted]")]
        for y in ("vz", "c16all"):
            agree, total = 0, 0
            for g in gs:
                vd = paired(SC[dev]["sel"] - SC[dev][y], DR[dev], B[dev]["groups"][g])
                vf = paired(SC[fresh]["sel"] - SC[fresh][y], DR[fresh], B[fresh]["groups"][g])
                if not vd or not vf:
                    continue
                same = bool(np.sign(vd[0]) == np.sign(vf[0]))
                agree, total = agree + same, total + 1
                res["S5"][f"{fam} sel - {y} {g}"] = {"dev": vd, "fresh": vf, "same_sign": same}
                L.append(f"| 所选 − {NAMES[y]} | {fam} {g} | {fmt(vd)} | {fmt(vf)} | {'是' if same else '否'} |")
            res["S5"][f"{fam} sel - {y} agreement"] = [agree, total]
            L.append(f"| 所选 − {NAMES[y]} | {fam}：同号的组 | {agree} / {total} | | |")
    L.append("")

    if any(x < 1 for x in checks.values()):
        print("STOP", {k: x for k, x in checks.items() if x < 1})
        sys.exit(1)
    L += ["## 核对", ""] + [f"- {k}: {100 * x:.0f}%" for k, x in checks.items()]
    res["checks"] = checks
    os.makedirs(OUT_DIR)
    json.dump(res, open(f"{OUT_DIR}/squeeze_0031.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(f"{OUT_DIR}/squeeze_0031.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
