"""SQZ-EXP-0029（看过预注册分析之后，探索性，不是判据）：IDEA-0006 的确认为什么没有复现开发集的增益，TextVQA 上覆盖为什么变差。
只读已封存的运行（EXP-0028、EXP-0029 与它们复用的运行），CPU；重抽同 analyze_0029（按图像 1000 次，种子 0）。

  python -B squeeze_0029.py     # 写 code/output/EXP-0029/squeeze/{squeeze_0029.md,squeeze_0029.json}；目录已存在就停

S1 开发集 → 确认集：各配置与 VisionZip-64 的差，按答案类型（VQAv2）/ 结构类型（GQA）拆成“题型构成的变化”与“题型内效应的变化”。
S2 逐题翻转：每题覆盖 16 与 VisionZip-64 得分不同的题有多少，变好与变差各多少（开发集、确认集、TextVQA）。
S3 开发集上的选择：三个候选的两集平均 Δ̄ 与所选的 Δ̄（来自 0028.json）。
S4 TextVQA：按“多数答案的词是否都在 OCR token 里”（读字题）分组，每题覆盖 16 − VisionZip-64、不剪枝 − VisionZip-64。
S5 VQAv2：被覆盖的不路由题按“题目是否点名图中的 COCO 物体”与颜色题分组（同 EXP-0028 的 locate_dev2b.py），开发集与确认集并列。
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0024 as a24  # noqa: E402
import analyze_0025 as a25  # noqa: E402
import analyze_0028 as a28  # noqa: E402
import analyze_0029 as a29  # noqa: E402
from squeeze_0018 import EVAL_DATA, OUT  # noqa: E402

OUT_DIR = f"{OUT}/EXP-0029/squeeze"
a28.FILES.update(a29.FILES)
DEV = {"vqav2_test3": "vqav2_dev2", "gqa_testdev": "gqa_val400"}
RUNS = dict(a28.RUNS)                                     # development runs (EXP-0028 and reused)
for (bench, meth), v in a29.run_table("c16all").items():  # confirmation runs; the selected one is c16all
    RUNS[(bench, "c16all" if meth == "sel" else meth)] = v
VARIANTS = ("c16all", "c16", "c16ctr", "t16")
NAMES = {"c16all": "每题覆盖 16（所选）", "c16": "覆盖 16", "c16ctr": "居中框覆盖 16", "t16": "不围框", "vz": "VisionZip-64",
         "vanilla": "不剪枝", "router": "K = 2 路由"}


def scores(bench, b, meths):
    sc = {}
    for m in meths:
        exp, rid = RUNS[(bench, m)]
        a = {}
        for line in open(f"{OUT}/{exp}/{exp}-{rid}/answers.jsonl", encoding="utf-8"):
            x = json.loads(line)
            a[str(x["question_id"])] = x["text"]
        sc[m] = a28.recompute(bench, b, a) if bench in a28.FILES else a25.recompute(bench, b, a)
    return sc


def draws_of(b):
    by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
    rng = np.random.default_rng(0)
    return [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(a29.N_BOOT)]


def paired(d, draws, m=None):
    m = np.ones(len(d), bool) if m is None else m
    if not m.any():
        return None
    bs = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
    return [100 * float(d[m].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(m.sum())]


fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）" if v else "—"   # noqa: E731


def main():
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} exists; results are never overwritten")
    sealed = {"conf": json.load(open(f"{OUT}/EXP-0029/analysis/0029.json", encoding="utf-8")),
              "dev": json.load(open(f"{OUT}/EXP-0028/analysis/0028.json", encoding="utf-8"))}
    res, checks, L = {"S1": {}, "S2": {}, "S3": {}, "S4": {}, "S5": {}}, {}, ["# SQZ-EXP-0029：确认为什么没有复现开发集的增益（探索性）", ""]
    B, SC, DR = {}, {}, {}
    for conf, dev in DEV.items():
        for bench in (dev, conf):
            B[bench] = b = a28.load_bench(bench)
            SC[bench] = scores(bench, b, ("vz",) + VARIANTS)
            DR[bench] = draws_of(b)
            for x in VARIANTS:                            # the overall pairs must equal the sealed analyses
                key, src = (f"{bench} {x} - vz", sealed["dev"]["pairs"]) if bench == dev else \
                           (f"{bench} {'sel' if x == 'c16all' else x} - vz", sealed["conf"]["pairs"])
                v = paired(SC[bench][x] - SC[bench]["vz"], DR[bench])
                checks[f"{bench} {x} - vz equals the sealed analysis"] = float(src.get(key) is not None and np.allclose(v, src[key]))

    # S1: composition vs within-type change
    L += ["## S1 开发集 → 确认集：题型构成的变化，还是题型内效应的变化", "",
          "差 = Σ 题型占比 × 题型内的差（题型互斥且覆盖全部题）。确认集的差 − 开发集的差 = 构成的变化（Σ（占比变化）× 开发集的题型内差）+ 题型内的变化（Σ 确认集占比 ×（题型内差的变化））。", ""]
    for conf, dev in DEV.items():
        part = [g for g in B[conf]["groups"] if (g in ("[number]", "[other]", "[yes/no]") if conf.startswith("vqav2") else g.startswith("[structural "))]
        assert all(g in B[dev]["groups"] for g in part)
        for bench in (dev, conf):
            tot = sum(B[bench]["groups"][g].astype(int) for g in part)
            checks[f"{bench} S1 groups partition the questions"] = float((tot == 1).all())
        L += [f"### {dev} → {conf}", "", "| 配置 | 开发集的差 | 确认集的差 | 构成的变化 | 题型内的变化 |", "| --- | --- | --- | --- | --- |"]
        rows = {}
        for x in VARIANTS:
            w, dlt = {}, {}
            for bench in (dev, conf):
                d = SC[bench][x] - SC[bench]["vz"]
                w[bench] = {g: float(B[bench]["groups"][g].mean()) for g in part}
                dlt[bench] = {g: 100 * float(d[B[bench]["groups"][g]].mean()) for g in part}
            tot = {bench: sum(w[bench][g] * dlt[bench][g] for g in part) for bench in (dev, conf)}
            comp = sum((w[conf][g] - w[dev][g]) * dlt[dev][g] for g in part)
            within = sum(w[conf][g] * (dlt[conf][g] - dlt[dev][g]) for g in part)
            checks[f"{conf} {x} S1 decomposition adds up"] = float(abs(tot[conf] - tot[dev] - comp - within) < 1e-9)
            rows[x] = {"dev": tot[dev], "conf": tot[conf], "composition": comp, "within": within,
                       "share": w, "delta_by_type": dlt}
            L.append(f"| {NAMES[x]} | {tot[dev]:+.2f} | {tot[conf]:+.2f} | {comp:+.2f} | {within:+.2f} |")
        L += ["", "各题型（占比：开发集 → 确认集；每题覆盖 16 − VisionZip-64：开发集 → 确认集）：", ""]
        L += [f"- {g}：{rows['c16all']['share'][dev][g]:.3f} → {rows['c16all']['share'][conf][g]:.3f}；"
              f"{rows['c16all']['delta_by_type'][dev][g]:+.2f} → {rows['c16all']['delta_by_type'][conf][g]:+.2f}" for g in part]
        L.append("")
        res["S1"][conf] = rows

    # S2: per-question flips
    L += ["## S2 逐题翻转：每题覆盖 16 与 VisionZip-64", "", "| 集 | 题数 | 得分不同的题 | 变好 | 变差 | 净差（百分点） |", "| --- | --- | --- | --- | --- | --- |"]
    tv = a25.load_bench("textvqa_noocr")
    SC["textvqa_noocr"] = scores("textvqa_noocr", tv, ("vz", "c16all", "vanilla", "router"))
    DR["textvqa_noocr"] = draws_of(tv)
    for bench in ("vqav2_dev2", "vqav2_test3", "gqa_val400", "gqa_testdev", "textvqa_noocr"):
        d = SC[bench]["c16all"] - SC[bench]["vz"]
        r = {"n": int(len(d)), "changed": int((d != 0).sum()), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "net": 100 * float(d.mean())}
        res["S2"][bench] = r
        L.append(f"| {bench} | {r['n']} | {r['changed']}（{r['changed'] / r['n']:.1%}） | {r['better']} | {r['worse']} | {r['net']:+.2f} |")
    L.append("")

    # S3: selection among three candidates on the development sets
    c = sealed["dev"]["criteria"]
    res["S3"] = {"mean": c["mean"], "selected": c["selected"]}
    m = c["mean"]
    L += ["## S3 开发集上的选择", "",
          f"- 三个候选的两集平均 Δ̄：" + "、".join(f"{k} {v:+.2f}" for k, v in m.items()) + f"；所选 {c['selected']} {m[c['selected']]:+.2f}，"
          f"三个的平均 {np.mean(list(m.values())):+.2f}。",
          f"- 所选在确认集上的两集平均：{np.mean([res['S1'][k]['c16all']['conf'] for k in DEV]):+.2f}。", ""]

    # S4: TextVQA reading questions
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
    L += ["## S4 TextVQA（不给 OCR）：读字题与其余题", "",
          "读字题：10 个标注答案中出现最多的那个，拆成字母数字词后全部出现在该题的 OCR token 里（OCR token 来自 LLaVA 的 val_v051_ocr 提示，本实验运行时不给模型）。", "",
          "| 组 | 题数 | VisionZip-64 准确率 | 每题覆盖 16 − VisionZip-64 | 不剪枝 − VisionZip-64 | VisionZip-64 − K = 2 路由 |", "| --- | --- | --- | --- | --- | --- |"]
    T = SC["textvqa_noocr"]
    for name, g in groups.items():
        r = {"n": int(g.sum()), "vz_acc": 100 * float(T["vz"][g].mean()),
             "c16all_vz": paired(T["c16all"] - T["vz"], DR["textvqa_noocr"], g),
             "vanilla_vz": paired(T["vanilla"] - T["vz"], DR["textvqa_noocr"], g),
             "vz_router": paired(T["vz"] - T["router"], DR["textvqa_noocr"], g)}
        res["S4"][name] = r
        L.append(f"| {name} | {r['n']} | {r['vz_acc']:.2f} | {fmt(r['c16all_vz'])} | {fmt(r['vanilla_vz'])} | {fmt(r['vz_router'])} |")
    v = paired(T["c16all"] - T["vz"], DR["textvqa_noocr"])
    checks["textvqa_noocr c16all - vz equals the sealed analysis"] = float(np.allclose(v, sealed["conf"]["pairs"]["textvqa_noocr sel - vz"]))
    L.append("")

    # S5: VQAv2 named / not named, colour
    L += ["## S5 VQAv2：被覆盖的不路由题，按是否点名图中的 COCO 物体与颜色题分组", "",
          "做法同 EXP-0028 的 locate_dev2b.py（analyze_0024.topic_masks）；覆盖与否取覆盖 16 的记录（开发集 EXP-0028-R13，确认集 EXP-0029-R09）。", "",
          "| 组 | 集 | 题数 | 覆盖 16 − VisionZip-64 | 覆盖 16 − 居中框 | 不围框 − VisionZip-64 | 每题覆盖 16 − VisionZip-64 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for bench in ("vqav2_dev2", "vqav2_test3"):
        b = B[bench]
        b["text"] = {str(q["question_id"]): q["text"] for q in map(json.loads, open(a28.FILES[bench][0], encoding="utf-8"))}
        exp, rid = RUNS[(bench, "c16")]
        recs = json.load(open(f"{OUT}/{exp}/{exp}-{rid}/vz.json"))["records"]
        cov = np.array([bool(x["cover"]) for x in recs])
        named = np.array([mk is not None for mk in a24.topic_masks(b)])
        colour = b["groups"]["[colour]"]
        G = {"点名图中 COCO 物体": cov & named, "没有点名": cov & ~named, "没有点名的颜色题": cov & ~named & colour,
             "没有点名的其他题": cov & ~named & ~colour, "点名的颜色题": cov & named & colour}
        s, dr = SC[bench], DR[bench]
        res["S5"][bench] = {}
        for name, g in G.items():
            r = {k: paired(s[x] - s[y], dr, g) for k, (x, y) in
                 {"c16_vz": ("c16", "vz"), "c16_c16ctr": ("c16", "c16ctr"), "t16_vz": ("t16", "vz"), "c16all_vz": ("c16all", "vz")}.items()}
            res["S5"][bench][name] = r
            L.append(f"| {name} | {bench} | {int(g.sum())} | {fmt(r['c16_vz'])} | {fmt(r['c16_c16ctr'])} | {fmt(r['t16_vz'])} | {fmt(r['c16all_vz'])} |")
    L.append("")

    if any(x < 1 for x in checks.values()):
        print("STOP", {k: x for k, x in checks.items() if x < 1})
        sys.exit(1)
    L += ["## 核对", ""] + [f"- {k}: {100 * x:.0f}%" for k, x in checks.items()]
    res["checks"] = checks
    os.makedirs(OUT_DIR)
    json.dump(res, open(f"{OUT_DIR}/squeeze_0029.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(f"{OUT_DIR}/squeeze_0029.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
