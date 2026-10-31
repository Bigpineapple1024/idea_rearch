"""SQZ-EXP-0041 (post-hoc, exploratory; written after reading EXP-0041's sealed 0041.md, before any number computed here):
where the virtual fill's gain over CDPruner-64 comes from, on the development sets.

  python -B squeeze_0041.py      # writes code/output/EXP-0041/squeeze/{squeeze_0041.md,squeeze_0041.json}

Runs (sealed, read only): fill / shuffle / orig / mass EXP-0041-R01..R08, b64 = CDPruner-64 EXP-0039-R01 / R02, unpruned
EXP-0033-R03 / R04. Scores are recomputed as analyze_0041.py does (analyze_0028's loader and scoring) and must equal its
sealed table; the bootstrap is the same (images resampled within each set, 1000 draws, seed 0).
S1 (how well the copies fit): questions split into thirds by fill's mean cosine of the pruned patches to their
   representatives (layout log, per question: CDPruner's selection depends on the question); fill - b64 and fill -
   shuffle in each third.
S2 (one token copied to many slots): thirds by the largest number of slots one kept token represents (count_max).
S3 (GQA position questions by detailed type): positionChoose / positionQuery / positionVerify / positionVerifyC: accuracy
   of each run, fill - b64, fill - shuffle, orig - b64, unpruned - fill.
S4 (does copying make one object count several times?): VQAv2 "number" questions whose answer and most common human answer
   read as a whole number (digits or zero..twenty): the mean answered number, the share above / below the human answer,
   and fill - b64 of the answered number (paired, by image).
S5 (flips): fill against b64, per set and on the yes/no questions by ground truth: questions that gain, lose or stay.
"""
import csv
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0028 as a28  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
DST = f"{OUT}/EXP-0041/squeeze"
BENCHES = ("vqav2_dev2", "gqa_val400")
RUNS = {"fill": ("EXP-0041", ("R01", "R02")), "shuffle": ("EXP-0041", ("R03", "R04")), "orig": ("EXP-0041", ("R05", "R06")),
        "mass": ("EXP-0041", ("R07", "R08")), "b64": ("EXP-0039", ("R01", "R02")), "vanilla": ("EXP-0033", ("R03", "R04"))}
POS_TYPES = ("positionChoose", "positionQuery", "positionVerify", "positionVerifyC")
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                    "sixteen seventeen eighteen nineteen twenty".split())}
N_BOOT = 1000
CHECKS = {}


def as_int(text):
    t = text.strip().lower().rstrip(".")
    if re.fullmatch(r"\d+", t):
        return int(t)
    return WORDS.get(t)


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    sealed = json.load(open(f"{OUT}/EXP-0041/analysis/0041.json", encoding="utf-8"))
    res = {"S1": {}, "S2": {}, "S3": {}, "S4": {}, "S5": {}}
    for bi, bench in enumerate(BENCHES):
        b = a28.load_bench(bench)
        n = len(b["order"])
        sc, ans = {}, {}
        for meth, (exp, rids) in RUNS.items():
            d = f"{OUT}/{exp}/{exp}-{rids[bi]}"
            a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
            sc[meth] = a28.recompute(bench, b, a)
            ans[meth] = [a[q] for q in b["order"]]
            CHECKS[f"{bench} {meth}: accuracy equals analyze_0041's sealed table"] = float(
                abs(100 * sc[meth].mean() - sealed["table"][bench][meth]["accuracy"]) < 1e-9)
        lay = json.load(open(f"{OUT}/EXP-0041/EXP-0041-{RUNS['fill'][1][bi]}/layout.json", encoding="utf-8"))["prefills"]
        CHECKS[f"{bench}: fill's layout log has one record per question"] = float(len(lay) == n)
        sim = np.array([r["sim_mean"] for r in lay])
        cmax = np.array([r["count_max"] for r in lay])
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]

        def paired(x, y, mask, values=None, scale=100.0):
            v = sc if values is None else values
            d = v[x] - v[y]
            if mask.sum() == 0:
                return None
            boot = [scale * d[i][mask[i]].mean() for i in draws if mask[i].any()]
            return [scale * float(d[mask].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(mask.sum())]

        for key, val in (("S1", sim), ("S2", cmax)):
            cut = np.quantile(val, [1 / 3, 2 / 3])
            third = np.digitize(val, cut)                  # 0, 1, 2
            res[key][bench] = {"cuts": [float(c) for c in cut]}
            for t, name in enumerate(("low", "middle", "high")):
                m = third == t
                res[key][bench][name] = {"n": int(m.sum()), "range": [float(val[m].min()), float(val[m].max())],
                                         "accuracy": {k: 100 * float(s[m].mean()) for k, s in sc.items()},
                                         "fill - b64": paired("fill", "b64", m), "fill - shuffle": paired("fill", "shuffle", m),
                                         "vanilla - fill": paired("vanilla", "fill", m)}
        if bench.startswith("gqa"):
            det = np.array([b["gts"][q]["detailed"] for q in b["order"]])
            for pt in POS_TYPES:
                m = det == pt
                res["S3"][pt] = {"n": int(m.sum()), "accuracy": {k: 100 * float(s[m].mean()) for k, s in sc.items()},
                                 **{f"{x} - {y}": paired(x, y, m) for x, y in (("fill", "b64"), ("fill", "shuffle"), ("orig", "b64"), ("vanilla", "fill"))}}
            gt = np.array([b["gts"][q]["answer"] if b["gts"][q]["answer"] in ("yes", "no") else None for q in b["order"]], dtype=object)
        else:
            gt = np.array([(lambda top: top if top in ("yes", "no") else None)(Counter(b["gts"][q]["answers"]).most_common(1)[0][0])
                           if b["gts"][q]["answer_type"] == "yes/no" else None for q in b["order"]], dtype=object)
            num = np.array([b["gts"][q]["answer_type"] == "number" for q in b["order"]])
            human = np.array([as_int(Counter(b["gts"][q]["answers"]).most_common(1)[0][0]) if nm else None
                              for q, nm in zip(b["order"], num)], dtype=object)
            said = {k: np.array([as_int(t) if nm else None for t, nm in zip(v, num)], dtype=object) for k, v in ans.items()}
            ok = num & np.array([h is not None for h in human]) & np.all([np.array([s is not None for s in said[k]]) for k in RUNS], axis=0)
            vals = {k: np.array([float(s) if o else np.nan for s, o in zip(said[k], ok)]) for k in RUNS}
            h = np.array([float(x) if o else np.nan for x, o in zip(human, ok)])
            res["S4"] = {"n_number": int(num.sum()), "n_parsed": int(ok.sum()),
                         "mean_answered": {k: float(np.nanmean(v)) for k, v in vals.items()}, "mean_human": float(np.nanmean(h)),
                         "share_above_human": {k: float(np.mean(v[ok] > h[ok])) for k, v in vals.items()},
                         "share_below_human": {k: float(np.mean(v[ok] < h[ok])) for k, v in vals.items()},
                         "share_equal_human": {k: float(np.mean(v[ok] == h[ok])) for k, v in vals.items()},
                         "fill - b64 (answered number)": paired("fill", "b64", ok, {k: np.nan_to_num(v) for k, v in vals.items()}, 1.0),
                         "vanilla - b64 (answered number)": paired("vanilla", "b64", ok, {k: np.nan_to_num(v) for k, v in vals.items()}, 1.0)}
        d = sc["fill"] - sc["b64"]
        flips = {"all": np.ones(n, bool), "gt yes": gt == "yes", "gt no": gt == "no"}
        res["S5"][bench] = {k: {"n": int(m.sum()), "gain": int((d[m] > 0).sum()), "loss": int((d[m] < 0).sum()), "same": int((d[m] == 0).sum())}
                            for k, m in flips.items()}
        yn = {k: np.array([yes_no(t) for t in v], dtype=object) for k, v in ans.items()}
        res["S5"][bench]["gt no: fill and b64 answers"] = dict(Counter(f"b64 {x} / fill {y}" for x, y, g in zip(yn["b64"], yn["fill"], gt) if g == "no"))
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/squeeze_0041.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x, k=2: "—" if x is None else f"{x[0]:+.{k}f} [{x[1]:+.{k}f}, {x[2]:+.{k}f}]（{x[3]}）"   # noqa: E731
    L = ["# SQZ-EXP-0041（事后探索，不是预注册）：虚拟填补的增益从哪来（开发集）", "", "## 核对", ""] + [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    for key, what in (("S1", "fill 的代表与剪掉的 patch 的平均余弦（每题）"), ("S2", "一个保留 token 最多代表的槽数（每题）")):
        L += ["", f"## {key}：按{what}三等分", ""]
        for bench in BENCHES:
            x = res[key][bench]
            for name in ("low", "middle", "high"):
                y = x[name]
                L.append(f"- {bench} {name}（{y['n']} 题，{y['range'][0]:.3f}～{y['range'][1]:.3f}）：fill − b64 {fmt(y['fill - b64'])}；fill − shuffle {fmt(y['fill - shuffle'])}；"
                         f"不剪枝 − fill {fmt(y['vanilla - fill'])}；准确率 " + "；".join(f"{k} {v:.2f}" for k, v in y["accuracy"].items()))
    L += ["", "## S3：GQA 位置题按细分题型", ""]
    for pt, y in res["S3"].items():
        L.append(f"- {pt}（{y['n']} 题）：准确率 " + "；".join(f"{k} {v:.2f}" for k, v in y["accuracy"].items())
                 + "；" + "；".join(f"{p} {fmt(y[p])}" for p in ("fill - b64", "fill - shuffle", "orig - b64", "vanilla - fill")))
    s4 = res["S4"]
    L += ["", "## S4：VQAv2 计数题（答案与人的答案都读得成整数）", "",
          f"- 计数题 {s4['n_number']} 题，六个运行与人的答案都读得成整数的 {s4['n_parsed']} 题；人的答案平均 {s4['mean_human']:.3f}",
          "- 答的数平均：" + "；".join(f"{k} {v:.3f}" for k, v in s4["mean_answered"].items()),
          "- 比人的答案大的比例：" + "；".join(f"{k} {v:.3f}" for k, v in s4["share_above_human"].items()),
          "- 比人的答案小的比例：" + "；".join(f"{k} {v:.3f}" for k, v in s4["share_below_human"].items()),
          "- 与人的答案相同的比例：" + "；".join(f"{k} {v:.3f}" for k, v in s4["share_equal_human"].items()),
          f"- 答的数 fill − b64 {fmt(s4['fill - b64 (answered number)'], 3)}；不剪枝 − b64 {fmt(s4['vanilla - b64 (answered number)'], 3)}"]
    L += ["", "## S5：fill 对 b64 的逐题翻转", ""]
    for bench in BENCHES:
        x = res["S5"][bench]
        L.append(f"- {bench}：" + "；".join(f"{k} {x[k]['n']} 题：变好 {x[k]['gain']}、变差 {x[k]['loss']}、不变 {x[k]['same']}" for k in ("all", "gt yes", "gt no")))
        L.append(f"  - “否”题上 b64 与 fill 的回答：" + "；".join(f"{k} {v}" for k, v in sorted(x["gt no: fill and b64 answers"].items())))
    open(f"{DST}/squeeze_0041.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {DST}; {len(CHECKS)} checks")


if __name__ == "__main__":
    main()
