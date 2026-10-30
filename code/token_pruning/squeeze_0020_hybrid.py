"""EXP-0020 squeeze follow-up (exploratory; written after reading squeeze_0020's output, so everything here is post hoc
on both test1 and test2):

  python squeeze_0020_hybrid.py      # writes code/output/EXP-0020/squeeze_hybrid/{squeeze_0020_hybrid.md,.json}

(a) routed questions that start with "do" lost on test2 (squeeze_0020 section 2): the same subset on test1, with examples;
(b) the gap to VisionZip-64 lies in the questions the router leaves to A (squeeze_0020 section 4): a composition that
    sends those questions to VisionZip-64 instead of A, scored from the existing runs on test1 and test2. It is not a
    method of this study and not tested; a lead for later work only.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from squeeze_0020 import OUT, ROUTE_START, RUNS, RUNS_TEST1, TYPES, fmt, load_set, majority, paired   # noqa: E402


def main():
    out_dir = f"{OUT}/EXP-0020/squeeze_hybrid"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sets = {"test1": load_set("EXP-0018", RUNS_TEST1, "llava_vqav2_val_test_annotations.jsonl", "llava_vqav2_val_test.jsonl"),
            "test2": load_set("EXP-0020", RUNS, "llava_vqav2_val_test2_annotations.jsonl", "llava_vqav2_val_test2.jsonl")}
    res, L = {}, ["# EXP-0020 squeeze 追加（探索性；看过 squeeze_0020 的输出后写，test1、test2 上都是事后分析）", ""]

    # (a) "do"-start questions
    L += ["## (a) 以 do 开头的题（被路由）", "", "| 子集 | 题数 | yes/no 占比 | 64：A′+B+V − A | 128：A′+B+V − A |", "| --- | --- | --- | --- | --- |"]
    res["do_start"] = {}
    for name, s in sets.items():
        m = np.array([s["qtext"][q].startswith("do ") for q in s["qids"]])
        r = {"n": int(m.sum()), "yesno_share": float((s["atype"][m] == "yes/no").mean()),
             "64": paired(s, "AV_64", "A_64", m), "128": paired(s, "AV_128", "A_128", m)}
        res["do_start"][name] = r
        L.append(f"| {name} | {r['n']} | {r['yesno_share']:.2f} | {fmt(r['64'])} | {fmt(r['128'])} |")
    s = sets["test2"]
    m = np.array([s["qtext"][q].startswith("do ") for q in s["qids"]])
    idx = np.flatnonzero(m)
    flips = [i for i in idx if s["scores"]["AV_64"][i] != s["scores"]["A_64"][i]]
    res["do_start_test2_flips"] = [{"question": s["qtext"][s["qids"][i]], "majority": str(majority(s["gts"][s["qids"][i]]["answers"])),
                                    "A_64": s["answers"]["A_64"][i], "AV_64": s["answers"]["AV_64"][i],
                                    "A_score": float(s["scores"]["A_64"][i]), "AV_score": float(s["scores"]["AV_64"][i])} for i in flips]
    L += ["", f"test2 上 64 档 A′+B+V 与 A 得分不同的 do 题（{len(flips)} 题）：", "", "| 题目 | 多数答案 | A | A′+B+V |", "| --- | --- | --- | --- |"]
    L += [f"| {r['question']} | {r['majority']} | {r['A_64']} ({r['A_score']:.1f}) | {r['AV_64']} ({r['AV_score']:.1f}) |" for r in res["do_start_test2_flips"]]
    # 2026-10-30 note: no rstrip(".") here, unlike the yes rate in squeeze_0020.py (line 172), so "yes." counts differently
    yes_rate = {k: float(np.mean([a.strip().lower() == "yes" for a, mm in zip(s["answers"][k], m) if mm])) for k in ("vanilla", "A_64", "AV_64")}
    maj_yes = float(np.mean([majority(s["gts"][q]["answers"]) == "yes" for q, mm in zip(s["qids"], m) if mm]))
    res["do_start_test2_yes_rate"] = {**yes_rate, "majority_yes_share": maj_yes}
    L += ["", "test2 的 do 题上答 yes 的比例：" + "，".join(f"{k} {v:.2f}" for k, v in yes_rate.items()) + f"；多数人答 yes 的比例 {maj_yes:.2f}"]

    # (b) composition: A'+B+V when routed, VisionZip-64 otherwise
    L += ["", "## (b) 组合：被路由的题用 A′+B+V-64，其余用 VisionZip-64（由现有运行拼出，未作为方法测试）", "",
          "| 子集 | 组合 | 组合 − VisionZip-64 | 组合 − 路由-64 | 组合 − A-64 |", "| --- | --- | --- | --- | --- |"]
    res["composition"] = {}
    for name, s in sets.items():
        route = np.array([s["qtext"][q].startswith(ROUTE_START) for q in s["qids"]])
        s["scores"]["comp"] = np.where(route, s["scores"]["AV_64"], s["scores"]["vz64"])
        s["scores"]["router_64x"] = np.where(route, s["scores"]["AV_64"], s["scores"]["A_64"])
        r = {"accuracy": 100 * float(s["scores"]["comp"].mean()), "minus_vz": paired(s, "comp", "vz64"),
             "minus_router": paired(s, "comp", "router_64x"), "minus_A": paired(s, "comp", "A_64"),
             "by_type_minus_vz": {t: paired(s, "comp", "vz64", s["atype"] == t) for t in TYPES}, "routed_share": float(route.mean())}
        res["composition"][name] = r
        L.append(f"| {name} | {r['accuracy']:.2f} | {fmt(r['minus_vz'])} | {fmt(r['minus_router'])} | {fmt(r['minus_A'])} |")
    L += [""] + [f"- {name} 按答案类型的 组合 − VisionZip-64：" + "；".join(f"{t} {fmt(v)}" for t, v in r["by_type_minus_vz"].items())
                 for name, r in res["composition"].items()]
    L += ["- test1 上的“路由-64”由 EXP-0018 的 A′+B+V 与 A 两次运行按同一规则拼出（squeeze_0018 第 4 节的做法），test2 上与实际运行 R02 逐题相同。"]
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/squeeze_0020_hybrid.json", "w"), indent=2)
    open(f"{out_dir}/squeeze_0020_hybrid.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
