"""EXP-0022 (plan v1): the question-form router, rule frozen, on GQA testdev-balanced (a third benchmark); router vs A at
64 (Q1) and 128 (Q2), and A'+B+V vs A by gold answer kind (Q3, replication of F38). Also reported (written before any run):
groups by GQA structural type, routed either/or questions vs the other routed questions, router - A'+B+V, the per-question
oracle against the same bound for A-64 and VisionZip-64, the yes rate on yes/no questions, and vanilla, VisionZip-64 and
SparseVLM-64 as references.

  python analyze_0022.py      # writes code/output/EXP-0022/analysis/{0022.md,0022.json}

Per-question scores come from each run's gqa_scores.jsonl and are recomputed here from answers.jsonl as a check.
Paired differences: images resampled with replacement 1000 times (seed 0), as in analyze_0020.py.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_0016 import counts_check   # noqa: E402
from analyze_query import row   # noqa: E402
from analyze_repope import EVAL_DATA, OUT   # noqa: E402

RUNS = {"A_64": "R01", "router_64": "R02", "A'+B+V_64": "R03", "A_128": "R04", "router_128": "R05",
        "vanilla": "R06", "visionzip64": "R07", "sparsevlm64": "R08"}
PAIRS = [("router_64", "A_64"), ("router_128", "A_128"), ("A'+B+V_64", "A_64"), ("router_64", "A'+B+V_64"),
         ("router_64", "visionzip64"), ("router_64", "sparsevlm64"), ("A_64", "visionzip64"), ("router_64", "vanilla")]
KINDS = ("yes/no", "other")
STRUCT = ("verify", "logical", "choose", "compare", "query")
N_BOOT = 1000
VIS = f"{OUT}/EXP-0015/visibility/visibility.npy"
LLAVA_PAPER_GQA = 62.0      # LLaVA-1.5-7B, the reproduction check of the plan


def main():
    out_dir = f"{OUT}/EXP-0022/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sys.path.insert(0, "D:/6CJNM007/idea_rearch/source/LLaVA")
    import run_eval
    qfile = f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced.jsonl"
    questions = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    gts = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{EVAL_DATA}/gqa/llava_gqa_testdev_balanced_annotations.jsonl", encoding="utf-8")}
    order = [str(q["question_id"]) for q in questions]
    text = {str(q["question_id"]): q["text"] for q in questions}
    routed_rule = np.array([run_eval.routes_to_query(q["text"]) for q in questions])
    qids = sorted(gts)
    pos = {q: i for i, q in enumerate(order)}
    images = sorted({g["image"] for g in gts.values()})
    img_idx = {im: k for k, im in enumerate(images)}
    img_of = np.array([img_idx[gts[q]["image"]] for q in qids])
    kind = np.array(["yes/no" if gts[q]["answer"] in ("yes", "no") else "other" for q in qids])
    struct = np.array([gts[q]["structural"] for q in qids])
    either_or = np.array([bool(re.search(r"\bor\b", text[q].split("\n")[0].lower())) for q in qids])
    routed = np.array([routed_rule[pos[q]] for q in qids])
    checks = {"R03 counts = round(30 S)": counts_check("EXP-0022/EXP-0022-R03", qfile, 30, VIS)}
    scores, answers, kept = {}, {}, {}
    for name, rid in RUNS.items():
        run = f"EXP-0022/EXP-0022-{rid}"
        ans = {str(json.loads(l)["question_id"]): json.loads(l)["text"] for l in open(f"{OUT}/{run}/answers.jsonl")}
        s = np.array([float(ans[q].rstrip(".").lower() == gts[q]["answer"]) for q in qids])
        logged = {json.loads(l)["question_id"]: json.loads(l)["score"] for l in open(f"{OUT}/{run}/gqa_scores.jsonl")}
        checks[f"{name}: recomputed scores equal gqa_scores.jsonl"] = float(np.allclose(s, [logged[q] for q in qids]))
        checks[f"{name}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(row(run)["headline"])) < 1e-6)
        scores[name], answers[name] = s, [ans[q] for q in qids]
        if os.path.exists(f"{OUT}/{run}/kept_tokens.json"):
            kept[name] = json.load(open(f"{OUT}/{run}/kept_tokens.json"))
    checks["vanilla within 1 point of LLaVA-1.5's reported 62.0"] = float(abs(100 * scores["vanilla"].mean() - LLAVA_PAPER_GQA) <= 1.0)
    r, a, av = kept["router_64"], kept["A_64"], kept["A'+B+V_64"]
    checks["router_64: logged decisions equal the rule on the question text"] = float(np.mean(np.array(r["routed"]) == routed_rule))
    checks["router_64: kept set = A'+B+V's when routed, A's otherwise"] = float(np.mean(
        [r["kept"][i] == (av["kept"][i] if routed_rule[i] else a["kept"][i]) for i in range(len(order))]))
    checks["router_64: answer = A'+B+V's when routed, A's otherwise"] = float(np.mean(
        [answers["router_64"][i] == (answers["A'+B+V_64"][i] if routed[i] else answers["A_64"][i]) for i in range(len(qids))]))
    for b in (64, 128):
        rb = kept[f"router_{b}"]
        if b == 128:
            checks["router_128: logged decisions equal the rule on the question text"] = float(np.mean(np.array(rb["routed"]) == routed_rule))
        checks[f"router_{b}: question part empty exactly when unrouted"] = float(np.mean(
            [(len(k) > 0) == bool(d) for k, d in zip(rb["kept_by_object_name"], routed_rule)]))
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]

    def paired(x, y, mask=None):
        d = scores[x] - scores[y]
        m = np.ones(len(d), bool) if mask is None else mask
        if m.sum() == 0:
            return None
        boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

    pairs = {}
    for x, y in PAIRS:
        pairs[f"{x} - {y}"] = paired(x, y)
        for k in KINDS:
            pairs[f"{x} - {y} [{k}]"] = paired(x, y, kind == k)
    for s in STRUCT:                                            # reported, not criteria
        pairs[f"router_64 - A_64 [structural {s}]"] = paired("router_64", "A_64", struct == s)
        pairs[f"A'+B+V_64 - A_64 [structural {s}]"] = paired("A'+B+V_64", "A_64", struct == s)
    pairs["A'+B+V_64 - A_64 [routed, either/or]"] = paired("A'+B+V_64", "A_64", routed & either_or)
    pairs["A'+B+V_64 - A_64 [routed, not either/or]"] = paired("A'+B+V_64", "A_64", routed & ~either_or)
    pairs["A'+B+V_64 - A_64 [not routed]"] = paired("A'+B+V_64", "A_64", ~routed)
    for x, y in (("A_64", "A'+B+V_64"), ("A_64", "visionzip64")):   # per-question oracle (an upper bound, not a method)
        scores[f"oracle({x}, {y})"] = np.maximum(scores[x], scores[y])
        pairs[f"oracle({x}, {y}) - {x}"] = paired(f"oracle({x}, {y})", x)
    yn = kind == "yes/no"
    yes_rate = {n: float(np.mean([answers[n][i].rstrip(".").lower() == "yes" for i in np.flatnonzero(yn)])) for n in RUNS}
    yes_rate["gold"] = float(np.mean([gts[qids[i]]["answer"] == "yes" for i in np.flatnonzero(yn)]))
    table = {n: {"accuracy": 100 * float(s.mean()), **{k: 100 * float(s[kind == k].mean()) for k in KINDS}} for n, s in scores.items()}
    q1 = pairs["router_64 - A_64"][1] > 0
    q2 = pairs["router_128 - A_128"][1] > 0
    q3 = pairs["A'+B+V_64 - A_64 [yes/no]"][1] > 0 and pairs["A'+B+V_64 - A_64 [other]"][2] < 0
    os.makedirs(out_dir)
    json.dump({"checks": checks, "table": table, "pairs": pairs, "Q1": q1, "Q2": q2, "Q3": q3, "n_questions": len(qids),
               "n_images": len(images), "routed_share": float(routed.mean()), "routed_either_or": int((routed & either_or).sum()),
               "routed_share_by_kind": {k: float(routed[kind == k].mean()) for k in KINDS},
               "n_by_kind": {k: int((kind == k).sum()) for k in KINDS}, "n_by_structural": {s: int((struct == s).sum()) for s in STRUCT},
               "yes_rate_on_yesno": yes_rate}, open(f"{out_dir}/0022.json", "w"), indent=2)
    L = [f"# EXP-0022（计划 v1）：GQA testdev-balanced（{len(qids)} 题，{len(images)} 张图）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", f"路由比例 {routed.mean():.3f}（yes/no {routed[yn].mean():.3f}，other {routed[~yn].mean():.3f}）；被路由的二选一题 {int((routed & either_or).sum())}"]
    L += ["", "## 准确率", "", "| 配置 | 全部 | yes/no | other |", "| --- | --- | --- | --- |"]
    L += [f"| {n} | {t['accuracy']:.2f} | {t['yes/no']:.2f} | {t['other']:.2f} |" for n, t in table.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间] | 题数 |", "| --- | --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] | {v[3]} |" for k, v in pairs.items() if v]
    L += ["", "yes/no 题上答 yes 的比例：" + "，".join(f"{k} {v:.3f}" for k, v in yes_rate.items())]
    L += ["", f"**判据**：Q1（64 档路由高于 A）{'成立' if q1 else '不成立'}；Q2（128 档）{'成立' if q2 else '不成立'}；"
              f"Q3（64 档 A′+B+V 在 yes/no 上高于 A、在其余题上低于 A）{'成立' if q3 else '不成立'}。"]
    open(f"{out_dir}/0022.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
