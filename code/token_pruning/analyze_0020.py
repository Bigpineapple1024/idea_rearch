"""EXP-0020 (plan v1.1): the question-form router on a fresh VQAv2 val subset (test2); router vs A at 64 (Q1) and 128 (Q2),
and the replication of EXP-0018's split by answer type (Q3). Also reported (v1.1, before any run): the per-question oracle
of A and A'+B+V (the better of the two on each question), an upper bound for any router over these two branches; and
(v1.2) vanilla, VisionZip-64 and SparseVLM-64 on the same questions as references.

  python analyze_0020.py      # writes code/output/EXP-0020/analysis/{0020.md,0020.json}

Per-question scores come from each run's vqa_scores.jsonl and are recomputed here with the same evaluator as a check.
Paired differences: images resampled with replacement 1000 times (seed 0), as in analyze_0018.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_0016 import counts_check   # noqa: E402
from analyze_query import row   # noqa: E402
from analyze_repope import EVAL_DATA, OUT   # noqa: E402

RUNS = {"A_64": "R01", "router_64": "R02", "A'+B+V_64": "R03", "A_128": "R04", "router_128": "R05", "A'+B+V_128": "R06",
        "vanilla": "R07", "visionzip64": "R08", "sparsevlm64": "R09"}           # R07-R09: references (plan v1.2)
PAIRS = [("router_64", "A_64"), ("router_128", "A_128"), ("A'+B+V_64", "A_64"), ("A'+B+V_128", "A_128"),
         ("router_64", "A'+B+V_64"), ("router_128", "A'+B+V_128"), ("router_64", "visionzip64"), ("router_64", "sparsevlm64"),
         ("A_64", "visionzip64"), ("router_64", "vanilla")]
TYPES = ("yes/no", "number", "other")
N_BOOT = 1000
VIS = f"{OUT}/EXP-0015/visibility/visibility.npy"


def main():
    out_dir = f"{OUT}/EXP-0020/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sys.path.insert(0, "D:/6CJNM007/idea_rearch/source/LLaVA")
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    import run_eval
    ev = TextVQAAccuracyEvaluator()
    qfile = f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test2.jsonl"
    questions = [json.loads(l) for l in open(qfile)]
    gts = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test2_annotations.jsonl")}
    order = [q["question_id"] for q in questions]                 # the loader's order: kept_tokens.json follows it
    routed_rule = np.array([run_eval.routes_to_query(q["text"]) for q in questions])
    qids = sorted(gts)
    pos = {q: i for i, q in enumerate(order)}
    images = sorted({g["image"] for g in gts.values()})
    img_of = np.array([images.index(gts[q]["image"]) for q in qids])
    atype = np.array([gts[q]["answer_type"] for q in qids])
    checks = {"R03 counts = round(30 S)": counts_check("EXP-0020/EXP-0020-R03", qfile, 30, VIS),
              "R06 counts = round(98 S)": counts_check("EXP-0020/EXP-0020-R06", qfile, 98, VIS)}
    scores, answers, kept = {}, {}, {}
    for name, rid in RUNS.items():
        run = f"EXP-0020/EXP-0020-{rid}"
        ans = {json.loads(l)["question_id"]: json.loads(l)["text"] for l in open(f"{OUT}/{run}/answers.jsonl")}
        s = np.array([ev._compute_answer_scores(gts[q]["answers"]).get(ev.answer_processor(ans[q]), 0.0) for q in qids])
        logged = {json.loads(l)["question_id"]: json.loads(l)["score"] for l in open(f"{OUT}/{run}/vqa_scores.jsonl")}
        checks[f"{name}: recomputed scores equal vqa_scores.jsonl"] = float(np.allclose(s, [logged[q] for q in qids]))
        checks[f"{name}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(row(run)["headline"])) < 1e-6)
        scores[name], answers[name] = s, [ans[q] for q in qids]
        if os.path.exists(f"{OUT}/{run}/kept_tokens.json"):                  # FastV-family runs only
            kept[name] = json.load(open(f"{OUT}/{run}/kept_tokens.json"))
    routed = np.array([routed_rule[pos[q]] for q in qids])
    for b, keep in ((64, 30), (128, 98)):
        r, a, av = kept[f"router_{b}"], kept[f"A_{b}"], kept[f"A'+B+V_{b}"]
        checks[f"router_{b}: logged decisions equal the rule on the question text"] = float(np.mean(np.array(r["routed"]) == routed_rule))
        same_kept = [r["kept"][i] == (av["kept"][i] if routed_rule[i] else a["kept"][i]) for i in range(len(order))]
        checks[f"router_{b}: kept set = A'+B+V's when routed, A's otherwise"] = float(np.mean(same_kept))
        same_ans = [answers[f"router_{b}"][i] == (answers[f"A'+B+V_{b}"][i] if routed[i] else answers[f"A_{b}"][i]) for i in range(len(qids))]
        checks[f"router_{b}: answer = A'+B+V's when routed, A's otherwise"] = float(np.mean(same_ans))
        checks[f"router_{b}: question part empty exactly when unrouted"] = float(np.mean(
            [(len(k) > 0) == bool(d) for k, d in zip(r["kept_by_object_name"], routed_rule)]))
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]
    pairs = {}
    for x, y in PAIRS:
        d = scores[x] - scores[y]
        boot = [100 * d[idx].mean() for idx in draws]
        pairs[f"{x} - {y}"] = [100 * float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        for t in TYPES:
            m = atype == t
            bt = [100 * d[idx][m[idx]].mean() for idx in draws]
            pairs[f"{x} - {y} [{t}]"] = [100 * float(d[m].mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    for b in (64, 128):                     # plan v1.1: per-question oracle of the two branches (an upper bound, not a method)
        scores[f"oracle_{b}"] = np.maximum(scores[f"A_{b}"], scores[f"A'+B+V_{b}"])
        d = scores[f"oracle_{b}"] - scores[f"A_{b}"]
        boot = [100 * d[idx].mean() for idx in draws]
        pairs[f"oracle_{b} - A_{b}"] = [100 * float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    table = {n: {"accuracy": 100 * float(s.mean()), **{t: 100 * float(s[atype == t].mean()) for t in TYPES}} for n, s in scores.items()}
    q1 = pairs["router_64 - A_64"][1] > 0
    q2 = pairs["router_128 - A_128"][1] > 0
    q3 = pairs["A'+B+V_64 - A_64 [yes/no]"][1] > 0 and pairs["A'+B+V_64 - A_64 [other]"][2] < 0
    os.makedirs(out_dir)
    json.dump({"checks": checks, "table": table, "pairs": pairs, "Q1": q1, "Q2": q2, "Q3": q3, "n_questions": len(qids),
               "n_images": len(images), "routed_share": float(routed.mean()),
               "routed_share_by_type": {t: float(routed[atype == t].mean()) for t in TYPES},
               "by_type_n": {t: int((atype == t).sum()) for t in TYPES}}, open(f"{out_dir}/0020.json", "w"), indent=2)
    L = [f"# EXP-0020（计划 v1）：VQAv2 验证集 test2（{len(qids)} 题，{len(images)} 张图）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", f"路由到查询的比例 {routed.mean():.3f}（" + "，".join(f"{t} {routed[atype == t].mean():.3f}" for t in TYPES) + "）"]
    L += ["", "## VQA 准确率", "", "| 配置 | 全部 | yes/no | number | other |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {n} | {r['accuracy']:.2f} | {r['yes/no']:.2f} | {r['number']:.2f} | {r['other']:.2f} |" for n, r in table.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间] |", "| --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] |" for k, v in pairs.items()]
    L += ["", f"**判据**：Q1（64 档路由高于 A）{'成立' if q1 else '不成立'}；Q2（128 档）{'成立' if q2 else '不成立'}；"
              f"Q3（64 档 A′+B+V 在 yes/no 上高于 A、在 other 上低于 A）{'成立' if q3 else '不成立'}。"]
    open(f"{out_dir}/0020.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
