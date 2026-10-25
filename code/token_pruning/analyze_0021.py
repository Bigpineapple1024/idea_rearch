"""EXP-0021 (plan v1): completes the VQAv2 test2 table. Router vs A at the 192 budget, and VisionZip-128 / SparseVLM-128
against EXP-0020's 128-budget runs on the same questions. Descriptive, no criteria.

  python analyze_0021.py      # writes code/output/EXP-0021/analysis/{0021.md,0021.json}

Scores and checks as in analyze_0020.py; paired differences resample images (1000 draws, seed 0).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_query import row   # noqa: E402
from analyze_repope import EVAL_DATA, OUT   # noqa: E402

RUNS = {"A_192": "EXP-0021/EXP-0021-R01", "router_192": "EXP-0021/EXP-0021-R02", "visionzip128": "EXP-0021/EXP-0021-R03",
        "sparsevlm128": "EXP-0021/EXP-0021-R04", "A_128": "EXP-0020/EXP-0020-R04", "router_128": "EXP-0020/EXP-0020-R05",
        "vanilla": "EXP-0020/EXP-0020-R07"}
PAIRS = [("router_192", "A_192"), ("router_128", "visionzip128"), ("router_128", "sparsevlm128"), ("A_128", "visionzip128"),
         ("router_192", "vanilla"), ("A_192", "A_128")]
TYPES = ("yes/no", "number", "other")
N_BOOT = 1000


def main():
    out_dir = f"{OUT}/EXP-0021/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sys.path.insert(0, "D:/6CJNM007/idea_rearch/source/LLaVA")
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    import run_eval
    ev = TextVQAAccuracyEvaluator()
    questions = [json.loads(l) for l in open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test2.jsonl")]
    gts = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test2_annotations.jsonl")}
    rule = [run_eval.routes_to_query(q["text"]) for q in questions]
    qids = sorted(gts)
    images = sorted({g["image"] for g in gts.values()})
    img_of = np.array([images.index(gts[q]["image"]) for q in qids])
    atype = np.array([gts[q]["answer_type"] for q in qids])
    checks, scores = {}, {}
    for name, run in RUNS.items():
        ans = {json.loads(l)["question_id"]: json.loads(l)["text"] for l in open(f"{OUT}/{run}/answers.jsonl")}
        s = np.array([ev._compute_answer_scores(gts[q]["answers"]).get(ev.answer_processor(ans[q]), 0.0) for q in qids])
        logged = {json.loads(l)["question_id"]: json.loads(l)["score"] for l in open(f"{OUT}/{run}/vqa_scores.jsonl")}
        checks[f"{name}: recomputed scores equal vqa_scores.jsonl"] = float(np.allclose(s, [logged[q] for q in qids]))
        checks[f"{name}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(row(run)["headline"])) < 1e-6)
        scores[name] = s
    k = json.load(open(f"{OUT}/{RUNS['router_192']}/kept_tokens.json"))
    checks["router_192: logged decisions equal the rule on the question text"] = float(np.mean(np.array(k["routed"]) == np.array(rule)))
    checks["router_192: question part empty exactly when unrouted"] = float(np.mean(
        [(len(x) > 0) == bool(d) for x, d in zip(k["kept_by_object_name"], rule)]))
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    by_img = [np.flatnonzero(img_of == i) for i in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[i] for i in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]
    pairs = {}
    for x, y in PAIRS:
        d = scores[x] - scores[y]
        pairs[f"{x} - {y}"] = [100 * float(d.mean()), *np.percentile([100 * d[idx].mean() for idx in draws], [2.5, 97.5]).tolist()]
        for t in TYPES:
            m = atype == t
            pairs[f"{x} - {y} [{t}]"] = [100 * float(d[m].mean()),
                                         *np.percentile([100 * d[idx][m[idx]].mean() for idx in draws], [2.5, 97.5]).tolist()]
    table = {n: {"accuracy": 100 * float(s.mean()), **{t: 100 * float(s[atype == t].mean()) for t in TYPES}} for n, s in scores.items()}
    os.makedirs(out_dir)
    json.dump({"checks": checks, "table": table, "pairs": pairs}, open(f"{out_dir}/0021.json", "w"), indent=2)
    L = ["# EXP-0021（计划 v1）：VQAv2 test2 的补全（192 档路由；128 档参照）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", "## VQA 准确率", "", "| 配置 | 全部 | yes/no | number | other |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {n} | {r['accuracy']:.2f} | {r['yes/no']:.2f} | {r['number']:.2f} | {r['other']:.2f} |" for n, r in table.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间] |", "| --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] |" for k, v in pairs.items()]
    open(f"{out_dir}/0021.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
