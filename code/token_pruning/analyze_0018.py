"""EXP-0018 (plan v1): VQA accuracy on the VQAv2 val test subset; A'+B+V vs A at 64 (Q1) and 128 (Q2), with references.

  python analyze_0018.py      # writes code/output/EXP-0018/analysis/{0018.md,0018.json}

Per-question scores come from each run's vqa_scores.jsonl (written by score_pope_mme.py) and are recomputed here from the
answers with the same evaluator as a check. Paired differences: images resampled with replacement 1000 times (seed 0).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_0016 import counts_check   # noqa: E402
from analyze_query import row   # noqa: E402
from analyze_repope import EVAL_DATA, OUT   # noqa: E402

RUNS = {"vanilla": "R09", "A_64": "R02", "A'+B+V_64": "R03", "A'+B_64": "R04", "visionzip64": "R05", "sparsevlm64": "R06",
        "A_128": "R07", "A'+B+V_128": "R08"}
PAIRS = [("A'+B+V_64", "A_64"), ("A'+B+V_128", "A_128"), ("A'+B_64", "A_64"), ("A'+B+V_64", "A'+B_64"),
         ("A'+B+V_64", "visionzip64"), ("A'+B+V_64", "sparsevlm64"), ("A'+B+V_64", "vanilla"), ("A_64", "vanilla")]
N_BOOT = 1000


def main():
    out_dir = f"{OUT}/EXP-0018/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sys.path.insert(0, "D:/6CJNM007/idea_rearch/source/LLaVA")
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    ev = TextVQAAccuracyEvaluator()
    qfile = f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test.jsonl"
    gts = {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test_annotations.jsonl")}
    qids = sorted(gts)
    images = sorted({g["image"] for g in gts.values()})
    img_of = np.array([images.index(gts[q]["image"]) for q in qids])
    atype = np.array([gts[q]["answer_type"] for q in qids])
    checks = {"R03 counts = round(30 S)": counts_check("EXP-0018/EXP-0018-R03", qfile, 30, f"{OUT}/EXP-0015/visibility/visibility.npy"),
              "R08 counts = round(98 S)": counts_check("EXP-0018/EXP-0018-R08", qfile, 98, f"{OUT}/EXP-0015/visibility/visibility.npy")}
    scores = {}
    for name, rid in RUNS.items():
        run = f"EXP-0018/EXP-0018-{rid}"
        ans = {json.loads(l)["question_id"]: json.loads(l)["text"] for l in open(f"{OUT}/{run}/answers.jsonl")}
        s = np.array([ev._compute_answer_scores(gts[q]["answers"]).get(ev.answer_processor(ans[q]), 0.0) for q in qids])
        logged = {json.loads(l)["question_id"]: json.loads(l)["score"] for l in open(f"{OUT}/{run}/vqa_scores.jsonl")}
        checks[f"{name}: recomputed scores equal vqa_scores.jsonl"] = float(np.allclose(s, [logged[q] for q in qids]))
        checks[f"{name}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(row(run)["headline"])) < 1e-6)
        scores[name] = s
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]
    pairs = {}
    for a, b in PAIRS:
        d = scores[a] - scores[b]
        boot = [100 * d[idx].mean() for idx in draws]
        pairs[f"{a} - {b}"] = [100 * float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        for t in ("yes/no", "number", "other"):
            m = atype == t
            bt = [100 * d[idx][m[idx]].mean() for idx in draws]
            pairs[f"{a} - {b} [{t}]"] = [100 * float(d[m].mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    table = {n: {"accuracy": 100 * float(s.mean()), **{t: 100 * float(s[atype == t].mean()) for t in ("yes/no", "number", "other")}}
             for n, s in scores.items()}
    q1 = pairs["A'+B+V_64 - A_64"][1] > 0
    q2 = pairs["A'+B+V_128 - A_128"][1] > 0
    os.makedirs(out_dir)
    json.dump({"checks": checks, "table": table, "pairs": pairs, "Q1": q1, "Q2": q2,
               "n_questions": len(qids), "n_images": len(images), "by_type_n": {t: int((atype == t).sum()) for t in ("yes/no", "number", "other")}},
              open(f"{out_dir}/0018.json", "w"), indent=2)
    L = ["# EXP-0018（计划 v1）：VQAv2 验证集子集（{} 题，{} 张图）".format(len(qids), len(images)), "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", "## VQA 准确率", "", "| 配置 | 全部 | yes/no | number | other |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {n} | {r['accuracy']:.2f} | {r['yes/no']:.2f} | {r['number']:.2f} | {r['other']:.2f} |" for n, r in table.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间] |", "| --- | --- |"]
    L += [f"| {k} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] |" for k, v in pairs.items()]
    L += ["", f"**判据**：Q1（64 档 A′+B+V 高于 A）{'成立' if q1 else '不成立'}；Q2（128 档）{'成立' if q2 else '不成立'}。"]
    open(f"{out_dir}/0018.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
