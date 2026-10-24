"""EXP-0014 (plan v1, step 12): A / A' / A'+B / A+B on POPE and MME.

  python analyze_query.py      # writes code/output/EXP-0014/analysis/{query.md,query.json}

A = strong baseline with padding excluded (POPE: EXP-0011-R07; MME: EXP-0014-R04); A' = uniform question query (R01,
R05); A'+B = IDF-weighted question query (R02, R06); A+B = IDF-weighted layer-1 attention rows (R03, R07). References:
the object-name query EXP-0011-R06 (POPE only), vanilla, FastV-64, VisionZip-64, SparseVLM-64 (EXP-0002) and the strong
baseline without padding exclusion (EXP-0010). Checks (stop if violated): every run's rescored headline equals its
registered one; the df table each IDF run wrote equals the offline recount. POPE: F1 over the three splits with the
original labels and RePOPE, paired bootstrap over questions within splits (analyze_repope.paired_boot). MME: Perception
total, paired bootstrap over images within subtasks (bootstrap_ci, 1000 resamples, seed 0, the same draws for both runs).
Criteria: S1 A'+B - A' and A'+B - A above 0 on POPE (both labels); S2 |A+B - A| <= 1 or its interval contains 0 (POPE);
S3 the MME-P interval of A'+B - A not below 0.
"""
import csv
import json
import os
import sys

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_repope import EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, paired_boot, pope_no, read_labels, score   # noqa: E402
from bootstrap_ci import PERCEPTION, load_mme, mme_stat   # noqa: E402

POPE = {"A": "EXP-0011/EXP-0011-R07", "A'": "EXP-0014/EXP-0014-R01", "A'+B": "EXP-0014/EXP-0014-R02",
        "A+B": "EXP-0014/EXP-0014-R03", "object_name": "EXP-0011/EXP-0011-R06", "vanilla": "EXP-0002/EXP-0002-R04",
        "fastv64": "EXP-0002/EXP-0002-R11", "visionzip64": "EXP-0002/EXP-0002-R17", "sparsevlm64": "EXP-0002/EXP-0002-R41"}
MME = {"A": "EXP-0014/EXP-0014-R04", "A'": "EXP-0014/EXP-0014-R05", "A'+B": "EXP-0014/EXP-0014-R06",
       "A+B": "EXP-0014/EXP-0014-R07", "vanilla": "EXP-0002/EXP-0002-R02", "strong_no_padding_excl": "EXP-0010/EXP-0010-R06",
       "visionzip64": "EXP-0002/EXP-0002-R35", "sparsevlm64": "EXP-0002/EXP-0002-R40"}
POPE_PAIRS = [("A'+B", "A'"), ("A'+B", "A"), ("A+B", "A"), ("A'", "A"), ("A'+B", "object_name"), ("A'+B", "visionzip64"),
              ("A'+B", "sparsevlm64"), ("A'+B", "vanilla")]
MME_PAIRS = [("A'+B", "A"), ("A'+B", "A'"), ("A+B", "A"), ("A'", "A"), ("A'+B", "visionzip64"), ("A'+B", "sparsevlm64"),
             ("A'+B", "vanilla")]
N_BOOT = 1000


def row(run):
    exp, run_id = run.split("/")
    with open(f"{OUT}/{exp}/runs.tsv") as f:
        r = {x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")}[run_id]
    if r["status"] != "completed":
        sys.exit(f"{run} is not completed")
    return r


def mme_boot(a, b, seed=0):
    sizes = {k: len(v) for k, v in a.items()}
    rng = np.random.default_rng(seed)
    full = {k: np.arange(n) for k, n in sizes.items()}
    d = []
    for _ in range(N_BOOT):
        idx = {k: rng.integers(0, n, n) for k, n in sizes.items()}
        d.append(mme_stat(a, idx) - mme_stat(b, idx))
    return [mme_stat(a, full) - mme_stat(b, full), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    out_dir = f"{OUT}/EXP-0014/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    checks = {}
    # IDF tables
    from transformers import AutoTokenizer
    import run_eval
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    for run in ("EXP-0014/EXP-0014-R02", "EXP-0014/EXP-0014-R03", "EXP-0014/EXP-0014-R06", "EXP-0014/EXP-0014-R07"):
        logged = json.load(open(f"{OUT}/{run}/answers.jsonl.idf.json"))
        ref = run_eval.question_idf(tok, logged["question_file"], "vicuna_v1")
        checks[f"{run}: df table equals the offline recount"] = (
            logged["n_questions"] == ref["n_questions"] and {int(k): v for k, v in logged["df"].items()} == ref["df"])

    # POPE
    order = []
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            order.append((q["question_id"], (q["category"], q["image"], OBJECT.match(q["text"]).group(1))))
    key_of = dict(order)
    original = read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json")
    repope = {k: v for k, v in read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json").items() if k in original}
    said, pope = {}, {}
    for name, run in POPE.items():
        ans = [json.loads(line) for line in open(f"{OUT}/{run}/answers.jsonl")]
        s = {key_of[a["question_id"]]: not pope_no(a["text"]) for a in ans}
        per_o, f1_o = score(s, original)
        per_r, f1_r = score(s, repope)
        checks[f"POPE {name} rescored F1 equals the registered headline"] = abs(f1_o - float(row(run)["headline"])) < 1e-6
        said[name] = s
        pope[name] = {"run": run, "f1": f1_o, "f1_repope": f1_r,
                      "accuracy": 100 * float(np.mean([per_o[x]["accuracy"] for x in SPLITS])),
                      "recall": 100 * float(np.mean([per_o[x]["recall"] for x in SPLITS])),
                      "precision": 100 * float(np.mean([per_o[x]["precision"] for x in SPLITS])), "fp": {}}
        for x in SPLITS:
            keys = [k for k in original if k[0] == x]
            pope[name]["fp"][x] = int(sum(s[k] and not original[k] for k in keys))
    pope_pairs = {f"{a} - {b}": {"original": paired_boot(said[a], said[b], original),
                                 "repope": paired_boot(said[a], said[b], repope)} for a, b in POPE_PAIRS}
    k02 = json.load(open(f"{OUT}/EXP-0014/EXP-0014-R02/kept_tokens.json"))["kept"]
    k06 = json.load(open(f"{OUT}/EXP-0011/EXP-0011-R06/kept_tokens.json"))["kept"]
    overlap = float(np.mean([len(set(a) & set(b)) / 30 for a, b in zip(k02, k06)]))

    # MME
    mme, items = {}, {}
    for name, run in MME.items():
        it = load_mme(f"{OUT}/{run}")
        items[name] = it
        full = {k: np.arange(len(v)) for k, v in it.items()}
        total = mme_stat(it, full)
        checks[f"MME {name} recomputed Perception equals the registered headline"] = abs(total - float(row(run)["headline"])) < 1e-6
        mme[name] = {"run": run, "perception": total,
                     "per_task": {t: 100 * float(it[t].mean() + it[t].all(axis=1).mean()) for t in PERCEPTION}}
    mme_pairs = {f"{a} - {b}": mme_boot(items[a], items[b]) for a, b in MME_PAIRS}
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print("STOP: checks failed:", failed)
        sys.exit(1)

    s1 = all(pope_pairs[p][lab][1] > 0 for p in ("A'+B - A'", "A'+B - A") for lab in ("original", "repope"))
    aa = pope_pairs["A+B - A"]["original"]
    s2 = abs(aa[0]) <= 1 or aa[1] <= 0 <= aa[2]
    s3 = mme_pairs["A'+B - A"][2] >= 0                     # the interval is not wholly below 0
    s3_up = mme_pairs["A'+B - A"][1] > 0
    summary = {"checks": checks, "pope": pope, "pope_pairs": pope_pairs, "kept_overlap_R02_vs_object_name": overlap,
               "mme": mme, "mme_pairs": mme_pairs, "S1": bool(s1), "S2": bool(s2), "S3_not_worse": bool(s3),
               "S3_better": bool(s3_up)}
    os.makedirs(out_dir)
    json.dump(summary, open(f"{out_dir}/query.json", "w"), indent=2)
    L = ["# EXP-0014（计划 v1，第 ⑫ 步）：A / A′ / A′+B / A+B", "", "## 核对", ""]
    L += [f"- {k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()]
    L += ["", "## POPE（三个子集均值）", "", "| 配置 | F1 原始 | F1 RePOPE | 准确率 | 召回 | 精确率 | 误报 adversarial / popular / random |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, r in pope.items():
        L.append(f"| {n}（{r['run']}） | {r['f1']:.2f} | {r['f1_repope']:.2f} | {r['accuracy']:.2f} | {r['recall']:.1f}% | "
                 f"{r['precision']:.1f}% | {r['fp']['adversarial']} / {r['fp']['popular']} / {r['fp']['random']} |")
    L += ["", "| 配对差（F1，题目在各子集内重抽 1000 次） | 原始标注 | RePOPE |", "| --- | --- | --- |"]
    for k, v in pope_pairs.items():
        o, r = v["original"], v["repope"]
        L.append(f"| {k} | {o[0]:+.2f} [{o[1]:+.2f}, {o[2]:+.2f}] | {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] |")
    L += ["", f"A′+B 与物体名查询（EXP-0011-R06）的保留集合平均重合 {100 * overlap:.1f}%。", "",
          "## MME（Perception 总分与各子任务）", "", "| 配置 | Perception | " + " | ".join(PERCEPTION) + " |",
          "| --- | --- | " + " | ".join("---" for _ in PERCEPTION) + " |"]
    for n, r in mme.items():
        L.append(f"| {n} | {r['perception']:.1f} | " + " | ".join(f"{r['per_task'][t]:.1f}" for t in PERCEPTION) + " |")
    L += ["", "| 配对差（MME-P，按子任务内的图像重抽 1000 次） | 差 [95% 区间] |", "| --- | --- |"]
    for k, v in mme_pairs.items():
        L.append(f"| {k} | {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] |")
    L += ["", f"**判据**：S1（POPE 上 A′+B 高于 A′ 与 A，两种标注）{'成立' if s1 else '不成立'}；S2（A+B 与 A 相差不超过 1 或区间含 0）"
              f"{'成立' if s2 else '不成立'}；S3（MME-P 上 A′+B − A 的区间不在 0 以下）{'成立' if s3 else '不成立'}"
              f"{'，并且区间在 0 以上' if s3_up else ''}。"]
    open(f"{out_dir}/query.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
