"""Do the pruning methods miss the same POPE objects? (SQZ-EXP-0002 addendum on F10)

  python analyze_missed_objects.py      # writes code/output/EXP-0002/deepdive/{missed_objects.md,missed_objects.json}

POPE's three splits share their positive questions (the same image-object pairs; only the negatives differ), so a
positive is identified by (image, object); this is checked. A positive is missed when the answer is "no" (scored like
llava/eval/eval_pope.py: first sentence, No/not/no); set comparisons use the adversarial split's answer, per-object
rates average the three splits. For the completed POPE runs of EXP-0002 (and EXP-0008-R05 when present): the miss
rate; for pairs of methods, the Jaccard overlap of their missed sets next to the value expected if the misses were
independent draws at the same rates; the share of vanilla's misses that each method also misses; and per-object miss
rates (objects with at least 10 positives) with their Spearman correlation between methods.
"""
import csv
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output")
EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
SPLITS = ("adversarial", "popular", "random")
OBJECT = re.compile(r"Is there an? (.+?) in the (?:image|imange)\?")     # POPE's "an" questions spell it "imange"
EXTRA = {"fastv64_prior": ("EXP-0008", "EXP-0008-R05")}


def pope_no(text):
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "No" in words or "not" in words or "no" in words


def load_questions():
    questions = {}
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            questions[q["question_id"]] = (q["category"], q["image"], OBJECT.match(q["text"]).group(1))
    labels = {}
    for split in SPLITS:
        with open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json") as f:
            for line in f:
                q = json.loads(line)
                labels[(split, q["image"], OBJECT.match(q["text"]).group(1))] = q["label"] == "yes"
    return questions, labels


def runs():
    found = {}
    with open(f"{OUT}/EXP-0002/runs.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["bench"] == "pope" and r["status"] == "completed":
                found.setdefault(r["config"], f"{OUT}/EXP-0002/{r['run_id']}")
    for config, (exp, run_id) in EXTRA.items():
        if os.path.exists(f"{OUT}/{exp}/{run_id}/answers.jsonl"):
            found[config] = f"{OUT}/{exp}/{run_id}"
    return found


def main():
    questions, labels = load_questions()
    positives = {s: {(img, obj) for (sp, img, obj), y in labels.items() if sp == s and y} for s in SPLITS}
    same_positives = positives["adversarial"] == positives["popular"] == positives["random"]
    pos = sorted(positives["adversarial"])
    index = {p: i for i, p in enumerate(pos)}
    missed = {}                                          # config -> (splits x positives) bool array: answered no
    for config, run_dir in runs().items():
        arr = np.zeros((len(SPLITS), len(pos)), dtype=bool)
        with open(f"{run_dir}/answers.jsonl") as f:
            for line in f:
                a = json.loads(line)
                split, img, obj = questions[a["question_id"]]
                if labels[(split, img, obj)]:
                    arr[SPLITS.index(split), index[(img, obj)]] = pope_no(a["text"])
        missed[config] = arr
    n = len(pos)
    rate = {c: float(m[0].mean()) for c, m in missed.items()}
    same_answer_across_splits = {c: float((m == m[0]).all(axis=0).mean()) for c, m in missed.items()}
    order = [c for c in ("vanilla", "fastv64", "fastv64_prior", "random64_s2021", "random64_s2022", "visionzip64",
                         "sparsevlm64", "fastv192", "random192_s2021", "visionzip192", "sparsevlm192") if c in missed]
    jac, jac_expected = {}, {}
    for a in order:
        for b in order:
            if a < b:
                sa, sb = missed[a][0], missed[b][0]
                jac[f"{a}|{b}"] = float((sa & sb).sum() / max((sa | sb).sum(), 1))
                ra, rb = sa.mean(), sb.mean()
                jac_expected[f"{a}|{b}"] = float(ra * rb / (ra + rb - ra * rb)) if ra + rb > 0 else 0.0
    vanilla_kept = {c: float((missed[c][0] & missed["vanilla"][0]).sum() / max(missed["vanilla"][0].sum(), 1))
                    for c in order if c != "vanilla"}
    objects = pd.Series([obj for _, obj in pos])
    counts = objects.value_counts()
    common = sorted(counts[counts >= 10].index)
    per_object = {c: {o: float(missed[c][:, (objects == o).to_numpy()].mean()) for o in common} for c in order}
    po = pd.DataFrame(per_object)
    corr = po.rank().corr().round(3)
    low_budget = [c for c in ("fastv64", "random64_s2021", "visionzip64", "sparsevlm64") if c in po]
    mean_low = po[low_budget].mean(axis=1).sort_values(ascending=False)
    summary = {"n_unique_positives": n, "positives_identical_across_splits": same_positives, "miss_rate": rate,
               "same_answer_across_splits": same_answer_across_splits, "jaccard": jac, "jaccard_if_independent": jac_expected,
               "share_of_vanilla_misses_also_missed": vanilla_kept, "objects_with_10_or_more_positives": len(common),
               "per_object_miss_rate": per_object, "per_object_spearman": corr.to_dict(),
               "hardest_objects_at_64": mean_low.head(10).round(3).to_dict(), "easiest_objects_at_64": mean_low.tail(10).round(3).to_dict()}
    out_dir = f"{OUT}/EXP-0002/deepdive"
    os.makedirs(out_dir, exist_ok=True)   # 2026-10-30 note: no overwrite check; do not rerun (README)
    with open(f"{out_dir}/missed_objects.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = [f"POPE 的 {n} 个正样本（三个 split 的正样本{'相同' if same_positives else '不同'}）；集合比较用 adversarial split 的回答。", "",
             "| 方法 | 漏答率（答 no 的正样本） | vanilla 漏掉的正样本中也漏掉的比例 | 三个 split 回答一致的比例 |", "| --- | --- | --- | --- |"]
    for c in order:
        lines.append(f"| {c} | {rate[c]:.1%} | {'—' if c == 'vanilla' else f'{vanilla_kept[c]:.1%}'} | {same_answer_across_splits[c]:.1%} |")
    lines += ["", "| 方法对 | 漏答集合的 Jaccard | 若独立漏答的期望 |", "| --- | --- | --- |"]
    for k in jac:
        a, b = k.split("|")
        if a in low_budget + ["vanilla", "fastv64_prior"] and b in low_budget + ["vanilla", "fastv64_prior"]:
            lines.append(f"| {a} vs {b} | {jac[k]:.3f} | {jac_expected[k]:.3f} |")
    lines += ["", f"按物体（正样本 ≥ 10 个的 {len(common)} 类）的漏答率，64 档四种方法平均最高与最低的物体：", "",
              "| 物体 | " + " | ".join(order) + " |", "| --- |" + " --- |" * len(order)]
    for o in list(mean_low.head(8).index) + list(mean_low.tail(5).index):
        lines.append(f"| {o} | " + " | ".join(f"{po.loc[o, c]:.2f}" for c in order) + " |")
    lines += ["", "按物体漏答率的 Spearman 相关（64 档与 vanilla）：", "", "| | " + " | ".join(low_budget + ["vanilla"]) + " |",
              "| --- |" + " --- |" * (len(low_budget) + 1)]
    for a in low_budget + ["vanilla"]:
        lines.append(f"| {a} | " + " | ".join(f"{corr.loc[a, b]:.2f}" for b in low_budget + ["vanilla"]) + " |")
    with open(f"{out_dir}/missed_objects.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
