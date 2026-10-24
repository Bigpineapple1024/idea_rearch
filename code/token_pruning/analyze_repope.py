"""Rescore the completed POPE runs with the RePOPE labels (F27 follow-up)

  python analyze_repope.py      # writes code/output/EXP-0002/deepdive4/{repope.md,repope.json}

RePOPE (Neuhaus & Hein, arXiv 2504.15707; YanNeu/RePOPE@bcf22e56, annotations/coco_repope_<split>.json; user-approved
download 2026-10-24) relabels POPE's 500 COCO images and drops ambiguous questions. Each RePOPE question is matched to
the LLaVA POPE question with the same split, image and object (RePOPE starts from the 3000-question splits, LLaVA's
random split has 2910; RePOPE questions without a LLaVA counterpart are counted and left out); answers are scored as in
llava/eval/eval_pope.py (first
sentence, "No"/"not"/"no" means no) and F1 is averaged over the three splits, the project's POPE headline. As a check,
the same code must reproduce every run's registered headline from the original labels. Paired differences use a
bootstrap over questions within each split (1000 resamples, seed 0, the same draws for both runs). The positives
without a COCO instance (deepdive3) are looked up in RePOPE.
"""
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IDEA_REARCH_REPO") or os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "code", "output")
EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
REPOPE = os.environ.get("REPOPE_DIR", "D:/6CJNM007/data/downloads/repope")
SPLITS = ("adversarial", "popular", "random")
OBJECT = re.compile(r"Is there an? (.+?) in the (?:image|imange)\?")     # POPE's "an" questions spell it "imange"
EXPS = ("EXP-0002", "EXP-0007", "EXP-0008", "EXP-0010")
PAIRS = [("fastv64_prior_strat", "random64_s2021"), ("fastv64_strat", "random64_s2021"),
         ("fastv64_prior", "random64_s2021"), ("fastv64", "random64_s2021"), ("sparsevlm64", "random_svp64_s2021"),
         ("visionzip64", "fastv64_prior_strat"), ("sparsevlm64", "fastv64_prior_strat"), ("visionzip64", "sparsevlm64"),
         ("random_svp64_s2021", "random_flat64_s2021"), ("fastv_svp64", "fastv_flat64")]
N_BOOT = 1000


def pope_no(text):
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "No" in words or "not" in words or "no" in words


def read_labels(path_of_split):
    labels = {}
    for split in SPLITS:
        with open(path_of_split(split)) as f:
            for line in f:
                q = json.loads(line)
                labels[(split, q["image"], OBJECT.match(q["text"]).group(1))] = q["label"] == "yes"
    return labels


def runs():
    found = {}
    for exp in EXPS:
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["bench"] == "pope" and r["status"] == "completed":
                    found.setdefault(r["config"], (f"{exp}/{r['run_id']}", float(r["headline"])))
    return found


def counts(said_yes, label):
    tp = int((said_yes & label).sum())
    fp = int((said_yes & ~label).sum())
    fn = int((~said_yes & label).sum())
    tn = int((~said_yes & ~label).sum())
    return tp, fp, fn, tn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def score(said, labels):
    """said: {(split, image, object): bool}; returns per-split arrays and the F1 mean over splits."""
    per = {}
    for split in SPLITS:
        keys = [k for k in labels if k[0] == split]
        y = np.array([said[k] for k in keys])
        lab = np.array([labels[k] for k in keys])
        tp, fp, fn, tn = counts(y, lab)
        per[split] = {"n": len(keys), "f1": f1(tp, fp, fn), "precision": tp / (tp + fp) if tp + fp else 0.0,
                      "recall": tp / (tp + fn) if tp + fn else 0.0, "accuracy": (tp + tn) / len(keys),
                      "yes_ratio": float(y.mean())}
    return per, 100 * float(np.mean([per[s]["f1"] for s in SPLITS]))


def paired_boot(said_a, said_b, labels, seed=0):
    rng = np.random.default_rng(seed)
    arrays = {}
    for split in SPLITS:
        keys = [k for k in labels if k[0] == split]
        arrays[split] = (np.array([said_a[k] for k in keys]), np.array([said_b[k] for k in keys]),
                         np.array([labels[k] for k in keys]))
    diffs = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        fa, fb = [], []
        for split in SPLITS:
            a, b, lab = arrays[split]
            idx = rng.integers(0, len(lab), len(lab))
            fa.append(f1(*counts(a[idx], lab[idx])[:3]))
            fb.append(f1(*counts(b[idx], lab[idx])[:3]))
        diffs[i] = 100 * (np.mean(fa) - np.mean(fb))
    point = 100 * (np.mean([f1(*counts(*arrays[s][::2])[:3]) for s in SPLITS])
                   - np.mean([f1(*counts(arrays[s][1], arrays[s][2])[:3]) for s in SPLITS]))
    return [float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    out_dir = f"{OUT}/EXP-0002/deepdive4"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} already exists; results are never overwritten")
    questions = {}
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            questions[q["question_id"]] = (q["category"], q["image"], OBJECT.match(q["text"]).group(1))
    original = read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json")
    repope = read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json")
    # RePOPE starts from the 3000-question POPE splits; LLaVA's random split has 2910 questions, so a few RePOPE
    # questions have no LLaVA counterpart (and no answers to score). They are counted and left out.
    unknown = {split: {"yes": sum(1 for k in repope if k[0] == split and k not in original and repope[k]),
                       "no": sum(1 for k in repope if k[0] == split and k not in original and not repope[k])}
               for split in SPLITS}
    repope = {k: v for k, v in repope.items() if k in original}
    change = {split: {"kept": sum(1 for k in original if k[0] == split and k in repope),
                      "removed": sum(1 for k in original if k[0] == split and k not in repope),
                      "yes_to_no": sum(1 for k in repope if k[0] == split and original[k] and not repope[k]),
                      "no_to_yes": sum(1 for k in repope if k[0] == split and not original[k] and repope[k])}
              for split in SPLITS}

    said, results = {}, {}
    for config, (run, headline) in runs().items():
        s = {}
        with open(f"{OUT}/{run}/answers.jsonl") as f:
            for line in f:
                a = json.loads(line)
                s[questions[a["question_id"]]] = not pope_no(a["text"])
        if len(s) != len(original):
            sys.exit(f"{run}: {len(s)} answers")
        per_o, f1_o = score(s, original)
        if abs(f1_o - headline) > 1e-6:
            sys.exit(f"{run}: rescored {f1_o} differs from the registered headline {headline}")
        per_r, f1_r = score(s, repope)
        said[config] = s
        results[config] = {"run": run, "f1_original": f1_o, "f1_repope": f1_r, "original": per_o, "repope": per_r}
    pairs = {f"{a} - {b}": {"original": paired_boot(said[a], said[b], original),
                            "repope": paired_boot(said[a], said[b], repope)} for a, b in PAIRS}

    with open(f"{OUT}/EXP-0002/deepdive3/object_boxes.json") as f:
        unmatched = [tuple(u) for u in json.load(f)["unmatched"]]
    status = {split: {"still_yes": 0, "now_no": 0, "removed": 0} for split in SPLITS}
    for img, obj in unmatched:
        for split in SPLITS:
            k = (split, img, obj)
            status[split]["removed" if k not in repope else "still_yes" if repope[k] else "now_no"] += 1

    summary = {"label_changes": change, "repope_questions_not_in_llava_pope": unknown, "runs": results, "pairs": pairs,
               "coco_unmatched_positives_in_repope": status}
    os.makedirs(out_dir)
    with open(f"{out_dir}/repope.json", "w") as f:
        json.dump(summary, f, indent=2)

    def mean_over(per, key):
        return 100 * np.mean([per[s][key] for s in SPLITS])

    lines = ["## RePOPE 与原始标注的差别（LLaVA 的 POPE 题目：adversarial、popular 各 3000 题，random 2910 题）", "",
             "| split | 保留 | 删除 | yes → no | no → yes | RePOPE 中不在 LLaVA 题目里的（yes / no） |",
             "| --- | --- | --- | --- | --- | --- |"]
    for split in SPLITS:
        c = change[split]
        lines.append(f"| {split} | {c['kept']} | {c['removed']} | {c['yes_to_no']} | {c['no_to_yes']} | "
                     f"{unknown[split]['yes']} / {unknown[split]['no']} |")
    lines += ["", "## 各运行的 POPE F1（三个 split 平均）", "",
              "| 配置 | 运行 | 原始标注 F1 | RePOPE F1 | 变化 | 召回 原始 → RePOPE | 精确率 原始 → RePOPE |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for config, r in sorted(results.items(), key=lambda kv: -kv[1]["f1_repope"]):
        lines.append(f"| {config} | {r['run']} | {r['f1_original']:.2f} | {r['f1_repope']:.2f} | "
                     f"{r['f1_repope'] - r['f1_original']:+.2f} | {mean_over(r['original'], 'recall'):.1f}% → "
                     f"{mean_over(r['repope'], 'recall'):.1f}% | {mean_over(r['original'], 'precision'):.1f}% → "
                     f"{mean_over(r['repope'], 'precision'):.1f}% |")
    lines += ["", "## 配对差（F1 百分点，95% 区间）", "", "| 配对 | 原始标注 | RePOPE |", "| --- | --- | --- |"]
    for k, v in pairs.items():
        o, r = v["original"], v["repope"]
        lines.append(f"| {k} | {o[0]:+.2f} [{o[1]:+.2f}, {o[2]:+.2f}] | {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] |")
    lines += ["", f"## COCO 中没有同名实例的 {len(unmatched)} 个正样本在 RePOPE 中", "",
              "| split | 仍为 yes | 改为 no | 删除 |", "| --- | --- | --- | --- |"]
    for split in SPLITS:
        st = status[split]
        lines.append(f"| {split} | {st['still_yes']} | {st['now_no']} | {st['removed']} |")
    with open(f"{out_dir}/repope.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
