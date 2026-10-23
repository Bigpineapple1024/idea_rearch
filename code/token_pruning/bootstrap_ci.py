"""Bootstrap 95% intervals for EXP-0002 headline metrics, alone or as a paired difference of two runs.

  python bootstrap_ci.py <run_dir>                 # interval of the run's headline metric
  python bootstrap_ci.py <run_dir_a> <run_dir_b>   # interval of a - b on the same resampled items

POPE: questions are resampled within each split; the statistic is the mean of the three split F1s
(scored exactly as llava/eval/eval_pope.py: first sentence, "No"/"not"/"no" means no).
MME: images (question pairs) are resampled within each subtask; the statistic is the Perception total,
sum over the 10 perception subtasks of (acc + acc+) * 100, read from the official tool's per-task files.
"""
import argparse
import json
import os

import numpy as np

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]


def pope_no(text):
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "No" in words or "not" in words or "no" in words


def load_pope(run_dir):
    questions = {}
    with open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl") as f:
        for line in f:
            q = json.loads(line)
            questions[q["question_id"]] = q["category"]
    with open(f"{run_dir}/answers.jsonl") as f:
        answers = [json.loads(line) for line in f]
    items = {}
    for split in ("adversarial", "popular", "random"):
        with open(f"{EVAL_DATA}/pope/coco/coco_pope_{split}.json") as f:
            labels = [json.loads(line)["label"] == "yes" for line in f]
        preds = [not pope_no(a["text"]) for a in answers if questions[a["question_id"]] == split]
        items[split] = np.array(list(zip(preds, labels[:len(preds)])), dtype=bool)  # eval_pope pairs in order
    return items


def pope_stat(items, idx):
    f1s = []
    for split, arr in items.items():
        pred, label = arr[idx[split], 0], arr[idx[split], 1]
        tp, fp, fn = np.sum(pred & label), np.sum(pred & ~label), np.sum(~pred & label)
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return 100 * float(np.mean(f1s))


def load_mme(run_dir):
    items = {}
    for task in PERCEPTION:
        with open(f"{run_dir}/mme_eval_tool/{task}.txt") as f:
            lines = [line.rstrip("\n").split("\t") for line in f]
        correct = []
        for _, _, gt, pred in lines:
            pred = pred.lower()
            label = pred if pred in ("yes", "no") else ("yes" if "yes" in pred[:4] else "no" if "no" in pred[:4] else "other")
            correct.append(label == gt.lower())
        items[task] = np.array(correct, dtype=bool).reshape(-1, 2)          # one image = two questions
    return items


def mme_stat(items, idx):
    total = 0.0
    for task, pairs in items.items():
        sample = pairs[idx[task]]
        total += 100 * (sample.mean() + sample.all(axis=1).mean())
    return float(total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    benches = {json.load(open(f"{r}/config.json"))["bench"] for r in a.runs}
    if len(benches) != 1 or len(a.runs) > 2:
        raise SystemExit("give one run, or two runs on the same benchmark")
    bench = benches.pop()
    load, stat = (load_pope, pope_stat) if bench == "pope" else (load_mme, mme_stat)
    data = [load(r) for r in a.runs]
    sizes = {k: len(v) for k, v in data[0].items()}
    full = {k: np.arange(n) for k, n in sizes.items()}
    point = [stat(d, full) for d in data]
    rng = np.random.default_rng(a.seed)
    draws = []
    for _ in range(a.n):
        idx = {k: rng.integers(0, n, n) for k, n in sizes.items()}
        vals = [stat(d, idx) for d in data]
        draws.append(vals[0] - vals[1] if len(vals) == 2 else vals[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    est = point[0] - point[1] if len(point) == 2 else point[0]
    print(json.dumps({"bench": bench, "runs": [os.path.basename(r.rstrip("/\\")) for r in a.runs],
                      "statistic": "difference" if len(point) == 2 else "value", "estimate": est,
                      "ci95": [float(lo), float(hi)], "n_boot": a.n, "seed": a.seed}))


if __name__ == "__main__":
    main()
