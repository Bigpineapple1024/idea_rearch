"""Score LLaVA answers on POPE or MME with the official tools and write metrics.json.

  python score_pope_mme.py pope <answers.jsonl> <out_dir>
  python score_pope_mme.py mme  <answers.jsonl> <out_dir> [--tag NAME]

POPE runs LLaVA's llava/eval/eval_pope.py. The headline number is the F1 averaged over the
adversarial / popular / random splits, in percent (LLaVA MODEL_ZOO: 85.9 for LLaVA-1.5-7B).
MME runs eval.zip's convert_answer_to_mme.py and the official eval_tool/calculation.py inside the
data folder (both use relative paths). The headline number is the Perception total (1510.7 for 7B).
The tools' raw output is kept in <out_dir>/score.log; MME's per-task files in <out_dir>/mme_eval_tool/.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

LLAVA_ROOT = os.environ.get("LLAVA_ROOT", "D:/6CJNM007/idea_rearch/source/LLaVA")
EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")


def run_tool(cmd, cwd, log):
    proc = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True)
    log.write(f"$ (cd {cwd} && python {' '.join(cmd)})\n{proc.stdout}{proc.stderr}\n")
    if proc.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed with exit code {proc.returncode}; see score.log")
    return proc.stdout


def score_pope(answers, log):
    out = run_tool([f"{LLAVA_ROOT}/llava/eval/eval_pope.py", "--annotation-dir", f"{EVAL_DATA}/pope/coco",
                    "--question-file", f"{EVAL_DATA}/pope/llava_pope_test.jsonl", "--result-file", answers],
                   cwd=EVAL_DATA, log=log)
    splits = {}
    for block in out.split("Category: ")[1:]:
        name, n = re.match(r"(\w+), # samples: (\d+)", block).groups()
        tp, fp, tn, fn = map(int, re.search(r"TP\tFP\tTN\tFN\t\n(\d+)\t(\d+)\t(\d+)\t(\d+)", block).groups())
        value = {k: float(re.search(rf"{label}: ([\d.]+)", block).group(1)) for k, label in
                 [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                  ("f1", "F1 score"), ("yes_ratio", "Yes ratio")]}
        splits[name] = {"n": int(n), "TP": tp, "FP": fp, "TN": tn, "FN": fn, **value}
    if sorted(splits) != ["adversarial", "popular", "random"]:
        raise SystemExit(f"unexpected POPE splits {sorted(splits)}")
    mean = lambda key: 100 * sum(s[key] for s in splits.values()) / len(splits)
    return {"benchmark": "pope", "headline": "f1_mean", "f1_mean": mean("f1"), "accuracy_mean": mean("accuracy"),
            "n_scored": sum(s["n"] for s in splits.values()), "splits": splits}


def score_mme(answers, out_dir, tag, log):
    mme = f"{EVAL_DATA}/MME"
    os.makedirs(f"{mme}/answers", exist_ok=True)
    shutil.copyfile(answers, f"{mme}/answers/{tag}.jsonl")
    run_tool(["convert_answer_to_mme.py", "--experiment", tag], cwd=mme, log=log)
    out = run_tool(["calculation.py", "--results_dir", f"answers/{tag}"], cwd=f"{mme}/eval_tool", log=log)
    shutil.copytree(f"{mme}/eval_tool/answers/{tag}", f"{out_dir}/mme_eval_tool", dirs_exist_ok=True)
    result = {"benchmark": "mme", "headline": "perception"}
    for section in out.split("=========== ")[1:]:
        name = section.split(" ")[0].lower()
        result[name] = float(re.search(r"total score: ([\d.]+)", section).group(1))
        result[f"{name}_tasks"] = {t: float(s) for t, s in re.findall(r"\t (\w+)  score: ([\d.]+)", section)}
    if len(result.get("perception_tasks", {})) != 10 or len(result.get("cognition_tasks", {})) != 4:
        raise SystemExit("unexpected MME task list; see score.log")
    return result


def score_vqav2(answers, out_dir, split, log):
    """VQA accuracy with LLaVA's m4c evaluator (EvalAI answer normalisation, min(1, matches / 3) averaged over the ten
    leave-one-out subsets of the human answers); per-question scores go to <out_dir>/vqa_scores.jsonl."""
    sys.path.insert(0, LLAVA_ROOT)
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    ev = TextVQAAccuracyEvaluator()
    gts = {}
    for line in open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_{split}_annotations.jsonl"):
        a = json.loads(line)
        gts[a["question_id"]] = a
    preds = [json.loads(line) for line in open(answers)]
    if sorted(p["question_id"] for p in preds) != sorted(gts):
        raise SystemExit("answers do not cover the question set exactly")
    by_type, rows = {}, []
    with open(f"{out_dir}/vqa_scores.jsonl", "w") as f:
        for p in preds:
            g = gts[p["question_id"]]
            s = ev._compute_answer_scores(g["answers"]).get(ev.answer_processor(p["text"]), 0.0)
            by_type.setdefault(g["answer_type"], []).append(s)
            rows.append(s)
            f.write(json.dumps({"question_id": p["question_id"], "image": g["image"], "answer_type": g["answer_type"],
                                "score": s}) + "\n")
    log.write(f"VQAv2 val {split}: {len(rows)} questions scored with llava.eval.m4c_evaluator.TextVQAAccuracyEvaluator\n")
    return {"benchmark": f"vqav2_{split}", "headline": "accuracy", "accuracy": 100 * sum(rows) / len(rows),
            "n_scored": len(rows), "by_answer_type": {t: {"n": len(v), "accuracy": 100 * sum(v) / len(v)} for t, v in by_type.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=["pope", "mme", "vqav2_test", "vqav2_dev"])
    p.add_argument("answers")
    p.add_argument("out_dir")
    p.add_argument("--tag", help="MME working name (default: the out_dir name)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    answers = os.path.abspath(a.answers)
    with open(answers) as f:
        n_answers = sum(1 for _ in f)
    with open(f"{a.out_dir}/score.log", "w") as log:
        if a.bench == "pope":
            metrics = score_pope(answers, log)
        elif a.bench.startswith("vqav2_"):
            metrics = score_vqav2(answers, a.out_dir, a.bench.split("_", 1)[1], log)
        else:
            metrics = score_mme(answers, a.out_dir, a.tag or os.path.basename(os.path.abspath(a.out_dir)), log)
    metrics["n_answers"] = n_answers
    with open(f"{a.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
