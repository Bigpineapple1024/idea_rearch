"""Score LLaVA answers on POPE, MME, the VQAv2 val subsets or GQA testdev-balanced and write metrics.json.

  python score_pope_mme.py pope <answers.jsonl> <out_dir>
  python score_pope_mme.py mme  <answers.jsonl> <out_dir> [--tag NAME]
  python score_pope_mme.py vqav2_test <answers.jsonl> <out_dir>    # or vqav2_dev (EXP-0018), vqav2_test2 (EXP-0020), vqav2_dev2 / vqav2_test3 (IDEA-0004, make_dev_sets.py; 2026-10-29 note: added to this list)
  python score_pope_mme.py gqa_testdev <answers.jsonl> <out_dir>   # EXP-0022 (gqa_val400 uses the same GQA scoring, IDEA-0004; 2026-10-29 note: added to this list)
  python score_pope_mme.py textvqa_noocr <answers.jsonl> <out_dir> # or textvqa_ocr, vstar (IDEA-0004, DEC-0036)
  python score_pope_mme.py gqa_val400b <answers.jsonl> <out_dir>   # or vqav2_test4: IDEA-0007's fresh sets (EXP-0031)
  python score_pope_mme.py gqa_val400c <answers.jsonl> <out_dir>   # or vqav2_test5: IDEA-0008's fresh sets (EXP-0035)
  python score_pope_mme.py gqa_val400d <answers.jsonl> <out_dir>   # or vqav2_test6: IDEA-0011's fresh sets (EXP-0042)

GQA follows LLaVA's scripts/convert_gqa_for_eval.py (answer.rstrip('.').lower()) and the official eval.py accuracy for
--tier testdev_balanced (exact match with the single gold answer); per-question scores go to <out_dir>/gqa_scores.jsonl.

VQAv2 uses LLaVA's m4c TextVQAAccuracyEvaluator (EvalAI normalisation, which also drops commas, unlike the official
vqaEval.py; leave-one-out min(1, matches / 3)); per-question scores go to <out_dir>/vqa_scores.jsonl.

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
    # 2026-10-30 note: the split list is checked, not that every question has an answer (n_scored is recorded)
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


def score_gqa(answers, out_dir, log, split="testdev_balanced", bench="gqa_testdev"):
    """GQA accuracy (testdev-balanced, or the val400 development subset of IDEA-0004): LLaVA's conversion (strip a final
    '.', lower-case) and exact match with the gold answer, as the official eval.py does; per-question scores go to
    <out_dir>/gqa_scores.jsonl."""
    gts = {}
    for line in open(f"{EVAL_DATA}/gqa/llava_gqa_{split}_annotations.jsonl", encoding="utf-8"):
        a = json.loads(line)
        gts[a["question_id"]] = a
    preds = [json.loads(line) for line in open(answers)]
    if sorted(str(p["question_id"]) for p in preds) != sorted(gts):
        raise SystemExit("answers do not cover the question set exactly")
    groups, rows = {}, []
    with open(f"{out_dir}/gqa_scores.jsonl", "w") as f:
        for p in preds:
            g = gts[str(p["question_id"])]
            s = float(p["text"].rstrip(".").lower() == g["answer"])
            kind = "yes/no" if g["answer"] in ("yes", "no") else "other"
            for key in (f"answer:{kind}", f"structural:{g['structural']}"):
                groups.setdefault(key, []).append(s)
            rows.append(s)
            f.write(json.dumps({"question_id": str(p["question_id"]), "image": g["image"], "answer_kind": kind,
                                "structural": g["structural"], "score": s}) + "\n")
    log.write(f"GQA {split}: {len(rows)} questions scored by exact match after rstrip('.').lower()\n")
    return {"benchmark": bench, "headline": "accuracy", "accuracy": 100 * sum(rows) / len(rows), "n_scored": len(rows),
            "by_group": {k: {"n": len(v), "accuracy": 100 * sum(v) / len(v)} for k, v in sorted(groups.items())}}


def score_textvqa(answers, out_dir, prompt, log):
    """TextVQA v0.5.1 val (IDEA-0004, DEC-0036), with or without LLaVA's OCR-token line: LLaVA's m4c evaluator, as its
    eval_textvqa.py, on the ten human answers; question ids are TextVQA's own (make_textvqa_vstar.py); per-question
    scores go to <out_dir>/textvqa_scores.jsonl."""
    sys.path.insert(0, LLAVA_ROOT)
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    ev = TextVQAAccuracyEvaluator()
    gts = {str(a["question_id"]): a for a in map(json.loads, open(f"{EVAL_DATA}/textvqa/llava_textvqa_val_annotations.jsonl", encoding="utf-8"))}
    preds = [json.loads(line) for line in open(answers)]
    if sorted(str(p["question_id"]) for p in preds) != sorted(gts):
        raise SystemExit("answers do not cover the question set exactly")
    rows = []
    with open(f"{out_dir}/textvqa_scores.jsonl", "w") as f:
        for p in preds:
            g = gts[str(p["question_id"])]
            s = ev._compute_answer_scores(g["answers"]).get(ev.answer_processor(p["text"]), 0.0)
            rows.append(s)
            f.write(json.dumps({"question_id": str(p["question_id"]), "image": g["image"], "score": s}) + "\n")
    log.write(f"TextVQA val ({prompt}): {len(rows)} questions scored with llava.eval.m4c_evaluator.TextVQAAccuracyEvaluator\n")
    return {"benchmark": f"textvqa_{prompt}", "headline": "accuracy", "accuracy": 100 * sum(rows) / len(rows), "n_scored": len(rows)}


def vstar_letter(text, n_options):
    """The option letter of an answer to "Answer with the option's letter from the given choices directly.": a leading
    letter (optionally in brackets or followed by '.', ':' or ')'), else the first standalone letter among the options."""
    letters = "ABCD"[:n_options]
    t = text.strip()
    m = re.match(r"^\(?([A-Za-z])\)?(?:[.:)\s]|$)", t)   # 2026-10-30 note: "A man ..." also reads as option A
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    m = re.search(rf"\b([{letters}])\b", t)
    return m.group(1) if m else None


def score_vstar(answers, out_dir, log):
    """V* Bench (191 multiple-choice questions, IDEA-0004, DEC-0036): the answer's option letter against the label;
    accuracy over all 191 and per category; unparsed answers count as wrong. Per-question scores go to
    <out_dir>/vstar_scores.jsonl."""
    gts = {str(a["question_id"]): a for a in map(json.loads, open(f"{EVAL_DATA}/vstar/llava_vstar_test_annotations.jsonl", encoding="utf-8"))}
    preds = [json.loads(line) for line in open(answers)]
    if sorted(str(p["question_id"]) for p in preds) != sorted(gts):
        raise SystemExit("answers do not cover the question set exactly")
    by_cat, rows, unparsed = {}, [], 0
    with open(f"{out_dir}/vstar_scores.jsonl", "w") as f:
        for p in preds:
            g = gts[str(p["question_id"])]
            letter = vstar_letter(p["text"], g["n_options"])
            unparsed += letter is None
            s = float(letter == g["label"])
            by_cat.setdefault(g["category"], []).append(s)
            rows.append(s)
            f.write(json.dumps({"question_id": str(p["question_id"]), "category": g["category"], "letter": letter, "score": s}) + "\n")
    log.write(f"V* Bench: {len(rows)} questions, option letter vs label; {unparsed} answers without a letter\n")
    return {"benchmark": "vstar", "headline": "accuracy", "accuracy": 100 * sum(rows) / len(rows), "n_scored": len(rows),
            "unparsed": unparsed, "by_category": {k: {"n": len(v), "accuracy": 100 * sum(v) / len(v)} for k, v in sorted(by_cat.items())}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=["pope", "mme", "vqav2_test", "vqav2_dev", "vqav2_test2", "vqav2_dev2", "vqav2_test3",
                                     "vqav2_test4", "vqav2_test5", "vqav2_test6", "gqa_testdev", "gqa_val400", "gqa_val400b",
                                     "gqa_val400c", "gqa_val400d",
                                     "textvqa_noocr", "textvqa_ocr", "vstar"])
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
        elif a.bench == "gqa_testdev":
            metrics = score_gqa(answers, a.out_dir, log)
        elif a.bench in ("gqa_val400", "gqa_val400b", "gqa_val400c", "gqa_val400d"):
            metrics = score_gqa(answers, a.out_dir, log, split=a.bench.split("_", 1)[1], bench=a.bench)
        elif a.bench.startswith("textvqa_"):
            metrics = score_textvqa(answers, a.out_dir, a.bench.split("_", 1)[1], log)
        elif a.bench == "vstar":
            metrics = score_vstar(answers, a.out_dir, log)
        else:
            metrics = score_mme(answers, a.out_dir, a.tag or os.path.basename(os.path.abspath(a.out_dir)), log)
    metrics["n_answers"] = n_answers
    with open(f"{a.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
