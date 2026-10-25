"""Does the "no" bias of F10 (POPE) also appear on MME? Yes ratio, recall on the yes questions and specificity on the no
questions of every completed MME run of EXP-0002, EXP-0007, EXP-0008 and EXP-0010 (a generality check for step 7).

  python analyze_mme_yes_ratio.py      # writes code/output/EXP-0002/deepdive2/{mme_yes_ratio.md,mme_yes_ratio.json}

Ground truth comes from MME_Benchmark_release_version the way llava's convert_answer_to_mme.py matches it; an answer
counts as yes or no by its first word (lower case), anything else as other. MME asks one yes and one no question per
image, so a model without bias answers yes to half of them.
"""
import csv
import json
import os

OUT = "D:/6CJNM007/idea_rearch/code/output"
MME = "D:/6CJNM007/data/llava_eval/MME/MME_Benchmark_release_version"
EXPS = ("EXP-0002", "EXP-0007", "EXP-0008", "EXP-0010")


def get_gt():
    gt = {}
    for category in os.listdir(MME):
        d = os.path.join(MME, category)
        if not os.path.isdir(d):
            continue
        qa = os.path.join(d, "questions_answers_YN") if os.path.exists(os.path.join(d, "images")) else d
        for file in os.listdir(qa):
            if file.endswith(".txt"):
                for line in open(os.path.join(qa, file), encoding="utf-8"):
                    question, answer = line.strip().split("\t")
                    gt[(category, file, question)] = answer
    return gt


def lookup(gt, a):
    category = a["question_id"].split("/")[0]
    file = a["question_id"].split("/")[-1].split(".")[0] + ".txt"
    prompt = a["prompt"].replace("Answer the question using a single word or phrase.", "").strip()
    if "Please answer yes or no." not in prompt:
        prompt += " Please answer yes or no."
        if (category, file, prompt) not in gt:
            prompt = prompt.replace(" Please answer yes or no.", "  Please answer yes or no.")
    return gt[(category, file, prompt)]


def main():
    gt = get_gt()
    rows = []
    for exp in EXPS:
        if not os.path.exists(f"{OUT}/{exp}/runs.tsv"):
            continue
        seen = set()
        with open(f"{OUT}/{exp}/runs.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["bench"] != "mme" or r["status"] != "completed" or r["config"] in seen:
                    continue
                seen.add(r["config"])
                n = yes = other = gt_yes = yes_on_yes = gt_no = no_on_no = 0
                by_cat = {}                                  # category -> [n, yes, gt_yes, yes_on_yes, gt_no, no_on_no]
                with open(f"{OUT}/{exp}/{r['run_id']}/answers.jsonl") as fa:
                    for line in fa:
                        a = json.loads(line)
                        truth = lookup(gt, a).strip().lower()
                        words = a["text"].strip().lower().replace(",", " ").replace(".", " ").split()
                        pred = words[0] if words and words[0] in ("yes", "no") else "other"
                        n += 1
                        yes += pred == "yes"
                        other += pred == "other"
                        c = by_cat.setdefault(a["question_id"].split("/")[0], [0, 0, 0, 0, 0, 0])
                        c[0] += 1
                        c[1] += pred == "yes"
                        if truth == "yes":
                            gt_yes += 1
                            yes_on_yes += pred == "yes"
                            c[2] += 1
                            c[3] += pred == "yes"
                        else:
                            gt_no += 1
                            no_on_no += pred == "no"
                            c[4] += 1
                            c[5] += pred == "no"
                rows.append({"run_id": r["run_id"], "config": r["config"], "n": n, "yes_ratio": yes / n,
                             "other_ratio": other / n, "recall_yes": yes_on_yes / gt_yes, "specificity_no": no_on_no / gt_no,
                             "n_gt_yes": gt_yes, "n_gt_no": gt_no, "perception": float(r["headline"]),
                             "by_category": {k: {"n": v[0], "yes_ratio": v[1] / v[0], "recall_yes": v[3] / v[2],
                                                 "specificity_no": v[5] / v[4]} for k, v in sorted(by_cat.items())}})
    os.makedirs(f"{OUT}/EXP-0002/deepdive2", exist_ok=True)
    with open(f"{OUT}/EXP-0002/deepdive2/mme_yes_ratio.json", "w") as f:
        json.dump(rows, f, indent=2)
    lines = ["MME 全部 2374 题（含 Cognition；每张图一道 yes、一道 no 题）：", "",
             "| 运行 | 配置 | yes 比例 % | yes 题的召回 % | no 题的特异度 % | 其他回答 % | MME-P |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['run_id']} | {r['config']} | {100 * r['yes_ratio']:.1f} | {100 * r['recall_yes']:.1f} | "
                     f"{100 * r['specificity_no']:.1f} | {100 * r['other_ratio']:.1f} | {r['perception']:.1f} |")
    keys = ("vanilla", "fastv64", "random64_s2021", "visionzip64", "sparsevlm64")
    pick = {r["config"]: r for r in rows if r["run_id"].startswith("EXP-0002") and r["config"] in keys}
    cats = list(pick["vanilla"]["by_category"])
    lines += ["", "按子任务的 yes 比例 / yes 题召回 / no 题特异度（%，64 档与 vanilla）：", "",
              "| 子任务 | 题数 | " + " | ".join(k for k in keys if k in pick) + " |", "| --- | --- |" + " --- |" * len(pick)]
    for c in cats:
        cells = []
        for k in keys:
            if k in pick:
                v = pick[k]["by_category"][c]
                cells.append(f"{100 * v['yes_ratio']:.0f} / {100 * v['recall_yes']:.0f} / {100 * v['specificity_no']:.0f}")
        lines.append(f"| {c} | {pick['vanilla']['by_category'][c]['n']} | " + " | ".join(cells) + " |")
    with open(f"{OUT}/EXP-0002/deepdive2/mme_yes_ratio.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
