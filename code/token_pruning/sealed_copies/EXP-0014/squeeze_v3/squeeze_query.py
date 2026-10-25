"""SQZ-EXP-0014 exploratory checks: (1) POPE questions where the object name gets < 90% of the IDF weight ("imange"):
do A'+B and the object-name query disagree more there? (2) MME: per subtask, question-level accuracy of A'+B vs A and the
highest-IDF tokens of the questions where A'+B loses."""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from transformers import AutoTokenizer   # noqa: E402
import run_eval   # noqa: E402
from analyze_repope import pope_no   # noqa: E402
from prompt_spans import question_spans   # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=False)
O = "D:/6CJNM007/idea_rearch/code/output"
E = "D:/6CJNM007/data/llava_eval"
tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
L = ["# SQZ-EXP-0014 探索性检查", ""]

qs = [json.loads(line) for line in open(f"{E}/pope/llava_pope_test.jsonl")]
idf = run_eval.question_idf(tok, f"{E}/pope/llava_pope_test.jsonl", "vicuna_v1")
share = []
for q in qs:
    ids = run_eval.prompt_ids_after_image(tok, q["text"], "vicuna_v1")
    w = np.array([idf["idf"][t] for t in ids])
    share.append(w[question_spans(tok, ids)[1]].sum() / w.sum())
share = np.array(share)
a = [not pope_no(json.loads(line)["text"]) for line in open(f"{O}/EXP-0014/EXP-0014-R02/answers.jsonl")]
b = [not pope_no(json.loads(line)["text"]) for line in open(f"{O}/EXP-0011/EXP-0011-R06/answers.jsonl")]
ka = json.load(open(f"{O}/EXP-0014/EXP-0014-R02/kept_tokens.json"))["kept"]
kb = json.load(open(f"{O}/EXP-0011/EXP-0011-R06/kept_tokens.json"))["kept"]
dis, ov = np.array(a) != np.array(b), np.array([len(set(x) & set(y)) / 30 for x, y in zip(ka, kb)])
low = share < 0.9
L += ["## POPE：物体名权重不足 90% 的题（拼写 imange）", "",
      f"- {low.sum()} 题：A′+B 与物体名查询答案不同的比例 {100 * dis[low].mean():.1f}%，保留集合重合 {100 * ov[low].mean():.1f}%",
      f"- 其余 {(~low).sum()} 题：答案不同 {100 * dis[~low].mean():.1f}%，重合 {100 * ov[~low].mean():.1f}%", ""]

midf = run_eval.question_idf(tok, f"{E}/MME/llava_mme.jsonl", "vicuna_v1")
SUFFIX = "\nAnswer the question using a single word or phrase."
L += ["## MME：各子任务逐题准确率（A′+B 对 A）与只有 A′+B 答错的题里 IDF 最高的词", "",
      "| 子任务 | 题数 | A 准确率 | A′+B 准确率 | 只有 A′+B 错 / 只有 A 错 | 只有 A′+B 错的题里 IDF 最高的词（前 6） |", "| --- | --- | --- | --- | --- | --- |"]
tool = lambda run, task: [l.rstrip("\n").split("\t") for l in open(f"{O}/EXP-0014/{run}/mme_eval_tool/{task}.txt", encoding="utf-8")]  # noqa: E731
for task in sorted(os.listdir(f"{O}/EXP-0014/EXP-0014-R04/mme_eval_tool")):
    task = task[:-4]
    ra, rb = tool("EXP-0014-R04", task), tool("EXP-0014-R06", task)
    ok = lambda r: r[3].strip().lower()[:3].startswith(r[2].strip().lower())   # noqa: E731
    acc_a, acc_b = [ok(r) for r in ra], [ok(r) for r in rb]
    top = Counter()
    for r, x, y in zip(ra, acc_a, acc_b):
        if x and not y:
            text = r[1].replace(" Please answer yes or no.", "").replace("Please answer yes or no.", "").strip()
            ids = run_eval.prompt_ids_after_image(tok, text + SUFFIX, "vicuna_v1")   # the prompt LLaVA's MME file uses
            t = max(ids, key=lambda t: midf["idf"].get(t, midf["unseen"]))
            top[tok.convert_ids_to_tokens([t])[0]] += 1
    only_b = sum(x and not y for x, y in zip(acc_a, acc_b))
    only_a = sum(y and not x for x, y in zip(acc_a, acc_b))
    L.append(f"| {task} | {len(ra)} | {100 * np.mean(acc_a):.1f}% | {100 * np.mean(acc_b):.1f}% | {only_b} / {only_a} | "
             + ", ".join(f"{k} {v}" for k, v in top.most_common(6)) + " |")
open(f"{out}/squeeze_query.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
