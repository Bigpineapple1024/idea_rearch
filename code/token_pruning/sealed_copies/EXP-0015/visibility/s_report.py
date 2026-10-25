"""EXP-0015, before R02/R03 (report only, no rule changes): the visible share S per question for POPE and each MME
subtask, from the question texts, the per-file IDF and the visibility table."""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from transformers import AutoTokenizer   # noqa: E402
import run_eval   # noqa: E402

E = "D:/6CJNM007/data/llava_eval"
vis = np.load("D:/6CJNM007/idea_rearch/code/output/EXP-0015/visibility/visibility.npy")
tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
out = []
for name, path in (("POPE", f"{E}/pope/llava_pope_test.jsonl"), ("MME", f"{E}/MME/llava_mme.jsonl")):
    idf = run_eval.question_idf(tok, path, "vicuna_v1")
    groups = defaultdict(list)
    for line in open(path):
        q = json.loads(line)
        ids = run_eval.prompt_ids_after_image(tok, q["text"], "vicuna_v1")
        w = np.array([idf["idf"].get(t, idf["unseen"]) for t in ids])
        s = float((w * vis[ids]).sum() / w.sum()) if w.sum() > 0 else 0.0
        groups[q["category"] if name == "MME" else "all"].append(s)
    for g, v in sorted(groups.items()):
        v = np.array(v)
        out.append(f"| {name} | {g} | {len(v)} | {v.mean():.3f} | {np.median(v):.3f} | {round(30 * v.mean())} |")
print("| 数据 | 组 | 题数 | S 均值 | S 中位数 | 平均按查询选的个数 |\n| --- | --- | --- | --- | --- | --- |")
print("\n".join(out))
