"""V* 上被路由的题按类别计数（只读题目文字与类别，不读答案、不调用模型）。"""
import collections
import json
import sys

sys.path.insert(0, "D:/6CJNM007/tmp/exp0024/code")
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
from run_eval import routes_to_query  # noqa: E402

c = collections.Counter()
for line in open("D:/6CJNM007/data/llava_eval/vstar/llava_vstar_test.jsonl", encoding="utf-8"):
    q = json.loads(line)
    c[(q["category"], routes_to_query(q["text"]))] += 1
print(dict(c))
