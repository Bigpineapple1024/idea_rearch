"""只读：EXP-0024 覆盖 16（R18 VQAv2 dev2、R17 GQA val400）每题框内保留个数的分布，看有多少题 30 个全在框里。"""
import json
from collections import Counter

O = "D:/6CJNM007/idea_rearch/code/output/EXP-0024"
for rid, bench in (("R18", "vqav2_dev2"), ("R17", "gqa_val400")):
    recs = [x for x in json.load(open(f"{O}/EXP-0024-{rid}/zoom.json"))["records"] if x["cover"]]
    n = len(recs)
    allin = sum(x["in_box"] == 30 for x in recs)
    base = Counter(min(x["baseline_in_box"] // 5 * 5, 30) for x in recs)
    side = Counter(x["side"] for x in recs)
    print(f"{bench} ({rid}): covered {n}; all 30 in the box {allin} ({allin / n:.1%}); "
          f"in_box mean {sum(x['in_box'] for x in recs) / n:.1f}; baseline_in_box mean {sum(x['baseline_in_box'] for x in recs) / n:.1f}")
    print("  baseline_in_box by 5s:", dict(sorted(base.items())))
    print("  box side:", dict(sorted(side.items())))
    print("  box area share of 576 (mean):", round(sum(x["side"] ** 2 for x in recs) / n / 576, 3))
