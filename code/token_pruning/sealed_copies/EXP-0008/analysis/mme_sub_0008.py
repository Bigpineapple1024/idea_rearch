"""MME perception subtasks of EXP-0008-R06 next to FastV-64 (EXP-0002-R10), random-64 s2021 (R28) and VisionZip-64 (R35)."""
import json

OUT = "D:/6CJNM007/idea_rearch/code/output"
runs = {"fastv64_prior": f"{OUT}/EXP-0008/EXP-0008-R06", "fastv64": f"{OUT}/EXP-0002/EXP-0002-R10",
        "random64_s2021": f"{OUT}/EXP-0002/EXP-0002-R28", "visionzip64": f"{OUT}/EXP-0002/EXP-0002-R35"}
m = {k: json.load(open(f"{v}/metrics.json")) for k, v in runs.items()}
tasks = list(m["fastv64_prior"]["perception_tasks"])
print("| 子任务 | " + " | ".join(runs) + " | 校正 − FastV | 校正 − 随机 |")
print("| --- |" + " --- |" * (len(runs) + 2))
for t in tasks:
    v = {k: m[k]["perception_tasks"][t] for k in runs}
    v = {k: (x["total"] if isinstance(x, dict) else x) for k, x in v.items()}
    print(f"| {t} | " + " | ".join(f"{v[k]:.1f}" for k in runs)
          + f" | {v['fastv64_prior'] - v['fastv64']:+.1f} | {v['fastv64_prior'] - v['random64_s2021']:+.1f} |")
print("| perception | " + " | ".join(f"{m[k]['perception']:.1f}" for k in runs) + " | | |")
