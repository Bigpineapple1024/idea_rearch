"""EXP-0010 details for SQZ-EXP-0010: POPE recall and yes ratio by split, MME perception subtasks, and the missed POPE
positives of the block-stratified FastV-64 runs next to FastV-64, its position-corrected version (EXP-0008), random-64
and VisionZip-64 (the same layout as analyze_0007_details.py).

  python analyze_0010_details.py      # writes code/output/EXP-0010/analysis/{details.md,details.json}
"""
import json
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_missed_objects import SPLITS, load_questions, pope_no  # noqa: E402

OUT = "D:/6CJNM007/idea_rearch/code/output"
POPE = {"fastv64_strat": "EXP-0010/EXP-0010-R03", "fastv64_prior_strat": "EXP-0010/EXP-0010-R05",
        "fastv64_prior": "EXP-0008/EXP-0008-R05", "fastv64": "EXP-0002/EXP-0002-R11",
        "random64_s2021": "EXP-0002/EXP-0002-R29", "visionzip64": "EXP-0002/EXP-0002-R17", "vanilla": "EXP-0002/EXP-0002-R04"}
MME = {"fastv64_strat": "EXP-0010/EXP-0010-R04", "fastv64_prior_strat": "EXP-0010/EXP-0010-R06",
       "fastv64_prior": "EXP-0008/EXP-0008-R06", "fastv64": "EXP-0002/EXP-0002-R10",
       "random64_s2021": "EXP-0002/EXP-0002-R28", "visionzip64": "EXP-0002/EXP-0002-R35", "vanilla": "EXP-0002/EXP-0002-R02"}
PAIRS = (("fastv64_strat", "fastv64"), ("fastv64_strat", "fastv64_prior"), ("fastv64_prior_strat", "fastv64_strat"),
         ("fastv64_prior_strat", "random64_s2021"), ("fastv64_prior_strat", "visionzip64"))
MME_DIFFS = (("fastv64_strat", "fastv64"), ("fastv64_strat", "fastv64_prior"), ("fastv64_prior_strat", "random64_s2021"))


def missed_positives():
    questions, labels = load_questions()
    pos = sorted({(img, obj) for (sp, img, obj), y in labels.items() if sp == "adversarial" and y})
    for sp in SPLITS:
        assert {(img, obj) for (s, img, obj), y in labels.items() if s == sp and y} == set(pos)
    index = {p: i for i, p in enumerate(pos)}
    missed = {}
    for c, run in POPE.items():
        arr = np.zeros((len(SPLITS), len(pos)), dtype=bool)
        seen = np.zeros_like(arr)
        with open(f"{OUT}/{run}/answers.jsonl") as f:
            for line in f:
                a = json.loads(line)
                split, img, obj = questions[a["question_id"]]
                if labels[(split, img, obj)]:
                    k, i = SPLITS.index(split), index[(img, obj)]
                    arr[k, i], seen[k, i] = pope_no(a["text"]), True
        assert seen.all(), c
        missed[c] = arr
    return pos, missed


def main():
    pope = {}
    for c, run in POPE.items():
        m = json.load(open(f"{OUT}/{run}/metrics.json"))
        s = m["splits"]
        pope[c] = {"f1_mean": m["f1_mean"], **{key: [s[k][key] for k in SPLITS]
                                               for key in ("recall", "precision", "yes_ratio", "TP", "FP")}}
    mme = {c: json.load(open(f"{OUT}/{run}/metrics.json")) for c, run in MME.items()}
    pos, missed = missed_positives()
    van = missed["vanilla"].any(axis=0)
    assert (missed["vanilla"] == missed["vanilla"][0]).all()
    per = {}
    for c, arr in missed.items():
        times = arr.sum(axis=0)
        per[c] = {"miss_rate": float(arr.mean()), "same_answer_all_splits": float(((times == 0) | (times == 3)).mean()),
                  "miss_rate_where_vanilla_correct": float(arr[:, ~van].mean()),
                  "miss_rate_where_vanilla_missed": float(arr[:, van].mean())}
    pairs = {}
    for a, b in PAIRS:
        d = missed[a].mean(axis=0) - missed[b].mean(axis=0)
        pairs[f"{a}|{b}"] = {"mean_difference": float(d.mean()), "positives_worse": int((d > 0).sum()),
                             "positives_better": int((d < 0).sum()), "positives_equal": int((d == 0).sum()),
                             "difference_where_vanilla_correct": float(d[~van].mean()),
                             "difference_where_vanilla_missed": float(d[van].mean())}
    summary = {"n_positives": len(pos), "vanilla_missed": int(van.sum()), "pope": pope, "missed": per, "pairs": pairs,
               "mme_perception": {c: m["perception"] for c, m in mme.items()},
               "mme_tasks": {c: m["perception_tasks"] for c, m in mme.items()}}
    with open(f"{OUT}/EXP-0010/analysis/details.json", "w") as f:
        json.dump(summary, f, indent=2)

    pct = lambda xs: " / ".join(f"{100 * x:.1f}" for x in xs)
    lines = ["POPE 按 split（adversarial / popular / random）：", "",
             "| 配置 | F1 平均 | 召回 % | 精确率 % | yes 比例 % | TP | FP |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for c, v in pope.items():
        lines.append(f"| {c} | {v['f1_mean']:.2f} | {pct(v['recall'])} | {pct(v['precision'])} | {pct(v['yes_ratio'])} | "
                     f"{' / '.join(map(str, v['TP']))} | {' / '.join(map(str, v['FP']))} |")
    lines += ["", f"漏答的正样本（{len(pos)} 个正样本；vanilla 漏掉 {int(van.sum())} 个）：", "",
              "| 配置 | 漏答率 % | 三个 split 回答相同 % | vanilla 答对的正样本上的漏答率 % | vanilla 漏掉的正样本上的漏答率 % |",
              "| --- | --- | --- | --- | --- |"]
    for c, v in per.items():
        lines.append(f"| {c} | {100 * v['miss_rate']:.1f} | {100 * v['same_answer_all_splits']:.1f} | "
                     f"{100 * v['miss_rate_where_vanilla_correct']:.1f} | {100 * v['miss_rate_where_vanilla_missed']:.1f} |")
    lines += ["", "逐个正样本的漏答比例之差（前者 − 后者，三个 split 平均）：", "",
              "| 比较 | 平均差（百分点） | 前者更差 / 更好 / 相同的正样本数 | vanilla 答对处的差 | vanilla 漏掉处的差 |",
              "| --- | --- | --- | --- | --- |"]
    for k, v in pairs.items():
        lines.append(f"| {k.replace('|', ' − ')} | {100 * v['mean_difference']:+.1f} | {v['positives_worse']} / "
                     f"{v['positives_better']} / {v['positives_equal']} | {100 * v['difference_where_vanilla_correct']:+.1f} | "
                     f"{100 * v['difference_where_vanilla_missed']:+.1f} |")
    tasks = list(mme["fastv64_strat"]["perception_tasks"])
    cols = list(MME)
    lines += ["", "MME Perception 子任务：", "", "| 子任务 | " + " | ".join(cols) + " | "
              + " | ".join(f"{a} − {b}" for a, b in MME_DIFFS) + " |", "| --- |" + " --- |" * (len(cols) + len(MME_DIFFS))]
    for t in tasks + ["perception"]:
        v = {c: (mme[c]["perception_tasks"][t] if t != "perception" else mme[c]["perception"]) for c in cols}
        lines.append(f"| {t} | " + " | ".join(f"{v[c]:.1f}" for c in cols) + " | "
                     + " | ".join(f"{v[a] - v[b]:+.1f}" for a, b in MME_DIFFS) + " |")
    with open(f"{OUT}/EXP-0010/analysis/details.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
