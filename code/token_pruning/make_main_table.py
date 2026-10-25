"""Draft main table for the paper: method x budget x benchmark, from the sealed runs (writing phase, no new computation).

  python make_main_table.py <out folder>      # writes main_table.md and main_table.json

POPE: F1 averaged over the three splits, original labels (metrics.json) and RePOPE (recomputed here, as analyze_repope);
MME: perception total; VQAv2: accuracy on test2 (EXP-0020; 192-budget router and A, and 128-budget references, from EXP-0021). A cell with no run is left empty ("—"); every cell lists its
run ID in the JSON, and every run must be completed.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
from analyze_repope import EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, pope_no, read_labels   # noqa: E402

# (method, budget) -> {benchmark: run}; budgets are 32-layer averages of visual tokens
CELLS = {
    ("vanilla", 576): {"pope": "EXP-0002/EXP-0002-R04", "mme": "EXP-0002/EXP-0002-R02", "vqav2": "EXP-0020/EXP-0020-R07"},
    ("FastV", 64): {"pope": "EXP-0002/EXP-0002-R11", "mme": "EXP-0002/EXP-0002-R10"},
    ("FastV", 128): {"pope": "EXP-0002/EXP-0002-R09", "mme": "EXP-0002/EXP-0002-R08"},
    ("FastV", 192): {"pope": "EXP-0002/EXP-0002-R07", "mme": "EXP-0002/EXP-0002-R06"},
    ("VisionZip", 64): {"pope": "EXP-0002/EXP-0002-R17", "mme": "EXP-0002/EXP-0002-R35", "vqav2": "EXP-0020/EXP-0020-R08"},
    ("VisionZip", 128): {"pope": "EXP-0002/EXP-0002-R15", "mme": "EXP-0002/EXP-0002-R14", "vqav2": "EXP-0021/EXP-0021-R03"},
    ("VisionZip", 192): {"pope": "EXP-0002/EXP-0002-R13", "mme": "EXP-0002/EXP-0002-R12"},
    ("SparseVLM", 64): {"pope": "EXP-0002/EXP-0002-R41", "mme": "EXP-0002/EXP-0002-R40", "vqav2": "EXP-0020/EXP-0020-R09"},
    ("SparseVLM", 128): {"pope": "EXP-0002/EXP-0002-R39", "mme": "EXP-0002/EXP-0002-R38", "vqav2": "EXP-0021/EXP-0021-R04"},
    ("SparseVLM", 192): {"pope": "EXP-0002/EXP-0002-R37", "mme": "EXP-0002/EXP-0002-R36"},
    ("A (strong baseline)", 64): {"pope": "EXP-0011/EXP-0011-R07", "mme": "EXP-0014/EXP-0014-R04", "vqav2": "EXP-0020/EXP-0020-R01"},
    ("A (strong baseline)", 128): {"pope": "EXP-0016/EXP-0016-R02", "mme": "EXP-0016/EXP-0016-R04", "vqav2": "EXP-0020/EXP-0020-R04"},
    ("A (strong baseline)", 192): {"pope": "EXP-0017/EXP-0017-R03", "mme": "EXP-0017/EXP-0017-R01", "vqav2": "EXP-0021/EXP-0021-R01"},
    ("A'+B+V", 64): {"pope": "EXP-0015/EXP-0015-R03", "mme": "EXP-0015/EXP-0015-R02", "vqav2": "EXP-0020/EXP-0020-R03"},
    ("A'+B+V", 128): {"pope": "EXP-0016/EXP-0016-R03", "mme": "EXP-0016/EXP-0016-R05", "vqav2": "EXP-0020/EXP-0020-R06"},
    ("A'+B+V", 192): {"pope": "EXP-0017/EXP-0017-R04", "mme": "EXP-0017/EXP-0017-R02"},
    # the router equals A'+B+V on POPE and MME-P (every question routed, check_route.py); its own runs are VQAv2 only
    ("router", 64): {"pope": "EXP-0015/EXP-0015-R03", "mme": "EXP-0015/EXP-0015-R02", "vqav2": "EXP-0020/EXP-0020-R02"},
    ("router", 128): {"pope": "EXP-0016/EXP-0016-R03", "mme": "EXP-0016/EXP-0016-R05", "vqav2": "EXP-0020/EXP-0020-R05"},
    ("router", 192): {"pope": "EXP-0017/EXP-0017-R04", "mme": "EXP-0017/EXP-0017-R02", "vqav2": "EXP-0021/EXP-0021-R02"},
}


def status(run):
    exp, rid = run.split("/")
    path = f"{OUT}/{exp}/runs.tsv"
    if not os.path.exists(path):
        return None
    rows = {r["run_id"]: r for r in csv.DictReader(open(path), delimiter="\t")}
    return rows.get(rid, {}).get("status")


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        sys.exit(f"{out} exists; results are never overwritten")
    qs = [json.loads(l) for l in open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl")]
    keys = [(q["category"], q["image"], OBJECT.match(q["text"]).group(1)) for q in qs]
    repope = read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json")

    def repope_f1(run):
        said = {json.loads(l)["question_id"]: not pope_no(json.loads(l)["text"]) for l in open(f"{OUT}/{run}/answers.jsonl")}
        f = []
        for s in SPLITS:
            idx = [i for i, k in enumerate(keys) if k[0] == s and k in repope]
            y = np.array([repope[keys[i]] for i in idx])
            p = np.array([said[qs[i]["question_id"]] for i in idx])
            tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
            f.append(2 * tp / (2 * tp + fp + fn))
        return 100 * float(np.mean(f))

    table, missing = {}, []
    for (method, budget), runs in CELLS.items():
        cell = {}
        for bench, run in runs.items():
            st = status(run)
            if st != "completed":
                missing.append(f"{method} {budget} {bench}: {run} ({st})")
                continue
            m = json.load(open(f"{OUT}/{run}/metrics.json"))
            if bench == "pope":
                cell["pope"], cell["repope"] = m["f1_mean"], repope_f1(run)
            elif bench == "mme":
                cell["mme"] = m["perception"]
            else:
                cell["vqav2"] = m["accuracy"]
            cell[f"{bench}_run"] = run
        table[f"{method} | {budget}"] = cell
    os.makedirs(out)
    json.dump({"table": table, "missing": missing}, open(f"{out}/main_table.json", "w"), indent=2)
    fmt = lambda c, k, d: f"{c[k]:.{d}f}" if k in c else "—"   # noqa: E731
    L = ["# 主表草稿（LLaVA-1.5-7B；预算为 32 层平均的视觉 token 数）", "",
         "| 方法 | 预算 | POPE F1 | RePOPE F1 | MME-P | VQAv2 test2 |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {k.split(' | ')[0]} | {k.split(' | ')[1]} | {fmt(c, 'pope', 2)} | {fmt(c, 'repope', 2)} | {fmt(c, 'mme', 1)} | {fmt(c, 'vqav2', 2)} |"
          for k, c in table.items()]
    L += ["", "路由在 POPE 与 MME-P 上的数字就是 A′+B+V 的（这两个 benchmark 的题全部被路由，check_route.py）。", ""]
    L += ["缺的格子：" if missing else "没有缺的格子。"] + [f"- {x}" for x in missing]
    open(f"{out}/main_table.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
