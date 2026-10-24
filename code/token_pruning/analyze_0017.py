"""EXP-0017 (plan v1): A'+B+V vs A at the 192 budget (K1 POPE, K2 MME), with the 192 references of EXP-0002.

  python analyze_0017.py      # writes code/output/EXP-0017/analysis/{0017.md,0017.json}
"""
import json
import os
import sys

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_0016 import counts_check, ref   # noqa: E402
from analyze_query import mme_boot, row   # noqa: E402
from analyze_repope import EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, paired_boot, pope_no, read_labels, score   # noqa: E402
from bootstrap_ci import PERCEPTION, load_mme, mme_stat   # noqa: E402


def main():
    out_dir = f"{OUT}/EXP-0017/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    VIS = f"{OUT}/EXP-0015/visibility/visibility.npy"
    mmeq, popeq = f"{EVAL_DATA}/MME/llava_mme.jsonl", f"{EVAL_DATA}/pope/llava_pope_test.jsonl"
    checks = {"R02 counts = round(166 S)": counts_check("EXP-0017/EXP-0017-R02", mmeq, 166, VIS),
              "R04 counts = round(166 S)": counts_check("EXP-0017/EXP-0017-R04", popeq, 166, VIS)}
    POPE = {"A_192": "EXP-0017/EXP-0017-R03", "A'+B+V_192": "EXP-0017/EXP-0017-R04", "fastv192": ref("fastv192", "pope"),
            "visionzip192": ref("visionzip192", "pope"), "sparsevlm192": ref("sparsevlm192", "pope"), "vanilla": "EXP-0002/EXP-0002-R04"}
    MME = {"A_192": "EXP-0017/EXP-0017-R01", "A'+B+V_192": "EXP-0017/EXP-0017-R02", "fastv192": ref("fastv192", "mme"),
           "visionzip192": ref("visionzip192", "mme"), "sparsevlm192": ref("sparsevlm192", "mme"), "vanilla": "EXP-0002/EXP-0002-R02"}
    key_of = {}
    for line in open(popeq):
        q = json.loads(line)
        key_of[q["question_id"]] = (q["category"], q["image"], OBJECT.match(q["text"]).group(1))
    original = read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json")
    repope = {k: v for k, v in read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json").items() if k in original}
    said, pope = {}, {}
    for n, run in POPE.items():
        s = {key_of[json.loads(l)["question_id"]]: not pope_no(json.loads(l)["text"]) for l in open(f"{OUT}/{run}/answers.jsonl")}
        per, f1 = score(s, original)
        checks[f"POPE {n} headline"] = float(abs(f1 - float(row(run)["headline"])) < 1e-6)
        said[n] = s
        pope[n] = {"run": run, "f1": f1, "f1_repope": score(s, repope)[1], "accuracy": 100 * float(np.mean([per[x]["accuracy"] for x in SPLITS]))}
    pairs = [("A'+B+V_192", "A_192"), ("A'+B+V_192", "visionzip192"), ("A'+B+V_192", "sparsevlm192"), ("A'+B+V_192", "vanilla")]
    pp = {f"{a} - {b}": {"original": paired_boot(said[a], said[b], original), "repope": paired_boot(said[a], said[b], repope)} for a, b in pairs}
    items, mme = {}, {}
    for n, run in MME.items():
        it = load_mme(f"{OUT}/{run}")
        items[n] = it
        tot = mme_stat(it, {k: np.arange(len(v)) for k, v in it.items()})
        checks[f"MME {n} headline"] = float(abs(tot - float(row(run)["headline"])) < 1e-6)
        mme[n] = {"run": run, "perception": tot, "per_task": {t: 100 * float(it[t].mean() + it[t].all(axis=1).mean()) for t in PERCEPTION}}
    mp = {f"{a} - {b}": mme_boot(items[a], items[b]) for a, b in pairs}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    k1 = all(pp["A'+B+V_192 - A_192"][lab][1] > 0 for lab in ("original", "repope"))
    k2 = mp["A'+B+V_192 - A_192"][1] > 0
    k2n = mp["A'+B+V_192 - A_192"][2] >= 0
    os.makedirs(out_dir)
    json.dump({"checks": checks, "pope": pope, "pope_pairs": pp, "mme": mme, "mme_pairs": mp, "K1": k1, "K2": k2, "K2_not_worse": k2n},
              open(f"{out_dir}/0017.json", "w"), indent=2)
    L = ["# EXP-0017（计划 v1）：192 档", "", "## 核对", ""] + [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", "## POPE", "", "| 配置 | F1 原始 | F1 RePOPE | 准确率 |", "| --- | --- | --- | --- |"]
    L += [f"| {n}（{r['run']}） | {r['f1']:.2f} | {r['f1_repope']:.2f} | {r['accuracy']:.2f} |" for n, r in pope.items()]
    L += ["", "| 配对差 | 原始 | RePOPE |", "| --- | --- | --- |"]
    L += [f"| {k} | {v['original'][0]:+.2f} [{v['original'][1]:+.2f}, {v['original'][2]:+.2f}] | {v['repope'][0]:+.2f} [{v['repope'][1]:+.2f}, {v['repope'][2]:+.2f}] |" for k, v in pp.items()]
    L += ["", "## MME", "", "| 配置 | MME-P | " + " | ".join(PERCEPTION) + " |", "| --- | --- | " + " | ".join("---" for _ in PERCEPTION) + " |"]
    L += [f"| {n} | {r['perception']:.1f} | " + " | ".join(f"{r['per_task'][t]:.1f}" for t in PERCEPTION) + " |" for n, r in mme.items()]
    L += ["", "| 配对差（MME-P） | 差 [95% 区间] |", "| --- | --- |"] + [f"| {k} | {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] |" for k, v in mp.items()]
    L += ["", f"**判据**：K1（192 档 POPE 提升）{'成立' if k1 else '不成立'}；K2（192 档 MME 提升）{'成立' if k2 else '不成立'}"
              f"{'；区间不在 0 以下（不变差）' if k2n and not k2 else ''}。"]
    open(f"{out_dir}/0017.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
