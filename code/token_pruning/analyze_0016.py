"""EXP-0016 (plan v1): H1 fixed split vs visibility on MME (64 budget); G1 / G2 A'+B+V vs A at the 128 budget.

  python analyze_0016.py      # writes code/output/EXP-0016/analysis/{0016.md,0016.json}
"""
import csv
import json
import os
import sys

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_query import mme_boot, row   # noqa: E402
from analyze_repope import EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, paired_boot, pope_no, read_labels, score   # noqa: E402
from bootstrap_ci import PERCEPTION, load_mme, mme_stat   # noqa: E402


def ref(config, bench):
    with open(f"{OUT}/EXP-0002/runs.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["config"] == config and r["bench"] == bench and r["status"] == "completed":
                return f"EXP-0002/{r['run_id']}"
    raise SystemExit(f"no EXP-0002 run for {config} x {bench}")


def counts_check(run, question_file, keep, vis_path):
    from transformers import AutoTokenizer
    import run_eval
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    vis = np.load(vis_path)
    idf = run_eval.question_idf(tok, question_file, "vicuna_v1")
    kq = json.load(open(f"{OUT}/{run}/kept_tokens.json"))["kept_by_object_name"]
    same = 0
    for k, line in zip(kq, open(question_file)):
        ids = run_eval.prompt_ids_after_image(tok, json.loads(line)["text"], "vicuna_v1")
        w = np.array([idf["idf"].get(t, idf["unseen"]) for t in ids])
        s = float((w * vis[ids]).sum() / w.sum()) if w.sum() > 0 else 0.0
        same += len(k) == round(keep * s)
    return same / len(kq)


def main():
    out_dir = f"{OUT}/EXP-0016/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    VIS = f"{OUT}/EXP-0015/visibility/visibility.npy"
    HALF = f"{OUT}/EXP-0016/visibility_half/visibility.npy"
    mmeq, popeq = f"{EVAL_DATA}/MME/llava_mme.jsonl", f"{EVAL_DATA}/pope/llava_pope_test.jsonl"
    checks = {"R01 counts = round(30 S), S = 0.5": counts_check("EXP-0016/EXP-0016-R01", mmeq, 30, HALF),
              "R05 counts = round(98 S)": counts_check("EXP-0016/EXP-0016-R05", mmeq, 98, VIS),
              "R03 counts = round(98 S)": counts_check("EXP-0016/EXP-0016-R03", popeq, 98, VIS)}
    MME = {"A'+B+V_64": "EXP-0015/EXP-0015-R02", "fixed_split_64": "EXP-0016/EXP-0016-R01", "A_64": "EXP-0014/EXP-0014-R04",
           "A_128": "EXP-0016/EXP-0016-R04", "A'+B+V_128": "EXP-0016/EXP-0016-R05", "fastv128": ref("fastv128", "mme"),
           "visionzip128": ref("visionzip128", "mme"), "sparsevlm128": ref("sparsevlm128", "mme")}
    POPE = {"A_128": "EXP-0016/EXP-0016-R02", "A'+B+V_128": "EXP-0016/EXP-0016-R03", "fastv128": ref("fastv128", "pope"),
            "visionzip128": ref("visionzip128", "pope"), "sparsevlm128": ref("sparsevlm128", "pope")}
    order = [(json.loads(l)["question_id"], json.loads(l)) for l in open(popeq)]
    key_of = {qid: (q["category"], q["image"], OBJECT.match(q["text"]).group(1)) for qid, q in order}
    original = read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json")
    repope = {k: v for k, v in read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json").items() if k in original}
    said, pope = {}, {}
    for n, run in POPE.items():
        s = {key_of[json.loads(l)["question_id"]]: not pope_no(json.loads(l)["text"]) for l in open(f"{OUT}/{run}/answers.jsonl")}
        per, f1 = score(s, original)
        checks[f"POPE {n} headline"] = float(abs(f1 - float(row(run)["headline"])) < 1e-6)
        said[n] = s
        pope[n] = {"run": run, "f1": f1, "f1_repope": score(s, repope)[1],
                   "accuracy": 100 * float(np.mean([per[x]["accuracy"] for x in SPLITS]))}
    pp = {f"{a} - {b}": {"original": paired_boot(said[a], said[b], original), "repope": paired_boot(said[a], said[b], repope)}
          for a, b in [("A'+B+V_128", "A_128"), ("A'+B+V_128", "visionzip128"), ("A'+B+V_128", "sparsevlm128")]}
    items, mme = {}, {}
    for n, run in MME.items():
        it = load_mme(f"{OUT}/{run}")
        items[n] = it
        tot = mme_stat(it, {k: np.arange(len(v)) for k, v in it.items()})
        checks[f"MME {n} headline"] = float(abs(tot - float(row(run)["headline"])) < 1e-6)
        mme[n] = {"run": run, "perception": tot, "per_task": {t: 100 * float(it[t].mean() + it[t].all(axis=1).mean()) for t in PERCEPTION}}
    mp = {f"{a} - {b}": mme_boot(items[a], items[b]) for a, b in [
        ("A'+B+V_64", "fixed_split_64"), ("fixed_split_64", "A_64"), ("A'+B+V_128", "A_128"), ("A'+B+V_128", "visionzip128"),
        ("A'+B+V_128", "sparsevlm128")]}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    h1 = mp["A'+B+V_64 - fixed_split_64"][1] > 0
    g1 = all(pp["A'+B+V_128 - A_128"][lab][1] > 0 for lab in ("original", "repope"))
    g2 = mp["A'+B+V_128 - A_128"][1] > 0
    g2p = mp["A'+B+V_128 - A_128"][0] > 0 and not g2
    summary = {"checks": checks, "pope": pope, "pope_pairs": pp, "mme": mme, "mme_pairs": mp, "H1": h1, "G1": g1, "G2": g2, "G2_not_worse": g2p}
    os.makedirs(out_dir)
    json.dump(summary, open(f"{out_dir}/0016.json", "w"), indent=2)
    L = ["# EXP-0016（计划 v1）", "", "## 核对", ""] + [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", "## POPE（128 档）", "", "| 配置 | F1 原始 | F1 RePOPE | 准确率 |", "| --- | --- | --- | --- |"]
    L += [f"| {n}（{r['run']}） | {r['f1']:.2f} | {r['f1_repope']:.2f} | {r['accuracy']:.2f} |" for n, r in pope.items()]
    L += ["", "| 配对差 | 原始 | RePOPE |", "| --- | --- | --- |"]
    L += [f"| {k} | {v['original'][0]:+.2f} [{v['original'][1]:+.2f}, {v['original'][2]:+.2f}] | {v['repope'][0]:+.2f} [{v['repope'][1]:+.2f}, {v['repope'][2]:+.2f}] |" for k, v in pp.items()]
    L += ["", "## MME", "", "| 配置 | MME-P | " + " | ".join(PERCEPTION) + " |", "| --- | --- | " + " | ".join("---" for _ in PERCEPTION) + " |"]
    L += [f"| {n} | {r['perception']:.1f} | " + " | ".join(f"{r['per_task'][t]:.1f}" for t in PERCEPTION) + " |" for n, r in mme.items()]
    L += ["", "| 配对差（MME-P） | 差 [95% 区间] |", "| --- | --- |"] + [f"| {k} | {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] |" for k, v in mp.items()]
    L += ["", f"**判据**：H1（可视性在切分之外有贡献）{'成立' if h1 else '不成立'}；G1（128 档 POPE 提升）{'成立' if g1 else '不成立'}；"
              f"G2（128 档 MME 提升）{'成立' if g2 else '不成立'}{'；点估计 > 0、区间含 0' if g2p else ''}。"]
    open(f"{out_dir}/0016.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
