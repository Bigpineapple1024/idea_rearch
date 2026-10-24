"""EXP-0015 (plan v1): IDF x visibility query with the visible share S of the budget (A'+B+V) against A, A'+B and the
references, on MME (V1) and POPE (V2).

  python analyze_vis.py      # writes code/output/EXP-0015/analysis/{vis.md,vis.json}

Checks (stop if violated): every rescored headline equals the registered one; in R02 and R03 the logged number of
query-chosen tokens equals round(30 * S) recomputed offline (>= 99% of the questions). V1: MME-P A'+B+V - A interval
above 0 (V1'): point above 0, interval containing 0). V2: POPE F1 A'+B+V - A above 0 with both label sets.
"""
import json
import os
import sys

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_query import mme_boot, row   # noqa: E402
from analyze_repope import EVAL_DATA, OBJECT, OUT, REPOPE, SPLITS, paired_boot, pope_no, read_labels, score   # noqa: E402
from bootstrap_ci import PERCEPTION, load_mme, mme_stat   # noqa: E402

POPE = {"A'+B+V": "EXP-0015/EXP-0015-R03", "A": "EXP-0011/EXP-0011-R07", "A'+B": "EXP-0014/EXP-0014-R02",
        "object_name": "EXP-0011/EXP-0011-R06", "visionzip64": "EXP-0002/EXP-0002-R17", "sparsevlm64": "EXP-0002/EXP-0002-R41",
        "vanilla": "EXP-0002/EXP-0002-R04"}
MME = {"A'+B+V": "EXP-0015/EXP-0015-R02", "A": "EXP-0014/EXP-0014-R04", "A'+B": "EXP-0014/EXP-0014-R06",
       "A+B": "EXP-0014/EXP-0014-R07", "visionzip64": "EXP-0002/EXP-0002-R35", "sparsevlm64": "EXP-0002/EXP-0002-R40",
       "vanilla": "EXP-0002/EXP-0002-R02"}
PAIRS = [("A'+B+V", "A"), ("A'+B+V", "A'+B"), ("A'+B+V", "visionzip64"), ("A'+B+V", "sparsevlm64"), ("A'+B+V", "vanilla")]


def share_check(run, question_file, tok, idf_fn, vis):
    import run_eval
    kept = json.load(open(f"{OUT}/{run}/kept_tokens.json"))["kept_by_object_name"]
    idf = run_eval.question_idf(tok, question_file, "vicuna_v1")
    same = 0
    texts = [json.loads(line)["text"] for line in open(question_file)]
    for k, text in zip(kept, texts):
        ids = run_eval.prompt_ids_after_image(tok, text, "vicuna_v1")
        w = np.array([idf["idf"].get(t, idf["unseen"]) for t in ids])
        s = float((w * vis[ids]).sum() / w.sum()) if w.sum() > 0 else 0.0
        same += len(k) == round(30 * s)
    return same / len(kept)


def main():
    out_dir = f"{OUT}/EXP-0015/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    vis = np.load(f"{OUT}/EXP-0015/visibility/visibility.npy")
    checks = {
        "R02 (MME) query-chosen counts equal round(30 S) (>= 99%)": share_check("EXP-0015/EXP-0015-R02", f"{EVAL_DATA}/MME/llava_mme.jsonl", tok, None, vis),
        "R03 (POPE) query-chosen counts equal round(30 S) (>= 99%)": share_check("EXP-0015/EXP-0015-R03", f"{EVAL_DATA}/pope/llava_pope_test.jsonl", tok, None, vis),
    }
    order = []
    for line in open(f"{EVAL_DATA}/pope/llava_pope_test.jsonl"):
        q = json.loads(line)
        order.append((q["question_id"], (q["category"], q["image"], OBJECT.match(q["text"]).group(1))))
    key_of = dict(order)
    original = read_labels(lambda s: f"{EVAL_DATA}/pope/coco/coco_pope_{s}.json")
    repope = {k: v for k, v in read_labels(lambda s: f"{REPOPE}/coco_repope_{s}.json").items() if k in original}
    said, pope = {}, {}
    for name, run in POPE.items():
        s = {key_of[json.loads(l)["question_id"]]: not pope_no(json.loads(l)["text"]) for l in open(f"{OUT}/{run}/answers.jsonl")}
        per_o, f1_o = score(s, original)
        _, f1_r = score(s, repope)
        checks[f"POPE {name} headline"] = 1.0 if abs(f1_o - float(row(run)["headline"])) < 1e-6 else 0.0
        said[name] = s
        pope[name] = {"f1": f1_o, "f1_repope": f1_r, "accuracy": 100 * float(np.mean([per_o[x]["accuracy"] for x in SPLITS])),
                      "recall": 100 * float(np.mean([per_o[x]["recall"] for x in SPLITS]))}
    pope_pairs = {f"{a} - {b}": {"original": paired_boot(said[a], said[b], original), "repope": paired_boot(said[a], said[b], repope)}
                  for a, b in PAIRS}
    mme, items = {}, {}
    for name, run in MME.items():
        it = load_mme(f"{OUT}/{run}")
        items[name] = it
        total = mme_stat(it, {k: np.arange(len(v)) for k, v in it.items()})
        checks[f"MME {name} headline"] = 1.0 if abs(total - float(row(run)["headline"])) < 1e-6 else 0.0
        mme[name] = {"perception": total, "per_task": {t: 100 * float(it[t].mean() + it[t].all(axis=1).mean()) for t in PERCEPTION}}
    mme_pairs = {f"{a} - {b}": mme_boot(items[a], items[b]) for a, b in PAIRS if b in items}
    if any(v < 0.99 for v in checks.values()):
        print("STOP:", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    m = mme_pairs["A'+B+V - A"]
    v1, v1p = m[1] > 0, m[0] > 0 and m[1] <= 0 <= m[2]
    v2 = all(pope_pairs["A'+B+V - A"][lab][1] > 0 for lab in ("original", "repope"))
    summary = {"checks": checks, "pope": pope, "pope_pairs": pope_pairs, "mme": mme, "mme_pairs": mme_pairs,
               "V1": bool(v1), "V1_prime": bool(v1p), "V2": bool(v2)}
    os.makedirs(out_dir)
    json.dump(summary, open(f"{out_dir}/vis.json", "w"), indent=2)
    L = ["# EXP-0015（计划 v1）：IDF × 可视性，名额按可见占比分配（A′+B+V）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    L += ["", "## POPE", "", "| 配置 | F1 原始 | F1 RePOPE | 准确率 | 召回 |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {n} | {r['f1']:.2f} | {r['f1_repope']:.2f} | {r['accuracy']:.2f} | {r['recall']:.1f}% |" for n, r in pope.items()]
    L += ["", "| 配对差 | 原始标注 | RePOPE |", "| --- | --- | --- |"]
    for k, v in pope_pairs.items():
        o, r = v["original"], v["repope"]
        L.append(f"| {k} | {o[0]:+.2f} [{o[1]:+.2f}, {o[2]:+.2f}] | {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] |")
    L += ["", "## MME", "", "| 配置 | MME-P | " + " | ".join(PERCEPTION) + " |", "| --- | --- | " + " | ".join("---" for _ in PERCEPTION) + " |"]
    L += [f"| {n} | {r['perception']:.1f} | " + " | ".join(f"{r['per_task'][t]:.1f}" for t in PERCEPTION) + " |" for n, r in mme.items()]
    L += ["", "| 配对差（MME-P） | 差 [95% 区间] |", "| --- | --- |"]
    L += [f"| {k} | {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] |" for k, v in mme_pairs.items()]
    L += ["", f"**判据**：V1（MME 上区间在 0 以上）{'成立' if v1 else '不成立'}；V1′（点估计 > 0、区间含 0）{'成立' if v1p else '不成立'}；"
              f"V2（POPE 上区间在 0 以上，两种标注）{'成立' if v2 else '不成立'}。"]
    open(f"{out_dir}/vis.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
