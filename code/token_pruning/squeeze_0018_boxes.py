"""EXP-0018 squeeze, part 2 (exploratory): do the question-chosen tokens land on the object the question names?

  python squeeze_0018_boxes.py      # writes code/output/EXP-0018/squeeze_boxes/{boxes.md,boxes.json}

For every VQAv2 test question that names a COCO category annotated in its image (same matching as squeeze_0018.py), the
object's patches come from its COCO boxes (analyze_object_boxes.object_record: patches covered >= 25%, on LLaVA's padded
square). Compared per question: the share of the question-chosen tokens (kept_by_object_name, n_q of them) inside the
object, the number of 6x6 blocks the object touches, and the paired score difference A'+B+V - A (64 and 128 budgets).
Hypothesis under test (H-forced): the block rotation makes the query take one token in each of n_q blocks, so when the
object touches fewer blocks than n_q, the extra picks fall outside it and displace FastV's choices in those blocks.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import GRID, object_record   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squeeze_0018 import COCO_INSTANCES, EVAL_DATA, OUT, plural_forms   # noqa: E402

BLOCKS = 6
N_BOOT = 1000


def block_of(p):
    return (p // GRID) * BLOCKS // GRID * BLOCKS + (p % GRID) * BLOCKS // GRID


def main():
    out_dir = f"{OUT}/EXP-0018/squeeze_boxes"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    gts = {a["question_id"]: a for a in map(json.loads, open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test_annotations.jsonl"))}
    qtext = {q["question_id"]: q["text"].split("\n")[0] for q in map(json.loads, open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test.jsonl"))}
    coco = json.load(open(COCO_INSTANCES, encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    ims = {im["file_name"]: im for im in coco["images"]}
    inst = defaultdict(list)
    for a in coco["annotations"]:
        inst[(a["image_id"], names[a["category_id"]])].append(a)
    pats = {n: re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(plural_forms(n), key=len, reverse=True)) + r")\b")
            for n in names.values()}

    def run(rid):
        d = f"{OUT}/EXP-0018/EXP-0018-{rid}"
        order = [json.loads(l)["question_id"] for l in open(f"{d}/answers.jsonl")]
        k = json.load(open(f"{d}/kept_tokens.json"))
        sc = {json.loads(l)["question_id"]: json.loads(l)["score"] for l in open(f"{d}/vqa_scores.jsonl")}
        return {q: (set(k["kept"][i]), set(k.get("kept_by_object_name", [[]] * len(order))[i])) for i, q in enumerate(order)}, sc

    runs = {n: run(r) for n, r in (("A_64", "R02"), ("AV_64", "R03"), ("A_128", "R07"), ("AV_128", "R08"))}
    rows = []
    for q, g in gts.items():
        im = ims[g["image"]]
        hit = [n for n, p in pats.items() if p.search(qtext[q].lower()) and inst.get((im["id"], n))]
        if not hit:
            continue
        anns = [a for n in hit for a in inst[(im["id"], n)]]
        rec = object_record(im, anns)
        obj = set(rec["object_patches"])
        r = {"qid": q, "image": g["image"], "answer_type": g["answer_type"], "question_type": g["question_type"],
             "n_object_patches": rec["n_object_patches"], "area_frac": rec["area_frac"],
             "n_object_blocks": len({block_of(p) for p in obj})}
        for keep, a, av in ((30, "A_64", "AV_64"), (98, "A_128", "AV_128")):
            ka, _ = runs[a][0][q]
            kv, kq = runs[av][0][q]
            r[f"nq_{keep}"] = len(kq)
            r[f"q_in_{keep}"] = len(kq & obj) / len(kq) if kq else float("nan")
            r[f"A_in_{keep}"] = len(ka & obj) / keep
            r[f"AV_in_{keep}"] = len(kv & obj) / keep
            r[f"diff_{keep}"] = runs[av][1][q] - runs[a][1][q]
        rows.append(r)

    images = sorted({r["image"] for r in rows})
    img_of = np.array([images.index(r["image"]) for r in rows])
    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]

    def ci(x, m):
        x = np.asarray(x, float)
        boot = [100 * x[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return [100 * float(x[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

    res = {"n_questions": len(rows), "n_images": len(images), "groups": {}}
    at = np.array([r["answer_type"] for r in rows])
    for keep in (30, 98):
        nq = np.array([r[f"nq_{keep}"] for r in rows])
        nb = np.array([r["n_object_blocks"] for r in rows])
        diff = np.array([r[f"diff_{keep}"] for r in rows])
        qin = np.array([r[f"q_in_{keep}"] for r in rows])
        res[f"mean_{keep}"] = {t: {"q_in": float(np.nanmean(qin[at == t])),
                                   "A_in": float(np.mean([r[f"A_in_{keep}"] for r, x in zip(rows, at) if x == t])),
                                   "AV_in": float(np.mean([r[f"AV_in_{keep}"] for r, x in zip(rows, at) if x == t])),
                                   "nq": float(nq[at == t].mean()), "object_blocks": float(nb[at == t].mean())}
                               for t in ("yes/no", "number", "other")}
        for t in ("yes/no", "number", "other"):
            for label, m in (("object touches >= n_q blocks", nb >= nq), ("object touches < n_q blocks", nb < nq),
                             ("query picks mostly inside (>= 0.5)", qin >= 0.5), ("query picks mostly outside (< 0.5)", qin < 0.5)):
                mm = (at == t) & m
                if mm.sum() >= 20:
                    res["groups"][f"{keep} | {t} | {label}"] = ci(diff, mm)
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/boxes.json", "w"), indent=2)
    L = [f"# EXP-0018 squeeze 第二部分：查询选的 token 是否落在题目点名的物体上（{len(rows)} 题，{len(images)} 张图；探索性）", ""]
    for keep in (30, 98):
        L += [f"## keep {keep}", "", "| 答案类型 | 平均 n_q | 物体覆盖的块数 | 查询选的落在物体内 | A 保留落在物体内 | A′+B+V 保留落在物体内 |",
              "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {t} | {v['nq']:.1f} | {v['object_blocks']:.1f} | {v['q_in']:.2f} | {v['A_in']:.2f} | {v['AV_in']:.2f} |"
              for t, v in res[f"mean_{keep}"].items()]
        L += ["", "| 分组 | 题数 | A′+B+V − A [95%] |", "| --- | --- | --- |"]
        L += [f"| {k.split(' | ', 1)[1]} | {v[3]} | {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}] |" for k, v in res["groups"].items()
              if k.startswith(f"{keep} |")]
        L.append("")
    open(f"{out_dir}/boxes.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
