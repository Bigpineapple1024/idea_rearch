"""EXP-0018 squeeze (exploratory, after the pre-registered analysis): where does A'+B+V lose to A on VQAv2 "other"?

  python squeeze_0018.py      # writes code/output/EXP-0018/squeeze/{squeeze_0018.md,squeeze_0018.json}

Reads only sealed run outputs (answers, vqa_scores, kept_tokens) and the subset annotations; no model calls.
Paired differences are resampled over images (1000 draws, seed 0), as in analyze_0018.py. Everything here is
post hoc on the test subset: a pattern found here must be re-tested on fresh data before it can be claimed.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO = "D:/6CJNM007/idea_rearch"
OUT = f"{REPO}/code/output"
EVAL_DATA = "D:/6CJNM007/data/llava_eval"
COCO_INSTANCES = "D:/6CJNM007/data/coco/annotations/instances_val2014.json"
RUNS = {"vanilla": "R09", "A_64": "R02", "AV_64": "R03", "AB_64": "R04", "vz64": "R05", "sv64": "R06",
        "A_128": "R07", "AV_128": "R08"}
N_BOOT = 1000
YESNO_START = ("is ", "are ", "does ", "do ", "did ", "can ", "could ", "was ", "were ", "has ", "have ", "will ",
               "would ", "should ", "shall ", "may ", "might ", "must ", "had ")


def load(rid, name):
    return [json.loads(l) for l in open(f"{OUT}/EXP-0018/EXP-0018-{rid}/{name}")]


def plural_forms(name):
    forms = {name}
    if name.endswith(("s", "sh", "ch", "x")):
        forms.add(name + "es")
    elif name.endswith("y") and name[-2] not in "aeiou":
        forms.add(name[:-1] + "ies")
    else:
        forms.add(name + "s")
    extra = {"person": ["people", "man", "men", "woman", "women", "boy", "girl", "child", "kid", "kids", "player", "guy", "lady"],
             "mouse": ["mice"], "knife": ["knives"], "sheep": ["sheep"], "skis": ["ski"], "dining table": ["table", "tables"],
             "tv": ["television", "tvs"], "couch": ["sofa"], "motorcycle": ["motorbike"], "airplane": ["plane", "planes", "jet"],
             "cell phone": ["phone", "phones", "cellphone"], "laptop": ["computer"], "bicycle": ["bike", "bikes"],
             "potted plant": ["plant", "plants"], "hot dog": ["hotdog", "hot dogs"], "donut": ["doughnut", "donuts"]}
    forms.update(extra.get(name, []))
    return forms


def main():
    out_dir = f"{OUT}/EXP-0018/squeeze"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    gts = {a["question_id"]: a for a in map(json.loads, open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test_annotations.jsonl"))}
    qtext = {q["question_id"]: q["text"].split("\n")[0] for q in map(json.loads, open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test.jsonl"))}
    qids = sorted(gts)
    images = sorted({g["image"] for g in gts.values()})
    img_of = np.array([images.index(gts[q]["image"]) for q in qids])
    atype = np.array([gts[q]["answer_type"] for q in qids])
    qtype = np.array([gts[q]["question_type"] for q in qids])
    scores, answers, kept, kept_q = {}, {}, {}, {}
    for name, rid in RUNS.items():
        sc = {r["question_id"]: r["score"] for r in load(rid, "vqa_scores.jsonl")}
        an = {r["question_id"]: r["text"] for r in load(rid, "answers.jsonl")}
        scores[name] = np.array([sc[q] for q in qids])
        answers[name] = [an[q] for q in qids]
        if name != "vanilla" and not name.startswith(("vz", "sv")):
            k = json.load(open(f"{OUT}/EXP-0018/EXP-0018-{rid}/kept_tokens.json"))
            order = [r["question_id"] for r in load(rid, "answers.jsonl")]
            pos = {q: i for i, q in enumerate(order)}
            kept[name] = [set(k["kept"][pos[q]]) for q in qids]
            if "kept_by_object_name" in k:
                kept_q[name] = [set(k["kept_by_object_name"][pos[q]]) for q in qids]

    # the kept-token order in kept_tokens.json must follow answers.jsonl (one prefill per question)
    checks = {}
    for name in kept:
        n = json.load(open(f"{OUT}/EXP-0018/EXP-0018-{RUNS[name]}/kept_tokens.json"))["n_prefills"]
        checks[f"{name}: one prefill per question"] = float(n == len(qids))

    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]

    def paired(a, b, mask=None):
        d = scores[a] - scores[b]
        m = np.ones(len(d), bool) if mask is None else mask
        if m.sum() == 0:
            return None
        boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return {"n": int(m.sum()), "diff": 100 * float(d[m].mean()), "lo": float(np.percentile(boot, 2.5)),
                "hi": float(np.percentile(boot, 97.5))}

    res = {"checks": checks}

    # 1. query share: n_q = |kept_by_object_name|; with n_q = 0 the selection must equal A's
    for keep, av, a in ((30, "AV_64", "A_64"), (98, "AV_128", "A_128")):
        nq = np.array([len(s) for s in kept_q[av]])
        same_set = np.array([kept[av][i] == kept[a][i] for i in range(len(qids))])
        same_ans = np.array([answers[av][i] == answers[a][i] for i in range(len(qids))])
        overlap = np.array([len(kept[av][i] & kept[a][i]) / keep for i in range(len(qids))])
        checks[f"{av}: n_q = 0 implies the same kept set as {a}"] = float(same_set[nq == 0].mean()) if (nq == 0).any() else 1.0
        bins = [(0, 0), (1, keep // 3), (keep // 3 + 1, 2 * keep // 3), (2 * keep // 3 + 1, keep)]
        rows = []
        for lo, hi in bins:
            m = (nq >= lo) & (nq <= hi)
            for t in ("all", "yes/no", "number", "other"):
                mm = m if t == "all" else m & (atype == t)
                r = paired(av, a, mm)
                if r:
                    rows.append({"n_q": f"{lo}-{hi}", "type": t, **r, "same_answer": float(same_ans[mm].mean()),
                                 "kept_overlap": float(overlap[mm].mean())})
        res[f"by_nq_{keep}"] = rows
        res[f"nq_share_{keep}"] = {t: float(nq[atype == t].mean() / keep) for t in ("yes/no", "number", "other")}
        res[f"same_answer_{keep}"] = float(same_ans.mean())

    # 2. question types with at least 40 questions, 64-token budget
    qt_rows = []
    for t in sorted(set(qtype)):
        m = qtype == t
        if m.sum() >= 40:
            r = paired("AV_64", "A_64", m)
            rz = paired("vz64", "A_64", m)
            qt_rows.append({"question_type": t, "answer_type": max(set(atype[m]), key=list(atype[m]).count), **r,
                            "vz_minus_A": rz["diff"], "A_acc": 100 * float(scores["A_64"][m].mean()),
                            "vanilla_acc": 100 * float(scores["vanilla"][m].mean())})
    res["by_question_type_64"] = sorted(qt_rows, key=lambda r: r["diff"])

    # 3. does the question name a COCO category, and is that category annotated in the image?
    cats = json.load(open(COCO_INSTANCES, encoding="utf-8"))
    names = {c["id"]: c["name"] for c in cats["categories"]}
    img_id = {f"COCO_val2014_{i['id']:012d}.jpg": i["id"] for i in cats["images"]}
    present = defaultdict(set)
    for a in cats["annotations"]:
        present[a["image_id"]].add(names[a["category_id"]])
    pats = {n: re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(plural_forms(n), key=len, reverse=True)) + r")\b") for n in names.values()}
    mention = np.zeros(len(qids), bool)
    mention_present = np.zeros(len(qids), bool)
    for i, q in enumerate(qids):
        text = qtext[q].lower()
        hit = {n for n, p in pats.items() if p.search(text)}
        mention[i] = bool(hit)
        mention_present[i] = bool(hit & present[img_id[gts[q]["image"]]])
    res["mention"] = {}
    for t in ("all", "yes/no", "number", "other"):
        tm = np.ones(len(qids), bool) if t == "all" else atype == t
        for label, m in (("names a COCO category present in the image", tm & mention_present),
                         ("names a COCO category not annotated in the image", tm & mention & ~mention_present),
                         ("names no COCO category", tm & ~mention)):
            res["mention"][f"{t} | {label}"] = {"64": paired("AV_64", "A_64", m), "128": paired("AV_128", "A_128", m),
                                                "vz64_minus_A64": paired("vz64", "A_64", m)}

    # 4. a label-free router by the question's first word (exploratory: chosen after seeing the test result)
    yesno_form = np.array([qtext[q].lower().startswith(YESNO_START) for q in qids])
    howmany = np.array([qtext[q].lower().startswith("how many") for q in qids])
    res["router_form_vs_answer_type"] = {"yes/no-form questions that are yes/no": float((atype[yesno_form] == "yes/no").mean()),
                                         "yes/no questions in yes/no form": float(yesno_form[atype == "yes/no"].mean()),
                                         "how-many questions that are number": float((atype[howmany] == "number").mean())}
    for keep, av, a in ((64, "AV_64", "A_64"), (128, "AV_128", "A_128")):
        for label, route in (("yes/no form -> query", yesno_form), ("yes/no form or how many -> query", yesno_form | howmany)):
            name = f"router_{keep}"
            scores[name] = np.where(route, scores[av], scores[a])
            r = paired(name, a)
            res[f"router {keep}: {label}"] = {**r, "accuracy": 100 * float(scores[name].mean()),
                                              "routed_share": float(route.mean())}

    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/squeeze_0018.json", "w"), indent=2)
    L = ["# EXP-0018 squeeze（探索性，事后分析，测试集上看到的模式需在新数据上复验）", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.1f}%" for k, v in checks.items()]
    for keep in (30, 98):
        L += ["", f"## 1. 按查询名额 n_q 分（keep {keep}）", "",
              f"- 平均查询占比：" + "，".join(f"{t} {v:.2f}" for t, v in res[f'nq_share_{keep}'].items()),
              f"- 与 A 答案相同的比例：{res[f'same_answer_{keep}']:.3f}", "",
              "| n_q | 类型 | 题数 | A′+B+V − A [95%] | 答案相同 | 保留集重合 |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r['n_q']} | {r['type']} | {r['n']} | {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['same_answer']:.2f} | {r['kept_overlap']:.2f} |"
              for r in res[f"by_nq_{keep}"]]
    L += ["", "## 2. 按题型（≥40 题，64 档，按差排序）", "",
          "| 题型 | 答案类型 | 题数 | A′+B+V − A [95%] | VisionZip − A | A | vanilla |", "| --- | --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r['question_type']} | {r['answer_type']} | {r['n']} | {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['vz_minus_A']:+.2f} | {r['A_acc']:.1f} | {r['vanilla_acc']:.1f} |"
          for r in res["by_question_type_64"]]
    L += ["", "## 3. 题目是否提到 COCO 类别（及该类别是否标注在图中）", "",
          "| 分组 | 题数 | 64：A′+B+V − A | 128：A′+B+V − A | VisionZip-64 − A-64 |", "| --- | --- | --- | --- | --- |"]
    for k, v in res["mention"].items():
        if v["64"]:
            L.append(f"| {k} | {v['64']['n']} | {v['64']['diff']:+.2f} [{v['64']['lo']:+.2f}, {v['64']['hi']:+.2f}] | "
                     f"{v['128']['diff']:+.2f} [{v['128']['lo']:+.2f}, {v['128']['hi']:+.2f}] | {v['vz64_minus_A64']['diff']:+.2f} |")
    L += ["", "## 4. 按题目首词路由（不看标注；事后在测试集上选出，只作探索）", ""]
    L += [f"- {k}: {v:.3f}" for k, v in res["router_form_vs_answer_type"].items()]
    L += [f"- {k}: 准确率 {v['accuracy']:.2f}，比 A {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]，路由到查询的比例 {v['routed_share']:.2f}"
          for k, v in res.items() if k.startswith("router ")]
    open(f"{out_dir}/squeeze_0018.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
