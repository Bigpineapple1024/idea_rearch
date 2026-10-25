"""EXP-0023 (plan v1): where do the answers of open questions lie, relative to the object the question names?
A CPU diagnosis for two candidate modules (topic-anchored neighbourhood; zoom within the budget). No model calls.

  python diag_0023.py      # writes code/output/EXP-0023/analysis/{diag_0023.md,diag_0023.json}

Object patches come from COCO boxes on LLaVA's padded square, 24x24 grid, a patch counts when a box covers >= 25% of it
(analyze_object_boxes.object_record). The named ("topic") object uses squeeze_0018's word matching. Rings are Chebyshev
distance 1..r from the topic patches. Data: VQAv2 test1 (with EXP-0018's runs) and dev (labels only); test2 and GQA unused.
Paired differences and accuracies resample images (1000 draws, seed 0).
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

TP = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, TP)
from analyze_object_boxes import GRID, object_record   # noqa: E402
from squeeze_0018 import COCO_INSTANCES, EVAL_DATA, OUT, plural_forms   # noqa: E402

N_BOOT = 1000
RUNS = {"A_64": "R02", "AV_64": "R03", "vz64": "R05", "vanilla": "R09"}      # EXP-0018 on test1


def dilate(mask, r):
    m = mask.reshape(GRID, GRID).copy()
    for _ in range(r):
        p = np.pad(m, 1)
        m = np.zeros_like(m)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                m |= p[dy:dy + GRID, dx:dx + GRID]
    return m.ravel()


def majority(answers):
    return Counter(answers).most_common(1)[0][0]


def load_split(name):
    gts = {a["question_id"]: a for a in map(json.loads, open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_{name}_annotations.jsonl"))}
    qtext = {q["question_id"]: q["text"].split("\n")[0] for q in map(json.loads, open(f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_{name}.jsonl"))}
    return gts, qtext


def main():
    out_dir = f"{OUT}/EXP-0023/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    coco = json.load(open(COCO_INSTANCES, encoding="utf-8"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    ims = {im["file_name"]: im for im in coco["images"]}
    inst = defaultdict(list)
    for a in coco["annotations"]:
        inst[(a["image_id"], names[a["category_id"]])].append(a)
    pats = {n: re.compile(r"\b(" + "|".join(re.escape(f) for f in sorted(plural_forms(n), key=len, reverse=True)) + r")\b")
            for n in names.values()}

    def records(split):
        gts, qtext = load_split(split)
        rows, skipped = [], Counter()
        for q, g in gts.items():
            if g["answer_type"] != "other":
                continue
            im = ims[g["image"]]
            topic = sorted(n for n, p in pats.items() if p.search(qtext[q].lower()) and inst.get((im["id"], n)))
            if not topic:
                continue
            tmask = np.zeros(GRID * GRID, bool)
            tmask[object_record(im, [a for n in topic for a in inst[(im["id"], n)]])["object_patches"]] = True
            if not tmask.any():
                skipped["topic covers no patch"] += 1
                continue
            maj = majority(g["answers"]).lower()
            ans_cat = sorted(n for n, p in pats.items() if p.search(maj) and n not in topic and inst.get((im["id"], n)))
            r = {"qid": q, "image": g["image"], "question": qtext[q], "majority": maj, "question_type": g["question_type"],
                 "topic": topic, "topic_mask": tmask, "n_topic": int(tmask.sum()), "answer_cat": ans_cat}
            if ans_cat:
                amask = np.zeros(GRID * GRID, bool)
                amask[object_record(im, [a for n in ans_cat for a in inst[(im["id"], n)]])["object_patches"]] = True
                if amask.any():
                    r["answer_mask"] = amask
                else:
                    skipped["answer object covers no patch"] += 1
            rows.append(r)
        return rows, skipped, gts

    res, L = {}, ["# EXP-0023（计划 v1）：开放式题的答案在哪里（CPU 诊断，事后探索）", ""]

    # N1: where the answer object lies relative to the topic object (labels only)
    L += ["## N1：答案物体相对话题物体的位置（S_rel，只看标注）", "",
          "| 数据 | S_open 题数 | S_rel 题数 | 内部 | 邻近（2 格环） | 远处 | 环取 1 格时的邻近 | 环取 3 格时的邻近 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    examples = {}
    for split in ("test", "dev"):
        rows, skipped, _ = records(split)
        rel = [r for r in rows if "answer_mask" in r]
        cls, near1, near3 = Counter(), 0, 0
        ex = defaultdict(list)
        for r in rel:
            a, t = r["answer_mask"], r["topic_mask"]
            inside = (a & t).sum() / a.sum()
            c = "inside" if inside >= 0.5 else ("adjacent" if (a & (t | dilate(t, 2))).sum() / a.sum() >= 0.5 else "far")
            cls[c] += 1
            near1 += inside < 0.5 and (a & (t | dilate(t, 1))).sum() / a.sum() >= 0.5
            near3 += inside < 0.5 and (a & (t | dilate(t, 3))).sum() / a.sum() >= 0.5
            if len(ex[c]) < 8:
                ex[c].append(f"{r['question']} → {r['majority']}（话题 {'/'.join(r['topic'])}，答案物体 {'/'.join(r['answer_cat'])}）")
        n = len(rel)
        res[f"N1_{split}"] = {"n_open": len(rows), "n_rel": n, "inside": cls["inside"] / n, "adjacent": cls["adjacent"] / n,
                              "far": cls["far"] / n, "adjacent_r1": near1 / n, "adjacent_r3": near3 / n, "skipped": dict(skipped)}
        v = res[f"N1_{split}"]
        L.append(f"| {'test1' if split == 'test' else 'dev'} | {len(rows)} | {n} | {v['inside']:.2f} | {v['adjacent']:.2f} | {v['far']:.2f} | "
                 f"{v['adjacent_r1']:.2f} | {v['adjacent_r3']:.2f} |")
        examples[split] = ex
    L += ["", "test1 的例子（每类最多 8 题）：", ""]
    for c in ("inside", "adjacent", "far"):
        L += [f"- **{ {'inside': '内部', 'adjacent': '邻近', 'far': '远处'}[c] }**："] + [f"  - {e}" for e in examples["test"][c]]

    # N2 / N3: test1 runs
    rows, _, gts = records("test")
    scores, kept = {}, {}
    for name, rid in RUNS.items():
        d = f"{OUT}/EXP-0018/EXP-0018-{rid}"
        scores[name] = {json.loads(l)["question_id"]: json.loads(l)["score"] for l in open(f"{d}/vqa_scores.jsonl")}
        if os.path.exists(f"{d}/kept_tokens.json"):
            order = [json.loads(l)["question_id"] for l in open(f"{d}/answers.jsonl")]
            k = json.load(open(f"{d}/kept_tokens.json"))
            if k["n_prefills"] != len(order):
                sys.exit(f"{rid}: one prefill per question expected")
            kept[name] = {q: set(k["kept"][i]) for i, q in enumerate(order)}
    images = sorted({r["image"] for r in rows})
    img_of = np.array([images.index(r["image"]) for r in rows])
    by_img = [np.flatnonzero(img_of == k) for k in range(len(images))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(images), len(images))]) for _ in range(N_BOOT)]

    def ci(x, m):
        x = np.asarray(x, float)
        if m.sum() == 0:
            return None
        boot = [100 * x[idx][m[idx]].mean() for idx in draws if m[idx].any()]
        return [100 * float(x[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

    fmt = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]（{v[3]}）" if v else "—"   # noqa: E731
    fa = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]（{v[3]}）" if v else "—"   # noqa: E731
    cnt = {}
    for name in ("A_64", "AV_64"):
        for part in ("topic", "ring", "answer", "else"):
            cnt[(name, part)] = np.zeros(len(rows))
        for i, r in enumerate(rows):
            ks = np.zeros(GRID * GRID, bool)
            ks[list(kept[name][r["qid"]])] = True
            t, ring = r["topic_mask"], dilate(r["topic_mask"], 2) & ~r["topic_mask"]
            cnt[(name, "topic")][i] = (ks & t).sum()
            cnt[(name, "ring")][i] = (ks & ring).sum()
            cnt[(name, "else")][i] = (ks & ~t & ~ring).sum()
            cnt[(name, "answer")][i] = (ks & r["answer_mask"]).sum() if "answer_mask" in r else np.nan
    sc = {n: np.array([scores[n][r["qid"]] for r in rows]) for n in RUNS}
    diff = sc["AV_64"] - sc["A_64"]
    is_rel = np.array(["answer_mask" in r for r in rows])
    is_attr = np.array([r["question_type"].startswith("what color") for r in rows])
    L += ["", f"## N2：A 与 A′+B+V 的 30 个保留 token 落在哪里（test1，64 档；S_open {len(rows)} 题，S_rel {int(is_rel.sum())} 题）", "",
          "| 配置 | 话题物体内 | 2 格环内 | 其他位置 | 答案物体上（S_rel） |", "| --- | --- | --- | --- | --- |"]
    res["N2_counts"] = {}
    for name in ("A_64", "AV_64"):
        v = {p: float(np.nanmean(cnt[(name, p)][is_rel] if p == "answer" else cnt[(name, p)])) for p in ("topic", "ring", "else", "answer")}
        res["N2_counts"][name] = v
        L.append(f"| {name} | {v['topic']:.2f} | {v['ring']:.2f} | {v['else']:.2f} | {v['answer']:.2f} |")
    dring = cnt[("AV_64", "ring")] - cnt[("A_64", "ring")]
    dans = cnt[("AV_64", "answer")] - cnt[("A_64", "answer")]
    res["N2_by_dring"] = {lab: ci(diff, m) for lab, m in (("dring<0", dring < 0), ("dring=0", dring == 0), ("dring>0", dring > 0), ("dring>=0", dring >= 0))}
    res["N2_by_dans"] = {lab: ci(diff, is_rel & m) for lab, m in (("dans<0", dans < 0), ("dans=0", dans == 0), ("dans>0", dans > 0))}
    L += ["", "A′+B+V − A 的 VQA 分数差，按 Δ环 = A′+B+V 环内个数 − A 环内个数分组（S_open）：", ""]
    L += [f"- {k}：{fmt(v)}" for k, v in res["N2_by_dring"].items()]
    L += ["", "按 Δ答案 = A′+B+V 落在答案物体上的个数 − A 的个数分组（S_rel）：", ""]
    L += [f"- {k}：{fmt(v)}" for k, v in res["N2_by_dans"].items()]

    # N3: dose on the topic object (accuracy by the number of kept tokens on it)
    L += ["", "## N3：按话题物体上保留的 token 数分组的准确率（test1，64 档）", "",
          "| 子集 | 配置 | 0 个 | 1～3 个 | ≥4 个 | 全部 |", "| --- | --- | --- | --- | --- | --- |"]
    res["N3"] = {}
    for sub, sm in (("S_attr（颜色题）", is_attr), ("S_open", np.ones(len(rows), bool))):
        for name in ("A_64", "AV_64"):
            c = cnt[(name, "topic")]
            g = {lab: ci(sc[name], sm & m) for lab, m in (("0", c == 0), ("1-3", (c >= 1) & (c <= 3)), (">=4", c >= 4), ("all", np.ones(len(rows), bool)))}
            res["N3"][f"{sub} | {name}"] = g
            L.append(f"| {sub} | {name} | {fa(g['0'])} | {fa(g['1-3'])} | {fa(g['>=4'])} | {fa(g['all'])} |")
        for name in ("vanilla", "vz64"):
            res["N3"][f"{sub} | {name}"] = {"all": ci(sc[name], sm)}
        L.append(f"| {sub} | vanilla / VisionZip-64 | — | — | — | {fa(res['N3'][f'{sub} | vanilla']['all'])} / {fa(res['N3'][f'{sub} | vz64']['all'])} |")
    res["N3_A_minus_vanilla_attr"] = ci(sc["A_64"] - sc["vanilla"], is_attr)
    L += ["", f"- 颜色题上 A-64 − vanilla：{fmt(res['N3_A_minus_vanilla_attr'])}"]
    # supplement (v1.1, written before running): the same dose split within thirds of the topic object's size
    sizes = np.array([r["n_topic"] for r in rows])
    cuts = np.percentile(sizes[is_attr], [100 / 3, 200 / 3]) if is_attr.any() else [0, 0]
    L += ["", "补充（计划 v1.1，运行前加入）：颜色题按话题物体大小三等分后，A-64 在各剂量组的准确率", "",
          "| 物体大小 | 0 个 | 1～3 个 | ≥4 个 |", "| --- | --- | --- | --- |"]
    res["N3_by_size"] = {}
    for lab, m in (("小", sizes <= cuts[0]), ("中", (sizes > cuts[0]) & (sizes <= cuts[1])), ("大", sizes > cuts[1])):
        c = cnt[("A_64", "topic")]
        g = {d: ci(sc["A_64"], is_attr & m & dm) for d, dm in (("0", c == 0), ("1-3", (c >= 1) & (c <= 3)), (">=4", c >= 4))}
        res["N3_by_size"][lab] = g
        L.append(f"| {lab} | {fa(g['0'])} | {fa(g['1-3'])} | {fa(g['>=4'])} |")

    # decision rules of the plan
    n1t, n1d = res["N1_test"], res["N1_dev"]
    neigh_1 = all(v["adjacent"] >= 0.40 and v["adjacent"] > v["inside"] for v in (n1t, n1d))
    lo, hi = res["N2_by_dring"]["dring<0"], res["N2_by_dring"]["dring>=0"]
    neigh_2 = bool(lo and hi and lo[0] < hi[0])
    far = any(v["far"] >= 0.50 for v in (n1t, n1d))
    a3 = res["N3"]["S_attr（颜色题） | A_64"]
    dose = all(a3[x] and a3[y] and a3[x][0] < a3[y][0] for x, y in (("0", "1-3"), ("1-3", ">=4")))
    gap = res["N3_A_minus_vanilla_attr"][0] <= -5
    res["decision"] = {"neighbourhood_N1": neigh_1, "neighbourhood_N2": neigh_2, "neighbourhood_promising": neigh_1 and neigh_2,
                       "neighbourhood_unlikely_far": far, "zoom_dose_monotone": dose, "zoom_gap_ge_5": gap, "zoom_promising": dose and gap}
    d = res["decision"]
    L += ["", "## 按计划写定的判读", "",
          f"- 邻域法：N1（两个数据上邻近 ≥40% 且高于内部）{'满足' if neigh_1 else '不满足'}；N2（Δ环 < 0 组低于 Δ环 ≥ 0 组）{'满足' if neigh_2 else '不满足'} → "
          f"{'有希望' if d['neighbourhood_promising'] else '不满足“有希望”'}；远处 ≥50%：{'是（不像）' if far else '否'}",
          f"- 放大法：颜色题剂量单调 {'是' if dose else '否'}；A 比 vanilla 低 ≥5 分 {'是' if gap else '否'} → {'有希望' if d['zoom_promising'] else '不满足“有希望”'}",
          "- 这些都是事后探索，据此设计的模块要在开发集上调、在新数据上预注册确认。"]
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/diag_0023.json", "w"), indent=2, ensure_ascii=False)
    open(f"{out_dir}/diag_0023.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
