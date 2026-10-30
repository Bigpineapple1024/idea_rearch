"""SQZ-EXP-0034 (written 2026-10-31 around 03:30, before running it; EXP-0034's sealed analysis had been read, EXP-0035
was running): what seeing the dropped patches at layer 0 (peek - nopeek) and the original position ids (nopeek -
DivPrune-64) do to yes/no questions.

  python -B squeeze_0034.py      # writes code/output/EXP-0034/squeeze/{squeeze_0034.md,squeeze_0034.json}

Runs (all completed and sealed): EXP-0034-R01..R06 (peek, nopeek, peektext on VQAv2 dev2 and GQA val400),
EXP-0032-R01, R02 (DivPrune-64), R03, R04 (K = 1 keep 47), EXP-0033-R03, R04 (unpruned).

S1 yes/no direction. Questions whose ground truth is yes or no (GQA: the answer; VQAv2: answer type 'yes/no' and the
   most common of the ten answers is yes or no). Per config: the share answered yes and the accuracy on gt-yes and on
   gt-no questions; paired peek - nopeek, nopeek - div64, peek - div64 within gt-yes and within gt-no.
S2 first-token log-odds (peek, nopeek, peektext have answer_logits.json): log(P(Yes) + P(yes)) - log(P(No) + P(no))
   of the first answer token; the mean by gt; paired peek - nopeek and peektext - nopeek.
S3 COCO objects (VQAv2 dev2 only; instances_val2014). Yes/no questions naming exactly one COCO category (its name or
   the name + s / es; 'people' for person; 'orange' left out, it is also a colour). Present (the category has an
   instance in the image) or absent. Present: the object's patches as in analyze_object_boxes.object_record; the share
   of them among the 61 kept (the same 61 in peek, nopeek and peektext), split at the median -> peek - nopeek; the
   mean patch row of the object's patches in thirds of the 24 rows (top, middle, bottom) -> nopeek - div64 and
   peek - nopeek. Absent: peek - nopeek and nopeek - div64.
S4 by detailed type (GQA 'detailed', VQAv2 'question_type'; groups with at least 60 questions): the three pairs.
S5 per-question flips (questions whose scores differ, better, worse): peek vs nopeek, nopeek vs div64, peek vs div64.
Bootstrap: resample images with replacement within each set, 1000 draws, seed 0 (as analyze_0034). Exploratory, many
groups, no multiple-comparison correction; nothing here changes EXP-0034's readings.
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

CODE = "D:/6CJNM007/idea_rearch/code/token_pruning"
sys.path.insert(0, CODE)
import analyze_object_boxes as ob  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
EV = "D:/6CJNM007/data/llava_eval"
DST = f"{OUT}/EXP-0034/squeeze"
BENCH = {"vqav2_dev2": (f"{EV}/vqav2_val/llava_vqav2_val_dev2.jsonl", f"{EV}/vqav2_val/llava_vqav2_val_dev2_annotations.jsonl", "vqa_scores.jsonl"),
         "gqa_val400": (f"{EV}/gqa/llava_gqa_val400.jsonl", f"{EV}/gqa/llava_gqa_val400_annotations.jsonl", "gqa_scores.jsonl")}
RUNS = {"peek": ("EXP-0034", {"vqav2_dev2": "R01", "gqa_val400": "R02"}, "fastv61k1_div_peek_kll"),
        "nopeek": ("EXP-0034", {"vqav2_dev2": "R03", "gqa_val400": "R04"}, "fastv61k1_div_nopeek_kll"),
        "peektext": ("EXP-0034", {"vqav2_dev2": "R05", "gqa_val400": "R06"}, "fastv61k1_div_peektext_kll"),
        "div64": ("EXP-0032", {"vqav2_dev2": "R01", "gqa_val400": "R02"}, "divprune64"),
        "k1div": ("EXP-0032", {"vqav2_dev2": "R03", "gqa_val400": "R04"}, "fastv64k1_div_kl"),
        "vanilla": ("EXP-0033", {"vqav2_dev2": "R03", "gqa_val400": "R04"}, "vanilla")}
PAIRS = [("peek", "nopeek"), ("nopeek", "div64"), ("peek", "div64")]
N_BOOT, MIN_GROUP, KEEP = 1000, 60, 61
CHECKS = {}


def run_dir(meth, bench):
    exp, ids, _ = RUNS[meth]
    return f"{OUT}/{exp}/{exp}-{ids[bench]}"


def check_runs():
    for meth, (exp, ids, cfg) in RUNS.items():
        rows = {r["run_id"]: r for r in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")}
        for bench, rid in ids.items():
            r = rows[f"{exp}-{rid}"]
            CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench}"] = float(r["status"] == "completed" and r["config"] == cfg and r["bench"] == bench)


def yes_no(text):
    t = text.strip().lower().lstrip("\"' ")
    w = re.split(r"[^a-z]+", t, maxsplit=1)[0] if t else ""
    return "yes" if w == "yes" else "no" if w == "no" else None


def load(bench):
    qf, af, sf = BENCH[bench]
    qs = [json.loads(l) for l in open(qf, encoding="utf-8")]
    ann = {str(a["question_id"]): a for a in (json.loads(l) for l in open(af, encoding="utf-8"))}
    order = [str(q["question_id"]) for q in qs]
    text = {str(q["question_id"]): q["text"].split("\n")[0] for q in qs}
    image = {str(q["question_id"]): q["image"] for q in qs}
    ims = sorted(set(image.values()))
    idx = {im: k for k, im in enumerate(ims)}
    img_of = np.array([idx[image[q]] for q in order])
    if bench == "gqa_val400":
        gt = [ann[q]["answer"] if ann[q]["answer"] in ("yes", "no") else None for q in order]
        dtype = [ann[q]["detailed"] for q in order]
    else:
        gt = []
        for q in order:
            top = Counter(ann[q]["answers"]).most_common(1)[0][0] if ann[q]["answer_type"] == "yes/no" else None
            gt.append(top if top in ("yes", "no") else None)
        dtype = [ann[q]["question_type"] for q in order]
    sc, said = {}, {}
    for meth in RUNS:
        d = run_dir(meth, bench)
        s = {}
        for l in open(f"{d}/{sf}", encoding="utf-8"):
            x = json.loads(l)
            s[str(x["question_id"])] = float(x["score"])
        a = {}
        for l in open(f"{d}/answers.jsonl", encoding="utf-8"):
            x = json.loads(l)
            a[str(x["question_id"])] = x["text"]
        CHECKS[f"{bench} {meth}: one score and one answer per question"] = float(sorted(s) == sorted(order) == sorted(a))
        sc[meth] = np.array([s[q] for q in order])
        said[meth] = [yes_no(a[q]) for q in order]
    by_img = [np.flatnonzero(img_of == k) for k in range(len(ims))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]
    return {"order": order, "text": text, "image": image, "gt": np.array(gt, dtype=object), "dtype": dtype, "sc": sc,
            "said": said, "draws": draws, "n": len(order)}


def paired(b, x, y, mask=None, values=None):
    """Mean difference x - y (in points for scores, raw for values) over mask, with the 95% bootstrap interval."""
    if values is None:
        d, k = b["sc"][x] - b["sc"][y], 100.0
    else:
        d, k = values[x] - values[y], 1.0
    m = np.ones(b["n"], bool) if mask is None else mask
    if m.sum() == 0:
        return None
    boot = [k * d[i][m[i]].mean() for i in b["draws"] if m[i].any()]
    return [k * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]


def fmt(x, digits=2):
    return "—" if x is None else f"{x[0]:+.{digits}f} [{x[1]:+.{digits}f}, {x[2]:+.{digits}f}]（{x[3]}）"


def s1(b):
    out = {}
    for g in ("yes", "no"):
        m = b["gt"] == g
        out[f"gt {g}"] = {"n": int(m.sum()),
                          "said_yes": {meth: float(np.mean([s == "yes" for s, mm in zip(b["said"][meth], m) if mm])) for meth in RUNS},
                          "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS},
                          "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in PAIRS}}
    return out


def s2(b, bench):
    lo = {}
    for meth in ("peek", "nopeek", "peektext"):
        x = json.load(open(f"{run_dir(meth, bench)}/answer_logits.json", encoding="utf-8"))
        CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == b["n"] == len(x["first_token"]))
        assert x["tokens"] == ["▁Yes", "▁No", "▁yes", "▁no"], x["tokens"]
        lp = np.array([r["logprobs"] for r in x["first_token"]])
        lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
        top = [r["argmax"] for r in x["first_token"]]
        yn = {x["ids"][0]: "yes", x["ids"][2]: "yes", x["ids"][1]: "no", x["ids"][3]: "no"}
        agree = [yn[t] == s for t, s in zip(top, b["said"][meth]) if t in yn]
        CHECKS[f"{bench} {meth}: answers agree with the logged first-token argmax (alignment, >= 99%)"] = float(np.mean(agree) >= 0.99)
    out = {}
    for g in ("yes", "no"):
        m = b["gt"] == g
        out[f"gt {g}"] = {"mean_log_odds": {meth: float(v[m].mean()) for meth, v in lo.items()},
                          "peek - nopeek": paired(b, "peek", "nopeek", m, lo), "peektext - nopeek": paired(b, "peektext", "nopeek", m, lo)}
    return out


def s3(b):
    coco = json.load(open(ob.COCO, encoding="utf-8"))
    images = {im["file_name"]: im for im in coco["images"]}
    cats = {c["name"]: c["id"] for c in coco["categories"] if c["name"] != "orange"}
    inst = defaultdict(list)
    for a in coco["annotations"]:
        inst[(a["image_id"], a["category_id"])].append(a)
    pats = {n: re.compile(r"\b" + re.escape(n) + r"(?:s|es)?\b") for n in cats}
    pats["person"] = re.compile(r"\b(?:person|persons|people)\b")
    kept = json.load(open(f"{run_dir('peek', 'vqav2_dev2')}/kept_tokens.json", encoding="utf-8"))["kept"]
    CHECKS["vqav2_dev2 peek: one kept list of 61 per question"] = float(len(kept) == b["n"] and all(len(set(k)) == KEEP for k in kept))
    rec = []
    for i, q in enumerate(b["order"]):
        if b["gt"][i] is None:
            continue
        hit = [n for n, p in pats.items() if p.search(b["text"][q].lower())]
        if len(hit) != 1:
            continue
        im = images[b["image"][q]]
        anns = inst.get((im["id"], cats[hit[0]]), [])
        if not anns:
            rec.append({"i": i, "present": False})
            continue
        o = ob.object_record(im, anns)
        p = np.array(o["object_patches"])
        rec.append({"i": i, "present": True, "kept_share": float(np.isin(p, kept[i]).mean()), "row": float(np.mean(p // ob.GRID)),
                    "n_patches": int(len(p))})
    pres = [r for r in rec if r["present"]]
    med = float(np.median([r["kept_share"] for r in pres]))

    def mask(rs):
        m = np.zeros(b["n"], bool)
        m[[r["i"] for r in rs]] = True
        return m
    groups = {"present": pres, "absent": [r for r in rec if not r["present"]],
              f"present, kept share < median ({med:.3f})": [r for r in pres if r["kept_share"] < med],
              f"present, kept share >= median ({med:.3f})": [r for r in pres if r["kept_share"] >= med],
              "present, top third (rows 0-7)": [r for r in pres if r["row"] < 8],
              "present, middle third (rows 8-15)": [r for r in pres if 8 <= r["row"] < 16],
              "present, bottom third (rows 16-23)": [r for r in pres if r["row"] >= 16]}
    out = {"n_named_one_category": len(rec), "n_present": len(pres), "median_kept_share": med,
           "mean_kept_share": float(np.mean([r["kept_share"] for r in pres])), "groups": {}}
    for g, rs in groups.items():
        m = mask(rs)
        out["groups"][g] = {"n": int(m.sum()), "gt_yes_share": float(np.mean(b["gt"][m] == "yes")) if m.any() else None,
                            "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in RUNS} if m.any() else {},
                            "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in PAIRS}}
    return out


def s4(b):
    out = {}
    for t, c in sorted(Counter(b["dtype"]).items(), key=lambda kv: -kv[1]):
        if c < MIN_GROUP:
            continue
        m = np.array([d == t for d in b["dtype"]])
        out[t] = {"n": c, "accuracy": {meth: 100 * float(b["sc"][meth][m].mean()) for meth in ("peek", "nopeek", "div64", "vanilla")},
                  "pairs": {f"{x} - {y}": paired(b, x, y, m) for x, y in PAIRS}}
    return out


def s5(b):
    out = {}
    for x, y in PAIRS:
        d = b["sc"][x] - b["sc"][y]
        out[f"{x} - {y}"] = {"changed": int((d != 0).sum()), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
    return out


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    check_runs()
    sealed = json.load(open(f"{OUT}/EXP-0034/analysis/0034.json", encoding="utf-8"))["pairs"]
    res = {}
    for bench in BENCH:
        b = load(bench)
        for x, y in PAIRS:
            CHECKS[f"{bench} {x} - {y}: the overall mean difference equals the sealed 0034.json"] = float(
                abs(100 * float((b["sc"][x] - b["sc"][y]).mean()) - sealed[f"{bench} {x} - {y}"][0]) < 1e-6)
        res[bench] = {"S1": s1(b), "S2": s2(b, bench), "S4": s4(b), "S5": s5(b)}
        if bench == "vqav2_dev2":
            res[bench]["S3"] = s3(b)
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/squeeze_0034.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = ["# SQZ-EXP-0034：第 0 层看被剪的 patch（peek − nopeek）与原始位置编号（nopeek − DivPrune-64）对是非题做了什么", "",
         "事后、探索性；分组多，没有多重比较校正；不改 EXP-0034 的读法。区间：每个集内按图像重抽 1000 次（种子 0）。", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    for bench in BENCH:
        r = res[bench]
        L += ["", f"## {bench}", "", "### S1 是非题的方向（gt = 标准答案）", "",
              "| gt | 题数 | 答“是”的比例：peek / nopeek / peektext / DivPrune-64 / K = 1 留 47 / 不剪枝 | 准确率：同上 |", "| --- | --- | --- | --- |"]
        for g, x in r["S1"].items():
            L.append(f"| {g} | {x['n']} | " + " / ".join(f"{x['said_yes'][m]:.3f}" for m in RUNS) + " | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in RUNS) + " |")
        L += ["", "| gt | peek − nopeek | nopeek − DivPrune-64 | peek − DivPrune-64 |", "| --- | --- | --- | --- |"]
        for g, x in r["S1"].items():
            L.append(f"| {g} | " + " | ".join(fmt(x["pairs"][f"{p} - {q}"]) for p, q in PAIRS) + " |")
        L += ["", "### S2 第一个答案 token 的对数几率 log(P(Yes)+P(yes)) − log(P(No)+P(no))", "",
              "| gt | 平均：peek / nopeek / peektext | peek − nopeek | peektext − nopeek |", "| --- | --- | --- | --- |"]
        for g, x in r["S2"].items():
            L.append(f"| {g} | " + " / ".join(f"{x['mean_log_odds'][m]:+.3f}" for m in ("peek", "nopeek", "peektext"))
                     + f" | {fmt(x['peek - nopeek'], 3)} | {fmt(x['peektext - nopeek'], 3)} |")
        if "S3" in r:
            s = r["S3"]
            L += ["", "### S3 题目里点名的 COCO 物体（是非题，只点名一类）", "",
                  f"点名一类的是非题 {s['n_named_one_category']} 道，其中物体在图中 {s['n_present']} 道；物体的 patch 有多少在留下的 61 个里："
                  f"平均 {s['mean_kept_share']:.3f}，中位数 {s['median_kept_share']:.3f}。", "",
                  "| 分组 | 题数 | gt 为“是”的比例 | 准确率 peek / nopeek / DivPrune-64 | peek − nopeek | nopeek − DivPrune-64 | peek − DivPrune-64 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
            for g, x in s["groups"].items():
                acc = " / ".join(f"{x['accuracy'][m]:.2f}" for m in ("peek", "nopeek", "div64")) if x["accuracy"] else "—"
                gy = f"{x['gt_yes_share']:.3f}" if x["gt_yes_share"] is not None else "—"
                L.append(f"| {g} | {x['n']} | {gy} | {acc} | " + " | ".join(fmt(x["pairs"][f"{p} - {q}"]) for p, q in PAIRS) + " |")
        L += ["", f"### S4 按细分题型（至少 {MIN_GROUP} 题）", "",
              "| 题型 | 题数 | 准确率 peek / nopeek / DivPrune-64 / 不剪枝 | peek − nopeek | nopeek − DivPrune-64 | peek − DivPrune-64 |", "| --- | --- | --- | --- | --- | --- |"]
        for t, x in r["S4"].items():
            L.append(f"| {t} | {x['n']} | " + " / ".join(f"{x['accuracy'][m]:.2f}" for m in ("peek", "nopeek", "div64", "vanilla")) + " | "
                     + " | ".join(fmt(x["pairs"][f"{p} - {q}"]) for p, q in PAIRS) + " |")
        L += ["", "### S5 逐题翻转（得分不同 / 变好 / 变差）", ""]
        L += [f"- {k}: {x['changed']} / {x['better']} / {x['worse']}" for k, x in r["S5"].items()]
    open(f"{DST}/squeeze_0034.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print("written", DST)


if __name__ == "__main__":
    main()
