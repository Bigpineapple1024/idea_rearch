"""EXP-0025 (plan v1): IDEA-0004's budgeted zoom on the detail benchmarks at the 64 budget: TextVQA v0.5.1 val without
OCR tokens (the main test), with OCR tokens (LLaVA's protocol, a reference) and V* Bench.

  python analyze_0025.py      # writes code/output/EXP-0025/analysis/{0025.md,0025.json}

The variant EXP-0024 selected (K = 2, 16 lattice tokens of the re-encoded crop, unrouted questions only) is spliced from
the zoom-everything run on the unrouted questions and the K = 2 router's answers on the routed ones (each prefill is
independent). The controls, same K and pick, are spliced the same way: the image's own tokens at the same lattice
(g16full, no second encoding) and the centred box with the located side (z16gctr). Criteria written before any run
(plan v0 and v1): T1, TextVQA without OCR, selected - router, 95% interval above 0; T2, the same on V*; T3, selected -
each control on TextVQA without OCR, both intervals above 0; "large" needs a lower bound of at least +2.0 on TextVQA or
+5.0 on V*. Per-question scores are recomputed from answers.jsonl (LLaVA's m4c evaluator on the ten human answers for
TextVQA, score_pope_mme.vstar_letter against the label for V*) and checked against each run's own score file. Paired
differences resample images with replacement (1000 draws, seed 0) within each set; V* has one question per image.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from squeeze_0018 import EVAL_DATA, OUT   # noqa: E402

LLAVA = "D:/6CJNM007/idea_rearch/source/LLaVA"
DATA = "D:/6CJNM007/data"
BENCHES = ("textvqa_noocr", "vstar", "textvqa_ocr")
IMAGES = {"textvqa_noocr": f"{DATA}/textvqa/train_images", "textvqa_ocr": f"{DATA}/textvqa/train_images",
          "vstar": f"{DATA}/downloads/vstar_bench"}
# (bench, method) -> run ID of the plan's run table
RUNS = {("textvqa_noocr", "router"): "R01", ("textvqa_noocr", "z16g"): "R02", ("vstar", "router"): "R03",
        ("vstar", "z16g"): "R04", ("textvqa_ocr", "router"): "R05", ("textvqa_ocr", "z16g"): "R06",
        ("textvqa_noocr", "vanilla"): "R07", ("vstar", "vanilla"): "R08", ("textvqa_noocr", "g16full"): "R09",
        ("textvqa_noocr", "z16gctr"): "R10"}
ZOOM = ("z16g", "g16full", "z16gctr")          # run on every question with 16 inserted tokens, spliced with the router
CROP = ("z16g", "z16gctr")                     # these encode a crop: one more vision tower pass per zoomed prefill
KEEP, N_ZOOM, N_BOOT = 30, 16, 1000
LARGE = {"textvqa_noocr": 2.0, "vstar": 5.0}


def load_bench(bench):
    """Questions in file order with the fields the breakdowns need; ids as strings."""
    import run_eval
    if bench == "vstar":
        qfile, afile = f"{EVAL_DATA}/vstar/llava_vstar_test.jsonl", f"{EVAL_DATA}/vstar/llava_vstar_test_annotations.jsonl"
    else:
        qfile = f"{EVAL_DATA}/textvqa/llava_textvqa_val_{bench.split('_')[1]}_qid.jsonl"
        afile = f"{EVAL_DATA}/textvqa/llava_textvqa_val_annotations.jsonl"
    qs = [json.loads(l) for l in open(qfile, encoding="utf-8")]
    gts = {str(a["question_id"]): a for a in map(json.loads, open(afile, encoding="utf-8"))}
    order = [str(q["question_id"]) for q in qs]
    if sorted(order) != sorted(gts):
        sys.exit(f"{bench}: questions and annotations differ")
    image = {str(q["question_id"]): q["image"] for q in qs}
    b = {"order": order, "gts": gts, "image": image,
         "routed": np.array([run_eval.routes_to_query(q["text"]) for q in qs])}
    images = sorted(set(image.values()))
    idx = {im: k for k, im in enumerate(images)}
    b["images"], b["img_of"] = images, np.array([idx[image[q]] for q in order])
    b["groups"] = {"[unrouted]": ~b["routed"], "[routed]": b["routed"]}
    if bench == "vstar":
        for c in sorted({gts[q]["category"] for q in order}):
            b["groups"][f"[{c}]"] = np.array([gts[q]["category"] == c for q in order])
    return b


def recompute(bench, b, answers):
    if bench == "vstar":
        from score_pope_mme import vstar_letter
        return np.array([float(vstar_letter(answers[q], b["gts"][q]["n_options"]) == b["gts"][q]["label"]) for q in b["order"]])
    sys.path.insert(0, LLAVA)
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    ev = TextVQAAccuracyEvaluator()
    return np.array([ev._compute_answer_scores(b["gts"][q]["answers"]).get(ev.answer_processor(answers[q]), 0.0)
                     for q in b["order"]])


def runs_rows():
    with open(f"{OUT}/EXP-0025/runs.tsv") as f:
        rows = {x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")}
    for (bench, meth), rid in RUNS.items():
        if rows.get(f"EXP-0025-{rid}", {}).get("status") != "completed":
            sys.exit(f"EXP-0025-{rid} ({meth}, {bench}) is not completed")
    return rows


def image_sizes(bench, b):
    from PIL import Image
    out = {}
    for im in b["images"]:
        with Image.open(os.path.join(IMAGES[bench], im)) as x:
            out[im] = x.size                                    # (width, height), as LLaVA's loader passes it
    return out


def main():
    out_dir = f"{OUT}/EXP-0025/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    from fastv_llava import _content_band
    rows = runs_rows()
    checks = {}
    res = {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "boxes": {}, "zoomed_share": {},
           "vision_passes_per_question": {}}
    for bench in BENCHES:
        b = load_bench(bench)
        n = len(b["order"])
        routed = b["routed"]
        sizes = image_sizes(bench, b)
        sc, ans, kept, zoom = {}, {}, {}, {}
        for (bb, meth), rid in RUNS.items():
            if bb != bench:
                continue
            d = f"{OUT}/EXP-0025/EXP-0025-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            s = recompute(bench, b, a)
            logf = "vstar_scores.jsonl" if bench == "vstar" else "textvqa_scores.jsonl"
            logged = {}
            for line in open(f"{d}/{logf}", encoding="utf-8"):
                x = json.loads(line)
                logged[str(x["question_id"])] = x["score"]
            checks[f"{bench} {meth}: recomputed scores equal {logf}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            r = rows[f"EXP-0025-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth], ans[meth] = s, [a[q] for q in b["order"]]
            if meth == "vanilla":
                continue
            kept[meth] = json.load(open(f"{d}/kept_tokens.json"))
            checks[f"{bench} {meth}: one kept record per question"] = float(kept[meth]["n_prefills"] == n)
            checks[f"{bench} {meth}: logged route decisions equal the rule on the question text"] = float(
                np.mean(np.array(kept[meth]["routed"]) == routed))
            if meth in ZOOM:
                z = json.load(open(f"{d}/zoom.json"))
                checks[f"{bench} {meth}: one zoom record per question, in order"] = float(
                    z["n_prefills"] == n and [x["index"] for x in z["records"]] == list(range(n)))
                zoom[meth] = z["records"]
        checks[f"{bench} router: {KEEP} image tokens kept in every prefill"] = float(np.mean([len(x) == KEEP for x in kept["router"]["kept"]]))
        zoomed = {}
        for meth in [m for m in ZOOM if m in zoom]:
            recs, k = zoom[meth], kept[meth]["kept"]
            zoomed[meth] = np.array([bool(x["zoom"]) for x in recs])
            checks[f"{bench} {meth}: zoomed on at least 99% of the questions (s_min 0)"] = float(zoomed[meth].mean() >= 0.99)
            want = [KEEP - N_ZOOM if zm else KEEP for zm in zoomed[meth]]
            checks[f"{bench} {meth}: kept image tokens = keep - 16 when zoomed, keep otherwise"] = float(
                np.mean([len(x) == w for x, w in zip(k, want)]))
            checks[f"{bench} {meth}: kept log's zoom_tokens = 16 exactly on the zoomed prefills"] = float(np.mean(
                [zt == (N_ZOOM if zm else 0) for zt, zm in zip(kept[meth]["zoom_tokens"], zoomed[meth])]))
            same = [ans[meth][i] == ans["router"][i] for i in range(n) if not zoomed[meth][i]]
            checks[f"{bench} {meth}: answers of the prefills not zoomed = router's"] = float(np.mean(same)) if same else 1.0
            res["zoomed_share"][f"{bench} {meth}"] = float(zoomed[meth].mean())
            res["vision_passes_per_question"][f"{bench} {meth}"] = 1 + (float(zoomed[meth].mean()) if meth in CROP else 0.0)
            if meth in CROP:
                checks[f"{bench} {meth}: every zoom record re-encodes a crop"] = float(np.mean(
                    [x.get("source") == "crop" and "crop_box" in x for x in recs if x["zoom"]]))
        res["vision_passes_per_question"][f"{bench} router"] = 1.0
        if "g16full" in zoom:
            recs = [x for x in zoom["g16full"] if x["zoom"]]
            checks[f"{bench} g16full: every zoom record is the image's own lattice tokens, no crop"] = float(np.mean(
                [x.get("source") == "full" and "crop_box" not in x and len(x.get("picked", [])) == N_ZOOM for x in recs]))
            both = [(x["grid_box"], y["grid_box"]) for x, y in zip(zoom["g16full"], zoom["z16g"]) if "grid_box" in x and "grid_box" in y]
            checks[f"{bench} g16full: boxes equal z16g's (same locate step)"] = float(np.mean([x == y for x, y in both])) if both else 0.0
        if "z16gctr" in zoom:
            both = [(x["side"], y["side"]) for x, y in zip(zoom["z16gctr"], zoom["z16g"]) if "side" in x and "side" in y]
            checks[f"{bench} z16gctr: box side equals z16g's"] = float(np.mean([x == y for x, y in both])) if both else 0.0
            ctr = []
            for i, x in enumerate(zoom["z16gctr"]):
                if "grid_box" in x:
                    r_lo, r_hi, c_lo, c_hi = _content_band(*sizes[b["image"][b["order"][i]]], 24)
                    r0, c0, r1, c1 = x["grid_box"]
                    ctr.append(abs((r0 + r1) / 2 - (r_lo + r_hi) / 2) <= 0.5 and abs((c0 + c1) / 2 - (c_lo + c_hi) / 2) <= 0.5)
            checks[f"{bench} z16gctr: box centred on the image's content band"] = float(np.mean(ctr)) if ctr else 0.0
        if "z16g" in zoom:
            located = [(i, x) for i, x in enumerate(zoom["z16g"]) if "side" in x]
            mag = [max(sizes[b["image"][b["order"][i]]]) / max(1.0, x["crop_box"][2] - x["crop_box"][0]) for i, x in located]
            res["boxes"][bench] = {"mean_side": float(np.mean([x["side"] for _, x in located])),
                                   "span_over_16": float(np.mean([x["span"] > 16 for _, x in located])),
                                   "mean_top8_inside": float(np.mean([x["top_inside"] for _, x in located])),
                                   "top1_inside": float(np.mean([x["top1_inside"] for _, x in located])),
                                   "median_long_side_px": float(np.median([max(v) for v in sizes.values()])),
                                   "median_magnification": float(np.median(mag))}
        for meth in [m for m in ZOOM if m in sc]:                   # unrouted-only splices of the zoom-everything runs
            sc[f"{meth}|unrouted"] = np.where(routed, sc["router"], sc[meth])
            res["vision_passes_per_question"][f"{bench} {meth}|unrouted"] = 1 + (
                float((zoomed[meth] & ~routed).mean()) if meth in CROP else 0.0)
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

        for v in [m for m in sc if m not in ("router", "vanilla")]:
            res["pairs"][f"{bench} {v} - router"] = paired(v, "router")
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} {v} - router {g}"] = paired(v, "router", m)
        if "vanilla" in sc:
            res["pairs"][f"{bench} vanilla - router"] = paired("vanilla", "router")
            for g, m in b["groups"].items():
                res["pairs"][f"{bench} vanilla - router {g}"] = paired("vanilla", "router", m)
        for ctrl in [c for c in ("g16full", "z16gctr") if c in sc]:   # attribution, spliced and as run
            for v in ("|unrouted", ""):
                res["pairs"][f"{bench} z16g{v} - {ctrl}{v}"] = paired(f"z16g{v}", f"{ctrl}{v}")
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in b["groups"].items() if m.any()}}
                               for v, s in sc.items()}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    P = res["pairs"]
    t1, t2 = P["textvqa_noocr z16g|unrouted - router"], P["vstar z16g|unrouted - router"]
    t3 = {c: P[f"textvqa_noocr z16g|unrouted - {c}|unrouted"] for c in ("g16full", "z16gctr")}
    T1, T2, T3 = t1[1] > 0, t2[1] > 0, all(x[1] > 0 for x in t3.values())
    large = t1[1] >= LARGE["textvqa_noocr"] or t2[1] >= LARGE["vstar"]
    if T1:
        verdict = "T1 成立：放大在看细节的题上有效；" + ("T3 也成立：增益可称为“定位后重新编码、看得更细”" if T3 else "T3 不成立：不能说增益来自看得更细")
    elif T2:
        verdict = "T1 不成立、T2 成立：只在分辨率极高、物体极小的图上有效，如实限定"
    else:
        verdict = "T1、T2 都不成立：放大在本设置下不成立（预算内只换 16 个 token 不够），回到第 ⑨ 步"
    wording = ("大提升" if large else "提升") if (T1 or T2) else "没有可检出的提升"
    gap = {}
    for bench, t in (("textvqa_noocr", t1), ("vstar", t2)):
        g = P[f"{bench} vanilla - router"][0]
        gap[bench] = t[0] / g if g else None
    res["criteria"] = {"T1": T1, "T2": T2, "T3": T3, "T1_pair": t1, "T2_pair": t2, "T3_pairs": t3, "large": large,
                       "verdict": verdict, "wording": wording, "gap_closed": gap}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0025.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0025（计划 v1）：预算内放大，看细节的题（TextVQA、V*，64 档）", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {m} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "框的统计：" + "；".join(f"{k} {json.dumps(x)}" for k, x in res["boxes"].items()),
          "", "被放大的题的比例：" + "，".join(f"{k} {x:.3f}" for k, x in res["zoomed_share"].items()),
          "", "每题视觉编码器前向次数：" + "，".join(f"{k} {x:.3f}" for k, x in res["vision_passes_per_question"].items()),
          "", "每题秒数：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    L += ["", "## 按计划写定的判据", "",
          f"- **T1**（TextVQA 不给 OCR，z16g|unrouted − router）：{fmt(t1)} → {'成立' if T1 else '不成立'}",
          f"- **T2**（V*，z16g|unrouted − router）：{fmt(t2)} → {'成立' if T2 else '不成立'}",
          f"- **T3**（TextVQA 不给 OCR）：减去全图 lattice {fmt(t3['g16full'])}，减去居中框 lattice {fmt(t3['z16gctr'])} → "
          f"{'成立' if T3 else '不成立'}",
          f"- **措辞**：{wording}（“大提升”要 TextVQA 下界 ≥ +2.0 或 V* 下界 ≥ +5.0）",
          f"- **判读**：{verdict}",
          "- **缩小的差距**（Δ ÷（vanilla − router））：" + "，".join(
              f"{k} {v:.2f}" if v is not None else f"{k} —" for k, v in gap.items())]
    open(f"{out_dir}/0025.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
