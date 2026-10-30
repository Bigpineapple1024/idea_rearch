"""EXP-0035 (plan v1, run only when EXP-0034's reading sends IDEA-0008 to confirmation): IDEA-0008 on fresh data, VQAv2
test5 and GQA val400c (make_confirm_sets2.py, drawn while EXP-0034 ran, before any EXP-0034 result), with TextVQA (no OCR
tokens) described.

  python analyze_0035.py      # writes code/output/EXP-0035/analysis/{0035.md,0035.json}

peek (R01, R02): K = 1, DivPrune's 61 chosen before layer 0, layer 0 lets every token see all patches, decoding never sees
the dropped ones; DivPrune-64 (R03, R04); nopeek (R05, R06): the same 61 at their original positions, layer 0 sees none of
the dropped patches; TextVQA without OCR tokens: peek (R07), DivPrune-64 (R08); unpruned (R09, R10). Q1 (primary, at about
equal FLOPs): peek - DivPrune-64, 95% interval above 0 on both fresh sets: holds; on one: partly; on neither: fails. Q2
(mechanism): peek - nopeek, read the same way. Q3 (described): peek - DivPrune-64 on TextVQA. Large (reported, not part of
the reading): the two-set mean of Q1 at least +1.5. Described alongside: by question type (GQA's attribute and choose
questions above all, EXP-0034's F78), against unpruned. Per-question scores are recomputed from answers.jsonl and checked
against each run's own score file; paired differences resample images with replacement (1000 draws, seed 0) within each
set (TextVQA: within its images), as in analyze_0031.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_0025 as a25  # noqa: E402
import analyze_0028 as a28  # noqa: E402
from squeeze_0018 import EVAL_DATA, OUT   # noqa: E402

GENERAL = ("vqav2_test5", "gqa_val400c")
BENCHES = GENERAL + ("textvqa_noocr",)
FILES = {"vqav2_test5": (f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test5.jsonl", f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test5_annotations.jsonl",
                         "D:/6CJNM007/data/coco/val2014"),
         "gqa_val400c": (f"{EVAL_DATA}/gqa/llava_gqa_val400c.jsonl", f"{EVAL_DATA}/gqa/llava_gqa_val400c_annotations.jsonl",
                         "D:/6CJNM007/data/gqa/images_val400c")}
SCORES = {"vqav2_test5": "vqa_scores.jsonl", "gqa_val400c": "gqa_scores.jsonl", "textvqa_noocr": "textvqa_scores.jsonl"}
RUNS = {("vqav2_test5", "peek"): ("R01", "fastv61k1_div_peek_kll"), ("gqa_val400c", "peek"): ("R02", "fastv61k1_div_peek_kll"),
        ("vqav2_test5", "div64"): ("R03", "divprune64"), ("gqa_val400c", "div64"): ("R04", "divprune64"),
        ("vqav2_test5", "nopeek"): ("R05", "fastv61k1_div_nopeek_kll"), ("gqa_val400c", "nopeek"): ("R06", "fastv61k1_div_nopeek_kll"),
        ("textvqa_noocr", "peek"): ("R07", "fastv61k1_div_peek_kll"), ("textvqa_noocr", "div64"): ("R08", "divprune64"),
        ("vqav2_test5", "vanilla"): ("R09", "vanilla"), ("gqa_val400c", "vanilla"): ("R10", "vanilla")}
NAMES = {"peek": "K = 1、留 61、第 0 层都看（peek）", "div64": "DivPrune-64", "nopeek": "同样的 61 个、第 0 层看不到被剪的（nopeek）",
         "vanilla": "不剪枝"}
KEEP, N_IMG, N_BOOT, LARGE = 61, 576, 1000, 1.5


def dev_reading():
    """EXP-0034's sealed reading; stops unless it sent IDEA-0008 to confirmation (the plan's condition)."""
    c = json.load(open(f"{OUT}/EXP-0034/analysis/0034.json", encoding="utf-8"))["criteria"]
    if not c["confirmation_next"]:
        sys.exit(f"EXP-0034 did not send IDEA-0008 to confirmation: {c['Q1_reading']}")
    return c


def load(bench):
    if bench in GENERAL:
        return a28.load_bench(bench)
    b = a25.load_bench(bench)
    sizes = a25.image_sizes(bench, b)
    b["size"] = [sizes[b["image"][q]] for q in b["order"]]
    return b


def recompute(bench, b, answers):
    return a28.recompute(bench, b, answers) if bench in GENERAL else a25.recompute(bench, b, answers)


def main():
    out_dir = f"{OUT}/EXP-0035/analysis"
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    dev = dev_reading()
    a28.FILES.update(FILES)                               # analyze_0028's loader on the fresh sets
    with open(f"{OUT}/EXP-0035/runs.tsv") as f:
        rows = {x["run_id"]: x for x in csv.DictReader(f, delimiter="\t")}
    for (bench, meth), (rid, cfg) in RUNS.items():
        x = rows.get(f"EXP-0035-{rid}", {})
        if x.get("status") != "completed":
            sys.exit(f"EXP-0035-{rid} ({meth}, {bench}) is not completed")
        if x.get("config") != cfg or x.get("bench") != bench:
            sys.exit(f"EXP-0035-{rid} has {x.get('config')} x {x.get('bench')}, planned {cfg} x {bench}")
    checks, res = {}, {"pairs": {}, "table": {}, "seconds_per_answer": {}, "peak_gib": {}, "logs": {}, "dev_reading": dev}
    for bench in BENCHES:
        b = load(bench)
        n = len(b["order"])
        sc, d_of = {}, {}
        for (bb, meth), (rid, cfg) in RUNS.items():
            if bb != bench:
                continue
            d = d_of[meth] = f"{OUT}/EXP-0035/EXP-0035-{rid}"
            a = {}
            for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
                x = json.loads(line)
                a[str(x["question_id"])] = x["text"]
            checks[f"{bench} {meth}: one answer per question"] = float(sorted(a) == sorted(b["order"]))
            s = recompute(bench, b, a)
            logged = {}
            for line in open(f"{d}/{SCORES[bench]}", encoding="utf-8"):
                x = json.loads(line)
                logged[str(x["question_id"])] = x["score"]
            checks[f"{bench} {meth}: recomputed scores equal {SCORES[bench]}"] = float(np.allclose(s, [logged[q] for q in b["order"]]))
            r = rows[f"EXP-0035-{rid}"]
            checks[f"{bench} {meth}: accuracy equals the registered headline"] = float(abs(100 * s.mean() - float(r["headline"])) < 1e-6)
            res["seconds_per_answer"][f"{bench} {meth}"] = float(r["s_per_answer"])
            res["peak_gib"][f"{bench} {meth}"] = float(r["peak_gib"])
            sc[meth] = s
        pre = json.load(open(f"{d_of['div64']}/div.json"))["records"]
        checks[f"{bench} div64: one selection record per question, keep 64"] = float(len(pre) == n and all(len(x["order"]) == 64 for x in pre))
        for meth in ("peek", "nopeek") if bench in GENERAL else ("peek",):
            mode = {"peek": "prefill", "nopeek": "none"}[meth]
            rec = json.load(open(f"{d_of[meth]}/div.json"))["records"]
            kept = json.load(open(f"{d_of[meth]}/kept_tokens.json"))["kept"]
            lt = json.load(open(f"{d_of[meth]}/layer_tokens.json"))
            checks[f"{bench} {meth}: one record, one kept list and one row of layer lengths per prefill, mode '{mode}'"] = float(
                len(rec) == n == len(kept) == lt["n_prefills"] == lt["n_complete"] and all(x.get("peek") == mode for x in rec))
            checks[f"{bench} {meth}: kept = the 61 logged"] = float(np.mean([len(x["order"]) == KEEP == len(set(x["order"]))
                                                                            and sorted(x["order"]) == sorted(k) for x, k in zip(rec, kept)]))
            same = float(np.mean([x["order"] == y["order"][:KEEP] for x, y in zip(rec, pre)]))
            checks[f"{bench} {meth}: its selection = the first 61 of DivPrune-64's (>= 99%)"] = float(same >= 0.99)
            res["logs"][f"{bench} {meth} order = DivPrune-64's first 61"] = same
            checks[f"{bench} {meth}: layer 0 sees the whole prompt, layers 1 to 31 that minus 515"] = float(np.mean(
                [len(t) == 32 and all(x == t[0] - (N_IMG - KEEP) for x in t[1:]) for t in lt["tokens_per_layer"]]))
        imgs = b["img_of"] if "img_of" in b else None
        if imgs is None:                                   # TextVQA: one image per question group, as analyze_0031
            ids = sorted({b["image"][q] for q in b["order"]})
            idx = {im: k for k, im in enumerate(ids)}
            imgs = np.array([idx[b["image"][q]] for q in b["order"]])
        by_img = [np.flatnonzero(imgs == k) for k in range(int(imgs.max()) + 1)]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

        pairs = [("peek", "div64")] + ([("peek", "nopeek"), ("nopeek", "div64"), ("vanilla", "peek"), ("vanilla", "div64")] if bench in GENERAL else [])
        groups = b.get("groups", {})
        for x, y in pairs:
            res["pairs"][f"{bench} {x} - {y}"] = paired(x, y)
            for g, m in groups.items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(x, y, m)
        res["table"][bench] = {v: {"accuracy": 100 * float(s.mean()), **{g: 100 * float(s[m].mean()) for g, m in groups.items() if m.any()}}
                               for v, s in sc.items()}
    if any(v < 0.99 for v in checks.values()):
        print("STOP", {k: v for k, v in checks.items() if v < 0.99})
        sys.exit(1)
    P = res["pairs"]
    word = {2: "成立（两个新集的区间都在 0 以上）", 1: "部分成立（只有一个新集的区间在 0 以上）", 0: "不成立（两个新集的区间都含 0 或在 0 以下）"}

    def two(x, y):
        v = {bench: P[f"{bench} {x} - {y}"] for bench in GENERAL}
        return v, float(np.mean([v[bench][0] for bench in GENERAL])), sum(v[bench][1] > 0 for bench in GENERAL)

    q1, m1, n1 = two("peek", "div64")
    q2, m2, n2 = two("peek", "nopeek")
    q3 = P["textvqa_noocr peek - div64"]
    res["criteria"] = {"Q1_pairs": q1, "Q1_mean": m1, "Q1": word[n1], "Q1_large": m1 >= LARGE, "Q2_pairs": q2, "Q2_mean": m2,
                       "Q2": word[n2], "Q3": q3}
    res["checks"] = checks
    os.makedirs(out_dir)
    json.dump(res, open(f"{out_dir}/0035.json", "w"), indent=2, ensure_ascii=False)
    fmt = lambda x: f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）" if x else "—"   # noqa: E731
    L = ["# EXP-0035（计划 v1）：IDEA-0008 的确认（VQAv2 test5、GQA val400c，新抽的数据；TextVQA 描述）", "",
         f"EXP-0034 的读法：{dev['Q1_reading']}（两集平均 {dev['Q1_mean']:+.2f}）", "", "## 核对", ""]
    L += [f"- {k}: {100 * x:.1f}%" for k, x in checks.items()]
    L += ["", "选择记录：" + "；".join(f"{k} {x:.4f}" for k, x in res["logs"].items())]
    for bench in BENCHES:
        t = res["table"][bench]
        cols = [g for g in next(iter(t.values())) if g != "accuracy"]
        L += ["", f"## 准确率：{bench}", "", "| 配置 | 全部 | " + " | ".join(cols) + " |", "| --- | --- |" + " --- |" * len(cols)]
        L += [f"| {NAMES[m]} | {x['accuracy']:.2f} | " + " | ".join(f"{x.get(g, float('nan')):.2f}" for g in cols) + " |" for m, x in t.items()]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"]
    L += [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    L += ["", "每题秒数（第 0 层按全部 576 个计算的实现）：" + "，".join(f"{k} {x:.4f}" for k, x in res["seconds_per_answer"].items()),
          "", "峰值显存（GiB）：" + "，".join(f"{k} {x:.2f}" for k, x in res["peak_gib"].items())]
    L += ["", "## 按计划写定的读法", "",
          f"- **Q1**（peek − DivPrune-64）：VQAv2 test5 {fmt(q1['vqav2_test5'])}，GQA val400c {fmt(q1['gqa_val400c'])}，两集平均 {m1:+.2f} → {word[n1]}"
          + ("；两集平均 ≥ +1.5，记为“提升大”" if m1 >= LARGE else ""),
          f"- **Q2**（peek − nopeek）：VQAv2 test5 {fmt(q2['vqav2_test5'])}，GQA val400c {fmt(q2['gqa_val400c'])}，两集平均 {m2:+.2f} → {word[n2]}",
          f"- **Q3**（描述，TextVQA 不给 OCR：peek − DivPrune-64）：{fmt(q3)}"]
    open(f"{out_dir}/0035.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
