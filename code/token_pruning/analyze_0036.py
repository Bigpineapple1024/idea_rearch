"""EXP-0036 (plan v1, written 2026-10-31 before any EXP-0035 or EXP-0036 result): IDEA-0008 and SQZ-EXP-0034's two
findings on three other benchmarks: MME (2374 questions; position, existence, count and colour subtasks), V* Bench (191;
relative position and direct attributes) and POPE (8910 yes/no existence questions).

  python analyze_0036.py           # writes code/output/EXP-0036/analysis/{0036.md,0036.json}
  python analyze_0036.py --dry     # the same code on stand-in runs (older runs on the same benchmarks) -> tmp; not results

Runs: peek (K = 1, DivPrune's 61 chosen before layer 0, layer 0 lets every token see all patches) R01 / R04 / R07,
DivPrune-64 R02 / R05 / R08, nopeek (the same 61 at their original positions, layer 0 sees none of the dropped patches)
R03 / R06 / R09, on MME / V* / POPE. Unpruned, reused: EXP-0002-R05 (MME), EXP-0025-R08 (V*), EXP-0002-R34 (POPE).
Per-question correctness: POPE with LLaVA's eval_pope.py rule (first sentence; "No", "not" or "no" as a word -> no) against
the k-th label of the question's split; MME with the official calculation.py rule on its per-task files (mme_eval_tool);
V* from vstar_scores.jsonl. Each is checked against the run's own metrics.json.
Q1 (generalisation): peek - DivPrune-64 per benchmark, per-question accuracy; 95% interval above 0 on 3 / 2 / 1 / 0
   benchmarks. Q2 (SQZ-EXP-0034's F88 on POPE): nopeek - DivPrune-64 on questions whose answer is "no", interval below 0
   -> "the lean towards yes reproduces on POPE"; peek - nopeek on them above 0 -> "the peek removes it"; peek - nopeek on
   "yes" questions below 0 -> "its cost reproduces". Q3 (F89): MME position and V* relative position pooled (136
   questions): nopeek - DivPrune-64 above 0 -> "the original position ids help position questions here too".
Described: MME scores per subtask, POPE accuracy / F1 / yes ratio per split, V* by category, peek - DivPrune-64 against
the unpruned run, first-token log-odds of peek and nopeek on POPE, prefill FLOPs (analyze_0034.macs).
Bootstrap: resample images with replacement within each benchmark (pooled: within each, then together), 1000 draws, seed 0.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
OUT = "D:/6CJNM007/idea_rearch/code/output"
EV = "D:/6CJNM007/data/llava_eval"
DRY = "--dry" in sys.argv
KEEP, N_IMG, N_BOOT = 61, 576, 1000
BENCHES = ("mme", "vstar", "pope")
if DRY:
    DST = "D:/6CJNM007/tmp/exp0036/dry_out"
    RUNS = {("mme", "peek"): ("EXP-0002", "R02", "vanilla"), ("mme", "div64"): ("EXP-0002", "R05", "vanilla"),
            ("mme", "nopeek"): ("EXP-0002", "R02", "vanilla"), ("vstar", "peek"): ("EXP-0025", "R08", "vanilla"),
            ("vstar", "div64"): ("EXP-0025", "R08", "vanilla"), ("vstar", "nopeek"): ("EXP-0025", "R08", "vanilla"),
            ("pope", "peek"): ("EXP-0002", "R04", "vanilla"), ("pope", "div64"): ("EXP-0002", "R34", "vanilla"),
            ("pope", "nopeek"): ("EXP-0002", "R04", "vanilla")}
else:
    DST = f"{OUT}/EXP-0036/analysis"
    RUNS = {("mme", "peek"): ("EXP-0036", "R01", "fastv61k1_div_peek_kll"), ("mme", "div64"): ("EXP-0036", "R02", "divprune64"),
            ("mme", "nopeek"): ("EXP-0036", "R03", "fastv61k1_div_nopeek_kll"), ("vstar", "peek"): ("EXP-0036", "R04", "fastv61k1_div_peek_kll"),
            ("vstar", "div64"): ("EXP-0036", "R05", "divprune64"), ("vstar", "nopeek"): ("EXP-0036", "R06", "fastv61k1_div_nopeek_kll"),
            ("pope", "peek"): ("EXP-0036", "R07", "fastv61k1_div_peek_kll"), ("pope", "div64"): ("EXP-0036", "R08", "divprune64"),
            ("pope", "nopeek"): ("EXP-0036", "R09", "fastv61k1_div_nopeek_kll")}
RUNS.update({("mme", "vanilla"): ("EXP-0002", "R05", "vanilla"), ("vstar", "vanilla"): ("EXP-0025", "R08", "vanilla"),
             ("pope", "vanilla"): ("EXP-0002", "R34", "vanilla")})
METHODS = ("peek", "div64", "nopeek", "vanilla")
NAMES = {"peek": "peek（K = 1、留 61、第 0 层都看）", "div64": "DivPrune-64", "nopeek": "nopeek（同样的 61 个、第 0 层看不到被剪的）", "vanilla": "不剪枝"}
MME_TASKS = {"Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
             "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]}
CHECKS = {}


def run_dir(bench, meth):
    exp, rid, _ = RUNS[(bench, meth)]
    return f"{OUT}/{exp}/{exp}-{rid}"


def check_runs():
    rows = {}
    for exp in sorted({e for e, _, _ in RUNS.values()}):
        rows.update({r["run_id"]: r for r in csv.DictReader(open(f"{OUT}/{exp}/runs.tsv", encoding="utf-8"), delimiter="\t")})
    for (bench, meth), (exp, rid, cfg) in RUNS.items():
        r = rows.get(f"{exp}-{rid}", {})
        CHECKS[f"{exp}-{rid} is a completed {cfg} x {bench} ({meth})"] = float(r.get("status") == "completed" and r.get("config") == cfg
                                                                             and r.get("bench") == bench)
    return rows


def pope_label(text):
    """LLaVA's eval_pope.py: keep the first sentence; "No", "not" or "no" as a word -> no, else yes."""
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "no" if ("No" in words or "not" in words or "no" in words) else "yes"


def mme_label(pred):
    """MME's calculation.py parse_pred_ans, on the lower-cased prediction as read from its per-task file."""
    if pred in ("yes", "no"):
        return pred
    p = pred[:4]
    return "yes" if "yes" in p else "no" if "no" in p else "other"


def load_pope():
    qs = [json.loads(l) for l in open(f"{EV}/pope/llava_pope_test.jsonl", encoding="utf-8")]
    labels = {}
    for split in ("adversarial", "popular", "random"):
        lab = [json.loads(l) for l in open(f"{EV}/pope/coco/coco_pope_{split}.json", encoding="utf-8")]
        mine = [q for q in qs if q["category"] == split]
        CHECKS[f"pope {split}: the k-th question is the k-th label's image and question"] = float(
            len(lab) == len(mine) and all(a["image"] == b["image"] and b["text"].startswith(a["text"]) for a, b in zip(lab, mine)))
        labels.update({str(b["question_id"]): a["label"] for a, b in zip(lab, mine)})
    order = [str(q["question_id"]) for q in qs]
    CHECKS["pope: question ids are unique"] = float(len(set(order)) == len(order))
    out = {"order": order, "image": [q["image"] for q in qs], "split": [q["category"] for q in qs], "gt": np.array([labels[q] for q in order]),
           "group": {"gt yes": np.array([labels[q] == "yes" for q in order]), "gt no": np.array([labels[q] == "no" for q in order])}, "sc": {}, "said": {}}
    for meth in METHODS:
        d = run_dir("pope", meth)
        a = {str(x["question_id"]): x["text"] for x in (json.loads(l) for l in open(f"{d}/answers.jsonl", encoding="utf-8"))}
        CHECKS[f"pope {meth}: one answer per question"] = float(sorted(a) == sorted(order))
        said = np.array([pope_label(a[q]) for q in order])
        out["said"][meth] = said
        out["sc"][meth] = (said == out["gt"]).astype(float)
        m = json.load(open(f"{d}/metrics.json", encoding="utf-8"))
        for split in ("adversarial", "popular", "random"):
            k = np.array([s == split for s in out["split"]])
            acc, yr = float(out["sc"][meth][k].mean()), float((said[k] == "yes").mean())
            CHECKS[f"pope {meth} {split}: accuracy and yes ratio equal metrics.json"] = float(
                abs(acc - m["splits"][split]["accuracy"]) < 1e-3 and abs(yr - m["splits"][split]["yes_ratio"]) < 1e-3)
    return out


def load_mme():
    out = {"sc": {}, "said": {}}
    keys = None
    for meth in METHODS:
        d = run_dir("mme", meth)
        rows = []
        for task in MME_TASKS["Perception"] + MME_TASKS["Cognition"]:
            for line in open(f"{d}/mme_eval_tool/{task}.txt", encoding="utf-8"):
                img, q, gt, pred = line.split("\t")
                rows.append((task, img, q, gt.lower(), mme_label(pred.lower())))
        k = [(t, i, q) for t, i, q, _, _ in rows]
        if keys is None:
            keys = k
            out["order"], out["task"] = k, [t for t, _, _ in k]
            out["image"] = [f"{t}/{i}" for t, i, _ in k]
            out["gt"] = np.array([g for _, _, _, g, _ in rows])
        CHECKS[f"mme {meth}: the same 2374 questions in the same order"] = float(k == keys and len(k) == 2374)
        out["said"][meth] = np.array([p for _, _, _, _, p in rows])
        out["sc"][meth] = (out["said"][meth] == out["gt"]).astype(float)
        m = json.load(open(f"{d}/metrics.json", encoding="utf-8"))
        ok = []
        for kind, tasks in MME_TASKS.items():
            for task in tasks:
                idx = [j for j, t in enumerate(out["task"]) if t == task]
                acc = out["sc"][meth][idx].mean()
                pairs = {}
                for j in idx:
                    pairs.setdefault(out["image"][j], []).append(out["sc"][meth][j])
                accp = np.mean([all(v) for v in pairs.values()])
                ok.append(abs(100 * (acc + accp) - m[f"{kind.lower()}_tasks"][task]) < 1e-6)
        CHECKS[f"mme {meth}: every subtask score (acc + acc+) equals metrics.json"] = float(all(ok))
    out["group"] = {f"[{t}]": np.array([x == t for x in out["task"]]) for t in ("position", "existence", "count", "color")}
    return out


def load_vstar():
    qs = [json.loads(l) for l in open(f"{EV}/vstar/llava_vstar_test.jsonl", encoding="utf-8")]
    order = [str(q["question_id"]) for q in qs]
    out = {"order": order, "image": [q["image"] for q in qs], "sc": {}}
    cat = None
    for meth in METHODS:
        d = run_dir("vstar", meth)
        s = {str(x["question_id"]): x for x in (json.loads(l) for l in open(f"{d}/vstar_scores.jsonl", encoding="utf-8"))}
        CHECKS[f"vstar {meth}: one score per question"] = float(sorted(s) == sorted(order))
        out["sc"][meth] = np.array([float(s[q]["score"]) for q in order])
        cat = [s[q]["category"] for q in order]
        m = json.load(open(f"{d}/metrics.json", encoding="utf-8"))
        CHECKS[f"vstar {meth}: accuracy equals metrics.json"] = float(abs(100 * out["sc"][meth].mean() - m["accuracy"]) < 1e-6)
    out["group"] = {f"[{c}]": np.array([x == c for x in cat]) for c in ("relative_position", "direct_attributes")}
    return out


def draws_for(b):
    ims = sorted(set(b["image"]))
    idx = {im: k for k, im in enumerate(ims)}
    img_of = np.array([idx[i] for i in b["image"]])
    by_img = [np.flatnonzero(img_of == k) for k in range(len(ims))]
    rng = np.random.default_rng(0)
    return [np.concatenate([by_img[k] for k in rng.integers(0, len(by_img), len(by_img))]) for _ in range(N_BOOT)]


def paired(b, x, y, mask=None, values=None):
    d = (b["sc"][x] - b["sc"][y]) if values is None else (values[x] - values[y])
    k = 100.0 if values is None else 1.0
    m = np.ones(len(d), bool) if mask is None else mask
    if m.sum() == 0:
        return None
    boot = [k * d[i][m[i]].mean() for i in b["draws"] if m[i].any()]
    return [k * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]


def pooled(parts, x, y):
    """Spatial questions pooled over benchmarks: each draw resamples images within each benchmark, then pools."""
    d_all, n_all = [], 0
    for b, m in parts:
        d_all.append(100 * (b["sc"][x] - b["sc"][y])[m])
        n_all += int(m.sum())
    mean = float(np.concatenate(d_all).mean())
    boot = []
    for t in range(N_BOOT):
        vals = []
        for b, m in parts:
            i = b["draws"][t]
            vals.append(100 * (b["sc"][x] - b["sc"][y])[i][m[i]])
        boot.append(np.concatenate(vals).mean())
    return [mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), n_all]


def kll_logs(bench, b):
    """peek / nopeek selection logs (as analyze_0034): one record per prefill with its mode, 61 kept, layer lengths; FLOPs."""
    import analyze_0034 as a34
    n = len(b["order"])
    pre = json.load(open(f"{run_dir(bench, 'div64')}/div.json"))["records"]
    CHECKS[f"{bench} div64: one selection record per question, keep 64"] = float(len(pre) == n and all(len(x["order"]) == 64 for x in pre))
    others = []
    for meth, mode in (("peek", "prefill"), ("nopeek", "none")):
        d = run_dir(bench, meth)
        rec = json.load(open(f"{d}/div.json"))["records"]
        kept = json.load(open(f"{d}/kept_tokens.json"))["kept"]
        lt = json.load(open(f"{d}/layer_tokens.json"))
        CHECKS[f"{bench} {meth}: one record, one kept list and one row of layer lengths per prefill, mode '{mode}'"] = float(
            len(rec) == n == len(kept) == lt["n_prefills"] == lt["n_complete"] and all(x.get("peek") == mode for x in rec))
        CHECKS[f"{bench} {meth}: kept = the 61 logged"] = float(np.mean([len(x["order"]) == KEEP == len(set(x["order"])) and sorted(x["order"]) == sorted(k)
                                                                        for x, k in zip(rec, kept)]))
        CHECKS[f"{bench} {meth}: its selection = the first 61 of DivPrune-64's (>= 99%)"] = float(np.mean([x["order"] == y["order"][:KEEP] for x, y in zip(rec, pre)]) >= 0.99)
        CHECKS[f"{bench} {meth}: layer 0 sees the whole prompt, layers 1 to 31 that minus 515"] = float(np.mean(
            [len(t) == 32 and all(v == t[0] - (N_IMG - KEEP) for v in t[1:]) for t in lt["tokens_per_layer"]]))
        others.append([t[0] - N_IMG for t in lt["tokens_per_layer"]])
    CHECKS[f"{bench}: peek and nopeek see the same prompts"] = float(others[0] == others[1])
    kept_p = [x["order"] for x in json.load(open(f"{run_dir(bench, 'peek')}/div.json"))["records"]]
    tot = {m: float(np.mean([a34.macs(o, sorted(k), m) for o, k in zip(others[0], kept_p)])) for m in ("div64", "vanilla", "nopeek", "peek_kv", "peek_impl")}
    return {"mean_non_image_tokens": float(np.mean(others[0])), "relative_to_div64": {m: tot[m] / tot["div64"] for m in tot}}


def log_odds(bench, b):
    lo = {}
    for meth in ("peek", "nopeek"):
        x = json.load(open(f"{run_dir(bench, meth)}/answer_logits.json", encoding="utf-8"))
        CHECKS[f"{bench} {meth}: one first-token record per question"] = float(x["n_prefills"] == len(b["order"]) == len(x["first_token"]))
        lp = np.array([r["logprobs"] for r in x["first_token"]])
        lo[meth] = np.logaddexp(lp[:, 0], lp[:, 2]) - np.logaddexp(lp[:, 1], lp[:, 3])
    return lo


def main():
    if os.path.exists(DST):
        sys.exit(f"{DST} exists; results are never overwritten")
    rows = check_runs()
    if any(v < 1 for v in CHECKS.values()):
        sys.exit(f"STOP {[k for k, v in CHECKS.items() if v < 1]}")
    data = {"mme": load_mme(), "vstar": load_vstar(), "pope": load_pope()}
    res = {"pairs": {}, "table": {}, "flops": {}, "seconds_per_answer": {}, "log_odds": {}}
    for bench in BENCHES:
        b = data[bench]
        b["draws"] = draws_for(b)
        for x, y in (("peek", "div64"), ("peek", "nopeek"), ("nopeek", "div64"), ("vanilla", "peek"), ("vanilla", "div64")):
            res["pairs"][f"{bench} {x} - {y}"] = paired(b, x, y)
            for g, m in b["group"].items():
                res["pairs"][f"{bench} {x} - {y} {g}"] = paired(b, x, y, m)
        res["table"][bench] = {meth: {"accuracy": 100 * float(b["sc"][meth].mean()), **{g: 100 * float(b["sc"][meth][m].mean()) for g, m in b["group"].items()}}
                               for meth in METHODS}
        for meth in METHODS:
            exp, rid, _ = RUNS[(bench, meth)]
            res["seconds_per_answer"][f"{bench} {meth}"] = float(rows[f"{exp}-{rid}"]["s_per_answer"])
        if not DRY:
            res["flops"][bench] = kll_logs(bench, b)
    p = data["pope"]
    res["pope_splits"] = {meth: {s: {"accuracy": 100 * float(p["sc"][meth][[x == s for x in p["split"]]].mean()),
                                     "yes_ratio": float((p["said"][meth][[x == s for x in p["split"]]] == "yes").mean())}
                                 for s in ("adversarial", "popular", "random")} for meth in METHODS}
    for meth in METHODS:
        m = json.load(open(f"{run_dir('pope', meth)}/metrics.json", encoding="utf-8"))
        res["pope_splits"][meth]["f1_mean"] = m["f1_mean"]
        mm = json.load(open(f"{run_dir('mme', meth)}/metrics.json", encoding="utf-8"))
        res["table"]["mme"][meth]["perception"], res["table"]["mme"][meth]["cognition"] = mm["perception"], mm["cognition"]
        res["table"]["mme"][meth]["tasks"] = {**mm["perception_tasks"], **mm["cognition_tasks"]}
    if not DRY:
        lo = log_odds("pope", p)
        res["log_odds"] = {g: {"mean": {k: float(v[m].mean()) for k, v in lo.items()}, "peek - nopeek": paired(p, "peek", "nopeek", m, lo)}
                           for g, m in p["group"].items()}
    if any(v < 1 for v in CHECKS.values()):
        print("STOP", {k: v for k, v in CHECKS.items() if v < 1})
        sys.exit(1)
    P = res["pairs"]
    n1 = sum(P[f"{bench} peek - div64"][1] > 0 for bench in BENCHES)
    q3 = pooled([(data["mme"], data["mme"]["group"]["[position]"]), (data["vstar"], data["vstar"]["group"]["[relative_position]"])], "nopeek", "div64")
    q3p = pooled([(data["mme"], data["mme"]["group"]["[position]"]), (data["vstar"], data["vstar"]["group"]["[relative_position]"])], "peek", "div64")
    res["criteria"] = {
        "Q1": {3: "三个 benchmark 上 peek 都高于 DivPrune-64", 2: "两个", 1: "一个", 0: "没有一个"}[n1], "Q1_n": n1,
        "Q2_lean": "偏“是”在 POPE 上重现" if P["pope nopeek - div64 gt no"][2] < 0 else "偏“是”在 POPE 上没有重现",
        "Q2_removes": "借看在 POPE 上去掉偏“是”" if P["pope peek - nopeek gt no"][1] > 0 else "借看在 POPE 上没有去掉偏“是”",
        "Q2_cost": "“是”题上的代价重现" if P["pope peek - nopeek gt yes"][2] < 0 else "“是”题上的代价没有重现",
        "Q3": "原始编号在 MME、V* 的位置题上也有帮助" if q3[1] > 0 else "原始编号在 MME、V* 的位置题上的帮助没有检出",
        "Q3_pooled_nopeek_div64": q3, "Q3_pooled_peek_div64": q3p}
    res["checks"] = CHECKS
    os.makedirs(DST)
    json.dump(res, open(f"{DST}/0036.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    fmt = lambda x: "—" if x is None else f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"   # noqa: E731
    L = [f"# EXP-0036（计划 v1）：IDEA-0008 在 MME、V*、POPE 上{'（替身数据的空跑，不是结果）' if DRY else ''}", "", "## 核对", ""]
    L += [f"- {k}: {100 * v:.0f}%" for k, v in CHECKS.items()]
    L += ["", "## 准确率（逐题；MME 另报官方分数）", "", "| 配置 | MME 逐题 | MME 感知 / 认知 | position / existence / count / color（逐题） | V* | 相对位置 / 直接属性 | POPE | F1 均值 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    T = res["table"]
    for meth in METHODS:
        mt, vt = T["mme"][meth], T["vstar"][meth]
        L.append(f"| {NAMES[meth]} | {mt['accuracy']:.2f} | {mt['perception']:.2f} / {mt['cognition']:.2f} | "
                 + " / ".join(f"{mt[f'[{t}]']:.2f}" for t in ("position", "existence", "count", "color"))
                 + f" | {vt['accuracy']:.2f} | {vt['[relative_position]']:.2f} / {vt['[direct_attributes]']:.2f} | {T['pope'][meth]['accuracy']:.2f} | {res['pope_splits'][meth]['f1_mean']:.2f} |")
    L += ["", "MME 每个子任务的官方分数（acc + acc+）：" + "；".join(f"{NAMES[m]}：" + "，".join(f"{k} {v:.1f}" for k, v in T["mme"][m]["tasks"].items()) for m in METHODS)]
    L += ["", "POPE 每个划分的准确率 / 答“是”的比例：" + "；".join(f"{NAMES[m]}：" + "，".join(f"{s} {v['accuracy']:.2f} / {v['yes_ratio']:.3f}" for s, v in res["pope_splits"][m].items() if s != "f1_mean") for m in METHODS)]
    L += ["", "## 配对差（百分点，按图像重抽 1000 次）", "", "| 比较 | 差 [95% 区间]（题数） |", "| --- | --- |"] + [f"| {k} | {fmt(x)} |" for k, x in P.items() if x]
    if res["log_odds"]:
        L += ["", "POPE 第一个答案 token 的对数几率（peek / nopeek；peek − nopeek）：" + "；".join(
            f"{g} {x['mean']['peek']:+.3f} / {x['mean']['nopeek']:+.3f}，{fmt(x['peek - nopeek'])}" for g, x in res["log_odds"].items())]
    for bench, f in res["flops"].items():
        L += ["", f"计算量（{bench}，按 LLaMA-7B 的结构，相对 DivPrune-64；平均每题非图像 token {f['mean_non_image_tokens']:.1f}）：" + "，".join(f"{k} {v:.4f}" for k, v in f["relative_to_div64"].items())]
    L += ["", "每题秒数：" + "，".join(f"{k} {v:.4f}" for k, v in res["seconds_per_answer"].items())]
    C = res["criteria"]
    L += ["", "## 按计划写定的读法", "",
          f"- **Q1**（peek − DivPrune-64）：MME {fmt(P['mme peek - div64'])}，V* {fmt(P['vstar peek - div64'])}，POPE {fmt(P['pope peek - div64'])} → {C['Q1']}",
          f"- **Q2**（POPE，标准答案为“否”的题）：nopeek − DivPrune-64 {fmt(P['pope nopeek - div64 gt no'])} → {C['Q2_lean']}；peek − nopeek {fmt(P['pope peek - nopeek gt no'])} → {C['Q2_removes']}；"
          f"“是”题 peek − nopeek {fmt(P['pope peek - nopeek gt yes'])} → {C['Q2_cost']}",
          f"- **Q3**（MME position 与 V* 相对位置合并）：nopeek − DivPrune-64 {fmt(q3)} → {C['Q3']}；peek − DivPrune-64 {fmt(q3p)}"]
    open(f"{DST}/0036.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-5:]))


if __name__ == "__main__":
    main()
