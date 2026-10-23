"""Tables for the EXP-0002 record and squeeze report, computed from the completed run folders.

  python analyze_exp0002.py            # writes code/output/EXP-0002/analysis/{tables.md,summary.json}

Reads runs.tsv (all runs, including failed ones) and each completed run's metrics.json, answers.jsonl and
answers.jsonl.meta.json. Vanilla comes from R34 (POPE) and R05 (MME): sdpa and 4 DataLoader workers like every
phase-B run, with answers identical to R04 and R02. Intervals: bootstrap_ci.py's resampling (1000 draws, seed 0),
paired where two runs are compared on the same items. Random selection is also summarised across its seeds.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bootstrap_ci as bc  # noqa: E402

OUT = os.path.join(REPO, "code", "output", "EXP-0002")
VANILLA = {"pope": "EXP-0002-R34", "mme": "EXP-0002-R05"}
N_BOOT = 1000
# method A - method B at the same nominal budget (no multiple-comparison correction; descriptive only)
METHOD_PAIRS = [("fastv192", "visionzip192"), ("fastv192", "sparsevlm192"), ("visionzip192", "sparsevlm192"),
                ("visionzip128", "sparsevlm128"), ("visionzip64", "sparsevlm64"), ("random64_s2021", "visionzip64")]


def runs():
    with open(f"{OUT}/runs.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load(run_id):
    d = f"{OUT}/{run_id}"
    with open(f"{d}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{d}/answers.jsonl.meta.json") as f:
        meta = json.load(f)
    with open(f"{d}/answers.jsonl") as f:
        answers = [json.loads(line)["text"] for line in f]
    return d, metrics, meta, answers


def interval(dirs, bench, seed=0):
    loader, stat = (bc.load_pope, bc.pope_stat) if bench == "pope" else (bc.load_mme, bc.mme_stat)
    data = [loader(d) for d in dirs]
    sizes = {k: len(v) for k, v in data[0].items()}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        idx = {k: rng.integers(0, n, n) for k, n in sizes.items()}
        vals = [stat(x, idx) for x in data]
        draws.append(vals[0] - vals[1] if len(vals) == 2 else vals[0])
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def pope_correct(run_dir):
    items = bc.load_pope(run_dir)
    return np.concatenate([arr[:, 0] == arr[:, 1] for arr in items.values()])


def main():
    table = runs()
    done, layer_runs = {}, {}
    for r in table:
        if r["status"] != "completed" or r["config"] == "vanilla_eager":
            continue
        if r["bench"] == "pope50":                                     # unscored per-layer token diagnostic
            layer_runs[r["config"]] = r["run_id"]                      # the latest run of each config wins
            continue
        done.setdefault((r["config"], r["bench"]), r["run_id"])     # first completed run of each pair
    done[("vanilla", "pope")], done[("vanilla", "mme")] = VANILLA["pope"], VANILLA["mme"]
    van = {b: load(VANILLA[b]) for b in ("pope", "mme")}
    van_correct = pope_correct(van["pope"][0])
    summary = {"runs_total": len(table), "runs_failed": [r["run_id"] for r in table if r["status"] != "completed"],
               "configs": {}}
    for (config, bench), run_id in sorted(done.items()):
        d, metrics, meta, answers = load(run_id)
        row = summary["configs"].setdefault(config, {})
        entry = {"run_id": run_id, "seconds_per_answer": meta["seconds_per_answer"],
                 "eval_peak_gpu_mem_gib": meta["eval_peak_gpu_mem_gib"], "num_workers": meta["method"].get("num_workers", 0),
                 "answers_changed_vs_vanilla": int(sum(a != b for a, b in zip(answers, van[bench][3]))),
                 "ci95": interval([d], bench)}
        if bench == "pope":
            correct = pope_correct(d)
            entry.update({"f1_mean": metrics["f1_mean"], "accuracy_mean": metrics["accuracy_mean"],
                          "splits": metrics["splits"], "right_to_wrong": int(np.sum(van_correct & ~correct)),
                          "wrong_to_right": int(np.sum(~van_correct & correct))})
        else:
            entry.update({"perception": metrics["perception"], "cognition": metrics["cognition"],
                          "total": metrics["perception"] + metrics["cognition"], "tasks": {**metrics["perception_tasks"],
                                                                                           **metrics["cognition_tasks"]}})
        if config != "vanilla":
            entry["diff_vs_vanilla_ci95"] = interval([d, van[bench][0]], bench)
        row[bench] = entry
    for budget in (192, 128, 64):                       # H1: attention ranking vs random at the same budget
        for bench in ("pope", "mme"):
            fastv = done.get((f"fastv{budget}", bench))
            seeds = [s for s in (2021, 2022, 2023) if (f"random{budget}_s{s}", bench) in done]
            for s in seeds:
                if fastv:
                    summary.setdefault("fastv_minus_random", {})[f"{budget}_{bench}_s{s}"] = interval(
                        [f"{OUT}/{fastv}", f"{OUT}/{done[(f'random{budget}_s{s}', bench)]}"], bench)
            if len(seeds) > 1:
                key = "f1_mean" if bench == "pope" else "perception"
                vals = [summary["configs"][f"random{budget}_s{s}"][bench][key] for s in seeds]
                summary.setdefault("random_across_seeds", {})[f"{budget}_{bench}"] = {
                    "seeds": seeds, "values": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    for config, run_id in layer_runs.items():
        with open(f"{OUT}/{run_id}/metrics.json") as f:
            m = json.load(f)
        if m.get("n_complete") != m["n_prefills"]:      # R43-R45 missed SparseVLM's 3 keyword-called layers
            continue
        per_layer = m["mean_visual_per_layer"]
        runs_of = []                                    # run-length encoding of the per-layer means
        for x in per_layer:
            if runs_of and abs(runs_of[-1][0] - x) < 1e-9:
                runs_of[-1][1] += 1
            else:
                runs_of.append([x, 1])
        summary.setdefault("layer_tokens", {})[config] = {
            "run_id": run_id, "n_prefills": m["n_prefills"], "mean_visual_over_layers": m["mean_visual_over_layers"],
            "mean_visual_per_layer": per_layer, "segments": runs_of}
    for a, b in METHOD_PAIRS:                           # exploratory, added after the record's first draft
        for bench in ("pope", "mme"):
            if (a, bench) in done and (b, bench) in done:
                key = "f1_mean" if bench == "pope" else "perception"
                summary.setdefault("method_pairs", {})[f"{a}-{b}_{bench}"] = {
                    "diff": summary["configs"][a][bench][key] - summary["configs"][b][bench][key],
                    "ci95": interval([f"{OUT}/{done[(a, bench)]}", f"{OUT}/{done[(b, bench)]}"], bench)}
    os.makedirs(f"{OUT}/analysis", exist_ok=True)
    with open(f"{OUT}/analysis/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_markdown(summary)


def _order(name):
    methods = ["vanilla", "fastv", "random", "visionzip", "sparsevlm"]
    method = next(m for m in methods if name.startswith(m))
    rest = name[len(method):]
    budget = int(rest.split("_")[0]) if rest else 576
    return methods.index(method), -budget, rest


def write_markdown(s):
    c = {k: s["configs"][k] for k in sorted(s["configs"], key=_order)}
    lines = ["| 配置 | POPE F1 均值 [95% 区间] | POPE 准确率 | MME-P [95% 区间] | MME-C | MME P+C | 秒/题（POPE / MME） | 峰值显存 GiB |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in c.items():
        p, m = row.get("pope"), row.get("mme")
        fmt_ci = lambda e, k: f"{e[k]:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]" if e else "NA"
        lines.append(f"| {name} | {fmt_ci(p, 'f1_mean')} | {p['accuracy_mean']:.2f} | " if p else f"| {name} | NA | NA | ")
        lines[-1] += (f"{fmt_ci(m, 'perception')} | {m['cognition']:.2f} | {m['total']:.2f} | " if m else "NA | NA | NA | ")
        lines[-1] += f"{p['seconds_per_answer'] if p else 'NA'} / {m['seconds_per_answer'] if m else 'NA'} | "
        lines[-1] += f"{max(e['eval_peak_gpu_mem_gib'] for e in (p, m) if e)} |"
    lines += ["", "| 配置 | POPE 改变的答案（对→错 / 错→对） | POPE 相对 vanilla [95% 区间] | MME 改变的答案 | MME-P 相对 vanilla [95% 区间] |",
              "| --- | --- | --- | --- | --- |"]
    for name, row in c.items():
        if name == "vanilla":
            continue
        p, m = row.get("pope"), row.get("mme")
        lines.append(f"| {name} | " + (f"{p['answers_changed_vs_vanilla']}（{p['right_to_wrong']} / {p['wrong_to_right']}） | "
                                        f"[{p['diff_vs_vanilla_ci95'][0]:.2f}, {p['diff_vs_vanilla_ci95'][1]:.2f}] | " if p else "NA | NA | ")
                     + (f"{m['answers_changed_vs_vanilla']} | [{m['diff_vs_vanilla_ci95'][0]:.2f}, {m['diff_vs_vanilla_ci95'][1]:.2f}] |"
                        if m else "NA | NA |"))
    tasks = list(c["vanilla"]["mme"]["tasks"])
    c = {k: v for k, v in c.items() if "mme" in v or "pope" in v}
    lines += ["", "| MME 子任务 | vanilla | " + " | ".join(n for n in c if n != "vanilla" and "mme" in c[n]) + " |",
              "| --- | --- | " + " | ".join("---" for n in c if n != "vanilla" and "mme" in c[n]) + " |"]
    for t in tasks:
        v = c["vanilla"]["mme"]["tasks"][t]
        lines.append(f"| {t} | {v:.1f} | " + " | ".join(f"{c[n]['mme']['tasks'][t] - v:+.1f}" for n in c
                                                          if n != "vanilla" and "mme" in c[n]) + " |")
    splits = ["adversarial", "popular", "random"]
    lines += ["", "| 配置 | " + " | ".join(f"{sp} F1 / yes 比例" for sp in splits) + " |", "| --- | --- | --- | --- |"]
    for name, row in c.items():
        if "pope" in row:
            lines.append(f"| {name} | " + " | ".join(f"{100 * row['pope']['splits'][sp]['f1']:.2f} / "
                                                      f"{100 * row['pope']['splits'][sp]['yes_ratio']:.1f}%" for sp in splits) + " |")
    if "fastv_minus_random" in s:
        lines += ["", "| 预算 × 评测 × 随机种子 | FastV − 随机，配对 bootstrap 95% 区间 |", "| --- | --- |"]
        lines += [f"| {k} | [{v[0]:.2f}, {v[1]:.2f}] |" for k, v in s["fastv_minus_random"].items()]
    if "random_across_seeds" in s:
        lines += ["", "| 预算 × 评测 | 随机选择各种子 | 均值 ± 标准差 |", "| --- | --- | --- |"]
        lines += [f"| {k} | {' / '.join(f'{x:.2f}' for x in v['values'])}（种子 {', '.join(map(str, v['seeds']))}） | "
                  f"{v['mean']:.2f} ± {v['sd']:.2f} |" for k, v in s["random_across_seeds"].items()]
    if "layer_tokens" in s:
        lines += ["", "| 诊断配置（50 道 POPE） | 运行 | 预填充次数 | 32 层平均视觉 token | 逐层视觉 token（值×层数，从第 0 层起） |",
                  "| --- | --- | --- | --- | --- |"]
        lines += [f"| {k} | {v['run_id']} | {v['n_prefills']} | {v['mean_visual_over_layers']:.2f} | "
                  + ", ".join(f"{x:g}×{n}" for x, n in v["segments"]) + " |" for k, v in s["layer_tokens"].items()]
    if "method_pairs" in s:
        lines += ["", "| 方法间比较（A − B × 评测，探索性） | 差值 | 配对 bootstrap 95% 区间 |", "| --- | --- | --- |"]
        lines += [f"| {k} | {v['diff']:+.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] |" for k, v in s["method_pairs"].items()]
    with open(f"{OUT}/analysis/tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
