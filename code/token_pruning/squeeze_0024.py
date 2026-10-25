"""SQZ-EXP-0024: exploratory squeeze of EXP-0024 (IDEA-0004's development runs), written after its sealed analysis was
seen. CPU only, from the sealed run outputs; no model calls.

  python squeeze_0024.py [out_dir]    # default code/output/EXP-0024/squeeze/{squeeze_0024.md,squeeze_0024.json}

Every split here was chosen after the analysis (exploratory, no multiplicity correction). Paired differences resample
images with replacement (1000 draws, seed 0) exactly as analyze_0024.py does; one pair is checked against the sealed
0024.json. Questions:
  1. lattice against pooled crop tokens (z16g - z16p);
  2. cover 16 (c16, the reallocation control): does its gain depend on the box hitting the named object, on the box
     sitting off the image's centre, on how few of the baseline's tokens were in the box; how many answers flip;
  3. the K = 1 lever by answer kind, and how much of K = 2's kept set K = 1 keeps;
  4. do the c16 and K = 1 gains fall on the same questions.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/6CJNM007/idea_rearch/code/token_pruning")
import analyze_0024 as an  # noqa: E402
from fastv_llava import _content_band  # noqa: E402

IMAGES = {"vqav2_dev2": ("D:/6CJNM007/data/llava_eval/vqav2_val/llava_vqav2_val_dev2.jsonl", "D:/6CJNM007/data/coco/val2014"),
          "gqa_val400": ("D:/6CJNM007/data/llava_eval/gqa/llava_gqa_val400.jsonl", "D:/6CJNM007/data/gqa/images_val400")}
METHODS = ("router", "z16p", "router_k1", "k1_z16p", "z16g", "p16full", "z16pctr", "c16")


def load_scores(bench, bi, b):
    sc, kept, zoom = {}, {}, {}
    for meth in METHODS:
        d = f"{an.OUT}/EXP-0024/EXP-0024-{an.RUNS[meth][bi]}"
        a = {}
        for line in open(f"{d}/answers.jsonl", encoding="utf-8"):
            x = json.loads(line)
            a[str(x["question_id"])] = x["text"]
        sc[meth] = an.recompute(bench, b, a)
        kept[meth] = json.load(open(f"{d}/kept_tokens.json"))["kept"]
        if os.path.exists(f"{d}/zoom.json"):
            zoom[meth] = json.load(open(f"{d}/zoom.json"))["records"]
    return sc, kept, zoom


def image_sizes(bench, b):
    from PIL import Image
    qfile, folder = IMAGES[bench]
    path = {str(q["question_id"]): q["image"] for q in map(json.loads, open(qfile, encoding="utf-8"))}
    cache, out = {}, []
    for q in b["order"]:
        p = path[q]
        if p not in cache:
            with Image.open(os.path.join(folder, p)) as im:
                cache[p] = im.size
        out.append(cache[p])
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else f"{an.OUT}/EXP-0024/squeeze"     # a test folder, or the sealed one
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} exists; results are never overwritten")
    sealed = json.load(open(f"{an.OUT}/EXP-0024/analysis/0024.json", encoding="utf-8"))
    res, checks = {}, {}
    for bi, bench in enumerate(an.BENCHES):
        b = an.load_bench(bench)
        n, routed = len(b["order"]), b["routed"]
        sc, kept, zoom = load_scores(bench, bi, b)
        for meth in ("z16p", "z16g"):
            sc[f"{meth}|unrouted"] = np.where(routed, sc["router"], sc[meth])
        by_img = [np.flatnonzero(b["img_of"] == k) for k in range(len(b["images"]))]
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by_img[k] for k in rng.integers(0, len(b["images"]), len(b["images"]))]) for _ in range(an.N_BOOT)]

        def paired(x, y, mask=None):
            d = sc[x] - sc[y]
            m = np.ones(n, bool) if mask is None else mask
            if m.sum() == 0:
                return None
            boot = [100 * d[idx][m[idx]].mean() for idx in draws if m[idx].any()]
            return [100 * float(d[m].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), int(m.sum())]

        r = res[bench] = {}
        got = paired("c16", "router")
        want = sealed["pairs"][f"{bench} c16 - router"]
        checks[f"{bench}: c16 - router equals the sealed analysis (same draws)"] = bool(np.allclose(got, want))
        # 1. lattice against pooled crop tokens
        r["z16g - z16p"] = paired("z16g", "z16p")
        r["z16g|unrouted - z16p|unrouted"] = paired("z16g|unrouted", "z16p|unrouted")
        # 2. cover 16
        recs = zoom["c16"]
        cov = np.array([bool(x["cover"]) for x in recs])
        checks[f"{bench}: c16 covers exactly the unrouted questions"] = bool((cov == ~routed).all())
        r["c16 - router [unrouted]"] = paired("c16", "router", cov)
        d = sc["c16"] - sc["router"]
        r["c16 flips on the unrouted questions"] = {"better": int((d[cov] > 0).sum()), "worse": int((d[cov] < 0).sum()),
                                                     "same": int((d[cov] == 0).sum())}
        base_in = np.array([x.get("baseline_in_box", -1) for x in recs])
        for lo, hi in ((0, 8), (9, 14), (15, 30)):
            m = cov & (base_in >= lo) & (base_in <= hi)
            r[f"c16 - router [baseline in box {lo}-{hi}]"] = paired("c16", "router", m)
        sizes = image_sizes(bench, b)
        off = np.zeros(n, bool)
        dist = np.full(n, np.nan)
        for i, x in enumerate(recs):
            if x["cover"]:
                r_lo, r_hi, c_lo, c_hi = _content_band(*sizes[i], an.GRID)
                r0, c0, r1, c1 = x["grid_box"]
                dist[i] = max(abs((r0 + r1) / 2 - (r_lo + r_hi) / 2), abs((c0 + c1) / 2 - (c_lo + c_hi) / 2))
        off = cov & (dist > 2)
        r["box centre distance from the content centre, patches (unrouted): median"] = float(np.nanmedian(dist))
        r["c16 - router [box near the centre, <= 2 patches]"] = paired("c16", "router", cov & ~off)
        r["c16 - router [box off the centre, > 2 patches]"] = paired("c16", "router", off)
        r["z16p - router [box near the centre]"] = paired("z16p", "router", cov & ~off)
        r["z16p - router [box off the centre]"] = paired("z16p", "router", off)
        changed = float(np.mean([x["changed"] for x in recs if x["cover"]]))
        r["c16 tokens changed: mean, share of the 30 kept"] = [changed, changed / 30]
        if bench == "vqav2_dev2":
            masks = an.topic_masks(b)
            hit, has = np.zeros(n, bool), np.zeros(n, bool)
            for i, (m, x) in enumerate(zip(masks, recs)):
                if m is not None and x["cover"]:
                    has[i], hit[i] = True, (m & an.box_mask(x["grid_box"])).sum() / m.sum() >= 0.5
            r["c16 - router [box hits the named object]"] = paired("c16", "router", has & hit)
            r["c16 - router [box misses the named object]"] = paired("c16", "router", has & ~hit)
            r["c16 - router [unrouted, no named COCO object]"] = paired("c16", "router", cov & ~has)
        # 3. the K = 1 lever
        for g, m in b["groups"].items():
            r[f"router_k1 - router {g}"] = paired("router_k1", "router", m)
        r["router_k1 - router [routed]"] = paired("router_k1", "router", routed)
        inc = [len(set(a) & set(k)) / len(a) for a, k in zip(kept["router"], kept["router_k1"])]
        r["share of K = 2's kept positions also kept at K = 1 (mean)"] = float(np.mean(inc))
        # 4. do the c16 and K = 1 gains fall on the same questions (unrouted)
        dk = sc["router_k1"] - sc["router"]
        up_c, up_k = cov & (d > 0), cov & (dk > 0)
        r["unrouted: P(K = 1 better | c16 better), P(K = 1 better)"] = [float(up_k[up_c].mean()) if up_c.any() else None,
                                                                       float(up_k[cov].mean())]
        r["unrouted: P(c16 better | K = 1 better), P(c16 better)"] = [float(up_c[up_k].mean()) if up_k.any() else None,
                                                                     float(up_c[cov].mean())]
    if not all(checks.values()):
        print("STOP", checks)
        sys.exit(1)
    os.makedirs(out_dir)
    json.dump({"checks": checks, "results": res}, open(f"{out_dir}/squeeze_0024.json", "w"), indent=2, ensure_ascii=False)

    def fmt(x):
        if isinstance(x, list) and len(x) == 4 and isinstance(x[3], int):
            return f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]（{x[3]}）"
        return json.dumps(x, ensure_ascii=False)

    L = ["# SQZ-EXP-0024：探索性深挖（看过封存的分析之后选的分组，未做多重比较校正）", "", "## 核对", ""]
    L += [f"- {k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()]
    for bench, r in res.items():
        L += ["", f"## {bench}", "", "| 项目 | 值 |", "| --- | --- |"]
        L += [f"| {k} | {fmt(x) if x is not None else '—'} |" for k, x in r.items()]
    open(f"{out_dir}/squeeze_0024.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
