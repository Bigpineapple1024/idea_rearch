"""analyze_0026.py（计划 v1.1）的空跑（不是结果）：EXP-0024 的参照运行与 EXP-0026 的运行都是合成的，只检查代码路径与核对项。

  python dryrun_0026.py                 # 全部 12 次运行
  python dryrun_0026.py --without-v11   # 没有 R09-R12（检查“未检验”的路径）

- 答案由标准答案按题号的固定规律构造（没有任何模型输出）：两个路由各一种规律；每个覆盖运行与 qvis 在不路由的题上换一种规律，
  被路由的题与同 K 的路由相同。
- 框：定位框放在内容带的左上角，边长 min(12, 内容带)；居中框边长相同、放在内容带中央。框内保留个数按“基线的框内个数
  （合成：8）+ n，上限 keep”。不围框（t16）：区域是固定的 16 个位置，全部保留。
- 全部写在 D:/6CJNM007/tmp/exp0026b/dry（或 dry_without_v11）之下。
"""
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter

WITHOUT = "--without-v11" in sys.argv
ROOT = "D:/6CJNM007/tmp/exp0026b/dry" + ("_without_v11" if WITHOUT else "")
sys.path.insert(0, "D:/6CJNM007/tmp/exp0026b/code")
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
sys.path.insert(2, "D:/6CJNM007/idea_rearch/source/LLaVA")
import analyze_0026 as an  # noqa: E402
from fastv_llava import _content_band  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_new", "D:/6CJNM007/tmp/exp0026b/code/score_pope_mme.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
pattern = {"router": lambda i: i % 3 != 0, "router_k1": lambda i: i % 5 != 0, "c16": lambda i: i % 7 != 0,
           "c16ctr": lambda i: i % 11 != 0, "k1_c16": lambda i: i % 13 != 0, "k1_c16ctr": lambda i: i % 17 != 0,
           "c8": lambda i: i % 19 != 0, "qvis": lambda i: i % 23 != 0, "t16": lambda i: i % 29 != 0}
TOP16 = list(range(300, 316))
rows = {"EXP-0024": [], "EXP-0026": []}
for bi, bench in enumerate(an.BENCHES):
    b = an.a24.load_bench(bench)
    n, routed = len(b["order"]), b["routed"]
    qfile, folder = an.QFILES[bench]
    path = {str(q["question_id"]): q["image"] for q in map(json.loads, open(qfile, encoding="utf-8"))}
    from PIL import Image
    sizes = {}
    for q in b["order"]:
        if path[q] not in sizes:
            with Image.open(os.path.join(folder, path[q])) as im:
                sizes[path[q]] = im.size
    if bench == "vqav2_dev2":
        gold = {q: Counter(b["gts"][q]["answers"]).most_common(1)[0][0] for q in b["order"]}
    else:
        gold = {q: b["gts"][q]["answer"] for q in b["order"]}
    for meth, (exp, ids) in an.RUNS.items():
        if WITHOUT and meth in an.OPTIONAL:
            continue
        rid = ids[bi]
        ref, keep, cov, ctr = an.SPEC[meth]
        d = f"{ROOT}/out/{exp}/{exp}-{rid}"
        os.makedirs(d)
        right = [pattern[meth](i) if (ref is None or not routed[i]) else pattern[ref](i) for i in range(n)]
        with open(f"{d}/answers.jsonl", "w", encoding="utf-8") as f:
            for i, q in enumerate(b["order"]):
                qid = int(q) if bench == "vqav2_dev2" else q
                f.write(json.dumps({"question_id": qid, "text": gold[q].capitalize() + "." if right[i] else "wrong"}) + "\n")
        with open(f"{d}/score.log", "w") as log:
            m = (score.score_vqav2(f"{d}/answers.jsonl", d, "dev2", log) if bench == "vqav2_dev2"
                 else score.score_gqa(f"{d}/answers.jsonl", d, log, split="val400", bench="gqa_val400"))
        kept, recs = [], []
        for i, q in enumerate(b["order"]):
            if ref is None or routed[i] or meth == "qvis":
                kept.append(list(range(keep)))
                if ref is not None and meth != "qvis":
                    recs.append({"index": i, "zoom": False, "cover": False, "skip": "routed"})
                continue
            w, h = sizes[path[q]]
            r_lo, r_hi, c_lo, c_hi = _content_band(w, h, 24)
            side = min(12, r_hi - r_lo + 1, c_hi - c_lo + 1)
            if ctr:
                cr, cc = (r_lo + r_hi) / 2, (c_lo + c_hi) / 2
                r0 = int(min(max(round(cr - (side - 1) / 2), r_lo), r_hi - side + 1))
                c0 = int(min(max(round(cc - (side - 1) / 2), c_lo), c_hi - side + 1))
            else:
                r0, c0 = r_lo, c_lo
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            rec = {"index": i, "zoom": False, "cover": True, "S": 0.5, "grid_box": list(box), "side": side, "span": 9,
                   "band": side, "top_inside": 6, "top1_inside": True, "baseline_in_box": 8}
            if meth == "t16":
                kept.append(sorted(TOP16 + [p for p in range(576) if p not in TOP16][:keep - 16]))
                rec.update(cover_top=TOP16, in_box=16, changed=10)
            else:
                inside = [r * 24 + c for r in range(box[0], box[2] + 1) for c in range(box[1], box[3] + 1)]
                outside = [p for p in range(576) if p not in set(inside)]
                in_box = min(8 + cov, keep, len(inside))
                kept.append(sorted(inside[:in_box] + outside[:keep - in_box]))
                rec.update(in_box=in_box, changed=in_box - 8)
            recs.append(rec)
        k = {"n_prefills": n, "keep": keep, "n_image_tokens": [576], "kept": kept}
        if meth != "qvis":                                     # like run_eval: no route decisions without --fastv-route
            k["routed"] = [bool(x) for x in routed]
        json.dump(k, open(f"{d}/kept_tokens.json", "w"))
        if ref is not None and meth != "qvis":
            json.dump({"n_prefills": n, "n_zoomed": 0, "n_covered": sum(r["cover"] for r in recs), "records": recs}, open(f"{d}/zoom.json", "w"))
        rows[exp].append([f"{exp}-{rid}", meth, bench, "completed", repr(m["accuracy"]), "1", "0.17", "14", "", "", ""])
for exp, rr in rows.items():
    with open(f"{ROOT}/out/{exp}/runs.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["run_id", "config", "bench", "status", "headline", "eval_s", "s_per_answer", "peak_gib", "start", "end", "sha256sums_sha256"])
        w.writerows(rr)
an.a24.OUT = f"{ROOT}/out"
an.main()
