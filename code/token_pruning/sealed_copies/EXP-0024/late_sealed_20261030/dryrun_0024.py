"""analyze_0024.py 的空跑（不是结果）：所有运行文件夹都是合成的，在任何 EXP-0024 运行之前只检查代码路径与核对项。

- 答案由标准答案按题号的固定规律构造（没有任何模型输出）：两个路由运行（K = 2、K = 1）各一种规律；每个放大或对照运行在
  “被放大”的题上换一种规律，其余题与同 K 的路由相同。
- “被放大”：放大全部题的配置是题号不是 150 的倍数的题；c16 是其中不路由的题。
- 放大记录、保留 token 记录（含 zoom_tokens）、runs.tsv 按 run_eval / exp0002 的格式写；计分用 score_pope_mme 的函数。
- 全部写在 D:/6CJNM007/tmp/exp0024/dry 之下。
"""
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter

ROOT = "D:/6CJNM007/tmp/exp0024/dry"
sys.path.insert(0, "D:/6CJNM007/tmp/exp0024/code")
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
sys.path.insert(2, "D:/6CJNM007/idea_rearch/source/LLaVA")
import analyze_0024 as an  # noqa: E402

_spec = importlib.util.spec_from_file_location("score_new", "D:/6CJNM007/tmp/exp0024/code/score_pope_mme.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
out = f"{ROOT}/out/EXP-0024"
BOX, CTR = [4, 4, 15, 15], [6, 6, 17, 17]
INSIDE = [r * 24 + c for r in range(4, 16) for c in range(4, 16)]
rows = []
pattern = {"router": lambda i: i % 3 != 0, "router_k1": lambda i: i % 5 != 0}
for k, m in enumerate(m for m in an.RUNS if m not in pattern):
    pattern[m] = (lambda mod: (lambda i: i % mod != 0))(7 + k)     # moduli distinct from the routers' 3 and 5
for bi, bench in enumerate(an.BENCHES):
    b = an.load_bench(bench)
    n = len(b["order"])
    routed = b["routed"]
    if bench == "vqav2_dev2":
        gold = {q: Counter(b["gts"][q]["answers"]).most_common(1)[0][0] for q in b["order"]}
    else:
        gold = {q: b["gts"][q]["answer"] for q in b["order"]}
    for meth, ids in an.RUNS.items():
        rid = ids[bi]
        ref, nz, keep, every = an.SPEC[meth]
        d = f"{out}/EXP-0024-{rid}"
        os.makedirs(d)
        if ref is None:
            zoomed = [False] * n
        else:
            zoomed = [i % 150 != 0 and (every or not routed[i]) for i in range(n)]
        base = pattern[ref] if ref else pattern[meth]
        right = [pattern[meth](i) if zoomed[i] else base(i) for i in range(n)]
        with open(f"{d}/answers.jsonl", "w") as f:
            for i, q in enumerate(b["order"]):
                qid = int(q) if bench == "vqav2_dev2" else q          # the loader writes the question file's own id type
                f.write(json.dumps({"question_id": qid, "text": gold[q].capitalize() + "." if right[i] else "wrong"}) + "\n")
        with open(f"{d}/score.log", "w") as log:
            m = (score.score_vqav2(f"{d}/answers.jsonl", d, "dev2", log) if bench == "vqav2_dev2"
                 else score.score_gqa(f"{d}/answers.jsonl", d, log, split="val400", bench="gqa_val400"))
        kept, recs, ztok = [], [], []
        for i in range(n):
            if ref is not None and zoomed[i] and not nz:            # cover: 20 inside the box, 10 outside
                kept.append(INSIDE[:20] + list(range(10)))
            else:
                kept.append(list(range(keep - nz if zoomed[i] else keep)))
            ztok.append(nz if zoomed[i] else 0)
            if ref is None:
                continue
            rec = {"index": i, "zoom": bool(zoomed[i] and nz), "cover": bool(zoomed[i] and not nz)}
            if routed[i] and not every:
                rec["skip"] = "routed"
            elif not zoomed[i]:
                rec.update(S=0.0, skip="S below s_min")
            else:
                rec.update(S=0.5, grid_box=CTR if meth == "z16pctr" else BOX, side=12, span=10 + i % 9, band=18,
                           top_inside=6, top1_inside=True)
                if meth == "p16full":
                    rec["source"] = "full"
                if not nz:
                    rec.update(baseline_in_box=4, in_box=20, changed=16)
            recs.append(rec)
        k = {"n_prefills": n, "keep": keep, "n_image_tokens": [576], "kept": kept, "routed": [bool(x) for x in routed]}
        if ref is not None:
            k["zoom_tokens"] = ztok
            json.dump({"n_prefills": n, "n_zoomed": sum(r["zoom"] for r in recs), "n_covered": sum(r["cover"] for r in recs),
                       "records": recs}, open(f"{d}/zoom.json", "w"))
        json.dump(k, open(f"{d}/kept_tokens.json", "w"))
        rows.append([f"EXP-0024-{rid}", meth, bench, "completed", repr(m["accuracy"]), "1", "0.17", "14", "", "", ""])
with open(f"{out}/runs.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["run_id", "config", "bench", "status", "headline", "eval_s", "s_per_answer", "peak_gib", "start", "end", "sha256sums_sha256"])
    w.writerows(rows)
an.OUT = f"{ROOT}/out"
an.main()
