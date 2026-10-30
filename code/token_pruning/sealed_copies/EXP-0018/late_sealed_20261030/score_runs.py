"""m4c (score_vqav2) vs lmms-eval / LLaVA-server definitions on every finished EXP-0018 run. Read-only.

  python -I -X utf8 score_runs.py
"""
import collections
import glob
import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location("verify_mod", "D:/6CJNM007/tmp/verify_comma_metric/verify.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
gts = {}
for line in open(v.ANN.format("test"), encoding="utf-8"):
    a = json.loads(line)
    gts[a["question_id"]] = a
for run in sorted(glob.glob("D:/6CJNM007/idea_rearch/code/output/EXP-0018/EXP-0018-R*")):
    path = f"{run}/answers.jsonl"
    preds = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []
    if sorted(p["question_id"] for p in preds) != sorted(gts):
        print(f"{os.path.basename(run)}: {len(preds)} answers, incomplete; skipped")
        continue
    m = json.load(open(f"{run}/metrics.json", encoding="utf-8")) if os.path.exists(f"{run}/metrics.json") else {}
    out = {}
    for name in ("m4c(score_vqav2)", "lmms-eval", "llava server"):
        f = v.DEFS[name]
        sc = {p["question_id"]: f(gts[p["question_id"]]["answers"], p["text"]) for p in preds}
        typ = collections.defaultdict(list)
        for q, s in sc.items():
            typ[gts[q]["answer_type"]].append(s)
        out[name] = (sc, 100 * sum(sc.values()) / len(sc), {t: round(100 * sum(x) / len(x), 4) for t, x in sorted(typ.items())})
    ns = sum(1 for p in preds if v.NOSPACE_COMMA.search(p["text"]) and not all(c.isdigit() for c in p["text"].replace(',', '')))
    print(f"{os.path.basename(run)}: metrics.json {m.get('accuracy')!r}; no-space ',' outputs {ns}")
    for name, (sc, acc, typ) in out.items():
        nd = sum(1 for q in sc if abs(sc[q] - out['m4c(score_vqav2)'][0][q]) > 1e-12)
        print(f"   {name:18s} {acc!r:22s} {typ}  per-question diffs vs m4c: {nd}")
