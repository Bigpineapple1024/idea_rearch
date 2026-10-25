"""CPU check of the V* scoring path (IDEA-0004, DEC-0036): letter parsing on typical answer forms, and score_vstar on
synthetic answer files: every label (must give 100%), every answer 'A' (must equal the share of label A), and
unparseable answers (0%, all counted as unparsed).

  python check_vstar_scoring.py <work dir>
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
work = sys.argv[1]
os.makedirs(work, exist_ok=True)
spec = importlib.util.spec_from_file_location("score_new", f"{HERE}/score_pope_mme.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
cases = {("A", 4): "A", ("(B)", 4): "B", ("C.", 4): "C", ("d", 4): "D", ("B) leather", 4): "B", ("The answer is B.", 4): "B",
         ("C", 2): None, ("left", 2): None, (" A\n", 2): "A"}
checks = {f"letter {t!r} with {n} options -> {want}": score.vstar_letter(t, n) == want for (t, n), want in cases.items()}
gts = [json.loads(l) for l in open("D:/6CJNM007/data/llava_eval/vstar/llava_vstar_test_annotations.jsonl", encoding="utf-8")]
for tag, answer in (("gold", lambda g: g["label"]), ("allA", lambda g: "A"), ("none", lambda g: "unclear")):
    d = f"{work}/{tag}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/answers.jsonl", "w") as f:
        for g in gts:
            f.write(json.dumps({"question_id": g["question_id"], "text": answer(g)}) + "\n")
    with open(f"{d}/score.log", "w") as log:
        m = score.score_vstar(f"{d}/answers.jsonl", d, log)
    share_a = 100 * sum(g["label"] == "A" for g in gts) / len(gts)
    want = {"gold": 100.0, "allA": share_a, "none": 0.0}[tag]
    checks[f"{tag}: accuracy {m['accuracy']:.2f} == {want:.2f}, unparsed {m['unparsed']}"] = (
        abs(m["accuracy"] - want) < 1e-9 and m["unparsed"] == (len(gts) if tag == "none" else 0))
for k, v in checks.items():
    print(("ok   " if v else "FAIL ") + k)
print(f"{sum(checks.values())}/{len(checks)} checks pass")
sys.exit(0 if all(checks.values()) else 1)
