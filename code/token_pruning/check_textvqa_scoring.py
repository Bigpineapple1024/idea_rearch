"""CPU check of the TextVQA scoring path (IDEA-0004, DEC-0036) on LLaVA's own LLaVA-1.5-13B predictions (eval.zip,
textvqa/answers/llava-v1.5-13b.jsonl; question ids are image ids, prompts carry the OCR line). They are remapped to
TextVQA's question ids by (image id, question) and scored with score_pope_mme.score_textvqa; LLaVA's eval_textvqa.py
scores the original file against the downloaded TextVQA_0.5.1_val.json. Both must agree (LLaVA reports 61.3 for 13B).

  python check_textvqa_scoring.py <work dir>
"""
import importlib.util
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = "D:/6CJNM007/data/llava_eval/textvqa"
ANN = "D:/6CJNM007/data/downloads/textvqa/TextVQA_0.5.1_val.json"
LLAVA = "D:/6CJNM007/idea_rearch/source/LLaVA"

work = sys.argv[1]
os.makedirs(work, exist_ok=True)
spec = importlib.util.spec_from_file_location("score_new", f"{HERE}/score_pope_mme.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
key = {(a["image_id"], a["question"].strip().lower()): a["question_id"] for a in map(json.loads, open(f"{EV}/llava_textvqa_val_annotations.jsonl", encoding="utf-8"))}
src = f"{EV}/answers/llava-v1.5-13b.jsonl"
preds = [json.loads(l) for l in open(src, encoding="utf-8")]
with open(f"{work}/llava13b_qid.jsonl", "w", encoding="utf-8") as f:
    for p in preds:
        qid = key[(p["question_id"], p["prompt"].split("\n")[0].strip().lower())]
        f.write(json.dumps({"question_id": qid, "text": p["text"]}) + "\n")
with open(f"{work}/score.log", "w") as log:
    ours = score.score_textvqa(f"{work}/llava13b_qid.jsonl", work, "ocr", log)["accuracy"]
r = subprocess.run([sys.executable, "-m", "llava.eval.eval_textvqa", "--annotation-file", ANN, "--result-file", src],
                   cwd=LLAVA, capture_output=True, text=True)
theirs = float(re.search(r"Accuracy: ([\d.]+)%", r.stdout).group(1))
print(f"13B predictions: {len(preds)}; ours {ours:.4f}, LLaVA's eval_textvqa.py {theirs:.2f} (paper 61.3)")
print("ok" if abs(ours - theirs) < 0.01 and abs(ours - 61.3) < 0.1 else "MISMATCH")   # 2026-10-30 note: exits 0 on MISMATCH too
