"""Label-free facts for the EXP-0022 plan, from the GQA question text only (no answers are on disk yet):
the share routed by the frozen rule, and how many routed questions are either/or choices ("... red or blue?")."""
import collections
import json
import re

ROUTE_START = ("is ", "are ", "does ", "do ", "did ", "can ", "could ", "was ", "were ", "has ", "have ", "had ", "will ",
               "would ", "should ", "shall ", "may ", "might ", "must ", "how many ")      # = run_eval.ROUTE_START
P = "D:/6CJNM007/data/llava_eval/gqa/llava_gqa_testdev_balanced.jsonl"
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
q = [r["text"].strip().split("\n")[0].strip().lower() for r in rows]
routed = [x.startswith(ROUTE_START) for x in q]
choice = [bool(re.search(r"\bor\b", x)) for x in q]
n = len(q)
print(f"{n} questions, {len({r['image'] for r in rows})} images")
print(f"routed {sum(routed)} ({sum(routed) / n:.3f})")
print(f"routed with ' or ' (either/or choice) {sum(r and c for r, c in zip(routed, choice))}; unrouted with ' or ' {sum((not r) and c for r, c in zip(routed, choice))}")
print("routed by start word:", collections.Counter(x.split(" ")[0] if not x.startswith("how many") else "how many" for x, r in zip(q, routed) if r).most_common())
print("unrouted by start word:", collections.Counter(x.split(" ")[0] for x, r in zip(q, routed) if not r).most_common(12))
print("prompt suffix check:", collections.Counter(r["text"].strip().split("\n")[1] if "\n" in r["text"].strip() else "<none>" for r in rows).most_common(3))
per_img = collections.Counter(r["image"] for r in rows)
print("questions per image: min", min(per_img.values()), "median", sorted(per_img.values())[len(per_img) // 2], "max", max(per_img.values()))
