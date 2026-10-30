"""Count no-space commas (the only strings where m4c and the official normaliser group answers differently) in every
answers.jsonl under code/output. Read-only.

  python -I -X utf8 scan_outputs.py
"""
import glob
import json
import re

NOSPACE_NONDIGIT = re.compile(r"(?<!\d),(?! )|,(?![ \d])")   # a comma with no space after it, other than digit,digit
tot, tot_ns, files, hits = 0, 0, 0, []
for path in sorted(glob.glob("D:/6CJNM007/idea_rearch/code/output/**/answers.jsonl", recursive=True)):
    n = 0
    try:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            t = json.loads(line).get("text", "")
            n += 1
            if isinstance(t, str) and NOSPACE_NONDIGIT.search(t):
                hits.append((path, t))
    except (OSError, ValueError) as e:
        print(f"skip {path}: {e}")
        continue
    files += 1
    tot += n
print(f"{files} answers.jsonl files, {tot} outputs; outputs with a no-space non-numeric comma: {len(hits)}")
for h in hits[:20]:
    print("   ", h)
