"""Stop a batch unless a _layers diagnostic run measured the intended per-layer visual-token profile.

  python check_layer_profile.py <run folder> 576x3,82x4,32x9,18x16

Compares metrics.json's mean_visual_per_layer (from --layer-token-log) with the expected value x count list and exits 1
on any difference, so a schedule that does not do what it should never reaches the full evaluations.
"""
import json
import sys

run_dir, expected = sys.argv[1], sys.argv[2]
want = [float(v) for part in expected.split(",") for v in [part.split("x")[0]] * int(part.split("x")[1])]
with open(f"{run_dir}/metrics.json") as f:
    got = json.load(f)["mean_visual_per_layer"]
ok = len(got) == len(want) and all(abs(g - w) < 1e-9 for g, w in zip(got, want))
print(("profile ok: " if ok else "profile MISMATCH: ") + json.dumps(got))
sys.exit(0 if ok else 1)
