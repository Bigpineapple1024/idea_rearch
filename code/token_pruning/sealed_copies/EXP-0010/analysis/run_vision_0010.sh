#!/usr/bin/env bash
# EXP-0010 (plan v1): FastV-64 with the ranking taken in turns from a 6x6 grid of image blocks, on the raw scores and
# after dividing out the position curve. Stops after the two subset diagnostics unless every kept set covers 30/36 blocks.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
cd /d/6CJNM007/idea_rearch/code/token_pruning
python exp0002.py EXP-0010-R01:fastv64_strat_kept:pope_img20 EXP-0010-R02:fastv64_prior_strat_kept:pope_img20 || exit 1
python analyze_kept_tokens.py EXP-0010 EXP-0010-R01 EXP-0010-R02 || exit 1
python - <<'EOF' || exit 1
import json, sys
stats = json.load(open("D:/6CJNM007/idea_rearch/code/output/EXP-0010/analysis/kept_summary.json"))
for s in stats:
    print(s["run_id"], "keep", s["keep"], "block coverage", s["block_coverage_mean"])
    if s["keep"] != 30 or abs(s["block_coverage_mean"] - 30 / 36) > 1e-9:
        sys.exit(f"{s['run_id']}: block coverage {s['block_coverage_mean']:.4f}, expected 30/36; stopping")
EOF
echo "coverage ok"
python exp0002.py EXP-0010-R03:fastv64_strat:pope EXP-0010-R04:fastv64_strat:mme EXP-0010-R05:fastv64_prior_strat:pope \
  EXP-0010-R06:fastv64_prior_strat:mme
echo "EXP-0010 done"
