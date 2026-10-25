#!/usr/bin/env bash
# Exploratory pairs for SQZ-EXP-0007 (not in plan v1): FastV against random selection on the flat profile, the
# counterpart of Q2. Writes code/output/EXP-0007/analysis/paired_bootstrap_extra.txt.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
cd /d/6CJNM007/idea_rearch/code/token_pruning
O=/d/6CJNM007/idea_rearch/code/output/EXP-0007
pair() {  # label, run a, run b
  echo "## $1"
  python bootstrap_ci.py "$O/$2" "$O/$3" 2>&1 | tail -3
  echo
}
{
  pair "POPE: fastv_flat64 - random_flat64 (exploratory)" EXP-0007-R06 EXP-0007-R05
  pair "MME: fastv_flat64 - random_flat64 (exploratory)" EXP-0007-R10 EXP-0007-R09
} > "$O/analysis/paired_bootstrap_extra.txt"
cat "$O/analysis/paired_bootstrap_extra.txt"
