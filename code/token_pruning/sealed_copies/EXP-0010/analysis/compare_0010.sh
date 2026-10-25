#!/usr/bin/env bash
# Paired bootstrap comparisons for EXP-0010 (plan v1): the block-stratified FastV-64 runs against FastV-64, random-64
# (seed 2021) and the position-corrected FastV-64 of EXP-0008, on the same questions. Writes
# code/output/EXP-0010/analysis/paired_bootstrap.txt.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
cd /d/6CJNM007/idea_rearch/code/token_pruning
O=/d/6CJNM007/idea_rearch/code/output
pair() {  # label, run a, run b
  echo "## $1"
  python bootstrap_ci.py "$O/${2%%-R*}/$2" "$O/${3%%-R*}/$3" 2>&1 | tail -3
  echo
}
one() {
  echo "## $1"
  python bootstrap_ci.py "$O/${2%%-R*}/$2" 2>&1 | tail -3
  echo
}
mkdir -p "$O/EXP-0010/analysis"
{
  for r in R03 R04 R05 R06; do one "EXP-0010-$r" "EXP-0010-$r"; done
  pair "POPE: fastv64_strat - fastv64_prior (rule 1-2)" EXP-0010-R03 EXP-0008-R05
  pair "POPE: fastv64_prior_strat - random64_s2021 (rule 3)" EXP-0010-R05 EXP-0002-R29
  pair "POPE: fastv64_strat - fastv64" EXP-0010-R03 EXP-0002-R11
  pair "POPE: fastv64_strat - random64_s2021" EXP-0010-R03 EXP-0002-R29
  pair "POPE: fastv64_prior_strat - fastv64" EXP-0010-R05 EXP-0002-R11
  pair "POPE: fastv64_prior_strat - fastv64_prior" EXP-0010-R05 EXP-0008-R05
  pair "POPE: fastv64_prior_strat - fastv64_strat" EXP-0010-R05 EXP-0010-R03
  pair "MME: fastv64_strat - fastv64_prior (rule 1-2)" EXP-0010-R04 EXP-0008-R06
  pair "MME: fastv64_prior_strat - random64_s2021 (rule 3)" EXP-0010-R06 EXP-0002-R28
  pair "MME: fastv64_strat - fastv64" EXP-0010-R04 EXP-0002-R10
  pair "MME: fastv64_strat - random64_s2021" EXP-0010-R04 EXP-0002-R28
  pair "MME: fastv64_prior_strat - fastv64" EXP-0010-R06 EXP-0002-R10
  pair "MME: fastv64_prior_strat - fastv64_prior" EXP-0010-R06 EXP-0008-R06
  pair "MME: fastv64_prior_strat - fastv64_strat" EXP-0010-R06 EXP-0010-R04
} > "$O/EXP-0010/analysis/paired_bootstrap.txt"
cat "$O/EXP-0010/analysis/paired_bootstrap.txt"
