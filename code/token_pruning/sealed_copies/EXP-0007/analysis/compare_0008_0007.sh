#!/usr/bin/env bash
# Paired bootstrap comparisons for EXP-0008 and EXP-0007 against each other and against EXP-0002 (same questions, same
# resampled items). Writes one text file per experiment into its analysis folder; run after the vision batch.
#   bash compare_0008_0007.sh 0008|0007
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
case "$1" in
0008)
  mkdir -p "$O/EXP-0008/analysis"
  {
    one "fastv64_prior POPE" EXP-0008-R05
    one "fastv64_prior MME" EXP-0008-R06
    pair "POPE: fastv64_prior - fastv64" EXP-0008-R05 EXP-0002-R11
    pair "POPE: fastv64_prior - random64_s2021" EXP-0008-R05 EXP-0002-R29
    pair "POPE: fastv64_prior - visionzip64" EXP-0008-R05 EXP-0002-R17
    pair "MME: fastv64_prior - fastv64" EXP-0008-R06 EXP-0002-R10
    pair "MME: fastv64_prior - random64_s2021" EXP-0008-R06 EXP-0002-R28
    pair "MME: fastv64_prior - visionzip64" EXP-0008-R06 EXP-0002-R35
  } > "$O/EXP-0008/analysis/paired_bootstrap.txt"
  cat "$O/EXP-0008/analysis/paired_bootstrap.txt"
  ;;
0007)
  mkdir -p "$O/EXP-0007/analysis"
  {
    for r in R03 R04 R05 R06 R07 R08 R09 R10; do one "EXP-0007-$r" "EXP-0007-$r"; done
    pair "POPE: random_svp64 - sparsevlm64" EXP-0007-R03 EXP-0002-R41
    pair "POPE: fastv_svp64 - random_svp64" EXP-0007-R04 EXP-0007-R03
    pair "POPE: random_svp64 - random_flat64" EXP-0007-R03 EXP-0007-R05
    pair "POPE: fastv_svp64 - fastv_flat64" EXP-0007-R04 EXP-0007-R06
    pair "POPE: random_flat64 - random64_s2021" EXP-0007-R05 EXP-0002-R29
    pair "POPE: fastv_flat64 - fastv64" EXP-0007-R06 EXP-0002-R11
    pair "MME: random_svp64 - sparsevlm64" EXP-0007-R07 EXP-0002-R40
    pair "MME: fastv_svp64 - random_svp64" EXP-0007-R08 EXP-0007-R07
    pair "MME: random_svp64 - random_flat64" EXP-0007-R07 EXP-0007-R09
    pair "MME: fastv_svp64 - fastv_flat64" EXP-0007-R08 EXP-0007-R10
    pair "MME: random_flat64 - random64_s2021" EXP-0007-R09 EXP-0002-R28
    pair "MME: fastv_flat64 - fastv64" EXP-0007-R10 EXP-0002-R10
  } > "$O/EXP-0007/analysis/paired_bootstrap.txt"
  cat "$O/EXP-0007/analysis/paired_bootstrap.txt"
  ;;
esac
