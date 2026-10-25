#!/usr/bin/env bash
# EXP-0011 plan v1.2, D3: implementation check on the dev set (R05), then the variant (R06) and the padding-only control
# (R07) on POPE. Stops unless R05 passes its check.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
cd /d/6CJNM007/idea_rearch/code/token_pruning
python exp0002.py EXP-0011-R05:fastv64_objcos_strat_nopad_kept:coco_dev || exit 1
python -X utf8 D:/6CJNM007/tmp/check_0011_d3.py || { echo "R05 check failed"; exit 1; }
echo "R05 check ok"
python exp0002.py EXP-0011-R06:fastv64_objcos_strat_nopad_kept:pope EXP-0011-R07:fastv64_prior_strat_nopad_kept:pope
echo "EXP-0011 R05-R07 done, exit $?"
