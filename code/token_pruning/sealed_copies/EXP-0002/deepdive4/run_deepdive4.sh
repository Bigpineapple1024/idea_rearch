#!/usr/bin/env bash
# EXP-0002 deepdive4: rescore the POPE runs with RePOPE (plan: /d/6CJNM007/tmp/deepdive4_plan.md, written first), then seal.
set -euo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export IDEA_REARCH_REPO=D:/6CJNM007/idea_rearch
D=/d/6CJNM007/idea_rearch/code/output/EXP-0002/deepdive4
python -I -X utf8 D:/6CJNM007/tmp/analyze_repope.py
cp /d/6CJNM007/tmp/deepdive4_plan.md "$D/plan.md"
cp /d/6CJNM007/data/downloads/repope/SHA256SUMS "$D/repope_inputs_SHA256SUMS"
cp /d/6CJNM007/data/downloads/repope/SOURCE.txt "$D/repope_inputs_SOURCE.txt"
bash /d/6CJNM007/tmp/seal_folder.sh EXP-0002 deepdive4 /d/6CJNM007/tmp/analyze_repope.py /d/6CJNM007/tmp/run_deepdive4.sh \
  /d/6CJNM007/tmp/get_repope.sh /d/6CJNM007/tmp/inspect_repope.py
