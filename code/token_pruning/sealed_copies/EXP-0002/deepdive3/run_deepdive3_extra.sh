#!/usr/bin/env bash
# deepdive3 follow-up checks, and the plan copy that the first run skipped (its final console print failed on cp932
# after object_boxes.md/json and positives.json were written).
set -euo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export IDEA_REARCH_REPO=D:/6CJNM007/idea_rearch
D=/d/6CJNM007/idea_rearch/code/output/EXP-0002/deepdive3
test ! -e "$D/plan.md"
cp /d/6CJNM007/tmp/deepdive3_plan.md "$D/plan.md"
python -I -X utf8 D:/6CJNM007/tmp/analyze_object_boxes_extra.py
ls -la "$D"
