#!/usr/bin/env bash
# EXP-0002 deepdive3: COCO boxes against the POPE misses (plan: /d/6CJNM007/tmp/deepdive3_plan.md, written first).
set -euo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export IDEA_REARCH_REPO=D:/6CJNM007/idea_rearch
python -I D:/6CJNM007/tmp/analyze_object_boxes.py
cp /d/6CJNM007/tmp/deepdive3_plan.md /d/6CJNM007/idea_rearch/code/output/EXP-0002/deepdive3/plan.md
