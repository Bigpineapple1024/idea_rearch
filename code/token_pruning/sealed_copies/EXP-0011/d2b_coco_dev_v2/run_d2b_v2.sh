#!/usr/bin/env bash
# EXP-0011 D2b offline statistics, version 2 (object AUC without the undefined cases), then the comparison with v1.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /d/6CJNM007/tmp/vision_d2
python -I -X utf8 /d/6CJNM007/tmp/vision_d2/analyze_layer_rows.py EXP-0011-R04 EXP-0011-R02 d2b_coco_dev_v2
echo "analysis exit $?"
python -I -X utf8 /d/6CJNM007/tmp/vision_d2/compare_d2b.py
echo "compare exit $?"
