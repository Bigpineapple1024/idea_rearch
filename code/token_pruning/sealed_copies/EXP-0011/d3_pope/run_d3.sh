#!/usr/bin/env bash
# EXP-0011 D3 analysis (CPU only).
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /d/6CJNM007/tmp/vision_d2
python -I -X utf8 /d/6CJNM007/tmp/vision_d2/analyze_d3.py
echo "exit $?"
