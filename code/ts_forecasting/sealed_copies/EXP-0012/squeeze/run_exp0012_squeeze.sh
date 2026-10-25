#!/usr/bin/env bash
# EXP-0012 squeeze (exploratory): CPU only, a few threads so the EXP-0011 GPU run keeps its CPU.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh itrans
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /d/6CJNM007/tmp
python -I -X utf8 /d/6CJNM007/tmp/ts_level/squeeze_level_predictor.py
echo "exit $?"
