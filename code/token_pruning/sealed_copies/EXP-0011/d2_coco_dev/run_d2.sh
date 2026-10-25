#!/usr/bin/env bash
# EXP-0011 D2 offline statistics (CPU only; tokenizer from the local HF cache).
#   bash run_d2.sh <run> <bench>
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /d/6CJNM007/tmp/vision_d2
python -I -X utf8 /d/6CJNM007/tmp/vision_d2/analyze_text_rows.py "$1" "$2"
echo "exit $?"
