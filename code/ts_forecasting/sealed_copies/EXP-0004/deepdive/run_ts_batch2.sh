#!/usr/bin/env bash
# Time-series GPU batch, run with code/ts-forecasting checked out:
#   EXP-0009 R01-R12 (ETTh1 validation predictions) -> its analysis -> EXP-0004 phase B (R21-R56, 36 trainings)
#   -> phase B tables and test-period level/shape splits -> EXP-0009 R13-R48 (validation predictions, plan v1.1)
#   -> validation-period splits.
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh itrans
cd /d/6CJNM007/idea_rearch/code/ts_forecasting
spec() {  # exp, first run number, method, datasets...
  local exp=$1 n=$2 method=$3; shift 3
  for ds in "$@"; do for pl in 96 192 336 720; do for seed in 2021 2022 2023; do
    printf '%s-R%02d:%s:%s:%s:%s\n' "$exp" "$n" "$method" "$ds" "$pl" "$seed"; n=$((n + 1))
  done; done; done
}
mapfile -t val1 < <(spec EXP-0009 1 itransformer_val ETTh1)
python exp_ts.py EXP-0009 "${val1[@]}" || exit 1
python analyze_level_shape.py ETTh1 EXP-0009 val || exit 1
echo "EXP-0009 ETTh1 done"
mapfile -t phaseb < <(spec EXP-0004 21 itransformer ETTh2 ETTm1 ETTm2)
python exp_ts.py EXP-0004 "${phaseb[@]}"
echo "phase B exit $?"
python analyze_exp0004.py
echo "analysis exit $?"
for ds in ETTh2 ETTm1 ETTm2; do python analyze_level_shape.py "$ds" EXP-0004; done
mapfile -t val2 < <(spec EXP-0009 13 itransformer_val ETTh2 ETTm1 ETTm2)
python exp_ts.py EXP-0009 "${val2[@]}"
echo "EXP-0009 phase B datasets exit $?"
for ds in ETTh2 ETTm1 ETTm2; do python analyze_level_shape.py "$ds" EXP-0009 val; done
echo "ts batch 2 done"
