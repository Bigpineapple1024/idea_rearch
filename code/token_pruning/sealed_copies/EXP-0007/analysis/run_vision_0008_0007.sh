#!/usr/bin/env bash
# Vision batch: EXP-0008 (position-corrected FastV, short) first, then EXP-0007 (profile vs selection rule).
set -uo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
cd /d/6CJNM007/idea_rearch/code/token_pruning
python exp0002.py EXP-0008-R01:fastv64_scores:pope_img20_blank EXP-0008-R02:fastv64_scores:pope_img20 || exit 1
python analyze_score_prior.py EXP-0008 EXP-0008-R01 EXP-0008-R02 || exit 1
python exp0002.py EXP-0008-R03:fastv64_prior_kept:pope_img20 EXP-0008-R04:fastv192_prior_kept:pope_img20 \
  EXP-0008-R05:fastv64_prior:pope EXP-0008-R06:fastv64_prior:mme
python analyze_kept_tokens.py EXP-0008 EXP-0008-R03 EXP-0008-R04
echo "EXP-0008 done"
python exp0002.py EXP-0007-R01:random_svp64_s2021_layers:pope50 EXP-0007-R02:fastv_svp64_layers:pope50
python check_layer_profile.py /d/6CJNM007/idea_rearch/code/output/EXP-0007/EXP-0007-R01 576x3,82x4,32x9,18x16 || exit 1
python check_layer_profile.py /d/6CJNM007/idea_rearch/code/output/EXP-0007/EXP-0007-R02 576x3,82x4,32x9,18x16 || exit 1
python exp0002.py EXP-0007-R03:random_svp64_s2021:pope EXP-0007-R04:fastv_svp64:pope EXP-0007-R05:random_flat64_s2021:pope \
  EXP-0007-R06:fastv_flat64:pope EXP-0007-R07:random_svp64_s2021:mme EXP-0007-R08:fastv_svp64:mme \
  EXP-0007-R09:random_flat64_s2021:mme EXP-0007-R10:fastv_flat64:mme
echo "EXP-0007 done"
