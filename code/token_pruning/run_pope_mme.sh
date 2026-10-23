#!/usr/bin/env bash
# POPE + MME for vanilla LLaVA-1.5-7B, VisionZip and FastV at the 192/128/64 budgets.
# Run from the LLaVA repository root (needs ./playground/data/eval prepared as in docs/Evaluation.md).
#   LLAVA_ROOT=~/LLaVA VISIONZIP_ROOT=~/VisionZip bash run_pope_mme.sh            # run
#   DRY_RUN=1 LLAVA_ROOT=... VISIONZIP_ROOT=... bash run_pope_mme.sh              # print commands only
# FastV keep counts give the same mean visual tokens over 32 layers as the budget with K=2:
#   keep = round((budget*32 - 2*576) / 30)  ->  192:166, 128:98, 64:30
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LLAVA_ROOT:?set LLAVA_ROOT to the haotian-liu/LLaVA checkout}"
: "${VISIONZIP_ROOT:?set VISIONZIP_ROOT to the dvlab-research/VisionZip checkout}"
MODEL="${MODEL:-liuhaotian/llava-v1.5-7b}"
BENCHES="${BENCHES:-pope mme}"
export VISIONZIP_ROOT
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

run() { if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "+ $*"; else "$@"; fi; }

configs=(
  "vanilla|--method vanilla"
  "visionzip192|--method visionzip --dominant 162 --contextual 30"
  "visionzip128|--method visionzip --dominant 108 --contextual 20"
  "visionzip64|--method visionzip --dominant 54 --contextual 10"
  "fastv192|--method fastv --fastv-k 2 --fastv-keep 166"
  "fastv128|--method fastv --fastv-k 2 --fastv-keep 98"
  "fastv64|--method fastv --fastv-k 2 --fastv-keep 30"
)

cd "$LLAVA_ROOT"
for cfg in "${configs[@]}"; do
  tag="${cfg%%|*}"; method_args="${cfg#*|}"
  for bench in $BENCHES; do
    case "$bench" in
      pope)
        q=./playground/data/eval/pope/llava_pope_test.jsonl
        img=./playground/data/eval/pope/val2014
        ans=./playground/data/eval/pope/answers/$tag.jsonl ;;
      mme)
        q=./playground/data/eval/MME/llava_mme.jsonl
        img=./playground/data/eval/MME/MME_Benchmark_release_version
        ans=./playground/data/eval/MME/answers/$tag.jsonl ;;
      *) echo "unknown bench $bench" >&2; exit 1 ;;
    esac
    # shellcheck disable=SC2086
    run python "$HERE/run_eval.py" $method_args -- --model-path "$MODEL" --question-file "$q" \
      --image-folder "$img" --answers-file "$ans" --temperature 0 --conv-mode vicuna_v1
    if [[ "$bench" == pope ]]; then
      run python llava/eval/eval_pope.py --annotation-dir ./playground/data/eval/pope/coco \
        --question-file "$q" --result-file "$ans"
    else
      run bash -c "cd ./playground/data/eval/MME && python convert_answer_to_mme.py --experiment $tag \
        && cd eval_tool && python calculation.py --results_dir answers/$tag"
    fi
  done
done
