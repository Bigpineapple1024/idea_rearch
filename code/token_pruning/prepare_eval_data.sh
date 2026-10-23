#!/usr/bin/env bash
# Download the EXP-0002 inputs into D:\6CJNM007 (DEC-0013) at pinned revisions.
#   source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp && bash prepare_eval_data.sh hf      # weights + MME images
#   source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp && bash prepare_eval_data.sh files   # everything else
# Weights go to HF_HOME (set by windows_env.sh); the rest to $DATA/downloads with a SHA256SUMS file.
set -euo pipefail
DATA=/d/6CJNM007/data
POPE_COMMIT=e3e39262c85a6a83f26cf5094022a782cb0df58d        # pinned in LLaVA docs/Evaluation.md
MME_TOOL_COMMIT=dd2950902889cd614d4edf606827240166d29381    # BradyFU/Awesome-Multimodal-Large-Language-Models, Evaluation branch
EVAL_ZIP_GDRIVE=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy           # eval.zip linked from LLaVA docs/Evaluation.md
GDOWN=/d/6CJNM007/miniforge3/Scripts/gdown.exe             # installed in the base env only
mkdir -p "$DATA/downloads"
cd "$DATA/downloads"

case "${1:?usage: prepare_eval_data.sh hf|files|extract}" in
  hf)
    python - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download

pins = {"liuhaotian/llava-v1.5-7b": "4481d270cc22fd5c4d1bb5df129622006ccd9234",
        "openai/clip-vit-large-patch14-336": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"}
for repo, sha in pins.items():
    # Fetch "main" so the cache records refs/main (LLaVA loads by repo name), then check it is the pinned commit.
    # One worker: huggingface_hub marks symlinks as supported before testing, so parallel workers on Windows
    # (no Developer Mode) race into os.symlink and fail with WinError 1314.
    path = snapshot_download(repo, revision="main", ignore_patterns=["*.h5", "*.msgpack", "*.onnx"], max_workers=1)
    assert path.replace("\\", "/").rstrip("/").endswith(sha), f"{repo}: main moved to {path}"
    print("ok", repo, sha, flush=True)
hf_hub_download("darkyarding/MME", "MME_Benchmark_release_version.zip", repo_type="dataset",
                revision="7056f44dac19de35e510b62d734bb2f7a6f64739", local_dir="D:/6CJNM007/data/downloads")
print("ok darkyarding/MME 7056f44dac19de35e510b62d734bb2f7a6f64739", flush=True)
EOF
    ;;
  files)
    for split in adversarial popular random; do
      curl -sSL --fail -o "coco_pope_$split.json" \
        "https://raw.githubusercontent.com/AoiDragon/POPE/$POPE_COMMIT/output/coco/coco_pope_$split.json"
    done
    curl -sSL --fail -o eval_tool.zip \
      "https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/$MME_TOOL_COMMIT/tools/eval_tool.zip"
    [[ -f eval.zip ]] || "$GDOWN" "$EVAL_ZIP_GDRIVE" -O eval.zip
    # images.cocodataset.org has no valid HTTPS certificate; the same S3 object (ETag 3925cc4d...-793) is reachable here.
    curl -sSL --fail -C - -o val2014.zip https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip
    [[ $(stat -c %s val2014.zip) == 6645013297 ]] || { echo "val2014.zip has the wrong size" >&2; exit 1; }
    ;;
  extract)
    # Layout: llava_eval/{pope,MME} from eval.zip; POPE annotations in pope/coco; MME subtasks directly under
    # MME/MME_Benchmark_release_version as LLaVA's convert_answer_to_mme.py expects (the HF zip nests them one
    # level deeper, in MME_Benchmark/); COCO images in coco/val2014 once the archive is complete.
    mkdir -p "$DATA/llava_eval/pope/coco" "$DATA/coco"
    unzip -q -o eval.zip -d "$DATA/llava_eval"     # whole archive: Git for Windows' unzip stops '*' at '/'
    cp coco_pope_*.json "$DATA/llava_eval/pope/coco/"
    unzip -q -o eval_tool.zip -d "$DATA/llava_eval/MME"
    mme="$DATA/llava_eval/MME/MME_Benchmark_release_version"
    if [[ ! -d "$mme/artwork" ]]; then
      unzip -q -o MME_Benchmark_release_version.zip -d "$DATA/llava_eval/MME"
      mv "$mme"/MME_Benchmark/* "$mme"/ && rmdir "$mme/MME_Benchmark"
    fi
    if [[ -d "$DATA/coco/val2014" ]]; then
      echo "coco/val2014 already extracted"
    elif [[ $(stat -c %s val2014.zip 2>/dev/null) == 6645013297 ]]; then
      unzip -q val2014.zip -d "$DATA/coco"
    else
      echo "val2014.zip not complete yet; run this stage again after the download"
    fi
    ;;
  *) echo "unknown stage $1" >&2; exit 1 ;;
esac
sha256sum -- *.zip *.json > SHA256SUMS 2>/dev/null || true
echo "done $1"
