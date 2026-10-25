#!/usr/bin/env bash
# Verify and extract the COCO 2014 instance annotations (user-approved download, 2026-10-24).
set -euo pipefail
source /d/6CJNM007/idea_rearch/code/env/windows_env.sh tp
python -I D:/6CJNM007/tmp/coco_ann_extract.py
cd /d/6CJNM007/data/downloads/coco_annotations_2014
sha256sum annotations_trainval2014.zip > SHA256SUMS
cd /d/6CJNM007/data/coco/annotations
sha256sum instances_val2014.json > SHA256SUMS
cat /d/6CJNM007/data/downloads/coco_annotations_2014/SHA256SUMS SHA256SUMS
