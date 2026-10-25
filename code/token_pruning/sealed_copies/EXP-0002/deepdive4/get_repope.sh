#!/usr/bin/env bash
# Download the three RePOPE annotation files at a pinned commit (user-approved, 2026-10-24) into their own empty folder.
set -euo pipefail
C=bcf22e5690d587cce34e1728af3efdcfaf747d29
D=/d/6CJNM007/data/downloads/repope
test ! -e "$D"
mkdir -p "$D"
cd "$D"
for s in random popular adversarial; do
  /c/Windows/System32/curl.exe -sSL --fail -o "coco_repope_$s.json" "https://raw.githubusercontent.com/YanNeu/RePOPE/$C/annotations/coco_repope_$s.json"
done
echo "YanNeu/RePOPE@$C annotations/ (license not stated in the README)" > SOURCE.txt
sha256sum coco_repope_*.json > SHA256SUMS
ls -la
cat SHA256SUMS
head -c 300 coco_repope_random.json; echo
wc -l coco_repope_*.json
