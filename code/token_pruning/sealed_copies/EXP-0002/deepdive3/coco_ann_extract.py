"""Record the sha256 of annotations_trainval2014.zip, list it, and extract instances_val2014.json only."""
import hashlib
import json
import os
import sys
import zipfile

ZIP = "D:/6CJNM007/data/downloads/coco_annotations_2014/annotations_trainval2014.zip"
DEST = "D:/6CJNM007/data/coco/annotations"
MEMBER = "annotations/instances_val2014.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("zip", os.path.getsize(ZIP), sha256(ZIP))
if os.path.exists(DEST):
    sys.exit(f"{DEST} already exists")
with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        print(" ", info.filename, info.file_size)
    os.makedirs(DEST)
    target = os.path.join(DEST, "instances_val2014.json")
    with z.open(MEMBER) as src, open(target, "wb") as dst:     # fixed target name: no path from the archive is used
        for chunk in iter(lambda: src.read(1 << 20), b""):
            dst.write(chunk)
print("extracted", target, os.path.getsize(target), sha256(target))
with open(target, encoding="utf-8") as f:
    d = json.load(f)
print({k: len(v) for k, v in d.items() if isinstance(v, list)})
