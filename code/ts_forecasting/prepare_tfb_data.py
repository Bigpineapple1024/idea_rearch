"""Extract TFB's ETTh1 files (the data DUET's paper uses) from forecasting.zip and record their checksums.

  python prepare_tfb_data.py [D:/6CJNM007/data/downloads/tfb/forecasting.zip] [D:/6CJNM007/data/tfb]

Only forecasting/ETTh1.csv and forecasting/FORECAST_META.csv are extracted (no macOS resource forks); the output
folder must not exist yet. SHA256SUMS is written next to the files.
"""
import hashlib
import os
import sys
import zipfile

archive = sys.argv[1] if len(sys.argv) > 1 else "D:/6CJNM007/data/downloads/tfb/forecasting.zip"
out = sys.argv[2] if len(sys.argv) > 2 else "D:/6CJNM007/data/tfb"
if os.path.exists(out):
    raise SystemExit(f"{out} already exists")
members = ["forecasting/ETTh1.csv", "forecasting/FORECAST_META.csv"]
with zipfile.ZipFile(archive) as z:
    for name in members:
        z.extract(name, out)
lines = []
for name in members:
    with open(os.path.join(out, name), "rb") as f:
        lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {name}\n")
with open(os.path.join(out, "SHA256SUMS"), "w") as f:
    f.writelines(lines)
print("".join(lines), end="")
