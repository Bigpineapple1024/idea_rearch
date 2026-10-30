"""After the full official images.zip downloaded: the 398 images read from it must equal the ones fetched earlier by HTTP
range requests from the same file (sha256 in data/gqa/images_SHA256SUMS). Read-only; prints the result."""
import hashlib
import os
import zipfile

Z = "D:/6CJNM007/data/downloads/gqa/images.zip"
SUMS = "D:/6CJNM007/data/gqa/images_SHA256SUMS"
want = dict(reversed(l.split("  ")) for l in open(SUMS).read().split("\n") if l)
with zipfile.ZipFile(Z) as z:
    members = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".jpg")}
    print(f"{len(members)} jpg members in the full archive")
    same = sum(hashlib.sha256(z.read(members[im])).hexdigest() == h for im, h in want.items())
print(f"{same}/{len(want)} images identical to the range-fetched copies")
