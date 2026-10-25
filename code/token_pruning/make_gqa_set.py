"""EXP-0022: GQA testdev-balanced annotations and images from the official GQA archives.

  python make_gqa_set.py <questions1.2.zip> <images.zip | https URL of images.zip>

Writes (never overwrites):
  D:/6CJNM007/data/llava_eval/gqa/llava_gqa_testdev_balanced_annotations.jsonl   one line per question: question_id, image,
      answer, structural / semantic / detailed type (from testdev_balanced_questions.json)
  D:/6CJNM007/data/gqa/images/<imageId>.jpg   the 398 images LLaVA's question file uses, and images_SHA256SUMS next to them

The images come from the official images.zip, either a local copy or the same file read with HTTP range requests (only
the central directory and the 398 members are fetched); zipfile checks every member's CRC-32 on read either way.
Checks (any failure stops): the answers cover LLaVA's 12578 question ids exactly, with the same image and question text.
"""
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

LLAVA_Q = "D:/6CJNM007/data/llava_eval/gqa/llava_gqa_testdev_balanced.jsonl"
ANN = "D:/6CJNM007/data/llava_eval/gqa/llava_gqa_testdev_balanced_annotations.jsonl"
IMG_DIR = "D:/6CJNM007/data/gqa/images"


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
        if len(data) != n:
            raise IOError(f"range read returned {len(data)} of {n} bytes")
        self.pos += n
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    qzip, izip = sys.argv[1], sys.argv[2]
    for path in (ANN, IMG_DIR):
        if os.path.exists(path):
            sys.exit(f"{path} exists; never overwritten")
    llava = [json.loads(l) for l in open(LLAVA_Q, encoding="utf-8")]
    with zipfile.ZipFile(qzip) as z:
        name = [n for n in z.namelist() if n.endswith("testdev_balanced_questions.json")]
        if len(name) != 1:
            sys.exit(f"testdev_balanced_questions.json not found once in {qzip}: {name}")
        official = json.loads(z.read(name[0]))
    ids = [str(r["question_id"]) for r in llava]
    if sorted(ids) != sorted(official):
        sys.exit(f"question ids differ: {len(set(ids) - set(official))} only in LLaVA's file, {len(set(official) - set(ids))} only in the official one")
    bad_img = [i for i, r in zip(ids, llava) if r["image"] != f"{official[i]['imageId']}.jpg"]
    bad_txt = [i for i, r in zip(ids, llava) if r["text"].split("\n")[0].strip() != official[i]["question"].strip()]
    if bad_img or bad_txt:
        sys.exit(f"{len(bad_img)} image and {len(bad_txt)} question-text mismatches, e.g. {(bad_img + bad_txt)[:3]}")
    images = sorted({r["image"] for r in llava})
    src = HTTPRangeFile(izip) if izip.startswith("https://") else open(izip, "rb")
    with zipfile.ZipFile(src) as z:
        members = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".jpg")}
        missing = [im for im in images if im not in members]
        if missing:
            sys.exit(f"{len(missing)} images not in the archive, e.g. {missing[:3]}")
        os.makedirs(IMG_DIR)
        sums = []
        for k, im in enumerate(images):
            data = z.read(members[im])                      # raises BadZipFile on a CRC mismatch
            with open(f"{IMG_DIR}/{im}", "wb") as f:
                f.write(data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  {im}")
            if k % 50 == 0:
                print(f"{k + 1}/{len(images)} images", flush=True)
    open(f"{os.path.dirname(IMG_DIR)}/images_SHA256SUMS", "w", newline="\n").write("\n".join(sums) + "\n")
    with open(ANN, "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            o = official[i]
            f.write(json.dumps({"question_id": i, "image": f"{o['imageId']}.jpg", "answer": o["answer"],
                                "structural": o["types"]["structural"], "semantic": o["types"]["semantic"],
                                "detailed": o["types"]["detailed"]}) + "\n")
    print(f"{len(ids)} questions, {len(images)} images; annotations and images written")
    print("source of images:", izip)


if __name__ == "__main__":
    main()
