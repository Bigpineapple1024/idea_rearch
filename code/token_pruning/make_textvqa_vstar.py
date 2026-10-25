"""IDEA-0004 confirmation data (DEC-0036: downloaded 2026-10-25 with the user's permission), prepared once:

  python -I make_textvqa_vstar.py

TextVQA v0.5.1 val: the annotation file and the official train + val image zip are checked against the sizes and md5
the server reported; only the 3,166 val images are extracted, into data/textvqa/train_images (LLaVA's folder name), with
a sha256 per image, and each must open. LLaVA's question file (eval.zip; with the OCR-token line; its question_id is the
image id) is matched to the annotations by (image id, question) and rewritten with TextVQA's own question ids, once with
the OCR line (llava_textvqa_val_ocr_qid.jsonl) and once without it (llava_textvqa_val_noocr_qid.jsonl); per-question
annotations go to llava_textvqa_val_annotations.jsonl.
V* Bench (data/downloads/vstar_bench, revision in vstar_bench.revision.json): the 191 questions, already in LLaVA's
format, are copied into data/llava_eval/vstar with their label, category, option count, target boxes and image size.
Never overwrites; checksum files are written next to the new files.
"""
import hashlib
import json
import os
import re
import sys
import zipfile

DL = "D:/6CJNM007/data/downloads"
EV = "D:/6CJNM007/data/llava_eval"
IMG = "D:/6CJNM007/data/textvqa/train_images"
EXPECT = {"TextVQA_0.5.1_val.json": (3116162, "90f839369ad2177b4b76610438603a6a"),
          "train_val_images.zip": (7072297970, "d12dd8098899044e4ae1af34db7ecfef")}
SUFFIX = "\nAnswer the question using a single word or phrase."


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_sums(folder, names, out):
    with open(f"{folder}/{out}", "w", newline="\n") as f:
        for name in names:
            f.write(f"{digest(f'{folder}/{name}', 'sha256')}  {name}\n")


def textvqa():
    from PIL import Image
    out = f"{EV}/textvqa"
    names = ("llava_textvqa_val_ocr_qid.jsonl", "llava_textvqa_val_noocr_qid.jsonl", "llava_textvqa_val_annotations.jsonl")
    if any(os.path.exists(f"{out}/{n}") for n in names) or os.path.exists(IMG):
        sys.exit("TextVQA files exist; never rebuilt")
    for name, (size, md5) in EXPECT.items():
        p = f"{DL}/textvqa/{name}"
        if os.path.getsize(p) != size or digest(p, "md5") != md5:
            sys.exit(f"{name}: size or md5 differs from the server's")
    data = json.load(open(f"{DL}/textvqa/TextVQA_0.5.1_val.json", encoding="utf-8"))["data"]
    images = sorted({d["image_id"] for d in data})
    if len(data) != 5000 or len(images) != 3166:
        sys.exit(f"unexpected TextVQA val: {len(data)} questions on {len(images)} images")
    key = {(d["image_id"], d["question"].strip().lower()): d for d in data}
    llava = [json.loads(l) for l in open(f"{out}/llava_textvqa_val_v051_ocr.jsonl", encoding="utf-8")]
    rows, text_differs = [], 0
    for q in llava:
        lines = q["text"].split("\n")
        if len(lines) != 3 or not lines[1].startswith("Reference OCR token:") or "\n" + lines[2] != SUFFIX:
            sys.exit(f"unexpected LLaVA prompt: {q['text']!r}")
        d = key.get((q["question_id"], lines[0].strip().lower()))
        if d is None:
            sys.exit(f"no annotation for {q['question_id']}: {lines[0]!r}")
        text_differs += lines[0] != d["question"]
        rows.append((d, q, lines))
    if len(key) != 5000 or len(rows) != 5000 or len({d["question_id"] for d, _, _ in rows}) != 5000:
        sys.exit("the (image, question) match is not one to one")
    with open(f"{out}/{names[0]}", "w", encoding="utf-8", newline="\n") as fo, \
            open(f"{out}/{names[1]}", "w", encoding="utf-8", newline="\n") as fn, \
            open(f"{out}/{names[2]}", "w", encoding="utf-8", newline="\n") as fa:
        for d, q, lines in rows:
            base = {"question_id": d["question_id"], "image": q["image"]}
            fo.write(json.dumps({**base, "text": q["text"], "category": "default"}) + "\n")
            fn.write(json.dumps({**base, "text": lines[0] + SUFFIX, "category": "default"}) + "\n")
            fa.write(json.dumps({**base, "image_id": d["image_id"], "question": d["question"], "answers": d["answers"]}) + "\n")
    write_sums(out, names, "SHA256SUMS_idea0004")
    os.makedirs(IMG)
    want = sorted(f"{i}.jpg" for i in images)
    sums, sizes = [], []
    with zipfile.ZipFile(f"{DL}/textvqa/train_val_images.zip") as z:
        members = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".jpg")}
        missing = [w for w in want if w not in members]
        if missing:
            sys.exit(f"{len(missing)} val images missing from the zip, e.g. {missing[:3]}")
        for name in want:
            blob = z.read(members[name])                      # zipfile checks each member's CRC-32
            with open(f"{IMG}/{name}", "wb") as f:
                f.write(blob)
            sums.append(f"{hashlib.sha256(blob).hexdigest()}  {name}")
            with Image.open(f"{IMG}/{name}") as im:
                im.load()
                sizes.append(im.size)
    with open(f"{IMG}_SHA256SUMS", "w", newline="\n") as f:
        f.write("\n".join(sums) + "\n")
    longest = sorted(max(s) for s in sizes)
    print(f"TextVQA val: 5000 questions on 3166 images extracted and opened; question text differs from the annotation "
          f"(case or spacing) in {text_differs}; long side min / median / max {longest[0]} / {longest[len(longest) // 2]} / {longest[-1]}")


def vstar():
    from PIL import Image
    out = f"{EV}/vstar"
    if os.path.exists(out):
        sys.exit(f"{out} exists; never rebuilt")
    src = f"{DL}/vstar_bench"
    qs = [json.loads(l) for l in open(f"{src}/test_questions.jsonl", encoding="utf-8")]
    if len(qs) != 191 or len({q["question_id"] for q in qs}) != 191:
        sys.exit("unexpected V* question file")
    os.makedirs(out)
    longest = []
    with open(f"{out}/llava_vstar_test.jsonl", "w", encoding="utf-8", newline="\n") as fq, \
            open(f"{out}/llava_vstar_test_annotations.jsonl", "w", encoding="utf-8", newline="\n") as fa:
        for q in qs:
            n_opt = len(re.findall(r"^\([A-D]\) ", q["text"], flags=re.M))
            if q["label"] not in "ABCD"[:n_opt]:
                sys.exit(f"label {q['label']} outside the {n_opt} options of question {q['question_id']}")
            meta = json.load(open(f"{src}/{os.path.splitext(q['image'])[0]}.json", encoding="utf-8"))
            with Image.open(f"{src}/{q['image']}") as im:
                w, h = im.size
            longest.append(max(w, h))
            fq.write(json.dumps({"question_id": q["question_id"], "image": q["image"], "text": q["text"], "category": q["category"]}) + "\n")
            fa.write(json.dumps({"question_id": q["question_id"], "image": q["image"], "label": q["label"], "category": q["category"],
                                 "n_options": n_opt, "target_object": meta["target_object"], "bbox": meta["bbox"], "width": w,
                                 "height": h}) + "\n")
    write_sums(out, ("llava_vstar_test.jsonl", "llava_vstar_test_annotations.jsonl"), "SHA256SUMS")
    longest.sort()
    print(f"V*: 191 questions; long side min / median / max {longest[0]} / {longest[95]} / {longest[-1]}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["vstar", "textvqa"]
    if "vstar" in which:
        vstar()
    if "textvqa" in which:
        textvqa()
