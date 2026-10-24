"""EXP-0015: the visibility of every vocabulary id, from the coco_dev images only (plan v1).

  python make_visibility.py <vocab_visibility.npz of the logging run> <output folder>   # writes visibility.npy, .md

For each image and vocabulary id, the peak = largest minus mean cosine of the id's input embedding to the image tokens'
input embeddings (how strongly the word picks out one region of that image). A word's raw visibility is the 95th
percentile of its peak over the images (it localises clearly in at least a few images); the visibility used by
query_vis is its percentile rank among all vocabulary ids, in [0, 1]. Nothing here sees POPE or MME images.
"""
import os
import sys

import numpy as np


def main():
    src, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        sys.exit(f"{out} exists; results are never overwritten")
    z = np.load(src)
    peak = z["max"].astype(np.float32) - z["mean"].astype(np.float32)          # (images, vocab)
    raw = np.percentile(peak, 95, axis=0)
    order = np.argsort(np.argsort(raw, kind="stable"), kind="stable")
    vis = order / (len(raw) - 1)
    os.makedirs(out)
    np.save(f"{out}/visibility.npy", vis.astype(np.float32))
    np.save(f"{out}/visibility_raw.npy", raw.astype(np.float32))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    top = np.argsort(-raw)[:40]
    low = np.argsort(raw)[:20]
    words = ["▁dog", "▁car", "▁pizza", "▁person", "▁table", "▁red", "▁two", "▁left", "▁the", "▁image", "▁answer", "▁Leon",
             "▁Graham", "▁movie", "▁city", "▁building"]
    ids = tok.convert_tokens_to_ids(words)
    lines = [f"# visibility from {peak.shape[0]} coco_dev images, {peak.shape[1]} vocabulary ids", "",
             "highest raw visibility: " + ", ".join(tok.convert_ids_to_tokens([int(t)])[0] for t in top), "",
             "lowest: " + ", ".join(tok.convert_ids_to_tokens([int(t)])[0] for t in low), "",
             "| token | raw | percentile |", "| --- | --- | --- |"]
    lines += [f"| {w} | {raw[i]:.4f} | {vis[i]:.3f} |" for w, i in zip(words, ids) if i != tok.unk_token_id and i < len(raw)]
    open(f"{out}/visibility.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
