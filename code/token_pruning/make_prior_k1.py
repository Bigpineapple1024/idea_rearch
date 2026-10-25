"""EXP-0024: FastV's position prior at K = 1, i.e. the mean layer-0 score of every image patch over the blank-image
prefills of EXP-0024-R01 (fastv64k1_scores on pope_img20_blank), as analyze_score_prior.py did for K = 2 (EXP-0008).

  python make_prior_k1.py      # writes code/output/EXP-0024/prior_k1/{position_prior.npy,prior_k1.json,SHA256SUMS}

Never overwrites; the folder is copied to D:/6CJNM007/backup/EXP-0024/prior_k1 and the copy is checked.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "code", "output")
BACKUP = "D:/6CJNM007/backup/EXP-0024/prior_k1"
GRID = 24


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    src = f"{OUT}/EXP-0024/EXP-0024-R01/scores.npy"
    dst = f"{OUT}/EXP-0024/prior_k1"
    if os.path.exists(dst) or os.path.exists(BACKUP):
        sys.exit(f"{dst} or its backup exists; never overwritten")
    blank = np.load(src)
    if blank.ndim != 2 or blank.shape[1] != GRID * GRID:
        sys.exit(f"unexpected score array {blank.shape}")
    prior = blank.mean(axis=0)
    os.makedirs(dst)
    np.save(f"{dst}/position_prior.npy", prior.astype(np.float32))
    k2 = np.load(f"{OUT}/EXP-0008/analysis/position_prior.npy")
    summary = {"source": "EXP-0024-R01/scores.npy", "source_sha256": sha256(src), "n_blank_prefills": int(blank.shape[0]),
               "row_mean_relative": (prior.reshape(GRID, GRID).mean(axis=1) / prior.mean()).tolist(),
               "bottom4_rows_share": float(prior.reshape(GRID, GRID)[20:].sum() / prior.sum()),
               "last_patch_share": float(prior[-1] / prior.sum()),
               "pearson_r_with_k2_prior": float(np.corrcoef(prior, k2)[0, 1])}
    json.dump(summary, open(f"{dst}/prior_k1.json", "w"), indent=1)
    with open(f"{dst}/SHA256SUMS", "w", newline="\n") as f:
        for name in ("position_prior.npy", "prior_k1.json"):
            f.write(f"{sha256(f'{dst}/{name}')}  {name}\n")
    shutil.copytree(dst, BACKUP)
    for name in ("position_prior.npy", "prior_k1.json", "SHA256SUMS"):
        if sha256(f"{dst}/{name}") != sha256(f"{BACKUP}/{name}"):
            sys.exit(f"backup of {name} differs")
    print(json.dumps({k: v for k, v in summary.items() if k != "row_mean_relative"}))


if __name__ == "__main__":
    main()
