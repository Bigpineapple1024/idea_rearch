"""Regression helper for check_zoom.py: with the fastv_llava / run_eval of --code-dir, run the router configuration (query_vis,
blocks, prior, padding excluded) on a tiny random LLaVA over a fixed sequence of prefills and decisions, and write the
kept sets and generated ids (JSON). Run once with the committed code and once with the new code; the two must match.

  python kept_dump.py --code-dir <dir> --workdir <dir> --out <file.json>
"""
import argparse
import json
import sys

p = argparse.ArgumentParser()
p.add_argument("--code-dir", required=True)
p.add_argument("--workdir", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
sys.path.insert(0, a.code_dir)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fastv_llava  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import BLOCKS, HEIGHT, KEEP, WIDTH, Pieces  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402

after = torch.arange(100, 100 + TEXT_LEN)
rng = np.random.default_rng(0)
idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
np.save(f"{a.workdir}/prior_zoom.npy", (torch.rand(576, generator=torch.Generator().manual_seed(3)) + 0.5).numpy())
np.save(f"{a.workdir}/vis_zoom.npy", rng.uniform(0, 1, 1000).astype(np.float32))
m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=f"{a.workdir}/prior_zoom.npy",
                       fastv_stratify=BLOCKS, text_rows_log=None, fastv_score="query_vis", fastv_exclude_padding=True,
                       fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=idf, fastv_visibility=f"{a.workdir}/vis_zoom.npy")
model = run_eval.apply_method(build_tiny_llava(a.workdir), m, Pieces(after.tolist()))
lm = model.get_model()
flag = [None]
lm.fastv_route = lambda ids: flag[0]
out = {"kept": [], "gen": [], "modules": {m.__name__: m.__file__.replace("\\", "/") for m in (fastv_llava, run_eval)}}
torch.manual_seed(1)
ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
with torch.no_grad():
    for i, d in enumerate([True, False, False, True, False]):
        flag[0] = d
        images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))
        g = model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=3, use_cache=True)
        out["kept"].append(lm.fastv_kept_image_positions.tolist())
        out["gen"].append(g[0].tolist())
json.dump(out, open(a.out, "w"))
print("ok", a.code_dir)
