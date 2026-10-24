"""CPU check of EXP-0011 D3's --fastv-score object_cos and --fastv-exclude-padding on a tiny random LLaVA.

  python check_object_cos.py --workdir D:/6CJNM007/tmp/tp_smoke

FastV, K = 2, keep 30, 6 x 6 blocks, through the official generate() path (smoke_test.build_tiny_llava) on a 640 x 480
image, with a stand-in tokenizer whose pieces spell a POPE question after the image. Checks: the score is the cosine of
each image token's input embedding to the object name's mean input embedding; the kept set equals the offline rule of
analyze_layer_rows.py (block rotation over the non-padding patches) on that score; no kept patch lies in the padding,
whose mask equals analyze_object_boxes.padding_patches; 30 patches are kept; the strong baseline with padding excluded
keeps 30 non-padding patches; the argument checks refuse the new options outside FastV and object_cos with a position
prior or an attention-row log.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402
from analyze_object_boxes import padding_patches  # noqa: E402
from fastv_llava import _padding_mask  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

PIECES = ["\u2581", "<0x0A>", "Is", "\u2581there", "\u2581a", "\u2581c", "at", "\u2581in", "\u2581the", "\u2581image", "?",
          "<0x0A>", "Answer", "\u2581briefly", ".", "\u2581A", "SS", "IST", "ANT", ":"]
OBJECT = [5, 6]                                   # token indices of the object name within the ids after the image
WIDTH, HEIGHT = 640, 480
SIDE, BLOCKS, KEEP = 24, 6, 30


class Pieces:
    def __init__(self, ids):
        self.map = dict(zip(ids, PIECES))

    def convert_ids_to_tokens(self, ids):
        return [self.map[i] for i in ids]


def stratified(score, cand):
    """analyze_layer_rows.stratified: fastv_llava._stratified_top on the candidate patches, sorted patch indices."""
    s = score[cand].tolist()
    cell = ((cand // SIDE) * BLOCKS // SIDE * BLOCKS + (cand % SIDE) * BLOCKS // SIDE).tolist()
    taken, rank = {}, [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cell[i], 0)
        taken[cell[i]] = rank[i] + 1
    return sorted(int(cand[i]) for i in sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:KEEP])


def run(workdir, after, score, prior=None):
    model = build_tiny_llava(workdir)
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=prior,
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score=score, fastv_exclude_padding=True)
    model = run_eval.apply_method(model, m, Pieces(after.tolist()))
    lm = model.get_model()
    kept = []

    def record(module, args, kwargs, output):
        if kwargs.get("inputs_embeds") is not None and kwargs["inputs_embeds"].shape[1] > 1:
            kept.append(module.fastv_kept_image_positions.tolist())

    lm.register_forward_hook(record, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
        embeds = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images, image_sizes=[(WIDTH, HEIGHT)])[4]
    return kept, lm, embeds


def refused(argv):
    saved = sys.argv
    sys.argv = ["run_eval.py"] + argv + ["--", "--model-path", "m"]
    try:
        run_eval.parse()
        return False
    except SystemExit:
        return True
    finally:
        sys.argv = saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    kept, lm, embeds = run(a.workdir, after, "object_cos")
    h = embeds[0].float()
    obj = [SYS_LEN + 576 + i for i in OBJECT]
    cos = torch.nn.functional.cosine_similarity(h[SYS_LEN:SYS_LEN + 576], h[obj].mean(0, keepdim=True), dim=-1)
    pad = padding_patches({"width": WIDTH, "height": HEIGHT})
    mask = _padding_mask(WIDTH, HEIGHT, SIDE).numpy()
    nonpad = np.setdiff1d(np.arange(576), pad)
    np.save(f"{a.workdir}/prior.npy", (torch.rand(576) + 0.5).numpy())
    kept_base, _, _ = run(a.workdir, after, "last", prior=f"{a.workdir}/prior.npy")
    checks = {
        "one prefill": len(kept) == 1,
        "score is the input-embedding cosine to the object name": bool(torch.allclose(lm.fastv_object_cos, cos, atol=1e-6)),
        "padding mask equals analyze_object_boxes.padding_patches": np.flatnonzero(mask).tolist() == pad.tolist(),
        "the image has padding (3 rows above, 3 below)": len(pad) == 6 * SIDE,
        "kept set equals the offline rule": kept[0] == stratified(cos.numpy(), nonpad),
        "30 patches kept, none in the padding": len(kept[0]) == KEEP and not set(kept[0]) & set(pad.tolist()),
        "strong baseline + exclude padding keeps 30 non-padding patches":
            len(kept_base[0]) == KEEP and not set(kept_base[0]) & set(pad.tolist()),
        "--fastv-score refused for random": refused(["--method", "random", "--seed", "1", "--fastv-score", "object_cos"]),
        "--fastv-exclude-padding refused for vanilla": refused(["--method", "vanilla", "--fastv-exclude-padding"]),
        "object_cos refused with a position prior": refused(["--method", "fastv", "--fastv-score", "object_cos",
                                                             "--fastv-score-prior", "x"]),
        "object_cos refused with --text-rows-log": refused(["--method", "fastv", "--fastv-score", "object_cos",
                                                            "--text-rows-log", "x"]),
        "object_cos accepted for fastv": not refused(["--method", "fastv", "--fastv-score", "object_cos",
                                                      "--fastv-exclude-padding", "--fastv-stratify", "6"]),
    }
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    print("padding patches:", len(pad), "kept:", kept[0][:10], "...")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
