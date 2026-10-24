"""CPU check of the EXP-0011 diagnostics on a tiny random LLaVA: --text-rows-log and --answer-logit-log must record
what they claim and change nothing.

  python check_text_rows_log.py --workdir D:/6CJNM007/tmp/tp_smoke

FastV with the position prior and 6x6 coverage (the strong baseline of 1A4B-0003), K=2, keep 30, a random positive
prior, through the official generate() path (smoke_test.build_tiny_llava). Checks: one row per text token after the
image and 576 columns; the last row equals FastV's score row; the logged ids are the prompt's ids after the image
token; the first generated token is the argmax of the logged first-token distribution; kept patches and generated
tokens are identical with and without logging; the .npz and .json writers round-trip; the argument checks refuse
--text-rows-log without FastV ranking and --answer-logit-log with SparseVLM.
"""
import argparse
import functools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

CANDIDATES = [3, 4, 5, 6]                         # the tiny model has no tokenizer: any four ids


def generate(workdir, log):
    model = build_tiny_llava(workdir)
    torch.manual_seed(2)
    np.save(f"{workdir}/prior.npy", (torch.rand(576) + 0.5).numpy())
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=30, fastv_schedule=None,
                           fastv_score_prior=f"{workdir}/prior.npy", fastv_stratify=6,
                           text_rows_log="on" if log else None)
    model = run_eval.apply_method(model, m)
    lm = model.get_model()
    kept = []

    def record_kept(module, args, kwargs, output):
        if kwargs.get("inputs_embeds") is not None and kwargs["inputs_embeds"].shape[1] > 1:
            kept.append(module.fastv_kept_image_positions.tolist())

    lm.register_forward_hook(record_kept, with_kwargs=True)
    if log:
        lm.register_forward_hook(run_eval._record_text_rows, with_kwargs=True)
        model.lm_head.register_forward_hook(functools.partial(run_eval._record_logits, CANDIDATES))
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    after = torch.randint(3, 1000, (TEXT_LEN,))
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        out = model.generate(ids, images=images, do_sample=False, max_new_tokens=3, use_cache=True)
    return out[0].tolist(), kept, after.tolist(), lm


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
    plain_out, plain_kept, _, _ = generate(a.workdir, log=False)
    out, kept, after, lm = generate(a.workdir, log=True)
    rows, ids = run_eval._TEXT_ROWS[0]
    # generate() from inputs_embeds returns the BOS id (1) in front of the 3 new tokens in transformers 4.37
    first = out[1] if len(out) == 4 and out[0] == 1 else out[0]
    checks = {
        "generated tokens unchanged by logging": out == plain_out,
        "kept patches unchanged by logging": kept == plain_kept,
        "one prefill logged": len(run_eval._TEXT_ROWS) == 1 and len(run_eval._LOGITS) == 1,
        "rows: one per text token after the image, 576 columns": rows.shape == (TEXT_LEN, 576),
        "ids are the prompt's ids after the image token": ids == after,
        "last row is FastV's score row": bool(np.allclose(rows[-1].astype(np.float32),
                                                          lm.fastv_last_scores.float().numpy(), atol=1e-3)),
        "first generated token is the logged argmax": run_eval._LOGITS[0]["argmax"] == first,
        "log-probabilities are <= 0": all(v <= 0 for v in run_eval._LOGITS[0]["logprobs"]),
        "--text-rows-log refused for random": refused(["--method", "random", "--fastv-keep", "30", "--seed", "1",
                                                       "--text-rows-log", "x"]),
        "--answer-logit-log refused for sparsevlm": refused(["--method", "sparsevlm", "--retained-tokens", "64",
                                                            "--answer-logit-log", "x"]),
        "--answer-logit-log accepted for vanilla": not refused(["--method", "vanilla", "--answer-logit-log", "x"]),
    }
    run_eval._write_text_rows(f"{a.workdir}/text_rows.npz")
    z = np.load(f"{a.workdir}/text_rows.npz")
    checks["npz round trip"] = (z["rows"].shape == (TEXT_LEN, 576) and z["offsets"].tolist() == [0, TEXT_LEN]
                                and z["token_ids"].tolist() == after)
    run_eval._write_logits(f"{a.workdir}/answer_logits.json", CANDIDATES)
    with open(f"{a.workdir}/answer_logits.json") as f:
        logged = json.load(f)
    checks["json round trip"] = logged["n_prefills"] == 1 and logged["ids"] == CANDIDATES
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    print("kept patches:", kept[0][:10], "... generated:", out, "logged argmax:", run_eval._LOGITS[0]["argmax"])
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
