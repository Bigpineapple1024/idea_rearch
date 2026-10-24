"""CPU check of EXP-0011 D2b's --layer-rows-log on a tiny random LLaVA: it must record what it claims.

  python check_layer_rows_log.py --workdir D:/6CJNM007/tmp/tp_smoke

Vanilla through the official generate() path (smoke_test.build_tiny_llava: 6 layers, 4 heads, 576 image tokens), with
a stand-in tokenizer whose pieces spell a POPE question after the image. Checks: one prefill logged; the spans are the
question sentence and the object name; the logged head-averaged rows, per-head rows and cosine similarities equal a
direct recomputation from the unwrapped forward with attentions and hidden states; the per-head rows average to the
head-averaged span rows; the generated tokens equal those of a run without logging; the .npz writer round-trips; the
argument check refuses --layer-rows-log for FastV and accepts it for vanilla.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

PIECES = ["\u2581", "<0x0A>", "Is", "\u2581there", "\u2581a", "\u2581c", "at", "\u2581in", "\u2581the", "\u2581image", "?",
          "<0x0A>", "Answer", "\u2581briefly", ".", "\u2581A", "SS", "IST", "ANT", ":"]
QUESTION, OBJECT = [2, 11], [5, 7]                # [first, last + 1] within the ids after the image


class Pieces:
    """Stand-in tokenizer: the ids after the image map to PIECES in order."""

    def __init__(self, ids):
        self.map = dict(zip(ids, PIECES))

    def convert_ids_to_tokens(self, ids):
        return [self.map[i] for i in ids]


def generate(workdir, after, log):
    model = build_tiny_llava(workdir)
    if log:
        run_eval._record_image_span(model)
        run_eval._log_layer_rows(model.get_model(), Pieces(after.tolist()))
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        out = model.generate(ids, images=images, do_sample=False, max_new_tokens=3, use_cache=True)
    return out[0].tolist(), model, ids, images


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
    run_eval.LAYER_ROWS_LAYERS = (0, 1, 2, 5)          # the tiny model has 6 layers
    after = torch.arange(100, 100 + TEXT_LEN)
    assert len(PIECES) == TEXT_LEN
    plain_out, _, _, _ = generate(a.workdir, after, log=False)
    out, model, ids, images = generate(a.workdir, after, log=True)
    lm = model.get_model()
    with torch.no_grad():
        embeds = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images)[4]
        ref = lm.layer_rows_orig_forward(inputs_embeds=embeds, output_attentions=True, output_hidden_states=True,
                                         return_dict=True)
    start, n_img = SYS_LEN, 576
    img = slice(start, start + n_img)
    q_pos = list(range(start + n_img + QUESTION[0], start + n_img + QUESTION[1]))
    o_pos = list(range(start + n_img + OBJECT[0], start + n_img + OBJECT[1]))
    rec = run_eval._LAYER_ROWS[0] if run_eval._LAYER_ROWS else {}
    def spans_of(m):
        return np.stack([m[-1, img].float().numpy(), m[q_pos, img].float().mean(0).numpy(), m[o_pos, img].float().mean(0).numpy()])

    avg_ref = np.stack([spans_of(ref.attentions[layer].mean(dim=1)[0]) for layer in run_eval.LAYER_ROWS_LAYERS])
    head_ref = np.stack([np.stack([ref.attentions[layer][0][:, pos, img].float().mean(1).numpy() for pos in (q_pos, o_pos)])
                         for layer in run_eval.HEAD_ROW_LAYERS])
    cos_ref = np.stack([np.stack([torch.nn.functional.cosine_similarity(
        ref.hidden_states[level][0][img].float(), ref.hidden_states[level][0][pos].float().mean(0, keepdim=True), dim=-1).numpy()
        for pos in (q_pos, o_pos)]) for level in run_eval.STATE_LEVELS])
    checks = {
        "one prefill logged": len(run_eval._LAYER_ROWS) == 1,
        "question span is the question sentence": rec.get("question") == QUESTION,
        "object span is the object name": rec.get("object") == OBJECT,
        "ids are the prompt's ids after the image": rec.get("ids") == after.tolist(),
        "head-averaged rows match a direct recomputation": rec["avg"].shape == (4, 3, 576)
            and bool(np.allclose(rec["avg"].astype(np.float32), avg_ref, atol=1e-3)),
        "per-head rows match a direct recomputation": rec["head"].shape == (2, 2, 4, 576)
            and bool(np.allclose(rec["head"].astype(np.float32), head_ref, atol=1e-3)),
        "per-head rows average to the head-averaged span rows": bool(np.allclose(
            rec["head"].astype(np.float32).mean(2), avg_ref[:2, 1:], atol=2e-3)),
        "cosine similarities match a direct recomputation": rec["cos"].shape == (3, 2, 576)
            and bool(np.allclose(rec["cos"], cos_ref, atol=1e-5)),
        "generated tokens unchanged by logging": out == plain_out,
        "--layer-rows-log refused for fastv": refused(["--method", "fastv", "--fastv-keep", "30", "--layer-rows-log", "x"]),
        "--layer-rows-log accepted for vanilla": not refused(["--method", "vanilla", "--layer-rows-log", "x"]),
    }
    run_eval._write_layer_rows(f"{a.workdir}/layer_rows.npz")
    z = np.load(f"{a.workdir}/layer_rows.npz")
    checks["npz round trip"] = (z["avg_rows"].shape == (1, 4, 3, 576) and z["head_rows"].shape == (1, 2, 2, 4, 576)
                                and z["cos"].shape == (1, 3, 2, 576) and z["offsets"].tolist() == [0, TEXT_LEN]
                                and z["token_ids"].tolist() == after.tolist() and z["question_span"].tolist() == [QUESTION]
                                and z["object_span"].tolist() == [OBJECT] and z["layers"].tolist() == [0, 1, 2, 5])
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    print("generated:", out, "plain:", plain_out)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
