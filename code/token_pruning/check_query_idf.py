"""CPU check of EXP-0014's --fastv-score query_mean / query_idf / attn_idf and of the IDF tables.

  python check_query_idf.py --workdir D:/6CJNM007/tmp/tp_smoke

On the tiny random LLaVA of check_object_cos.py (K = 2, keep 30, 6 x 6 blocks, padding excluded): query_idf's score equals
the cosine of each image token's input embedding to the weighted mean input embedding of the tokens after the image,
recomputed; the kept set is the block rotation of that score; query_mean equals query_idf with equal weights; query_idf
with all weight on the object name keeps what object_cos keeps; attn_idf with all weight on the last token keeps what
FastV's last-token score keeps (blocks, no prior). With the real tokenizer (CPU, offline): question_idf gives zero weight
to tokens in every question; on coco_dev3 and POPE the object name's share of the weights is reported with the heaviest
other token where it is below 0.9 (a property of the method, not a requirement). The argument checks refuse the new
scores where they do not apply.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402
from analyze_object_boxes import padding_patches  # noqa: E402
from check_mix_names import stratified  # noqa: E402
from check_object_cos import BLOCKS, HEIGHT, KEEP, OBJECT, WIDTH, Pieces, refused  # noqa: E402
from prompt_spans import question_spans  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")


def run(workdir, after, score, idf=None, prior=None):
    model = build_tiny_llava(workdir)
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=prior,
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score=score, fastv_exclude_padding=True,
                           fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=idf)
    model = run_eval.apply_method(model, m, Pieces(after.tolist()))
    lm = model.get_model()
    run_eval._KEPT.clear()
    lm.register_forward_hook(run_eval._record_kept, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
        h = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images,
                                                       image_sizes=[(WIDTH, HEIGHT)])[4][0].float()
    return list(run_eval._KEPT), lm, h


def tokenizer_checks():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    out = {}
    for name, path in (("coco_dev3", f"{EVAL_DATA}/pope/llava_coco_dev3.jsonl"), ("pope", f"{EVAL_DATA}/pope/llava_pope_test.jsonl")):
        idf = run_eval.question_idf(tok, path, "vicuna_v1")
        texts = [json.loads(line)["text"] for line in open(path)]
        shares = []
        for text in texts:
            ids = run_eval.prompt_ids_after_image(tok, text, "vicuna_v1")
            w = np.array([idf["idf"][t] for t in ids])
            obj = question_spans(tok, ids)[1]
            shares.append(w[obj].sum() / w.sum())
        shares = np.array(shares)
        everywhere = [t for t, c in idf["df"].items() if c == idf["n_questions"]]
        out[f"{name}: tokens present in every question get weight 0 ({len(everywhere)} tokens)"] = all(
            abs(idf["idf"][t]) < 1e-12 for t in everywhere) and len(everywhere) > 10
        print(f"{name}: object-name share of the IDF weights min {shares.min():.3f}, median {np.median(shares):.3f}, "
              f"{int((shares < 0.9).sum())} of {len(shares)} questions below 0.9 (reported, not required)")
        low = Counter()
        for k in np.flatnonzero(shares < 0.9):
            ids = run_eval.prompt_ids_after_image(tok, texts[k], "vicuna_v1")
            obj = set(question_spans(tok, ids)[1])
            top = max((idf["idf"][t], j) for j, t in enumerate(ids) if j not in obj)
            low[tok.convert_ids_to_tokens([ids[top[1]]])[0]] += 1
        print(f"   heaviest non-object token in those questions: {low.most_common(5)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    pad = padding_patches({"width": WIDTH, "height": HEIGHT})
    nonpad = np.setdiff1d(np.arange(576), pad)
    rng = np.random.default_rng(0)
    idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
    kept_idf, lm, h = run(a.workdir, after, "query_idf", idf)
    w = torch.tensor([idf["idf"][int(t)] for t in after])
    q = (w[:, None] * h[SYS_LEN + 576:SYS_LEN + 576 + TEXT_LEN]).sum(0, keepdim=True) / w.sum()
    ref = torch.nn.functional.cosine_similarity(h[SYS_LEN:SYS_LEN + 576], q, dim=-1)
    equal = {"idf": {int(t): 1.0 for t in after}, "unseen": 1.0}
    kept_eq, _, _ = run(a.workdir, after, "query_idf", equal)
    kept_mean, _, _ = run(a.workdir, after, "query_mean")
    obj_only = {"idf": {int(t): (1.0 if i in OBJECT else 0.0) for i, t in enumerate(after)}, "unseen": 0.0}
    kept_obj_q, _, _ = run(a.workdir, after, "query_idf", obj_only)
    kept_obj, _, _ = run(a.workdir, after, "object_cos")
    last_only = {"idf": {int(t): (1.0 if i == TEXT_LEN - 1 else 0.0) for i, t in enumerate(after)}, "unseen": 0.0}
    kept_attn_last, lm_attn, _ = run(a.workdir, after, "attn_idf", last_only)
    kept_last, _, _ = run(a.workdir, after, "last")
    kept_attn_idf, _, _ = run(a.workdir, after, "attn_idf", idf)
    checks = {
        "query_idf: score equals the cosine to the weighted mean input embedding, recomputed": bool(
            torch.allclose(lm.fastv_input_score.float(), ref, atol=1e-5)),
        "query_idf: kept set is the block rotation of the score over non-padding patches":
            kept_idf[0] == sorted(stratified(ref.numpy(), nonpad, KEEP)),
        "query_mean equals query_idf with equal weights": kept_mean[0] == kept_eq[0],
        "query_idf with all weight on the object name keeps what object_cos keeps": kept_obj_q[0] == kept_obj[0],
        "attn_idf with all weight on the last token keeps what FastV's last-token score keeps": kept_attn_last[0] == kept_last[0],
        "attn_idf with IDF weights differs from the last token": kept_attn_idf[0] != kept_last[0],
        "30 kept, none in the padding": all(len(k[0]) == KEEP and not set(k[0]) & set(pad.tolist())
                                            for k in (kept_idf, kept_mean, kept_attn_idf)),
    }
    checks.update(tokenizer_checks())
    checks.update({
        "query_idf refused with a position prior": refused(["--method", "fastv", "--fastv-score", "query_idf",
                                                            "--fastv-score-prior", "x"]),
        "attn_idf refused with --text-rows-log": refused(["--method", "fastv", "--fastv-score", "attn_idf",
                                                          "--text-rows-log", "x"]),
        "query_mean refused with a schedule": refused(["--method", "fastv", "--fastv-score", "query_mean",
                                                       "--fastv-schedule", "2:30"]),
        "query_idf refused for random": refused(["--method", "random", "--seed", "1", "--fastv-score", "query_idf"]),
        "query_idf accepted with blocks and padding exclusion": not refused([
            "--method", "fastv", "--fastv-score", "query_idf", "--fastv-stratify", "6", "--fastv-exclude-padding"]),
    })
    import exp0002
    for name, kind in (("qmean", "query_mean"), ("qidf", "query_idf"), ("attnidf", "attn_idf")):
        checks[f"exp0002 fastv64_{name}_strat_nopad_kl = the variant with --fastv-score {kind}"] = (
            exp0002.CONFIGS[f"fastv64_{name}_strat_nopad_kl"][1] == [
                x if x != "object_cos" else kind for x in exp0002.CONFIGS["fastv64_objcos_strat_nopad"][1]])
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
