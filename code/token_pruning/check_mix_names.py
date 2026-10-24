"""CPU check of EXP-0013's --fastv-score mix, --name-cos-log and the coco_dev3 name spans.

  python check_mix_names.py --workdir D:/6CJNM007/tmp/tp_smoke

On a tiny random LLaVA through the official generate() path (smoke_test.build_tiny_llava; 640 x 480 image, a stand-in
tokenizer spelling a POPE question after the image), FastV K = 2, keep 30, 6 x 6 blocks, padding excluded, a position
prior: the mix kept set equals the offline rule (n by the object-name cosine in blocks, then the rest by the prior-divided
attention score in blocks, empty blocks first), its object-name part is logged, 30 kept patches lie in 30 blocks; mix with
n = 30 keeps what object_cos keeps and mix with n = 0 what the strong baseline keeps. The name-cosine hook stores one
array per distinct image (two prefills on one image, one on another), equal to the cosine of the image tokens' input
embeddings to each name's mean token embedding. With the real LLaVA tokenizer (CPU, offline): _name_token_ids gives, for
every coco_dev3 question, the ids of the object name in the prompt as model_vqa_loader builds it. The argument checks
refuse the new options where they do not apply.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402
from analyze_object_boxes import padding_patches  # noqa: E402
from check_object_cos import BLOCKS, HEIGHT, KEEP, OBJECT, SIDE, WIDTH, Pieces, refused  # noqa: E402
from fastv_llava import _record_image_span  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")


def cell(p):
    return (p // SIDE) * BLOCKS // SIDE * BLOCKS + (p % SIDE) * BLOCKS // SIDE


def stratified(score, cand, k, taken=None):
    """fastv_llava._stratified_top on the candidate patches with blocks already holding kept patches; patch indices."""
    s = score[cand].tolist()
    cells = [cell(int(p)) for p in cand]
    taken, rank = dict(taken or {}), [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cells[i], 0)
        taken[cells[i]] = rank[i] + 1
    return [int(cand[i]) for i in sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:k]]


def run(workdir, after, score, prior, mix_question=None):
    model = build_tiny_llava(workdir)
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=prior,
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score=score, fastv_exclude_padding=True,
                           fastv_mix_question=mix_question)
    model = run_eval.apply_method(model, m, Pieces(after.tolist()))
    lm = model.get_model()
    run_eval._KEPT.clear()
    run_eval._KEPT_QUESTION.clear()
    lm.register_forward_hook(run_eval._record_kept, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
    return list(run_eval._KEPT), list(run_eval._KEPT_QUESTION), lm


def name_cos_check(workdir, after):
    model = build_tiny_llava(workdir)
    _record_image_span(model)
    lm = model.get_model()
    name_ids = {"cat": [5, 6], "dog": [7], "dining table": [11, 12, 13]}
    for k in run_eval._NAME_COS:
        run_eval._NAME_COS[k] = {} if k == "seen" else []
    names = run_eval._log_name_cos(lm, name_ids)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(2)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    first, second = torch.randn(1, 3, 336, 336), torch.randn(1, 3, 336, 336)
    embeds = []
    with torch.no_grad():
        for images in (first, first, second):
            model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2,
                           use_cache=True)
            embeds.append(model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images,
                                                                      image_sizes=[(WIDTH, HEIGHT)])[4][0].float())
    path = f"{workdir}/name_cos.npz"
    run_eval._write_name_cos(path, names, name_ids)
    z = np.load(path)
    w = lm.embed_tokens.weight.detach().float()
    ok_values = True
    for slot, e in ((0, embeds[0]), (1, embeds[2])):
        img = torch.nn.functional.normalize(e[SYS_LEN:SYS_LEN + 576], dim=-1)
        ref = torch.stack([torch.nn.functional.cosine_similarity(e[SYS_LEN:SYS_LEN + 576], w[name_ids[n]].mean(0, keepdim=True),
                                                                 dim=-1) for n in names], dim=1)
        ok_values &= bool(np.allclose(z["cos"][slot].astype(np.float32), ref.numpy(), atol=2e-3)) and img.shape[0] == 576
    return {"name cosines: one array per distinct image, index per prefill": z["index"].tolist() == [0, 0, 1]
            and z["cos"].shape == (2, 576, 3),
            "name cosines equal the input-embedding cosine to each name's mean token embedding": ok_values,
            "names and their ids are stored": z["names"].tolist() == names
            and [json.loads(s) for s in z["name_ids"]] == [name_ids[n] for n in names]}


def tokenizer_check():
    from transformers import AutoTokenizer
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    with open(f"{EVAL_DATA}/pope/llava_coco_dev3.jsonl") as f:
        questions = [json.loads(line) for line in f]
    with open(f"{EVAL_DATA}/pope/llava_coco_dev3_labels.jsonl") as f:
        labels = [json.loads(line) for line in f]
    with open(run_eval.COCO_INSTANCES, encoding="utf-8") as f:
        names = [c["name"] for c in sorted(json.load(f)["categories"], key=lambda c: c["id"])]
    name_ids = run_eval._name_token_ids(tokenizer, names, "vicuna_v1")
    same = 0
    for q, lab in zip(questions, labels):
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + q["text"])
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
        after = ids[ids.index(IMAGE_TOKEN_INDEX) + 1:]
        _, o_idx, found = run_eval.question_spans(tokenizer, after)
        same += found == lab["object"] and [after[i] for i in o_idx] == name_ids[lab["object"]]
    return {f"80 names; the object-name ids of all {len(questions)} coco_dev3 prompts equal _name_token_ids":
            len(names) == 80 and same == len(questions) == 3600,
            "coco_dev3 questions and labels align": [q["question_id"] for q in questions] == [l["question_id"] for l in labels]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    np.save(f"{a.workdir}/prior.npy", (torch.rand(576, generator=torch.Generator().manual_seed(3)) + 0.5).numpy())
    prior = f"{a.workdir}/prior.npy"
    pad = padding_patches({"width": WIDTH, "height": HEIGHT})
    nonpad = np.setdiff1d(np.arange(576), pad)
    kept15, part15, lm = run(a.workdir, after, "mix", prior, 15)
    cos = lm.fastv_object_cos.float().numpy()
    att = lm.fastv_last_scores.float().numpy() / np.load(prior)
    first = stratified(cos, nonpad, 15)
    rest = np.setdiff1d(nonpad, first)
    filled = {}
    for q in first:
        filled[cell(q)] = filled.get(cell(q), 0) + 1
    offline = sorted(first + stratified(att, rest, KEEP - 15, filled))
    kept30, _, _ = run(a.workdir, after, "mix", prior, 30)
    kept0, _, _ = run(a.workdir, after, "mix", prior, 0)
    kept_obj, _, _ = run(a.workdir, after, "object_cos", None)
    kept_base, _, _ = run(a.workdir, after, "last", prior)
    checks = {
        "one prefill": len(kept15) == 1 == len(part15),
        "mix 15: kept set equals the offline two-stage rule": kept15[0] == offline,
        "mix 15: logged object-name part equals the first stage": part15[0] == sorted(first),
        "mix 15: 30 kept, none in the padding, in 30 different blocks":
            len(kept15[0]) == KEEP and not set(kept15[0]) & set(pad.tolist()) and len({cell(q) for q in kept15[0]}) == KEEP,
        "mix 30 keeps what object_cos keeps": kept30[0] == kept_obj[0],
        "mix 0 keeps what the strong baseline keeps (prior, blocks, no padding)": kept0[0] == kept_base[0],
        "the three rules differ here": kept15[0] != kept_obj[0] and kept15[0] != kept_base[0],
    }
    checks.update(name_cos_check(a.workdir, after))
    checks.update(tokenizer_check())
    checks.update({
        "mix refused without --fastv-mix-question": refused(["--method", "fastv", "--fastv-score", "mix",
                                                             "--fastv-stratify", "6"]),
        "--fastv-mix-question refused without mix": refused(["--method", "fastv", "--fastv-mix-question", "5"]),
        "mix refused without blocks": refused(["--method", "fastv", "--fastv-score", "mix", "--fastv-mix-question", "5"]),
        "mix refused with --text-rows-log": refused(["--method", "fastv", "--fastv-score", "mix", "--fastv-mix-question",
                                                     "5", "--fastv-stratify", "6", "--text-rows-log", "x"]),
        "mix refused with a schedule": refused(["--method", "fastv", "--fastv-score", "mix", "--fastv-mix-question", "5",
                                                "--fastv-stratify", "6", "--fastv-schedule", "2:30"]),
        "--name-cos-log refused for visionzip": refused(["--method", "visionzip", "--name-cos-log", "x"]),
        "mix accepted with blocks, prior and padding exclusion": not refused([
            "--method", "fastv", "--fastv-score", "mix", "--fastv-mix-question", "15", "--fastv-stratify", "6",
            "--fastv-score-prior", "x", "--fastv-exclude-padding"]),
        "--name-cos-log accepted for vanilla": not refused(["--method", "vanilla", "--name-cos-log", "x"]),
    })
    import exp0002
    args = exp0002.CONFIGS["fastv64_mix15_kl"][1]
    checks["exp0002 fastv64_mix15 = strong baseline + mix 15"] = (
        args == exp0002.CONFIGS["fastv64_prior_strat_nopad"][1] + ["--fastv-score", "mix", "--fastv-mix-question", "15"])
    checks["exp0002 has coco_dev3 and the _ksl, vanilla_names configs"] = (
        "coco_dev3" in exp0002.BENCHES and "fastv64_objcos_strat_nopad_ksl" in exp0002.CONFIGS
        and "fastv64_prior_strat_nopad_ksl" in exp0002.CONFIGS and "vanilla_names" in exp0002.CONFIGS)
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    print("mix 15 kept:", kept15[0][:10], "... object-name part:", part15[0][:8], "...")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
