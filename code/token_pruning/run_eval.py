"""Run LLaVA's model_vqa_loader evaluation with a visual-token reduction method switched on.

Covers the benchmarks that LLaVA-1.5 evaluates through llava.eval.model_vqa_loader
(POPE, GQA, TextVQA, MME, VQAv2, VizWiz). All LLaVA arguments are passed through unchanged.

  python run_eval.py --method vanilla   -- --model-path liuhaotian/llava-v1.5-7b ...
  python run_eval.py --method fastv --fastv-k 2 --fastv-keep 30 -- ...
  python run_eval.py --method visionzip --dominant 54 --contextual 10 -- ...
  python run_eval.py --method random --fastv-k 2 --fastv-keep 30 --seed 2021 -- ...
  python run_eval.py --method sparsevlm --retained-tokens 64 -- ...   # in the sparsevlm env

VisionZip is imported from its repository root (set VISIONZIP_ROOT or PYTHONPATH): the
visionzip 0.1.3 wheel on PyPI ships no code.
`random` is FastV with the attention ranking replaced by a seeded uniform draw (same K, same count).
`sparsevlm` runs SparseVLM's own fork of model_vqa_loader (pip install -e source/SparseVLMs), which
prunes inside generate() from `retained_tokens`; it refuses to run against the official LLaVA loader.
Its decoder layers call FlashAttention-2, which has no Windows build; sparsevlm_sdpa.py swaps in SDPA.
"""
import argparse
import functools
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
from llava.eval import model_vqa_loader as loader
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutputWithPast

from fastv_llava import _record_image_span, apply_fastv
from prompt_spans import question_spans

COCO_INSTANCES = os.environ.get("COCO_INSTANCES", "D:/6CJNM007/data/coco/annotations/instances_val2014.json")


def _schedule(text):
    stages = [tuple(int(x) for x in part.split(":")) for part in text.split(",")]
    if any(len(s) != 2 for s in stages) or [s[0] for s in stages] != sorted({s[0] for s in stages}):
        raise argparse.ArgumentTypeError("expected increasing 'layer:keep' pairs, e.g. 3:82,7:32,16:18")
    return stages


def parse():
    ours = argparse.ArgumentParser()
    ours.add_argument("--method", choices=["vanilla", "fastv", "visionzip", "random", "sparsevlm"], required=True)
    ours.add_argument("--fastv-k", type=int, default=2)
    ours.add_argument("--fastv-keep", type=int, default=288)
    ours.add_argument("--seed", type=int, default=None, help="random: seed of the token draw")
    ours.add_argument("--fastv-schedule", type=_schedule, default=None,
                      help="fastv/random: prune in stages 'layer:keep,layer:keep,...' instead of --fastv-k/--fastv-keep")
    ours.add_argument("--dominant", type=int, default=54)
    ours.add_argument("--contextual", type=int, default=10)
    ours.add_argument("--retained-tokens", type=int, default=None, help="sparsevlm: 32-layer average budget")
    ours.add_argument("--attn-implementation", choices=["eager", "sdpa"], default=None,
                      help="load the LLM with this attention kernel (default: transformers' choice, sdpa in 4.37)")
    ours.add_argument("--num-workers", type=int, default=4,
                      help="DataLoader workers (LLaVA's default 4; 0 loads images in this process)")
    ours.add_argument("--layer-token-log", default=None,
                      help="diagnostic: write the tokens entering each LLM layer during every prefill (JSON)")
    ours.add_argument("--kept-token-log", default=None,
                      help="diagnostic, fastv/random only: write the image-patch indices kept in every prefill (JSON)")
    ours.add_argument("--score-log", default=None,
                      help="diagnostic, fastv only: save FastV's attention score of every image patch per prefill (.npy)")
    ours.add_argument("--text-rows-log", default=None,
                      help="diagnostic, fastv only: save the scoring layer's head-averaged attention rows of every text "
                           "token after the image over the image patches, with the token ids, per prefill (.npz)")
    ours.add_argument("--answer-logit-log", default=None,
                      help="diagnostic, not sparsevlm: write the log-probabilities of Yes/No/yes/no as the first answer "
                           "token, and the most likely first token, per prefill (JSON)")
    ours.add_argument("--layer-rows-log", default=None,
                      help="diagnostic, vanilla only: per prefill, the head-averaged attention rows of the last token, "
                           "the question sentence and the object name over the image patches at several layers, their "
                           "per-head rows at layers 0 and 1, and the cosine similarity of every image token's hidden "
                           "state to the two spans' mean states at hidden_states[0..2] (.npz); the prefill runs eager")
    ours.add_argument("--fastv-score-prior", default=None,
                      help="diagnostic, fastv only: .npy with one value per image patch; scores are divided by it")
    ours.add_argument("--fastv-stratify", type=int, default=None,
                      help="diagnostic, fastv only: take the ranking in turns from an n x n grid of image blocks")
    ours.add_argument("--fastv-visibility", default=None,
                      help="fastv, --fastv-score query_vis: .npy with one visibility in [0, 1] per vocabulary id (EXP-0015)")
    ours.add_argument("--vocab-visibility-log", default=None,
                      help="diagnostic: per image, the largest and the mean cosine over the image tokens' input embeddings "
                           "for every vocabulary embedding (.npz, EXP-0015)")
    ours.add_argument("--fastv-score", choices=["last", "object_cos", "mix", "contrast_max", "contrast_mean", "query_mean",
                                                "query_idf", "attn_idf", "query_vis"],
                      default="last",
                      help="fastv: rank by the last token's attention at layer K-1 (FastV), or, for POPE-style questions, "
                           "by the cosine of each image token's input embedding to the object name's (EXP-0011 D3), or "
                           "keep --fastv-mix-question tokens by that cosine and the rest by FastV's score (EXP-0013), or "
                           "by that cosine minus the largest / mean cosine to the other 80 COCO names (EXP-0013); "
                           "query_mean / query_idf: by the cosine to the uniform / IDF-weighted mean input embedding of "
                           "all tokens after the image, IDF over the question file; attn_idf: the scoring layer's "
                           "attention rows of those tokens, IDF-weighted (EXP-0014)")
    ours.add_argument("--fastv-route", choices=["yesno_howmany"], default=None,
                      help="fastv, --fastv-score query_vis (EXP-0020): only questions in yes/no form (first word an "
                           "auxiliary verb) or starting with 'how many' take query_vis; the others take the strong "
                           "baseline's path (FastV's prior-divided score in blocks)")
    ours.add_argument("--fastv-zoom", type=int, default=0,
                      help="fastv, --fastv-score query_vis (IDEA-0004): insert this many tokens (a square) of a re-encoded "
                           "crop of the box the question's query points at; the stage keeps that many fewer image tokens")
    ours.add_argument("--fastv-zoom-pick", choices=["grid", "query", "pool"], default="grid",
                      help="which crop tokens: a uniform lattice, the best query cosine per block of the crop, or block "
                           "means of the crop's tokens")
    ours.add_argument("--fastv-zoom-source", choices=["crop", "full"], default="crop",
                      help="control: 'full' inserts block means of the image's own tokens instead (needs pick pool)")
    ours.add_argument("--fastv-zoom-centre", action="store_true",
                      help="control: the box keeps its side but sits at the image's centre")
    ours.add_argument("--fastv-zoom-on", choices=["unrouted", "all"], default="unrouted",
                      help="zoom / cover the router's unrouted questions only (needs --fastv-route), or every question")
    ours.add_argument("--fastv-zoom-box", default="8,8,16",
                      help="top_m,min_side,max_side: the box around the top_m tokens by query cosine, its side in patches")
    ours.add_argument("--fastv-zoom-smin", type=float, default=0.2,
                      help="locate only when the question's visible share S is at least this")
    ours.add_argument("--fastv-cover", type=int, default=0,
                      help="fastv, query_vis (IDEA-0004), unrouted questions: keep the strong baseline's own count of "
                           "tokens inside the box plus this many, the rest outside")
    ours.add_argument("--fastv-zoom-log", default=None, help="write one zoom / cover record per prefill (JSON)")
    ours.add_argument("--fastv-mix-question", type=int, default=None,
                      help="fastv, --fastv-score mix: number of kept tokens chosen by the object-name cosine")
    ours.add_argument("--fastv-exclude-padding", action="store_true",
                      help="fastv: never keep patches lying wholly in the padding of LLaVA's square image (EXP-0011 D3)")
    ours.add_argument("--name-cos-log", default=None,
                      help="diagnostic: per image, the cosine of every image token's input embedding to the mean input "
                           "embedding of each COCO category name as it appears in a POPE question (.npz, EXP-0013)")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    method_args = ours.parse_args(argv[:split])

    llava = argparse.ArgumentParser()               # same arguments as model_vqa_loader.__main__
    llava.add_argument("--model-path", type=str, default="facebook/opt-350m")
    llava.add_argument("--model-base", type=str, default=None)
    llava.add_argument("--image-folder", type=str, default="")
    llava.add_argument("--question-file", type=str, default="tables/question.jsonl")
    llava.add_argument("--answers-file", type=str, default="answer.jsonl")
    llava.add_argument("--conv-mode", type=str, default="llava_v1")
    llava.add_argument("--num-chunks", type=int, default=1)
    llava.add_argument("--chunk-idx", type=int, default=0)
    llava.add_argument("--temperature", type=float, default=0.2)
    llava.add_argument("--top_p", type=float, default=None)
    llava.add_argument("--num_beams", type=int, default=1)
    llava.add_argument("--max_new_tokens", type=int, default=128)
    llava_args = llava.parse_args(argv[split + 1:])
    if method_args.method == "random" and method_args.seed is None:
        ours.error("--method random needs --seed")
    if method_args.kept_token_log and method_args.method not in ("fastv", "random"):
        ours.error("--kept-token-log needs --method fastv or random")
    if (method_args.score_log or method_args.fastv_score_prior or method_args.fastv_stratify
            or method_args.text_rows_log) and method_args.method != "fastv":
        ours.error("--score-log, --fastv-score-prior, --fastv-stratify and --text-rows-log need --method fastv")
    if method_args.answer_logit_log and method_args.method == "sparsevlm":
        ours.error("--answer-logit-log is not wired for SparseVLM's generate()")
    if method_args.layer_rows_log and method_args.method != "vanilla":
        ours.error("--layer-rows-log needs --method vanilla")
    if (method_args.fastv_score != "last" or method_args.fastv_exclude_padding) and method_args.method != "fastv":
        ours.error("--fastv-score and --fastv-exclude-padding need --method fastv")
    if method_args.fastv_score in ("object_cos", "contrast_max", "contrast_mean", "query_mean", "query_idf",
                                   "attn_idf") and (method_args.text_rows_log or method_args.fastv_score_prior):
        ours.error(f"--fastv-score {method_args.fastv_score} has no attention rows to log and takes no position prior")
    if method_args.fastv_score in ("query_mean", "query_idf", "attn_idf", "query_vis") and method_args.fastv_schedule:
        ours.error(f"--fastv-score {method_args.fastv_score} works with a single stage")
    if (method_args.fastv_score == "query_vis") != bool(method_args.fastv_visibility):
        ours.error("--fastv-score query_vis and --fastv-visibility go together")
    if method_args.fastv_score == "query_vis" and (method_args.text_rows_log or not method_args.fastv_stratify):
        ours.error("--fastv-score query_vis needs --fastv-stratify and no --text-rows-log")
    if method_args.fastv_route and method_args.fastv_score != "query_vis":
        ours.error("--fastv-route needs --fastv-score query_vis")
    zoom_on = method_args.fastv_zoom or method_args.fastv_cover
    if zoom_on and method_args.fastv_score != "query_vis":
        ours.error("--fastv-zoom and --fastv-cover need --fastv-score query_vis")
    if zoom_on and method_args.fastv_zoom_on == "unrouted" and not method_args.fastv_route:
        ours.error("--fastv-zoom-on unrouted needs --fastv-route")
    if method_args.fastv_zoom and round(method_args.fastv_zoom ** 0.5) ** 2 != method_args.fastv_zoom:
        ours.error("--fastv-zoom must be a square number")
    if (method_args.fastv_zoom_log is not None) != bool(zoom_on):
        ours.error("--fastv-zoom-log goes with --fastv-zoom or --fastv-cover")
    if method_args.fastv_cover and method_args.fastv_zoom_on != "unrouted":
        ours.error("--fastv-cover works on the unrouted questions only (--fastv-zoom-on unrouted)")
    if method_args.fastv_zoom_source == "full" and (method_args.fastv_zoom_pick != "pool" or not method_args.fastv_zoom):
        ours.error("--fastv-zoom-source full needs --fastv-zoom n and --fastv-zoom-pick pool")
    if method_args.fastv_zoom_centre and not zoom_on:
        ours.error("--fastv-zoom-centre needs --fastv-zoom or --fastv-cover")
    if method_args.vocab_visibility_log and method_args.method in ("visionzip", "sparsevlm"):
        ours.error("--vocab-visibility-log needs the 576 image tokens at the LLM input (vanilla, fastv or random)")
    if (method_args.fastv_score == "mix") != (method_args.fastv_mix_question is not None):
        ours.error("--fastv-score mix and --fastv-mix-question go together")
    if method_args.fastv_score == "mix" and (method_args.text_rows_log or not method_args.fastv_stratify
                                             or method_args.fastv_schedule):
        ours.error("--fastv-score mix needs --fastv-stratify, a single stage and no --text-rows-log")
    if method_args.name_cos_log and method_args.method in ("visionzip", "sparsevlm"):
        ours.error("--name-cos-log needs the 576 image tokens at the LLM input (vanilla, fastv or random)")
    method_args.conv_mode = llava_args.conv_mode          # the contrast vocabulary is tokenised in the same template
    method_args.question_file = llava_args.question_file  # IDF weights are counted over the evaluated questions
    if method_args.method == "sparsevlm":
        if method_args.retained_tokens is None:
            ours.error("--method sparsevlm needs --retained-tokens")
        if "retained_tokens" not in inspect.getsource(loader.eval_model):
            ours.error(f"--method sparsevlm needs SparseVLM's model_vqa_loader, but {loader.__file__} is not it")
        llava_args.retained_tokens = method_args.retained_tokens
    return method_args, llava_args


def apply_method(model, m, tokenizer=None):
    schedule = getattr(m, "fastv_schedule", None)
    if m.method == "fastv":
        prior_path = getattr(m, "fastv_score_prior", None)
        prior = torch.from_numpy(np.load(prior_path)) if prior_path else None
        score_kind = getattr(m, "fastv_score", "last")
        span = (lambda ids: question_spans(tokenizer, ids)[1]) if score_kind != "last" else None
        contrast_ids = None
        if score_kind.startswith("contrast"):                  # the 80 COCO names, as they appear in a POPE question
            contrast_ids = getattr(m, "fastv_contrast_ids", None) or list(
                _name_token_ids(tokenizer, coco_names(), getattr(m, "conv_mode", "vicuna_v1")).values())
        weights = None
        if score_kind == "query_mean":
            weights = lambda ids: [1.0] * len(ids)             # noqa: E731
        elif score_kind in ("query_idf", "attn_idf", "query_vis"):     # IDF of each token over the evaluated question file
            idf = getattr(m, "fastv_idf", None) or question_idf(tokenizer, m.question_file, getattr(m, "conv_mode", "vicuna_v1"))
            m.fastv_idf = idf
            weights = lambda ids: [idf["idf"].get(t, idf["unseen"]) for t in ids]   # noqa: E731
        route = None
        if getattr(m, "fastv_route", None) == "yesno_howmany":       # decided from the prompt text after the image
            route = lambda ids: routes_to_query(tokenizer.decode(ids))   # noqa: E731
        zoom, cover = None, getattr(m, "fastv_cover", 0)
        if getattr(m, "fastv_zoom", 0) or cover:                       # IDEA-0004
            top_m, min_side, max_side = (int(x) for x in getattr(m, "fastv_zoom_box", "8,8,16").split(","))
            zoom = {"n": getattr(m, "fastv_zoom", 0), "pick": getattr(m, "fastv_zoom_pick", "grid"),
                    "on": getattr(m, "fastv_zoom_on", "unrouted"), "top_m": top_m, "min_side": min_side,
                    "max_side": max_side, "s_min": getattr(m, "fastv_zoom_smin", 0.2), "cover": bool(cover),
                    "source": getattr(m, "fastv_zoom_source", "crop"), "centre": bool(getattr(m, "fastv_zoom_centre", False))}
        return apply_fastv(model, k=m.fastv_k, keep=m.fastv_keep, schedule=schedule, score_prior=prior,
                           stratify=getattr(m, "fastv_stratify", None),
                           log_text_rows=bool(getattr(m, "text_rows_log", None)), score_kind=score_kind,
                           object_span=span if score_kind in ("object_cos", "mix", "contrast_max", "contrast_mean") else None,
                           exclude_padding=bool(getattr(m, "fastv_exclude_padding", False)),
                           mix_question=getattr(m, "fastv_mix_question", None), contrast_ids=contrast_ids,
                           token_weights=weights,
                           visibility=np.load(m.fastv_visibility) if score_kind == "query_vis" else None, route=route,
                           zoom=zoom, cover=cover)
    if m.method == "random":
        return apply_fastv(model, k=m.fastv_k, keep=m.fastv_keep, seed=m.seed, schedule=schedule)
    if m.method == "visionzip":
        root = os.environ.get("VISIONZIP_ROOT")
        if root:
            sys.path.insert(0, root)
        from visionzip import visionzip
        return visionzip(model, dominant=m.dominant, contextual=m.contextual)
    return model


def _give_worker_args(llava_args, worker_id):
    loader.args = llava_args


_LAYER_TOKENS = []           # one dict per prefill: layer index -> sequence length entering that LLM layer
_LAYER_EXTRA = []            # IDEA-0004: zoom tokens inserted after the image in each prefill (0 without zoom)


def _record_layer_tokens(index, module, args, kwargs):
    # SparseVLM passes hidden_states by keyword to its pruning layers (2, 6, 15) and positionally to the others
    hidden = args[0] if args else kwargs.get("hidden_states")
    if hidden is not None and hidden.shape[1] > 1:      # prefill only; a decode step carries one token
        if index == 0:
            _LAYER_TOKENS.append({})
            _LAYER_EXTRA.append(int(getattr(_ZOOM_MODEL[0], "fastv_extra", 0)) if _ZOOM_MODEL else 0)
        _LAYER_TOKENS[-1][index] = int(hidden.shape[1])


def _write_layer_tokens(path, n_layers):
    # Visual tokens per layer = length - text tokens, with text = layer-0 length - 576 - zoom tokens: valid for methods
    # that give layer 0 all 576 patches (vanilla, FastV, random, SparseVLM; with zoom, the inserted tokens count as
    # visual), not for VisionZip, which shrinks them before the LLM.
    rows = [(row, e) for row, e in zip(_LAYER_TOKENS, _LAYER_EXTRA) if sorted(row) == list(range(n_layers))]
    visual = [[row[i] - (row[0] - 576 - e) for i in range(n_layers)] for row, e in rows]
    per_layer = [sum(col) / len(col) for col in zip(*visual)] if visual else []
    record = {"n_prefills": len(_LAYER_TOKENS), "n_complete": len(rows), "n_layers": n_layers,
              "tokens_per_layer": [[row.get(i) for i in range(n_layers)] for row in _LAYER_TOKENS],
              "mean_visual_per_layer": per_layer,
              "mean_visual_over_layers": sum(per_layer) / len(per_layer) if per_layer else None}
    if _ZOOM_MODEL:
        record["zoom_tokens"] = _LAYER_EXTRA
    with open(path, "w") as f:
        json.dump(record, f, indent=1)


_KEPT = []                   # one list per prefill: indices (0..n_img-1, row-major patch order) of the kept image tokens
_KEPT_N_IMG = set()          # image-token count of each prefill (576 for LLaVA-1.5 at 336 px)
_KEPT_QUESTION = []          # --fastv-score mix: the kept patches chosen by the object-name cosine, per prefill
_ROUTED = []                 # --fastv-route: whether each prefill took query_vis
_KEPT_ZOOM = []              # IDEA-0004: zoom tokens kept besides the listed image patches, per prefill


def _record_kept(module, args, kwargs, output):
    embeds = kwargs.get("inputs_embeds")
    if embeds is not None and embeds.shape[1] > 1:      # prefill only: generate() decodes from input_ids
        _KEPT.append(module.fastv_kept_image_positions.tolist())
        _KEPT_N_IMG.add(int(module.fastv_image_span[1]))
        if getattr(module, "fastv_score_kind", None) in ("mix", "query_vis"):
            _KEPT_QUESTION.append(module.fastv_kept_question_positions.tolist())
        if getattr(module, "fastv_route", None) is not None:
            _ROUTED.append(module.fastv_routed)
        if getattr(module, "fastv_zoom", None):
            _KEPT_ZOOM.append(int(module.fastv_extra))


def _write_kept(path, keep):
    record = {"n_prefills": len(_KEPT), "keep": keep, "n_image_tokens": sorted(_KEPT_N_IMG), "kept": _KEPT}
    if _KEPT_QUESTION:
        record["kept_by_object_name"] = _KEPT_QUESTION
    if _ROUTED:
        record["routed"] = _ROUTED
    if _KEPT_ZOOM:
        record["zoom_tokens"] = _KEPT_ZOOM
    with open(path, "w") as f:
        json.dump(record, f)


_NAME_COS = {"cos": [], "index": [], "seen": {}}     # one (576, n_names) array per distinct image, an index per prefill


def prompt_ids_after_image(tokenizer, text, conv_mode):
    """The prompt ids after the image token for one question, built exactly as model_vqa_loader builds the prompt."""
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
    return ids[ids.index(IMAGE_TOKEN_INDEX) + 1:]


def question_idf(tokenizer, question_file, conv_mode):
    """IDF of every token id after the image over the questions of question_file (each question one document; plus-one
    smoothing as in BERTScore): log((N + 1) / (df + 1)); a token absent from the file gets log(N + 1)."""
    import math
    with open(question_file) as f:
        texts = [json.loads(line)["text"] for line in f]
    df = {}
    for text in texts:
        for t in set(prompt_ids_after_image(tokenizer, text, conv_mode)):
            df[t] = df.get(t, 0) + 1
    n = len(texts)
    return {"n_questions": n, "idf": {t: math.log((n + 1) / (c + 1)) for t, c in df.items()}, "unseen": math.log(n + 1),
            "df": df}


ROUTE_START = ("is ", "are ", "does ", "do ", "did ", "can ", "could ", "was ", "were ", "has ", "have ", "had ", "will ",
               "would ", "should ", "shall ", "may ", "might ", "must ", "how many ")


def routes_to_query(prompt_after_image):
    """EXP-0020's router: True when the question (the first non-empty line of the prompt after the image) is in yes/no
    form (starts with an auxiliary verb) or asks 'how many', the forms whose answer turns on the named object itself."""
    question = prompt_after_image.strip().split("\n")[0].strip().lower()
    return question.startswith(ROUTE_START)


def coco_names():
    """The 80 COCO category names in category-id order."""
    with open(COCO_INSTANCES, encoding="utf-8") as f:
        return [c["name"] for c in sorted(json.load(f)["categories"], key=lambda c: c["id"])]


def _name_token_ids(tokenizer, names, conv_mode):
    """The ids of each name inside a POPE question, from the prompt exactly as model_vqa_loader builds it."""
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    out = {}
    for name in names:
        question = f"Is there {'an' if name[0] in 'aeiou' else 'a'} {name} in the image?"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question
                            + "\nAnswer the question using a single word or phrase.")
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
        after = ids[ids.index(IMAGE_TOKEN_INDEX) + 1:]
        _, o_idx, found = question_spans(tokenizer, after)
        if found != name:
            raise RuntimeError(f"name span of {name!r} reads {found!r}")
        out[name] = [after[i] for i in o_idx]
    return out


def _log_name_cos(lm, name_ids):
    """Forward hook on the LlamaModel: at each prefill, the cosine of every image token's input embedding to each name's
    mean input embedding (context-free lookups, as the object-name score uses), stored once per distinct image."""
    names = list(name_ids)
    weight = lm.embed_tokens.weight
    name_emb = torch.stack([weight[torch.tensor(name_ids[n], device=weight.device)].float().mean(0) for n in names])
    name_emb = torch.nn.functional.normalize(name_emb, dim=-1)

    def hook(module, args, kwargs, output):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None or embeds.shape[1] == 1:
            return
        start, n_img = module.fastv_image_span
        img = embeds[0, start:start + n_img].float()
        key = (n_img, float(img.sum()), float((img * img).sum()), float(img[0, 0]), float(img[-1, -1]))
        if key not in _NAME_COS["seen"]:
            _NAME_COS["seen"][key] = len(_NAME_COS["cos"])
            _NAME_COS["cos"].append((torch.nn.functional.normalize(img, dim=-1) @ name_emb.T).cpu().numpy())
        _NAME_COS["index"].append(_NAME_COS["seen"][key])

    lm.register_forward_hook(hook, with_kwargs=True)
    return names


_VOCAB_VIS = {"max": [], "mean": [], "seen": set()}     # per distinct image: largest and mean cosine of every vocab id


def _log_vocab_visibility(lm):
    """Forward hook on the LlamaModel: at each prefill of a new image, the cosine of every image token's input embedding to
    every vocabulary embedding, reduced over the image tokens to its largest and mean value (one vector per image)."""
    vocab = torch.nn.functional.normalize(lm.embed_tokens.weight.detach().float(), dim=-1)

    def hook(module, args, kwargs, output):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None or embeds.shape[1] == 1:
            return
        start, n_img = module.fastv_image_span
        img = embeds[0, start:start + n_img].float()
        key = (n_img, float(img.sum()), float((img * img).sum()), float(img[0, 0]), float(img[-1, -1]))
        if key in _VOCAB_VIS["seen"]:
            return
        _VOCAB_VIS["seen"].add(key)
        sims = torch.nn.functional.normalize(img, dim=-1) @ vocab.T            # (n_img, vocab)
        _VOCAB_VIS["max"].append(sims.max(0).values.half().cpu().numpy())
        _VOCAB_VIS["mean"].append(sims.mean(0).half().cpu().numpy())

    lm.register_forward_hook(hook, with_kwargs=True)


def _write_vocab_visibility(path):
    np.savez_compressed(path, max=np.stack(_VOCAB_VIS["max"]), mean=np.stack(_VOCAB_VIS["mean"]))


def _write_name_cos(path, names, name_ids):
    np.savez_compressed(path, cos=np.stack(_NAME_COS["cos"]).astype(np.float16), index=np.array(_NAME_COS["index"]),
                        names=np.array(names), name_ids=np.array([json.dumps(name_ids[n]) for n in names]))


_SCORES = []                 # one vector per prefill: FastV's score of every image patch, before any prior


def _record_scores(module, args, kwargs, output):
    embeds = kwargs.get("inputs_embeds")
    if embeds is not None and embeds.shape[1] > 1:
        _SCORES.append(module.fastv_last_scores.float().cpu().numpy())


_TEXT_ROWS = []              # one (rows, token ids) pair per prefill: attention of each text token after the image


def _record_text_rows(module, args, kwargs, output):
    embeds = kwargs.get("inputs_embeds")
    if embeds is not None and embeds.shape[1] > 1:
        rows, ids = module.fastv_text_rows.half().cpu().numpy(), module.fastv_text_after_ids
        if rows.shape[0] != len(ids):
            raise RuntimeError(f"{rows.shape[0]} attention rows for {len(ids)} text tokens after the image")
        _TEXT_ROWS.append((rows, ids))


def _write_text_rows(path):
    lengths = [len(ids) for _, ids in _TEXT_ROWS]
    np.savez_compressed(path, rows=np.concatenate([rows for rows, _ in _TEXT_ROWS]),
                        offsets=np.cumsum([0] + lengths),
                        token_ids=np.concatenate([np.asarray(ids, dtype=np.int64) for _, ids in _TEXT_ROWS]))


ANSWER_TOKENS = ["▁Yes", "▁No", "▁yes", "▁no"]
_LOGITS = []                 # one dict per prefill: log-probabilities of ANSWER_TOKENS as the first answer token


def _record_logits(candidate_ids, module, args, output):
    if output.shape[1] > 1:          # prefill: the last position predicts the first answer token
        logp = torch.log_softmax(output[0, -1].float(), dim=-1)
        _LOGITS.append({"logprobs": logp[candidate_ids].tolist(), "argmax": int(logp.argmax())})


def _write_logits(path, candidate_ids):
    with open(path, "w") as f:
        json.dump({"tokens": ANSWER_TOKENS, "ids": candidate_ids, "n_prefills": len(_LOGITS), "first_token": _LOGITS}, f)


LAYER_ROWS_LAYERS = (0, 1, 2, 4, 8, 16, 31)      # head-averaged rows (EXP-0011 D2b)
HEAD_ROW_LAYERS = (0, 1)                         # per-head rows: the layers before FastV's K = 2
STATE_LEVELS = (0, 1, 2)                         # hidden_states index: input embeddings, outputs of layers 0 and 1
_LAYER_ROWS = []             # one dict per prefill: the D2b statistics of that prefill


def _contiguous(idx):
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise RuntimeError(f"token span {idx} is not contiguous")
    return [idx[0], idx[-1] + 1]


def _log_layer_rows(lm, tokenizer):
    """Wrap LlamaModel.forward so that each prefill also returns attentions and hidden states (which makes it run the
    eager kernel), keep the D2b statistics, and return the output without them; decode steps are left alone."""
    orig = lm.forward
    lm.layer_rows_orig_forward = orig

    def forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None or embeds.shape[1] == 1:
            return orig(*args, **kwargs)
        out = orig(*args, **{**kwargs, "output_attentions": True, "output_hidden_states": True, "return_dict": True})
        start, n_img = lm.fastv_image_span
        ids = lm.fastv_text_after_ids
        q_idx, o_idx, _ = question_spans(tokenizer, ids)
        img = slice(start, start + n_img)
        q_pos = [start + n_img + i for i in q_idx]
        o_pos = [start + n_img + i for i in o_idx]
        avg = []
        for layer in LAYER_ROWS_LAYERS:
            a = out.attentions[layer].mean(dim=1)[0]                    # head average, as FastV computes it
            avg.append(torch.stack([a[-1, img].float(), a[q_pos, img].float().mean(0), a[o_pos, img].float().mean(0)]))
        head = [torch.stack([out.attentions[layer][0][:, pos, img].float().mean(1) for pos in (q_pos, o_pos)])
                for layer in HEAD_ROW_LAYERS]
        cos = []
        for level in STATE_LEVELS:
            h = out.hidden_states[level][0].float()
            cos.append(torch.stack([torch.nn.functional.cosine_similarity(h[img], h[pos].mean(0, keepdim=True), dim=-1)
                                    for pos in (q_pos, o_pos)]))
        _LAYER_ROWS.append({"avg": torch.stack(avg).half().cpu().numpy(), "head": torch.stack(head).half().cpu().numpy(),
                            "cos": torch.stack(cos).cpu().numpy(), "ids": list(ids),
                            "question": _contiguous(q_idx), "object": _contiguous(o_idx)})
        return BaseModelOutputWithPast(last_hidden_state=out.last_hidden_state, past_key_values=out.past_key_values)

    lm.forward = forward


def _write_layer_rows(path):
    lengths = [len(r["ids"]) for r in _LAYER_ROWS]
    np.savez_compressed(path, avg_rows=np.stack([r["avg"] for r in _LAYER_ROWS]),
                        head_rows=np.stack([r["head"] for r in _LAYER_ROWS]),
                        cos=np.stack([r["cos"] for r in _LAYER_ROWS]),
                        question_span=np.array([r["question"] for r in _LAYER_ROWS]),
                        object_span=np.array([r["object"] for r in _LAYER_ROWS]),
                        offsets=np.cumsum([0] + lengths),
                        token_ids=np.concatenate([np.asarray(r["ids"], dtype=np.int64) for r in _LAYER_ROWS]),
                        layers=np.array(LAYER_ROWS_LAYERS), head_layers=np.array(HEAD_ROW_LAYERS),
                        state_levels=np.array(STATE_LEVELS))


def main():
    method_args, llava_args = parse()
    original_load = loader.load_pretrained_model
    times = {"start": time.time()}

    def load_and_patch(*a, **kw):
        if method_args.attn_implementation:
            kw["attn_implementation"] = method_args.attn_implementation   # LLaVA's builder forwards kwargs
        tokenizer, model, image_processor, context_len = original_load(*a, **kw)
        model = apply_method(model, method_args, tokenizer)
        if method_args.fastv_zoom or method_args.fastv_cover:    # IDEA-0004: the original image of every prefill, in order
            lm = model.get_model()
            lm.fastv_image_processor = image_processor
            with open(os.path.expanduser(llava_args.question_file)) as f:
                lm.fastv_zoom_paths = [os.path.join(llava_args.image_folder, json.loads(line)["image"]) for line in f]
            if llava_args.num_chunks != 1:
                raise SystemExit("--fastv-zoom / --fastv-cover assume one chunk (prefills follow the question file)")
            _ZOOM_MODEL.append(lm)
        if method_args.layer_token_log:
            times["n_layers"] = len(model.get_model().layers)
            for i, layer in enumerate(model.get_model().layers):
                layer.register_forward_pre_hook(functools.partial(_record_layer_tokens, i), with_kwargs=True)
        if method_args.kept_token_log:
            model.get_model().register_forward_hook(_record_kept, with_kwargs=True)
        if method_args.score_log:
            model.get_model().register_forward_hook(_record_scores, with_kwargs=True)
        if method_args.text_rows_log:
            model.get_model().register_forward_hook(_record_text_rows, with_kwargs=True)
        if method_args.answer_logit_log:
            ids = tokenizer.convert_tokens_to_ids(ANSWER_TOKENS)
            if tokenizer.unk_token_id in ids:
                raise SystemExit(f"answer tokens {ANSWER_TOKENS} map to {ids}, which includes the unknown token")
            times["answer_ids"] = ids
            model.lm_head.register_forward_hook(functools.partial(_record_logits, ids))
        if method_args.layer_rows_log:
            _record_image_span(model)
            _log_layer_rows(model.get_model(), tokenizer)
        if method_args.name_cos_log:
            if method_args.method == "vanilla" and not method_args.layer_rows_log:
                _record_image_span(model)                # FastV and random record the image span themselves
            times["name_ids"] = _name_token_ids(tokenizer, coco_names(), llava_args.conv_mode)
            times["names"] = _log_name_cos(model.get_model(), times["name_ids"])
        if method_args.vocab_visibility_log:
            if method_args.method == "vanilla" and not (method_args.layer_rows_log or method_args.name_cos_log):
                _record_image_span(model)
            _log_vocab_visibility(model.get_model())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times["loaded"] = time.time()                # evaluation timing starts after loading
        return tokenizer, model, image_processor, context_len

    loader.load_pretrained_model = load_and_patch
    loader.args = llava_args                         # CustomDataset reads the module-level `args`
    if method_args.method == "sparsevlm" and os.name == "nt":
        import sparsevlm_sdpa
        sparsevlm_sdpa.apply()

    def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
        # Same dataset, collate_fn and order as LLaVA's create_data_loader. Workers spawned on Windows re-import
        # model_vqa_loader without its module-level `args` (set only under its __main__), so hand it over here.
        dataset = loader.CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
        return DataLoader(dataset, batch_size=batch_size, num_workers=method_args.num_workers, shuffle=False,
                          collate_fn=loader.collate_fn, worker_init_fn=functools.partial(_give_worker_args, llava_args))

    loader.create_data_loader = create_data_loader
    loader.eval_model(llava_args)
    torch.cuda.synchronize()
    end = time.time()
    answers_file = os.path.expanduser(llava_args.answers_file)
    with open(answers_file) as f:
        n_answers = sum(1 for _ in f)
    eval_seconds = end - times["loaded"]
    meta = {"method": vars(method_args), "llava_args": vars(llava_args), "loader": loader.__file__,
            "n_answers": n_answers, "load_seconds": round(times["loaded"] - times["start"], 1),
            "eval_seconds": round(eval_seconds, 1), "seconds_per_answer": round(eval_seconds / max(n_answers, 1), 4),
            "eval_peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "wall_seconds": round(end - times["start"], 1), "gpu": torch.cuda.get_device_name(0),
            "flash_attention_stand_in": "sdpa" if method_args.method == "sparsevlm" and os.name == "nt" else None}
    with open(answers_file + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if getattr(method_args, "fastv_idf", None):                # the IDF table the query weights came from
        idf = method_args.fastv_idf
        with open(answers_file + ".idf.json", "w") as f:
            json.dump({"question_file": llava_args.question_file, "n_questions": idf["n_questions"],
                       "unseen": idf["unseen"], "df": {str(t): c for t, c in sorted(idf["df"].items())}}, f)
    if method_args.layer_token_log:
        _write_layer_tokens(method_args.layer_token_log, times["n_layers"])
    if method_args.kept_token_log:
        schedule = method_args.fastv_schedule
        _write_kept(method_args.kept_token_log, schedule[-1][1] if schedule else method_args.fastv_keep)
    if method_args.score_log:
        np.save(method_args.score_log, np.stack(_SCORES))
    if method_args.text_rows_log:
        _write_text_rows(method_args.text_rows_log)
    if method_args.answer_logit_log:
        _write_logits(method_args.answer_logit_log, times["answer_ids"])
    if method_args.layer_rows_log:
        _write_layer_rows(method_args.layer_rows_log)
    if method_args.name_cos_log:
        _write_name_cos(method_args.name_cos_log, times["names"], times["name_ids"])
    if method_args.vocab_visibility_log:
        _write_vocab_visibility(method_args.vocab_visibility_log)
    if method_args.fastv_zoom_log:                              # IDEA-0004: one record per prefill
        log = _ZOOM_MODEL[0].fastv_zoom_log if _ZOOM_MODEL else []
        with open(method_args.fastv_zoom_log, "w") as f:
            json.dump({"n_prefills": len(log), "n_zoomed": sum(r["zoom"] for r in log),
                       "n_covered": sum(r["cover"] for r in log), "records": log}, f)


_ZOOM_MODEL = []


if __name__ == "__main__":
    main()
