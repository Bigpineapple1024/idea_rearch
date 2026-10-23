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

from fastv_llava import apply_fastv


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
    ours.add_argument("--fastv-score-prior", default=None,
                      help="diagnostic, fastv only: .npy with one value per image patch; scores are divided by it")
    ours.add_argument("--fastv-stratify", type=int, default=None,
                      help="diagnostic, fastv only: take the ranking in turns from an n x n grid of image blocks")
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
    if (method_args.score_log or method_args.fastv_score_prior or method_args.fastv_stratify) and method_args.method != "fastv":
        ours.error("--score-log, --fastv-score-prior and --fastv-stratify need --method fastv")
    if method_args.method == "sparsevlm":
        if method_args.retained_tokens is None:
            ours.error("--method sparsevlm needs --retained-tokens")
        if "retained_tokens" not in inspect.getsource(loader.eval_model):
            ours.error(f"--method sparsevlm needs SparseVLM's model_vqa_loader, but {loader.__file__} is not it")
        llava_args.retained_tokens = method_args.retained_tokens
    return method_args, llava_args


def apply_method(model, m):
    schedule = getattr(m, "fastv_schedule", None)
    if m.method == "fastv":
        prior_path = getattr(m, "fastv_score_prior", None)
        prior = torch.from_numpy(np.load(prior_path)) if prior_path else None
        return apply_fastv(model, k=m.fastv_k, keep=m.fastv_keep, schedule=schedule, score_prior=prior,
                           stratify=getattr(m, "fastv_stratify", None))
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


def _record_layer_tokens(index, module, args, kwargs):
    # SparseVLM passes hidden_states by keyword to its pruning layers (2, 6, 15) and positionally to the others
    hidden = args[0] if args else kwargs.get("hidden_states")
    if hidden is not None and hidden.shape[1] > 1:      # prefill only; a decode step carries one token
        if index == 0:
            _LAYER_TOKENS.append({})
        _LAYER_TOKENS[-1][index] = int(hidden.shape[1])


def _write_layer_tokens(path, n_layers):
    # Visual tokens per layer = length - text tokens, with text = layer-0 length - 576: valid for methods that give
    # layer 0 all 576 patches (vanilla, FastV, random, SparseVLM), not for VisionZip, which shrinks them before the LLM.
    complete = [row for row in _LAYER_TOKENS if sorted(row) == list(range(n_layers))]
    visual = [[row[i] - (row[0] - 576) for i in range(n_layers)] for row in complete]
    per_layer = [sum(col) / len(col) for col in zip(*visual)] if visual else []
    with open(path, "w") as f:
        json.dump({"n_prefills": len(_LAYER_TOKENS), "n_complete": len(complete), "n_layers": n_layers,
                   "tokens_per_layer": [[row.get(i) for i in range(n_layers)] for row in _LAYER_TOKENS],
                   "mean_visual_per_layer": per_layer,
                   "mean_visual_over_layers": sum(per_layer) / len(per_layer) if per_layer else None}, f, indent=1)


_KEPT = []                   # one list per prefill: indices (0..n_img-1, row-major patch order) of the kept image tokens
_KEPT_N_IMG = set()          # image-token count of each prefill (576 for LLaVA-1.5 at 336 px)


def _record_kept(module, args, kwargs, output):
    embeds = kwargs.get("inputs_embeds")
    if embeds is not None and embeds.shape[1] > 1:      # prefill only: generate() decodes from input_ids
        _KEPT.append(module.fastv_kept_image_positions.tolist())
        _KEPT_N_IMG.add(int(module.fastv_image_span[1]))


def _write_kept(path, keep):
    with open(path, "w") as f:
        json.dump({"n_prefills": len(_KEPT), "keep": keep, "n_image_tokens": sorted(_KEPT_N_IMG), "kept": _KEPT}, f)


_SCORES = []                 # one vector per prefill: FastV's score of every image patch, before any prior


def _record_scores(module, args, kwargs, output):
    embeds = kwargs.get("inputs_embeds")
    if embeds is not None and embeds.shape[1] > 1:
        _SCORES.append(module.fastv_last_scores.float().cpu().numpy())


def main():
    method_args, llava_args = parse()
    original_load = loader.load_pretrained_model
    times = {"start": time.time()}

    def load_and_patch(*a, **kw):
        if method_args.attn_implementation:
            kw["attn_implementation"] = method_args.attn_implementation   # LLaVA's builder forwards kwargs
        tokenizer, model, image_processor, context_len = original_load(*a, **kw)
        model = apply_method(model, method_args)
        if method_args.layer_token_log:
            times["n_layers"] = len(model.get_model().layers)
            for i, layer in enumerate(model.get_model().layers):
                layer.register_forward_pre_hook(functools.partial(_record_layer_tokens, i), with_kwargs=True)
        if method_args.kept_token_log:
            model.get_model().register_forward_hook(_record_kept, with_kwargs=True)
        if method_args.score_log:
            model.get_model().register_forward_hook(_record_scores, with_kwargs=True)
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
    if method_args.layer_token_log:
        _write_layer_tokens(method_args.layer_token_log, times["n_layers"])
    if method_args.kept_token_log:
        schedule = method_args.fastv_schedule
        _write_kept(method_args.kept_token_log, schedule[-1][1] if schedule else method_args.fastv_keep)
    if method_args.score_log:
        np.save(method_args.score_log, np.stack(_SCORES))


if __name__ == "__main__":
    main()
