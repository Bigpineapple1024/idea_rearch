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
"""
import argparse
import functools
import inspect
import json
import os
import sys
import time

import torch
from llava.eval import model_vqa_loader as loader
from torch.utils.data import DataLoader

from fastv_llava import apply_fastv


def parse():
    ours = argparse.ArgumentParser()
    ours.add_argument("--method", choices=["vanilla", "fastv", "visionzip", "random", "sparsevlm"], required=True)
    ours.add_argument("--fastv-k", type=int, default=2)
    ours.add_argument("--fastv-keep", type=int, default=288)
    ours.add_argument("--seed", type=int, default=None, help="random: seed of the token draw")
    ours.add_argument("--dominant", type=int, default=54)
    ours.add_argument("--contextual", type=int, default=10)
    ours.add_argument("--retained-tokens", type=int, default=None, help="sparsevlm: 32-layer average budget")
    ours.add_argument("--attn-implementation", choices=["eager", "sdpa"], default=None,
                      help="load the LLM with this attention kernel (default: transformers' choice, sdpa in 4.37)")
    ours.add_argument("--num-workers", type=int, default=4,
                      help="DataLoader workers (LLaVA's default 4; 0 loads images in this process)")
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
    if method_args.method == "sparsevlm":
        if method_args.retained_tokens is None:
            ours.error("--method sparsevlm needs --retained-tokens")
        if "retained_tokens" not in inspect.getsource(loader.eval_model):
            ours.error(f"--method sparsevlm needs SparseVLM's model_vqa_loader, but {loader.__file__} is not it")
        llava_args.retained_tokens = method_args.retained_tokens
    return method_args, llava_args


def apply_method(model, m):
    if m.method == "fastv":
        return apply_fastv(model, k=m.fastv_k, keep=m.fastv_keep)
    if m.method == "random":
        return apply_fastv(model, k=m.fastv_k, keep=m.fastv_keep, seed=m.seed)
    if m.method == "visionzip":
        root = os.environ.get("VISIONZIP_ROOT")
        if root:
            sys.path.insert(0, root)
        from visionzip import visionzip
        return visionzip(model, dominant=m.dominant, contextual=m.contextual)
    return model


def _give_worker_args(llava_args, worker_id):
    loader.args = llava_args


def main():
    method_args, llava_args = parse()
    original_load = loader.load_pretrained_model
    times = {"start": time.time()}

    def load_and_patch(*a, **kw):
        if method_args.attn_implementation:
            kw["attn_implementation"] = method_args.attn_implementation   # LLaVA's builder forwards kwargs
        tokenizer, model, image_processor, context_len = original_load(*a, **kw)
        model = apply_method(model, method_args)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times["loaded"] = time.time()                # evaluation timing starts after loading
        return tokenizer, model, image_processor, context_len

    loader.load_pretrained_model = load_and_patch
    loader.args = llava_args                         # CustomDataset reads the module-level `args`

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
            "wall_seconds": round(end - times["start"], 1), "gpu": torch.cuda.get_device_name(0)}
    with open(answers_file + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
