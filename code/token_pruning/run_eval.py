"""Run LLaVA's model_vqa_loader evaluation with a visual-token reduction method switched on.

Covers the benchmarks that LLaVA-1.5 evaluates through llava.eval.model_vqa_loader
(POPE, GQA, TextVQA, MME, VQAv2, VizWiz). All LLaVA arguments are passed through unchanged.

  python run_eval.py --method vanilla   -- --model-path liuhaotian/llava-v1.5-7b ...
  python run_eval.py --method fastv --fastv-k 2 --fastv-keep 30 -- ...
  python run_eval.py --method visionzip --dominant 54 --contextual 10 -- ...

VisionZip is imported from its repository root (set VISIONZIP_ROOT or PYTHONPATH): the
visionzip 0.1.3 wheel on PyPI ships no code.
"""
import argparse
import json
import os
import sys
import time

from llava.eval import model_vqa_loader as loader

from fastv_llava import apply_fastv


def parse():
    ours = argparse.ArgumentParser()
    ours.add_argument("--method", choices=["vanilla", "fastv", "visionzip"], required=True)
    ours.add_argument("--fastv-k", type=int, default=2)
    ours.add_argument("--fastv-keep", type=int, default=288)
    ours.add_argument("--dominant", type=int, default=54)
    ours.add_argument("--contextual", type=int, default=10)
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
    return method_args, llava.parse_args(argv[split + 1:])


def apply_method(model, m):
    if m.method == "fastv":
        return apply_fastv(model, k=m.fastv_k, keep=m.fastv_keep)
    if m.method == "visionzip":
        root = os.environ.get("VISIONZIP_ROOT")
        if root:
            sys.path.insert(0, root)
        from visionzip import visionzip
        return visionzip(model, dominant=m.dominant, contextual=m.contextual)
    return model


def main():
    method_args, llava_args = parse()
    original_load = loader.load_pretrained_model

    def load_and_patch(*a, **kw):
        tokenizer, model, image_processor, context_len = original_load(*a, **kw)
        return tokenizer, apply_method(model, method_args), image_processor, context_len

    loader.load_pretrained_model = load_and_patch
    loader.args = llava_args                         # CustomDataset reads the module-level `args`
    if os.name == "nt":
        # Windows spawns DataLoader workers; they re-import model_vqa_loader without `args` and fail
        # in CustomDataset.__getitem__, so load the data in this process instead.
        original_create = loader.create_data_loader

        def create_in_process(*a, **kw):
            return original_create(*a, **{**kw, "num_workers": 0})

        loader.create_data_loader = create_in_process
    start = time.time()
    loader.eval_model(llava_args)
    meta = {"method": vars(method_args), "llava_args": vars(llava_args), "wall_seconds": round(time.time() - start, 1)}
    with open(os.path.expanduser(llava_args.answers_file) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
