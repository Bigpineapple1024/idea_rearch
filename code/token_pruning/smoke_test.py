"""CPU smoke test: run each method through the official LLaVA generate() path on a tiny random model.

Checks that the code paths execute and how many tokens each LLM layer sees during prefill.
Random weights: the generated tokens and timings say nothing about accuracy or real speed.

  python smoke_test.py                     # all methods, each in its own process
  python smoke_test.py --method fastv      # one method
Requires: LLaVA installed, VISIONZIP_ROOT pointing at the VisionZip repository.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import torch

SYS_LEN, TEXT_LEN = 35, 20
METHODS = {
    "vanilla": {},
    "fastv_k2_keep288": {"method": "fastv", "fastv_k": 2, "fastv_keep": 288},
    "fastv_k2_keep30": {"method": "fastv", "fastv_k": 2, "fastv_keep": 30},
    "visionzip_54_10": {"method": "visionzip", "dominant": 54, "contextual": 10},
    "random_k2_keep30_seed0": {"method": "random", "fastv_k": 2, "fastv_keep": 30, "seed": 0},
}


def build_tiny_llava(workdir):
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
    from llava.model import LlavaLlamaForCausalLM
    from llava.model.language_model.llava_llama import LlavaConfig

    torch.manual_seed(0)
    clip_dir = os.path.join(workdir, "tiny-clip-336")
    if not os.path.isdir(clip_dir):
        # CLIP-L/14-336 geometry (24 layers, 576 patches); VisionZip hard-codes the 24-layer schedule
        clip_cfg = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=24,
                                    num_attention_heads=4, image_size=336, patch_size=14)
        CLIPVisionModel(clip_cfg).save_pretrained(clip_dir)
        CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336}).save_pretrained(clip_dir)
    cfg = LlavaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                      mm_vision_tower=clip_dir, mm_hidden_size=64, mm_projector_type="mlp2x_gelu",
                      mm_vision_select_layer=-2, mm_vision_select_feature="patch", image_aspect_ratio="pad",
                      mm_use_im_start_end=False, use_cache=True)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)                         # same LLM weights whether or not the CLIP dir was cached
    model = LlavaLlamaForCausalLM(cfg)
    model.get_vision_tower().load_model()
    return model.eval()


def run_one(name, workdir):
    spec = METHODS[name]
    model = build_tiny_llava(workdir)
    if spec:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_eval import apply_method
        model = apply_method(model, argparse.Namespace(**{"fastv_k": 2, "fastv_keep": 288, "seed": None,
                                                          "dominant": 54, "contextual": 10, **spec}))
    from llava.constants import IMAGE_TOKEN_INDEX

    seen = []
    hooks = [layer.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[1]))
             for layer in model.get_model().layers]
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]),
                     torch.randint(3, 1000, (TEXT_LEN,))]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    start = time.time()
    with torch.no_grad():
        out = model.generate(ids, images=images, do_sample=False, max_new_tokens=4, use_cache=True)
    for h in hooks:
        h.remove()
    n_layers = len(model.get_model().layers)
    prefill = seen[:n_layers]
    return {"method": name, "status": "ok", "prefill_tokens_per_layer": prefill,
            "visual_tokens_per_layer": [n - SYS_LEN - TEXT_LEN for n in prefill],
            "decode_calls": len(seen) // n_layers - 1, "generated_ids": out[0].tolist(),
            "cpu_seconds": round(time.time() - start, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "token_pruning_smoke"))
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    if a.method:
        try:
            print(json.dumps(run_one(a.method, a.workdir)))
        except Exception as e:  # report, do not mask
            print(json.dumps({"method": a.method, "status": f"error: {type(e).__name__}: {e}"}))
        return
    for name in METHODS:                        # VisionZip patches classes globally: one process per method
        res = subprocess.run([sys.executable, __file__, "--method", name, "--workdir", a.workdir],
                             capture_output=True, text=True)
        lines = [l for l in res.stdout.splitlines() if l.startswith("{")]
        print(lines[-1] if lines else json.dumps({"method": name, "status": "crashed", "stderr": res.stderr[-800:]}))


if __name__ == "__main__":
    main()
