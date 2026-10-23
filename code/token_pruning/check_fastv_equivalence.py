"""Check fastv_llava.fastv_forward against FastV's official lmms-eval implementation.

Same random tiny Llama weights, same tokens, the official constants (image span 35..611, K=3, keep 288).
Prints the per-layer KV-cache lengths of both and the relative difference of the last hidden state
for the prefill and 5 greedy-free decode steps (random next tokens).

  python check_fastv_equivalence.py <path to pkunlp-icler/FastV checkout>
"""
import copy
import importlib.util
import sys

import torch
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache

from fastv_llava import bind_fastv

spec = importlib.util.spec_from_file_location("fastv_kvcache", f"{sys.argv[1]}/src/FastV/lmms-eval/fastv_kvcache.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

SEQ, STEPS = 35 + 576 + 24, 5


def run(model):
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (1, SEQ))
    nxt = torch.randint(0, 1000, (STEPS, 1))
    outs = []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        outs.append(out.last_hidden_state[:, -1])
        pkv = out.past_key_values
        for s in range(STEPS):
            out = model(input_ids=nxt[s:s + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            outs.append(out.last_hidden_state[:, -1])
    cache = pkv if isinstance(pkv, DynamicCache) else DynamicCache.from_legacy_cache(pkv)
    return outs, [cache.get_seq_length(i) for i in range(len(cache))]


cfg = LlamaConfig(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)
cfg._attn_implementation = "sdpa"
torch.manual_seed(0)
ref_model = official.FastVLlamaModel(cfg).eval()
ours = copy.deepcopy(ref_model)
bind_fastv(ours, k=3, keep=288)                  # also applies the README rotary fix used by both
ours.fastv_image_span = (35, 576)

ref_out, ref_cache = run(ref_model)
our_out, our_cache = run(ours)
print("KV cache length per layer  official:", ref_cache, " fastv_llava:", our_cache)
for step, (a, b) in enumerate(zip(our_out, ref_out)):
    print(f"{'prefill' if step == 0 else f'decode {step}'}: relative L2 difference = {((a - b).norm() / b.norm()).item():.2e}")
