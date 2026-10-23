"""Code-level probe of FastV's lmms-eval KV-cache implementation (fastv_kvcache.py).

Random-weight tiny Llama on CPU. Checks implementation behaviour only (cache sizes,
position handling, crash conditions); says nothing about accuracy on real models.
Usage: python probe_fastv.py <path to FastV repo>
"""
import importlib.util
import sys

import torch
from transformers import LlamaConfig
from transformers.models.llama import modeling_llama as ml

repo = sys.argv[1]
spec = importlib.util.spec_from_file_location("fastv_kvcache", f"{repo}/src/FastV/lmms-eval/fastv_kvcache.py")
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)

SYS, IMG, TXT = 35, 576, 24          # LLaVA-1.5 layout hard-coded in fastv_kvcache.py (35, 576)
SEQ = SYS + IMG + TXT
K, R = 3, 0.5                         # values hard-coded in fastv_kvcache.py lines 119-120


def make_config(attn):
    cfg = LlamaConfig(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)
    cfg._attn_implementation = attn
    return cfg


def patched_rotary_forward(self, x, seq_len=None):
    # README step 5: use seq_len = position_ids.max()+1; equivalent here to not truncating the cache.
    return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)


def cache_lengths(pkv):
    return [pkv.get_seq_length(i) for i in range(len(pkv))]


def run(attn, patch_rotary):
    torch.manual_seed(0)
    orig = ml.LlamaRotaryEmbedding.forward
    if patch_rotary:
        ml.LlamaRotaryEmbedding.forward = patched_rotary_forward
    try:
        model = fk.FastVLlamaModel(make_config(attn)).eval()
        ids = torch.randint(0, 1000, (1, SEQ))
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            pkv = out.past_key_values
            lens = cache_lengths(ml.DynamicCache.from_legacy_cache(pkv) if isinstance(pkv, tuple) else pkv)
            nxt = torch.randint(0, 1000, (1, 1))
            out2 = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        return {"status": "ok", "prefill_cache_len_per_layer": lens,
                "decode_hidden_norm": round(out2.last_hidden_state.norm().item(), 4)}
    except Exception as e:  # report the failure mode, do not hide it
        return {"status": f"error: {type(e).__name__}: {str(e)[:160]}"}
    finally:
        ml.LlamaRotaryEmbedding.forward = orig


def reference_cache_lengths():
    kept_img = int(IMG * R)
    return [SEQ] * K + [SYS + kept_img + TXT] * (6 - K)


if __name__ == "__main__":
    print("expected per-layer cache length without duplication:", reference_cache_lengths())
    for attn in ("eager", "sdpa"):
        for patch in (False, True):
            print(f"attn={attn:5s} rotary_patch={patch!s:5s} ->", run(attn, patch))
