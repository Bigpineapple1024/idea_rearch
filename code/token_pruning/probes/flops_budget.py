"""Prefill FLOPs / KV-cache size of LLaVA-1.5-7B's LLM under different "64 visual token" definitions.

Per-layer FLOPs formula copied from SparseVLM modelling_sparse_llama.py line 219: 4nd^2 + 2n^2d + 3ndm.
LLaVA-1.5-7B LLM (Vicuna-7B): 32 layers, d=4096, FFN m=11008. System prompt 35 tokens, text T tokens.
"""
D, M, LAYERS, SYS = 4096, 11008, 32, 35


def layer_flops(n):
    return 4 * n * D**2 + 2 * n**2 * D + 3 * n * D * M


def schedule(name, vis_per_layer, text):
    flops = sum(layer_flops(SYS + v + text) for v in vis_per_layer)
    kv_mb = sum(2 * (SYS + v + text) * D * 2 for v in vis_per_layer) / 2**20   # K and V, fp16
    return name, sum(vis_per_layer) / LAYERS, flops / 1e12, kv_mb


def fastv(k, kept):
    return [576] * k + [kept] * (LAYERS - k)


def sparsevlm(a, b, c):                       # pruning before layers 2, 6, 15
    return [576] * 2 + [a] * 4 + [b] * 9 + [c] * 17


for text in (20, 60):
    rows = [
        schedule("vanilla (576)", [576] * LAYERS, text),
        schedule("VisionZip 54+10", [64] * LAYERS, text),
        schedule("FastV K=2, keep 30 (mean 64)", fastv(2, 30), text),
        schedule("FastV K=2, R=50% (keep 288)", fastv(2, 288), text),
        schedule("SparseVLM v1.0 '64' (66,30,17)", sparsevlm(66, 30, 17), text),
        schedule("SparseVLM+ v2.0 '64' (66,34,20)", sparsevlm(66, 34, 20), text),
    ]
    base = rows[0][2]
    print(f"\ntext tokens = {text}")
    print(f"{'method':34s} {'mean vis/layer':>14s} {'TFLOPs':>8s} {'% of vanilla':>12s} {'KV cache MB':>11s}")
    for name, mean_v, tf, kv in rows:
        print(f"{name:34s} {mean_v:14.1f} {tf:8.2f} {100 * tf / base:11.1f}% {kv:11.1f}")
