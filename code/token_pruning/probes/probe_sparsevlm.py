"""Code-level probes of SparseVLM (Gumpest/SparseVLMs, main branch).

1. attn_postprocess_topk with batch size 2 (score.py).
2. What the 64/96/128/192 "retained tokens" budgets mean per layer (score.py schedules).
3. SparseVLM+ (USE_VERSION=2_0) "gravity correction" divisor: RoPE logits of all-ones q/k,
   computed exactly as modelling_sparse_llama.py lines 243-251, for LLaVA-1.5-7B head_dim=128.
Usage: python probe_sparsevlm.py <path to SparseVLMs repo>
"""
import importlib.util
import math
import os
import sys

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

repo = sys.argv[1]


def load_score(version):
    os.environ["USE_VERSION"] = version
    os.environ["RETAIN_TOKN"] = "64"
    spec = importlib.util.spec_from_file_location(f"score_{version}", f"{repo}/llava/model/language_model/score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = load_score("1_0")
score_v2 = load_score("2_0")

# 1. batch-size-2 behaviour of the top-k mask
print("== 1. attn_postprocess_topk with B=2")
torch.manual_seed(0)
B, H, V0, NV = 2, 4, 35, 576
L = V0 + NV + 20
attn = torch.rand(B, H, L, L)
raters = (torch.zeros(5, dtype=torch.long), torch.arange(5))       # torch.where-style tuple, 5 rater tokens
mask, _, _ = score.attn_postprocess_topk(attn, V0, NV, V0 + NV, raters, layer_idx=2)
print("kept visual tokens per sample:", mask.sum(dim=1).tolist(), "(budget at layer 2 =", score.sparse_token_dict[64][0], ")")

# 2. budgets: average visual tokens per LLM layer (32 layers, pruning before layers 2, 6, 15)
print("\n== 2. budget schedules (visual tokens seen by each of the 32 LLM layers)")
for name, mod in (("v1.0", score), ("v2.0", score_v2)):
    for budget, (a, b, c) in sorted(mod.sparse_token_dict.items()):
        per_layer = [576] * 2 + [a] * 4 + [b] * 9 + [c] * 17
        print(f"{name} RETAIN_TOKN={budget:3d}: stages={a, b, c}  mean over layers={sum(per_layer) / 32:6.1f}"
              f"  first 2 layers=576  last 17 layers={c}")

# 3. gravity-correction divisor
print("\n== 3. SparseVLM+ divisor rope_weights[q, k] for all-ones q/k (head_dim=128, base=10000)")
head_dim, seq = 128, 700
rot = LlamaRotaryEmbedding(head_dim, max_position_embeddings=4096)
q = torch.ones(1, 1, seq, head_dim)
k = torch.ones(1, 1, seq, head_dim)
pos = torch.arange(seq).unsqueeze(0)
cos, sin = rot(k, seq_len=seq)
qr, kr = apply_rotary_pos_emb(q, k, cos, sin, pos)
w = (qr @ kr.transpose(2, 3) / math.sqrt(head_dim))[0, 0]
text_q = 650                                       # a text token after 35 system + 576 visual tokens
row = w[text_q, 35:611]
dist = text_q - torch.arange(35, 611)
print(f"divisor for text query at {text_q} over the 576 visual keys: min={row.min():.3f} max={row.max():.3f}")
print("visual keys with divisor <= 0:", int((row <= 0).sum()), "| with 0 < divisor < 0.5:",
      int(((row > 0) & (row < 0.5)).sum()))
for d in (40, 100, 200, 300, 400, 500, 600):
    print(f"  distance {d:3d}: divisor = {w[text_q, text_q - d]:.3f}")
