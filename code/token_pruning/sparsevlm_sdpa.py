"""SDPA stand-in for FlashAttention-2 inside SparseVLM (Gumpest/SparseVLMs v1.5) on Windows.

SparseVLM's builder attaches LlamaDynamicvitFlashAttention2 to all 32 decoder layers (llava/model/builder.py
L114-121); its forward ends in transformers' LlamaFlashAttention2._flash_attention_forward, i.e. flash_attn_func.
flash-attn has no official Windows build, so apply() replaces only that last call with
torch.nn.functional.scaled_dot_product_attention: the same exact attention through another kernel, so numbers
differ at rounding level (as between sdpa and eager). SparseVLM's token scoring (attn_logits, computed explicitly
earlier in the same forward) is untouched. Batch size 1 without padding only, as SparseVLM itself requires.

  python sparsevlm_sdpa.py     # CPU check against explicit softmax attention (prefill and decode steps)
"""
import torch
import torch.nn.functional as F


def sdpa_flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length,
                                 dropout=0.0, softmax_scale=None):
    if attention_mask is not None and not bool(attention_mask.all()):
        raise NotImplementedError("padded inputs are not supported; SparseVLM runs with batch size 1")
    q, k, v = (t.transpose(1, 2) for t in (query_states, key_states, value_states))   # (b, s, h, d) -> (b, h, s, d)
    # flash_attn_func(causal=True) aligns the mask bottom-right: a decode step (one query over the key cache)
    # sees every key; a prefill (as many queries as keys) is ordinary causal attention.
    causal = self.is_causal and query_length > 1
    if causal and q.shape[-2] != k.shape[-2]:
        raise NotImplementedError("causal attention over a longer key cache does not occur in SparseVLM's eval path")
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)


def apply():
    from llava.model.language_model import modelling_sparse_llama as sparse
    sparse.LlamaDynamicvitFlashAttention2._flash_attention_forward = sdpa_flash_attention_forward


def _check():
    from types import SimpleNamespace

    torch.manual_seed(0)
    layer = SimpleNamespace(is_causal=True)
    for q_len, k_len in ((7, 7), (1, 12)):                     # prefill, then one decode step over a cache
        q, k, v = torch.randn(1, q_len, 4, 16), torch.randn(1, k_len, 4, 16), torch.randn(1, k_len, 4, 16)
        got = sdpa_flash_attention_forward(layer, q, k, v, None, q_len)
        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(2, 3) / 16 ** 0.5
        if q_len > 1:
            scores = scores.masked_fill(torch.ones(q_len, k_len, dtype=torch.bool).triu(1), float("-inf"))
        want = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2)
        print(f"q_len={q_len} k_len={k_len}: max |sdpa - explicit| = {(got - want).abs().max().item():.2e}")


if __name__ == "__main__":
    _check()
