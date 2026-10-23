"""FastV for the official LLaVA codebase (haotian-liu/LLaVA, transformers 4.37.2), with KV cache.

Same selection rule as FastV's lmms-eval implementation (src/FastV/lmms-eval/fastv_kvcache.py):
before layer K, keep the `keep` image tokens with the highest head-averaged attention from the
last prompt token at layer K-1; all other tokens are kept; original position ids are kept.

Differences from that file, all deliberate:
- K and the number of kept tokens are arguments instead of constants (3, 0.5) in the code.
- The image span is read from the actual input instead of the hard-coded 35/576.
- Layer K-1 runs once (with output_attentions) instead of twice, so its KV cache is not duplicated.
- Rotary embeddings are not truncated to the pruned length (FastV README, lmms-eval step 5).
Batch size 1 without padding only, as in all three reference implementations.

With a seed, the kept image tokens are drawn uniformly at random instead (same K, same count), so the
attention ranking can be compared with chance at an identical budget; layer K-1 then skips its attention map.
"""
import types

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import modeling_llama

from llava.constants import IMAGE_TOKEN_INDEX


def _full_rotary_forward(self, x, seq_len=None):
    if seq_len is not None and seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
    return self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype)


def _causal_mask(hidden):
    n = hidden.shape[1]
    return modeling_llama._prepare_4d_causal_attention_mask(None, (1, n), hidden, 0)


def fastv_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                  inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                  return_dict=None):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    bsz, q_len, _ = inputs_embeds.shape
    if bsz != 1:
        raise ValueError("fastv_forward supports batch size 1 only")
    use_cache = self.config.use_cache if use_cache is None else use_cache
    return_legacy = not isinstance(past_key_values, Cache)
    cache = None
    if use_cache:
        cache = past_key_values if isinstance(past_key_values, Cache) else DynamicCache.from_legacy_cache(past_key_values)
    past_len = cache.get_seq_length(0) if cache is not None else 0
    if position_ids is None:
        position_ids = torch.arange(past_len, past_len + q_len, device=inputs_embeds.device).unsqueeze(0)

    k, keep = self.fastv_k, self.fastv_keep
    prefill = q_len > 1
    hidden = inputs_embeds
    attn = None
    for idx, layer in enumerate(self.layers):
        if prefill and idx == k:
            start, n_img = self.fastv_image_span
            if self.fastv_generator is None:
                score = attn.mean(dim=1)[0, -1, start:start + n_img]
            else:
                score = torch.rand(n_img, generator=self.fastv_generator, device=self.fastv_generator.device)
                score = score.to(hidden.device)
            top = score.topk(min(keep, n_img)).indices + start
            keep_idx = torch.cat((torch.arange(start, device=hidden.device), top,
                                  torch.arange(start + n_img, hidden.shape[1], device=hidden.device))).sort().values
            hidden = hidden[:, keep_idx]
            position_ids = position_ids[:, keep_idx]
            self.fastv_kept_image_positions = top.sort().values - start
        mask = _causal_mask(hidden) if prefill else None
        need_attn = prefill and idx == k - 1 and self.fastv_generator is None
        out = layer(hidden, attention_mask=mask, position_ids=position_ids, past_key_value=cache,
                    output_attentions=need_attn, use_cache=use_cache)
        hidden = out[0]
        if need_attn:
            attn = out[1]
    hidden = self.norm(hidden)
    next_cache = None
    if use_cache:
        next_cache = cache.to_legacy_cache() if return_legacy else cache
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=next_cache)


def _record_image_span(model):
    lm = model.get_model()
    orig = model.prepare_inputs_labels_for_multimodal

    def wrapped(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        result = orig(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes)
        if images is not None and input_ids is not None and input_ids.shape[1] != 1:
            hits = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
            if hits.numel() != 1:
                raise ValueError("FastV wrapper expects exactly one image per prompt")
            embeds = result[4]
            lm.fastv_image_span = (int(hits[0, 0]), embeds.shape[1] - (input_ids.shape[1] - 1))
        return result

    model.prepare_inputs_labels_for_multimodal = wrapped


def bind_fastv(llama_model, k, keep, seed=None):
    """Turn a transformers LlamaModel into FastV in place (image span set by the caller).

    seed=None keeps FastV's attention ranking; an integer selects random image tokens from that seed.
    """
    if k < 1:
        raise ValueError("K must be >= 1: the scores come from layer K-1")
    modeling_llama.LlamaRotaryEmbedding.forward = _full_rotary_forward
    llama_model.fastv_k, llama_model.fastv_keep = k, keep
    llama_model.fastv_generator = None
    if seed is not None:
        device = llama_model.embed_tokens.weight.device
        llama_model.fastv_generator = torch.Generator(device=device).manual_seed(seed)
    llama_model.forward = types.MethodType(fastv_forward, llama_model)
    return llama_model


def apply_fastv(model, k=2, keep=288, seed=None):
    """Enable FastV (or its random-selection control when seed is given) on a loaded LlavaLlamaForCausalLM."""
    bind_fastv(model.get_model(), k, keep, seed)
    _record_image_span(model)
    return model
