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
A schedule of several (layer, keep) stages prunes in steps (e.g. to copy another method's per-layer token counts);
with a single stage the computation, including the random draws, is the same as before.
With stratify=n (diagnostic), the ranking is taken in turns from an n x n grid of image blocks instead of globally,
so the kept tokens cover as many blocks as the count allows; without it the computation is the same as before.
With log_text_rows (diagnostic, FastV ranking only), the first stage also keeps the head-averaged attention rows of
every text token after the image over the image tokens (the last of them is FastV's score row) and the ids of those
tokens; nothing in the computation changes.
With score_kind="object_cos" (EXP-0011 D3, POPE-style questions), the ranking is instead the cosine similarity of each
image token's input embedding to the mean input embedding of the object name's tokens (object_span maps the ids after
the image to their indices); no attention map is needed. With exclude_padding, patches lying wholly in the padding of
LLaVA's centred square image (expand2square) are never kept. Without either option the computation is as before.
With score_kind="mix" (EXP-0013, blocks only), mix_question of the kept tokens are taken by the object-name cosine and the
rest by FastV's (prior-divided) attention score, the second rotation filling the blocks the first left empty.
"""
import types

import numpy as np
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


def _block_of(patches, side, blocks):
    return ((patches // side) * blocks // side * blocks + (patches % side) * blocks // side).tolist()


def _stratified_top(score, k, patches, side, blocks, taken=None):
    """Indices of k scores taken in turns from a blocks x blocks grid over the side x side patch image: first the best
    patch of every block, then every block's second best, and so on, each round in order of score. taken counts patches
    already kept per block; those blocks then wait until the empty ones have their turn."""
    cell = _block_of(patches, side, blocks)
    s = score.float().tolist()
    taken, rank = dict(taken or {}), [0] * len(s)
    for i in sorted(range(len(s)), key=lambda i: -s[i]):
        rank[i] = taken.get(cell[i], 0)
        taken[cell[i]] = rank[i] + 1
    chosen = sorted(range(len(s)), key=lambda i: (rank[i], -s[i]))[:k]
    return torch.tensor(chosen, dtype=torch.long, device=score.device)


def _padding_mask(width, height, side):
    """Patches of the side x side grid lying wholly in the padding that expand2square adds around a width x height image
    (the same geometry as analyze_object_boxes.padding_patches)."""
    s = max(width, height)
    ox = (s - width) // 2 if height > width else 0
    oy = (s - height) // 2 if width > height else 0
    p = s / side
    edges = np.arange(side + 1) * p
    cx = np.clip(np.minimum(edges[1:], ox + width) - np.maximum(edges[:-1], ox), 0, None) / p
    cy = np.clip(np.minimum(edges[1:], oy + height) - np.maximum(edges[:-1], oy), 0, None) / p
    return torch.from_numpy(np.outer(cy, cx).ravel() <= 1e-9)


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

    stages = self.fastv_stages                  # {layer: keep}; FastV itself is {K: keep}
    prefill = q_len > 1
    hidden = inputs_embeds
    attn = None
    object_cos = self.fastv_score_kind == "object_cos"
    mix = self.fastv_score_kind == "mix"
    if prefill:
        start, n_img = self.fastv_image_span
        kept = torch.arange(n_img, device=hidden.device)       # original patch index of each image token still present
        if object_cos or mix:                                  # question-aware score from the input embeddings
            obj = [start + n_img + i for i in self.fastv_object_span(self.fastv_text_after_ids)]
            h = inputs_embeds[0].float()
            self.fastv_object_cos = torch.nn.functional.cosine_similarity(
                h[start:start + n_img], h[obj].mean(0, keepdim=True), dim=-1)
        allowed = torch.ones(n_img, dtype=torch.bool, device=hidden.device)
        if self.fastv_exclude_padding:
            allowed = ~_padding_mask(*self.fastv_image_size, round(n_img ** 0.5)).to(hidden.device)
    for idx, layer in enumerate(self.layers):
        if prefill and idx in stages:
            n_cur = kept.numel()
            if self.fastv_generator is None and object_cos:
                score = self.fastv_object_cos[kept]
                self.fastv_last_scores = score
            elif self.fastv_generator is None:
                score = attn.mean(dim=1)[0, -1, start:start + n_cur]
                self.fastv_last_scores = score                  # before any prior, for the score-log diagnostic
                if self.fastv_log_text_rows and idx == min(stages):     # diagnostic: rows of the text after the image
                    self.fastv_text_rows = attn.mean(dim=1)[0, start + n_cur:, start:start + n_cur]
                if self.fastv_score_prior is not None:          # diagnostic: divide out a content-free position curve
                    score = score.float() / self.fastv_score_prior.to(score.device)[kept]
            else:
                score = torch.rand(n_cur, generator=self.fastv_generator, device=self.fastv_generator.device)
                score = score.to(hidden.device)
            pos = allowed[kept].nonzero().squeeze(1)           # positions among the present image tokens that may stay
            count = min(stages[idx], pos.numel())
            if mix:                                            # part by the object name, the rest by FastV's score
                side, blocks = round(n_img ** 0.5), self.fastv_stratify
                n_q = min(self.fastv_mix_question, count)
                first = pos[_stratified_top(self.fastv_object_cos[kept][pos], n_q, kept[pos], side, blocks)]
                rest = pos[~torch.isin(pos, first)]
                filled = {}
                for cell in _block_of(kept[first], side, blocks):
                    filled[cell] = filled.get(cell, 0) + 1
                second = rest[_stratified_top(score[rest], count - n_q, kept[rest], side, blocks, filled)]
                top = torch.cat((first, second)).sort().values
                self.fastv_kept_question_positions = kept[first].sort().values
            elif self.fastv_stratify is None:
                top = pos[score[pos].topk(count).indices].sort().values
            else:
                side = round(n_img ** 0.5)
                if side * side != n_img:
                    raise ValueError(f"stratify needs a square patch grid, got {n_img} image tokens")
                top = pos[_stratified_top(score[pos], count, kept[pos], side, self.fastv_stratify)].sort().values
            keep_idx = torch.cat((torch.arange(start, device=hidden.device), top + start,
                                  torch.arange(start + n_cur, hidden.shape[1], device=hidden.device)))
            hidden = hidden[:, keep_idx]
            position_ids = position_ids[:, keep_idx]
            kept = kept[top]
            self.fastv_kept_image_positions = kept
        mask = _causal_mask(hidden) if prefill else None
        need_attn = prefill and idx + 1 in stages and self.fastv_generator is None and not object_cos
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
            lm.fastv_text_after_ids = input_ids[0, int(hits[0, 0]) + 1:].tolist()
            lm.fastv_image_size = tuple(image_sizes[0]) if image_sizes is not None else None     # (width, height)
        return result

    model.prepare_inputs_labels_for_multimodal = wrapped


def bind_fastv(llama_model, k, keep, seed=None, schedule=None, score_prior=None, stratify=None, log_text_rows=False,
               score_kind="last", object_span=None, exclude_padding=False, mix_question=None):
    """Turn a transformers LlamaModel into FastV in place (image span set by the caller).

    seed=None keeps FastV's attention ranking; an integer selects random image tokens from that seed.
    schedule, a list of (layer, keep) pairs, replaces the single stage (k, keep): before each listed layer the image
    tokens still present are cut to `keep`, ranked by the last token's attention at the layer before (or at random).
    score_prior, a float tensor with one value per image patch, divides the attention scores before the ranking
    (diagnostic: a position curve measured on blank images).
    stratify, an int n, takes the ranking in turns from an n x n grid of image blocks (diagnostic: coverage control).
    log_text_rows keeps the first stage's attention rows of the text after the image (diagnostic, FastV ranking only).
    score_kind "object_cos" ranks by the input-embedding cosine to the object name, whose token indices among the ids
    after the image object_span returns; exclude_padding never keeps patches lying wholly in the square's padding.
    score_kind "mix" keeps mix_question tokens by that cosine and the rest by FastV's score (after score_prior, if any),
    both taken in blocks (stratify required, single stage).
    """
    if score_kind not in ("last", "object_cos", "mix"):
        raise ValueError(f"unknown score_kind {score_kind!r}")
    if score_kind == "object_cos" and (object_span is None or seed is not None or score_prior is not None or log_text_rows):
        raise ValueError("object_cos needs object_span and takes no seed, position prior or attention-row log")
    if score_kind == "mix" and (object_span is None or seed is not None or stratify is None or log_text_rows
                                or mix_question is None or not 0 <= mix_question <= keep or schedule):
        raise ValueError("mix needs object_span, stratify and 0 <= mix_question <= keep; no seed, schedule or row log")
    llama_model.fastv_score_kind = score_kind
    llama_model.fastv_mix_question = mix_question
    llama_model.fastv_kept_question_positions = None
    llama_model.fastv_object_span = object_span
    llama_model.fastv_exclude_padding = exclude_padding
    llama_model.fastv_score_prior = None if score_prior is None else score_prior.float()
    llama_model.fastv_stratify = stratify
    llama_model.fastv_last_scores = None
    llama_model.fastv_log_text_rows = log_text_rows
    llama_model.fastv_text_rows = None
    stages = dict(schedule) if schedule else {k: keep}
    if min(stages) < 1:
        raise ValueError("every stage layer must be >= 1: the scores come from the layer before it")
    modeling_llama.LlamaRotaryEmbedding.forward = _full_rotary_forward
    llama_model.fastv_k, llama_model.fastv_keep = k, keep
    llama_model.fastv_stages = stages
    llama_model.fastv_generator = None
    if seed is not None:
        device = llama_model.embed_tokens.weight.device
        llama_model.fastv_generator = torch.Generator(device=device).manual_seed(seed)
    llama_model.forward = types.MethodType(fastv_forward, llama_model)
    return llama_model


def apply_fastv(model, k=2, keep=288, seed=None, schedule=None, score_prior=None, stratify=None, log_text_rows=False,
                score_kind="last", object_span=None, exclude_padding=False, mix_question=None):
    """Enable FastV (or its random-selection control when seed is given) on a loaded LlavaLlamaForCausalLM."""
    bind_fastv(model.get_model(), k, keep, seed, schedule, score_prior, stratify, log_text_rows, score_kind, object_span,
               exclude_padding, mix_question)
    _record_image_span(model)
    return model
