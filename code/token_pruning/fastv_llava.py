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
With score_kind="contrast_max" or "contrast_mean" (EXP-0013), the object-name cosine is compared with the cosine of the
same image token to a vocabulary of other names (contrast_ids: the token ids of each name; a name spelled with the asked
object's ids is left out): the score is the asked cosine minus the largest, or minus the mean, of the others.
With score_kind="query_mean" or "query_idf" (EXP-0014, any question), the query is a weighted mean of the input
embeddings of all tokens after the image, with weights from token_weights (uniform, or IDF over the question file), and
each image token is ranked by its cosine to that query. score_kind="attn_idf" weights the scoring layer's attention rows
of the same tokens by those weights instead (FastV's attention, no position prior).
With score_kind="query_vis" (EXP-0015, blocks only), each token's weight is its IDF times its visibility (a value per
vocabulary id in [0, 1]); round(keep * S) tokens, S = sum(IDF * visibility) / sum(IDF), are taken by the cosine to that
weighted query and the rest by FastV's (prior-divided) score, as in "mix".
With route (EXP-0020, query_vis only), a callable on the ids after the image: a question it returns False for takes the
strong baseline's path instead (FastV's prior-divided score taken in blocks, padding never kept), so query_vis applies
only to the routed questions; fastv_routed records the decision of the last prefill.
With zoom (IDEA-0004, query_vis only), a question whose query is visible enough (S >= s_min) and that zoom["on"] selects
("unrouted": the router's other questions; "all") is located and looked at closer, inside the same budget: the image
tokens most similar to the IDF x visibility query give a square box on the 24 x 24 grid (zoom_box: never smaller than
the side holding the vision tower's own resolution, and kept inside the image so the crop is square); that box is
cropped from the original image (zoom_paths, one per prefill in question order), re-encoded by the vision tower and
projector, and n of the crop's tokens are inserted right after the image tokens: a lattice over the crop ("grid"), the
best query cosine per block ("query"), or block means of the crop's tokens ("pool"). With source "full" the n tokens are
instead the image's own tokens, no second encoding: block means with pick "pool", the same lattice over the image's
content rows and columns with pick "grid" (EXP-0025); with centre the box keeps its side but sits at the image's centre
(the controls). The inserted tokens are never pruned and the stage keeps keep - n image tokens, so
the visual tokens after the stage are still keep. With cover (IDEA-0004, unrouted questions only), the strong
baseline's own count of kept tokens inside the box plus cover is taken inside the box (by its score, in blocks) and the
rest outside. fastv_zoom_log holds one record per prefill. Without zoom and cover nothing changes.
"""
import math
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


def _square_geometry(width, height, side):
    """expand2square's square: its side in pixels, the image's offset in it, and one patch's size in pixels."""
    s = max(width, height)
    return s, ((s - width) // 2 if height > width else 0), ((s - height) // 2 if width > height else 0), s / side


def _content_band(width, height, side):
    """First and last rows and columns of the side x side grid lying at least half inside a width x height image on
    expand2square's square: (r_lo, r_hi, c_lo, c_hi), inclusive."""
    _, ox, oy, p = _square_geometry(width, height, side)
    edges = np.arange(side + 1) * p
    out = []
    for o, extent in ((oy, height), (ox, width)):
        cover = np.clip(np.minimum(edges[1:], o + extent) - np.maximum(edges[:-1], o), 0, None) / p
        idx = np.flatnonzero(cover >= 0.5)
        idx = idx if idx.size else np.array([int(cover.argmax())])
        out += [int(idx[0]), int(idx[-1])]
    return out[0], out[1], out[2], out[3]


def zoom_box(cos, allowed, side, width, height, top_m, min_side, max_side, centre=False, native=336):
    """A square box on the side x side grid around the image tokens most similar to the query. Its side is the top_m
    allowed tokens' span, clamped to [floor, max_side], floor being the larger of min_side and the side that holds
    `native` original pixels (the crop is then never enlarged past the image's own resolution). It is centred on their
    score-weighted centre (on the image's centre with centre=True) and kept inside the image's content band (rows and
    columns at least half inside the image), shrunk to the band if needed, so the crop stays square; only when the band
    is narrower than min_side (very elongated images) does the box merely stay inside the grid, its crop then carrying
    padding. Returns the grid box (r0, c0, r1, c1), inclusive, its pixel box (x0, y0, x1, y1) on the original image and a
    dict: span, side, band, how many of the top_m tokens and whether the top one lie inside."""
    score = cos.float().masked_fill(~allowed, float("-inf"))
    top = score.topk(min(top_m, int(allowed.sum()))).indices
    rows, cols = (top // side).float(), (top % side).float()
    w = score[top] - score[top].min() + 1e-6
    span = int(max(rows.max() - rows.min(), cols.max() - cols.min()).item()) + 1
    size = min(max(span, min_side, math.ceil(side * native / max(width, height))), max_side)
    r_lo, r_hi, c_lo, c_hi = _content_band(width, height, side)
    band = min(r_hi - r_lo + 1, c_hi - c_lo + 1)
    if band >= min_side:
        size = min(size, band)
        lo_r, hi_r, lo_c, hi_c = r_lo, r_hi - size + 1, c_lo, c_hi - size + 1
    else:
        lo_r, hi_r, lo_c, hi_c = 0, side - size, 0, side - size
    if centre:
        cr, cc = (r_lo + r_hi) / 2, (c_lo + c_hi) / 2
    else:
        cr, cc = float((w * rows).sum() / w.sum()), float((w * cols).sum() / w.sum())
    r0 = int(min(max(round(cr - (size - 1) / 2), lo_r), hi_r))
    c0 = int(min(max(round(cc - (size - 1) / 2), lo_c), hi_c))
    r1, c1 = r0 + size - 1, c0 + size - 1
    _, ox, oy, p = _square_geometry(width, height, side)
    x0, x1 = max(0.0, c0 * p - ox), min(float(width), (c1 + 1) * p - ox)
    y0, y1 = max(0.0, r0 * p - oy), min(float(height), (r1 + 1) * p - oy)
    inside = (rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1)
    info = {"span": span, "side": size, "band": band, "top_inside": int(inside.sum()), "top1_inside": bool(inside[0])}
    return (r0, c0, r1, c1), (x0, y0, x1, y1), info


def _content_lines(pad, side):
    """Rows and columns of a side x side token grid that are not wholly padding (pad: one bool per token, or None)."""
    keep = torch.ones(side, side, dtype=torch.bool) if pad is None else ~pad.cpu().reshape(side, side)
    return keep.any(1).nonzero().squeeze(1).tolist(), keep.any(0).nonzero().squeeze(1).tolist()


def zoom_pick(n, side, how, cos=None, pad=None):
    """n = g * g crop tokens: the centres of a g x g lattice over the crop's content ("grid"), or the best query cosine in
    each of g x g blocks among its non-padding tokens ("query"). pad marks the tokens wholly in the padding that
    process_images adds around a non-square crop (none for the square crops zoom_box gives inside the image)."""
    g = round(n ** 0.5)
    if g * g != n:
        raise ValueError("the number of zoom tokens must be a square")
    if how == "grid":
        rows, cols = _content_lines(pad, side)
        at_r = [rows[0] + int((i + 0.5) * (rows[-1] - rows[0] + 1) / g) for i in range(g)]
        at_c = [cols[0] + int((i + 0.5) * (cols[-1] - cols[0] + 1) / g) for i in range(g)]
        return torch.tensor([r * side + c for r in at_r for c in at_c], dtype=torch.long)
    cand = torch.arange(side * side) if pad is None else (~pad.cpu()).nonzero().squeeze(1)
    return cand[_stratified_top(cos.cpu()[cand], n, cand, side, g).cpu()].sort().values


def zoom_pool(feats, n, side, pad=None):
    """n = g * g block means of a side x side grid of tokens (feats: side * side x d), the blocks splitting its content
    rows and columns (those not wholly padding) as evenly as possible."""
    g = round(n ** 0.5)
    if g * g != n:
        raise ValueError("the number of zoom tokens must be a square")
    rows, cols = _content_lines(pad, side)
    if len(rows) < g or len(cols) < g:
        rows, cols = list(range(side)), list(range(side))
    grid = feats.reshape(side, side, -1)
    out = []
    for rg in np.array_split(np.array(rows), g):
        for cg in np.array_split(np.array(cols), g):
            r = torch.as_tensor(rg, device=feats.device)
            c = torch.as_tensor(cg, device=feats.device)
            out.append(grid[r][:, c].reshape(-1, grid.shape[-1]).float().mean(0))
    return torch.stack(out).to(feats.dtype)


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
    contrast = self.fastv_score_kind in ("contrast_max", "contrast_mean")
    query = self.fastv_score_kind in ("query_mean", "query_idf")
    attn_weighted = self.fastv_score_kind == "attn_idf"
    query_vis = self.fastv_score_kind == "query_vis"
    if query_vis and prefill and self.fastv_route is not None:     # EXP-0020: the other questions take the baseline's path
        self.fastv_routed = bool(self.fastv_route(self.fastv_text_after_ids))
        query_vis = self.fastv_routed
        if not query_vis:
            self.fastv_visible_share, self.fastv_mix_count = None, 0
            self.fastv_kept_question_positions = torch.zeros(0, dtype=torch.long)
    object_cos = self.fastv_score_kind == "object_cos" or contrast or query     # ranked by an input-embedding score alone
    mix = self.fastv_score_kind == "mix" or query_vis
    extra = getattr(self, "fastv_extra", 0) if prefill else 0      # zoom tokens inserted right after the image (IDEA-0004)
    if prefill:
        start, n_img = self.fastv_image_span
        kept = torch.arange(n_img, device=hidden.device)       # original patch index of each image token still present
        if query or attn_weighted:                             # one weight per token after the image
            w = torch.tensor(self.fastv_token_weights(self.fastv_text_after_ids), dtype=torch.float32,
                             device=hidden.device)
            self.fastv_query_weights = w if float(w.sum()) > 0 else torch.ones_like(w)
        if query_vis:                                          # IDF x visibility; S sets the question's share of the budget
            idf = torch.tensor(self.fastv_token_weights(self.fastv_text_after_ids), dtype=torch.float32,
                               device=hidden.device)
            vis = self.fastv_visibility.to(hidden.device)[torch.tensor(self.fastv_text_after_ids, device=hidden.device)]
            w = idf * vis.float()
            self.fastv_visible_share = float(w.sum() / idf.sum()) if float(idf.sum()) > 0 else 0.0
            self.fastv_query_weights = w if float(w.sum()) > 0 else torch.ones_like(w)
        if query or query_vis:                                 # the weighted mean of the question's input embeddings
            h = inputs_embeds[0].float()
            w = self.fastv_query_weights
            t0 = start + n_img + extra                         # the text after the image (and after any zoom tokens)
            q = (w[:, None] * h[t0:t0 + w.numel()]).sum(0, keepdim=True) / w.sum()
            self.fastv_object_cos = torch.nn.functional.cosine_similarity(h[start:start + n_img], q, dim=-1)
            self.fastv_input_score = self.fastv_object_cos
        elif object_cos or mix:                                # question-aware score from the input embeddings
            span = self.fastv_object_span(self.fastv_text_after_ids)
            obj = [start + n_img + extra + i for i in span]
            h = inputs_embeds[0].float()
            self.fastv_object_cos = torch.nn.functional.cosine_similarity(
                h[start:start + n_img], h[obj].mean(0, keepdim=True), dim=-1)
            self.fastv_input_score = self.fastv_object_cos
            if contrast:                                       # the asked name against the other names
                asked = [self.fastv_text_after_ids[i] for i in span]
                other = [j for j, ids in enumerate(self.fastv_contrast_ids) if ids != asked]
                alt = torch.nn.functional.normalize(h[start:start + n_img], dim=-1) @ self.fastv_contrast_emb[other].T
                ref = alt.max(1).values if self.fastv_score_kind == "contrast_max" else alt.mean(1)
                self.fastv_input_score = self.fastv_object_cos - ref
        allowed = torch.ones(n_img, dtype=torch.bool, device=hidden.device)
        if self.fastv_exclude_padding:
            allowed = ~_padding_mask(*self.fastv_image_size, round(n_img ** 0.5)).to(hidden.device)
    for idx, layer in enumerate(self.layers):
        if prefill and idx in stages:
            n_cur = kept.numel()
            if self.fastv_generator is None and object_cos:
                score = self.fastv_input_score[kept]
                self.fastv_last_scores = score
            elif self.fastv_generator is None and attn_weighted:   # weighted attention rows of the text after the image
                t0 = start + n_cur + extra
                rows = attn.mean(dim=1)[0, t0:t0 + self.fastv_query_weights.numel(), start:start + n_cur]
                w = self.fastv_query_weights.to(rows.device)
                score = (w[:, None] * rows.float()).sum(0) / w.sum()
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
            count = min(stages[idx] - extra, pos.numel())       # zoom tokens count towards the stage's budget
            cover = getattr(self, "fastv_cover_mask", None) if not mix else None
            if mix:                                            # part by the object name, the rest by FastV's score
                side, blocks = round(n_img ** 0.5), self.fastv_stratify
                n_q = round(count * self.fastv_visible_share) if query_vis else min(self.fastv_mix_question, count)
                self.fastv_mix_count = n_q
                first = pos[_stratified_top(self.fastv_object_cos[kept][pos], n_q, kept[pos], side, blocks)]
                rest = pos[~torch.isin(pos, first)]
                filled = {}
                for cell in _block_of(kept[first], side, blocks):
                    filled[cell] = filled.get(cell, 0) + 1
                second = rest[_stratified_top(score[rest], count - n_q, kept[rest], side, blocks, filled)]
                top = torch.cat((first, second)).sort().values
                self.fastv_kept_question_positions = kept[first].sort().values
            elif cover is not None:            # IDEA-0004: the baseline's own in-box count plus `cover` taken in the box
                side, blocks = round(n_img ** 0.5), self.fastv_stratify
                box = cover.to(hidden.device)
                base = pos[_stratified_top(score[pos], count, kept[pos], side, blocks)]    # the baseline's own choice
                a_in = int(box[kept[base]].sum())
                inside, outside = pos[box[kept[pos]]], pos[~box[kept[pos]]]
                n_c = min(a_in + self.fastv_cover, count, inside.numel())
                first = inside[_stratified_top(score[inside], n_c, kept[inside], side, blocks)]
                filled = {}
                for cell in _block_of(kept[first], side, blocks):
                    filled[cell] = filled.get(cell, 0) + 1
                second = outside[_stratified_top(score[outside], count - n_c, kept[outside], side, blocks, filled)]
                top = torch.cat((first, second)).sort().values
                self.fastv_cover_count = n_c
                if self.fastv_zoom_log:                        # this prefill's record, written by _zoom just before
                    self.fastv_zoom_log[-1].update(baseline_in_box=a_in, in_box=n_c,
                                                   changed=int((~torch.isin(top, base)).sum()))
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


def _zoom(model, lm, result):
    """IDEA-0004: locate the question's object with the IDF x visibility query and, for zoom, insert n tokens of a
    re-encoded crop of that box right after the image tokens; for cover, mark the box for the stage. One record per
    prefill in lm.fastv_zoom_log."""
    cfg, ids = lm.fastv_zoom, lm.fastv_text_after_ids
    index = lm.fastv_zoom_count
    lm.fastv_zoom_count += 1
    rec = {"index": index, "zoom": False, "cover": False}
    if cfg["on"] == "unrouted" and lm.fastv_route(ids):
        rec["skip"] = "routed"
        lm.fastv_zoom_log.append(rec)
        return result
    idf = torch.tensor(lm.fastv_token_weights(ids), dtype=torch.float32)
    w = idf * lm.fastv_visibility[torch.tensor(ids)].float()
    share = float(w.sum() / idf.sum()) if float(idf.sum()) > 0 else 0.0
    rec["S"] = share
    if share < cfg["s_min"] or float(w.sum()) <= 0:
        rec["skip"] = "S below s_min"
        lm.fastv_zoom_log.append(rec)
        return result
    embeds = result[4]
    start, n_img = lm.fastv_image_span
    side = round(n_img ** 0.5)
    width, height = lm.fastv_image_size
    h = embeds[0].float()
    w = w.to(h.device)
    q = (w[:, None] * h[start + n_img:start + n_img + w.numel()]).sum(0, keepdim=True) / w.sum()
    cos = torch.nn.functional.cosine_similarity(h[start:start + n_img], q, dim=-1)
    allowed = ~_padding_mask(width, height, side).to(h.device)
    native = lm.fastv_image_processor.crop_size["height"] if getattr(lm, "fastv_image_processor", None) else 336
    grid, pixels, info = zoom_box(cos, allowed, side, width, height, cfg["top_m"], cfg["min_side"], cfg["max_side"],
                                  centre=cfg.get("centre", False), native=native)
    rec.update(grid_box=list(grid), pixel_box=[round(v, 1) for v in pixels], **info)
    if cfg.get("cover"):
        r0, c0, r1, c1 = grid
        box = torch.zeros(side, side, dtype=torch.bool)
        box[r0:r1 + 1, c0:c1 + 1] = True
        lm.fastv_cover_mask = box.ravel()
        rec["cover"] = True
    n = cfg.get("n", 0)
    if n:
        if cfg.get("source", "crop") == "full":                # controls: the image's own tokens, no second encoding
            if cfg["pick"] == "pool":                          # block means
                tokens = zoom_pool(embeds[0, start:start + n_img], n, side, ~allowed)
            else:                                              # the lattice over the image's content (EXP-0025)
                pick = zoom_pick(n, side, "grid", pad=~allowed).to(embeds.device)
                tokens = embeds[0, start:start + n_img][pick]
                rec["picked"] = pick.tolist()
            rec["source"] = "full"
        else:
            from PIL import Image
            from llava.mm_utils import process_images
            crop_box = [round(v) for v in pixels]
            with Image.open(lm.fastv_zoom_paths[index]) as im:
                if im.size != (width, height):
                    raise ValueError(f"zoom image {lm.fastv_zoom_paths[index]} is {im.size}, the prompt's is {(width, height)}")
                crop = im.convert("RGB").crop(tuple(crop_box))
            pix = process_images([crop], lm.fastv_image_processor, model.config).to(device=embeds.device, dtype=embeds.dtype)
            with torch.no_grad():
                feats = model.encode_images(pix)[0]
            cpad = _padding_mask(crop.width, crop.height, side).to(feats.device)
            rec.update(source="crop", crop_box=crop_box, crop_padding_tokens=int(cpad.sum()))
            if cfg["pick"] == "pool":
                tokens = zoom_pool(feats, n, side, cpad)
            else:
                ccos = torch.nn.functional.cosine_similarity(feats.float(), q, dim=-1) if cfg["pick"] == "query" else None
                pick = zoom_pick(n, side, cfg["pick"], ccos, cpad).to(feats.device)
                tokens = feats[pick]
                rec.update(picked=pick.tolist(), picked_on_padding=int(cpad[pick].sum()))
        cut = start + n_img
        result = list(result)
        result[4] = torch.cat((embeds[:, :cut], tokens[None].to(embeds.dtype), embeds[:, cut:]), dim=1)
        if result[2] is not None:                              # attention mask and position ids, if the caller passed them
            result[2] = torch.ones(1, result[4].shape[1], dtype=result[2].dtype, device=result[2].device)
        if result[1] is not None:
            result[1] = torch.arange(result[4].shape[1], device=result[1].device).unsqueeze(0)
        result = tuple(result)
        lm.fastv_extra = n
        rec.update(zoom=True, n=n)
    lm.fastv_zoom_log.append(rec)
    return result


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
            lm.fastv_extra, lm.fastv_cover_mask = 0, None
            if getattr(lm, "fastv_zoom", None):
                result = _zoom(model, lm, result)
        return result

    model.prepare_inputs_labels_for_multimodal = wrapped


def bind_fastv(llama_model, k, keep, seed=None, schedule=None, score_prior=None, stratify=None, log_text_rows=False,
               score_kind="last", object_span=None, exclude_padding=False, mix_question=None, contrast_ids=None,
               token_weights=None, visibility=None, route=None, zoom=None, cover=0):
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
    score_kind "contrast_max" / "contrast_mean" subtracts from that cosine the largest / mean cosine to the names in
    contrast_ids (a list of token-id lists; each name is the mean of its tokens' input embeddings).
    """
    if score_kind not in ("last", "object_cos", "mix", "contrast_max", "contrast_mean", "query_mean", "query_idf",
                          "attn_idf", "query_vis"):
        raise ValueError(f"unknown score_kind {score_kind!r}")
    weighted = score_kind in ("query_mean", "query_idf", "attn_idf", "query_vis")
    if weighted != (token_weights is not None):
        raise ValueError("query_mean, query_idf, attn_idf and query_vis need token_weights, and only they take it")
    if weighted and score_kind != "query_vis" and (seed is not None or score_prior is not None or log_text_rows or schedule):
        raise ValueError(f"{score_kind} takes no seed, position prior, attention-row log or schedule")
    if (score_kind == "query_vis") != (visibility is not None):
        raise ValueError("query_vis needs visibility, and only it takes it")
    if score_kind == "query_vis" and (seed is not None or stratify is None or log_text_rows or schedule):
        raise ValueError("query_vis needs stratify; no seed, schedule or row log")
    if route is not None and score_kind != "query_vis":
        raise ValueError("route works with query_vis only")
    if (zoom or cover) and score_kind != "query_vis":
        raise ValueError("zoom and cover work with query_vis only (they locate with its query)")
    if zoom and zoom["on"] == "unrouted" and route is None:
        raise ValueError("zoom on the unrouted questions needs route")
    if zoom and not 0 <= zoom.get("n", 0) < keep:
        raise ValueError("zoom n must be in [0, keep)")
    if cover and zoom and zoom["on"] != "unrouted":
        raise ValueError("cover works on the unrouted questions only (routed ones take query_vis, which it does not change)")
    if zoom and zoom.get("source", "crop") == "full" and zoom.get("pick") not in ("pool", "grid"):
        raise ValueError("zoom source 'full' is a context control: it needs pick 'pool' or 'grid'")
    llama_model.fastv_zoom = zoom
    llama_model.fastv_cover = cover
    llama_model.fastv_zoom_count, llama_model.fastv_zoom_log = 0, []
    llama_model.fastv_extra, llama_model.fastv_cover_mask, llama_model.fastv_cover_count = 0, None, None
    llama_model.fastv_route = route
    llama_model.fastv_routed = None
    llama_model.fastv_token_weights = token_weights
    llama_model.fastv_visibility = None if visibility is None else torch.as_tensor(visibility).float()
    llama_model.fastv_visible_share = None
    llama_model.fastv_mix_count = None
    if score_kind in ("object_cos", "contrast_max", "contrast_mean") and (
            object_span is None or seed is not None or score_prior is not None or log_text_rows):
        raise ValueError(f"{score_kind} needs object_span and takes no seed, position prior or attention-row log")
    if score_kind.startswith("contrast") != bool(contrast_ids):
        raise ValueError("the contrast scores need contrast_ids, and only they take it")
    llama_model.fastv_contrast_ids = [list(ids) for ids in contrast_ids] if contrast_ids else None
    if contrast_ids:
        w = llama_model.embed_tokens.weight
        emb = torch.stack([w[torch.tensor(ids, device=w.device)].detach().float().mean(0) for ids in contrast_ids])
        llama_model.fastv_contrast_emb = torch.nn.functional.normalize(emb, dim=-1)
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
                score_kind="last", object_span=None, exclude_padding=False, mix_question=None, contrast_ids=None,
                token_weights=None, visibility=None, route=None, zoom=None, cover=0):
    """Enable FastV (or its random-selection control when seed is given) on a loaded LlavaLlamaForCausalLM."""
    bind_fastv(model.get_model(), k, keep, seed, schedule, score_prior, stratify, log_text_rows, score_kind, object_span,
               exclude_padding, mix_question, contrast_ids, token_weights, visibility, route, zoom, cover)
    _record_image_span(model)
    return model
