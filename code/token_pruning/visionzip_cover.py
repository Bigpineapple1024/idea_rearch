"""IDEA-0006: locate-then-cover on VisionZip (official LLaVA codebase, LLaVA-1.5, batch size 1, one image per prompt).

VisionZip (source/VisionZip/visionzip/clip_encoder.py) keeps the CLS token and `dominant - 1` patches with the largest
CLS attention summed over heads at the vision tower's layer -2, and merges the other patches into `contextual` tokens by
key similarity. With cover n, the question's IDF x visibility query (fastv_llava's query_vis: the weighted mean of the
input embeddings of the tokens after the image) locates a square box on the 24 x 24 grid with fastv_llava.zoom_box, from
the cosine of every patch's LLM input embedding (the projector on layer -2, which plain LLaVA feeds the LLM) to the query,
so the box is the one IDEA-0005 locates for the same question and image. VisionZip's own count of dominant patches inside
the box plus n are then taken inside it (patches not wholly in the padding), by CLS attention, and the rest outside it, by
CLS attention; the contextual merge is VisionZip's, over the patches left. With on="unrouted" the questions the router
sends to the query path (EXP-0020: yes/no, how many) keep VisionZip's choice; with centre the box keeps its side but sits
at the centre of the image's content band (the localisation control); with top (EXP-0028 v1.1, the no-box control) the
region is instead the `top` patches (not wholly padding) most similar to the query, wherever they lie. A prefill without
a query (routed, or no visible weight) runs VisionZip's forward unchanged. model.vz_cover_log holds one record per
prefill.
"""
import types

import torch

from llava.constants import IMAGE_TOKEN_INDEX

from fastv_llava import _padding_mask, zoom_box


def _cover(state, hidden_states, cls_sum, own, dtype):
    """The dominant patches (indices 0..n_img-1) of one image: VisionZip's own count inside the located box plus
    state["cover"] taken inside it by CLS attention, the rest outside it by CLS attention. own: VisionZip's choice.
    dtype: the images' dtype, to which plain LLaVA casts layer -2 before the projector (CLIPVisionTower.forward)."""
    if hidden_states.shape[0] != 1:
        raise ValueError("visionzip_cover covers one image per prefill")
    own, d = own[0], own.shape[1]
    n_img = hidden_states.shape[1] - 1
    side = round(n_img ** 0.5)
    width, height = state["size"]
    # the patches' LLM input embeddings, as plain LLaVA computes them: with device_map="auto" the vision tower stays in
    # float32 (llava/model/builder.py) and its features are cast to the images' dtype (float16) before the projector
    proj = state["projector"](hidden_states[:, 1:].to(dtype))[0].float()
    cos = torch.nn.functional.cosine_similarity(proj, state["query"].to(proj.device), dim=-1)
    allowed = ~_padding_mask(width, height, side).to(proj.device)
    top_m, min_side, max_side = state["box"]
    grid, pixels, info = zoom_box(cos, allowed, side, width, height, top_m, min_side, max_side, centre=state["centre"],
                                  native=state["native"])
    if state["top"]:                                     # no-box control: the patches most similar to the query
        top = cos.masked_fill(~allowed, float("-inf")).topk(min(state["top"], int(allowed.sum()))).indices
        box = torch.zeros(n_img, dtype=torch.bool, device=proj.device)
        box[top] = True
        state["record"]["cover_top"] = sorted(top.tolist())
    else:
        r0, c0, r1, c1 = grid
        box = torch.zeros(side, side, dtype=torch.bool)
        box[r0:r1 + 1, c0:c1 + 1] = True
        box = box.ravel().to(proj.device)
    region = box & allowed
    a_in = int(region[own].sum())
    inside, outside = region.nonzero().squeeze(1), (~box).nonzero().squeeze(1)
    n_c = min(a_in + state["cover"], d, inside.numel())
    if outside.numel() < d - n_c:
        raise ValueError(f"the box leaves {outside.numel()} patches outside it for {d - n_c} dominant ones")
    s = cls_sum[0]
    first = inside[s[inside].topk(n_c).indices]
    second = outside[s[outside].topk(d - n_c).indices]
    chosen = torch.cat((first, second))
    state["record"].update(cover=True, grid_box=list(grid), pixel_box=[round(v, 1) for v in pixels], **info,
                           region=inside.numel(), baseline_in_box=a_in, in_box=n_c, changed=int((~torch.isin(chosen, own)).sum()),
                           dominant=sorted(chosen.tolist()), baseline=sorted(own.tolist()))
    return chosen[None]


@torch.no_grad()
def _forward(self, images):
    """VisionZip's CLIPVisionTower forward (clip_encoder.py, the batch-tensor branch), line for line, except that the
    dominant patches come from _cover when the prefill hook left a query."""
    if type(images) is list:
        raise ValueError("visionzip_cover takes a batch tensor of images (LLaVA-1.5), not a list")
    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
    attn_weights = image_forward_outs.attentions[-2]
    hidden_states = image_forward_outs.hidden_states[-2]
    metric = self.vision_tower.vision_model.encoder.layers[-2].metric
    dominant_num = self.vision_tower._info["dominant"]
    contextual_num = self.vision_tower._info["contextual"]

    ## Dominant Visual Tokens
    cls_idx = 0
    cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]
    cls_attention_sum = cls_attention.sum(dim=1)
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
    state = self.vz_cover
    if state["query"] is not None:                       # IDEA-0006: the located box takes its share of the patches
        topk_indices = _cover(state, hidden_states, cls_attention_sum, topk_indices - 1, images.dtype) + 1
    elif state["record"] is not None:
        state["record"]["dominant"] = sorted((topk_indices[0] - 1).tolist())
    all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)

    mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])

    ### Filter
    metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])

    hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])

    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    ## Contextual Visual Tokens
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
    target_hidden = hidden_states_filtered[:, target_indices, :]

    contextual_tokens = target_hidden + aggregated_hidden

    # Merge with target hidden states and concatenate
    hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

    return hidden_states_save, all_indices


def apply(model, cover, token_weights, visibility, route=None, on="unrouted", centre=False, box=(8, 8, 16), top=0):
    """Turn VisionZip (visionzip.visionzip already applied) into locate-then-cover on a loaded LlavaLlamaForCausalLM.

    token_weights maps the ids after the image to one IDF each; visibility holds one value in [0, 1] per vocabulary id;
    route, a callable on those ids, is the router: with on="unrouted" a question it returns True for keeps VisionZip's
    choice. box is (top_m, min_side, max_side) of fastv_llava.zoom_box; top > 0 replaces the box by the top patches most
    similar to the query (the box is still located and logged)."""
    if cover < 1:
        raise ValueError("cover must be at least 1")
    if on not in ("unrouted", "all"):
        raise ValueError(f"unknown on {on!r}")
    if on == "unrouted" and route is None:
        raise ValueError("cover on the unrouted questions needs route")
    if top < 0 or (top and centre):
        raise ValueError("top must be >= 0 and does not go with centre")
    lm = model.get_model()
    tower = lm.get_vision_tower()
    if not hasattr(tower.vision_tower, "_info"):
        raise ValueError("apply visionzip.visionzip before visionzip_cover.apply")
    state = {"cover": cover, "centre": bool(centre), "box": tuple(box), "top": top, "native": tower.config.image_size,
             "projector": lm.mm_projector, "query": None, "size": None, "record": None}
    tower.vz_cover = state
    tower.forward = types.MethodType(_forward, tower)     # VisionZip patched the class; this instance takes the copy
    visibility = torch.as_tensor(visibility).float()
    log = []
    model.vz_cover_log = log
    orig = model.prepare_inputs_labels_for_multimodal

    def wrapped(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        if images is None or input_ids is None or input_ids.shape[1] == 1:
            return orig(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes)
        hits = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
        if input_ids.shape[0] != 1 or hits.numel() != 1:
            raise ValueError("visionzip_cover expects batch size 1 and exactly one image per prompt")
        ids = input_ids[0, int(hits[0, 0]) + 1:].tolist()
        rec = {"index": len(log), "cover": False}
        if on == "unrouted" and route(ids):
            rec["skip"] = "routed"
        else:
            idf = torch.tensor(token_weights(ids), dtype=torch.float32)
            w = idf * visibility[torch.tensor(ids)]
            rec["S"] = float(w.sum() / idf.sum()) if float(idf.sum()) > 0 else 0.0
            if float(w.sum()) <= 0:
                rec["skip"] = "no visible weight"
            else:
                if image_sizes is None:
                    raise ValueError("visionzip_cover needs image_sizes (the box stays inside the image)")
                h = lm.embed_tokens(torch.tensor(ids, device=lm.embed_tokens.weight.device)).float()
                w = w.to(h.device)
                state["query"] = (w[:, None] * h).sum(0, keepdim=True) / w.sum()
                state["size"] = tuple(image_sizes[0])     # (width, height)
        state["record"] = rec
        log.append(rec)
        try:
            return orig(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes)
        finally:
            state["query"], state["size"], state["record"] = None, None, None

    model.prepare_inputs_labels_for_multimodal = wrapped
    return model
