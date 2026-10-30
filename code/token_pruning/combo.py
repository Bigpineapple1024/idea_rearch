"""EXP-0033 (step 4, round 2): exploratory combinations of two methods, all before the LLM (LLaVA-1.5, batch size 1,
one image per prompt), so every LLM layer sees the kept image tokens only.

- apply_vzcand_div(model, cand, keep, log)  (X2): VisionZip's dominant score (the vision tower's CLS attention at layer -2,
  summed over heads) gives the top `cand` patches as candidates; DivPrune's max-min order on their projector outputs
  keeps `keep` of them.
- apply_div_qroute(model, keep, weights, visibility, route, log)  (X3): DivPrune-64 as the base. For a question the
  router sends to the query (IDEA-0003's rule on the text after the image), the first round(keep * S) patches are the
  ones most similar to the IDF x visibility query (IDEA-0002's formulas, the same as fastv_llava's query_vis, without
  blocks), and the max-min order continues from them; other questions keep DivPrune's own choice.
- apply_div_ctx(model, keep, contextual, log)  (X4): DivPrune keeps `keep` patches; the other patches are merged into
  `contextual` tokens by VisionZip's contextual rule (evenly spaced targets, each other patch to its most similar target,
  target + mean of its members), on the vision tower's patch features, which then go through the projector. VisionZip
  measures similarity with layer -2's keys; this uses the features' cosine instead (an approximation, EXP-0033 plan).
X1 (VisionZip-128, then plain FastV keeping 60 from layer 2) needs nothing here: run_eval applies visionzip, then fastv.

Every function logs one record per encoded image; the logs are what check_combo.py and analyze_0033.py read.
"""
import types

import torch

from divprune import maxmin_order

IMAGE_TOKEN_INDEX = -200


def maxmin_from(x, m, start):
    """The greedy max-min order of maxmin_order, continued from the rows in `start` (kept first, in their order)."""
    if not start:
        return maxmin_order(x, m)
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    d = 1 - x @ x.T
    n = d.shape[0]
    m = min(m, n)
    order = list(start)[:m]
    taken = torch.zeros(n, dtype=torch.bool, device=d.device)
    taken[torch.tensor(order, device=d.device)] = True
    near = d[torch.tensor(order, device=d.device)].min(dim=0).values
    while len(order) < m:
        k = int(near.masked_fill(taken, float("-inf")).argmax())
        order.append(k)
        taken[k] = True
        near = torch.minimum(near, d[k])
    return order


@torch.no_grad()
def _tower(model, images):
    """Plain LLaVA's vision tower features (layer -2, patches only) and VisionZip's dominant score per patch."""
    vt = model.get_model().get_vision_tower()
    if type(images) is list:
        raise ValueError("the combinations take a batch tensor of images (LLaVA-1.5), not a list")
    outs = vt.vision_tower(images.to(device=vt.device, dtype=vt.dtype), output_hidden_states=True, output_attentions=True)
    cls = outs.attentions[-2][:, :, 0, 1:].sum(dim=1)
    return vt.feature_select(outs).to(images.dtype), cls


def apply_vzcand_div(model, cand, keep, log):
    if not 0 < keep <= cand <= 576:
        raise ValueError("vzcand_div needs 0 < keep <= cand <= 576")

    def encode_images(self, images):
        feats, cls = _tower(self, images)
        proj = self.get_model().mm_projector(feats)
        out = []
        for b in range(proj.shape[0]):
            cands = cls[b].topk(cand).indices.sort().values
            order = maxmin_order(proj[b, cands], keep)
            kept = sorted(cands[torch.tensor(order, device=cands.device)].tolist())
            out.append(proj[b, kept])
            log.append({"index": len(log), "cand": cands.tolist(), "order": order, "kept": kept})
        return torch.stack(out)

    model.encode_images = types.MethodType(encode_images, model)
    return model


def _stash_text(model):
    """Keep the prompt ids after the image of the current prefill, before LLaVA encodes the image."""
    orig = model.prepare_inputs_labels_for_multimodal

    def wrapped(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        if images is not None and input_ids is not None and input_ids.shape[1] != 1:
            hits = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
            if hits.numel() != 1:
                raise ValueError("the combinations expect exactly one image per prompt")
            model.combo_text_ids = input_ids[0, int(hits[0, 0]) + 1:].tolist()
        return orig(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=image_sizes)

    model.prepare_inputs_labels_for_multimodal = wrapped
    model.combo_text_ids = None


def apply_div_qroute(model, keep, weights, visibility, route, log):
    if not 0 < keep <= 576:
        raise ValueError("div_qroute needs 0 < keep <= 576")
    plain = model.encode_images
    vis_table = torch.as_tensor(visibility).float()
    _stash_text(model)

    def encode_images(self, images):
        feats = plain(images)
        ids = self.combo_text_ids
        if ids is None or feats.shape[0] != 1:
            raise ValueError("div_qroute needs the prompt of the current prefill and one image")
        rec = {"index": len(log), "routed": bool(route(ids))}
        if rec["routed"]:
            idf = torch.tensor(weights(ids), dtype=torch.float32)
            vis = vis_table[torch.tensor(ids)]
            w = idf * vis
            share = float(w.sum() / idf.sum()) if float(idf.sum()) > 0 else 0.0
            w = w if float(w.sum()) > 0 else torch.ones_like(w)
            emb = self.get_model().embed_tokens(torch.tensor(ids, device=feats.device)).float()
            q = (w.to(emb.device)[:, None] * emb).sum(0, keepdim=True) / w.sum().to(emb.device)
            cos = torch.nn.functional.cosine_similarity(feats[0].float(), q, dim=-1)
            n_q = round(keep * share)
            first = cos.topk(n_q).indices.tolist() if n_q > 0 else []
            order = maxmin_from(feats[0], keep, first)
            rec.update({"share": share, "n_query": n_q, "first": first})
        else:
            order = maxmin_order(feats[0], keep)
        kept = sorted(order)
        rec.update({"order": order, "kept": kept})
        log.append(rec)
        return feats[:, kept]

    model.encode_images = types.MethodType(encode_images, model)
    return model


def contextual_merge(h, contextual):
    """VisionZip's contextual tokens (clip_encoder.py's rule) over the rows of h (n x d), the cosine of h as the metric."""
    n = h.shape[0]
    mn = torch.nn.functional.normalize(h.float(), dim=-1)
    step = max(1, n // contextual)
    targets = torch.arange(0, n, step, device=h.device)[:contextual]
    others = ~torch.isin(torch.arange(n, device=h.device), targets)
    sim = mn[others] @ mn[targets].T
    onehot = torch.zeros(int(others.sum()), contextual, dtype=h.dtype, device=h.device)
    onehot.scatter_(1, sim.argmax(dim=1).unsqueeze(-1), 1)
    counts = onehot.sum(dim=0).clamp(min=1).unsqueeze(-1)
    agg = (onehot.T @ h[others]) / counts
    return h[targets] + agg, targets.tolist()


def apply_div_ctx(model, keep, contextual, log):
    if not (0 < keep and 0 < contextual and keep + contextual <= 576):
        raise ValueError("div_ctx needs keep > 0, contextual > 0 and keep + contextual <= 576")

    def encode_images(self, images):
        vt = self.get_model().get_vision_tower()
        feats = vt(images)
        proj = self.get_model().mm_projector(feats)
        out = []
        for b in range(proj.shape[0]):
            order = maxmin_order(proj[b], keep)
            kept = sorted(order)
            kept_set = set(kept)
            rest = [i for i in range(proj.shape[1]) if i not in kept_set]
            ctx, targets = contextual_merge(feats[b, rest], contextual)
            out.append(torch.cat([proj[b, kept], self.get_model().mm_projector(ctx.to(feats.dtype))], dim=0))
            log.append({"index": len(log), "order": order, "kept": kept, "targets": [rest[t] for t in targets]})
        return torch.stack(out)

    model.encode_images = types.MethodType(encode_images, model)
    return model
