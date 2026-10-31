"""IDEA-0011: give the LLM the pruned image back as a layout, not as tokens. After a selection before the LLM (CDPruner or
DivPrune keeping n of the N = 576 patch tokens), every pruned patch slot is filled, in every layer, with the key and value
of its most similar kept token (cosine of the projected features, the similarity both selections use), rotated to the
slot's own position. The kept image tokens and the text then attend to a gap-free 24 x 24 image at the unpruned positions,
while only the n kept tokens are computed (their queries, attention outputs and MLP); the slots cost a gather and a
rotation per layer and the extra attention pairs. Related: DyMU's virtual token unmerging (arXiv 2504.17040: merged
tokens expanded back to the full length with their tracked positions, then re-merged by averaging after each layer);
here nothing is merged or averaged: the kept tokens stay exact and the pruned slots are key / value copies only.

apply_layout(model, mode, log=None, seed=0), after the selection (run_eval: apply_method, then this):
- "orig": the kept image tokens keep their original positions (start + patch index) and the text after the image is
  numbered as in the unpruned prompt (start + N + t); nothing else changes (the gaps stay).
- "fill": as "orig", and every query attends to the N slots instead of the n kept tokens, causally by position (a query
  at position p sees the slots at positions <= p); the generated tokens see them too.
- "shuffle": as "fill", but the pruned slots' representatives are permuted among the pruned slots (a CPU generator seeded
  once per run, so the draws follow the question order): each kept token fills as many slots as in "fill", at other places.
- "mass": the base's contiguous numbering, no slots; each kept token's attention logit, as a key, gets + log(c_i) in every
  layer for every query, c_i = 1 + the number of pruned patches whose most similar kept token it is (the attention mass
  its slots would have in "fill" if they all sat at its own position).
The KV cache keeps only the real tokens (so generate's bookkeeping of the cache length is unchanged); in "fill" and
"shuffle" each layer's N rotated slot keys and values are kept beside it for the decoding steps of the same prompt.
Batch size 1, one image per prompt, transformers 4.37's LlamaAttention / LlamaSdpaAttention arithmetic (pretraining_tp 1).
"""
import math
import types

import torch
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half

from fastv_llava import _full_rotary_forward, _record_image_span

MODES = ("orig", "fill", "shuffle", "mass")
N_SLOTS = 576


@torch.no_grad()
def assignment(features, kept, shuffle_generator=None):
    """rep (N,) long: for each patch slot, the index in the kept list (sorted, as fed to the LLM) of the kept token whose
    projected feature is most similar (cosine, float32); kept slots map to themselves. With shuffle_generator the pruned
    slots' entries are permuted among the pruned slots. Also the mean cosine of the pruned slots to their representative
    (before any shuffle)."""
    f = torch.nn.functional.normalize(features.float(), dim=-1)
    kept = torch.as_tensor(kept, dtype=torch.long, device=f.device)
    sim = f @ f[kept].T
    rep = sim.argmax(dim=1)
    rep[kept] = torch.arange(kept.numel(), device=f.device)
    pruned = torch.ones(f.shape[0], dtype=torch.bool, device=f.device)
    pruned[kept] = False
    idx = pruned.nonzero().squeeze(1)
    sim_mean = float(sim[idx, rep[idx]].mean()) if idx.numel() else 1.0
    if shuffle_generator is not None and idx.numel() > 1:
        perm = torch.randperm(idx.numel(), generator=shuffle_generator).to(f.device)
        rep[idx] = rep[idx][perm]
    return rep, sim_mean


def remap_positions(pid, start, kept, n_slots=N_SLOTS):
    """Contiguous positions of the pruned prompt -> the unpruned prompt's: the i-th image token -> start + kept[i], the
    tokens after the image -> + (n_slots - n); the system prompt is unchanged."""
    kept = torch.as_tensor(kept, dtype=torch.long, device=pid.device)
    n = kept.numel()
    out = pid.clone()
    img = (pid >= start) & (pid < start + n)
    out[img] = start + kept[pid[img] - start]
    after = pid >= start + n
    out[after] = pid[after] + (n_slots - n)
    return out


def _stash(model):
    """The selection's features and kept patches of the current image: set by divprune / cdpruner's encode_images."""
    last = getattr(model, "prune_last", None)
    if last is None:
        raise ValueError("vfill needs a selection before the LLM that records prune_last (divprune or cdpruner)")
    model.prune_last = None
    return last["features"], last["kept"]


def apply_layout(model, mode, log=None, seed=0):
    """Apply one of MODES on top of the selection already applied to `model`. log: a dict that gets the mode and one
    record per prefill."""
    if mode not in MODES:
        raise ValueError(f"layout mode must be one of {MODES}")
    lm = model.get_model()
    if getattr(lm, "layout_mode", None) is not None:
        raise ValueError("apply_layout was already applied to this model")
    if not (hasattr(model, "divprune_keep") or hasattr(model, "cdpruner_keep")):
        raise ValueError("apply_layout needs --method divprune or cdpruner")
    _record_image_span(model)
    lm.layout_mode, lm.layout_ctx, lm.layout_slots = mode, None, {}
    if log is not None:
        log.update({"mode": mode, "seed": seed if mode == "shuffle" else None, "prefills": []})
    gen = torch.Generator().manual_seed(seed) if mode == "shuffle" else None
    plain_encode = model.encode_images

    def encode_images(self, images):
        out = plain_encode(images)
        feats, kept = _stash(self)
        rep, sim_mean = assignment(feats, kept, gen)
        n = len(kept)
        counts = torch.bincount(rep, minlength=n)
        lm.layout_ctx = {"kept": torch.as_tensor(kept, dtype=torch.long), "rep": rep, "counts": counts, "used": False}
        if log is not None:
            log["prefills"].append({"n": n, "slots": int(rep.numel()), "count_max": int(counts.max()),
                                    "count_ones": int((counts == 1).sum()), "sim_mean": round(sim_mean, 6),
                                    "rep": rep.tolist(), "layers": 0, "decode_steps": 0})
        return out

    model.encode_images = types.MethodType(encode_images, model)
    plain_forward = lm.forward

    def forward(*args, **kw):
        if args:
            raise ValueError("vfill: LlamaModel.forward is called with keywords in LLaVA")
        x = kw["inputs_embeds"] if kw.get("inputs_embeds") is not None else kw["input_ids"]
        q_len = x.shape[1]
        pkv = kw.get("past_key_values")
        past = 0 if pkv is None else (pkv.get_seq_length() if isinstance(pkv, Cache) else pkv[0][0].shape[2])
        ctx = lm.layout_ctx
        if q_len > 1:                                         # the prefill of a new prompt
            if ctx is None or ctx["used"] or past != 0:
                raise RuntimeError("vfill: a prefill without a fresh selection of its image")
            start, n = lm.fastv_image_span
            if n != ctx["kept"].numel():
                raise RuntimeError("vfill: the image span differs from the selection")
            ctx.update(used=True, start=start, n=n)
            lm.layout_slots = {}
            if log is not None:
                log["prefills"][-1].update(start=start, len=q_len)
        pid = kw.get("position_ids")
        if pid is None:
            pid = torch.arange(past, past + q_len, device=x.device).unsqueeze(0)
        if mode != "mass":
            pid = remap_positions(pid, ctx["start"], ctx["kept"], N_SLOTS)
        if q_len > 1 and log is not None:                     # the prompt's positions as the layers see them
            log["prefills"][-1]["positions_ok"] = bool(torch.equal(pid[0].cpu(), expected_positions(
                q_len, ctx["start"], ctx["kept"], mode)))
        kw["position_ids"] = pid
        return plain_forward(**kw)

    lm.forward = forward
    for layer in lm.layers:
        att = layer.self_attn
        if att.config.pretraining_tp != 1:
            raise ValueError("vfill assumes pretraining_tp 1")
        if mode != "mass":                                    # positions beyond the cache length
            att.rotary_emb.forward = types.MethodType(_full_rotary_forward, att.rotary_emb)
        if mode != "orig":
            att.forward = types.MethodType(_make_attention(lm, mode, log), att)
    return model


def expected_positions(q_len, start, kept, mode):
    """The prefill's position ids under `mode` (for the log's check)."""
    kept = torch.as_tensor(kept, dtype=torch.long).cpu()
    n = kept.numel()
    if mode == "mass":
        return torch.arange(q_len)
    return torch.cat((torch.arange(start), start + kept, torch.arange(start + N_SLOTS, start + N_SLOTS + q_len - start - n)))


def _make_attention(lm, mode, log):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False,
                use_cache=False, **kwargs):
        if output_attentions:
            raise ValueError("vfill does not return attention maps")
        ctx = lm.layout_ctx
        bsz, q_len, _ = hidden_states.size()
        if bsz != 1:
            raise ValueError("vfill supports batch size 1 only")
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_len = q_len + (past_key_value.get_usable_length(q_len, self.layer_idx) if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(v, seq_len=kv_len)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        if past_key_value is not None:
            k_all, v_all = past_key_value.update(kr, v, self.layer_idx, {"sin": sin, "cos": cos})
        else:
            k_all, v_all = kr, v
        start, n = ctx["start"], ctx["n"]
        prefill = q_len > 1
        dtype, dev = qr.dtype, qr.device
        if log is not None:                                   # how many layers used the layout, per prefill and step
            rec = log["prefills"][-1]
            if prefill:
                rec["layers"] = 1 if self.layer_idx == 0 else rec["layers"] + 1
            elif self.layer_idx == 0:
                rec["decode_steps"] += 1
        if mode == "mass":
            keys, vals = k_all, v_all
            bias = torch.zeros(k_all.shape[2], dtype=dtype, device=dev)
            bias[start:start + n] = ctx["counts"].to(dev).to(dtype).log()
            if prefill:
                past = k_all.shape[2] - q_len
                cols = torch.arange(k_all.shape[2], device=dev)
                rows = torch.arange(past, past + q_len, device=dev)
                causal = torch.where(cols[None, :] <= rows[:, None], 0.0, torch.finfo(dtype).min).to(dtype)
                mask = (causal + bias[None, :])[None, None]
            else:
                mask = bias[None, None, None, :].expand(1, 1, q_len, -1)
        else:
            if prefill:
                rep = ctx["rep"].to(dev)
                slot_pos = start + torch.arange(N_SLOTS, device=dev)
                kp = k[:, :, start:start + n][:, :, rep]                       # the representatives' keys, before RoPE
                cs, sn = cos[slot_pos][None, None], sin[slot_pos][None, None]
                lm.layout_slots[self.layer_idx] = (kp * cs + rotate_half(kp) * sn, v[:, :, start:start + n][:, :, rep])
            ks, vs = lm.layout_slots[self.layer_idx]
            keys = torch.cat((k_all[:, :, :start], ks, k_all[:, :, start + n:]), dim=2)
            vals = torch.cat((v_all[:, :, :start], vs, v_all[:, :, start + n:]), dim=2)
            if prefill:
                key_pos = torch.cat((torch.arange(start, device=dev), start + torch.arange(N_SLOTS, device=dev),
                                     torch.arange(start + n, k_all.shape[2], device=dev) + (N_SLOTS - n)))
                mask = torch.where(key_pos[None, :] <= position_ids[0][:, None], 0.0, torch.finfo(dtype).min).to(dtype)[None, None]
            else:
                mask = None                                   # one new token, the last position: every key is visible
        keys = repeat_kv(keys, self.num_key_value_groups)
        vals = repeat_kv(vals, self.num_key_value_groups)
        if qr.device.type == "cuda":
            qr, keys, vals = qr.contiguous(), keys.contiguous(), vals.contiguous()
        out = torch.nn.functional.scaled_dot_product_attention(qr, keys, vals, attn_mask=mask)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(out), None, past_key_value

    return forward


@torch.no_grad()
def reference_logits(ref, embeds, start, kept, rep, mode, counts=None):
    """For the check: the layers of a plain LLaVA (same weights) by hand, eager arithmetic with a float32 softmax, on the
    pruned prompt's embeddings; `mode` as in apply_layout. Returns the last row's logits."""
    lm = ref.get_model()
    L = embeds.shape[1]
    kept = torch.as_tensor(kept, dtype=torch.long)
    n = kept.numel()
    pos = expected_positions(L, start, kept, mode)
    h = embeds
    for layer in lm.layers:
        att = layer.self_attn
        x = layer.input_layernorm(h)
        q = att.q_proj(x).view(1, L, att.num_heads, att.head_dim).transpose(1, 2)
        k = att.k_proj(x).view(1, L, att.num_key_value_heads, att.head_dim).transpose(1, 2)
        v = att.v_proj(x).view(1, L, att.num_key_value_heads, att.head_dim).transpose(1, 2)
        big = int(pos.max()) + N_SLOTS + 2
        cos, sin = att.rotary_emb.cos_cached[:big].to(q.dtype), att.rotary_emb.sin_cached[:big].to(q.dtype)
        if big > att.rotary_emb.max_seq_len_cached:
            raise ValueError("reference: positions beyond the rotary table")
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin, pos.unsqueeze(0))
        if mode in ("fill", "shuffle"):
            slot_pos = start + torch.arange(N_SLOTS)
            kp = k[:, :, start:start + n][:, :, rep]
            ks = kp * cos[slot_pos][None, None] + rotate_half(kp) * sin[slot_pos][None, None]
            vs = v[:, :, start:start + n][:, :, rep]
            keys = torch.cat((kr[:, :, :start], ks, kr[:, :, start + n:]), dim=2)
            vals = torch.cat((v[:, :, :start], vs, v[:, :, start + n:]), dim=2)
            key_pos = torch.cat((torch.arange(start), slot_pos, pos[start + n:]))
        else:
            keys, vals, key_pos = kr, v, pos
        w = (qr @ keys.transpose(2, 3)).float() / math.sqrt(att.head_dim)
        w = w.masked_fill(key_pos[None, None, None, :] > pos[None, None, :, None], float("-inf"))
        if mode == "mass":
            b = torch.zeros(keys.shape[2])
            b[start:start + n] = torch.as_tensor(counts).float().log()
            w = w + b
        w = torch.softmax(w, dim=-1).to(q.dtype)
        a = (w @ vals).transpose(1, 2).reshape(1, L, att.hidden_size)
        h = h + att.o_proj(a)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return ref.lm_head(lm.norm(h))[0, -1]
