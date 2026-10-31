"""IDEA-0010: the image tokens frozen in chosen LLM layers (ShortV, Yuan et al., ICCV 2025, arXiv 2504.00502: in a frozen
layer "the visual tokens do not attend to other tokens, and only the text tokens function as queries"), written here for
this pipeline from the paper; ShortV's code was not downloaded. In a frozen layer the image tokens keep their input hidden
states (no attention output, no MLP update), while every other token attends to them as usual: their keys and values in
that layer come from those input states, and the prefill caches them so that the generated tokens see the same.

- apply_freeze(model, layers, log): forward hooks on the chosen decoder layers put the image rows of the layer's output
  back to the layer's input during each prefill (the image span recorded by fastv_llava._record_image_span); decoding
  steps (one token at a time) are left alone. Computing a frozen layer for every row and then restoring the image rows
  gives the same text rows as computing the text queries only (causal attention; check_freeze.py); the FLOPs of a frozen
  image token are counted as its key and value projections (analysis), not as run here.
- parse_layers("0,2,3,12,17-31", 32) -> [0, 2, 3, 12, 17, ..., 31].
- SHORTV_LLAVA15_7B: ShortV's Table 8, the 19 layers it freezes in LLaVA-1.5-7B (N = 19, 55% FLOPs), chosen there by the
  KL divergence of the output logits on 40 samples (20 GQA, 20 Flickr30K).
"""
import torch

from fastv_llava import _record_image_span

SHORTV_LLAVA15_7B = (31, 29, 30, 28, 0, 26, 27, 25, 24, 22, 23, 21, 2, 3, 20, 18, 17, 12, 19)
SHORTV_SPEC = "0,2,3,12,17-31"


def parse_layers(spec, n_layers):
    """Comma-separated layer indices and ranges a-b (inclusive) -> sorted list; every index in 0..n_layers-1, no repeats."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            if a > b:
                raise ValueError(f"bad range {part!r}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out) or not out or min(out) < 0 or max(out) >= n_layers:
        raise ValueError(f"layers {spec!r}: need distinct indices in 0..{n_layers - 1}")
    return sorted(out)


def apply_freeze(model, layers, log=None):
    """Freeze the image tokens in `layers` (decoder layer indices) during every prefill. log: a dict that gets the layers,
    and per prefill the image span, the prompt length and how many layers restored it."""
    lm = model.get_model()
    if getattr(lm, "freeze_layers", None) is not None:
        raise ValueError("apply_freeze was already applied to this model")
    _record_image_span(model)
    lm.freeze_layers = list(layers)
    if log is not None:
        log.update({"layers": list(layers), "prefills": []})
    saved = {}

    def pre(i, module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] > 1:                                    # prefill (decoding feeds one token at a time)
            start, n = lm.fastv_image_span
            if i == layers[0] and log is not None:
                log["prefills"].append({"start": start, "n_img": n, "len": h.shape[1], "restored": 0})
            saved[i] = h[:, start:start + n].clone()

    def post(i, module, args, kwargs, output):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] > 1:
            start, n = lm.fastv_image_span
            out = output[0] if isinstance(output, tuple) else output
            out[:, start:start + n] = saved.pop(i)
            if log is not None:
                log["prefills"][-1]["restored"] += 1
        return output

    for i in layers:
        layer = lm.layers[i]
        layer.register_forward_pre_hook(lambda mod, args, kwargs, i=i: pre(i, mod, args, kwargs), with_kwargs=True)
        layer.register_forward_hook(lambda mod, args, kwargs, output, i=i: post(i, mod, args, kwargs, output), with_kwargs=True)
    return model


@torch.no_grad()
def frozen_text_rows(layer, h, start, n, mask, pos):
    """A frozen layer computed with the non-image rows as the only queries (for the check): keys and values from every
    row's input, the attention output, the residual and the MLP for the non-image rows; the image rows unchanged.
    transformers 4.37's LlamaDecoderLayer / LlamaAttention, eager arithmetic."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    att = layer.self_attn
    b, L, _ = h.shape
    x = layer.input_layernorm(h)
    q_rows = torch.cat((torch.arange(start), torch.arange(start + n, L)))
    q = att.q_proj(x[:, q_rows]).view(b, len(q_rows), att.num_heads, att.head_dim).transpose(1, 2)
    k = att.k_proj(x).view(b, L, att.num_key_value_heads, att.head_dim).transpose(1, 2)
    v = att.v_proj(x).view(b, L, att.num_key_value_heads, att.head_dim).transpose(1, 2)
    cos, sin = att.rotary_emb(v, seq_len=L)
    qf = torch.zeros(b, att.num_heads, L, att.head_dim, dtype=q.dtype)
    qf[:, :, q_rows] = q
    qf, k = apply_rotary_pos_emb(qf, k, cos, sin, pos)
    q = qf[:, :, q_rows]
    k, v = repeat_kv(k, att.num_key_value_groups), repeat_kv(v, att.num_key_value_groups)
    w = q @ k.transpose(2, 3) / att.head_dim ** 0.5 + mask[:, :, q_rows]
    w = torch.softmax(w.float(), dim=-1).to(q.dtype)
    a = (w @ v).transpose(1, 2).reshape(b, len(q_rows), att.hidden_size)
    y = h[:, q_rows] + att.o_proj(a)
    y = y + layer.mlp(layer.post_attention_layernorm(y))
    out = h.clone()
    out[:, q_rows] = y
    return out
