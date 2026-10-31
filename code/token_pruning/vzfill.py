"""IDEA-0011 on VisionZip (EXP-0044): the layout vfill.py's "fill" needs, from VisionZip's own selection (Yang et al., CVPR
2025; source/VisionZip, read only, not changed). VisionZip-64 gives the LLM 1 + d + c tokens of layer -2 through the
projector: the CLS token and the d dominant patches (by CLS attention, in index order), then c contextual tokens, each a
target patch (every step-th of the remaining patches in index order) plus the mean of the remaining patches whose keys are
most similar to it. The layout: the CLS token gets a slot of its own before the 24 x 24 grid (slot 0, position start),
patch p gets slot 1 + p (position start + 1 + p); a dominant patch fills its own slot and a contextual token its target
patch's slot; every other patch slot holds the dominant or contextual token whose projected feature is most similar
(cosine, float32) to the patch's own projected feature (the projector on its layer -2 hidden state, the input embedding
plain LLaVA would give it); the CLS token represents no patch. So 577 slots, and the text after the image is numbered
as if the image had 577 tokens.

  apply(model)   # after visionzip.visionzip(model, ...) and before vfill.apply_layout(model, "fill")

What VisionZip feeds the LLM is not changed (check_vzfill.py); the extra cost is the projector on the 576 patch states
(for the similarities) and the slots of vfill.
"""
import types

import torch

N_PATCH = 576


def layout(dominant, n_contextual):
    """The slot of each image token in the LLM's order (CLS, the dominant patches in index order, the contextual tokens in
    target order) and the target patches, from the dominant patch indices (0..575), as VisionZip picks the targets."""
    dom = sorted(int(p) for p in dominant)
    taken = set(dom)
    rest = [p for p in range(N_PATCH) if p not in taken]
    step = max(1, len(rest) // n_contextual)
    targets = [rest[i] for i in range(0, len(rest), step)][:n_contextual]
    return [0] + [1 + p for p in dom] + [1 + t for t in targets], dom, targets


@torch.no_grad()
def assign(patch_feats, token_feats, kept_slots):
    """rep (1 + N_PATCH,) long: each slot's representative (an index into the LLM's image tokens). Slot 0 (CLS) and the
    tokens' own slots map to themselves; every other patch slot to the most similar non-CLS token (cosine, float32). Also
    the mean cosine of those slots to their representative."""
    f = torch.nn.functional.normalize(patch_feats.float(), dim=-1)
    g = torch.nn.functional.normalize(token_feats.float(), dim=-1)
    sim = f @ g[1:].T                                    # token 0 is CLS
    rep = torch.empty(1 + N_PATCH, dtype=torch.long, device=f.device)
    rep[1:] = sim.argmax(dim=1) + 1
    rep[0] = 0
    own = torch.as_tensor(kept_slots, dtype=torch.long, device=f.device)
    rep[own] = torch.arange(own.numel(), device=f.device)
    free = torch.ones(1 + N_PATCH, dtype=torch.bool, device=f.device)
    free[own] = False
    idx = free.nonzero().squeeze(1)
    sim_mean = float(sim[idx - 1, rep[idx] - 1].mean()) if idx.numel() else 1.0
    return rep, sim_mean


def apply(model):
    lm = model.get_model()
    vt = lm.get_vision_tower()
    clip = vt.vision_tower
    if not hasattr(clip, "_info") or not hasattr(model, "encode_images_visionzip"):
        raise ValueError("vzfill needs visionzip.visionzip applied first")
    if getattr(model, "vz_layout", False):
        raise ValueError("vzfill was already applied to this model")
    n_ctx = clip._info["contextual"]
    plain_clip, plain_tower, plain_encode = clip.forward, vt.forward, model.encode_images_visionzip

    def clip_forward(*args, **kwargs):                  # keep layer -2's states (VisionZip reads the same ones)
        out = plain_clip(*args, **kwargs)
        vt.vz_hidden = out.hidden_states[-2]
        return out

    def tower_forward(images):                           # keep VisionZip's CLS + dominant indices
        feats, idx = plain_tower(images)
        vt.vz_indices = idx
        return feats, idx

    def encode(self, images):
        vt.vz_hidden = vt.vz_indices = None
        out = plain_encode(images)
        hs, idx = vt.vz_hidden, vt.vz_indices
        if hs is None or idx is None or out.shape[0] != 1 or hs.shape[1] != 1 + N_PATCH:
            raise RuntimeError("vzfill: one image per prefill, VisionZip's states of this image")
        kept, dom, targets = layout(idx[0, 1:] - 1, n_ctx)
        if len(kept) != out.shape[1]:
            raise RuntimeError("vzfill: VisionZip's token count differs from the layout")
        patch = lm.mm_projector(hs[:, 1:])
        rep, sim_mean = assign(patch[0], out[0], kept)
        self.prune_last = {"kept": kept, "rep": rep.cpu(), "n_slots": 1 + N_PATCH, "sim_mean": sim_mean, "dominant": dom, "targets": targets}
        return out

    clip.forward = clip_forward
    vt.forward = tower_forward
    model.encode_images_visionzip = types.MethodType(encode, model)
    model.vz_layout = True
    return model
