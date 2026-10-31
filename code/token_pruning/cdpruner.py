"""CDPruner (Zhang et al., arXiv 2506.10967), ported from its official code (github.com/Theia-4869/CDPruner at 9541616c,
Apache-2.0; source/CDPruner) for LLaVA-1.5 with batch size 1 and one image per prompt: before the LLM, keep `keep` of the
576 projected patch tokens by the conditional DPP of llava/model/llava_arch.py encode_images (lines 140-187), with the
query relevance from CLIP's text tower as in llava/model/multimodal_encoder/clip_encoder.py forward (lines 60-98).

- Relevance: the vision tower's layer -2 patch features through CLIP's post_layernorm and visual_projection, against the
  CLIP text embedding(s) of the question (the prompt text without "\\nAnswer the question using a single word or phrase.",
  as the official model_vqa_loader passes it; longer than 77 tokens -> zero-padded 77-token segments, one embedding each);
  relevance = mean over segments of the negative cosine, min-max normalised over the image's patches.
- Kernel: relevance_i * cos(projected_i, projected_j) * relevance_j; greedy MAP inference of the DPP for `keep` steps.
- The kept tokens go to the LLM in their original order, numbered contiguously (as the official code: position_ids None).
- The image's projected features and kept patches stay in model.prune_last (IDEA-0011's vfill reads them; nothing in
  the computation changes).
- The question for the current prefill: combo._stash_text keeps the prompt ids after the image; text_of maps them back to
  the question file's text (prompt_ids_after_image in run_eval, exactly as model_vqa_loader builds the prompt).
Precision as the official code under LLaVA's device_map="auto": the vision tower, CLIP's text tower and visual_projection
in float32, the projected features in float16, normalised, then float32 for the kernel.
"""
import types

import torch

from combo import _stash_text

SUFFIX = "\nAnswer the question using a single word or phrase."
CLIP = "openai/clip-vit-large-patch14-336"


def load_aux(device, name=CLIP):
    """CLIP's text tower (with projection), its tokenizer and the vision model's visual_projection, float32 (official
    load_text_tower; only the projection of CLIPVisionModelWithProjection is kept)."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast, CLIPVisionModelWithProjection
    vproj = CLIPVisionModelWithProjection.from_pretrained(name).visual_projection.to(device).eval()
    text = CLIPTextModelWithProjection.from_pretrained(name).to(device).eval()
    for p in list(vproj.parameters()) + list(text.parameters()):
        p.requires_grad_(False)
    return {"visual_projection": vproj, "text_tower": text, "tokenizer": CLIPTokenizerFast.from_pretrained(name)}


@torch.no_grad()
def text_embeds(aux, text, device):
    """Official clip_encoder.forward lines 81-89: one embedding per 77-token segment of the zero-padded token ids."""
    tower, L = aux["text_tower"], aux["text_tower"].config.max_position_embeddings
    t = aux["tokenizer"](text=text, return_tensors="pt")
    segment = (t.input_ids.shape[1] - 1) // L + 1
    padding = L * segment - t.input_ids.shape[1]
    t = {k: torch.cat([v, v.new_zeros((v.shape[0], padding))], dim=1).reshape(-1, L).to(device=device) for k, v in t.items()}
    return tower(**t).text_embeds


@torch.no_grad()
def cdpp_select(image_features, image_embeds, text_embeds_, keep):
    """Official encode_images lines 150-185 (B images): the selected patch indices, sorted, (B, keep); and the selection
    order (B, keep)."""
    B, N, _ = image_features.shape
    device = image_features.device
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
    image_normalized = image_normalized.float()
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds_ = text_embeds_ / text_embeds_.norm(dim=-1, keepdim=True)
    relevance = torch.matmul(image_embeds, text_embeds_.t())
    relevance = (-relevance).mean(dim=-1)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    cis = torch.zeros((keep, B, N), device=device)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    select_idx = torch.empty((keep, B), dtype=torch.long, device=device)
    for i in range(keep):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j
        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')
    order = select_idx.t()
    return torch.sort(order).values, order


def apply_cdpruner(model, keep, text_of, aux, log=None):
    """Replace LLaVA's encode_images: keep `keep` projected patches by the conditional DPP for the current question."""
    if not 0 < keep <= 576:
        raise ValueError("cdpruner needs 0 < keep <= 576")
    _stash_text(model)

    def encode_images(self, images):
        vt = self.get_model().get_vision_tower()
        if type(images) is list or images.shape[0] != 1:
            raise ValueError("cdpruner takes one image per prompt (LLaVA-1.5, batch size 1)")
        ids = self.combo_text_ids
        if ids is None or tuple(ids) not in text_of:
            raise ValueError("cdpruner: the current prompt is not one of the question file's")
        question = text_of[tuple(ids)].replace(SUFFIX, "")
        outs = vt.vision_tower(images.to(device=vt.device, dtype=vt.dtype), output_hidden_states=True)
        image_outputs = vt.feature_select(outs)
        image_features = self.get_model().mm_projector(image_outputs.to(images.dtype))
        image_embeds = aux["visual_projection"](vt.vision_tower.vision_model.post_layernorm(image_outputs).float())
        kept, order = cdpp_select(image_features, image_embeds, text_embeds(aux, question, image_features.device), keep)
        if log is not None:
            log.append({"index": len(log), "order": order[0].tolist(), "kept": kept[0].tolist(), "text": question})
        self.prune_last = {"features": image_features[0], "kept": kept[0].tolist()}     # IDEA-0011 (vfill.py)
        return image_features[:, kept[0]]

    model.encode_images = types.MethodType(encode_images, model)
    model.cdpruner_keep, model.cdpruner_log = keep, log
    return model
