"""DivPrune-style selection (Alvar et al., CVPR 2025, arXiv 2503.02175), written here from the paper's Algorithm 1; the
official code was not downloaded. Max-min diversity with the cosine distance d = 1 - cos: the first token is the one
whose nearest neighbour is farthest away, then repeatedly the candidate whose nearest selected token is farthest away.
The greedy order does not depend on how many are taken, so the first 47 of a 64-token selection are the 47-token one.

- apply_divprune(model, keep, log): before the LLM (DivPrune's default, "layer 0"): LLaVA's encode_images keeps the
  `keep` projector outputs of each image chosen this way, in their original order, so the LLM sees only those.
- The last encoded image's projected features and kept patches stay in model.prune_last (IDEA-0011's vfill reads
  them; nothing in the computation changes).
- maxmin_order is also used inside the LLM by fastv_llava's score kind "div" (K = 1: layer 0 sees all patches, the
  selection is made on the same input embeddings and applied from layer 1 on).
"""
import types

import torch


def maxmin_order(x, m):
    """The first m rows of x (n x d) in the greedy max-min order, as a list of row indices in selection order."""
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    d = 1 - x @ x.T
    n = d.shape[0]
    m = min(m, n)
    off = d.clone()
    off.fill_diagonal_(float("inf"))
    first = int(off.min(dim=1).values.argmax())
    order = [first]
    taken = torch.zeros(n, dtype=torch.bool, device=d.device)
    taken[first] = True
    near = d[first].clone()                        # each candidate's distance to its nearest selected row
    while len(order) < m:
        k = int(near.masked_fill(taken, float("-inf")).argmax())
        order.append(k)
        taken[k] = True
        near = torch.minimum(near, d[k])
    return order


def apply_divprune(model, keep, log=None):
    """Keep `keep` of each image's projected patch tokens by maxmin_order, before the LLM. log: a list that gets one
    record per encoded image: the selection order and the kept patch indices (sorted, as fed to the LLM)."""
    plain = model.encode_images

    def encode_images(self, images):
        feats = plain(images)
        out = []
        for b in range(feats.shape[0]):
            order = maxmin_order(feats[b], keep)
            idx = sorted(order)
            out.append(feats[b, idx])
            self.prune_last = {"features": feats[b], "kept": idx}     # IDEA-0011 (vfill.py); batch size 1 there
            if log is not None:
                log.append({"index": len(log), "order": order, "kept": idx})
        return torch.stack(out)

    model.encode_images = types.MethodType(encode_images, model)
    model.divprune_keep, model.divprune_log = keep, log
    return model
