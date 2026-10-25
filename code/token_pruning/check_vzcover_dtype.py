"""CPU check of the dtype fix in IDEA-0006's --vz-cover. EXP-0028 R03-R12 failed on the GPU before their first answer:
"mat1 and mat2 must have the same dtype, but got Float and Half" in visionzip_cover._cover.

  python check_vzcover_dtype.py --workdir D:/6CJNM007/tmp/exp0028/vz_smoke

In the real runs LLaVA is loaded with device_map="auto", so the vision tower stays in float32 (llava/model/builder.py
casts it to float16 only for other device maps) while the projector, the LLM and the images are float16; plain LLaVA
casts layer -2 to the images' dtype before the projector (CLIPVisionTower.forward), and VisionZip casts its 64 tokens
the same way. check_vzcover.py's tiny random LLaVA is all float32, so it could not see this. Here the tiny LLaVA is
set up as the real one, with bfloat16 in place of float16 (the CPU has no float16 matmul). Checks: the committed
visionzip_cover (15e75ed) fails on this setup with the GPU's error; the fixed one runs; a routed prefill keeps
VisionZip's tokens and answer; an unrouted one locates the box from plain LLaVA's image input embeddings computed the
plain way (float32 tower, cast to bfloat16, projector), keeps the cover rule's dominant patches, and the LLM receives
the projector on CLS and those patches (exactly) and on VisionZip's merge of the others (to bfloat16 rounding).
"""
import argparse
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
os.environ.setdefault("VISIONZIP_ROOT", "D:/6CJNM007/idea_rearch/source/VisionZip")
import check_vzcover as cv  # noqa: E402
import run_eval  # noqa: E402
import visionzip_cover  # noqa: E402
from fastv_llava import _padding_mask, zoom_box  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

DT = torch.bfloat16
GIT, REPO = "D:/6CJNM007/git/cmd/git.exe", "D:/6CJNM007/idea_rearch"
FAILED = "15e75ed:code/token_pruning/visionzip_cover.py"      # the version EXP-0028 R03-R12 ran
D, CTX, W, H = cv.D, cv.CTX, cv.WIDTH, cv.HEIGHT


def build(workdir):
    model = build_tiny_llava(workdir).to(DT)
    model.get_model().get_vision_tower().to(torch.float32)      # as device_map="auto" leaves LLaVA's vision tower
    return model


def run(workdir, after, idf, texts, cover, module):
    """run_eval's VisionZip (+ cover) on the real dtype layout; module is the visionzip_cover run_eval imports."""
    m = argparse.Namespace(method="visionzip", dominant=D + 1, contextual=CTX, vz_cover=cover, vz_cover_on="unrouted",
                           vz_cover_centre=False, vz_cover_box="8,8,16", vz_visibility=f"{workdir}/vis_vz.npy",
                           fastv_idf=idf, vz_cover_top=0)
    tok = cv.Text()
    saved = sys.modules.get("visionzip_cover")
    sys.modules["visionzip_cover"] = module                  # apply_method imports it by name
    try:
        model = run_eval.apply_method(build(workdir), m, tok)
    finally:
        sys.modules["visionzip_cover"] = saved
    lm = model.get_model()
    got = {"embeds": [], "gen": [], "images": []}

    def pre_lm(mod, args, kwargs):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1:
            got["embeds"].append(e.detach().clone())

    lm.register_forward_pre_hook(pre_lm, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    with torch.no_grad():
        for i, t in enumerate(texts):
            tok.text = t
            images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i)).to(DT)
            got["images"].append(images)
            out = model.generate(ids, images=images, image_sizes=[(W, H)], do_sample=False, max_new_tokens=2, use_cache=True)
            got["gen"].append(out.tolist())
    return model, lm, got


def reference(model, images, after, idf, vis, cover):
    """Offline, on the same dtype layout: VisionZip's own patches, the box from plain LLaVA's image input embeddings
    (cv.PLAIN_FORWARD casts layer -2 to the images' dtype, then the projector), the cover rule, the 64 image tokens."""
    lm = model.get_model()
    tower = lm.get_vision_tower()
    with torch.no_grad():
        outs = tower.vision_tower(images.to(device=tower.device, dtype=tower.dtype), output_hidden_states=True, output_attentions=True)
        hs, attn = outs.hidden_states[-2], outs.attentions[-2]
        metric = tower.vision_tower.vision_model.encoder.layers[-2].metric
        cls = attn[0, :, 0, 1:].sum(0)
        own = cls.topk(D).indices
        idf_t = torch.tensor([idf["idf"].get(int(t), idf["unseen"]) for t in after], dtype=torch.float32)
        w = idf_t * torch.as_tensor(vis).float()[after]
        h = lm.embed_tokens(after).float()
        q = (w[:, None] * h).sum(0, keepdim=True) / w.sum()
        plain = lm.mm_projector(cv.PLAIN_FORWARD(tower, images))[0].float()
        cos = torch.nn.functional.cosine_similarity(plain, q, dim=-1)
        allowed = ~_padding_mask(W, H, 24)
        grid, _, info = zoom_box(cos, allowed, 24, W, H, 8, 8, 16)
        box = torch.zeros(24, 24, dtype=torch.bool)
        box[grid[0]:grid[2] + 1, grid[1]:grid[3] + 1] = True
        region = box.ravel() & allowed
        a_in = int(region[own].sum())
        inside, outside = region.nonzero().squeeze(1), (~box.ravel()).nonzero().squeeze(1)
        n_c = min(a_in + cover, D, inside.numel())
        dom = torch.cat((inside[cls[inside].topk(n_c).indices], outside[cls[outside].topk(D - n_c).indices]))
        feats = lm.mm_projector(cv.merge(hs[0], metric[0], dom.tolist())[None].to(images.dtype))[0]
    return {"hs_dtype": hs.dtype, "own": sorted(own.tolist()), "dom": sorted(dom.tolist()), "grid": list(grid),
            "info": info, "a_in": a_in, "n_c": n_c, "feats": feats, "plain": plain,
            "cast_proj": lm.mm_projector(hs[:, 1:].to(images.dtype))[0].float()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    rng = np.random.default_rng(0)
    idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
    vis = rng.uniform(0, 1, 1000).astype(np.float32)
    np.save(f"{a.workdir}/vis_vz.npy", vis)
    checks = {}
    checks["this check imports the fixed visionzip_cover from its own folder"] = cv.under(visionzip_cover.__file__, HERE)
    old_path = f"{a.workdir}/visionzip_cover_15e75ed.py"
    with open(old_path, "wb") as f:
        f.write(subprocess.run([GIT, "-C", REPO, "show", FAILED], capture_output=True, check=True).stdout)
    spec = importlib.util.spec_from_file_location("visionzip_cover_15e75ed", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    texts = [cv.ROUTED, cv.UNROUTED, cv.UNROUTED]

    # A. the committed version on the real dtype layout: the GPU's error
    try:
        run(a.workdir, after, idf, texts, 16, old)
        err = ""
    except RuntimeError as e:
        err = str(e)
    checks["the committed visionzip_cover (15e75ed) fails on this layout: mat1 and mat2 must have the same dtype"] = (
        "must have the same dtype" in err and "Float" in err)

    # B. the fixed version
    model, lm, got = run(a.workdir, after, idf, texts, 16, visionzip_cover)
    model_n, _, got_n = run(a.workdir, after, idf, texts, 0, visionzip_cover)
    tower = lm.get_vision_tower()
    proj_dt = next(lm.mm_projector.parameters()).dtype
    checks["layout: vision tower float32, projector and LLM bfloat16, images bfloat16"] = (
        tower.dtype == torch.float32 and proj_dt == DT and lm.embed_tokens.weight.dtype == DT
        and all(x.dtype == DT for x in got["images"]))
    log = model.vz_cover_log
    checks["the fixed visionzip_cover runs: three prefills, the routed one skipped, two covered"] = (
        len(got["gen"]) == 3 and [r["cover"] for r in log] == [False, True, True] and log[0].get("skip") == "routed")
    checks["routed prefill: image tokens and answer equal VisionZip's without cover"] = (
        torch.equal(got["embeds"][0], got_n["embeds"][0]) and got["gen"][0] == got_n["gen"][0])
    ok_box = ok_rule = ok_dom = ok_ctx = ok_cast = True
    gap = 0.0
    for j in (1, 2):
        rec, r = log[j], reference(model, got["images"][j], after, idf, vis, 16)
        ok_cast &= r["hs_dtype"] == torch.float32 and torch.equal(r["plain"], r["cast_proj"])
        ok_box &= rec["grid_box"] == r["grid"] and rec["side"] == r["info"]["side"]
        ok_rule &= (rec["dominant"] == r["dom"] and rec["baseline"] == r["own"] and rec["baseline_in_box"] == r["a_in"]
                    and rec["in_box"] == r["n_c"] == min(r["a_in"] + 16, D))
        tok = got["embeds"][j][0, SYS_LEN:SYS_LEN + 1 + D + CTX]
        ok_dom &= torch.equal(tok[:1 + D], r["feats"][:1 + D])
        d = (tok[1 + D:].float() - r["feats"][1 + D:].float()).abs().max() / r["feats"][1 + D:].float().abs().max()
        gap = max(gap, float(d))
        ok_ctx &= bool(d <= 0.02)
    checks["layer -2 is float32 here; the projector on it cast to bfloat16 = plain LLaVA's image input embeddings (exact)"] = bool(ok_cast)
    checks["unrouted prefills: box = zoom_box on the query cosine over plain LLaVA's image input embeddings"] = bool(ok_box)
    checks["unrouted prefills: dominant = VisionZip's in-box count + 16 inside the box by CLS attention, the rest outside"] = bool(ok_rule)
    checks["unrouted prefills: the LLM's CLS and dominant tokens = projector on those patches, cast to bfloat16 (exact)"] = bool(ok_dom)
    checks[f"unrouted prefills: the contextual tokens = projector on VisionZip's merge, to bfloat16 rounding (max {gap:.4f} of the scale)"] = bool(ok_ctx)
    checks["the cover changed the unrouted prefills' tokens (the check is not vacuous)"] = all(
        log[j]["changed"] > 0 and not torch.equal(got["embeds"][j], got_n["embeds"][j]) for j in (1, 2))
    for name, ok_ in checks.items():
        print(("ok   " if ok_ else "FAIL ") + name)
    print(f"{sum(checks.values())}/{len(checks)} checks pass")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
