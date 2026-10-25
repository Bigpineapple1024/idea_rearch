"""CPU check of IDEA-0006's --vz-cover (locate-then-cover on VisionZip) on a tiny random LLaVA (smoke_test.build_tiny_llava).

  python check_vzcover.py --workdir D:/6CJNM007/tmp/exp0028/vz_smoke

VisionZip-64 (dominant 54: CLS + 53 patches; contextual 10) on a 640 x 480 image. Checks: without --vz-cover run_eval's
VisionZip is visionzip.visionzip itself, and the copied tower forward equals VisionZip's when there is no query; with
cover on the unrouted questions a routed prefill keeps VisionZip's tokens and answers; an unrouted one locates the box
the IDF x visibility query points at (cosine over plain LLaVA's image input embeddings, as IDEA-0005), keeps
VisionZip's own in-box count plus n inside it by CLS attention and the rest outside, and the LLM receives the projector
on CLS, those patches in index order and VisionZip's merge of the others (written out here independently); the
centred box, cover on every question, cover 32, the no-box control (the 16 patches most similar to the query, EXP-0028
v1.1) and a question without visible weight; argument and config checks.
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
os.environ.setdefault("VISIONZIP_ROOT", "D:/6CJNM007/idea_rearch/source/VisionZip")
import exp0002  # noqa: E402
import fastv_llava  # noqa: E402
import run_eval  # noqa: E402
import visionzip_cover  # noqa: E402
from fastv_llava import _content_band, _padding_mask, zoom_box  # noqa: E402
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

PLAIN_FORWARD = CLIPVisionTower.forward          # plain LLaVA's, saved before VisionZip patches the class
WIDTH, HEIGHT = 640, 480
D, CTX = 53, 10                                  # dominant patches besides CLS, contextual tokens (VisionZip-64)
ROUTED = "Is there a cat in the image?\nAnswer the question using a single word or phrase."
UNROUTED = "What color is the cat?\nAnswer the question using a single word or phrase."


class Text:
    """Tokenizer stand-in: decode gives the prompt text the router reads, switched per prefill."""
    text = ""

    def decode(self, ids):
        return self.text


def under(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


def refused(argv):
    saved = sys.argv
    sys.argv = ["run_eval.py"] + argv + ["--", "--model-path", "m"]
    try:
        run_eval.parse()
        return False
    except SystemExit:
        return True
    finally:
        sys.argv = saved


def run(workdir, after, idf, texts, cover=16, on="unrouted", centre=False, vis="vis_vz.npy", top=0):
    m = argparse.Namespace(method="visionzip", dominant=D + 1, contextual=CTX, vz_cover=cover, vz_cover_on=on,
                           vz_cover_centre=centre, vz_cover_box="8,8,16", vz_visibility=f"{workdir}/{vis}", fastv_idf=idf,
                           vz_cover_top=top)
    tok = Text()
    model = run_eval.apply_method(build_tiny_llava(workdir), m, tok)
    lm = model.get_model()
    got = {"embeds": [], "gen": [], "images": []}

    def pre_lm(module, args, kwargs):
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
            images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))
            got["images"].append(images)
            out = model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
            got["gen"].append(out.tolist())
    return model, lm, got


def merge(hs, metric, dominant):
    """VisionZip's output for one image, written out: CLS and the dominant patches in index order, then CTX tokens, each a
    target (at a uniform step over the other patches) plus the mean of the other patches most similar to it by the
    normalised layer -2 key means."""
    keep = [0] + sorted(p + 1 for p in dominant)
    rest = [i for i in range(hs.shape[0]) if i not in set(keep)]
    mn = metric[rest] / metric[rest].norm(dim=-1, keepdim=True)
    step = max(1, len(rest) // CTX)
    targets = list(range(0, len(rest), step))[:CTX]
    others = [j for j in range(len(rest)) if j not in set(targets)]
    assign = torch.bmm(mn[others][None], mn[targets].T[None])[0].argmax(1).tolist()     # same kernel as VisionZip's
    ctx = []
    for t in range(CTX):
        members = [rest[others[j]] for j in range(len(others)) if assign[j] == t]
        agg = hs[members].sum(0) / max(1, len(members)) if members else torch.zeros_like(hs[0])
        ctx.append(hs[rest[targets[t]]] + agg)
    return torch.cat([hs[keep], torch.stack(ctx)])


def reference(model, images, after, idf, vis, cover, centre=False, top=0):
    """Offline: VisionZip's own dominant patches, the located box (or, with top, the top patches by query cosine), the
    cover rule and the 64 image input embeddings."""
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
        plain = lm.mm_projector(PLAIN_FORWARD(tower, images))[0].float()     # plain LLaVA's image input embeddings
        cos = torch.nn.functional.cosine_similarity(plain, q, dim=-1)
        allowed = ~_padding_mask(WIDTH, HEIGHT, 24)
        grid, _, info = zoom_box(cos, allowed, 24, WIDTH, HEIGHT, 8, 8, 16, centre=centre)
        box = torch.zeros(24, 24, dtype=torch.bool)
        box[grid[0]:grid[2] + 1, grid[1]:grid[3] + 1] = True
        box = box.ravel()
        top_set = sorted(cos.masked_fill(~allowed, float("-inf")).topk(top).indices.tolist()) if top else None
        if top:
            box = torch.zeros(576, dtype=torch.bool)
            box[top_set] = True
        region = box & allowed
        a_in = int(region[own].sum())
        inside, outside = region.nonzero().squeeze(1), (~box).nonzero().squeeze(1)
        n_c = min(a_in + cover, D, inside.numel())
        dom = torch.cat((inside[cls[inside].topk(n_c).indices], outside[cls[outside].topk(D - n_c).indices]))
        feats = lm.mm_projector(merge(hs[0], metric[0], dom.tolist())[None])[0]
        own_feats = lm.mm_projector(merge(hs[0], metric[0], own.tolist())[None])[0]
    return {"own": sorted(own.tolist()), "dom": sorted(dom.tolist()), "grid": list(grid), "info": info, "a_in": a_in,
            "n_c": n_c, "region": region, "feats": feats, "own_feats": own_feats, "plain_proj": plain, "top": top_set,
            "cls": cls,
            "proj_from_layer": lm.mm_projector(hs[:, 1:])[0].float()}


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
    vis0 = vis.copy()
    vis0[after.numpy()] = 0.0
    np.save(f"{a.workdir}/vis_zero.npy", vis0)
    checks = {}
    checks["this check imports the new run_eval, exp0002, fastv_llava and visionzip_cover"] = all(
        under(m.__file__, HERE) for m in (run_eval, exp0002, fastv_llava, visionzip_cover))
    img_len = SYS_LEN + 1 + D + CTX + TEXT_LEN

    # A. without --vz-cover: run_eval's VisionZip is visionzip.visionzip itself; the copied forward is VisionZip's
    model0, lm0, got0 = run(a.workdir, after, idf, [ROUTED, UNROUTED], cover=0)
    from visionzip import visionzip
    direct = visionzip(build_tiny_llava(a.workdir), dominant=D + 1, contextual=CTX)
    emb_d, gen_d = [], []
    direct.get_model().register_forward_pre_hook(
        lambda mod, args, kw: emb_d.append(kw["inputs_embeds"].clone()) if kw.get("inputs_embeds") is not None and kw["inputs_embeds"].shape[1] > 1 else None,
        with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    with torch.no_grad():
        for images in got0["images"]:
            gen_d.append(direct.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True).tolist())
    checks["without --vz-cover: image tokens and answers equal visionzip.visionzip's"] = (
        len(emb_d) == 2 and all(torch.equal(x, y) for x, y in zip(emb_d, got0["embeds"])) and gen_d == got0["gen"])
    checks["without --vz-cover: the tower keeps VisionZip's class forward, no hook"] = (
        "forward" not in vars(lm0.get_vision_tower()) and not hasattr(model0, "vz_cover_log"))
    tower = lm0.get_vision_tower()
    tower.vz_cover = {"query": None, "record": None}
    with torch.no_grad():
        ref_out = tower.forward(got0["images"][0])                     # VisionZip's (the class attribute it patched)
        our_out = visionzip_cover._forward(tower, got0["images"][0])
    checks["the copied tower forward without a query = VisionZip's forward (features and indices)"] = (
        torch.equal(ref_out[0], our_out[0]) and torch.equal(ref_out[1], our_out[1]) and ref_out[0].shape[1] == 1 + D + CTX)

    # B. cover 16 on the unrouted questions: [routed, unrouted, unrouted]
    model, lm, got = run(a.workdir, after, idf, [ROUTED, UNROUTED, UNROUTED])
    log = model.vz_cover_log
    r0 = reference(model, got["images"][0], after, idf, vis, 16)
    checks["routed prefill: not covered, logged as routed with VisionZip's own dominant patches"] = (
        log[0]["skip"] == "routed" and not log[0]["cover"] and log[0]["dominant"] == r0["own"])
    model_n, _, got_n = run(a.workdir, after, idf, [ROUTED, UNROUTED, UNROUTED], cover=0)
    checks["routed prefill: image tokens and answer equal VisionZip's (no cover)"] = (
        torch.equal(got["embeds"][0], got_n["embeds"][0]) and got["gen"][0] == got_n["gen"][0])
    checks["routed prefill: the LLM's image tokens = projector on VisionZip's output, written out"] = torch.allclose(
        got["embeds"][0][0, SYS_LEN:SYS_LEN + 1 + D + CTX], r0["own_feats"], atol=1e-5)
    ok_box = ok_rule = ok_feats = ok_log = ok_plain = True
    for j in (1, 2):
        rec, r = log[j], reference(model, got["images"][j], after, idf, vis, 16)
        ok_box &= rec["cover"] and "skip" not in rec and rec["grid_box"] == r["grid"] and rec["side"] == r["info"]["side"]
        ok_rule &= rec["dominant"] == r["dom"] and rec["baseline"] == r["own"] and len(rec["dominant"]) == D
        ok_log &= (rec["baseline_in_box"] == r["a_in"] and rec["in_box"] == r["n_c"] and rec["region"] == int(r["region"].sum())
                   and int(r["region"][torch.tensor(rec["dominant"])].sum()) == rec["in_box"]
                   and rec["changed"] == len(set(rec["dominant"]) - set(rec["baseline"]))
                   and rec["in_box"] == min(rec["baseline_in_box"] + 16, D))
        ok_feats &= torch.allclose(got["embeds"][j][0, SYS_LEN:SYS_LEN + 1 + D + CTX], r["feats"], atol=1e-5)
        ok_plain &= torch.allclose(r["plain_proj"], r["proj_from_layer"], atol=1e-6)
    checks["unrouted prefills: covered; box = zoom_box on the query cosine over plain LLaVA's image input embeddings"] = bool(ok_box)
    checks["the box's embeddings: projector on layer -2 = plain LLaVA's image input embeddings"] = bool(ok_plain)
    checks["unrouted prefills: dominant = VisionZip's in-box count + 16 inside the box by CLS attention, the rest outside"] = bool(ok_rule)
    checks["unrouted prefills: logged baseline / in-box counts, tokens changed"] = bool(ok_log)
    checks["unrouted prefills: the LLM's image tokens = projector on CLS + those patches + VisionZip's merge of the rest"] = bool(ok_feats)
    checks["every prefill gives the LLM 64 image tokens"] = [e.shape[1] for e in got["embeds"]] == [img_len] * 3
    checks["the cover changed the unrouted prefills' tokens (the check is not vacuous)"] = all(
        log[j]["changed"] > 0 and not torch.equal(got["embeds"][j], got_n["embeds"][j]) for j in (1, 2))
    checks["the log is one JSON-serialisable record per prefill, in order"] = (
        [r["index"] for r in log] == [0, 1, 2] and len(json.dumps(log)) > 0)

    # C. centred box (the localisation control): the located side, at the content band's centre
    model, lm, got = run(a.workdir, after, idf, [UNROUTED], centre=True)
    rec = model.vz_cover_log[0]
    r_loc = reference(model, got["images"][0], after, idf, vis, 16)
    r_ctr = reference(model, got["images"][0], after, idf, vis, 16, centre=True)
    r_lo, r_hi, c_lo, c_hi = _content_band(WIDTH, HEIGHT, 24)
    g = r_ctr["grid"]
    checks["centre: box = the centred box with the located side, at the content band's centre"] = (
        rec["grid_box"] == g and rec["side"] == r_loc["info"]["side"] and g != r_loc["grid"]
        and abs((g[0] + g[2]) / 2 - (r_lo + r_hi) / 2) <= 0.5 and abs((g[1] + g[3]) / 2 - (c_lo + c_hi) / 2) <= 0.5)
    checks["centre: dominant = the cover rule in the centred box; image tokens follow"] = (
        rec["dominant"] == r_ctr["dom"] and rec["in_box"] == r_ctr["n_c"]
        and torch.allclose(got["embeds"][0][0, SYS_LEN:SYS_LEN + 1 + D + CTX], r_ctr["feats"], atol=1e-5))

    # D. cover on every question: a routed question is covered too
    model, lm, got = run(a.workdir, after, idf, [ROUTED], on="all")
    rec, r = model.vz_cover_log[0], reference(model, got["images"][0], after, idf, vis, 16)
    checks["cover on all: a routed question is covered by the same rule"] = (
        rec["cover"] and "skip" not in rec and rec["grid_box"] == r["grid"] and rec["dominant"] == r["dom"])

    # E. cover 32: VisionZip's in-box count + 32, at most 53, inside
    model, lm, got = run(a.workdir, after, idf, [UNROUTED], cover=32)
    rec, r = model.vz_cover_log[0], reference(model, got["images"][0], after, idf, vis, 32)
    checks["cover 32: in-box = min(VisionZip's in-box count + 32, 53, box), dominant by the rule"] = (
        rec["in_box"] == r["n_c"] == min(r["a_in"] + 32, D, int(r["region"].sum())) and rec["dominant"] == r["dom"]
        and torch.allclose(got["embeds"][0][0, SYS_LEN:SYS_LEN + 1 + D + CTX], r["feats"], atol=1e-5))

    # E2. the no-box control (EXP-0028 v1.1): the 16 patches most similar to the query all kept, the other 37 by CLS
    # attention outside them; the box is still located and logged
    model, lm, got = run(a.workdir, after, idf, [UNROUTED], top=16)
    rec, r = model.vz_cover_log[0], reference(model, got["images"][0], after, idf, vis, 16, top=16)
    r_box = reference(model, got["images"][0], after, idf, vis, 16)
    rest = [p for p in range(576) if p not in set(r["top"])]
    others = sorted(torch.tensor(rest)[r["cls"][rest].topk(D - 16).indices].tolist())
    checks["no-box cover: the 16 most similar patches all kept, the other 37 = top CLS attention outside them"] = (
        rec["cover"] and rec["cover_top"] == r["top"] and len(r["top"]) == 16 and set(r["top"]) <= set(rec["dominant"])
        and sorted(set(rec["dominant"]) - set(r["top"])) == others and rec["dominant"] == r["dom"] and rec["in_box"] == 16
        and rec["baseline_in_box"] == len(set(r["own"]) & set(r["top"])) and rec["region"] == 16)
    checks["no-box cover: the located box still logged (= cover 16's); image tokens follow"] = (
        rec["grid_box"] == r_box["grid"] and torch.allclose(got["embeds"][0][0, SYS_LEN:SYS_LEN + 1 + D + CTX], r["feats"], atol=1e-5))

    # F. no visible weight: not covered, VisionZip's choice
    model, lm, got = run(a.workdir, after, idf, [UNROUTED], vis="vis_zero.npy")
    rec, r = model.vz_cover_log[0], reference(model, got["images"][0], after, idf, vis, 16)
    checks["no visible weight: not covered, VisionZip's own dominant patches, S = 0"] = (
        rec.get("skip") == "no visible weight" and not rec["cover"] and rec["dominant"] == r["own"] and rec["S"] == 0.0
        and torch.allclose(got["embeds"][0][0, SYS_LEN:SYS_LEN + 1 + D + CTX], r["own_feats"], atol=1e-5))

    # G. apply's own refusals
    def apply_refuses(**kw):
        mm = build_tiny_llava(a.workdir)
        if kw.pop("vz", True):
            mm = visionzip(mm, dominant=D + 1, contextual=CTX)
        try:
            visionzip_cover.apply(mm, kw.pop("cover", 16), lambda ids: [1.0] * len(ids), vis, **kw)
            return False
        except ValueError:
            return True
    checks["apply refuses cover 0, 'unrouted' without a router, an unknown 'on', top with centre, a model without VisionZip"] = (
        apply_refuses(cover=0, route=lambda ids: False) and apply_refuses() and apply_refuses(on="some", route=lambda ids: False)
        and apply_refuses(top=16, centre=True, route=lambda ids: False) and apply_refuses(vz=False, route=lambda ids: False))

    # H. arguments
    vz = ["--method", "visionzip", "--dominant", "54", "--contextual", "10"]
    ok = ["--vz-cover", "16", "--vz-visibility", "v", "--vz-log", "z"]
    checks["--vz-cover refused with --method fastv"] = refused(["--method", "fastv"] + ok)
    checks["--vz-cover refused without --vz-visibility"] = refused(vz + ["--vz-cover", "16", "--vz-log", "z"])
    checks["--vz-cover refused without --vz-log"] = refused(vz + ["--vz-cover", "16", "--vz-visibility", "v"])
    checks["--vz-log refused without --vz-cover"] = refused(vz + ["--vz-log", "z"])
    checks["--vz-visibility refused without --vz-cover"] = refused(vz + ["--vz-visibility", "v"])
    checks["--vz-cover-centre / --vz-cover-on all refused without --vz-cover"] = (
        refused(vz + ["--vz-cover-centre"]) and refused(vz + ["--vz-cover-on", "all"]))
    checks["--vz-cover -1 refused"] = refused(vz + ["--vz-cover", "-1", "--vz-visibility", "v", "--vz-log", "z"])
    checks["--vz-cover-top refused without --vz-cover, and with --vz-cover-centre"] = (
        refused(vz + ["--vz-cover-top", "16"]) and refused(vz + ok + ["--vz-cover-top", "16", "--vz-cover-centre"]))
    for extra in ([], ["--vz-cover-centre"], ["--vz-cover-on", "all"], ["--vz-cover-top", "16"]):
        checks[f"accepted: visionzip --vz-cover 16 {' '.join(extra)}"] = not refused(vz + ok + extra)
    checks["accepted: plain visionzip"] = not refused(vz)

    # I. configs
    C = exp0002.CONFIGS
    base = ["--method", "visionzip", "--dominant", "54", "--contextual", "10"]
    cov = ["--vz-cover", "16", "--vz-visibility", exp0002.VISIBILITY]
    want = {"visionzip64": base, "visionzip64_c16": base + cov, "visionzip64_c16ctr": base + cov + ["--vz-cover-centre"],
            "visionzip64_c32": base + ["--vz-cover", "32", "--vz-visibility", exp0002.VISIBILITY],
            "visionzip64_c16all": base + cov + ["--vz-cover-on", "all"], "visionzip64_t16": base + cov + ["--vz-cover-top", "16"]}
    for name, args in want.items():
        checks[f"exp0002 {name} as planned, in the tp env"] = C[name] == ("tp", args)
    checks["exp0002 VISIBILITY is EXP-0015's table under the repository, the one the FastV router uses"] = (
        exp0002.VISIBILITY == f"{exp0002.REPO}/code/output/EXP-0015/visibility/visibility.npy"     # REPO: two levels up
        and exp0002.VISIBILITY in C["fastv64k1_qroute_c16"][1]
        and os.path.exists("D:/6CJNM007/idea_rearch/code/output/EXP-0015/visibility/visibility.npy"))
    src = inspect.getsource(exp0002.run_one)
    checks["exp0002.run_one adds --vz-log {run_dir}/vz.json to the cover configs"] = (
        'if "--vz-cover" in method_args:' in src and '"--vz-log", f"{run_dir}/vz.json"' in src)
    checks["benches vqav2_dev2, gqa_val400, vqav2_test3, gqa_testdev registered with existing files"] = all(
        os.path.exists(exp0002.BENCHES[k]["question_file"]) for k in ("vqav2_dev2", "gqa_val400", "vqav2_test3", "gqa_testdev"))
    for name, ok_ in checks.items():
        print(("ok   " if ok_ else "FAIL ") + name)
    print(f"{sum(checks.values())}/{len(checks)} checks pass")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
