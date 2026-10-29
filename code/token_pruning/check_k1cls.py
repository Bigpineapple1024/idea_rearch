"""CPU check of IDEA-0007's CLS ranking at K = 1 (--fastv-score cls, --fastv-cls-unrouted) on a tiny random LLaVA
(smoke_test.build_tiny_llava).

  python check_k1cls.py --workdir D:/6CJNM007/tmp/exp0030/smoke

K = 1, keep 47 on a 640 x 480 image (padding never kept). Checks: the tower forward that returns the attention maps gives
plain LLaVA's features, and its score is the CLS attention summed over heads at layer -2, computed here independently;
with --fastv-score cls layer 0 sees all 576 patches and from layer 1 on the 47 non-padding patches with the largest
score stay; with --fastv-cls-unrouted a routed prefill keeps exactly what the K = 1 router keeps and answers the same,
an unrouted one keeps the 47 by the score; with cover 16 a routed prefill is the K = 1 router + cover 16's, an unrouted
one locates the same box and takes the CLS ranking's own in-box count plus 16 inside it by the score, the rest outside;
the per-prefill log; the 53 patches with the largest score are VisionZip's own dominant patches (visionzip.visionzip on
the same image, imported last because it patches CLIP's classes); argument and config checks. The FastV paths without
the new options are checked against the committed code by check_zoom.py, run from this folder.
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
from check_object_cos import BLOCKS, HEIGHT, WIDTH, Pieces, refused  # noqa: E402
from fastv_llava import _padding_mask  # noqa: E402
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

PLAIN_FORWARD = CLIPVisionTower.forward          # plain LLaVA's, saved before anything patches the class
KEEP = 47


def under(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


def run(workdir, after, idf, decisions, score="query_vis", cls_unrouted=False, cover=0):
    """One model, one prefill per decision (routed True / False; ignored for score cls), images seeded 10, 11, ..."""
    qv = score == "query_vis"
    m = argparse.Namespace(method="fastv", fastv_k=1, fastv_keep=KEEP, fastv_schedule=None,
                           fastv_score_prior=f"{workdir}/prior_k1.npy" if qv else None, fastv_stratify=BLOCKS if qv else None,
                           text_rows_log=None, fastv_score=score, fastv_exclude_padding=True, fastv_mix_question=None,
                           fastv_contrast_ids=None, fastv_idf=idf if qv else None,
                           fastv_visibility=f"{workdir}/vis_k1.npy" if qv else None, fastv_route="yesno_howmany" if qv else None,
                           fastv_zoom=0, fastv_zoom_pick="grid", fastv_zoom_on="unrouted", fastv_zoom_box="8,8,16",
                           fastv_zoom_smin=0.0, fastv_cover=cover, fastv_zoom_source="crop", fastv_zoom_centre=False,
                           fastv_cover_top=0, fastv_cls_unrouted=cls_unrouted)
    model = run_eval.apply_method(build_tiny_llava(workdir), m, Pieces(after.tolist()))
    lm = model.get_model()
    flag = [None]
    if qv:
        lm.fastv_route = lambda ids: flag[0]
    got = {"embeds": [], "len0": [], "len1": [], "kept": [], "scores": [], "gen": [], "images": []}

    def pre_lm(module, args, kwargs):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1:
            got["embeds"].append(e.detach().clone())

    def pre0(module, args, kwargs):
        if args[0].shape[1] > 1:
            got["len0"].append(args[0].shape[1])

    def pre1(module, args, kwargs):
        if args[0].shape[1] > 1:
            got["len1"].append(args[0].shape[1])

    def post_lm(module, args, kwargs, output):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1:
            got["kept"].append(module.fastv_kept_image_positions.tolist())
            got["scores"].append(module.fastv_last_scores.detach().clone().float())

    lm.register_forward_pre_hook(pre_lm, with_kwargs=True)
    lm.layers[0].register_forward_pre_hook(pre0, with_kwargs=True)
    lm.layers[1].register_forward_pre_hook(pre1, with_kwargs=True)
    lm.register_forward_hook(post_lm, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    with torch.no_grad():
        for i, d in enumerate(decisions):
            flag[0] = d
            images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))
            got["images"].append(images)
            out = model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
            got["gen"].append(out.tolist())
    return model, lm, got


def cls_score(model, images):
    """Independently: the CLS attention summed over heads at the vision tower's layer -2, and plain LLaVA's features."""
    tower = model.get_model().get_vision_tower()
    with torch.no_grad():
        outs = tower.vision_tower(images.to(device=tower.device, dtype=tower.dtype), output_hidden_states=True, output_attentions=True)
        return outs.attentions[-2][0, :, 0, 1:].sum(0).float(), PLAIN_FORWARD(tower, images)


def top_allowed(score, allowed, k):
    return sorted(allowed.nonzero().squeeze(1)[score[allowed].topk(k).indices].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    rng = np.random.default_rng(0)
    idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
    np.save(f"{a.workdir}/prior_k1.npy", (torch.rand(576, generator=torch.Generator().manual_seed(3)) + 0.5).numpy())
    np.save(f"{a.workdir}/vis_k1.npy", rng.uniform(0, 1, 1000).astype(np.float32))
    allowed = ~_padding_mask(WIDTH, HEIGHT, 24)
    checks = {}
    checks["this check imports the new fastv_llava, run_eval and exp0002"] = all(
        under(mod.__file__, HERE) for mod in (fastv_llava, run_eval, exp0002))
    checks["the test image has padding (640 x 480)"] = int((~allowed).sum()) > 0

    # A. the tower forward with the attention maps: plain LLaVA's features, VisionZip's score
    model, lm, got = run(a.workdir, after, idf, [None, None], score="cls")
    tower = lm.get_vision_tower()
    ref0, plain0 = cls_score(model, got["images"][0])
    ref1, _ = cls_score(model, got["images"][1])
    with torch.no_grad():
        feats = tower(got["images"][0])
    checks["the tower forward returns plain LLaVA's features"] = torch.equal(feats, plain0)
    checks["its score is the CLS attention summed over heads at layer -2"] = torch.equal(tower.fastv_cls_score[0].float(), ref0)
    e = got["embeds"][0][0]
    checks["the LLM's image input embeddings are the projector on plain LLaVA's features"] = torch.equal(
        e[SYS_LEN:SYS_LEN + 576], model.get_model().mm_projector(plain0)[0])

    # B. --fastv-score cls: all 576 at layer 0, then the 47 non-padding patches with the largest score
    log = lm.fastv_cls_log
    checks["cls: layer 0 sees all 576 patches, layer 1 on 47, in both prefills"] = (
        got["len0"] == [SYS_LEN + 576 + TEXT_LEN] * 2 and got["len1"] == [SYS_LEN + KEEP + TEXT_LEN] * 2)
    checks["cls: kept = the 47 non-padding patches with the largest CLS score"] = (
        got["kept"][0] == top_allowed(ref0, allowed, KEEP) and got["kept"][1] == top_allowed(ref1, allowed, KEEP))
    checks["cls: the stage ranked by that score"] = torch.equal(got["scores"][0], ref0) and torch.equal(got["scores"][1], ref1)
    checks["cls: one log record per prefill; used; own = kept; top53 = the 53 patches with the largest score"] = (
        len(log) == 2 and all(r["used"] for r in log) and [r["own"] for r in log] == got["kept"]
        and log[0]["top53"] == sorted(ref0.topk(53).indices.tolist()) and log[1]["top53"] == sorted(ref1.topk(53).indices.tolist()))
    checks["cls: kept patches never lie wholly in the padding"] = all(bool(allowed[k].all()) for k in got["kept"])
    top53 = log[0]["top53"]

    # C. --fastv-cls-unrouted: routed = the K = 1 router; unrouted = the CLS ranking
    _, lm_r, base = run(a.workdir, after, idf, [True, False])
    _, lm_c, got = run(a.workdir, after, idf, [True, False], cls_unrouted=True)
    checks["cls_unrouted: the routed prefill keeps what the K = 1 router keeps and generates the same"] = (
        got["kept"][0] == base["kept"][0] and got["gen"][0] == base["gen"][0] and torch.equal(got["embeds"][0], base["embeds"][0]))
    checks["cls_unrouted: the unrouted prefill keeps the 47 non-padding patches with the largest CLS score"] = (
        got["kept"][1] == top_allowed(ref1, allowed, KEEP) and torch.equal(got["scores"][1], ref1))
    checks["cls_unrouted: the K = 1 router's unrouted prefill keeps something else (the test is not vacuous)"] = (
        base["kept"][1] != got["kept"][1])
    checks["cls_unrouted: lengths 576 at layer 0, 47 after, in both prefills"] = (
        got["len0"] == [SYS_LEN + 576 + TEXT_LEN] * 2 and got["len1"] == [SYS_LEN + KEEP + TEXT_LEN] * 2)
    lg = lm_c.fastv_cls_log
    checks["cls_unrouted: the log marks only the unrouted prefill as ranked by the score"] = (
        len(lg) == 2 and not lg[0]["used"] and "own" not in lg[0] and lg[1]["used"] and lg[1]["own"] == got["kept"][1])
    checks["cls_unrouted: routed decisions recorded as before"] = lm_c.fastv_routed is False and lm_r.fastv_routed is False

    # D. with cover 16: routed = the K = 1 router + cover 16; unrouted = the same box, the cover rule by the CLS score
    _, lm_b, base = run(a.workdir, after, idf, [True, False], cover=16)
    _, lm_x, got = run(a.workdir, after, idf, [True, False], cls_unrouted=True, cover=16)
    checks["cover: the routed prefill keeps what the K = 1 router + cover 16 keeps and generates the same"] = (
        got["kept"][0] == base["kept"][0] and got["gen"][0] == base["gen"][0])
    rb, rx = lm_b.fastv_zoom_log[1], lm_x.fastv_zoom_log[1]
    checks["cover: the unrouted prefill locates the same box"] = rx["cover"] and rb["cover"] and rx["grid_box"] == rb["grid_box"]
    r0, c0, r1, c1 = rx["grid_box"]
    box = torch.zeros(24, 24, dtype=torch.bool)
    box[r0:r1 + 1, c0:c1 + 1] = True
    box = box.ravel()
    own = torch.tensor(top_allowed(ref1, allowed, KEEP))
    a_in = int(box[own].sum())
    inside, outside = (allowed & box).nonzero().squeeze(1), (allowed & ~box).nonzero().squeeze(1)
    n_c = min(a_in + 16, KEEP, inside.numel())
    want = sorted(inside[ref1[inside].topk(n_c).indices].tolist() + outside[ref1[outside].topk(KEEP - n_c).indices].tolist())
    checks["cover: kept = the CLS ranking's own in-box count + 16 inside the box by the score, the rest outside"] = (
        got["kept"][1] == want and rx["baseline_in_box"] == a_in and rx["in_box"] == n_c and n_c > a_in)
    checks["cover: the log's own choice is the CLS ranking's, before the cover"] = (
        lm_x.fastv_cls_log[1]["used"] and lm_x.fastv_cls_log[1]["own"] == own.tolist())
    checks["cover: the K = 1 router + cover 16 keeps something else on the unrouted prefill (not vacuous)"] = base["kept"][1] != want

    # E. arguments and configs
    k1 = ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "47"]
    ok_cls = k1 + ["--fastv-score", "cls", "--fastv-exclude-padding", "--fastv-cls-log", "x.json"]
    checks["arguments: cls with its log is accepted"] = not refused(ok_cls)
    checks["arguments: cls without the log is refused"] = refused(ok_cls[:-2])
    checks["arguments: cls with a position prior, blocks or cover is refused"] = all(refused(ok_cls + extra) for extra in (
        ["--fastv-score-prior", "p.npy"], ["--fastv-stratify", "6"], ["--fastv-cover", "16"]))
    router = exp0002.CONFIGS["fastv64k1_qroute"][1]
    checks["arguments: cls-unrouted on the K = 1 router (with the log) is accepted"] = not refused(
        router + ["--fastv-cls-unrouted", "--fastv-cls-log", "x.json"])
    checks["arguments: cls-unrouted without the router, or without query_vis, is refused"] = (
        refused([x for x in router if x not in ("--fastv-route", "yesno_howmany")] + ["--fastv-cls-unrouted", "--fastv-cls-log", "x.json"])
        and refused(k1 + ["--fastv-cls-unrouted", "--fastv-cls-log", "x.json"]))
    checks["arguments: the log without cls is refused"] = refused(router + ["--fastv-cls-log", "x.json"])
    C = exp0002.CONFIGS
    checks["configs: the three IDEA-0007 configs and their _kl variants"] = (
        C["fastv64k1_cls"][1] == k1 + ["--fastv-score", "cls", "--fastv-exclude-padding"]
        and C["fastv64k1_qroute_cls"][1] == C["fastv64k1_qroute"][1] + ["--fastv-cls-unrouted"]
        and C["fastv64k1_qroute_cls_c16"][1] == C["fastv64k1_qroute_c16"][1] + ["--fastv-cls-unrouted"]
        and all(C[f"{n}_kl"] == C[n] for n in ("fastv64k1_cls", "fastv64k1_qroute_cls", "fastv64k1_qroute_cls_c16")))
    src = inspect.getsource(exp0002.run_one)
    checks["configs: run_one adds the CLS log to the CLS configs"] = '"--fastv-cls-log", f"{run_dir}/cls.json"' in src
    checks["configs: each IDEA-0007 config (with the logs run_one adds) parses"] = all(not refused(
        C[n][1] + ["--fastv-cls-log", "x.json", "--kept-token-log", "k.json", "--answer-logit-log", "l.json"]
        + (["--fastv-zoom-log", "z.json"] if "--fastv-cover" in C[n][1] else []))
        for n in ("fastv64k1_cls", "fastv64k1_qroute_cls", "fastv64k1_qroute_cls_c16"))
    checks["configs: the K = 1 budget is 63.53 tokens per layer, within 64"] = abs((576 + 31 * KEEP) / 32 - 63.53125) < 1e-9

    # F. the 53 patches with the largest score are VisionZip's dominant patches (visionzip patches CLIP's classes, so last)
    root = os.environ["VISIONZIP_ROOT"]
    sys.path.insert(0, root)
    from visionzip import visionzip
    vz = visionzip(build_tiny_llava(a.workdir), dominant=54, contextual=10)
    image0 = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10))      # run B's first image
    with torch.no_grad():
        _, idx = vz.get_model().get_vision_tower()(image0)
    checks["VisionZip's own dominant patches = the 53 with the largest CLS score"] = sorted((idx[0, 1:] - 1).tolist()) == top53

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_k1cls.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
