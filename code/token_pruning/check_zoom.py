"""CPU check of IDEA-0004's --fastv-zoom / --fastv-cover on a tiny random LLaVA (smoke_test.build_tiny_llava).

  python check_zoom.py --workdir D:/6CJNM007/tmp/tp_smoke

Router configuration of EXP-0020 (query_vis, K = 2, keep 30, 6 x 6 blocks, prior, padding excluded) on a 640 x 480 image.
Checks: without zoom and cover the new code keeps exactly what the committed code keeps (kept_dump.py, two processes);
with zoom on the unrouted questions a routed prefill is untouched, an unrouted one carries n more tokens at layer 0 and
the same number of tokens as before after the stage (keep visual tokens), the inserted tokens equal the vision tower +
projector on the logged crop at the picked positions (lattice), or its block means (pool), or block means of the image's
own tokens (source full), and the box is the one the query (read from the text after the zoom tokens) points at, or the
centred box with the same side; with zoom on every question the routed query_vis cosine still reads the question's
text; with cover, the baseline's own in-box count plus cover is taken inside the box by its rule and the rest outside;
the box stays square and inside the image, never below the native-resolution side, and the picks skip a crop's padding;
the layer-token and kept-token logs count the zoom tokens; argument, config and bench checks.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
import exp0002  # noqa: E402
import fastv_llava  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import BLOCKS, HEIGHT, KEEP, WIDTH, Pieces, refused  # noqa: E402
from fastv_llava import _content_band, _padding_mask, _stratified_top, zoom_box, zoom_pick, zoom_pool  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

PY = sys.executable
COMMITTED = "D:/6CJNM007/idea_rearch/code/token_pruning"


def under(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


def run(workdir, img_path, after, idf, decisions, n=0, pick="grid", on="unrouted", cover=0, source="crop", centre=False,
        kept_log=False):
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=f"{workdir}/prior_zoom.npy",
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score="query_vis", fastv_exclude_padding=True,
                           fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=idf, fastv_visibility=f"{workdir}/vis_zoom.npy",
                           fastv_route="yesno_howmany", fastv_zoom=n, fastv_zoom_pick=pick, fastv_zoom_on=on, fastv_zoom_box="8,8,16",
                           fastv_zoom_smin=0.0, fastv_cover=cover, fastv_zoom_source=source, fastv_zoom_centre=centre)
    model = run_eval.apply_method(build_tiny_llava(workdir), m, Pieces(after.tolist()))
    lm = model.get_model()
    flag = [None]
    lm.fastv_route = lambda ids: flag[0]
    lm.fastv_image_processor = model.get_vision_tower().image_processor
    lm.fastv_zoom_paths = [img_path] * len(decisions)
    got = {"embeds": [], "len0": [], "len2": [], "kept": [], "obj_cos": [], "n_q": [], "scores": [], "cover_count": []}

    def pre_lm(module, args, kwargs):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1:
            got["embeds"].append(e.detach().clone())

    def pre0(module, args, kwargs):
        if args[0].shape[1] > 1:
            got["len0"].append(args[0].shape[1])

    def pre2(module, args, kwargs):
        if args[0].shape[1] > 1:
            got["len2"].append(args[0].shape[1])

    def post_lm(module, args, kwargs, output):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1:
            got["kept"].append(module.fastv_kept_image_positions.tolist())
            got["obj_cos"].append(None if not flag[0] else module.fastv_object_cos.detach().clone())
            got["n_q"].append(module.fastv_mix_count)
            got["scores"].append(module.fastv_last_scores.detach().clone())
            got["cover_count"].append(module.fastv_cover_count)
            module.fastv_cover_count = None

    lm.register_forward_pre_hook(pre_lm, with_kwargs=True)
    lm.layers[0].register_forward_pre_hook(pre0, with_kwargs=True)
    lm.layers[2].register_forward_pre_hook(pre2, with_kwargs=True)
    lm.register_forward_hook(post_lm, with_kwargs=True)
    if kept_log:                                          # run_eval's own kept-token hook
        for lst in (run_eval._KEPT, run_eval._KEPT_QUESTION, run_eval._ROUTED, run_eval._KEPT_ZOOM):
            lst.clear()
        run_eval._KEPT_N_IMG.clear()
        lm.register_forward_hook(run_eval._record_kept, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    with torch.no_grad():
        for i, d in enumerate(decisions):
            flag[0] = d
            images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))
            model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
    return model, lm, got


def weights(lm, after):
    idf = torch.tensor(lm.fastv_token_weights(after.tolist()), dtype=torch.float32)
    return idf * lm.fastv_visibility[after].float(), idf


def query_cos(lm, after, e, start, n_img, n):
    """The IDF x visibility query read from the text after the image and the n zoom tokens, and every image token's cosine."""
    w, _ = weights(lm, after)
    q = (w[:, None] * e[start + n_img + n:start + n_img + n + w.numel()]).sum(0, keepdim=True) / w.sum()
    return torch.nn.functional.cosine_similarity(e[start:start + n_img], q, dim=-1)


def block_means(feats, rows, cols, g):
    """Reference for zoom_pool: means over np.array_split groups of the given rows and columns."""
    grid = feats.reshape(24, 24, -1)
    return torch.stack([grid[list(rg)][:, list(cg)].reshape(-1, grid.shape[-1]).mean(0)
                        for rg in np.array_split(np.array(rows), g) for cg in np.array_split(np.array(cols), g)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    rng = np.random.default_rng(0)
    idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
    np.save(f"{a.workdir}/prior_zoom.npy", (torch.rand(576, generator=torch.Generator().manual_seed(3)) + 0.5).numpy())
    np.save(f"{a.workdir}/vis_zoom.npy", rng.uniform(0, 1, 1000).astype(np.float32))
    from PIL import Image
    img_path = f"{a.workdir}/zoom_640x480.png"
    Image.fromarray(np.random.default_rng(5).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)).save(img_path)
    checks = {}

    # A. nothing changes without zoom and cover: committed code vs new code, separate processes
    outs = {}
    for tag, code in (("committed", COMMITTED), ("new", HERE)):
        r = subprocess.run([PY, "-B", f"{HERE}/kept_dump.py", "--code-dir", code, "--workdir", a.workdir, "--out", f"{a.workdir}/kept_{tag}.json"],
                           capture_output=True, text=True)
        outs[tag] = json.load(open(f"{a.workdir}/kept_{tag}.json")) if r.returncode == 0 else {"error": r.stderr[-800:]}
    checks["without zoom and cover: kept sets and answers equal the committed code's"] = all(
        k in outs[t] for t in outs for k in ("kept", "gen")) and all(outs["committed"][k] == outs["new"][k] for k in ("kept", "gen"))
    checks["the regression ran the committed modules against the new ones"] = (
        all(under(f, COMMITTED) for f in outs["committed"].get("modules", {"none": ""}).values())
        and all(under(f, HERE) for f in outs["new"].get("modules", {"none": ""}).values()))
    checks["this check imports the new fastv_llava, run_eval and exp0002"] = all(
        under(m.__file__, HERE) for m in (fastv_llava, run_eval, exp0002))

    start, n_img, n = SYS_LEN, 576, 16
    base_len = SYS_LEN + n_img + TEXT_LEN
    allowed = ~_padding_mask(WIDTH, HEIGHT, 24)
    from llava.mm_utils import process_images

    def crop_feats(model, lm, rec):
        with Image.open(img_path) as im:
            crop = im.convert("RGB").crop(tuple(rec["crop_box"]))
        with torch.no_grad():
            return model.encode_images(process_images([crop], lm.fastv_image_processor, model.config))[0].float(), crop

    # B. zoom (lattice) on the unrouted questions; run_eval's kept-token log
    model, lm, got = run(a.workdir, img_path, after, idf, [True, False, False], n=16, pick="grid", kept_log=True)
    log = lm.fastv_zoom_log
    checks["routed prefill: no zoom, logged as routed"] = log[0].get("skip") == "routed" and got["len0"][0] == base_len
    checks["unrouted prefills: zoomed, n more tokens at layer 0"] = all(r["zoom"] for r in log[1:]) and got["len0"][1:] == [base_len + n] * 2
    checks["after the stage: keep visual tokens in every prefill"] = got["len2"] == [SYS_LEN + KEEP + TEXT_LEN] * 3
    checks["unrouted prefills keep keep - n image tokens"] = [len(k) for k in got["kept"]] == [KEEP, KEEP - n, KEEP - n]
    ok_tokens, ok_box, square = True, True, True
    for j in (1, 2):
        rec, e = log[j], got["embeds"][j][0].float()
        feats, crop = crop_feats(model, lm, rec)
        ok_tokens &= torch.allclose(e[start + n_img:start + n_img + n], feats[rec["picked"]], atol=1e-5)
        grid, _, _ = zoom_box(query_cos(lm, after, e, start, n_img, n), allowed, 24, WIDTH, HEIGHT, 8, 8, 16)
        ok_box &= list(grid) == rec["grid_box"]
        square &= abs(crop.width - crop.height) <= 1 and rec["crop_padding_tokens"] == 0 and rec["picked_on_padding"] == 0
    checks["inserted tokens = vision tower + projector on the logged crop at the picked positions"] = bool(ok_tokens)
    checks["the box is the one the query (text after the zoom tokens) points at"] = bool(ok_box)
    checks["the crop is square (within 1 px), carries no padding, no pick on padding"] = bool(square)
    run_eval._write_kept(f"{a.workdir}/kept_log.json", KEEP)
    kl = json.load(open(f"{a.workdir}/kept_log.json"))
    checks["kept-token log: zoom_tokens per prefill = [0, n, n], kept lists keep, keep - n, keep - n"] = (
        kl.get("zoom_tokens") == [0, n, n] and [len(k) for k in kl["kept"]] == [KEEP, KEEP - n, KEEP - n])

    # C. zoom on every question: the routed query_vis cosine still reads the question's text
    model, lm, got = run(a.workdir, img_path, after, idf, [True], n=16, pick="query", on="all")
    e = got["embeds"][0][0].float()
    w, idf_t = weights(lm, after)
    checks["zoom on all: routed prefill zoomed, its query_vis cosine reads the text after the zoom tokens"] = (
        lm.fastv_zoom_log[0]["zoom"] and torch.allclose(got["obj_cos"][0].float(), query_cos(lm, after, e, start, n_img, n), atol=1e-5))
    checks["zoom on all: n_q = round((keep - n) * S)"] = got["n_q"][0] == round((KEEP - n) * float(w.sum() / idf_t.sum()))
    checks["zoom on all: layer-0 and post-stage lengths"] = got["len0"] == [base_len + n] and got["len2"] == [SYS_LEN + KEEP + TEXT_LEN]

    # D. cover: the baseline's own in-box count plus 8 inside the box by its rule, the rest outside
    model, lm, got = run(a.workdir, img_path, after, idf, [False], cover=8)
    rec = lm.fastv_zoom_log[0]
    r0, c0, r1, c1 = rec["grid_box"]
    box = torch.zeros(24, 24, dtype=torch.bool)
    box[r0:r1 + 1, c0:c1 + 1] = True
    box = box.ravel()
    score = got["scores"][0].float() / torch.from_numpy(np.load(f"{a.workdir}/prior_zoom.npy")).float()
    pos = allowed.nonzero().squeeze(1)
    base = pos[_stratified_top(score[pos], KEEP, pos, 24, BLOCKS)]
    a_in = int(box[base].sum())
    inside, outside = pos[box[pos]], pos[~box[pos]]
    n_c = min(a_in + 8, KEEP, inside.numel())
    first = inside[_stratified_top(score[inside], n_c, inside, 24, BLOCKS)]
    filled = {}
    for cell in ((first // 24) * BLOCKS // 24 * BLOCKS + (first % 24) * BLOCKS // 24).tolist():
        filled[cell] = filled.get(cell, 0) + 1
    second = outside[_stratified_top(score[outside], KEEP - n_c, outside, 24, BLOCKS, filled)]
    expect = sorted(torch.cat((first, second)).tolist())
    kept0 = torch.tensor(got["kept"][0])
    checks["cover: kept set = baseline's in-box count + 8 inside by its rule, the rest outside"] = got["kept"][0] == expect
    checks["cover: in-box count = baseline's + 8, logged with the tokens changed"] = (
        int(box[kept0].sum()) == n_c and rec["baseline_in_box"] == a_in and rec["in_box"] == n_c
        and rec["changed"] == int((~torch.isin(kept0, base)).sum()) and rec["changed"] >= min(8, KEEP - a_in))
    checks["cover: logged, keep tokens, no zoom tokens"] = (rec["cover"] and not rec["zoom"] and got["cover_count"][0] == n_c
                                                           and len(got["kept"][0]) == KEEP and got["len0"] == [base_len])

    # G. pooled crop: block means of the crop's tokens over 4 x 4 blocks
    model, lm, got = run(a.workdir, img_path, after, idf, [False], n=16, pick="pool")
    rec, e = lm.fastv_zoom_log[0], got["embeds"][0][0].float()
    feats, crop = crop_feats(model, lm, rec)
    checks["pool: inserted tokens = 6 x 6 block means of the crop's 576 tokens"] = (
        rec["zoom"] and "picked" not in rec and torch.allclose(e[start + n_img:start + n_img + n], block_means(feats, range(24), range(24), 4), atol=1e-5))

    # H. source full: block means of the image's own tokens over its content rows (no crop)
    model, lm, got = run(a.workdir, img_path, after, idf, [False], n=16, pick="pool", source="full")
    rec, e = lm.fastv_zoom_log[0], got["embeds"][0][0].float()
    rows = [r for r in range(24) if not bool(_padding_mask(WIDTH, HEIGHT, 24).reshape(24, 24)[r].all())]
    checks["full: inserted tokens = block means of the image's own tokens over its content rows; no crop"] = (
        rec["zoom"] and rec.get("source") == "full" and "crop_box" not in rec and rows == list(range(3, 21))
        and torch.allclose(e[start + n_img:start + n_img + n], block_means(e[start:start + n_img], rows, range(24), 4), atol=1e-5))

    # I. centre: same side as located, centred on the image's content band
    model, lm, got = run(a.workdir, img_path, after, idf, [False], n=16, pick="grid", centre=True)
    rec, e = lm.fastv_zoom_log[0], got["embeds"][0][0].float()
    cos = query_cos(lm, after, e, start, n_img, n)
    g_loc, _, i_loc = zoom_box(cos, allowed, 24, WIDTH, HEIGHT, 8, 8, 16)
    g_ctr, _, i_ctr = zoom_box(cos, allowed, 24, WIDTH, HEIGHT, 8, 8, 16, centre=True)
    r_lo, r_hi, c_lo, c_hi = _content_band(WIDTH, HEIGHT, 24)
    mid_r, mid_c = (g_ctr[0] + g_ctr[2]) / 2, (g_ctr[1] + g_ctr[3]) / 2
    checks["centre: logged box = centred box, same side as the located one, centred on the content band"] = (
        list(g_ctr) == rec["grid_box"] and i_ctr["side"] == i_loc["side"] and abs(mid_r - (r_lo + r_hi) / 2) <= 0.5
        and abs(mid_c - (c_lo + c_hi) / 2) <= 0.5)

    # E. the box and the picks on other image shapes
    for wd, ht in ((640, 480), (480, 640), (500, 500), (640, 427)):
        al = ~_padding_mask(wd, ht, 24)
        rl, rh, cl, ch = _content_band(wd, ht, 24)
        floor = min(max(8, math.ceil(24 * 336 / max(wd, ht))), 16, rh - rl + 1, ch - cl + 1)
        ok = True
        for peak in al.nonzero().squeeze(1).tolist()[::7]:            # single peaks over the image, edges included
            cos = torch.full((576,), -1.0)
            cos[peak] = 1.0
            (r0, c0, r1, c1), px, info = zoom_box(cos, al, 24, wd, ht, 8, 8, 16)
            ok &= (r1 - r0 == c1 - c0 and r1 - r0 + 1 >= floor and r1 - r0 + 1 <= 16 and rl <= r0 and r1 <= rh and cl <= c0 and c1 <= ch
                   and 0 <= px[0] < px[2] <= wd and 0 <= px[1] < px[3] <= ht and abs((px[2] - px[0]) - (px[3] - px[1])) <= 1.0
                   and info["top1_inside"] == (r0 <= peak // 24 <= r1 and c0 <= peak % 24 <= c1))
        checks[f"box {wd}x{ht}: square, side in [native floor, 16], inside the content band, square pixel box"] = bool(ok)
    wd, ht = 2000, 300                                                  # band (4 rows) narrower than min_side: padded crop
    al = ~_padding_mask(wd, ht, 24)
    cos = torch.full((576,), -1.0)
    cos[int(al.nonzero()[len(al.nonzero()) // 2])] = 1.0
    _, px, info = zoom_box(cos, al, 24, wd, ht, 8, 8, 16)
    cw, ch_ = round(px[2]) - round(px[0]), round(px[3]) - round(px[1])
    cpad = _padding_mask(cw, ch_, 24)
    gp = zoom_pick(16, 24, "grid", pad=cpad)
    qp = zoom_pick(16, 24, "query", torch.rand(576, generator=torch.Generator().manual_seed(4)), pad=cpad)
    pooled = zoom_pool(torch.randn(576, 8), 16, 24, cpad)
    checks["2000x300: band < min_side, crop padded; grid and query picks avoid its padding; pool finite"] = (
        info["band"] < 8 and bool(cpad.any()) and not bool(cpad[gp].any()) and not bool(cpad[qp].any())
        and len(set(gp.tolist())) == 16 and bool(torch.isfinite(pooled).all()))
    gp = zoom_pick(16, 24, "grid")
    checks["grid pick on a square crop: 16 lattice positions on rows {3,9,15,21}"] = len(set(gp.tolist())) == 16 and sorted({int(x) // 24 for x in gp}) == [3, 9, 15, 21]
    qp = zoom_pick(16, 24, "query", torch.rand(576, generator=torch.Generator().manual_seed(2)))
    checks["query pick: one token in each of the 4 x 4 blocks"] = len({(int(x) // 24 * 4 // 24, int(x) % 24 * 4 // 24) for x in qp}) == 16
    try:
        zoom_pick(10, 24, "grid")
        checks["zoom_pick refuses a non-square n"] = False
    except ValueError:
        checks["zoom_pick refuses a non-square n"] = True

    # J. the layer-token log counts the zoom tokens as visual
    run_eval._LAYER_TOKENS[:] = [{i: v for i, v in enumerate([647, 647, 85, 85])}, {i: v for i, v in enumerate([631, 631, 85, 85])}]
    run_eval._LAYER_EXTRA[:] = [16, 0]
    run_eval._ZOOM_MODEL[:] = [lm]
    run_eval._write_layer_tokens(f"{a.workdir}/layer_log.json", 4)
    ll = json.load(open(f"{a.workdir}/layer_log.json"))
    checks["layer-token log: zoomed prefill 592, 592, 30, 30 visual tokens; unzoomed 576, 576, 30, 30"] = (
        ll["mean_visual_per_layer"] == [584.0, 584.0, 30.0, 30.0] and ll["zoom_tokens"] == [16, 0])
    run_eval._LAYER_TOKENS.clear()
    run_eval._LAYER_EXTRA.clear()
    run_eval._ZOOM_MODEL.clear()

    # F. arguments, configs, benches
    base = ["--method", "fastv", "--fastv-score", "query_vis", "--fastv-visibility", "x", "--fastv-stratify", "6",
            "--fastv-score-prior", "x", "--fastv-exclude-padding"]
    r = ["--fastv-route", "yesno_howmany"]
    checks["--fastv-zoom refused without query_vis"] = refused(["--method", "fastv", "--fastv-zoom", "16", "--fastv-zoom-log", "z"])
    checks["--fastv-zoom-on unrouted refused without --fastv-route"] = refused(base + ["--fastv-zoom", "16", "--fastv-zoom-log", "z"])
    checks["--fastv-zoom 10 refused (not a square)"] = refused(base + r + ["--fastv-zoom", "10", "--fastv-zoom-log", "z"])
    checks["--fastv-zoom without --fastv-zoom-log refused"] = refused(base + r + ["--fastv-zoom", "16"])
    checks["--fastv-cover with --fastv-zoom-on all refused"] = refused(base + r + ["--fastv-cover", "8", "--fastv-zoom-on", "all", "--fastv-zoom-log", "z"])
    checks["--fastv-zoom-source full without pool refused"] = refused(base + r + ["--fastv-zoom", "16", "--fastv-zoom-source", "full", "--fastv-zoom-log", "z"])
    checks["--fastv-zoom-centre without zoom refused"] = refused(base + r + ["--fastv-zoom-centre"])
    for extra in (["--fastv-zoom", "16"], ["--fastv-cover", "8"], ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool"],
                  ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool", "--fastv-zoom-source", "full"],
                  ["--fastv-zoom", "16", "--fastv-zoom-centre"], ["--fastv-cover", "16", "--fastv-zoom-smin", "0"]):
        checks[f"accepted with the router: {' '.join(extra)}"] = not refused(base + r + extra + ["--fastv-zoom-log", "z"])
    C = exp0002.CONFIGS
    for b in (64, 128, 192):
        q = C[f"fastv{b}_qroute"][1]
        checks[f"exp0002 fastv{b}_qroute_z16g_kl / _c8_kl = the router + zoom / cover"] = (
            C[f"fastv{b}_qroute_z16g_kl"][1] == q + ["--fastv-zoom", "16", "--fastv-zoom-pick", "grid"]
            and C[f"fastv{b}_qroute_c8_kl"][1] == q + ["--fastv-cover", "8"])
    q, k1 = C["fastv64_qroute"][1], C["fastv64k1_qroute"][1]
    diff = [(x, y) for x, y in zip(q, k1) if x != y]
    all0 = ["--fastv-zoom-on", "all", "--fastv-zoom-smin", "0"]
    checks["exp0002 fastv64k1_qroute = the router at K 1, keep 47, the K = 1 prior; nothing else differs"] = (
        len(q) == len(k1) and sorted(diff) == sorted([("2", "1"), ("30", "47"), (exp0002.SCORE_PRIOR, exp0002.PRIOR_K1)])
        and k1[k1.index("--fastv-k") + 1] == "1" and k1[k1.index("--fastv-keep") + 1] == "47")
    want = {"fastv64_qroute_z16g_all0": q + ["--fastv-zoom", "16", "--fastv-zoom-pick", "grid"] + all0,
            "fastv64_qroute_z16p_all0": q + ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool"] + all0,
            "fastv64_qroute_p16full_all0": q + ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool", "--fastv-zoom-source", "full"] + all0,
            "fastv64_qroute_z16pctr_all0": q + ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool", "--fastv-zoom-centre"] + all0,
            "fastv64_qroute_c16": q + ["--fastv-cover", "16", "--fastv-zoom-smin", "0"],
            "fastv64k1_qroute_z16p_all0": k1 + ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool"] + all0}
    for name, args in want.items():
        checks[f"exp0002 {name} (and _kl) as planned"] = C[name][1] == args and C[f"{name}_kl"][1] == args
    checks["exp0002 fastv64k1_scores = plain FastV at K 1, keep 47"] = C["fastv64k1_scores"][1] == ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "47"]
    checks["budgets: K1 keep 47 = 63.5, + 16 zoom tokens = 64.0; K2 keep 30 + 16 zoom tokens = 65.1 per layer"] = (
        abs((576 + 31 * 47) / 32 - 63.53) < 0.01 and abs((592 + 31 * 47) / 32 - 64.03) < 0.01 and abs((2 * 592 + 30 * 30) / 32 - 65.125) < 1e-9)
    checks["benches vqav2_dev2, vqav2_test3, gqa_val400 registered with existing files"] = all(
        os.path.exists(exp0002.BENCHES[k]["question_file"]) and os.path.exists(exp0002.BENCHES[k]["annotations"][0])
        for k in ("vqav2_dev2", "vqav2_test3", "gqa_val400"))
    src = open(f"{HERE}/score_pope_mme.py", encoding="utf-8").read()
    checks["score_pope_mme knows the three benches"] = all(f'"{k}"' in src for k in ("vqav2_dev2", "vqav2_test3", "gqa_val400"))
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    if not checks["without zoom and cover: kept sets and answers equal the committed code's"]:
        print(json.dumps({k: (v if isinstance(v, dict) and "error" in v else "...") for k, v in outs.items()}))
    print(f"{sum(checks.values())}/{len(checks)} checks pass")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
