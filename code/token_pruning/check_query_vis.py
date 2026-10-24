"""CPU check of EXP-0015's --fastv-score query_vis, --vocab-visibility-log and make_visibility.py.

  python check_query_vis.py --workdir D:/6CJNM007/tmp/tp_smoke

Tiny random LLaVA of check_object_cos.py (K = 2, keep 30, 6 x 6 blocks, padding excluded, a position prior): with
visibility 1 everywhere query_vis keeps what query_idf keeps; with visibility 0 it keeps what the strong baseline keeps;
with random visibility the kept set equals the offline rule (round(30 * S) by the cosine to the IDF x visibility query,
S = sum(IDF * vis) / sum(IDF), the rest by the prior-divided attention score, empty blocks first) and the logged
object-name part equals the first stage. The vocabulary log stores one row per distinct image, equal to the largest and
mean cosine recomputed. make_visibility.py gives values in [0, 1] ordered as the raw 95th percentiles. Argument checks.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval  # noqa: E402
from analyze_object_boxes import padding_patches  # noqa: E402
from check_mix_names import cell, stratified  # noqa: E402
from check_object_cos import BLOCKS, HEIGHT, KEEP, WIDTH, Pieces, refused  # noqa: E402
from fastv_llava import _record_image_span  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402


def run(workdir, after, score, idf=None, prior=None, vis_path=None):
    model = build_tiny_llava(workdir)
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=prior,
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score=score, fastv_exclude_padding=True,
                           fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=idf, fastv_visibility=vis_path)
    model = run_eval.apply_method(model, m, Pieces(after.tolist()))
    lm = model.get_model()
    run_eval._KEPT.clear()
    run_eval._KEPT_QUESTION.clear()
    lm.register_forward_hook(run_eval._record_kept, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
        h = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images,
                                                       image_sizes=[(WIDTH, HEIGHT)])[4][0].float()
    return list(run_eval._KEPT), list(run_eval._KEPT_QUESTION), lm, h


def vocab_log_check(workdir, after):
    model = build_tiny_llava(workdir)
    _record_image_span(model)
    lm = model.get_model()
    for k in ("max", "mean"):
        run_eval._VOCAB_VIS[k] = []
    run_eval._VOCAB_VIS["seen"] = set()
    run_eval._log_vocab_visibility(lm)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(2)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    a, b = torch.randn(1, 3, 336, 336), torch.randn(1, 3, 336, 336)
    embeds = []
    with torch.no_grad():
        for images in (a, a, b):
            model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
            embeds.append(model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images,
                                                                      image_sizes=[(WIDTH, HEIGHT)])[4][0].float())
    path = f"{workdir}/vocab_visibility.npz"
    run_eval._write_vocab_visibility(path)
    z = np.load(path)
    w = torch.nn.functional.normalize(lm.embed_tokens.weight.detach().float(), dim=-1)
    ok = z["max"].shape == (2, w.shape[0])
    for row, e in ((0, embeds[0]), (1, embeds[2])):
        sims = torch.nn.functional.normalize(e[SYS_LEN:SYS_LEN + 576], dim=-1) @ w.T
        ok &= np.allclose(z["max"][row].astype(np.float32), sims.max(0).values.numpy(), atol=2e-3)
        ok &= np.allclose(z["mean"][row].astype(np.float32), sims.mean(0).numpy(), atol=2e-3)
    return {"vocabulary log: one row per distinct image, largest and mean cosine recomputed": bool(ok)}, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    pad = padding_patches({"width": WIDTH, "height": HEIGHT})
    nonpad = np.setdiff1d(np.arange(576), pad)
    rng = np.random.default_rng(0)
    idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
    prior = f"{a.workdir}/prior_vis.npy"
    np.save(prior, (torch.rand(576, generator=torch.Generator().manual_seed(3)) + 0.5).numpy())
    paths = {}
    for name, vec in (("ones", np.ones(1000)), ("zeros", np.zeros(1000)), ("rand", rng.uniform(0, 1, 1000))):
        paths[name] = f"{a.workdir}/vis_{name}.npy"
        np.save(paths[name], vec.astype(np.float32))
    k1, _, _, _ = run(a.workdir, after, "query_vis", idf, prior, paths["ones"])
    k_idf, _, _, _ = run(a.workdir, after, "query_idf", idf)
    k0, _, _, _ = run(a.workdir, after, "query_vis", idf, prior, paths["zeros"])
    k_base, _, _, _ = run(a.workdir, after, "last", None, prior)
    kr, qr, lm, h = run(a.workdir, after, "query_vis", idf, prior, paths["rand"])
    vis = np.load(paths["rand"])
    wi = np.array([idf["idf"][int(t)] for t in after])
    w = wi * vis[after.numpy()]
    s = w.sum() / wi.sum()
    q = (torch.tensor(w, dtype=torch.float32)[:, None] * h[SYS_LEN + 576:SYS_LEN + 576 + TEXT_LEN]).sum(0, keepdim=True) / float(w.sum())
    cos = torch.nn.functional.cosine_similarity(h[SYS_LEN:SYS_LEN + 576], q, dim=-1).numpy()
    att = lm.fastv_last_scores.float().numpy() / np.load(prior)
    n_q = round(KEEP * s)
    first = stratified(cos, nonpad, n_q)
    filled = {}
    for x in first:
        filled[cell(x)] = filled.get(cell(x), 0) + 1
    offline = sorted(first + stratified(att, np.setdiff1d(nonpad, first), KEEP - n_q, filled))
    checks = {
        "visibility 1 everywhere: query_vis keeps what query_idf keeps": k1[0] == k_idf[0],
        "visibility 0 everywhere: query_vis keeps what the strong baseline keeps": k0[0] == k_base[0],
        f"random visibility: S = {s:.3f}, n_q = {n_q}, logged share and count match":
            abs(lm.fastv_visible_share - s) < 1e-5 and lm.fastv_mix_count == n_q,
        "random visibility: kept set equals the offline two-stage rule": kr[0] == offline,
        "random visibility: logged object-name part equals the first stage": qr[0] == sorted(first),
        "30 kept, none in the padding": len(kr[0]) == KEEP and not set(kr[0]) & set(pad.tolist()),
    }
    vc, vpath = vocab_log_check(a.workdir, after)
    checks.update(vc)
    out = f"{a.workdir}/vis_table_check"
    if os.path.exists(out):
        import shutil
        shutil.rmtree(out)
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", f"{HERE}/make_visibility.py", vpath, out], capture_output=True, text=True,
                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if r.returncode == 0:
        v, raw = np.load(f"{out}/visibility.npy"), np.load(f"{out}/visibility_raw.npy")
        checks["make_visibility: values in [0, 1], ordered as the raw 95th percentiles"] = bool(
            v.min() == 0 and v.max() == 1 and np.all(np.diff(v[np.argsort(raw, kind='stable')]) > 0))
    else:
        print(r.stderr[-1500:])
        checks["make_visibility ran"] = False
    checks.update({
        "query_vis refused without --fastv-visibility": refused(["--method", "fastv", "--fastv-score", "query_vis",
                                                                 "--fastv-stratify", "6"]),
        "--fastv-visibility refused without query_vis": refused(["--method", "fastv", "--fastv-visibility", "x"]),
        "query_vis refused without blocks": refused(["--method", "fastv", "--fastv-score", "query_vis", "--fastv-visibility", "x"]),
        "query_vis accepted with blocks, prior and padding exclusion": not refused([
            "--method", "fastv", "--fastv-score", "query_vis", "--fastv-visibility", "x", "--fastv-stratify", "6",
            "--fastv-score-prior", "x", "--fastv-exclude-padding"]),
        "--vocab-visibility-log refused for visionzip": refused(["--method", "visionzip", "--vocab-visibility-log", "x"]),
    })
    import exp0002
    checks["exp0002 fastv64_qvis_kl = strong baseline + query_vis"] = exp0002.CONFIGS["fastv64_qvis_kl"][1] == (
        exp0002.CONFIGS["fastv64_prior_strat_nopad"][1] + ["--fastv-score", "query_vis", "--fastv-visibility", exp0002.VISIBILITY])
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
