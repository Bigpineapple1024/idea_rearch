"""CPU check of EXP-0032's DivPrune selection (divprune.py, --method divprune, --fastv-score div) on a tiny random
LLaVA (smoke_test.build_tiny_llava).

  python -B check_divprune.py --workdir D:/6CJNM007/tmp/exp0032/smoke

Checks: maxmin_order equals a direct reading of the paper's Algorithm 1 (float64, loops over a precomputed distance
matrix) on random data, and its first 47 picks are its 47-pick selection; --method divprune: the LLM sees the system
prompt, 64 image tokens and the text, the 64 are plain LLaVA's projector outputs at the selected patches in their
original order, one log record per image; --fastv-score div at K = 1: layer 0 sees all 576 patches, from layer 1 on the
first 47 of the same selection on the same input embeddings stay, one log record per prefill in selection order;
argument and config checks (in tmp both exp0002.py variants, one per possible base; in the repository the committed
exp0002.py). The FastV paths without the new option are checked against the committed code by check_zoom.py and
check_k1cls.py, run from a folder with the new files.
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
import divprune  # noqa: E402
import fastv_llava  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import HEIGHT, WIDTH, Pieces, refused  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

KEEP_PRE, KEEP_K1 = 64, 47


def under(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


def reference_order(x, m):
    """Algorithm 1 read directly: R all tokens, E empty; the first pick maximises the distance to its nearest other
    token, every later one the distance to its nearest token in E (cosine distance, float64)."""
    v = x.double()
    v = v / v.norm(dim=1, keepdim=True)
    dist = (1 - v @ v.T).tolist()
    R, E = list(range(len(dist))), []
    D = [min(dist[i][j] for j in R if j != i) for i in R]
    E.append(R.pop(int(np.argmax(D))))
    while len(E) < m:
        D = [min(dist[i][j] for j in E) for i in R]
        E.append(R.pop(int(np.argmax(D))))
    return E


def run(workdir, after, m, n_prefill=2):
    """One model with method args m, one prefill per image seeded 10, 11, ...; records the LLM inputs and lengths."""
    model = run_eval.apply_method(build_tiny_llava(workdir), m, Pieces(after.tolist()))
    lm = model.get_model()
    got = {"embeds": [], "len0": [], "len1": [], "kept": [], "gen": [], "images": []}

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
        if e is not None and e.shape[1] > 1 and m.method == "fastv":
            got["kept"].append(module.fastv_kept_image_positions.tolist())

    lm.register_forward_pre_hook(pre_lm, with_kwargs=True)
    lm.layers[0].register_forward_pre_hook(pre0, with_kwargs=True)
    lm.layers[1].register_forward_pre_hook(pre1, with_kwargs=True)
    lm.register_forward_hook(post_lm, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    with torch.no_grad():
        for i in range(n_prefill):
            images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))
            got["images"].append(images)
            out = model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
            got["gen"].append(out.tolist())
    return model, lm, got


def fastv_div_args():
    return argparse.Namespace(method="fastv", fastv_k=1, fastv_keep=KEEP_K1, fastv_schedule=None, fastv_score_prior=None,
                              fastv_stratify=None, text_rows_log=None, fastv_score="div", fastv_exclude_padding=False,
                              fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=None, fastv_visibility=None,
                              fastv_route=None, fastv_zoom=0, fastv_zoom_pick="grid", fastv_zoom_on="unrouted",
                              fastv_zoom_box="8,8,16", fastv_zoom_smin=0.0, fastv_cover=0, fastv_zoom_source="crop",
                              fastv_zoom_centre=False, fastv_cover_top=0, fastv_cls_unrouted=False)


def config_modules():
    """The exp0002.py variants beside this file (in tmp, before the commit: one per possible base), else exp0002.py."""
    tags = [t for t in ("0030", "0031") if os.path.exists(f"{HERE}/exp0002_on{t}.py")]
    out = {}
    for tag in tags or [""]:
        name = f"exp0002_on{tag}" if tag else "exp0002"
        spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[tag or "committed"] = mod
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    checks = {}
    checks["this check imports the new divprune, fastv_llava and run_eval"] = all(
        under(mod.__file__, HERE) for mod in (divprune, fastv_llava, run_eval))

    # A. the greedy order against a direct reading of Algorithm 1
    same, prefix = [], []
    for seed, (n, d, m) in enumerate([(576, 32, 64), (576, 4096, 64), (200, 8, 60), (100, 16, 100)]):
        x = torch.randn(n, d, generator=torch.Generator().manual_seed(seed))
        got = divprune.maxmin_order(x, m)
        same.append(got == reference_order(x, m))
        prefix.append(divprune.maxmin_order(x, 47) == got[:47])
    checks["maxmin_order = Algorithm 1 read directly (4 random cases, up to 576 tokens, up to 4096 dims)"] = all(same)
    checks["maxmin_order: the first 47 picks are the 47-pick selection"] = all(prefix)
    x = torch.randn(30, 8, generator=torch.Generator().manual_seed(7))
    o = divprune.maxmin_order(x, 30)
    checks["maxmin_order: taking all tokens gives each once"] = sorted(o) == list(range(30))
    checks["maxmin_order: m larger than the tokens is capped"] = divprune.maxmin_order(x, 40) == o

    # B. --method divprune: 64 projector outputs at the selected patches, before the LLM
    md = argparse.Namespace(method="divprune", divprune_keep=KEEP_PRE, divprune_log="x.json", fastv_schedule=None)
    model, lm, got = run(a.workdir, after, md)
    plain = [type(model).encode_images(model, im)[0] for im in got["images"]]       # the class method: no selection
    ref64 = [reference_order(f, KEEP_PRE) for f in plain]
    checks["divprune: the LLM sees the system prompt, 64 image tokens and the text, in both prefills"] = (
        got["len0"] == [SYS_LEN + KEEP_PRE + TEXT_LEN] * 2 and got["len1"] == [SYS_LEN + KEEP_PRE + TEXT_LEN] * 2)
    checks["divprune: the 64 image tokens are plain LLaVA's projector outputs at the selected patches, in order"] = all(
        torch.equal(e[0, SYS_LEN:SYS_LEN + KEEP_PRE], f[sorted(r)]) for e, f, r in zip(got["embeds"], plain, ref64))
    log = model.divprune_log
    checks["divprune: one log record per image: the selection order and the sorted kept patches"] = (
        len(log) == 2 and [r["order"] for r in log] == ref64 and [r["kept"] for r in log] == [sorted(r) for r in ref64])
    checks["divprune: the text embeddings are unchanged around the image"] = all(
        torch.equal(e[0, :SYS_LEN], got["embeds"][0][0, :SYS_LEN]) for e in got["embeds"])
    gen_pre = got["gen"]

    # C. --fastv-score div at K = 1: all 576 at layer 0, then the first 47 of the same selection
    model_k, lm_k, got = run(a.workdir, after, fastv_div_args())
    feats = [e[0, SYS_LEN:SYS_LEN + 576] for e in got["embeds"]]
    checks["div: the LLM's image input embeddings are plain LLaVA's projector outputs (all 576)"] = all(
        torch.equal(f, p) for f, p in zip(feats, plain))
    checks["div: layer 0 sees all 576 patches, layer 1 on 47, in both prefills"] = (
        got["len0"] == [SYS_LEN + 576 + TEXT_LEN] * 2 and got["len1"] == [SYS_LEN + KEEP_K1 + TEXT_LEN] * 2)
    checks["div: kept = the first 47 of DivPrune's selection on the same features"] = all(
        k == sorted(r[:KEEP_K1]) for k, r in zip(got["kept"], ref64))
    dl = lm_k.fastv_div_log
    checks["div: one log record per prefill, the kept patches in selection order"] = (
        len(dl) == 2 and [r["order"] for r in dl] == [r[:KEEP_K1] for r in ref64])
    checks["div: no attention map is asked of layer 0"] = "or div_now)" in inspect.getsource(fastv_llava.fastv_forward)
    checks["div and divprune generate (two tokens each)"] = all(len(g[0]) >= 1 for g in gen_pre + got["gen"])

    # D. arguments and configs
    ok_pre = ["--method", "divprune", "--divprune-keep", "64", "--divprune-log", "x.json"]
    checks["arguments: divprune with keep and log is accepted"] = not refused(ok_pre)
    checks["arguments: divprune without keep, without the log, or keep out of 1..576 is refused"] = (
        refused(ok_pre[:2] + ok_pre[4:]) and refused(ok_pre[:4]) and refused(["--method", "divprune", "--divprune-keep", "0",
                                                                               "--divprune-log", "x.json"])
        and refused(["--method", "divprune", "--divprune-keep", "577", "--divprune-log", "x.json"]))
    checks["arguments: --divprune-keep or --divprune-log without divprune is refused"] = (
        refused(["--method", "vanilla", "--divprune-keep", "64"]) and refused(["--method", "vanilla", "--divprune-log", "x.json"]))
    k1 = ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "47"]
    ok_div = k1 + ["--fastv-score", "div", "--fastv-div-log", "x.json"]
    checks["arguments: fastv div with its log is accepted"] = not refused(ok_div)
    checks["arguments: div without the log, or the log without div, is refused"] = (
        refused(ok_div[:-2]) and refused(k1 + ["--fastv-div-log", "x.json"]))
    checks["arguments: div with padding exclusion, blocks, a prior, a schedule or cover is refused"] = all(refused(ok_div + extra) for extra in (
        ["--fastv-exclude-padding"], ["--fastv-stratify", "6"], ["--fastv-score-prior", "p.npy"],
        ["--fastv-schedule", "1:47"], ["--fastv-cover", "16"]))
    try:
        fastv_llava.bind_fastv(build_tiny_llava(a.workdir).get_model(), 1, KEEP_K1, score_kind="div", exclude_padding=True)
        bound_refuses = False
    except ValueError:
        bound_refuses = True
    checks["bind_fastv: div with padding exclusion is refused"] = bound_refuses
    for tag, e in config_modules().items():
        C = e.CONFIGS
        checks[f"configs ({tag}): divprune64, fastv64k1_div and its _kl variant"] = (
            C["divprune64"] == ("tp", ["--method", "divprune", "--divprune-keep", "64"])
            and C["fastv64k1_div"] == ("tp", k1 + ["--fastv-score", "div"]) and C["fastv64k1_div_kl"] == C["fastv64k1_div"])
        src = inspect.getsource(e.run_one)
        checks[f"configs ({tag}): run_one adds the two logs"] = (
            '"--fastv-div-log", f"{run_dir}/div.json"' in src and '"--divprune-log", f"{run_dir}/div.json"' in src)
        checks[f"configs ({tag}): each config with the logs run_one adds parses"] = (
            not refused(C["divprune64"][1] + ["--divprune-log", "x.json"])
            and not refused(C["fastv64k1_div"][1] + ["--fastv-div-log", "x.json", "--kept-token-log", "k.json",
                                                     "--answer-logit-log", "l.json"]))
    checks["configs: the K = 1 budget is 63.53 tokens per layer, within 64"] = abs((576 + 31 * KEEP_K1) / 32 - 63.53125) < 1e-9

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_divprune.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
