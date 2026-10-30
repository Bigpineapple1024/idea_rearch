"""CPU check of EXP-0033's exploratory combinations (combo.py; run_eval --method vzcand_div / div_qroute / div_ctx and
--vz-fastv) on a tiny random LLaVA (smoke_test.build_tiny_llava).

  python -B check_combo.py --workdir D:/6CJNM007/tmp/exp0033/smoke

References are written out here independently (Algorithm 1 read directly in float64, VisionZip's contextual rule as a
loop, the IDF x visibility query and S recomputed from the stand-in tables):
- X2 vzcand_div: the candidates are the top 128 patches by VisionZip's dominant score (CLS attention at the tower's
  layer -2, summed over heads, from a separate forward); the 64 kept are the first 64 of the max-min order on the
  candidates' projector outputs; the LLM sees exactly those 64 projector outputs, in index order.
- X3 div_qroute: a routed prompt takes round(64 S) patches by the cosine to the IDF x visibility query, then the
  max-min order continues from them; an unrouted prompt keeps DivPrune-64's choice.
- X4 div_ctx: 54 by the max-min order and 10 contextual tokens by VisionZip's rule on the tower features, through the
  projector.
- X1 --vz-fastv (checked last: VisionZip patches classes globally): layers 0 and 1 see VisionZip-128's 128 image tokens,
  equal to VisionZip-128 alone, layer 2 on 60 kept by plain FastV.
- Arguments and configs (in tmp both exp0002 variants, one per possible base; in the repository the committed one).
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
os.environ.setdefault("VISIONZIP_ROOT", "D:/6CJNM007/idea_rearch/source/VisionZip")
import combo  # noqa: E402
import divprune  # noqa: E402
import run_eval  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

WIDTH, HEIGHT = 640, 480
ROUTED = "Is there a cat in the image?\nAnswer the question using a single word or phrase."
UNROUTED = "What color is the cat?\nAnswer the question using a single word or phrase."


class Text:
    """Tokenizer stand-in: decode gives the prompt text the router reads, switched per prefill."""
    text = UNROUTED

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


def reference_from(x, m, start=()):
    """Algorithm 1 read directly (float64), optionally continued from the rows in `start`."""
    v = x.double()
    v = v / v.norm(dim=1, keepdim=True)
    dist = (1 - v @ v.T).tolist()
    R = [i for i in range(len(dist)) if i not in set(start)]
    E = list(start)
    if not E:
        D = [min(dist[i][j] for j in R if j != i) for i in R]
        E.append(R.pop(int(np.argmax(D))))
    while len(E) < m:
        D = [min(dist[i][j] for j in E) for i in R]
        E.append(R.pop(int(np.argmax(D))))
    return E


def merge_reference(h, contextual):
    """VisionZip's contextual tokens (clip_encoder.py) written as a loop over the rows of h, cosine of h as the metric."""
    n = h.shape[0]
    step = max(1, n // contextual)
    targets = list(range(0, n, step))[:contextual]
    hn = h.double() / h.double().norm(dim=1, keepdim=True)
    groups = {t: [] for t in targets}
    for i in range(n):
        if i in groups:
            continue
        sims = [float(hn[i] @ hn[t]) for t in targets]
        groups[targets[int(np.argmax(sims))]].append(i)
    out = []
    for t in targets:
        agg = h[groups[t]].double().mean(0) if groups[t] else torch.zeros(h.shape[1], dtype=torch.float64)
        out.append(h[t].double() + agg)
    return torch.stack(out), targets


def run(workdir, m, texts, tok=None):
    """One model with method args m, one prefill per text (images seeded 10, 11, ...); records inputs and lengths."""
    tok = tok or Text()
    model = run_eval.apply_method(build_tiny_llava(workdir), m, tok)
    lm = model.get_model()
    got = {"embeds": [], "len": {0: [], 1: [], 2: []}, "images": [], "gen": [], "kept": []}

    def pre_lm(module, args, kwargs):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1:
            got["embeds"].append(e.detach().clone())

    def pre_layer(i):
        def hook(module, args, kwargs):
            if args[0].shape[1] > 1:
                got["len"][i].append(args[0].shape[1])
        return hook

    def post_lm(module, args, kwargs, output):
        e = kwargs.get("inputs_embeds")
        if e is not None and e.shape[1] > 1 and getattr(module, "fastv_kept_image_positions", None) is not None:
            got["kept"].append(module.fastv_kept_image_positions.tolist())

    lm.register_forward_pre_hook(pre_lm, with_kwargs=True)
    for i in (0, 1, 2):
        lm.layers[i].register_forward_pre_hook(pre_layer(i), with_kwargs=True)
    lm.register_forward_hook(post_lm, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    after = torch.arange(100, 100 + TEXT_LEN)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    with torch.no_grad():
        for i, text in enumerate(texts):
            tok.text = text
            images = torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))
            got["images"].append(images)
            out = model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2,
                                 use_cache=True)
            got["gen"].append(out.tolist())
    return model, lm, got, after


@torch.no_grad()
def tower_reference(model, images):
    """A separate forward of the tiny tower: plain LLaVA's patch features, VisionZip's dominant score, the projector."""
    vt = model.get_model().get_vision_tower()
    outs = vt.vision_tower(images.to(device=vt.device, dtype=vt.dtype), output_hidden_states=True, output_attentions=True)
    cls = outs.attentions[-2][:, :, 0, 1:].sum(dim=1)[0]
    feats = vt.feature_select(outs).to(images.dtype)[0]
    return feats, cls, model.get_model().mm_projector(feats)


def config_modules():
    tags = [t for t in ("0030", "0031") if os.path.exists(f"{HERE}/exp0002_on0032_{t}.py")]
    out = {}
    for tag in tags or [""]:
        name = f"exp0002_on0032_{tag}" if tag else "exp0002"
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
    checks = {}
    checks["this check imports the new combo, divprune and run_eval"] = all(under(x.__file__, HERE) for x in (combo, divprune, run_eval))

    # 0. the continued max-min order
    for seed, (n, d, m, s) in enumerate([(576, 64, 64, 9), (200, 8, 60, 0), (100, 16, 50, 49)]):
        x = torch.randn(n, d, generator=torch.Generator().manual_seed(seed))
        start = list(torch.randperm(n, generator=torch.Generator().manual_seed(100 + seed))[:s].tolist())
        checks[f"maxmin_from = Algorithm 1 continued from {s} given rows ({n} x {d}, {m} picks)"] = combo.maxmin_from(x, m, start) == reference_from(x, m, start)
    x = torch.randn(80, 8, generator=torch.Generator().manual_seed(5))
    checks["maxmin_from with no start rows is maxmin_order"] = combo.maxmin_from(x, 30, []) == divprune.maxmin_order(x, 30)

    # X2: CLS candidates, then max-min
    m2 = argparse.Namespace(method="vzcand_div", combo_cand=128, combo_keep=64, combo_log="x.json", fastv_schedule=None)
    model, lm, got, after = run(a.workdir, m2, [UNROUTED, UNROUTED])
    ok_emb, ok_log, ok_sub = [], [], []
    for i, im in enumerate(got["images"]):
        feats, cls, proj = tower_reference(model, im)
        cand = sorted(cls.topk(128).indices.tolist())
        order = reference_from(proj[cand], 64)
        kept = sorted(cand[j] for j in order)
        ok_emb.append(torch.equal(got["embeds"][i][0, SYS_LEN:SYS_LEN + 64], proj[kept]))
        r = model.combo_log[i]
        ok_log.append(r["cand"] == cand and r["order"] == order and r["kept"] == kept)
        ok_sub.append(set(r["kept"]) <= set(cand))
    checks["X2: the LLM sees the system prompt, 64 image tokens and the text, in both prefills"] = got["len"][0] == [SYS_LEN + 64 + TEXT_LEN] * 2
    checks["X2: the 64 image tokens are the projector outputs of the max-min choice among the CLS top 128"] = all(ok_emb)
    checks["X2: one log record per image: the candidates, the order on them and the kept patches"] = all(ok_log) and len(model.combo_log) == 2
    checks["X2: the kept patches are candidates"] = all(ok_sub)

    # X3: routed prompts take round(64 S) by the query first; unrouted ones keep DivPrune-64's choice
    rng = np.random.default_rng(3)
    vis = rng.uniform(0, 1, 1000).astype(np.float32)
    np.save(f"{a.workdir}/vis_combo.npy", vis)
    idf_vals = {int(t): float(v) for t, v in zip(range(100, 100 + TEXT_LEN), rng.uniform(0, 3, TEXT_LEN))}
    idf_vals[100] = 0.0
    idf = {"idf": idf_vals, "unseen": 1.0, "n_questions": 10, "df": {}}
    m3 = argparse.Namespace(method="div_qroute", combo_keep=64, combo_visibility=f"{a.workdir}/vis_combo.npy", combo_log="x.json",
                            fastv_idf=idf, fastv_schedule=None)
    model, lm, got, after = run(a.workdir, m3, [ROUTED, UNROUTED])
    ids = after.tolist()
    w_idf = torch.tensor([idf_vals.get(t, 1.0) for t in ids], dtype=torch.float64)
    w = w_idf * torch.tensor(vis[ids], dtype=torch.float64)
    share = float(w.sum() / w_idf.sum())
    emb = lm.embed_tokens(after).double()
    q = (w[:, None] * emb).sum(0) / w.sum()
    feats = [type(model).encode_images(model, im)[0] for im in got["images"]]   # plain LLaVA's projector outputs
    n_q = round(64 * share)
    cos = torch.nn.functional.cosine_similarity(feats[0].double(), q.unsqueeze(0), dim=-1)
    first = cos.topk(n_q).indices.tolist()
    ref_routed = reference_from(feats[0], 64, first)
    ref_unrouted = reference_from(feats[1], 64)
    r0, r1 = model.combo_log
    checks["X3: the routed prompt is routed, the other is not"] = r0["routed"] is True and r1["routed"] is False
    checks["X3: routed: S and round(64 S) as recomputed from the IDF and visibility tables"] = abs(r0["share"] - share) < 1e-6 and r0["n_query"] == n_q and 0 < n_q < 64
    checks["X3: routed: the first ones are the patches most similar to the query, in that order"] = r0["first"] == first
    checks["X3: routed: the order continues by max-min from them; the LLM sees those 64"] = (
        r0["order"] == ref_routed and torch.equal(got["embeds"][0][0, SYS_LEN:SYS_LEN + 64], feats[0][sorted(ref_routed)]))
    checks["X3: unrouted: DivPrune-64's own choice"] = (
        r1["order"] == ref_unrouted and torch.equal(got["embeds"][1][0, SYS_LEN:SYS_LEN + 64], feats[1][sorted(ref_unrouted)]))
    checks["X3: the LLM sees 64 image tokens in both prefills"] = got["len"][0] == [SYS_LEN + 64 + TEXT_LEN] * 2

    # X4: 54 by max-min, 10 contextual tokens by VisionZip's rule on the tower features
    m4 = argparse.Namespace(method="div_ctx", combo_keep=54, combo_contextual=10, combo_log="x.json", fastv_schedule=None)
    model, lm, got, after = run(a.workdir, m4, [UNROUTED, UNROUTED])
    ok_emb, ok_log = [], []
    for i, im in enumerate(got["images"]):
        tf, _, proj = tower_reference(model, im)
        order = reference_from(proj, 54)
        kept = sorted(order)
        rest = [j for j in range(576) if j not in set(kept)]
        ctx, targets = merge_reference(tf[rest], 10)
        proj_ctx = model.get_model().mm_projector(ctx.to(tf.dtype))
        e = got["embeds"][i][0, SYS_LEN:SYS_LEN + 64]
        ok_emb.append(torch.equal(e[:54], proj[kept]) and torch.allclose(e[54:], proj_ctx, atol=1e-5, rtol=1e-4))
        r = model.combo_log[i]
        ok_log.append(r["order"] == order and r["kept"] == kept and r["targets"] == [rest[t] for t in targets])
    checks["X4: the LLM sees 54 + 10 image tokens in both prefills"] = got["len"][0] == [SYS_LEN + 64 + TEXT_LEN] * 2
    checks["X4: the 54 are the max-min choice's projector outputs, the 10 VisionZip's merge of the rest through the projector"] = all(ok_emb)
    checks["X4: one log record per image: the order, the kept patches and the merge targets"] = all(ok_log)

    # X1 (last: VisionZip patches classes globally): VisionZip-128, then plain FastV keeping 60 from layer 2
    base = dict(method="visionzip", dominant=108, contextual=20, vz_cover=0, fastv_schedule=None)
    _, _, got_vz, _ = run(a.workdir, argparse.Namespace(**base, vz_fastv=None), [UNROUTED, UNROUTED])
    model, lm, got, _ = run(a.workdir, argparse.Namespace(**base, vz_fastv="2,60"), [UNROUTED, UNROUTED])
    full, cut = SYS_LEN + 128 + TEXT_LEN, SYS_LEN + 60 + TEXT_LEN
    checks["X1: layers 0 and 1 see VisionZip-128's 128 image tokens, layer 2 on 60, in both prefills"] = (
        got["len"][0] == [full] * 2 and got["len"][1] == [full] * 2 and got["len"][2] == [cut] * 2)
    checks["X1: the 128 image tokens equal VisionZip-128 alone"] = all(
        torch.equal(x[0, SYS_LEN:SYS_LEN + 128], y[0, SYS_LEN:SYS_LEN + 128]) for x, y in zip(got["embeds"], got_vz["embeds"]))
    checks["X1: VisionZip-128 alone keeps 128 at every layer"] = got_vz["len"][2] == [full] * 2
    checks["X1: FastV keeps 60 of the 128 (positions within the image span)"] = (
        len(got["kept"]) == 2 and all(len(k) == 60 and all(0 <= p < 128 for p in k) for k in got["kept"]))
    checks["X1: the budget is 64.25 tokens per layer"] = abs((2 * 128 + 30 * 60) / 32 - 64.25) < 1e-12

    # arguments and configs
    ok = {"X2": ["--method", "vzcand_div", "--combo-cand", "128", "--combo-keep", "64", "--combo-log", "x.json"],
          "X3": ["--method", "div_qroute", "--combo-keep", "64", "--combo-visibility", "v.npy", "--combo-log", "x.json"],
          "X4": ["--method", "div_ctx", "--combo-keep", "54", "--combo-contextual", "10", "--combo-log", "x.json"],
          "X1": ["--method", "visionzip", "--dominant", "108", "--contextual", "20", "--vz-fastv", "2,60"]}
    checks["arguments: the four combinations are accepted"] = not any(refused(v) for v in ok.values())
    checks["arguments: a combination without its keep, its log or its own option is refused"] = all(refused(v) for v in (
        ok["X2"][:6], ok["X2"][:2] + ok["X2"][4:], ok["X2"][:2] + ok["X2"][2:4] + ok["X2"][6:],
        ok["X3"][:4] + ok["X3"][6:], ok["X4"][:4] + ok["X4"][6:]))
    checks["arguments: the combination options without their method are refused"] = all(refused(v) for v in (
        ["--method", "vanilla", "--combo-keep", "64"], ["--method", "divprune", "--divprune-keep", "64", "--divprune-log", "x.json",
                                                        "--combo-cand", "128"],
        ["--method", "vanilla", "--combo-visibility", "v.npy"], ["--method", "vanilla", "--combo-contextual", "10"],
        ["--method", "fastv", "--vz-fastv", "2,60"],
        ["--method", "visionzip", "--vz-fastv", "2,60", "--vz-cover", "16", "--vz-visibility", "v.npy"]))
    for tag, e in config_modules().items():
        C = e.CONFIGS
        checks[f"configs ({tag}): the four combinations"] = (
            C["vz128_fastv60"] == ("tp", C["visionzip128"][1] + ["--vz-fastv", "2,60"])
            and C["visionzip128"][1] == ["--method", "visionzip", "--dominant", "108", "--contextual", "20"]
            and C["vzcand128_div64"] == ("tp", ok["X2"][:6])
            and C["div64_qroute"][1][:4] == ["--method", "div_qroute", "--combo-keep", "64"]
            and C["div64_qroute"][1][4] == "--combo-visibility"
            and C["div64_qroute"][1][5].endswith("EXP-0015/visibility/visibility.npy") and os.path.exists(C["div64_qroute"][1][5])
            and C["div54_ctx10"] == ("tp", ok["X4"][:6]))
        checks[f"configs ({tag}): fastv64, vanilla and divprune64 are unchanged"] = (
            C["fastv64"] == ("tp", ["--method", "fastv", "--fastv-k", "2", "--fastv-keep", "30"])
            and C["vanilla"] == ("tp", ["--method", "vanilla"])
            and C["divprune64"] == ("tp", ["--method", "divprune", "--divprune-keep", "64"]))
        src = inspect.getsource(e.run_one)
        checks[f"configs ({tag}): run_one adds the combination log"] = '"--combo-log", f"{run_dir}/combo.json"' in src
        checks[f"configs ({tag}): each combination with the log run_one adds parses"] = not any(refused(C[c][1] + extra) for c, extra in (
            ("vzcand128_div64", ["--combo-log", "x.json"]), ("div64_qroute", ["--combo-log", "x.json"]),
            ("div54_ctx10", ["--combo-log", "x.json"]), ("vz128_fastv60", []),
            ("vz128_fastv60_kll", ["--kept-token-log", "k.json", "--answer-logit-log", "l.json", "--layer-token-log", "t.json"])))
        kll = ('    if config.endswith("_kll"):      # EXP-0033 X1: the kept patches, the log-probabilities and the tokens entering each layer\n'
               '        method_args = method_args + ["--kept-token-log", f"{run_dir}/kept_tokens.json", "--answer-logit-log",\n'
               '                                     f"{run_dir}/answer_logits.json", "--layer-token-log", f"{run_dir}/layer_tokens.json"]\n')
        checks[f"configs ({tag}): vz128_fastv60_kll is vz128_fastv60, the only _kll config; run_one gives it the three logs"] = (
            C["vz128_fastv60_kll"] == C["vz128_fastv60"] and [c for c in C if c.endswith("_kll")] == ["vz128_fastv60_kll"]
            and src.count(kll) == 1)
    checks["arguments: the kept-patch log is refused for VisionZip without --vz-fastv"] = refused(
        ["--method", "visionzip", "--dominant", "108", "--contextual", "20", "--kept-token-log", "k.json"])

    for name, okv in checks.items():
        print(f"{'ok  ' if okv else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_combo.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
