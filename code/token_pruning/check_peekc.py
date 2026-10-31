"""CPU check of IDEA-0009's --fastv-pos-after contiguous (with --fastv-div-peek prefill / none) on a tiny random LLaVA.

  python -B check_peekc.py --workdir D:/6CJNM007/tmp/exp0038/smoke

K = 1, keep 61, DivPrune's selection made before layer 0 (as EXP-0034). Layer 0 runs on the whole prompt at its own
positions with the mode's mask ("prefill": every token sees the patches to be dropped; "none": no token does); from layer 1
on, the system prompt, the kept patches and the text are numbered contiguously (0, 1, 2, ...), as when the same tokens are
fed to the LLM without the others; while decoding, layer 0 numbers the generated tokens after the whole prompt and layers
1 to 31 after the shortened one, and they never see the dropped patches. References are written out here by running the
tiny model's layers by hand on a plain LLaVA with the same weights. Checks: one log record per prefill with the mode;
layer 0 sees the whole prompt and layer 1 on 61 patches; the prefill logits equal the hand-run reference; generate equals
the reference's greedy decoding without a cache; the recorded shift is 515; contiguous differs from the original numbering
(EXP-0034's peek / nopeek); argument, binding and config checks. check_peek.py (EXP-0034, no pos_after) is re-run separately.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_peek as cp  # noqa: E402
import divprune  # noqa: E402
import fastv_llava  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import HEIGHT, WIDTH, Pieces, refused  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

KEEP, N_IMG, N_NEW = 61, 576, 3


def margs(peek, pos_after):
    m = cp.margs(peek)
    m.fastv_pos_after = pos_after
    return m


def manual_logits_c(ref, embeds, kept, mode, n_prompt):
    """Layer 0 on the whole sequence at its own positions with the mode's mask (generated tokens, after n_prompt, never see
    the dropped patches); then the system prompt, the kept patches, the text and the generated tokens, numbered 0, 1, 2,
    ... through the other layers; logits of the last position."""
    lm = ref.get_model()
    n = embeds.shape[1]
    pos = torch.arange(n).unsqueeze(0)
    drop = torch.ones(N_IMG, dtype=torch.bool)
    drop[kept] = False
    cols = SYS_LEN + drop.nonzero().squeeze(1)
    mask = cp.causal(n, embeds).clone()
    rows = {"prefill": [], "none": list(range(n))}[mode] + list(range(n_prompt, n))
    if rows:
        r = torch.tensor(sorted(set(rows)))
        mask[:, :, r[:, None], cols[None, :]] = torch.finfo(mask.dtype).min
    h = lm.layers[0](embeds, attention_mask=mask, position_ids=pos)[0]
    keep_idx = torch.cat((torch.arange(SYS_LEN), SYS_LEN + kept, torch.arange(SYS_LEN + N_IMG, n)))
    h = h[:, keep_idx]
    pos = torch.arange(h.shape[1]).unsqueeze(0)
    mask = cp.causal(h.shape[1], h)
    for layer in lm.layers[1:]:
        h = layer(h, attention_mask=mask, position_ids=pos)[0]
    return ref.lm_head(lm.norm(h))[0, -1]


def manual_generate_c(ref, embeds, kept, mode, n_new):
    lm, out, e = ref.get_model(), [], embeds
    n_prompt = embeds.shape[1]
    for _ in range(n_new):
        tok = int(manual_logits_c(ref, e, kept, mode, n_prompt).argmax())
        out.append(tok)
        if tok == ref.config.eos_token_id:
            break
        e = torch.cat((e, lm.embed_tokens(torch.tensor([[tok]]))), dim=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    ids = cp.prompt_ids(after)
    checks = {}
    checks["this check imports the new divprune, fastv_llava, run_eval and check_peek"] = all(
        cp.under(m.__file__, HERE) for m in (divprune, fastv_llava, run_eval, cp))
    spec = {"peekc": ("prefill", "contiguous"), "nopeekc": ("none", "contiguous"), "peek": ("prefill", None), "nopeek": ("none", None)}
    models = {n: run_eval.apply_method(build_tiny_llava(a.workdir), margs(*s), Pieces(after.tolist())) for n, s in spec.items()}
    ref = build_tiny_llava(a.workdir)
    lens, logits, caches, gens = {}, {}, {}, {}
    for name, model in models.items():
        lm = model.get_model()
        lens[name] = []
        lm.layers[0].register_forward_pre_hook(lambda mod, args, kw, n=name: lens[n].append(("l0", args[0].shape[1])) if args[0].shape[1] > 1 else None, with_kwargs=True)
        lm.layers[1].register_forward_pre_hook(lambda mod, args, kw, n=name: lens[n].append(("l1", args[0].shape[1])) if args[0].shape[1] > 1 else None, with_kwargs=True)
        logits[name], caches[name], gens[name] = [], [], []
        with torch.no_grad():
            for i in range(2):
                out = model(input_ids=ids, images=cp.image(i), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
                logits[name].append(out.logits[0, -1])
                pkv = out.past_key_values
                caches[name].append([pkv[l][0].shape[2] for l in range(len(pkv))])
                g = model.generate(ids, images=cp.image(i), image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=N_NEW, use_cache=True)
                gens[name].append(g[0].tolist())
    with torch.no_grad():
        feats = [type(ref).encode_images(ref, cp.image(i))[0] for i in range(2)]
        order = [divprune.maxmin_order(f, 64) for f in feats]
        kept61 = [torch.tensor(sorted(o[:KEEP])) for o in order]
        embeds = [ref.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, cp.image(i), image_sizes=[(WIDTH, HEIGHT)])[4] for i in range(2)]
        for name in ("peekc", "nopeekc"):
            mode = spec[name][0]
            lm = models[name].get_model()
            log = lm.fastv_div_log
            checks[f"{name}: one log record per prefill, the first 61 of DivPrune's order, with the mode"] = (
                len(log) == 4 and all(r["order"] == order[[0, 0, 1, 1][j]][:KEEP] for j, r in enumerate(log)) and all(r.get("peek") == mode for r in log))
            checks[f"{name}: layer 0 sees the whole prompt, layer 1 on 61 patches (every prefill)"] = (
                [x for x in lens[name] if x[0] == "l0"] == [("l0", SYS_LEN + N_IMG + TEXT_LEN)] * 4
                and [x for x in lens[name] if x[0] == "l1"] == [("l1", SYS_LEN + KEEP + TEXT_LEN)] * 4)
            ref_l = [manual_logits_c(ref, embeds[i], kept61[i], mode, embeds[i].shape[1]) for i in range(2)]
            checks[f"{name}: prefill logits = the hand-run reference with contiguous numbering from layer 1 (atol 1e-5)"] = all(
                torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(logits[name], ref_l))
            checks[f"{name}: the shift is 515 and the cache keeps every token at layer 0, 61 patches from layer 1"] = (
                lm.fastv_pos_shift == N_IMG - KEEP
                and all(c[0] == SYS_LEN + N_IMG + TEXT_LEN and c[1:] == [SYS_LEN + KEEP + TEXT_LEN] * (len(c) - 1) for c in caches[name]))
            ref_g = [manual_generate_c(ref, embeds[i], kept61[i], mode, N_NEW) for i in range(2)]
            bos = ref.config.bos_token_id
            checks[f"{name}: generate = the reference's greedy decoding without a cache"] = all(
                g[0] == bos and g[1:1 + len(r)] == r and len(r) >= 1 for g, r in zip(gens[name], ref_g))
            other = name[:-1]
            checks[f"{name}: differs from {other} (the original numbering)"] = all(
                not torch.allclose(x, y, atol=1e-4) for x, y in zip(logits[name], logits[other]))
        checks["peek and nopeek (no pos_after) give no shift"] = (models["peek"].get_model().fastv_pos_shift is None
                                                                 and models["nopeek"].get_model().fastv_pos_shift is None)
    k1 = ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "61", "--fastv-score", "div", "--fastv-div-log", "x.json"]
    checks["arguments: --fastv-pos-after contiguous with --fastv-div-peek prefill / none is accepted"] = all(
        not refused(k1 + ["--fastv-div-peek", m, "--fastv-pos-after", "contiguous"]) for m in ("prefill", "none"))
    checks["arguments: --fastv-pos-after without --fastv-div-peek or with another value is refused"] = (
        refused(k1 + ["--fastv-pos-after", "contiguous"]) and refused(k1 + ["--fastv-div-peek", "prefill", "--fastv-pos-after", "sparse"]))
    try:
        fastv_llava.bind_fastv(build_tiny_llava(a.workdir).get_model(), 1, KEEP, score_kind="div", pos_after="contiguous")
        checks["bind_fastv: pos_after without div_peek is refused"] = False
    except ValueError:
        checks["bind_fastv: pos_after without div_peek is refused"] = True
    C = cp.config_module().CONFIGS
    base = ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "61", "--fastv-score", "div", "--fastv-div-peek"]
    checks["configs: fastv61k1_div_peekc_kll, fastv61k1_div_nopeekc_kll and divprune61"] = (
        C["fastv61k1_div_peekc_kll"] == ("tp", base + ["prefill", "--fastv-pos-after", "contiguous"])
        and C["fastv61k1_div_nopeekc_kll"] == ("tp", base + ["none", "--fastv-pos-after", "contiguous"])
        and C["divprune61"] == ("tp", ["--method", "divprune", "--divprune-keep", "61"]))
    checks["configs: each new config with the logs run_one adds parses"] = (
        all(not refused(C[n][1] + ["--fastv-div-log", "x.json", "--kept-token-log", "k.json", "--answer-logit-log", "l.json", "--layer-token-log", "t.json"])
            for n in ("fastv61k1_div_peekc_kll", "fastv61k1_div_nopeekc_kll"))
        and not refused(C["divprune61"][1] + ["--divprune-log", "d.json"]))
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_peekc.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
