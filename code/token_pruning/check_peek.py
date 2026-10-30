"""CPU check of EXP-0034's --fastv-div-peek (IDEA-0008) on a tiny random LLaVA (smoke_test.build_tiny_llava).

  python -B check_peek.py --workdir D:/6CJNM007/tmp/exp0034/smoke

K = 1, keep 61, --fastv-score div (DivPrune's max-min selection on the image tokens' input embeddings) on a 640 x 480
image. The three modes say what layer 0 lets see the patches to be dropped: "prefill" every token (as K = 1), "none" no
token, "text" only the text after the image; in all three the generated tokens never see the dropped patches (their
layer-0 cache entries are masked out while decoding). References are written out here, by running the tiny model's own layers by hand on a plain LLaVA with
the same weights: layer 0 on the whole prompt with the mode's mask, then only the system prompt, the kept patches and the
text (at their original positions) through layers 1 to 31, and greedy decoding without a cache in which the generated
tokens never see the dropped patches. Checks: the kept patches are the first 61 of DivPrune's order on the same
features; layer 0 sees the whole prompt and layer 1 on 61 patches; the prefill logits of each mode equal its reference,
"prefill" also equals the plain K = 1 path with keep 61, and "none" also equals DivPrune's 61 handed to the LLM at their
original positions; the modes differ from each other; the cache keeps every token at layer 0 and the dropped columns are
recorded; generate equals the reference's greedy tokens; one log record per
prefill with the mode; argument, binding and config checks, and the run record's hashes of .npy input tables.
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, "D:/6CJNM007/idea_rearch/code/token_pruning")
import divprune  # noqa: E402
import fastv_llava  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import HEIGHT, WIDTH, Pieces, refused  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402
from transformers.models.llama import modeling_llama  # noqa: E402

KEEP, N_IMG, N_NEW = 61, 576, 3
MODES = ("prefill", "none", "text")


def under(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


def margs(peek):
    return argparse.Namespace(method="fastv", fastv_k=1, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=None,
                              fastv_stratify=None, text_rows_log=None, fastv_score="div", fastv_exclude_padding=False,
                              fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=None, fastv_visibility=None,
                              fastv_route=None, fastv_zoom=0, fastv_zoom_pick="grid", fastv_zoom_on="unrouted",
                              fastv_zoom_box="8,8,16", fastv_zoom_smin=0.0, fastv_cover=0, fastv_zoom_source="crop",
                              fastv_zoom_centre=False, fastv_cover_top=0, fastv_cls_unrouted=False, fastv_div_peek=peek)


def prompt_ids(after):
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    return torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)


def image(i):
    return torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))


def causal(n, like):
    return modeling_llama._prepare_4d_causal_attention_mask(None, (1, n), like, 0)


def manual_logits(ref, embeds, kept, mode, n_prompt):
    """The tiny model's layers by hand: layer 0 on the whole sequence with the mode's mask (generated tokens, after
    n_prompt, never see the dropped patches), then the system prompt, the kept patches, the text and the generated tokens
    at their original positions through the other layers; logits of the last position."""
    lm = ref.get_model()
    n = embeds.shape[1]
    pos = torch.arange(n).unsqueeze(0)
    drop = torch.ones(N_IMG, dtype=torch.bool)
    drop[kept] = False
    cols = SYS_LEN + drop.nonzero().squeeze(1)
    mask = causal(n, embeds).clone()
    rows = {"prefill": [], "none": list(range(n)), "text": list(range(SYS_LEN, SYS_LEN + N_IMG))}[mode] + list(range(n_prompt, n))
    if rows:
        r = torch.tensor(sorted(set(rows)))
        mask[:, :, r[:, None], cols[None, :]] = torch.finfo(mask.dtype).min
    h = lm.layers[0](embeds, attention_mask=mask, position_ids=pos)[0]
    keep_idx = torch.cat((torch.arange(SYS_LEN), SYS_LEN + kept, torch.arange(SYS_LEN + N_IMG, n)))
    h, pos = h[:, keep_idx], pos[:, keep_idx]
    mask = causal(h.shape[1], h)
    for layer in lm.layers[1:]:
        h = layer(h, attention_mask=mask, position_ids=pos)[0]
    return ref.lm_head(lm.norm(h))[0, -1]


def manual_generate(ref, embeds, kept, mode, n_new):
    """Greedy decoding without a cache, each step recomputed by manual_logits."""
    lm, out, e = ref.get_model(), [], embeds
    n_prompt = embeds.shape[1]
    for _ in range(n_new):
        tok = int(manual_logits(ref, e, kept, mode, n_prompt).argmax())
        out.append(tok)
        if tok == ref.config.eos_token_id:
            break
        e = torch.cat((e, lm.embed_tokens(torch.tensor([[tok]]))), dim=1)
    return out


def config_module():
    spec = importlib.util.spec_from_file_location("exp0002", f"{HERE}/exp0002.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    after = torch.arange(100, 100 + TEXT_LEN)
    ids = prompt_ids(after)
    checks = {}
    checks["this check imports the new divprune, fastv_llava and run_eval"] = all(
        under(m.__file__, HERE) for m in (divprune, fastv_llava, run_eval))

    models = {mode: run_eval.apply_method(build_tiny_llava(a.workdir), margs(mode), Pieces(after.tolist())) for mode in MODES}
    models["plain"] = run_eval.apply_method(build_tiny_llava(a.workdir), margs(None), Pieces(after.tolist()))
    ref = build_tiny_llava(a.workdir)                       # plain LLaVA, same weights; built after the rotary patch
    lens, logits, caches, gens = {}, {}, {}, {}
    for name, model in models.items():
        lm = model.get_model()
        lens[name] = []
        lm.layers[0].register_forward_pre_hook(lambda mod, args, kw, n=name: lens[n].append(("l0", args[0].shape[1])) if args[0].shape[1] > 1 else None, with_kwargs=True)
        lm.layers[1].register_forward_pre_hook(lambda mod, args, kw, n=name: lens[n].append(("l1", args[0].shape[1])) if args[0].shape[1] > 1 else None, with_kwargs=True)
        logits[name], caches[name], gens[name] = [], [], []
        with torch.no_grad():
            for i in range(2):
                out = model(input_ids=ids, images=image(i), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
                logits[name].append(out.logits[0, -1])
                pkv = out.past_key_values
                caches[name].append([pkv[l][0].shape[2] for l in range(len(pkv))])
                g = model.generate(ids, images=image(i), image_sizes=[(WIDTH, HEIGHT)], do_sample=False,
                                   max_new_tokens=N_NEW, use_cache=True)
                gens[name].append(g[0].tolist())
    with torch.no_grad():
        plain_feats = [type(ref).encode_images(ref, image(i))[0] for i in range(2)]
        ref_order = [divprune.maxmin_order(f, 64) for f in plain_feats]
        kept61 = [torch.tensor(sorted(o[:KEEP])) for o in ref_order]
        embeds = [ref.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, image(i), image_sizes=[(WIDTH, HEIGHT)])[4]
                  for i in range(2)]
        for mode in MODES:
            lm = models[mode].get_model()
            log = lm.fastv_div_log
            # the log holds one record per prefill: the two forwards and the two generate prefills
            checks[f"{mode}: one log record per prefill, the first 61 of DivPrune's order, with the mode"] = (
                len(log) == 4 and all(r["order"] == ref_order[[0, 0, 1, 1][j]][:KEEP] for j, r in enumerate(log))
                and all(r.get("peek") == mode for r in log))
            checks[f"{mode}: layer 0 sees the whole prompt, layer 1 on 61 patches (every prefill)"] = (
                [x for x in lens[mode] if x[0] == "l0"] == [("l0", SYS_LEN + N_IMG + TEXT_LEN)] * 4
                and [x for x in lens[mode] if x[0] == "l1"] == [("l1", SYS_LEN + KEEP + TEXT_LEN)] * 4)
            ref_l = [manual_logits(ref, embeds[i], kept61[i], mode, embeds[i].shape[1]) for i in range(2)]
            checks[f"{mode}: prefill logits = the hand-run reference (atol 1e-5)"] = all(
                torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(logits[mode], ref_l))
            checks[f"{mode}: the cache keeps every token at layer 0 (as plain K = 1) and the 515 dropped columns are recorded"] = (
                all(c[0] == SYS_LEN + N_IMG + TEXT_LEN and c[1:] == [SYS_LEN + KEEP + TEXT_LEN] * (len(c) - 1) for c in caches[mode])
                and lm.fastv_peek_drop is not None and lm.fastv_peek_drop.tolist() == [
                    SYS_LEN + j for j in range(N_IMG) if j not in set(kept61[1].tolist())])
            ref_g = [manual_generate(ref, embeds[i], kept61[i], mode, N_NEW) for i in range(2)]
            bos = ref.config.bos_token_id                      # generate from embeddings starts its output with BOS
            checks[f"{mode}: generate = the reference's greedy decoding without a cache"] = all(
                g[0] == bos and g[1:1 + len(r)] == r and len(r) >= 1 for g, r in zip(gens[mode], ref_g))
        checks["prefill: prefill logits = the plain K = 1 path with keep 61 (exactly)"] = all(
            torch.equal(x, y) for x, y in zip(logits["prefill"], logits["plain"]))
        checks["plain K = 1: layer 0 keeps every token in the cache"] = all(
            c[0] == SYS_LEN + N_IMG + TEXT_LEN and c[1] == SYS_LEN + KEEP + TEXT_LEN for c in caches["plain"])
        # "none" = DivPrune's 61 projector outputs handed to the LLM at their original positions
        lmr = ref.get_model()
        same = []
        for i in range(2):
            keep_idx = torch.cat((torch.arange(SYS_LEN), SYS_LEN + kept61[i], torch.arange(SYS_LEN + N_IMG, embeds[i].shape[1])))
            e, pos = embeds[i][:, keep_idx], keep_idx.unsqueeze(0)
            h = e
            mask = causal(h.shape[1], h)
            for layer in lmr.layers:
                h = layer(h, attention_mask=mask, position_ids=pos)[0]
            same.append(torch.allclose(ref.lm_head(lmr.norm(h))[0, -1], logits["none"][i], atol=1e-5, rtol=1e-4))
        checks["none: prefill logits = DivPrune's 61 handed to the LLM at their original positions"] = all(same)
        diff = lambda u, v: all(not torch.allclose(x, y, atol=1e-4) for x, y in zip(logits[u], logits[v]))   # noqa: E731
        checks["the three modes give different prefill logits"] = diff("prefill", "none") and diff("prefill", "text") and diff("none", "text")
        checks["generate runs for every mode and the plain path"] = all(len(g[0]) >= 1 for g in gens.values())

    # arguments, binding, configs
    k1 = ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "61", "--fastv-score", "div", "--fastv-div-log", "x.json"]
    checks["arguments: --fastv-div-peek prefill / none / text with div at K = 1 is accepted"] = all(
        not refused(k1 + ["--fastv-div-peek", m]) for m in MODES)
    checks["arguments: --fastv-div-peek with another mode name, without div, at K = 2 or with another method is refused"] = (
        refused(k1 + ["--fastv-div-peek", "all"])
        and refused(["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "61", "--fastv-div-peek", "none"])
        and refused(["--method", "fastv", "--fastv-k", "2", "--fastv-keep", "61", "--fastv-score", "div", "--fastv-div-log", "x.json",
                     "--fastv-div-peek", "none"])
        and refused(["--method", "divprune", "--divprune-keep", "61", "--divprune-log", "x.json", "--fastv-div-peek", "none"]))
    bad = []
    for kw in ({"score_kind": "last", "div_peek": "none"}, {"score_kind": "div", "div_peek": "all"}):
        try:
            fastv_llava.bind_fastv(build_tiny_llava(a.workdir).get_model(), 1, KEEP, **kw)
            bad.append(False)
        except ValueError:
            bad.append(True)
    try:
        fastv_llava.bind_fastv(build_tiny_llava(a.workdir).get_model(), 2, KEEP, score_kind="div", div_peek="none")
        bad.append(False)
    except ValueError:
        bad.append(True)
    checks["bind_fastv: div_peek without div, with an unknown mode or at K = 2 is refused"] = all(bad)
    e = config_module()
    C = e.CONFIGS
    base = ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "61", "--fastv-score", "div", "--fastv-div-peek"]
    checks["configs: fastv61k1_div_peek / nopeek / peektext and their _kll variants"] = all(
        C[f"fastv61k1_div_{n}"] == ("tp", base + [m]) and C[f"fastv61k1_div_{n}_kll"] == C[f"fastv61k1_div_{n}"]
        for n, m in (("peek", "prefill"), ("nopeek", "none"), ("peektext", "text")))
    src = inspect.getsource(e.run_one)
    checks["configs: run_one adds the div log and, for _kll, the kept, log-probability and layer logs"] = (
        '"--fastv-div-log", f"{run_dir}/div.json"' in src and 'config.endswith("_kll")' in src)
    checks["configs: each new config with the logs run_one adds parses"] = all(
        not refused(C[f"fastv61k1_div_{n}_kll"][1] + ["--fastv-div-log", "x.json", "--kept-token-log", "k.json",
                                                    "--answer-logit-log", "l.json", "--layer-token-log", "t.json"])
        for n in ("peek", "nopeek", "peektext"))
    checks["configs: the run record hashes .npy input tables (input_tables_sha256)"] = (
        '"input_tables_sha256": {p: sha256(p) for p in method_args if p.endswith(".npy") and os.path.isfile(p)}' in src)
    r = 2 * 4096 ** 2 / (4 * 4096 ** 2 + 3 * 4096 * 11008)
    checks["budget: with layer 0 computing only keys and values of the 515 dropped patches, 61 kept cost <= 64 per layer"] = (
        (32 * KEEP + (N_IMG - KEEP) * r) / 32 <= 64 and abs((576 + 31 * KEEP) / 32 - 77.09375) < 1e-9)

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_peek.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
