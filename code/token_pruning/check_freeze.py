"""CPU check of IDEA-0010's --freeze-layers (freeze.py) on a tiny random LLaVA (smoke_test.build_tiny_llava, 6 layers).

  python -B check_freeze.py --workdir D:/6CJNM007/tmp/exp0040/smoke

Models: DivPrune keep 20 before the LLM and vanilla (576 image tokens), each with the image tokens frozen in layers 0, 2
and 3 (as run_eval's load_and_patch does: apply_method, then freeze.apply_freeze), and DivPrune keep 20 without freezing.
References are run by hand on plain LLaVAs with the same weights: a frozen layer is computed with the non-image rows as the
only queries (freeze.frozen_text_rows: keys and values from every row's input, the image rows unchanged), the other layers
as usual. Checks: frozen_text_rows equals the full layer on the non-image rows (so computing a frozen layer for every row
and putting the image rows back is the same); in the frozen models the image rows leave each frozen layer exactly as they
entered it and change in the other layers; the prefill logits equal the reference; generate equals the reference's greedy
decoding without a cache; without freezing the logits equal the plain reference; one log record per prefill with the image
span and three restorations; parse_layers and ShortV's list; argument and config checks.
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
import freeze  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import HEIGHT, WIDTH, refused  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

KEEP, N_IMG, N_NEW, FROZEN = 20, 576, 3, [0, 2, 3]


def margs(method):
    return argparse.Namespace(method=method, divprune_keep=KEEP if method == "divprune" else None)


def manual_logits(ref, embeds, n_img, frozen):
    """The tiny model's layers by hand; frozen layers with the non-image rows as the only queries."""
    lm = ref.get_model()
    n = embeds.shape[1]
    pos = torch.arange(n).unsqueeze(0)
    mask = cp.causal(n, embeds)
    h = embeds
    for i, layer in enumerate(lm.layers):
        h = freeze.frozen_text_rows(layer, h, SYS_LEN, n_img, mask, pos) if i in frozen else layer(h, attention_mask=mask, position_ids=pos)[0]
    return ref.lm_head(lm.norm(h))[0, -1]


def manual_generate(ref, embeds, n_img, frozen, n_new):
    lm, out, e = ref.get_model(), [], embeds
    for _ in range(n_new):
        tok = int(manual_logits(ref, e, n_img, frozen).argmax())
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
    checks = {"this check imports the new freeze, divprune, run_eval and check_peek": all(
        cp.under(m.__file__, HERE) for m in (freeze, divprune, run_eval, cp))}
    spec = {"div_f": ("divprune", FROZEN), "van_f": ("vanilla", FROZEN), "div": ("divprune", None)}
    models, logs, rows = {}, {}, {}
    for name, (method, frozen) in spec.items():
        model = run_eval.apply_method(build_tiny_llava(a.workdir), margs(method))
        logs[name] = {}
        if frozen:
            freeze.apply_freeze(model, frozen, logs[name])
        else:                                      # the span only, for the row checks (no change to the computation)
            fastv_llava._record_image_span(model)
        rows[name] = []
        lm = model.get_model()
        for i, layer in enumerate(lm.layers):      # registered after freeze's hooks: they see the restored output
            def pre(mod, args, kw, i=i, name=name):
                if args[0].shape[1] > 1:
                    s, n = lm_span(models[name])
                    rows[name].append((i, "in", args[0][0, s:s + n].clone()))

            def post(mod, args, kw, out, i=i, name=name):
                if args[0].shape[1] > 1:
                    s, n = lm_span(models[name])
                    rows[name].append((i, "out", (out[0] if isinstance(out, tuple) else out)[0, s:s + n].clone()))
            layer.register_forward_pre_hook(pre, with_kwargs=True)
            layer.register_forward_hook(post, with_kwargs=True)
        models[name] = model
    logits, gens = {}, {}
    with torch.no_grad():
        for name, model in models.items():
            logits[name], gens[name] = [], []
            for i in range(2):
                out = model(input_ids=ids, images=cp.image(i), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
                logits[name].append(out.logits[0, -1])
                g = model.generate(ids, images=cp.image(i), image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=N_NEW, use_cache=True)
                gens[name].append(g[0].tolist())
        refs = {"divprune": divprune.apply_divprune(build_tiny_llava(a.workdir), KEEP), "vanilla": build_tiny_llava(a.workdir)}
        for name, (method, frozen) in spec.items():
            ref = refs[method]
            n_img = KEEP if method == "divprune" else N_IMG
            embeds = [ref.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, cp.image(i), image_sizes=[(WIDTH, HEIGHT)])[4] for i in range(2)]
            checks[f"{name}: the prompt has {n_img} image tokens after the {SYS_LEN}-token system prompt"] = all(
                e.shape[1] == SYS_LEN + n_img + TEXT_LEN for e in embeds) and lm_span(models[name]) == (SYS_LEN, n_img)
            fz = frozen or []
            ref_l = [manual_logits(ref, e, n_img, fz) for e in embeds]
            checks[f"{name}: prefill logits = the hand-run reference (atol 1e-5)"] = all(
                torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(logits[name], ref_l))
            ref_g = [manual_generate(ref, e, n_img, fz, N_NEW) for e in embeds]
            bos = ref.config.bos_token_id
            checks[f"{name}: generate = the reference's greedy decoding without a cache"] = all(
                g[0] == bos and g[1:1 + len(r)] == r and len(r) >= 1 for g, r in zip(gens[name], ref_g))
            pairs = {}
            for layer_i, kind, t in rows[name]:
                pairs.setdefault(layer_i, {"in": [], "out": []})[kind].append(t)
            same = {i: all(torch.equal(x, y) for x, y in zip(v["in"], v["out"])) for i, v in pairs.items()}
            n_pref = len(pairs[0]["in"])
            checks[f"{name}: the image rows leave the frozen layers {fz} unchanged and change in the others ({n_pref} prefills)"] = (
                n_pref == 4 and all(same[i] == (i in fz) for i in range(len(models[name].get_model().layers))))
            if frozen:
                lg = logs[name]
                checks[f"{name}: one log record per prefill with the span and {len(fz)} restorations"] = (
                    lg["layers"] == fz and len(lg["prefills"]) == 4
                    and all(r == {"start": SYS_LEN, "n_img": n_img, "len": SYS_LEN + n_img + TEXT_LEN, "restored": len(fz)} for r in lg["prefills"]))
        # frozen_text_rows against the full layer on the non-image rows (the hand-run reference's building block)
        ref = refs["divprune"]
        e = ref.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, cp.image(0), image_sizes=[(WIDTH, HEIGHT)])[4]
        n = e.shape[1]
        pos, mask = torch.arange(n).unsqueeze(0), cp.causal(n, e)
        h, ok = e, True
        rows_q = torch.cat((torch.arange(SYS_LEN), torch.arange(SYS_LEN + KEEP, n)))
        for layer in ref.get_model().layers:
            full = layer(h, attention_mask=mask, position_ids=pos)[0]
            mine = freeze.frozen_text_rows(layer, h, SYS_LEN, KEEP, mask, pos)
            ok = ok and torch.allclose(mine[:, rows_q], full[:, rows_q], atol=1e-5, rtol=1e-4) and torch.equal(mine[:, SYS_LEN:SYS_LEN + KEEP], h[:, SYS_LEN:SYS_LEN + KEEP])
            h = full
        checks["frozen_text_rows = the full layer on the non-image rows, image rows unchanged (every layer)"] = ok
        checks["div_f differs from div (freezing changes the answer logits)"] = all(not torch.allclose(x, y, atol=1e-4) for x, y in zip(logits["div_f"], logits["div"]))
    checks["parse_layers: ShortV's spec gives its 19 layers"] = (
        freeze.parse_layers(freeze.SHORTV_SPEC, 32) == sorted(freeze.SHORTV_LLAVA15_7B) and len(freeze.SHORTV_LLAVA15_7B) == 19
        and freeze.parse_layers("1, 4-6", 32) == [1, 4, 5, 6])
    bad = []
    for s in ("1,1", "32", "5-3", "", "-1", "a"):
        try:
            freeze.parse_layers(s, 32)
            bad.append(s)
        except ValueError:
            pass
    checks["parse_layers: repeats, out of range, reversed ranges and junk are refused"] = not bad
    try:
        m = run_eval.apply_method(build_tiny_llava(a.workdir), margs("divprune"))
        freeze.apply_freeze(m, [1])
        freeze.apply_freeze(m, [2])
        checks["apply_freeze twice is refused"] = False
    except ValueError:
        checks["apply_freeze twice is refused"] = True
    fl = ["--freeze-layers", freeze.SHORTV_SPEC]
    checks["arguments: --freeze-layers with divprune, cdpruner or vanilla is accepted (with --freeze-log)"] = all(
        not refused(x + fl + ["--freeze-log", "f.json"]) for x in (["--method", "divprune", "--divprune-keep", "127", "--divprune-log", "d.json"],
                                                                    ["--method", "cdpruner", "--cdpruner-keep", "127"], ["--method", "vanilla"]))
    checks["arguments: --freeze-layers with fastv, a bad spec, or --freeze-log alone is refused"] = (
        refused(["--method", "fastv", "--fastv-k", "2", "--fastv-keep", "30"] + fl)
        and refused(["--method", "divprune", "--divprune-keep", "127", "--freeze-layers", "0,40"])
        and refused(["--method", "divprune", "--divprune-keep", "127", "--freeze-log", "f.json"]))
    C = cp.config_module().CONFIGS
    want = {}
    for b in ("divprune", "cdpruner"):
        want[f"{b}127_logits"] = ("tp", ["--method", b, f"--{b}-keep", "127"])
        want[f"{b}127_f19_logits"] = ("tp", ["--method", b, f"--{b}-keep", "127", "--freeze-layers", "0,2,3,12,17-31"])
        want[f"{b}64_f19_logits"] = ("tp", ["--method", b, f"--{b}-keep", "64", "--freeze-layers", "0,2,3,12,17-31"])
    checks["configs: the six IDEA-0010 configurations"] = all(C.get(k) == v for k, v in want.items())
    checks["configs: each with the logs run_one adds parses"] = all(
        not refused(v[1] + [f"--{v[1][1]}-log", "x.json", "--answer-logit-log", "l.json"] + (["--freeze-log", "f.json"] if "--freeze-layers" in v[1] else []))
        for v in want.values())
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_freeze.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


def lm_span(model):
    return model.get_model().fastv_image_span


if __name__ == "__main__":
    main()
