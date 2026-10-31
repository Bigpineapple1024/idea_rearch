"""CPU check of IDEA-0011's --layout (vfill.py) on a tiny random LLaVA (smoke_test.build_tiny_llava, 6 layers).

  python -B check_vfill.py --workdir D:/6CJNM007/tmp/exp0041/smoke_check_vfill

Models: DivPrune keep 20 before the LLM with each layout mode (as run_eval's load_and_patch does: apply_method, then
vfill.apply_layout); DivPrune keep 576 (nothing pruned) with each mode; plain vanilla. References are run by hand on a
plain DivPrune LLaVA with the same weights (vfill.reference_logits: eager arithmetic, float32 softmax, the pruned prompt's
embeddings at the mode's positions; in fill and shuffle the keys and values of the 576 slots are the representatives',
rotated to the slots' positions, and a query sees the keys at positions <= its own; in mass each kept key gets
+ log(count)). Checks: the prefill logits equal the reference; generate equals the reference's greedy decoding without a
cache; with nothing pruned every mode equals vanilla; the logged assignment is the most similar kept token (cosine of the
projected features) with the kept slots mapped to themselves, and shuffle's is a permutation of it over the pruned slots
(same counts); the positions the layers saw are the expected ones; one log record per prefill with the layers that used
the layout and the decoding steps; argument and config checks.
EXP-0044, the compact cache (fill and shuffle): its prefill logits and generations are identical (torch.equal) to the
plain implementation's, nothing is kept beside the KV cache, the cache's image part holds the keys before RoPE, and the
cache has the base's length. VisionZip's layout (vzfill.py) is checked by check_vzfill.py in its own process (VisionZip
patches CLIP's classes).
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
import run_eval  # noqa: E402
import vfill  # noqa: E402
from check_object_cos import HEIGHT, WIDTH, refused  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

KEEP, N_IMG, N_NEW = 20, 576, 3
MODES = vfill.MODES


def margs(keep):
    return argparse.Namespace(method="divprune", divprune_keep=keep)


def build(workdir, keep, mode, compact=False):
    model = run_eval.apply_method(build_tiny_llava(workdir), margs(keep))
    log = {}
    if mode is not None:
        vfill.apply_layout(model, mode, log, seed=0, compact=compact)
    return model, log


def run(model):
    ids = cp.prompt_ids(torch.arange(100, 100 + TEXT_LEN))
    logits, gens = [], []
    with torch.no_grad():
        for i in range(2):
            out = model(input_ids=ids, images=cp.image(i), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
            logits.append(out.logits[0, -1])
            g = model.generate(ids, images=cp.image(i), image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=N_NEW, use_cache=True)
            gens.append(g[0].tolist())
    return ids, logits, gens


def manual_generate(ref, embeds, kept, rep, mode, counts):
    lm, out, e = ref.get_model(), [], embeds
    for _ in range(N_NEW):
        tok = int(vfill.reference_logits(ref, e, SYS_LEN, kept, rep, mode, counts).argmax())
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
    checks = {"this check imports the new vfill, divprune, run_eval and check_peek": all(
        cp.under(m.__file__, HERE) for m in (vfill, divprune, run_eval, cp))}
    ref_log = []
    ref = divprune.apply_divprune(build_tiny_llava(a.workdir), KEEP, ref_log)
    results = {}
    for mode in MODES:
        model, log = build(a.workdir, KEEP, mode)
        ids, logits, gens = run(model)
        results[mode] = (logits, gens, log)
        recs = log["prefills"]
        checks[f"{mode}: one log record per prefill (4: two forward passes, two generates)"] = log["mode"] == mode and len(recs) == 4
        checks[f"{mode}: the layers saw the expected positions"] = all(r.get("positions_ok") for r in recs)
        want_layers = 0 if mode == "orig" else len(model.get_model().layers)
        checks[f"{mode}: every layer used the layout in every prefill ({want_layers})"] = all(r["layers"] == want_layers for r in recs)
        ok_l = ok_g = ok_rep = True
        for i in range(2):
            with torch.no_grad():
                e = ref.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, cp.image(i), image_sizes=[(WIDTH, HEIGHT)])[4]
            feats, kept = ref.prune_last["features"], ref.prune_last["kept"]
            plain_rep, _ = vfill.assignment(feats, kept)
            for j in (2 * i, 2 * i + 1):                      # this image's forward pass and its generate
                rep = torch.tensor(recs[j]["rep"])
                kept_t = torch.tensor(kept)
                counts = torch.bincount(rep, minlength=len(kept))
                if mode == "shuffle":
                    pruned = torch.ones(N_IMG, dtype=torch.bool)
                    pruned[kept_t] = False
                    ok_rep = ok_rep and torch.equal(rep[kept_t], torch.arange(len(kept))) and torch.equal(
                        rep[pruned].sort().values, plain_rep[pruned].sort().values) and not torch.equal(rep, plain_rep)
                else:
                    ok_rep = ok_rep and torch.equal(rep, plain_rep)
                ok_rep = ok_rep and recs[j]["n"] == KEEP and recs[j]["slots"] == N_IMG and recs[j]["start"] == SYS_LEN and (
                    recs[j]["len"] == SYS_LEN + KEEP + TEXT_LEN)
                if j == 2 * i:
                    r = vfill.reference_logits(ref, e, SYS_LEN, kept, rep, mode, counts)
                    ok_l = ok_l and torch.allclose(logits[i], r, atol=1e-5, rtol=1e-4)
                else:
                    rg = manual_generate(ref, e, kept, rep, mode, counts)
                    g = gens[i]
                    ok_g = ok_g and g[0] == ref.config.bos_token_id and g[1:1 + len(rg)] == rg and len(rg) >= 1
                    steps = recs[j]["decode_steps"]
                    ok_g = ok_g and (steps == len(rg) - 1 if mode != "orig" else steps == 0)
            checks[f"{mode}: forward passes have no decoding steps"] = all(recs[k]["decode_steps"] == 0 for k in (0, 2))
        checks[f"{mode}: the assignment is the most similar kept token, kept slots to themselves"
               + (" (shuffle: permuted over the pruned slots, same counts)" if mode == "shuffle" else "")
               + "; span and prompt length logged"] = ok_rep
        checks[f"{mode}: prefill logits = the hand-run reference (atol 1e-5)"] = ok_l
        checks[f"{mode}: generate = the reference's greedy decoding without a cache; decoding steps logged"] = ok_g
    # the base itself, and the layout against it
    base, _ = build(a.workdir, KEEP, None)
    _, base_l, _ = run(base)
    with torch.no_grad():
        e0 = ref.prepare_inputs_labels_for_multimodal(cp.prompt_ids(torch.arange(100, 100 + TEXT_LEN)), None, None, None, None,
                                                      cp.image(0), image_sizes=[(WIDTH, HEIGHT)])[4]
    kept0 = ref.prune_last["kept"]
    checks["mass with every count 1 = the base (reference)"] = torch.allclose(
        vfill.reference_logits(ref, e0, SYS_LEN, kept0, torch.zeros(N_IMG, dtype=torch.long), "mass", torch.ones(KEEP)),
        base_l[0], atol=1e-5, rtol=1e-4)
    checks["every mode changes the answer logits of the base"] = all(
        not torch.allclose(results[m][0][0], base_l[0], atol=1e-4) for m in MODES)
    # nothing pruned: every mode is vanilla
    van = build_tiny_llava(a.workdir)
    _, van_l, van_g = run(van)
    for mode in MODES:
        model, log = build(a.workdir, N_IMG, mode)
        _, l, g = run(model)
        checks[f"keep 576, {mode}: = vanilla (logits atol 1e-5, generate)"] = all(
            torch.allclose(x, y, atol=1e-5, rtol=1e-4) for x, y in zip(l, van_l)) and g == van_g
    # EXP-0044: the compact cache gives the same outputs, keeps nothing beside the cache, and the cache's image part holds
    # the keys before RoPE (rotated at the image tokens' positions they are the plain implementation's cached keys)
    for mode in ("fill", "shuffle"):
        model, log = build(a.workdir, KEEP, mode, compact=True)
        ids, logits, gens = run(model)
        checks[f"compact {mode}: prefill logits identical to the plain implementation's (torch.equal)"] = all(
            torch.equal(x, y) for x, y in zip(logits, results[mode][0]))
        checks[f"compact {mode}: generations identical to the plain implementation's"] = gens == results[mode][1]
        checks[f"compact {mode}: nothing kept beside the KV cache; logged as compact"] = (
            model.get_model().layout_slots == {} and log["compact"] is True and log["mode"] == mode and len(log["prefills"]) == 4)
        cm, _ = build(a.workdir, KEEP, mode, compact=True)
        pm, _ = build(a.workdir, KEEP, mode)
        with torch.no_grad():
            oc = cm(input_ids=ids, images=cp.image(0), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
            op = pm(input_ids=ids, images=cp.image(0), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
        ctx = cm.get_model().layout_ctx
        start, n = ctx["start"], ctx["n"]
        pos = start + ctx["kept"]
        ok = torch.equal(oc.logits, op.logits)
        for li, layer in enumerate(cm.get_model().layers):
            kc, vc = oc.past_key_values[li]
            kq, vq = op.past_key_values[li]
            c = layer.self_attn.rotary_emb.cos_cached.to(kc.dtype)[pos][None, None]
            s = layer.self_attn.rotary_emb.sin_cached.to(kc.dtype)[pos][None, None]
            img = kc[:, :, start:start + n]
            ok = ok and kc.shape == kq.shape and kc.shape[2] == SYS_LEN + KEEP + TEXT_LEN and torch.equal(vc, vq) and (
                torch.equal(kc[:, :, :start], kq[:, :, :start]) and torch.equal(kc[:, :, start + n:], kq[:, :, start + n:])
                and torch.equal(img * c + vfill.rotate_half(img) * s, kq[:, :, start:start + n])
                and not torch.equal(img, kq[:, :, start:start + n]))
        checks[f"compact {mode}: the cache has the base's length, its image part the keys before RoPE, the rest as plain"] = ok
    try:
        build(a.workdir, KEEP, "orig", compact=True)
        checks["compact with orig is refused"] = False
    except ValueError:
        checks["compact with orig is refused"] = True
    # positions
    pid = torch.arange(SYS_LEN + KEEP + 4).unsqueeze(0)
    kept = list(range(0, 2 * KEEP, 2))
    got = vfill.remap_positions(pid, SYS_LEN, kept)[0]
    checks["remap_positions: system prompt kept, image tokens to start + patch, text after + (576 - n)"] = torch.equal(
        got, torch.cat((torch.arange(SYS_LEN), SYS_LEN + torch.tensor(kept), SYS_LEN + N_IMG + torch.arange(4))))
    checks["remap_positions: a decoding step continues after the unpruned prompt"] = int(
        vfill.remap_positions(torch.tensor([[SYS_LEN + KEEP + 7]]), SYS_LEN, kept)[0, 0]) == SYS_LEN + N_IMG + 7
    # refusals
    try:
        m, _ = build(a.workdir, KEEP, "fill")
        vfill.apply_layout(m, "orig")
        checks["apply_layout twice is refused"] = False
    except ValueError:
        checks["apply_layout twice is refused"] = True
    try:
        vfill.apply_layout(build_tiny_llava(a.workdir), "fill")
        checks["apply_layout without a selection before the LLM is refused"] = False
    except ValueError:
        checks["apply_layout without a selection before the LLM is refused"] = True
    try:
        vfill.apply_layout(run_eval.apply_method(build_tiny_llava(a.workdir), margs(KEEP)), "gaps")
        checks["an unknown mode is refused"] = False
    except ValueError:
        checks["an unknown mode is refused"] = True
    # arguments and configs
    ok_args = all(not refused(x + ["--layout", m, "--layout-log", "l.json"]) for m in MODES for x in (
        ["--method", "divprune", "--divprune-keep", "64", "--divprune-log", "d.json"], ["--method", "cdpruner", "--cdpruner-keep", "64"]))
    checks["arguments: --layout with divprune or cdpruner is accepted (with --layout-log, any mode)"] = ok_args
    checks["arguments: --layout with vanilla or fastv, with --freeze-layers, a bad mode, or --layout-log alone is refused"] = (
        refused(["--method", "vanilla", "--layout", "fill"])
        and refused(["--method", "fastv", "--fastv-k", "2", "--fastv-keep", "30", "--layout", "fill"])
        and refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout", "fill", "--freeze-layers", "0,2"])
        and refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout", "gaps"])
        and refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout-log", "l.json"]))
    C = cp.config_module().CONFIGS
    want = {f"{b}64_{m}_logits": ("tp", ["--method", b, f"--{b}-keep", "64", "--layout", m]) for b in ("divprune", "cdpruner") for m in MODES}
    checks["configs: the eight IDEA-0011 configurations"] = all(C.get(k) == v for k, v in want.items())
    checks["configs: each with the logs run_one adds parses"] = all(
        not refused(v[1] + [f"--{v[1][1]}-log", "x.json", "--answer-logit-log", "l.json", "--layout-log", "y.json"]) for v in want.values())
    checks["arguments: --layout-compact with fill or shuffle is accepted; with orig, mass or without --layout refused"] = (
        not refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout", "fill", "--layout-compact"])
        and not refused(["--method", "divprune", "--divprune-keep", "64", "--divprune-log", "d.json", "--layout", "shuffle", "--layout-compact"])
        and refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout", "orig", "--layout-compact"])
        and refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout", "mass", "--layout-compact"])
        and refused(["--method", "cdpruner", "--cdpruner-keep", "64", "--layout-compact"]))
    checks["arguments: --layout fill with plain visionzip is accepted; another mode or with --vz-fastv refused"] = (
        not refused(["--method", "visionzip", "--dominant", "54", "--contextual", "10", "--layout", "fill"])
        and refused(["--method", "visionzip", "--dominant", "54", "--contextual", "10", "--layout", "orig"])
        and refused(["--method", "visionzip", "--dominant", "108", "--contextual", "20", "--vz-fastv", "2,60", "--layout", "fill"]))
    vz, fc = C.get("visionzip64_fill_logits"), C.get("cdpruner64_fillc_logits")
    checks["configs: EXP-0044's visionzip64_fill_logits and cdpruner64_fillc_logits, with the logs run_one adds"] = (
        vz == ("tp", ["--method", "visionzip", "--dominant", "54", "--contextual", "10", "--layout", "fill"])
        and fc == ("tp", ["--method", "cdpruner", "--cdpruner-keep", "64", "--layout", "fill", "--layout-compact"])
        and not refused(vz[1] + ["--answer-logit-log", "l.json", "--layout-log", "y.json"])
        and not refused(fc[1] + ["--cdpruner-log", "x.json", "--answer-logit-log", "l.json", "--layout-log", "y.json"]))
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_vfill.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
