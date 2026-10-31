"""CPU check of EXP-0044's fill on VisionZip-64 (vzfill.py with vfill.py) on a tiny random LLaVA (smoke_test.build_tiny_llava).

  python -B check_vzfill.py --workdir D:/6CJNM007/tmp/exp0044/smoke_check_vzfill

VisionZip-64 (dominant 54: CLS + 53 patches; contextual 10), built as run_eval does (apply_method, then vzfill.apply and
vfill.apply_layout). Checks: with vzfill the LLM receives exactly VisionZip's 64 image tokens; the logged dominant patches
are VisionZip's own (CLS attention at layer -2, computed here) and the targets its contextual rule's (every step-th of
the other patches); each image token's slot is 0 (CLS), 1 + p (dominant patch p) or 1 + its target; every other patch
slot's representative is the most similar non-CLS token by the cosine of the projector on the patch's layer -2 state
against the 64 tokens (computed here); 577 slots, the expected positions, the layout in every layer, one record per
prefill; the prefill logits equal the hand-run reference (vfill.reference_logits with the 577 slots) and generate equals
its greedy decoding without a cache; the compact cache on top gives identical outputs; refusals.
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("VISIONZIP_ROOT", "D:/6CJNM007/idea_rearch/source/VisionZip")
import run_eval  # noqa: E402
import vfill  # noqa: E402
import vzfill  # noqa: E402
from smoke_test import SYS_LEN, TEXT_LEN, build_tiny_llava  # noqa: E402

WIDTH, HEIGHT = 640, 480
D, CTX, N_NEW, N_PATCH = 53, 10, 3, 576


def under(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


def build(workdir, fill, compact=False):
    m = argparse.Namespace(method="visionzip", dominant=D + 1, contextual=CTX)
    model = run_eval.apply_method(build_tiny_llava(workdir), m)
    log = {}
    if fill:
        vzfill.apply(model)
        vfill.apply_layout(model, "fill", log, compact=compact)
    embeds = []
    model.get_model().register_forward_pre_hook(
        lambda mod, args, kw: embeds.append(kw["inputs_embeds"].detach().clone())
        if kw.get("inputs_embeds") is not None and kw["inputs_embeds"].shape[1] > 1 else None, with_kwargs=True)
    return model, log, embeds


def prompt():
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    return torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), torch.arange(100, 100 + TEXT_LEN)]).unsqueeze(0)


def images(i):
    return torch.randn(1, 3, 336, 336, generator=torch.Generator().manual_seed(10 + i))


def run(model):
    ids = prompt()
    logits, gens = [], []
    with torch.no_grad():
        for i in range(2):
            out = model(input_ids=ids, images=images(i), image_sizes=[(WIDTH, HEIGHT)], use_cache=True)
            logits.append(out.logits[0, -1])
            gens.append(model.generate(ids, images=images(i), image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=N_NEW,
                                       use_cache=True)[0].tolist())
    return logits, gens


def own_layout(model, img, tokens):
    """VisionZip's dominant patches and targets for one image, and each slot's representative, computed here."""
    lm = model.get_model()
    tower = lm.get_vision_tower()
    with torch.no_grad():
        outs = tower.vision_tower(img.to(device=tower.device, dtype=tower.dtype), output_hidden_states=True, output_attentions=True)
        hs, attn = outs.hidden_states[-2], outs.attentions[-2]
        dom = sorted(attn[0, :, 0, 1:].sum(0).topk(D).indices.tolist())
        rest = [p for p in range(N_PATCH) if p not in set(dom)]
        targets = rest[::max(1, len(rest) // CTX)][:CTX]
        kept = [0] + [1 + p for p in dom] + [1 + t for t in targets]
        f = torch.nn.functional.normalize(lm.mm_projector(hs[:, 1:])[0].float(), dim=-1)
        g = torch.nn.functional.normalize(tokens.float(), dim=-1)
        rep = torch.cat((torch.zeros(1, dtype=torch.long), (f @ g[1:].T).argmax(1) + 1))
        rep[torch.tensor(kept)] = torch.arange(len(kept))
    return dom, targets, kept, rep


def manual_generate(ref, embeds, kept, rep):
    lm, out, e = ref.get_model(), [], embeds
    for _ in range(N_NEW):
        tok = int(vfill.reference_logits(ref, e, SYS_LEN, kept, rep, "fill").argmax())
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
    checks = {"this check imports the new run_eval, vfill and vzfill": all(under(m.__file__, HERE) for m in (run_eval, vfill, vzfill))}
    base, _, base_e = build(a.workdir, False)
    base_l, base_g = run(base)
    model, log, fill_e = build(a.workdir, True)
    logits, gens = run(model)
    n_layers = len(model.get_model().layers)
    recs = log["prefills"]
    checks["the LLM receives VisionZip's 64 image tokens unchanged (every prefill)"] = (
        len(base_e) == len(fill_e) == 4 and all(torch.equal(x, y) for x, y in zip(base_e, fill_e)) and base_e[0].shape[1] == SYS_LEN + 1 + D + CTX + TEXT_LEN)
    checks["one log record per prefill (4), mode fill, not compact"] = log["mode"] == "fill" and log["compact"] is False and len(recs) == 4
    checks["each record: 64 image tokens after the system prompt, 577 slots, the expected positions, the layout in every layer"] = all(
        r["n"] == 1 + D + CTX and r["slots"] == 1 + N_PATCH and r["start"] == SYS_LEN and r["positions_ok"] and r["layers"] == n_layers
        and r["len"] == SYS_LEN + 1 + D + CTX + TEXT_LEN for r in recs)
    ok_lay = ok_rep = ok_l = ok_g = True
    for i in range(2):
        e = base_e[2 * i]
        dom, targets, kept, rep = own_layout(base, images(i), e[0, SYS_LEN:SYS_LEN + 1 + D + CTX])
        for j in (2 * i, 2 * i + 1):
            r = recs[j]
            ok_lay = ok_lay and r["dominant"] == dom and r["targets"] == targets and r["kept"] == kept and len(set(kept)) == len(kept)
            ok_rep = ok_rep and torch.equal(torch.tensor(r["rep"]), rep)
        ok_l = ok_l and torch.allclose(logits[i], vfill.reference_logits(base, e, SYS_LEN, kept, rep, "fill"), atol=1e-5, rtol=1e-4)
        rg = manual_generate(base, e, kept, rep)
        ok_g = ok_g and gens[i][0] == base.config.bos_token_id and gens[i][1:1 + len(rg)] == rg and len(rg) >= 1
    checks["dominant patches = VisionZip's (CLS attention at layer -2), targets = its contextual rule's; slots 0, 1 + p, 1 + target"] = ok_lay
    checks["representatives: the most similar non-CLS token (projector on layer -2, cosine); own slots to themselves; CLS slot to CLS"] = ok_rep
    checks["prefill logits = the hand-run reference with 577 slots (atol 1e-5)"] = ok_l
    checks["generate = the reference's greedy decoding without a cache"] = ok_g
    checks["the fill changes the answer logits of VisionZip-64"] = all(not torch.allclose(x, y, atol=1e-4) for x, y in zip(logits, base_l))
    cm, clog, _ = build(a.workdir, True, compact=True)
    cl, cg = run(cm)
    checks["compact on VisionZip: identical prefill logits and generations, nothing kept beside the cache"] = (
        all(torch.equal(x, y) for x, y in zip(cl, logits)) and cg == gens and cm.get_model().layout_slots == {} and clog["compact"] is True)
    try:
        vzfill.apply(run_eval.apply_method(build_tiny_llava(a.workdir), argparse.Namespace(method="divprune", divprune_keep=64)))
        checks["vzfill without VisionZip is refused"] = False
    except ValueError:
        checks["vzfill without VisionZip is refused"] = True
    try:
        m2, _, _ = build(a.workdir, False)
        vzfill.apply(m2)
        vfill.apply_layout(m2, "orig")
        checks["vfill on VisionZip with a mode other than fill is refused"] = False
    except ValueError:
        checks["vfill on VisionZip with a mode other than fill is refused"] = True
    try:
        m3, _, _ = build(a.workdir, False)
        vzfill.apply(m3)
        vzfill.apply(m3)
        checks["vzfill twice is refused"] = False
    except ValueError:
        checks["vzfill twice is refused"] = True
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"{sum(bool(v) for v in checks.values())}/{len(checks)} checks pass")
    with open(f"{a.workdir}/check_vzfill.json", "w") as f:
        json.dump({k: bool(v) for k, v in checks.items()}, f, indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
