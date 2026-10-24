"""CPU check of EXP-0020's --fastv-route yesno_howmany (query_vis only for questions in yes/no form or asking "how many").

  python check_route.py --workdir D:/6CJNM007/tmp/tp_smoke

Tiny random LLaVA of check_object_cos.py (K = 2, keep 30, 6 x 6 blocks, padding excluded, a position prior, random
visibility): a routed prefill keeps what query_vis keeps, an unrouted one what the strong baseline keeps, also when the two
alternate on one model (no state carried over); the kept-token log records the decisions and an empty question part for
unrouted prefills. With LLaVA's tokenizer, the decision made from the prompt ids after the image equals the rule applied
to the raw question text on every question of the VQAv2 subsets, POPE and MME, and POPE and MME's ten perception subtasks
are routed entirely (so there the router is query_vis; some cognition questions start "The image shows ..."). Argument and
config checks.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp0002  # noqa: E402
import run_eval  # noqa: E402
from check_object_cos import BLOCKS, HEIGHT, KEEP, WIDTH, Pieces, refused  # noqa: E402
from smoke_test import SYS_LEN, build_tiny_llava  # noqa: E402

EVAL_DATA = os.environ.get("LLAVA_EVAL_DATA", "D:/6CJNM007/data/llava_eval")
QUESTION_FILES = {"vqav2_test": f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test.jsonl",
                  "vqav2_test2": f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_test2.jsonl",
                  "vqav2_dev": f"{EVAL_DATA}/vqav2_val/llava_vqav2_val_dev.jsonl",
                  "pope": f"{EVAL_DATA}/pope/llava_pope_test.jsonl", "mme": f"{EVAL_DATA}/MME/llava_mme.jsonl"}


def tiny_kept(workdir, after, idf, prior, vis_path, score, decisions):
    """Kept sets and logged records of one prefill per decision on one model (decision None: no router)."""
    model = build_tiny_llava(workdir)
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=prior,
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score=score, fastv_exclude_padding=True,
                           fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=idf,
                           fastv_visibility=vis_path if score == "query_vis" else None)
    model = run_eval.apply_method(model, m, Pieces(after.tolist()))
    lm = model.get_model()
    flag = [None]
    if decisions[0] is not None:
        lm.fastv_route = lambda ids: flag[0]              # the decision itself is tested below with the real tokenizer
    for store in (run_eval._KEPT, run_eval._KEPT_QUESTION, run_eval._ROUTED):
        store.clear()
    lm.register_forward_hook(run_eval._record_kept, with_kwargs=True)
    from llava.constants import IMAGE_TOKEN_INDEX
    torch.manual_seed(1)
    ids = torch.cat([torch.randint(3, 1000, (SYS_LEN,)), torch.tensor([IMAGE_TOKEN_INDEX]), after]).unsqueeze(0)
    images = torch.randn(1, 3, 336, 336)
    with torch.no_grad():
        for d in decisions:
            flag[0] = d
            model.generate(ids, images=images, image_sizes=[(WIDTH, HEIGHT)], do_sample=False, max_new_tokens=2, use_cache=True)
    return list(run_eval._KEPT), list(run_eval._KEPT_QUESTION), list(run_eval._ROUTED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    a = p.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    from smoke_test import TEXT_LEN
    after = torch.arange(100, 100 + TEXT_LEN)
    rng = np.random.default_rng(0)
    idf = {"idf": {int(t): float(rng.uniform(0, 3)) for t in after}, "unseen": 5.0}
    prior = f"{a.workdir}/prior_route.npy"
    np.save(prior, (torch.rand(576, generator=torch.Generator().manual_seed(3)) + 0.5).numpy())
    vis_path = f"{a.workdir}/vis_route.npy"
    np.save(vis_path, rng.uniform(0, 1, 1000).astype(np.float32))
    k_vis, q_vis, _ = tiny_kept(a.workdir, after, idf, prior, vis_path, "query_vis", [None])
    k_base, _, _ = tiny_kept(a.workdir, after, idf, prior, None, "last", [None])
    seq = [True, False, True, False, False, True]
    k_seq, q_seq, r_seq = tiny_kept(a.workdir, after, idf, prior, vis_path, "query_vis", seq)
    checks = {
        "query_vis and the strong baseline keep different sets here (the test can tell them apart)": k_vis[0] != k_base[0],
        "alternating decisions: routed prefills keep what query_vis keeps": all(k == k_vis[0] for k, d in zip(k_seq, seq) if d),
        "alternating decisions: unrouted prefills keep what the strong baseline keeps": all(k == k_base[0] for k, d in zip(k_seq, seq) if not d),
        "log: one decision per prefill, as made": r_seq == seq,
        "log: question part as query_vis's when routed, empty when not": all(
            q == (q_vis[0] if d else []) for q, d in zip(q_seq, seq)),
    }

    class Decoder:
        def decode(self, ids):
            return {1: "\nIs there a dog in the image?\nAnswer", 2: "\nWhat is the man holding?\nAnswer",
                    3: " \nHow many dogs are there?\nAnswer"}[ids[0]]
    m = argparse.Namespace(method="fastv", fastv_k=2, fastv_keep=KEEP, fastv_schedule=None, fastv_score_prior=prior,
                           fastv_stratify=BLOCKS, text_rows_log=None, fastv_score="query_vis", fastv_exclude_padding=True,
                           fastv_mix_question=None, fastv_contrast_ids=None, fastv_idf=idf, fastv_visibility=vis_path,
                           fastv_route="yesno_howmany")
    lm = run_eval.apply_method(build_tiny_llava(a.workdir), m, Decoder()).get_model()
    checks["apply_method wires the rule through tokenizer.decode"] = [lm.fastv_route([i]) for i in (1, 2, 3)] == [True, False, True]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    summary = {}
    for bench, path in QUESTION_FILES.items():
        qs = [json.loads(line) for line in open(path)]
        from_ids = [run_eval.routes_to_query(tok.decode(run_eval.prompt_ids_after_image(tok, q["text"], "vicuna_v1"))) for q in qs]
        from_text = [run_eval.routes_to_query(q["text"]) for q in qs]
        summary[bench] = {"n": len(qs), "routed": sum(from_ids), "agree": sum(x == y for x, y in zip(from_ids, from_text))}
        if bench == "mme":                                   # MME-P, the headline, is the ten perception subtasks
            cognition = ("commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning")
            perception = [r for q, r in zip(qs, from_ids) if q["category"] not in cognition]
            summary["mme_perception"] = {"n": len(perception), "routed": sum(perception)}
            summary["mme_unrouted_by_category"] = {}
            for q, r in zip(qs, from_ids):
                if not r:
                    summary["mme_unrouted_by_category"][q["category"]] = summary["mme_unrouted_by_category"].get(q["category"], 0) + 1
        checks[f"{bench}: decision from the prompt ids equals the rule on the question text ({len(qs)} questions)"] = from_ids == from_text
    checks["POPE: every question routed (router = query_vis there)"] = summary["pope"]["routed"] == summary["pope"]["n"]
    checks["MME perception subtasks: every question routed (router = query_vis on MME-P)"] = (
        summary["mme_perception"]["routed"] == summary["mme_perception"]["n"])
    checks.update({
        "--fastv-route refused without query_vis": refused(["--method", "fastv", "--fastv-route", "yesno_howmany"]),
        "--fastv-route accepted with query_vis, blocks, prior and padding exclusion": not refused([
            "--method", "fastv", "--fastv-score", "query_vis", "--fastv-visibility", "x", "--fastv-stratify", "6",
            "--fastv-score-prior", "x", "--fastv-exclude-padding", "--fastv-route", "yesno_howmany"]),
    })
    for b in (64, 128, 192):
        checks[f"exp0002 fastv{b}_qroute_kl = fastv{b}_qvis + the router"] = exp0002.CONFIGS[f"fastv{b}_qroute_kl"][1] == (
            exp0002.CONFIGS[f"fastv{b}_qvis"][1] + ["--fastv-route", "yesno_howmany"])
    print(json.dumps(summary))
    for name, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + name)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
