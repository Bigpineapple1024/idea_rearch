"""CPU checks of the CDPruner port (cdpruner.py) against the official code (source/CDPruner at 9541616c):

1. cdpp_select = the official encode_images lines 150-185, copied verbatim below as `official`, on random inputs (several
   seeds, keep values, one or two text segments): identical selections and orders.
2. text_embeds: for a short question (one segment), the zero-padded 77-token embedding equals CLIP's text tower on the
   unpadded tokens (the pooled EOS state does not see later tokens); a long text (> 77 tokens) gives one embedding per
   segment.
3. visual_projection and the text tower load from the local CLIP-L/336 snapshot (offline), float32.
4. The prompt-ids -> question map of every question file the port may run on has no collision with different texts.
5. run_eval.parse: valid cdpruner arguments parse; --method cdpruner without --cdpruner-keep, --cdpruner-keep 0 and
   --cdpruner-log with another method are refused; vanilla and divprune64 still parse; exp0002 has cdpruner64.
Imports only from this script's folder (the tmp copy before the commit, code/token_pruning after it). CPU, no GPU.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import cdpruner  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name)


def official(image_features, image_embeds, text_embeds, visual_token_num):
    # --- verbatim from source/CDPruner/llava/model/llava_arch.py lines 144-185 (encode_images, after the towers) ---
    B, N, C = image_features.shape
    device = image_features.device
    index_masks = torch.ones(B, N, dtype=torch.bool, device=device)

    # [CDPruner] Calculate cosine similarity
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
    image_normalized = image_normalized.float() # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)

    # [CDPruner] Calculate query relevance
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
    relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
    relevance = (-relevance).mean(dim=-1) # (B, N)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)

    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)

    # [CDPruner] Fast MAP inference of conditional DPP
    cis = torch.zeros((visual_token_num, B, N), device=device) # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
    select_idx = torch.empty((visual_token_num, B), dtype=torch.long, device=device) # (T, B)
    for i in range(visual_token_num):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')

    order = select_idx.t().clone()
    select_idx = torch.sort(select_idx.t()).values # (B, T)
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    index_masks.scatter_(1, select_idx, True)
    return select_idx, order, index_masks


def main():
    # 1. the selection
    # full-rank inputs (feature size 640 > 576 patches; the real projected features have 4096): distinct selections
    for seed in range(4):
        for keep in (16, 64, 128):
            for m in (1, 2):
                g = torch.Generator().manual_seed(seed)
                feats = torch.randn(1, 576, 640, generator=g)
                emb = torch.randn(1, 576, 48, generator=g)
                txt = torch.randn(m, 48, generator=g)
                a_kept, a_order = cdpruner.cdpp_select(feats, emb, txt, keep)
                b_kept, b_order, mask = official(feats, emb, txt, keep)
                check(f"selection = official (seed {seed}, keep {keep}, {m} text segment(s))",
                      torch.equal(a_kept, b_kept) and torch.equal(a_order, b_order) and int(mask.sum()) == keep and len(set(a_kept[0].tolist())) == keep)
    # rank-deficient kernel (feature size 96 < keep 128): the greedy MAP repeats indices after the rank is used up, in both
    g = torch.Generator().manual_seed(0)
    feats, emb, txt = torch.randn(1, 576, 96, generator=g), torch.randn(1, 576, 48, generator=g), torch.randn(1, 48, generator=g)
    a_kept, a_order = cdpruner.cdpp_select(feats, emb, txt, 128)
    b_kept, b_order, _ = official(feats, emb, txt, 128)
    check("rank-deficient input: the same (degenerate) selection as the official code", torch.equal(a_kept, b_kept) and torch.equal(a_order, b_order))
    # 3. the CLIP pieces, float32, from the local snapshot
    aux = cdpruner.load_aux("cpu")
    check("visual_projection 1024 -> 768, float32", tuple(aux["visual_projection"].weight.shape) == (768, 1024) and aux["visual_projection"].weight.dtype == torch.float32)
    check("text tower float32, 77 positions", next(aux["text_tower"].parameters()).dtype == torch.float32 and aux["text_tower"].config.max_position_embeddings == 77)
    # 2. text segments
    q = "What color is the umbrella on the left?"
    e = cdpruner.text_embeds(aux, q, "cpu")
    with torch.no_grad():
        ref = aux["text_tower"](**aux["tokenizer"](text=q, return_tensors="pt")).text_embeds
    check("short question: one segment, equal to the unpadded embedding", e.shape == (1, 768) and torch.allclose(e, ref, atol=1e-5))
    long_q = " ".join(["Is the small red object next to the large blue cylinder made of metal or rubber"] * 8)
    n_tok = aux["tokenizer"](text=long_q, return_tensors="pt").input_ids.shape[1]
    e2 = cdpruner.text_embeds(aux, long_q, "cpu")
    check(f"long text ({n_tok} tokens): one embedding per 77-token segment", e2.shape == ((n_tok - 1) // 77 + 1, 768) and n_tok > 77)
    # 4. prompt ids -> question text
    from transformers import AutoTokenizer
    import run_eval
    tok = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)
    EV = "D:/6CJNM007/data/llava_eval"
    for f in ("vqav2_val/llava_vqav2_val_dev2.jsonl", "gqa/llava_gqa_val400.jsonl", "MME/llava_mme.jsonl", "vstar/llava_vstar_test.jsonl",
              "pope/llava_pope_test.jsonl", "vqav2_val/llava_vqav2_val_test5.jsonl", "gqa/llava_gqa_val400c.jsonl"):
        texts = [json.loads(l)["text"] for l in open(f"{EV}/{f}", encoding="utf-8")]
        mp = {}
        bad = 0
        for t in texts:
            k = tuple(run_eval.prompt_ids_after_image(tok, t, "vicuna_v1"))
            bad += k in mp and mp[k] != t
            mp[k] = t
        check(f"{f}: {len(texts)} questions, {len(mp)} distinct prompts, no collision with a different text", bad == 0)
    # 5. argument parsing (run_eval.parse) and the exp0002 configuration
    import exp0002
    tail = ["--", "--model-path", "liuhaotian/llava-v1.5-7b", "--question-file", "q.jsonl", "--image-folder", ".", "--answers-file", "a.jsonl",
            "--conv-mode", "vicuna_v1"]

    def try_parse(args):
        argv = sys.argv
        sys.argv = ["run_eval.py"] + args + tail
        try:
            return run_eval.parse()[0]
        except SystemExit:
            return None
        finally:
            sys.argv = argv
    m = try_parse(["--method", "cdpruner", "--cdpruner-keep", "64", "--cdpruner-log", "c.json"])
    check("valid cdpruner arguments parse", m is not None and m.cdpruner_keep == 64 and m.question_file == "q.jsonl")
    check("--method cdpruner without --cdpruner-keep is refused", try_parse(["--method", "cdpruner"]) is None)
    check("--cdpruner-keep 0 is refused", try_parse(["--method", "cdpruner", "--cdpruner-keep", "0"]) is None)
    check("--cdpruner-log with another method is refused", try_parse(["--method", "vanilla", "--cdpruner-log", "c.json"]) is None)
    check("the vanilla arguments still parse", try_parse(["--method", "vanilla"]) is not None)
    check("divprune64 still parses (with the log exp0002 adds)", try_parse(exp0002.CONFIGS["divprune64"][1] + ["--divprune-log", "d.json"]) is not None)
    check("exp0002 has cdpruner64 = --method cdpruner --cdpruner-keep 64",
          exp0002.CONFIGS["cdpruner64"] == ("tp", ["--method", "cdpruner", "--cdpruner-keep", "64"]))
    n_pass = sum(ok for _, ok in RESULTS)
    print(f"{n_pass}/{len(RESULTS)} checks pass")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
