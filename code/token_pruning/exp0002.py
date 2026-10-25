"""Run the vision-direction evaluations (EXP-0002 onwards) with the per-run records the research process requires.

  python exp0002.py EXP-0002-R01:vanilla:pope EXP-0002-R02:vanilla:mme ...    # in the tp env, sequentially

The experiment is the run ID without its -Rnn suffix. Each run gets code/output/<EXP>/<run>/ with cmd.txt,
config.json, git.txt (this repo + external commits + model revisions), seed.txt, env.txt, gpu.txt, stdout.log,
answers.jsonl(+.meta.json), score.log, metrics.json, status.json and SHA256SUMS, and is copied to
D:/6CJNM007/backup/<EXP>/<run>/ (checksums re-verified); code/output/<EXP>/runs.tsv gets one row per run.
A run ID is never reused: a failed run keeps its folder and the retry gets a new ID.
"""
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = f"{REPO}/source"
BASH = "D:/6CJNM007/git/bin/bash.exe"
ENV_SH = "/d/6CJNM007/idea_rearch/code/env/windows_env.sh"
GIT = "D:/6CJNM007/git/cmd/git.exe"
HF_HUB = "D:/6CJNM007/cache/huggingface/hub"
DATA = "D:/6CJNM007/data"
BACKUP = "D:/6CJNM007/backup"
MODEL = "liuhaotian/llava-v1.5-7b"

BENCHES = {
    "pope": {"question_file": f"{DATA}/llava_eval/pope/llava_pope_test.jsonl", "image_folder": f"{DATA}/coco/val2014",
             "annotations": [f"{DATA}/llava_eval/pope/coco/coco_pope_{s}.json" for s in ("adversarial", "popular", "random")]},
    "mme": {"question_file": f"{DATA}/llava_eval/MME/llava_mme.jsonl",
            "image_folder": f"{DATA}/llava_eval/MME/MME_Benchmark_release_version", "annotations": []},
    # diagnostic only: 50 POPE questions spread evenly over the file (17/16/17 per split), not scored
    "pope50": {"question_file": f"{DATA}/llava_eval/pope/llava_pope_test_sub50.jsonl",
               "image_folder": f"{DATA}/coco/val2014", "annotations": [], "score": False},
    # diagnostic only: every question (all three splits) about 20 images, from make_pope_subset.py; not scored
    "pope_img20": {"question_file": f"{DATA}/llava_eval/pope/llava_pope_test_img20.jsonl",
                   "image_folder": f"{DATA}/coco/val2014", "annotations": [], "score": False},
    # diagnostic only: the pope_img20 questions on one blank grey image (make_blank_questions.py); not scored
    "pope_img20_blank": {"question_file": f"{DATA}/llava_eval/pope/llava_pope_test_img20_blank.jsonl",
                         "image_folder": f"{DATA}/blank", "annotations": [], "score": False},
    # selection set for EXP-0011: POPE-style questions on 300 COCO val2014 images POPE does not use (make_coco_dev.py);
    # choices are made here so POPE stays a test set; not scored by score_pope_mme.py
    "coco_dev": {"question_file": f"{DATA}/llava_eval/pope/llava_coco_dev.jsonl",
                 "image_folder": f"{DATA}/coco/val2014", "annotations": [], "score": False},
    # selection set for EXP-0013: coco_dev plus POPE's popular and adversarial negatives on the same 300 images
    # (make_coco_dev3.py); scored offline by analyze_fp.py, not by score_pope_mme.py
    "coco_dev3": {"question_file": f"{DATA}/llava_eval/pope/llava_coco_dev3.jsonl",
                  "image_folder": f"{DATA}/coco/val2014", "annotations": [], "score": False},
    # EXP-0018: VQAv2 val subsets on local COCO val2014 images (make_vqav2_sets.py), scored by score_pope_mme.py; the
    # VQAv2 archives' checksums are in downloads/vqav2/SHA256SUMS, the extracted JSONs' in vqav2/SHA256SUMS
    "vqav2_test": {"question_file": f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_test.jsonl",
                   "image_folder": f"{DATA}/coco/val2014",
                   "annotations": [f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_test_annotations.jsonl"],
                   "archives_sha256": f"{DATA}/downloads/vqav2/SHA256SUMS"},
    "vqav2_dev": {"question_file": f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_dev.jsonl",
                  "image_folder": f"{DATA}/coco/val2014",
                  "annotations": [f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_dev_annotations.jsonl"],
                  "archives_sha256": f"{DATA}/downloads/vqav2/SHA256SUMS"},
    # EXP-0020: a second, fresh VQAv2 val test subset (1000 other images, make_vqav2_sets.py test2)
    "vqav2_test2": {"question_file": f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_test2.jsonl",
                    "image_folder": f"{DATA}/coco/val2014",
                    "annotations": [f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_test2_annotations.jsonl"],
                    "archives_sha256": f"{DATA}/downloads/vqav2/SHA256SUMS"},
    # EXP-0022: GQA testdev-balanced (LLaVA's question file from eval.zip; answers and the 398 images from the official
    # GQA archives by make_gqa_set.py; the images' checksums are in gqa/images_SHA256SUMS), scored by score_pope_mme.py
    "gqa_testdev": {"question_file": f"{DATA}/llava_eval/gqa/llava_gqa_testdev_balanced.jsonl",
                    "image_folder": f"{DATA}/gqa/images",
                    "annotations": [f"{DATA}/llava_eval/gqa/llava_gqa_testdev_balanced_annotations.jsonl"],
                    "archives_sha256": f"{DATA}/downloads/gqa/SHA256SUMS"},
    # IDEA-0004 (make_dev_sets.py, drawn once before any development run): dev2 and GQA val400 for tuning the new module,
    # test3 kept for its pre-registered confirmation. Checksums: vqav2_val/SHA256SUMS_idea0004, gqa/SHA256SUMS_val400 and,
    # for the 400 GQA images, gqa/images_val400_SHA256SUMS (written when they were extracted)
    "vqav2_dev2": {"question_file": f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_dev2.jsonl",
                   "image_folder": f"{DATA}/coco/val2014",
                   "annotations": [f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_dev2_annotations.jsonl"],
                   "archives_sha256": f"{DATA}/downloads/vqav2/SHA256SUMS"},
    "vqav2_test3": {"question_file": f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_test3.jsonl",
                    "image_folder": f"{DATA}/coco/val2014",
                    "annotations": [f"{DATA}/llava_eval/vqav2_val/llava_vqav2_val_test3_annotations.jsonl"],
                    "archives_sha256": f"{DATA}/downloads/vqav2/SHA256SUMS"},
    "gqa_val400": {"question_file": f"{DATA}/llava_eval/gqa/llava_gqa_val400.jsonl",
                   "image_folder": f"{DATA}/gqa/images_val400",
                   "annotations": [f"{DATA}/llava_eval/gqa/llava_gqa_val400_annotations.jsonl"],
                   "archives_sha256": f"{DATA}/downloads/gqa/SHA256SUMS"},
    # IDEA-0004 confirmation benchmarks (DEC-0036, make_textvqa_vstar.py): TextVQA v0.5.1 val with TextVQA's own question
    # ids, without and with LLaVA's OCR-token line; V* Bench's 191 multiple-choice questions (revision d9ae62c9)
    "textvqa_noocr": {"question_file": f"{DATA}/llava_eval/textvqa/llava_textvqa_val_noocr_qid.jsonl",
                      "image_folder": f"{DATA}/textvqa/train_images",
                      "annotations": [f"{DATA}/llava_eval/textvqa/llava_textvqa_val_annotations.jsonl"],
                      "archives_sha256": f"{DATA}/downloads/textvqa/SHA256SUMS"},
    "textvqa_ocr": {"question_file": f"{DATA}/llava_eval/textvqa/llava_textvqa_val_ocr_qid.jsonl",
                    "image_folder": f"{DATA}/textvqa/train_images",
                    "annotations": [f"{DATA}/llava_eval/textvqa/llava_textvqa_val_annotations.jsonl"],
                    "archives_sha256": f"{DATA}/downloads/textvqa/SHA256SUMS"},
    "vstar": {"question_file": f"{DATA}/llava_eval/vstar/llava_vstar_test.jsonl",
              "image_folder": f"{DATA}/downloads/vstar_bench",
              "annotations": [f"{DATA}/llava_eval/vstar/llava_vstar_test_annotations.jsonl"],
              "archives_sha256": f"{DATA}/downloads/vstar_bench/SHA256SUMS"},
}
# Budgets are 32-layer averages of visual tokens (code/token_pruning/README.md, "预算的口径").
FASTV_KEEP = {192: 166, 128: 98, 64: 30}
VISIONZIP = {192: (162, 30), 128: (108, 20), 64: (54, 10)}
CONFIGS = {"vanilla": ("tp", ["--method", "vanilla"]),
           # gate diagnostic: LLaVA's MODEL_ZOO numbers predate transformers' sdpa default
           "vanilla_eager": ("tp", ["--method", "vanilla", "--attn-implementation", "eager"])}
for b in (192, 128, 64):
    CONFIGS[f"fastv{b}"] = ("tp", ["--method", "fastv", "--fastv-k", "2", "--fastv-keep", str(FASTV_KEEP[b])])
    CONFIGS[f"visionzip{b}"] = ("tp", ["--method", "visionzip", "--dominant", str(VISIONZIP[b][0]),
                                       "--contextual", str(VISIONZIP[b][1])])
    CONFIGS[f"sparsevlm{b}"] = ("sparsevlm", ["--method", "sparsevlm", "--retained-tokens", str(b)])
    for seed in (2021, 2022, 2023):
        CONFIGS[f"random{b}_s{seed}"] = ("tp", ["--method", "random", "--fastv-k", "2", "--fastv-keep",
                                                str(FASTV_KEEP[b]), "--seed", str(seed)])
# EXP-0007: FastV's ranking or random selection on SparseVLM's measured per-layer token counts (EXP-0002-R47~R49:
# 576 up to layer 2, then three stages), and the flat K=2 profile with about the same 32-layer average
SVP = {192: "3:309,7:204,16:113", 128: "3:312,7:117,16:39", 64: "3:82,7:32,16:18"}
FLAT_KEEP = {192: 182, 128: 117, 64: 49}        # (2*576 + 30*keep)/32 = 206.6/145.7/81.9 vs 206.5/145.4/82.25
for b in (192, 128, 64):
    CONFIGS[f"fastv_svp{b}"] = ("tp", ["--method", "fastv", "--fastv-schedule", SVP[b]])
    CONFIGS[f"fastv_flat{b}"] = ("tp", ["--method", "fastv", "--fastv-k", "2", "--fastv-keep", str(FLAT_KEEP[b])])
    for seed in (2021, 2022, 2023):
        CONFIGS[f"random_svp{b}_s{seed}"] = ("tp", ["--method", "random", "--fastv-schedule", SVP[b], "--seed", str(seed)])
        CONFIGS[f"random_flat{b}_s{seed}"] = ("tp", ["--method", "random", "--fastv-k", "2", "--fastv-keep",
                                                     str(FLAT_KEEP[b]), "--seed", str(seed)])
# EXP-0008: FastV with its scores divided by the content-free position curve measured on blank images
# (analyze_score_prior.py writes the curve after the blank-image run)
SCORE_PRIOR = f"{REPO}/code/output/EXP-0008/analysis/position_prior.npy"
for b in (192, 128, 64):
    CONFIGS[f"fastv{b}_prior"] = ("tp", CONFIGS[f"fastv{b}"][1] + ["--fastv-score-prior", SCORE_PRIOR])
# EXP-0010: the ranking taken in turns from a 6x6 grid of 4x4-patch blocks (the blocks of the coverage statistic),
# on the raw scores (coverage without removing the position term) and after dividing out the position curve
for b in (192, 128, 64):
    CONFIGS[f"fastv{b}_strat"] = ("tp", CONFIGS[f"fastv{b}"][1] + ["--fastv-stratify", "6"])
    CONFIGS[f"fastv{b}_prior_strat"] = ("tp", CONFIGS[f"fastv{b}_prior"][1] + ["--fastv-stratify", "6"])
# EXP-0011 D3 (plan v1.2): rank by the cosine of each image token's input embedding to the object name's (POPE-style
# questions only), in blocks, padding never kept; and the padding-exclusion control on the strong baseline
for b in (192, 128, 64):
    CONFIGS[f"fastv{b}_objcos_strat_nopad"] = ("tp", CONFIGS[f"fastv{b}"][1] + [
        "--fastv-stratify", "6", "--fastv-score", "object_cos", "--fastv-exclude-padding"])
    CONFIGS[f"fastv{b}_prior_strat_nopad"] = ("tp", CONFIGS[f"fastv{b}_prior_strat"][1] + ["--fastv-exclude-padding"])
# EXP-0013: the same budget split between the two rankings: mix<n> keeps n tokens by the object-name cosine and the rest
# by the strong baseline's score (prior, blocks, padding never kept), the second rotation filling the empty blocks first
for n in (20, 15, 10):
    CONFIGS[f"fastv64_mix{n}"] = ("tp", CONFIGS["fastv64_prior_strat_nopad"][1] + [
        "--fastv-score", "mix", "--fastv-mix-question", str(n)])
# EXP-0013: the object-name cosine minus the largest / mean cosine of the same image token to the other COCO names,
# otherwise as the EXP-0011 variant (blocks, padding never kept)
for kind in ("max", "mean"):
    CONFIGS[f"fastv64_contrast{kind}_strat_nopad"] = ("tp", CONFIGS["fastv64"][1] + [
        "--fastv-stratify", "6", "--fastv-score", f"contrast_{kind}", "--fastv-exclude-padding"])
# EXP-0014 (steps 9-12, any question): the query is the uniform (qmean, A') or IDF-weighted (qidf, A'+B) mean input
# embedding of all tokens after the image; attnidf (A+B) weights the layer-1 attention rows of those tokens by IDF.
# Blocks and padding exclusion as in the EXP-0011 variant; IDF is counted over the evaluated question file.
for name, kind in (("qmean", "query_mean"), ("qidf", "query_idf"), ("attnidf", "attn_idf")):
    CONFIGS[f"fastv64_{name}_strat_nopad"] = ("tp", CONFIGS["fastv64"][1] + [
        "--fastv-stratify", "6", "--fastv-score", kind, "--fastv-exclude-padding"])
# EXP-0015: IDF x visibility (visibility per vocabulary id from the coco_dev images, make_visibility.py); round(30 * S)
# tokens by that query, the rest by the strong baseline's score (prior, blocks, padding never kept)
VISIBILITY = f"{REPO}/code/output/EXP-0015/visibility/visibility.npy"
CONFIGS["fastv64_qvis"] = ("tp", CONFIGS["fastv64_prior_strat_nopad"][1] + [
    "--fastv-score", "query_vis", "--fastv-visibility", VISIBILITY])
# EXP-0016: the same with visibility 0.5 for every id (a fixed 15 / 15 split, the IDF query unchanged), and query_vis at
# the 128 budget (keep 98)
VISIBILITY_HALF = f"{REPO}/code/output/EXP-0016/visibility_half/visibility.npy"
CONFIGS["fastv64_qvis_half"] = ("tp", CONFIGS["fastv64_prior_strat_nopad"][1] + [
    "--fastv-score", "query_vis", "--fastv-visibility", VISIBILITY_HALF])
CONFIGS["fastv128_qvis"] = ("tp", CONFIGS["fastv128_prior_strat_nopad"][1] + [
    "--fastv-score", "query_vis", "--fastv-visibility", VISIBILITY])
# EXP-0019: the strong baseline without the position prior (last-token row, blocks, padding never kept), the matching
# control for EXP-0014's A+B (attn_idf has no prior)
CONFIGS["fastv64_strat_nopad"] = ("tp", CONFIGS["fastv64_strat"][1] + ["--fastv-exclude-padding"])
# EXP-0017: query_vis at the 192 budget (keep 166)
CONFIGS["fastv192_qvis"] = ("tp", CONFIGS["fastv192_prior_strat_nopad"][1] + [
    "--fastv-score", "query_vis", "--fastv-visibility", VISIBILITY])
# EXP-0020: query_vis only for questions in yes/no form or asking "how many"; the others take the strong baseline's path
for b in (64, 128, 192):
    CONFIGS[f"fastv{b}_qroute"] = ("tp", CONFIGS[f"fastv{b}_qvis"][1] + ["--fastv-route", "yesno_howmany"])
# IDEA-0004 (EXP-0024): on the router's unrouted (open) questions, locate the question's object with the query and either
# insert n tokens of a re-encoded crop of it (zoom; z16g / z16q / z9g: 16 or 9 tokens, lattice or best query cosine per
# block) or keep the baseline's in-box count plus 8 inside its box (cover, c8); _all zooms every question. The tokens after
# the stage stay keep; the 32-layer average rises by n * K / 32 (+1.0 for n = 16 at K = 2), and each zoom adds a vision
# tower and projector pass on the crop.
for b in (64, 128, 192):
    base = CONFIGS[f"fastv{b}_qroute"][1]
    CONFIGS[f"fastv{b}_qroute_z16g"] = ("tp", base + ["--fastv-zoom", "16", "--fastv-zoom-pick", "grid"])
    CONFIGS[f"fastv{b}_qroute_z16q"] = ("tp", base + ["--fastv-zoom", "16", "--fastv-zoom-pick", "query"])
    CONFIGS[f"fastv{b}_qroute_z9g"] = ("tp", base + ["--fastv-zoom", "9", "--fastv-zoom-pick", "grid"])
    CONFIGS[f"fastv{b}_qroute_c8"] = ("tp", base + ["--fastv-cover", "8"])
    CONFIGS[f"fastv{b}_qroute_z16g_all"] = ("tp", base + ["--fastv-zoom", "16", "--fastv-zoom-pick", "grid",
                                                          "--fastv-zoom-on", "all"])
# EXP-0024 (after the pre-run code and design review): the zoom configs run on every question with s_min 0, so the
# unrouted-only result comes from splicing with the router run; z16p pools the crop's tokens into 4 x 4 block means;
# p16full inserts block means of the image's own tokens (no second encoding, the context control); z16pctr keeps the
# located side but centres the box (the localisation control); c16 keeps the baseline's in-box count plus 16 in the box
# (the reallocation control). K = 1 keeps 47 tokens: (576 + 31 x 47) / 32 = 63.5 tokens per layer, 64.0 with 16 zoom
# tokens; its position prior is the blank-image score curve at layer 0 (EXP-0024-R01, fastv64k1_scores).
PRIOR_K1 = f"{REPO}/code/output/EXP-0024/prior_k1/position_prior.npy"


def _at_k1(args):
    a = list(args)
    a[a.index("--fastv-k") + 1], a[a.index("--fastv-keep") + 1] = "1", "47"
    a[a.index("--fastv-score-prior") + 1] = PRIOR_K1
    return a


ALL0 = ["--fastv-zoom-on", "all", "--fastv-zoom-smin", "0"]
CONFIGS["fastv64k1"] = ("tp", ["--method", "fastv", "--fastv-k", "1", "--fastv-keep", "47"])
CONFIGS["fastv64k1_qroute"] = ("tp", _at_k1(CONFIGS["fastv64_qroute"][1]))
CONFIGS["fastv64_qroute_z16g_all0"] = ("tp", CONFIGS["fastv64_qroute"][1] + ["--fastv-zoom", "16", "--fastv-zoom-pick", "grid"] + ALL0)
CONFIGS["fastv64_qroute_z16p_all0"] = ("tp", CONFIGS["fastv64_qroute"][1] + ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool"] + ALL0)
CONFIGS["fastv64_qroute_p16full_all0"] = ("tp", CONFIGS["fastv64_qroute"][1] + [
    "--fastv-zoom", "16", "--fastv-zoom-pick", "pool", "--fastv-zoom-source", "full"] + ALL0)
CONFIGS["fastv64_qroute_z16pctr_all0"] = ("tp", CONFIGS["fastv64_qroute"][1] + [
    "--fastv-zoom", "16", "--fastv-zoom-pick", "pool", "--fastv-zoom-centre"] + ALL0)
CONFIGS["fastv64_qroute_c16"] = ("tp", CONFIGS["fastv64_qroute"][1] + ["--fastv-cover", "16", "--fastv-zoom-smin", "0"])
CONFIGS["fastv64k1_qroute_z16p_all0"] = ("tp", CONFIGS["fastv64k1_qroute"][1] + ["--fastv-zoom", "16", "--fastv-zoom-pick", "pool"] + ALL0)
# EXP-0025 (plan v1): the attribution controls for the selected lattice zoom, same K and pick. g16full inserts the image's
# own tokens at the same 4 x 4 lattice over its content (no second encoding); z16gctr keeps the located side but centres
# the box.
CONFIGS["fastv64_qroute_g16full_all0"] = ("tp", CONFIGS["fastv64_qroute"][1] + [
    "--fastv-zoom", "16", "--fastv-zoom-pick", "grid", "--fastv-zoom-source", "full"] + ALL0)
CONFIGS["fastv64_qroute_z16gctr_all0"] = ("tp", CONFIGS["fastv64_qroute"][1] + [
    "--fastv-zoom", "16", "--fastv-zoom-pick", "grid", "--fastv-zoom-centre"] + ALL0)
# IDEA-0005 (EXP-0026): locate-then-cover on the unrouted questions. c16ctr keeps cover 16's side but centres the box (the
# localisation control); c8s0 moves fewer tokens into the box (cover 8, s_min 0 like c16; the older c8 keeps s_min 0.2);
# the K = 1 versions take the router at K = 1, keep 47 (63.5 tokens per layer).
CONFIGS["fastv64_qroute_c16ctr"] = ("tp", CONFIGS["fastv64_qroute_c16"][1] + ["--fastv-zoom-centre"])
CONFIGS["fastv64_qroute_c8s0"] = ("tp", CONFIGS["fastv64_qroute"][1] + ["--fastv-cover", "8", "--fastv-zoom-smin", "0"])
CONFIGS["fastv64k1_qroute_c16"] = ("tp", CONFIGS["fastv64k1_qroute"][1] + ["--fastv-cover", "16", "--fastv-zoom-smin", "0"])
CONFIGS["fastv64k1_qroute_c16ctr"] = ("tp", CONFIGS["fastv64k1_qroute_c16"][1] + ["--fastv-zoom-centre"])
# EXP-0026 v1.1, the no-box control: the 16 tokens most similar to the question are kept anywhere, the rest by the
# baseline's rule outside them (the cover rule with that set as its region).
CONFIGS["fastv64_qroute_t16"] = ("tp", CONFIGS["fastv64_qroute_c16"][1] + ["--fastv-cover-top", "16"])
# "<config>_scores": FastV with --score-log, for the score-curve diagnostic
CONFIGS.update({f"{name}_scores": spec for name, spec in list(CONFIGS.items()) if name.startswith("fastv")})
# "<config>_layers": the same run with --layer-token-log, for the per-layer token-count diagnostic
CONFIGS.update({f"{name}_layers": spec for name, spec in list(CONFIGS.items())})
# "<config>_kept": FastV or random selection with --kept-token-log, for the kept-patch position diagnostic
CONFIGS.update({f"{name}_kept": spec for name, spec in list(CONFIGS.items())
                if name.startswith(("fastv", "random")) and not name.endswith(("_layers", "_scores"))})
# EXP-0011 (1A4B-0003 D2): "<config>_rows" is FastV with the kept patches, the scoring layer's attention rows of every
# text token after the image, and the first answer token's log-probabilities; "<config>_logits" is any method except
# SparseVLM with the log-probabilities only (candidate 2's diagnostic). Neither changes the computation.
BASE = [name for name in CONFIGS if not name.endswith(("_layers", "_kept", "_scores"))]
CONFIGS.update({f"{name}_rows": CONFIGS[name] for name in BASE if name.startswith("fastv")})
CONFIGS.update({f"{name}_logits": CONFIGS[name] for name in BASE if not name.startswith("sparsevlm")})
# EXP-0011 D2b: vanilla with --layer-rows-log (attention rows at several layers, per-head rows at layers 0-1 and
# hidden-state similarities of the question spans); the prefill runs the eager kernel, so answers may differ slightly
CONFIGS["vanilla_layerrows"] = CONFIGS["vanilla"]
# EXP-0013: "<config>_ksl" logs the kept patches, the first stage's score vector and the first answer token's
# log-probabilities in one run; "<config>_kl" the kept patches and the log-probabilities; "vanilla_names" is vanilla with
# the log-probabilities and the input-embedding cosine of every image token to the 80 COCO names. None of them changes
# the computation.
CONFIGS.update({f"{name}_ksl": CONFIGS[name] for name in BASE if name.startswith("fastv")})
CONFIGS.update({f"{name}_kl": CONFIGS[name] for name in BASE if name.startswith("fastv")})
CONFIGS["vanilla_names"] = CONFIGS["vanilla"]
# EXP-0015: vanilla with the log-probabilities and, per image, the largest and mean cosine of every vocabulary embedding
CONFIGS["vanilla_vocabvis"] = CONFIGS["vanilla"]


def experiment(run_id):
    return run_id.rsplit("-R", 1)[0]


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout


def in_env(env, command):
    return [BASH, "-c", f"source {ENV_SH} {env} && {command}"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def git_record():
    lines = [f"repo HEAD {sh([GIT, '-C', REPO, 'rev-parse', 'HEAD']).strip()} "
             f"({sh([GIT, '-C', REPO, 'branch', '--show-current']).strip()})",
             "repo status --porcelain:", sh([GIT, "-C", REPO, "status", "--porcelain"]).rstrip(),
             "repo diff HEAD:", sh([GIT, "-C", REPO, "diff", "HEAD"]).rstrip(), ""]
    for name in ("LLaVA", "SparseVLMs", "VisionZip", "FastV"):
        dirty = "clean" if not sh([GIT, "-C", f"{SRC}/{name}", "status", "--porcelain"]).strip() else "MODIFIED"
        lines.append(f"source/{name} {sh([GIT, '-C', f'{SRC}/{name}', 'rev-parse', 'HEAD']).strip()} {dirty}")
    for repo in ("models--liuhaotian--llava-v1.5-7b", "models--openai--clip-vit-large-patch14-336"):
        with open(f"{HF_HUB}/{repo}/refs/main") as f:
            lines.append(f"hf {repo} refs/main {f.read().strip()}")
    return "\n".join(lines) + "\n"


def run_one(run_id, config, bench):
    env, method_args = CONFIGS[config]
    b = BENCHES[bench]
    out = f"{REPO}/code/output/{experiment(run_id)}"
    run_dir = f"{out}/{run_id}"
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_id} already exists; give the retry a new run ID")
    os.makedirs(run_dir)
    answers = f"{run_dir}/answers.jsonl"
    if config.endswith("_layers"):
        method_args = method_args + ["--layer-token-log", f"{run_dir}/layer_tokens.json"]
    if config.endswith("_kept"):
        method_args = method_args + ["--kept-token-log", f"{run_dir}/kept_tokens.json"]
    if config.endswith("_scores"):
        method_args = method_args + ["--score-log", f"{run_dir}/scores.npy"]
    if config.endswith("_rows"):
        method_args = method_args + ["--kept-token-log", f"{run_dir}/kept_tokens.json",
                                     "--text-rows-log", f"{run_dir}/text_rows.npz",
                                     "--answer-logit-log", f"{run_dir}/answer_logits.json"]
    if config.endswith("_logits"):
        method_args = method_args + ["--answer-logit-log", f"{run_dir}/answer_logits.json"]
    if config.endswith("_layerrows"):
        method_args = method_args + ["--layer-rows-log", f"{run_dir}/layer_rows.npz"]
    if config.endswith(("_ksl", "_kl")):
        method_args = method_args + ["--kept-token-log", f"{run_dir}/kept_tokens.json",
                                     "--answer-logit-log", f"{run_dir}/answer_logits.json"]
    if config.endswith("_ksl"):
        method_args = method_args + ["--score-log", f"{run_dir}/scores.npy"]
    if "--fastv-zoom" in method_args or "--fastv-cover" in method_args:     # IDEA-0004: one record per prefill
        method_args = method_args + ["--fastv-zoom-log", f"{run_dir}/zoom.json"]
    if config.endswith("_names"):
        method_args = method_args + ["--answer-logit-log", f"{run_dir}/answer_logits.json",
                                     "--name-cos-log", f"{run_dir}/name_cos.npz"]
    if config.endswith("_vocabvis"):
        method_args = method_args + ["--answer-logit-log", f"{run_dir}/answer_logits.json",
                                     "--vocab-visibility-log", f"{run_dir}/vocab_visibility.npz"]
    eval_cmd = ("cd {here} && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_MININTERVAL=30 VISIONZIP_ROOT={vz} "
                "python run_eval.py {margs} -- --model-path {model} --question-file {q} --image-folder {img} "
                "--answers-file {ans} --temperature 0 --conv-mode vicuna_v1").format(
        here=HERE, vz=f"{SRC}/VisionZip", margs=" ".join(map(shlex.quote, method_args)), model=MODEL,
        q=b["question_file"], img=b["image_folder"], ans=answers)
    score_cmd = [sys.executable, f"{HERE}/score_pope_mme.py", bench, answers, run_dir, "--tag", run_id]
    with open(f"{run_dir}/cmd.txt", "w") as f:
        scoring = f"# scoring (tp env)\n{shlex.join(score_cmd)}\n" if b.get("score", True) else "# not scored (diagnostic)\n"
        f.write(f"# env: {env}\n{shlex.join(in_env(env, eval_cmd))}\n{scoring}")
    config_rec = {"experiment": experiment(run_id), "run_id": run_id, "config": config, "bench": bench, "env": env,
                  "method_args": method_args, "model": MODEL, "decoding": "greedy (temperature 0), conv vicuna_v1",
                  "question_file": b["question_file"], "question_file_sha256": sha256(b["question_file"]),
                  "image_folder": b["image_folder"],
                  "annotations_sha256": {os.path.basename(p): sha256(p) for p in b["annotations"]},
                  "data_archives_sha256_file": b.get("archives_sha256", f"{DATA}/downloads/SHA256SUMS")}
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config_rec, f, indent=2)
    seed = method_args[method_args.index("--seed") + 1] if "--seed" in method_args else "none"
    with open(f"{run_dir}/seed.txt", "w") as f:
        f.write(f"token-selection seed: {seed}\ndecoding: greedy, no sampling\n")
    with open(f"{run_dir}/git.txt", "w") as f:
        f.write(git_record())
    with open(f"{run_dir}/env.txt", "w") as f:
        f.write(sh(in_env(env, "python --version && python -m pip freeze")))
    with open(f"{run_dir}/gpu.txt", "w") as f:
        f.write(sh(["nvidia-smi"]))

    status = {"run_id": run_id, "start": now()}
    with open(f"{run_dir}/stdout.log", "w") as log:
        status["eval_exit"] = subprocess.run(in_env(env, eval_cmd), stdout=log, stderr=subprocess.STDOUT).returncode
    if status["eval_exit"] == 0 and b.get("score", True):
        with open(f"{run_dir}/stdout.log", "a") as log:
            status["score_exit"] = subprocess.run(score_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    elif status["eval_exit"] == 0 and os.path.exists(f"{run_dir}/layer_tokens.json"):
        with open(f"{run_dir}/layer_tokens.json") as f:                # unscored diagnostic: the token counts
            layers = json.load(f)
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump({"benchmark": bench, "headline": "mean_visual_over_layers",
                       "mean_visual_over_layers": layers["mean_visual_over_layers"],
                       "mean_visual_per_layer": layers["mean_visual_per_layer"], "n_prefills": layers["n_prefills"],
                       "n_complete": layers.get("n_complete"), "n_layers": layers.get("n_layers")}, f, indent=2)
        status["score_exit"] = 0 if layers.get("n_complete") == layers["n_prefills"] else 1
    elif status["eval_exit"] == 0 and os.path.exists(f"{run_dir}/kept_tokens.json"):
        with open(f"{run_dir}/kept_tokens.json") as f:                 # unscored diagnostic: one kept set per question
            kept = json.load(f)
        with open(b["question_file"]) as f:
            n_questions = sum(1 for _ in f)
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump({"benchmark": bench, "headline": "n_prefills", "n_prefills": kept["n_prefills"],
                       "n_questions": n_questions, "keep": kept["keep"], "n_image_tokens": kept["n_image_tokens"],
                       "all_sets_full": all(len(s) == kept["keep"] for s in kept["kept"])}, f, indent=2)
        status["score_exit"] = 0 if kept["n_prefills"] == n_questions else 1
    elif status["eval_exit"] == 0 and os.path.exists(f"{run_dir}/layer_rows.npz"):
        import numpy as np                                             # unscored diagnostic: one record per question
        z = np.load(f"{run_dir}/layer_rows.npz")
        with open(b["question_file"]) as f:
            n_questions = sum(1 for _ in f)
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump({"benchmark": bench, "headline": "n_prefills", "n_prefills": int(z["avg_rows"].shape[0]),
                       "n_questions": n_questions, "avg_rows_shape": list(z["avg_rows"].shape),
                       "head_rows_shape": list(z["head_rows"].shape), "cos_shape": list(z["cos"].shape)}, f, indent=2)
        status["score_exit"] = 0 if z["avg_rows"].shape[0] == n_questions else 1
    elif status["eval_exit"] == 0 and os.path.exists(f"{run_dir}/scores.npy"):
        import numpy as np                                             # unscored diagnostic: one score vector per question
        scores = np.load(f"{run_dir}/scores.npy")
        with open(b["question_file"]) as f:
            n_questions = sum(1 for _ in f)
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump({"benchmark": bench, "headline": "n_prefills", "n_prefills": int(scores.shape[0]),
                       "n_questions": n_questions, "n_image_tokens": int(scores.shape[1])}, f, indent=2)
        status["score_exit"] = 0 if scores.shape[0] == n_questions else 1
    elif status["eval_exit"] == 0 and os.path.exists(f"{run_dir}/answer_logits.json"):
        with open(f"{run_dir}/answer_logits.json") as f:              # unscored diagnostic: log-probabilities only
            logits = json.load(f)
        with open(b["question_file"]) as f:
            n_questions = sum(1 for _ in f)
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump({"benchmark": bench, "headline": "n_prefills", "n_prefills": logits["n_prefills"],
                       "n_questions": n_questions}, f, indent=2)
        status["score_exit"] = 0 if logits["n_prefills"] == n_questions else 1
    status["end"] = now()
    status["status"] = "completed" if status.get("score_exit") == 0 else "failed"
    with open(f"{run_dir}/status.json", "w") as f:
        json.dump(status, f, indent=2)

    files = sorted(os.path.relpath(os.path.join(d, n), run_dir).replace("\\", "/")
                   for d, _, names in os.walk(run_dir) for n in names)
    sums = {p: sha256(f"{run_dir}/{p}") for p in files}
    # LF line endings (from EXP-0022 on): earlier runs wrote CRLF in text mode, and Git Bash's `sha256sum -c` needs
    # `tr -d '\r' < SHA256SUMS | sha256sum -c -` for those (full drift check, 2026-10-25)
    with open(f"{run_dir}/SHA256SUMS", "w", newline="\n") as f:
        f.writelines(f"{h}  {p}\n" for p, h in sums.items())
    backup = f"{BACKUP}/{experiment(run_id)}/{run_id}"
    shutil.copytree(run_dir, backup)
    if any(sha256(f"{backup}/{p}") != h for p, h in sums.items()):
        raise SystemExit(f"backup of {run_id} does not match its checksums")

    metrics, meta = {}, {}
    if status["status"] == "completed":
        with open(f"{run_dir}/metrics.json") as f:
            metrics = json.load(f)
        with open(answers + ".meta.json") as f:
            meta = json.load(f)
    headline = metrics.get(metrics.get("headline", ""), "NA")
    row = [run_id, config, bench, status["status"], str(headline), str(meta.get("eval_seconds", "NA")),
           str(meta.get("seconds_per_answer", "NA")), str(meta.get("eval_peak_gpu_mem_gib", "NA")),
           status["start"], status["end"], sha256(f"{run_dir}/SHA256SUMS")]
    new = not os.path.exists(f"{out}/runs.tsv")
    with open(f"{out}/runs.tsv", "a") as f:
        if new:
            f.write("run_id\tconfig\tbench\tstatus\theadline\teval_s\ts_per_answer\tpeak_gib\tstart\tend\tsha256sums_sha256\n")
        f.write("\t".join(row) + "\n")
    shutil.copyfile(f"{out}/runs.tsv", f"{BACKUP}/{experiment(run_id)}/runs.tsv")   # runs.tsv had no backup before EXP-0022
    print("\t".join(row), flush=True)
    return status["status"]


def main():
    for spec in sys.argv[1:]:
        run_id, config, bench = spec.split(":")
        if config not in CONFIGS or bench not in BENCHES:
            raise SystemExit(f"unknown config or bench in {spec}")
    for spec in sys.argv[1:]:
        run_one(*spec.split(":"))


if __name__ == "__main__":
    main()
