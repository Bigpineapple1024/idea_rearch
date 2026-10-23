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
# "<config>_layers": the same run with --layer-token-log, for the per-layer token-count diagnostic
CONFIGS.update({f"{name}_layers": spec for name, spec in list(CONFIGS.items())})
# "<config>_kept": FastV or random selection with --kept-token-log, for the kept-patch position diagnostic
CONFIGS.update({f"{name}_kept": spec for name, spec in list(CONFIGS.items())
                if name.startswith(("fastv", "random")) and not name.endswith("_layers")})


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
    eval_cmd = ("cd {here} && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_MININTERVAL=30 VISIONZIP_ROOT={vz} "
                "python run_eval.py {margs} -- --model-path {model} --question-file {q} --image-folder {img} "
                "--answers-file {ans} --temperature 0 --conv-mode vicuna_v1").format(
        here=HERE, vz=f"{SRC}/VisionZip", margs=" ".join(map(shlex.quote, method_args)), model=MODEL,
        q=b["question_file"], img=b["image_folder"], ans=answers)
    score_cmd = [sys.executable, f"{HERE}/score_pope_mme.py", bench, answers, run_dir, "--tag", run_id]
    with open(f"{run_dir}/cmd.txt", "w") as f:
        scoring = f"# scoring (tp env)\n{shlex.join(score_cmd)}\n" if b.get("score", True) else "# not scored (diagnostic)\n"
        f.write(f"# env: {env}\n{shlex.join(in_env(env, eval_cmd))}\n{scoring}")
    config_rec = {"experiment": "EXP-0002", "run_id": run_id, "config": config, "bench": bench, "env": env,
                  "method_args": method_args, "model": MODEL, "decoding": "greedy (temperature 0), conv vicuna_v1",
                  "question_file": b["question_file"], "question_file_sha256": sha256(b["question_file"]),
                  "image_folder": b["image_folder"],
                  "annotations_sha256": {os.path.basename(p): sha256(p) for p in b["annotations"]},
                  "data_archives_sha256_file": f"{DATA}/downloads/SHA256SUMS"}
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
    status["end"] = now()
    status["status"] = "completed" if status.get("score_exit") == 0 else "failed"
    with open(f"{run_dir}/status.json", "w") as f:
        json.dump(status, f, indent=2)

    files = sorted(os.path.relpath(os.path.join(d, n), run_dir).replace("\\", "/")
                   for d, _, names in os.walk(run_dir) for n in names)
    sums = {p: sha256(f"{run_dir}/{p}") for p in files}
    with open(f"{run_dir}/SHA256SUMS", "w") as f:
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
