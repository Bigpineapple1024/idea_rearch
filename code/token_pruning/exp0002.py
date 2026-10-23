"""Run EXP-0002 evaluations with the per-run records the research process requires.

  python exp0002.py EXP-0002-R01:vanilla:pope EXP-0002-R02:vanilla:mme ...    # in the tp env, sequentially

Each run gets code/output/EXP-0002/<run>/ with cmd.txt, config.json, git.txt (this repo + external commits +
model revisions), seed.txt, env.txt, gpu.txt, stdout.log, answers.jsonl(+.meta.json), score.log, metrics.json,
status.json and SHA256SUMS, and is copied to D:/6CJNM007/backup/EXP-0002/<run>/ (checksums re-verified).
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
OUT = f"{REPO}/code/output/EXP-0002"
BACKUP = "D:/6CJNM007/backup/EXP-0002"
MODEL = "liuhaotian/llava-v1.5-7b"

BENCHES = {
    "pope": {"question_file": f"{DATA}/llava_eval/pope/llava_pope_test.jsonl", "image_folder": f"{DATA}/coco/val2014",
             "annotations": [f"{DATA}/llava_eval/pope/coco/coco_pope_{s}.json" for s in ("adversarial", "popular", "random")]},
    "mme": {"question_file": f"{DATA}/llava_eval/MME/llava_mme.jsonl",
            "image_folder": f"{DATA}/llava_eval/MME/MME_Benchmark_release_version", "annotations": []},
}
# Budgets are 32-layer averages of visual tokens (code/token_pruning/README.md, "预算的口径").
FASTV_KEEP = {192: 166, 128: 98, 64: 30}
VISIONZIP = {192: (162, 30), 128: (108, 20), 64: (54, 10)}
CONFIGS = {"vanilla": ("tp", ["--method", "vanilla"])}
for b in (192, 128, 64):
    CONFIGS[f"fastv{b}"] = ("tp", ["--method", "fastv", "--fastv-k", "2", "--fastv-keep", str(FASTV_KEEP[b])])
    CONFIGS[f"visionzip{b}"] = ("tp", ["--method", "visionzip", "--dominant", str(VISIONZIP[b][0]),
                                       "--contextual", str(VISIONZIP[b][1])])
    CONFIGS[f"sparsevlm{b}"] = ("sparsevlm", ["--method", "sparsevlm", "--retained-tokens", str(b)])
    for seed in (2021, 2022, 2023):
        CONFIGS[f"random{b}_s{seed}"] = ("tp", ["--method", "random", "--fastv-k", "2", "--fastv-keep",
                                                str(FASTV_KEEP[b]), "--seed", str(seed)])


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
    run_dir = f"{OUT}/{run_id}"
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_id} already exists; give the retry a new run ID")
    os.makedirs(run_dir)
    answers = f"{run_dir}/answers.jsonl"
    eval_cmd = ("cd {here} && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_MININTERVAL=30 VISIONZIP_ROOT={vz} "
                "python run_eval.py {margs} -- --model-path {model} --question-file {q} --image-folder {img} "
                "--answers-file {ans} --temperature 0 --conv-mode vicuna_v1").format(
        here=HERE, vz=f"{SRC}/VisionZip", margs=" ".join(map(shlex.quote, method_args)), model=MODEL,
        q=b["question_file"], img=b["image_folder"], ans=answers)
    score_cmd = [sys.executable, f"{HERE}/score_pope_mme.py", bench, answers, run_dir, "--tag", run_id]
    with open(f"{run_dir}/cmd.txt", "w") as f:
        f.write(f"# env: {env}\n{shlex.join(in_env(env, eval_cmd))}\n# scoring (tp env)\n{shlex.join(score_cmd)}\n")
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
    if status["eval_exit"] == 0:
        with open(f"{run_dir}/stdout.log", "a") as log:
            status["score_exit"] = subprocess.run(score_cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    status["end"] = now()
    status["status"] = "completed" if status.get("score_exit") == 0 else "failed"
    with open(f"{run_dir}/status.json", "w") as f:
        json.dump(status, f, indent=2)

    files = sorted(os.path.relpath(os.path.join(d, n), run_dir).replace("\\", "/")
                   for d, _, names in os.walk(run_dir) for n in names)
    sums = {p: sha256(f"{run_dir}/{p}") for p in files}
    with open(f"{run_dir}/SHA256SUMS", "w") as f:
        f.writelines(f"{h}  {p}\n" for p, h in sums.items())
    shutil.copytree(run_dir, f"{BACKUP}/{run_id}")
    if any(sha256(f"{BACKUP}/{run_id}/{p}") != h for p, h in sums.items()):
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
    new = not os.path.exists(f"{OUT}/runs.tsv")
    with open(f"{OUT}/runs.tsv", "a") as f:
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
