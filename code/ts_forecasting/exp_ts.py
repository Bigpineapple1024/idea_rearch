"""Run time-series reproduction runs with the per-run records the research process requires.

  python exp_ts.py <EXP> <run_id>:<method>:<dataset>:<pred_len>:<seed>[:direct] ...     # sequentially

  itransformer  official run.py as is (":direct", seed 2023 only) or through itransformer_seed.py (any seed)

Each run gets code/output/<EXP>/<run_id>/ as its working directory, so the repository's own outputs
(checkpoints/, results/, test_results/, result_*.txt) land there next to cmd.txt, config.json, git.txt
(this repo + the external commit), seed.txt, env.txt, gpu.txt, stdout.log, metrics.json, status.json and
SHA256SUMS. The folder is then copied to D:/6CJNM007/backup/<EXP>/<run_id>/ and the checksums re-verified.
A run ID is never reused.
"""
import datetime
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = f"{REPO}/source"
BASH = "D:/6CJNM007/git/bin/bash.exe"
ENV_SH = "/d/6CJNM007/idea_rearch/code/env/windows_env.sh"
GIT = "D:/6CJNM007/git/cmd/git.exe"
ETT = "D:/6CJNM007/data/ETT-small"
BACKUP = "D:/6CJNM007/backup"


def itransformer(dataset, pred_len, seed, direct):
    """Official scripts/multivariate_forecasting/ETT/iTransformer_<dataset>.sh arguments, plus --num_workers 0."""
    width = {96: 256, 192: 256, 336: 512, 720: 512}[pred_len] if dataset == "ETTh1" else 128
    args = ["--is_training", "1", "--root_path", f"{ETT}/", "--data_path", f"{dataset}.csv",
            "--model_id", f"{dataset}_96_{pred_len}", "--model", "iTransformer", "--data", dataset, "--features", "M",
            "--seq_len", "96", "--pred_len", str(pred_len), "--e_layers", "2", "--enc_in", "7", "--dec_in", "7",
            "--c_out", "7", "--des", "Exp", "--d_model", str(width), "--d_ff", str(width), "--itr", "1",
            "--num_workers", "0"]
    if direct:
        if seed != 2023:
            raise SystemExit("a direct run.py run always uses its hard-coded seed 2023")
        return "itrans", ["python", f"{SRC}/iTransformer/run.py", *args], "iTransformer"
    return "itrans", ["python", f"{HERE}/itransformer_seed.py", f"{SRC}/iTransformer", str(seed), "--", *args], "iTransformer"


METHODS = {"itransformer": itransformer}


def tslib_metrics(run_dir):
    """TSLib-style repos save results/<setting>/metrics.npy = [mae, mse, rmse, mape, mspe] on the full test set."""
    found = glob.glob(f"{run_dir}/results/*/metrics.npy")
    if len(found) != 1:
        raise SystemExit(f"expected one results/*/metrics.npy, found {len(found)}")
    mae, mse, rmse, mape, mspe = np.load(found[0]).tolist()
    pred = np.load(os.path.join(os.path.dirname(found[0]), "pred.npy"), mmap_mode="r")
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe, "n_test_windows": int(pred.shape[0]),
            "setting": os.path.basename(os.path.dirname(found[0]))}


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


def run_one(exp, spec):
    run_id, method, dataset, pred_len, seed, *flags = spec.split(":")
    pred_len, seed, direct = int(pred_len), int(seed), flags == ["direct"]
    env, cmd, repo = METHODS[method](dataset, pred_len, seed, direct)
    run_dir = f"{REPO}/code/output/{exp}/{run_id}"
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_id} already exists; give the retry a new run ID")
    os.makedirs(run_dir)
    command = f"cd {run_dir} && " + shlex.join(cmd)
    with open(f"{run_dir}/cmd.txt", "w") as f:
        f.write(f"# env: {env}\n{shlex.join(in_env(env, command))}\n")
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump({"experiment": exp, "run_id": run_id, "method": method, "dataset": dataset, "seq_len": 96,
                   "pred_len": pred_len, "seed": seed, "direct_official_entry": direct, "env": env, "command": cmd,
                   "data_file": f"{ETT}/{dataset}.csv", "data_sha256": sha256(f"{ETT}/{dataset}.csv")}, f, indent=2)
    with open(f"{run_dir}/seed.txt", "w") as f:
        f.write(f"{seed}\n")
    with open(f"{run_dir}/git.txt", "w") as f:
        f.write(f"repo HEAD {sh([GIT, '-C', REPO, 'rev-parse', 'HEAD']).strip()} "
                f"({sh([GIT, '-C', REPO, 'branch', '--show-current']).strip()})\n"
                f"repo status --porcelain:\n{sh([GIT, '-C', REPO, 'status', '--porcelain']).rstrip()}\n"
                f"repo diff HEAD:\n{sh([GIT, '-C', REPO, 'diff', 'HEAD']).rstrip()}\n\n"
                f"source/{repo} {sh([GIT, '-C', f'{SRC}/{repo}', 'rev-parse', 'HEAD']).strip()} "
                f"{'clean' if not sh([GIT, '-C', f'{SRC}/{repo}', 'status', '--porcelain']).strip() else 'MODIFIED'}\n")
    with open(f"{run_dir}/env.txt", "w") as f:
        f.write(sh(in_env(env, "python --version && python -m pip freeze")))
    with open(f"{run_dir}/gpu.txt", "w") as f:
        f.write(sh(["nvidia-smi"]))

    status = {"run_id": run_id, "start": now()}
    with open(f"{run_dir}/stdout.log", "w") as log:
        status["exit"] = subprocess.run(in_env(env, command), stdout=log, stderr=subprocess.STDOUT).returncode
    status["end"] = now()
    metrics = {}
    if status["exit"] == 0:
        metrics = tslib_metrics(run_dir)
        with open(f"{run_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    status["status"] = "completed" if metrics else "failed"
    with open(f"{run_dir}/status.json", "w") as f:
        json.dump(status, f, indent=2)

    files = sorted(os.path.relpath(os.path.join(d, n), run_dir).replace("\\", "/")
                   for d, _, names in os.walk(run_dir) for n in names)
    sums = {p: sha256(f"{run_dir}/{p}") for p in files}
    with open(f"{run_dir}/SHA256SUMS", "w") as f:
        f.writelines(f"{h}  {p}\n" for p, h in sums.items())
    shutil.copytree(run_dir, f"{BACKUP}/{exp}/{run_id}")
    if any(sha256(f"{BACKUP}/{exp}/{run_id}/{p}") != h for p, h in sums.items()):
        raise SystemExit(f"backup of {run_id} does not match its checksums")

    row = [run_id, method, dataset, str(pred_len), str(seed), "direct" if direct else "wrapper", status["status"],
           str(metrics.get("mse", "NA")), str(metrics.get("mae", "NA")), str(metrics.get("n_test_windows", "NA")),
           status["start"], status["end"], sha256(f"{run_dir}/SHA256SUMS")]
    table = f"{REPO}/code/output/{exp}/runs.tsv"
    new = not os.path.exists(table)
    with open(table, "a") as f:
        if new:
            f.write("run_id\tmethod\tdataset\tpred_len\tseed\tentry\tstatus\tmse\tmae\tn_test\tstart\tend\tsha256sums_sha256\n")
        f.write("\t".join(row) + "\n")
    print("\t".join(row), flush=True)


def main():
    exp, specs = sys.argv[1], sys.argv[2:]
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in (5, 6) or parts[1] not in METHODS:
            raise SystemExit(f"bad run spec {spec}")
    for spec in specs:
        run_one(exp, spec)


if __name__ == "__main__":
    main()
