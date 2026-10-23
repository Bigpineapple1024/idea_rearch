"""Run time-series reproduction runs with the per-run records the research process requires.

  python exp_ts.py <EXP> <run_id>:<method>:<dataset>:<pred_len>:<seed>[:direct] ...     # sequentially

  itransformer  official run.py as is (":direct", seed 2023 only) or through itransformer_seed.py (any seed)
  itransformer_val  no training: the checkpoint of the EXP-0004 iTransformer run with the same dataset, length and
                seed, scored on the validation split through itransformer_val.py (metrics.json describes validation)
  tqnet         official run.py through tqnet_save_pred.py (scripts/TQNet/etth1.sh arguments)
  wpmixer       official run_LTF.py through wpmixer_droplast.py (scripts/Unified/ETTh1_Unified_setup.sh arguments)
  duet          official TFB entry through duet_test_stochasticity.py (DUET_unified_seq_len_96.sh, ETTh1-96, TFB data)

Each run gets code/output/<EXP>/<run_id>/ as its working directory, so the repository's own outputs
(checkpoints/, results/, test_results/, result_*.txt) land there next to cmd.txt, config.json, git.txt
(this repo + the external commit), seed.txt, env.txt, gpu.txt, stdout.log, metrics.json, status.json and
SHA256SUMS. WPMixer and DUET read their data from and write into their own folder, so they run in an export of
the pinned commit (git archive) inside the run folder, and source/ stays untouched.
The folder is then copied to D:/6CJNM007/backup/<EXP>/<run_id>/ and the checksums re-verified.
A run ID is never reused.
"""
import datetime
import glob
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = f"{REPO}/source"
BASH = "D:/6CJNM007/git/bin/bash.exe"
ENV_SH = "/d/6CJNM007/idea_rearch/code/env/windows_env.sh"
GIT = "D:/6CJNM007/git/cmd/git.exe"
ETT = "D:/6CJNM007/data/ETT-small"
TFB = "D:/6CJNM007/data/tfb/forecasting"           # prepare_tfb_data.py
BACKUP = "D:/6CJNM007/backup"
# scripts/Unified/ETTh1_Unified_setup.sh, per prediction length
WPMIXER_ETTH1 = {
    96: {"learning_rate": "0.002991949", "wavelet": "coif4", "level": 1, "tfactor": 3, "dfactor": 3, "dropout": "0.1",
         "embedding_dropout": "0.2", "d_model": 256},
    192: {"learning_rate": "0.001472121", "wavelet": "coif5", "level": 3, "tfactor": 7, "dfactor": 8, "dropout": "0.4",
          "embedding_dropout": "0.2", "d_model": 256},
    336: {"learning_rate": "0.001581057", "wavelet": "sym4", "level": 3, "tfactor": 3, "dfactor": 8, "dropout": "0.2",
          "embedding_dropout": "0.0", "d_model": 256},
    720: {"learning_rate": "0.000380284", "wavelet": "sym2", "level": 3, "tfactor": 5, "dfactor": 8, "dropout": "0.4",
          "embedding_dropout": "0.0", "d_model": 128}}
DUET_ETTH1_96 = ('{"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, '
                 '"fc_dropout": 0.1, "horizon": 96, "k": 2, "loss": "MAE", "lr": 0.0005, "lradj": "type3", "n_heads": 8, '
                 '"norm": true, "num_epochs": 100, "num_experts": 4, "patch_len": 48, "patience": 5, "seq_len": 96}')


def snapshot(repo, run_dir):
    """Export source/<repo> at HEAD into <run_dir>/<repo> and return that path."""
    tar = subprocess.run([GIT, "-C", f"{SRC}/{repo}", "archive", "--format=tar", "HEAD"], capture_output=True,
                         check=True).stdout
    dest = f"{run_dir}/{repo}"
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest)
    return dest


def itransformer(dataset, pred_len, seed, direct, run_dir):
    """Official scripts/multivariate_forecasting/ETT/iTransformer_<dataset>.sh arguments, plus --num_workers 0."""
    width = {96: 256, 192: 256, 336: 512, 720: 512}[pred_len] if dataset == "ETTh1" else 128
    args = ["--is_training", "1", "--root_path", f"{ETT}/", "--data_path", f"{dataset}.csv",
            "--model_id", f"{dataset}_96_{pred_len}", "--model", "iTransformer", "--data", dataset, "--features", "M",
            "--seq_len", "96", "--pred_len", str(pred_len), "--e_layers", "2", "--enc_in", "7", "--dec_in", "7",
            "--c_out", "7", "--des", "Exp", "--d_model", str(width), "--d_ff", str(width), "--itr", "1",
            "--num_workers", "0"]
    data = [f"{ETT}/{dataset}.csv"]
    if direct:
        if seed != 2023:
            raise SystemExit("a direct run.py run always uses its hard-coded seed 2023")
        return "itrans", ["python", f"{SRC}/iTransformer/run.py", *args], "iTransformer", data
    return ("itrans", ["python", f"{HERE}/itransformer_seed.py", f"{SRC}/iTransformer", str(seed), "--", *args],
            "iTransformer", data)


def trained_run(exp, method, dataset, pred_len, seed):
    """The first completed run of <exp> with this method, dataset, length and seed (runs.tsv order)."""
    with open(f"{REPO}/code/output/{exp}/runs.tsv") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    for r in rows:
        if r[1:5] == [method, dataset, str(pred_len), str(seed)] and r[6] == "completed":
            return f"{REPO}/code/output/{exp}/{r[0]}"
    raise SystemExit(f"no completed {method} run of {exp} for {dataset}-{pred_len}, seed {seed}")


def itransformer_val(dataset, pred_len, seed, direct, run_dir):
    """The trained EXP-0004 run's checkpoint, copied into ./checkpoints/<setting>/, scored on the validation split."""
    if direct:
        raise SystemExit("itransformer_val has no direct entry")
    source = trained_run("EXP-0004", "itransformer", dataset, pred_len, seed)
    found = glob.glob(f"{source}/checkpoints/*/checkpoint.pth")
    if len(found) != 1:
        raise SystemExit(f"expected one checkpoint in {source}, found {len(found)}")
    setting = os.path.basename(os.path.dirname(found[0]))
    os.makedirs(f"{run_dir}/checkpoints/{setting}")
    shutil.copyfile(found[0], f"{run_dir}/checkpoints/{setting}/checkpoint.pth")
    with open(f"{run_dir}/source_run.txt", "w") as f:
        f.write(f"{os.path.basename(source)}\n{found[0]}\n")
    _, cmd, repo, data = itransformer(dataset, pred_len, seed, False, run_dir)
    args = cmd[cmd.index("--") + 1:]
    args[args.index("--is_training") + 1] = "0"
    return ("itrans", ["python", f"{HERE}/itransformer_val.py", f"{SRC}/iTransformer", "--", *args], repo,
            [*data, f"{run_dir}/checkpoints/{setting}/checkpoint.pth"])


def tqnet(dataset, pred_len, seed, direct, run_dir):
    """scripts/TQNet/etth1.sh arguments with --random_seed <seed>, plus --num_workers 0."""
    if dataset != "ETTh1" or direct:
        raise SystemExit("tqnet: only the ETTh1 script's arguments are wired in")
    args = ["--is_training", "1", "--root_path", f"{ETT}/", "--data_path", "ETTh1.csv", "--model_id",
            f"ETTh1_96_{pred_len}", "--model", "TQNet", "--data", "ETTh1", "--features", "M", "--seq_len", "96",
            "--pred_len", str(pred_len), "--enc_in", "7", "--cycle", "24", "--train_epochs", "30", "--patience", "5",
            "--dropout", "0.5", "--itr", "1", "--batch_size", "256", "--learning_rate", "0.001",
            "--random_seed", str(seed), "--num_workers", "0"]
    return ("tqnet", ["python", f"{HERE}/tqnet_save_pred.py", f"{SRC}/TQNet", "--", *args], "TQNet",
            [f"{ETT}/ETTh1.csv"])


def wpmixer(dataset, pred_len, seed, direct, run_dir):
    """scripts/Unified/ETTh1_Unified_setup.sh arguments (with its --use_amp) and --seed <seed>, in an export."""
    if dataset != "ETTh1" or direct:
        raise SystemExit("wpmixer: only the ETTh1 unified-setup arguments are wired in")
    p = WPMIXER_ETTH1[pred_len]
    root = snapshot("WPMixer", run_dir)
    os.makedirs(f"{root}/data/ETT")
    shutil.copyfile(f"{ETT}/ETTh1.csv", f"{root}/data/ETT/ETTh1.csv")      # run_LTF.py reads ./data/ETT/
    args = ["--model", "WPMixer", "--task_name", "long_term_forecast", "--data", "ETTh1", "--seq_len", "96",
            "--pred_len", str(pred_len), "--d_model", str(p["d_model"]), "--tfactor", str(p["tfactor"]),
            "--dfactor", str(p["dfactor"]), "--wavelet", p["wavelet"], "--level", str(p["level"]), "--patch_len", "16",
            "--stride", "8", "--batch_size", "128", "--learning_rate", p["learning_rate"], "--lradj", "type3",
            "--dropout", p["dropout"], "--embedding_dropout", p["embedding_dropout"], "--patience", "5",
            "--train_epochs", "10", "--use_amp", "--seed", str(seed)]
    return ("wpmixer", ["python", f"{HERE}/wpmixer_droplast.py", root, "--", *args], "WPMixer",
            [f"{root}/data/ETT/ETTh1.csv"])


def duet(dataset, pred_len, seed, direct, run_dir):
    """DUET_unified_seq_len_96.sh, ETTh1-96, with --seed <seed>, on TFB's data, in an export."""
    if (dataset, pred_len) != ("ETTh1", 96) or direct:
        raise SystemExit("duet: only ETTh1-96 is wired in")
    root = snapshot("DUET", run_dir)
    os.makedirs(f"{root}/dataset/forecasting")
    for name in ("ETTh1.csv", "FORECAST_META.csv"):                  # TFB reads <root>/dataset/forecasting/
        shutil.copyfile(f"{TFB}/{name}", f"{root}/dataset/forecasting/{name}")
    args = ["--config-path", "rolling_forecast_config.json", "--data-name-list", "ETTh1.csv",
            "--strategy-args", '{"horizon": 96}', "--model-name", "duet.DUET", "--model-hyper-params", DUET_ETTH1_96,
            "--deterministic", "full", "--gpus", "0", "--num-workers", "1", "--timeout", "60000",
            "--save-path", "ETTh1/DUET", "--seed", str(seed)]
    return ("duet", ["python", f"{HERE}/duet_test_stochasticity.py", root, "--resamples", "1", "--save-pred", run_dir,
                     "--", *args], "DUET", [f"{root}/dataset/forecasting/ETTh1.csv"])


METHODS = {"itransformer": itransformer, "itransformer_val": itransformer_val, "tqnet": tqnet, "wpmixer": wpmixer,
           "duet": duet}


def tslib_metrics(run_dir):
    """TSLib-style repos save results/<setting>/metrics.npy = [mae, mse, rmse, mape, mspe] on the full test set."""
    found = glob.glob(f"{run_dir}/results/*/metrics.npy")
    if len(found) != 1:
        raise SystemExit(f"expected one results/*/metrics.npy, found {len(found)}")
    mae, mse, rmse, mape, mspe = np.load(found[0]).tolist()
    pred = np.load(os.path.join(os.path.dirname(found[0]), "pred.npy"), mmap_mode="r")
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe, "n_test_windows": int(pred.shape[0]),
            "setting": os.path.basename(os.path.dirname(found[0]))}


def result_line(run_dir, tag):
    with open(f"{run_dir}/stdout.log", errors="replace") as f:
        lines = [line for line in f if line.startswith(tag + " ")]
    if len(lines) != 1:
        raise SystemExit(f"expected one {tag} line in stdout.log, found {len(lines)}")
    return json.loads(lines[0][len(tag) + 1:])


def saved_scores(run_dir, pred_name, true_name):
    pred, true = np.load(f"{run_dir}/{pred_name}"), np.load(f"{run_dir}/{true_name}")
    return {"mse": float(np.mean((pred - true) ** 2)), "mae": float(np.mean(np.abs(pred - true))),
            "n_test_windows": int(pred.shape[0])}


def metrics_of(method, run_dir):
    if method in ("itransformer", "itransformer_val"):
        return tslib_metrics(run_dir)
    if method == "tqnet":                       # official result.txt line "mse:<x>, mae:<y>" and the saved arrays
        with open(f"{run_dir}/result.txt") as f:
            official = [line for line in f if line.startswith("mse:")]
        if len(official) != 1:
            raise SystemExit(f"expected one mse line in result.txt, found {len(official)}")
        mse_text, mae_text = official[0].strip().split(", ")
        return {**saved_scores(run_dir, "pred.npy", "true.npy"), "official_mse": float(mse_text.split(":")[1]),
                "official_mae": float(mae_text.split(":")[1])}
    if method == "wpmixer":                     # all windows is the primary score; the repository's drop_last too
        r = result_line(run_dir, "DROPLAST_RESULT")
        return {"mse": r["all_windows"]["mse"], "mae": r["all_windows"]["mae"],
                "n_test_windows": r["all_windows"]["windows_scored"],
                "drop_last_mse": r["official_drop_last_true"]["mse"], "drop_last_mae": r["official_drop_last_true"]["mae"],
                "drop_last_windows": r["official_drop_last_true"]["windows_scored"], "setting": r["setting"]}
    r = result_line(run_dir, "STOCHASTICITY_RESULT")
    return {"mse": r["official"]["mse_norm"], "mae": r["official"]["mae_norm"], "n_test_windows": r["windows"],
            "mask_resample": r["resampled_masks"]}


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
    run_dir = f"{REPO}/code/output/{exp}/{run_id}"
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_id} already exists; give the retry a new run ID")
    os.makedirs(run_dir)
    env, cmd, repo, data = METHODS[method](dataset, pred_len, seed, direct, run_dir)
    command = f"cd {run_dir} && " + shlex.join(cmd)
    with open(f"{run_dir}/cmd.txt", "w") as f:
        f.write(f"# env: {env}\n{shlex.join(in_env(env, command))}\n")
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump({"experiment": exp, "run_id": run_id, "method": method, "dataset": dataset, "seq_len": 96,
                   "pred_len": pred_len, "seed": seed, "direct_official_entry": direct, "env": env, "command": cmd,
                   "data_file": data[0], "data_sha256": sha256(data[0]),
                   "data_files_sha256": {p: sha256(p) for p in data}}, f, indent=2)
    with open(f"{run_dir}/seed.txt", "w") as f:
        f.write(f"{seed}\n")
    exported = os.path.isdir(f"{run_dir}/{repo}")
    with open(f"{run_dir}/git.txt", "w") as f:
        f.write(f"repo HEAD {sh([GIT, '-C', REPO, 'rev-parse', 'HEAD']).strip()} "
                f"({sh([GIT, '-C', REPO, 'branch', '--show-current']).strip()})\n"
                f"repo status --porcelain:\n{sh([GIT, '-C', REPO, 'status', '--porcelain']).rstrip()}\n"
                f"repo diff HEAD:\n{sh([GIT, '-C', REPO, 'diff', 'HEAD']).rstrip()}\n\n"
                f"source/{repo} {sh([GIT, '-C', f'{SRC}/{repo}', 'rev-parse', 'HEAD']).strip()} "
                f"{'clean' if not sh([GIT, '-C', f'{SRC}/{repo}', 'status', '--porcelain']).strip() else 'MODIFIED'}"
                f"{f'; run from a git archive export of that commit in {repo}/' if exported else ''}\n")
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
        try:
            metrics = metrics_of(method, run_dir)
        except (SystemExit, OSError, ValueError, KeyError) as e:   # keep the run folder and record why
            status["metrics_error"] = str(e)
        if metrics:
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
