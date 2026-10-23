"""Run DUET through its official TFB entry point and measure how much the test score moves when
only the random channel mask is re-sampled at inference time.

DUET's Mahalanobis_mask draws a hard Gumbel sample in every forward pass, including model.eval().
This script leaves the official prediction untouched (same RNG stream as a normal run) and, for
each test batch, also predicts with --resamples other torch seeds, restoring the RNG afterwards.

  python duet_test_stochasticity.py <DUET checkout> --resamples 10 -- <run_benchmark.py arguments>

Prints one JSON line with the official mse_norm/mae_norm and the spread over the resampled masks.
Scores are computed on the training-set StandardScaler scale, as TFB's *_norm metrics are.
"""
import argparse
import json
import os
import runpy
import sys

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--resamples", type=int, default=10)
argv = sys.argv[1:]
split = argv.index("--")
a = p.parse_args(argv[:split])
bench_args = argv[split + 1:]

root = os.path.abspath(a.root)
os.chdir(root)
sys.path.insert(0, root)

from ts_benchmark.baselines import deep_forecasting_model_base as base  # noqa: E402
from ts_benchmark.evaluation.strategy import rolling_forecast as rf  # noqa: E402

state = {"official": [], "resampled": [[] for _ in range(a.resamples)], "targets": None, "scaler": None}

original_rolling = base.DeepForecastingModelBase._perform_rolling_predictions


def rolling_with_resamples(self, horizon, input_np, all_mark, device):
    official = original_rolling(self, horizon, input_np, all_mark, device)
    state["official"].append(np.asarray(official))
    rng = torch.get_rng_state()
    for k in range(a.resamples):
        torch.manual_seed(10_000 + k)
        state["resampled"][k].append(np.asarray(original_rolling(self, horizon, input_np, all_mark, device)))
    torch.set_rng_state(rng)
    return official


original_eval_batch = rf.RollingForecastEvalBatchMaker.make_batch_eval


def capture_targets(self, horizon):
    out = original_eval_batch(self, horizon)
    state["targets"] = out["target"]
    return out


original_scaler = rf.RollingForecast._get_eval_scaler


def capture_scaler(self, data, ratio):
    scaler = original_scaler(self, data, ratio)
    state["scaler"] = scaler
    return scaler


base.DeepForecastingModelBase._perform_rolling_predictions = rolling_with_resamples
rf.RollingForecastEvalBatchMaker.make_batch_eval = capture_targets
rf.RollingForecast._get_eval_scaler = capture_scaler

sys.argv = ["run_benchmark.py"] + bench_args
runpy.run_path(os.path.join(root, "scripts", "run_benchmark.py"), run_name="__main__")


def scores(pred_norm_batches):
    pred = np.concatenate(pred_norm_batches, axis=0)
    tgt = np.asarray(state["targets"])
    tgt_norm = state["scaler"].transform(tgt.reshape(-1, tgt.shape[-1])).reshape(tgt.shape)
    pred = pred[:, -tgt.shape[1]:, :tgt.shape[2]]
    return float(((pred - tgt_norm) ** 2).mean()), float(np.abs(pred - tgt_norm).mean())


official = scores(state["official"])
resampled = [scores(r) for r in state["resampled"]]
mse = np.array([r[0] for r in resampled])
print("STOCHASTICITY_RESULT " + json.dumps({
    "windows": int(len(state["targets"])),
    "official": {"mse_norm": official[0], "mae_norm": official[1]},
    "resampled_masks": {"n": a.resamples, "mse_min": float(mse.min()), "mse_max": float(mse.max()),
                        "mse_mean": float(mse.mean()), "mse_std": float(mse.std(ddof=1)) if len(mse) > 1 else 0.0},
}))
