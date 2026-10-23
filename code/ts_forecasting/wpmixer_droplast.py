"""Run WPMixer's official run_LTF.py and score the same checkpoint on the test set twice:
with the repository's test loader (drop_last=True) and with every test window (drop_last=False).

  python wpmixer_droplast.py <WPMixer checkout> -- <run_LTF.py arguments>

Prints one JSON line with both scores and the number of test windows each one used, and saves the all-windows
predictions and targets (standardised scale) as pred_all_windows.npy and true_all_windows.npy in the directory it
was started from. The repository files are not modified; data_provider and metric are swapped only inside this
process.
"""
import json
import os
import runpy
import sys

import numpy as np

start_dir = os.getcwd()
root = os.path.abspath(sys.argv[1])
llm_args = sys.argv[sys.argv.index("--") + 1:]
os.chdir(root)
sys.path.insert(0, root)

from torch.utils.data import DataLoader  # noqa: E402

import exp.exp_main as exp_main  # noqa: E402

official_provider = exp_main.data_provider
official_test = exp_main.Exp_Main.test
official_metric = exp_main.metric
used = {}


def all_windows_provider(args, flag):
    dataset, loader = official_provider(args, flag)
    if flag == "test":
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataset, loader


def counting(provider, key):
    def wrapped(args, flag):
        dataset, loader = provider(args, flag)
        if flag == "test":
            used[key] = {"windows_in_test_set": len(dataset),
                         "windows_scored": len(loader) * args.batch_size if loader.drop_last else len(dataset)}
        return dataset, loader
    return wrapped


def saving_metric(pred, true):
    np.save(os.path.join(start_dir, "pred_all_windows.npy"), pred)
    np.save(os.path.join(start_dir, "true_all_windows.npy"), true)
    return official_metric(pred, true)


def test_both(self, setting):
    exp_main.data_provider = counting(official_provider, "official_drop_last_true")
    mse_official, mae_official = official_test(self, setting)
    exp_main.data_provider = counting(all_windows_provider, "all_windows")
    exp_main.metric = saving_metric
    try:
        mse_all, mae_all = official_test(self, setting)
    finally:
        exp_main.data_provider = official_provider
        exp_main.metric = official_metric
    print("DROPLAST_RESULT " + json.dumps({
        "setting": setting,
        "official_drop_last_true": {"mse": float(mse_official), "mae": float(mae_official), **used["official_drop_last_true"]},
        "all_windows": {"mse": float(mse_all), "mae": float(mae_all), **used["all_windows"]},
    }))
    return mse_official, mae_official


exp_main.Exp_Main.test = test_both
sys.argv = ["run_LTF.py"] + llm_args
runpy.run_path(os.path.join(root, "run_LTF.py"), run_name="__main__")
