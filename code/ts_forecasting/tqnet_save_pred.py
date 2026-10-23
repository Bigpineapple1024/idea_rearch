"""Run TQNet's official run.py and keep the test predictions it scores, without modifying the repository.

TQNet's Exp_Main.test computes metric(preds, trues) over every test window but leaves its np.save calls commented
out. This wrapper replaces exp_main.metric, inside this process only, with one that first saves preds and trues
(standardised scale, windows x pred_len x variables) as pred.npy and true.npy in the working directory and then
returns the original metric unchanged.

  python tqnet_save_pred.py <TQNet checkout> -- <run.py arguments>
"""
import os
import runpy
import sys

import numpy as np

root = os.path.abspath(sys.argv[1])
run_args = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, root)

import exp.exp_main as exp_main  # noqa: E402

official_metric = exp_main.metric


def saving_metric(pred, true):
    np.save("pred.npy", pred)
    np.save("true.npy", true)
    return official_metric(pred, true)


exp_main.metric = saving_metric
sys.argv = [os.path.join(root, "run.py")] + run_args
runpy.run_path(os.path.join(root, "run.py"), run_name="__main__")
