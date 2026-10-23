"""Do DUET (TFB data) and iTransformer/TQNet/WPMixer (ETDataset CSV) score the same ETTh1 test windows?

  python compare_tfb_ett.py <TFB ETTh1.csv> <ETDataset ETTh1.csv> <DUET run dir> <iTransformer run dir> > out.json

1. Raw values: TFB's long table (date, data, cols) pivoted to wide, against the first rows of ETDataset's wide CSV:
   dates, column names and the largest absolute difference.
2. Test windows: DUET's saved true.npy (from duet_test_stochasticity.py --save-pred) against iTransformer's
   results/*/true.npy, both on the training-set standardised scale: window counts, the column matching, the largest
   absolute difference and the number of windows that differ by more than 1e-4.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

tfb_csv, ett_csv, duet_dir, itr_dir = sys.argv[1:5]
report = {}

long = pd.read_csv(tfb_csv)
wide_tfb = long.pivot(index="date", columns="cols", values="data")
ett = pd.read_csv(ett_csv)
n = len(wide_tfb)
ett_head = ett.iloc[:n].set_index("date")
report["tfb_rows"], report["ett_rows"] = n, len(ett)
report["tfb_columns"], report["ett_columns"] = sorted(wide_tfb.columns), list(ett_head.columns)
report["same_dates"] = bool((pd.to_datetime(wide_tfb.index) == pd.to_datetime(ett_head.index)).all())
common = [c for c in ett_head.columns if c in wide_tfb.columns]
report["raw_max_abs_diff"] = float(np.abs(wide_tfb.loc[:, common].to_numpy() - ett_head.loc[:, common].to_numpy()).max())

duet_true = np.load(os.path.join(duet_dir, "true.npy"))
itr_true = np.load(glob.glob(os.path.join(itr_dir, "results", "*", "true.npy"))[0])
report["duet_windows"], report["itransformer_windows"] = list(duet_true.shape), list(itr_true.shape)
if duet_true.shape == itr_true.shape:
    # match each DUET variable to the iTransformer variable it equals best on the first window
    mapping = [int(np.argmin([np.abs(duet_true[0, :, i] - itr_true[0, :, j]).max() for j in range(itr_true.shape[2])]))
               for i in range(duet_true.shape[2])]
    aligned = itr_true[:, :, mapping]
    diff = np.abs(duet_true - aligned)
    report["column_mapping_duet_to_itransformer"] = mapping
    report["window_max_abs_diff"] = float(diff.max())
    report["windows_differing_over_1e-4"] = int((diff.max(axis=(1, 2)) > 1e-4).sum())
print(json.dumps(report, indent=2))
