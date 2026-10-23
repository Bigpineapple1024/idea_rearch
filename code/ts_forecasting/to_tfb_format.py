"""Convert a wide ETT CSV (date + one column per variable) into TFB's long format and metadata.

TFB/DUET read <root>/dataset/forecasting/<name>.csv with columns date,data,cols and take the
series length from FORECAST_META.csv. With length 14400 for ETTh1, TFB's split (tv_ratio 0.8,
train_ratio_in_tv 0.75) gives 8640/2880/2880 rows, the same 12/4/4-month split used by
TSLib-style repositories (TQNet, WPMixer).

  python to_tfb_format.py <ETTh1.csv> <DUET checkout> [--length 14400]
"""
import argparse
import os

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("wide_csv")
p.add_argument("duet_root")
p.add_argument("--length", type=int, default=14400)
a = p.parse_args()

wide = pd.read_csv(a.wide_csv).iloc[:a.length]
name = os.path.basename(a.wide_csv)
long = wide.melt(id_vars="date", var_name="cols", value_name="data")[["date", "data", "cols"]]
out_dir = os.path.join(a.duet_root, "dataset", "forecasting")
os.makedirs(out_dir, exist_ok=True)
long.to_csv(os.path.join(out_dir, name), index=False)
meta = pd.DataFrame([{"file_name": name, "freq": "hourly", "if_univariate": False, "size": "large",
                      "length": len(wide), "trend": "", "seasonal": "", "stationary": "", "transition": "",
                      "shifting": "", "correlation": ""}])
meta.to_csv(os.path.join(out_dir, "FORECAST_META.csv"), index=False)
print(f"wrote {len(long)} rows ({len(wide)} time steps x {wide.shape[1] - 1} variables) to {out_dir}")
