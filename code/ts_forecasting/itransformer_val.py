"""Validation-period predictions from a trained iTransformer checkpoint, without modifying the official code.

run.py with --is_training 0 loads ./checkpoints/<setting>/checkpoint.pth and calls exp.test(setting, test=1), which
scores the test split in order with batch size 1 and saves results/<setting>/pred.npy, true.npy and metrics.npy. This
wrapper replaces the data_provider that experiments/exp_long_term_forecasting.py imported, so that exp.test() gets the
validation split with those same loader settings, then executes run.py as __main__. Everything run.py then saves
describes the validation windows. Nothing is trained.

  python itransformer_val.py <iTransformer checkout> -- <run.py arguments with --is_training 0>
"""
import os
import sys

root = os.path.abspath(sys.argv[1])
run_args = sys.argv[sys.argv.index("--") + 1:]
if run_args[run_args.index("--is_training") + 1] != "0":
    raise SystemExit("itransformer_val.py needs --is_training 0")
sys.path.insert(0, root)                     # run.py imports experiments/, data_provider/, model/ from its own folder
from torch.utils.data import DataLoader      # noqa: E402

import data_provider.data_factory as factory                # noqa: E402
import experiments.exp_long_term_forecasting as long_term   # noqa: E402


def validation_as_test(args, flag):
    if flag != "test":
        return factory.data_provider(args, flag)
    data_set = factory.data_dict[args.data](
        root_path=args.root_path, data_path=args.data_path, flag="val", size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, timeenc=0 if args.embed != "timeF" else 1, freq=args.freq)
    print("val (handed to test)", len(data_set))
    return data_set, DataLoader(data_set, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=True)


long_term.data_provider = validation_as_test
path = os.path.join(root, "run.py")
with open(path, encoding="utf-8") as f:
    source = f.read()
sys.argv = [path] + run_args
exec(compile(source, path, "exec"), {"__name__": "__main__", "__file__": path})
