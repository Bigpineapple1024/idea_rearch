"""Code-level probe: does DUETModel give different outputs for the same input in eval mode?

Random weights, ETTh1-like shape (7 channels, lookback 96, horizon 96), unified-script hyper-parameters.
Also reports how often the sampled channel mask differs between two calls.

  python probe_duet_mask.py <DUET checkout>
"""
import os
import sys
from types import SimpleNamespace

import torch

root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)
from ts_benchmark.baselines.duet.duet import MODEL_HYPER_PARAMS  # noqa: E402
from ts_benchmark.baselines.duet.models.duet_model import DUETModel  # noqa: E402

cfg = dict(MODEL_HYPER_PARAMS)
cfg.update({"CI": 1, "d_ff": 512, "d_model": 512, "dropout": 0.1, "e_layers": 2, "factor": 3, "fc_dropout": 0.1,
            "k": 2, "n_heads": 8, "num_experts": 4, "seq_len": 96, "pred_len": 96, "enc_in": 7})
torch.manual_seed(0)
model = DUETModel(SimpleNamespace(**cfg)).eval()
x = torch.randn(4, 96, 7)

with torch.no_grad():
    y1, _ = model(x)
    y2, _ = model(x)
    m1 = model.mask_generator(x.transpose(1, 2))
    m2 = model.mask_generator(x.transpose(1, 2))
    torch.manual_seed(123)
    y3, _ = model(x)
    torch.manual_seed(123)
    y4, _ = model(x)

print("model.training:", model.training)
print("same input, two eval calls -> max |y1 - y2| =", f"{(y1 - y2).abs().max().item():.4e}")
print("channel-mask entries that differ between two calls:", int((m1 != m2).sum()), "of", m1.numel())
print("same torch seed before each call -> max |y3 - y4| =", f"{(y3 - y4).abs().max().item():.4e}")
