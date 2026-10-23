"""Code-level probes of VisionZip (dvlab-research/VisionZip, visionzip/clip_encoder.py + utils.py).

1. Random-weight CLIP-L/14-336 layout (24 layers, 576 patches), tiny width: output token count,
   whether the CLS token is passed on, which layer's attention/metric is used.
2. Crafted input: all non-dominant tokens identical -> what a contextual (merged) token equals.
Usage: python probe_visionzip.py <path to VisionZip repo>
"""
import importlib
import sys
import types

import torch
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers.models.clip import modeling_clip as mc

repo = sys.argv[1]
pkg = types.ModuleType("vzpkg")                    # load utils/clip_encoder without visionzip/__init__ (needs llava)
pkg.__path__ = [f"{repo}/visionzip"]
sys.modules["vzpkg"] = pkg
vz_utils = importlib.import_module("vzpkg.utils")
vz_enc = importlib.import_module("vzpkg.clip_encoder")

DOMINANT, CONTEXTUAL = 54, 10                      # README default: model = visionzip(model, dominant=54, contextual=10)

# 1. tiny random CLIP vision tower with the LLaVA-1.5 geometry
torch.manual_seed(0)
cfg = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=24, num_attention_heads=4,
                       image_size=336, patch_size=14)
clip = CLIPVisionModel(cfg).eval()
vz_utils.apply_info(clip, dominant_num=DOMINANT - 1, contextual_num=CONTEXTUAL)   # as visionzip() in main.py
mc.CLIPEncoderLayer.forward = vz_utils.CLIP_EncoderLayer_forward
mc.CLIPAttention.forward = vz_utils.CLIPAttention_forward

tower = types.SimpleNamespace(vision_tower=clip, device=torch.device("cpu"), dtype=torch.float32)
images = torch.randn(1, 3, 336, 336)
with torch.no_grad():
    feats, keep_idx = vz_enc.CLIPVisionTower_VisionZip.forward(tower, images)
    ref = clip(images, output_hidden_states=True)   # plain pass for comparison (patched forwards still apply)
print("== 1. random CLIP tower")
print("output tokens:", feats.shape[1], f"(dominant incl. CLS = {DOMINANT}, contextual = {CONTEXTUAL})")
print("token 0 equals CLS hidden state of layer -2:", torch.allclose(feats[0, 0], ref.hidden_states[-2][0, 0], atol=1e-5))
print("index 0 (CLS) in keep_idx:", bool((keep_idx == 0).any()))
kept = keep_idx[0, 1:].sort().values
print("dominant patch tokens are passed in spatial order:",
      torch.allclose(feats[0, 1:DOMINANT], ref.hidden_states[-2][0, kept], atol=1e-5))
print("metric layer = encoder layer with r=1:", vz_utils.parse_r(24, clip.r).index(1), "(0-based; 22 = second to last)")

# 2. crafted outputs: every token has the same hidden vector v
print("\n== 2. merge rule on identical tokens")
N, D = 577, 8
v = torch.ones(1, N, D)
attn = torch.rand(1, 4, N, N).softmax(-1)
metric_holder = types.SimpleNamespace(metric=torch.randn(1, N, 16))
fake_out = types.SimpleNamespace(attentions=[attn, attn], hidden_states=[v, v])
fake_tower = types.SimpleNamespace(
    _info={"dominant": DOMINANT - 1, "contextual": CONTEXTUAL},
    vision_model=types.SimpleNamespace(encoder=types.SimpleNamespace(layers=[metric_holder, metric_holder])),
    __call__=None)
fake = types.SimpleNamespace(vision_tower=lambda *a, **k: fake_out, device="cpu", dtype=torch.float32)
fake.vision_tower._info = fake_tower._info
fake.vision_tower.vision_model = fake_tower.vision_model
out, _ = vz_enc.CLIPVisionTower_VisionZip.forward(fake, torch.zeros(1, 3, 2, 2))
print("dominant token value:", out[0, 1, 0].item(), "| contextual token values:", sorted(set(out[0, DOMINANT:, 0].tolist())))
