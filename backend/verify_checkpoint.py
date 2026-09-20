import sys
import os
import torch
import safetensors.torch
import numpy as np

_CURRENT = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_CURRENT)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)
if _CURRENT not in sys.path:
    sys.path.insert(0, _CURRENT)

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from backend.models.bigearthnet_loader import DEFAULT_CHECKPOINT_DIR

ckpt = DEFAULT_CHECKPOINT_DIR
print(f"Loading checkpoint from: {ckpt}")

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(ckpt)
weights = safetensors.torch.load_file(os.path.join(ckpt, "model.safetensors"))
load_res = model.load_state_dict(weights, strict=True)
p_count = sum(p.numel() for p in model.parameters())

model.eval()
model.to("cpu")

torch.manual_seed(42)
x = torch.ones((1, 10, 120, 120), dtype=torch.float32) * 0.5
with torch.no_grad():
    logits = model(x)
probs = torch.sigmoid(logits)

print(f"CHECKPOINT_LOADED: True")
print(f"PARAM_COUNT: {p_count}")
print(f"MISSING_KEYS: {len(load_res.missing_keys)}")
print(f"UNEXPECTED_KEYS: {len(load_res.unexpected_keys)}")
print(f"INPUT_SHAPE: {tuple(x.shape)}")
print(f"OUTPUT_SHAPE: {tuple(logits.shape)}")
print(f"PROB_MIN: {probs.min().item():.8f}")
print(f"PROB_MAX: {probs.max().item():.8f}")
print(f"SAMPLE_PROBABILITIES: {[round(p, 6) for p in probs[0][:5].tolist()]}")
