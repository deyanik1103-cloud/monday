"""Converts the trained PyTorch model (ckpt_chat.pt) into a NumPy file (monday_weights.npz)
for hosting, then CHECKS that the NumPy version gives the same answers as PyTorch.

Run it again every time you retrain:   python export_weights.py
"""
import argparse
import json

import numpy as np
import torch

from model import GPT, GPTConfig
from numpy_gpt import NumpyGPT

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt_chat.pt")
p.add_argument("--out", default="monday_weights.npz")
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cpu")
state = ck["model"]
arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items() if not k.endswith(".mask")}
arrays["config_json"] = np.array(json.dumps(ck["config"]))
np.savez(args.out, **arrays)
print(f"Saved {args.out}")

# ---- check: PyTorch and NumPy must agree ----
model = GPT(GPTConfig(**ck["config"]))
model.load_state_dict(state)
model.eval()
vocab = ck["config"]["vocab_size"]
ids = [int(i) for i in np.random.default_rng(0).integers(0, vocab, size=24)]
with torch.no_grad():
    reference = model(torch.tensor([ids]))[0][0].numpy()
mine = NumpyGPT.load(args.out).forward(ids, last_only=False)
diff = float(np.abs(reference - mine).max())
print(f"Biggest difference between PyTorch and NumPy: {diff:.2e}")
if diff < 1e-3:
    print("OK: the NumPy brain matches the PyTorch brain.")
else:
    raise SystemExit("MISMATCH: do not deploy this file. Send me this message.")
