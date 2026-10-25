"""Loads the trained model and writes text."""
import argparse

import torch

from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--prompt", default="\n")
p.add_argument("--tokens", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(args.ckpt, map_location=device)
chars = ck["chars"]
stoi = {c: i for i, c in enumerate(chars)}
model = GPT(GPTConfig(**ck["config"])).to(device)
model.load_state_dict(ck["model"])

prompt = "".join(c for c in args.prompt if c in stoi) or chars[0]
idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
out = model.generate(idx, args.tokens, args.temperature, args.top_k)[0].tolist()
print("".join(chars[i] for i in out))
