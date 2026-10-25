"""Trains Monday on data/chat.txt using our own BPE tokenizer."""
import argparse
import os
import time

import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/chat.txt")
p.add_argument("--tok", default="tokenizer.json")
p.add_argument("--out", default="ckpt_chat.pt")
p.add_argument("--vocab_size", type=int, default=500)
p.add_argument("--retrain_tokenizer", action="store_true")
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--block_size", type=int, default=128)
p.add_argument("--n_embd", type=int, default=128)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--eval_every", type=int, default=250)
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

text = open(args.data, encoding="utf-8").read()

# ---- tokenizer: train our own BPE (once), then reuse ----
if os.path.exists(args.tok) and not args.retrain_tokenizer:
    tok = BPETokenizer.load(args.tok)
    print(f"Loaded tokenizer from {args.tok}")
else:
    print("Training our BPE tokenizer...")
    tok = BPETokenizer()
    tok.train(text, args.vocab_size)
    tok.save(args.tok)
    print(f"Tokenizer saved to {args.tok}")

ids = tok.encode(text)
print(f"{len(text):,} characters -> {len(ids):,} tokens | vocabulary size {tok.vocab_size}")
data = torch.tensor(ids, dtype=torch.long)
if len(data) < args.block_size + 2:
    raise SystemExit("Dataset is too small for this block_size.")
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(d):
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i:i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
                n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout)
model = GPT(cfg).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for name, d in [("train", train_data), ("val", val_data)]:
        losses = [model(*get_batch(d))[1].item() for _ in range(20)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def save():
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, args.out)


start = time.time()
for step in range(args.steps + 1):
    if step % args.eval_every == 0:
        l = estimate_loss()
        print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} | {time.time()-start:.0f}s")
        save()
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

save()
print(f"Training finished. Model saved to {args.out}")
print("Now run:  python chat.py")
