"""Monday's brain, runs with NumPy only (no PyTorch).

Training needs PyTorch, but ANSWERING only needs the forward pass of the model, which is
just a few matrix multiplications. Writing it in plain NumPy means the hosted app needs
about 30 MB of memory instead of several hundred, and it installs in seconds.
"""
import json
import math

import numpy as np


def erf(x):
    """Error function (Abramowitz and Stegun 7.1.26, error below 1.5e-7)."""
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-a * a))


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class NumpyGPT:
    def __init__(self, weights, config):
        self.w = weights
        self.cfg = config
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.block_size = config["block_size"]

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        config = json.loads(str(data["config_json"]))
        weights = {k: data[k].astype(np.float32) for k in data.files if k != "config_json"}
        return cls(weights, config)

    @property
    def n_params(self):
        # the output layer shares its weights with the token embedding, so count them once
        return int(sum(v.size for k, v in self.w.items() if k != "head.weight"))

    def forward(self, ids, last_only=True):
        """ids: list of token ids. Returns logits for the last position (or for every position)."""
        w = self.w
        ids = np.asarray(ids[-self.block_size:])
        T = len(ids)
        x = w["tok_emb.weight"][ids] + w["pos_emb.weight"][:T]            # (T, C)
        C = x.shape[1]
        hs = C // self.n_head
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)                 # True = "future, not allowed"

        for i in range(self.n_layer):
            p = f"blocks.{i}."
            # ---- self-attention ----
            h = layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, self.n_head, hs).transpose(1, 0, 2)           # (heads, T, hs)
            k = k.reshape(T, self.n_head, hs).transpose(1, 0, 2)
            v = v.reshape(T, self.n_head, hs).transpose(1, 0, 2)
            att = (q @ k.transpose(0, 2, 1)) / math.sqrt(hs)               # (heads, T, T)
            att = np.where(causal, -np.inf, att)
            att = softmax(att)
            y = (att @ v).transpose(1, 0, 2).reshape(T, C)
            x = x + (y @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"])
            # ---- feed-forward ----
            h = layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = gelu(h @ w[p + "ff.net.0.weight"].T + w[p + "ff.net.0.bias"])
            x = x + (h @ w[p + "ff.net.2.weight"].T + w[p + "ff.net.2.bias"])

        x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
        if last_only:
            x = x[-1]
        return x @ w["head.weight"].T

    # ------------------------------------------------------------------
    # Fast generation with a key/value cache.
    # The slow way recomputes every earlier word for each new word. But earlier words never change,
    # so we remember their keys and values and only compute the ONE new word. Same answers, far less work.
    # ------------------------------------------------------------------
    def new_cache(self):
        C = self.w["tok_emb.weight"].shape[1]
        hs = C // self.n_head
        shape = (self.n_head, self.block_size, hs)
        return {
            "len": 0,
            "k": [np.zeros(shape, dtype=np.float32) for _ in range(self.n_layer)],
            "v": [np.zeros(shape, dtype=np.float32) for _ in range(self.n_layer)],
        }

    def forward_cached(self, ids, cache):
        """Feed NEW token ids (after the ones already in the cache). Returns logits for the last one."""
        w = self.w
        ids = np.asarray(ids)
        Tn = len(ids)
        start = cache["len"]
        total = start + Tn
        if total > self.block_size:
            raise ValueError("sequence longer than block_size")
        x = w["tok_emb.weight"][ids] + w["pos_emb.weight"][start:total]     # (Tn, C)
        C = x.shape[1]
        hs = C // self.n_head
        if Tn > 1:   # new word j (absolute position start+j) may only look at positions up to start+j
            hidden = np.arange(total)[None, :] > (start + np.arange(Tn))[:, None]

        for i in range(self.n_layer):
            p = f"blocks.{i}."
            h = layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = h @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(Tn, self.n_head, hs).transpose(1, 0, 2)           # (heads, Tn, hs)
            cache["k"][i][:, start:total] = k.reshape(Tn, self.n_head, hs).transpose(1, 0, 2)
            cache["v"][i][:, start:total] = v.reshape(Tn, self.n_head, hs).transpose(1, 0, 2)
            K = cache["k"][i][:, :total]                                     # all keys so far
            V = cache["v"][i][:, :total]
            att = (q @ K.transpose(0, 2, 1)) / math.sqrt(hs)                 # (heads, Tn, total)
            if Tn > 1:
                att = np.where(hidden, -np.inf, att)
            att = softmax(att)
            y = (att @ V).transpose(1, 0, 2).reshape(Tn, C)
            x = x + (y @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"])
            h = layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = gelu(h @ w[p + "ff.net.0.weight"].T + w[p + "ff.net.0.bias"])
            x = x + (h @ w[p + "ff.net.2.weight"].T + w[p + "ff.net.2.bias"])

        cache["len"] = total
        last = layer_norm(x[-1], w["ln_f.weight"], w["ln_f.bias"])
        return last @ w["head.weight"].T
