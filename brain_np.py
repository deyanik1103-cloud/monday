"""Same interface as brain.py, but uses NumPy only. This is what the hosted app runs."""
import os
from collections import OrderedDict

import numpy as np

from familiarity import Familiarity
from numpy_gpt import NumpyGPT, softmax
from tokenizer import BPETokenizer

UNSURE = "I am not sure about that yet. I am still learning, and my knowledge is small."


class Monday:
    def __init__(self, weights="monday_weights.npz", tok="tokenizer.json", data="data/chat.txt",
                 temperature=0.4, top_k=10, max_tokens=100):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"{weights} not found. Run:  python export_weights.py")
        self.tok = BPETokenizer.load(tok)
        self.net = NumpyGPT.load(weights)
        self.block = self.net.block_size
        self.end = self.tok.special["<|end|>"]
        self.familiarity = Familiarity.from_file(data)
        self.n_params = self.net.n_params
        self.vocab_size = self.tok.vocab_size
        self.temperature, self.top_k, self.max_tokens = temperature, top_k, max_tokens
        self.rng = np.random.default_rng()
        self._answers = OrderedDict()      # remembers recent answers so repeat questions are instant

    def _pick(self, logits):
        logits = logits / max(self.temperature, 1e-5)
        if self.top_k:
            kth = np.sort(logits)[-min(self.top_k, len(logits))]
            logits = np.where(logits < kth, -np.inf, logits)
        probs = softmax(logits)
        probs = probs / probs.sum()
        return int(self.rng.choice(len(probs), p=probs))

    def _generate(self, question):
        ids = self.tok.encode(f"<|user|>{question.strip()}<|assistant|>")[-(self.block - 1):]
        cache = self.net.new_cache()
        logits = self.net.forward_cached(ids, cache)          # read the whole question once
        out = []
        limit = min(self.max_tokens, self.block - len(ids))   # the model can only hold block_size words
        for n in range(limit):
            nxt = self._pick(logits)
            if nxt == self.end:
                break
            out.append(nxt)
            if n + 1 < limit:
                logits = self.net.forward_cached([nxt], cache)  # only the ONE new word is computed
        return self.tok.decode(out)

    def reply(self, question, show_guess=False):
        known, score, unseen = self.familiarity.check(question)
        key = " ".join(question.lower().split())
        if known and not show_guess and key in self._answers:
            guess = self._answers[key]
            self._answers.move_to_end(key)
        else:
            guess = self._generate(question) if (known or show_guess) else None
            if known and guess is not None and not show_guess:
                self._answers[key] = guess
                if len(self._answers) > 300:
                    self._answers.popitem(last=False)
        return {
            "answer": guess if known else UNSURE,
            "known": known,
            "score": round(score, 2),
            "unseen_words": unseen,
            "guess": guess if not known else None,
        }
