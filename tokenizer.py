"""A byte-level BPE tokenizer, written from scratch (no libraries needed).

Idea: start with raw bytes (256 tokens). Repeatedly find the most common pair of
neighbouring tokens and merge it into one new token. Common words and word pieces
like " the" or "ing" become single tokens, so the model reads words instead of letters.
"""
import json
import re
from collections import Counter

SPECIALS = ["<|user|>", "<|assistant|>", "<|end|>"]
# splits text into word-like pieces (a word keeps its leading space)
WORD_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+|.", re.S)
SPECIAL_PATTERN = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}                                   # (tokenA, tokenB) -> new token id
        self.vocab = {i: bytes([i]) for i in range(256)}   # token id -> bytes
        self._cache = {}
        self._set_specials()

    def _set_specials(self):
        base = 256 + len(self.merges)
        self.special = {s: base + i for i, s in enumerate(SPECIALS)}
        self.inv_special = {i: s for s, i in self.special.items()}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(SPECIALS)

    def train(self, text, vocab_size, verbose=True):
        # count each distinct word once (much faster than scanning the whole text every time)
        counts = Counter()
        for chunk in SPECIAL_PATTERN.split(text):
            if chunk in SPECIALS:
                continue
            for w in WORD_PATTERN.findall(chunk):
                counts[tuple(w.encode("utf-8"))] += 1
        words = dict(counts)

        n_merges = vocab_size - 256 - len(SPECIALS)
        for step in range(n_merges):
            pairs = Counter()
            for w, c in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += c
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:        # nothing left worth merging
                break
            new_id = 256 + len(self.merges)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            new_words = {}
            for w, c in words.items():
                nw = tuple(merge(w, best, new_id))
                new_words[nw] = new_words.get(nw, 0) + c
            words = new_words
            if verbose and (step + 1) % 100 == 0:
                print(f"  tokenizer: {step + 1} merges done")
        self._cache = {}
        self._set_specials()

    def _encode_word(self, w):
        if w in self._cache:
            return self._cache[w]
        ids = list(w.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = merge(ids, best, self.merges[best])
        self._cache[w] = ids
        return ids

    def encode(self, text):
        ids = []
        for chunk in SPECIAL_PATTERN.split(text):
            if chunk in self.special:
                ids.append(self.special[chunk])
            elif chunk:
                for w in WORD_PATTERN.findall(chunk):
                    ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        out = b""
        for i in ids:
            if i in self.inv_special:
                out += self.inv_special[i].encode("utf-8")
            else:
                out += self.vocab[i]
        return out.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for i, (a, b) in enumerate(data["merges"]):
            new_id = 256 + i
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok._set_specials()
        return tok
