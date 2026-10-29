"""Taught answers: the notes Monday keeps from its teacher.

The loop: visitors ask something Monday is not sure about -> it is logged -> you run
`python teach.py` and write the answers -> they are saved in taught.json.
Monday answers from taught.json right away (after you push), and learns them properly into the
neural network the next time you retrain.

This layer is only used when the model says "I am not sure", so it never overrides what the
model already knows.
"""
import json
import os

from familiarity import close, words

PATH = "taught.json"

# Question frames that do not change the topic. "what is haskell" and "tell me about haskell" are one question.
FRAME = {"what", "whats", "is", "are", "was", "a", "an", "the", "of", "to", "in", "on", "do", "does", "did", "can", "you",
         "i", "me", "my", "tell", "about", "explain", "define", "describe", "mean", "meaning", "who", "which",
         "give", "show", "know", "it", "this", "that", "and"}


def load(path=PATH):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault("entries", [])      # {"questions": [...], "answer": "...", "added": "2026-10-28"}
    data.setdefault("skipped", [])      # questions you chose never to be asked about again
    return data


def save(data, path=PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def key(text):
    """A stable id for a question: its words, lowercase, in order, with fillers removed."""
    return " ".join(words(text))


def _canonical(user_words, known_words):
    """Fix one typo in longer words by mapping them onto the taught spelling."""
    out = set()
    for u in user_words:
        hit = u
        if u not in known_words and len(u) >= 5:
            for k in known_words:
                if len(k) >= 5 and close(u, k):
                    hit = k
                    break
        out.add(hit)
    return out


class Taught:
    def __init__(self, path=PATH):
        self.path = path
        self._mtime = None
        self.entries = []
        self._reload()

    def _reload(self):
        mtime = os.path.getmtime(self.path) if os.path.exists(self.path) else None
        if mtime != self._mtime or mtime is None:
            self._mtime = mtime
            self.entries = load(self.path)["entries"]

    def lookup(self, question):
        """The taught answer for this question, or None."""
        self._reload()
        user = set(words(question))
        if not user:
            return None
        best, best_score = None, 0.0
        for e in self.entries:
            for q in e["questions"]:
                taught = set(words(q))
                if not taught:
                    continue
                u = _canonical(user, taught)
                core_t, core_u = taught - FRAME, u - FRAME
                if core_t and core_t == core_u:          # same topic words, different frame
                    score = 1.0
                else:
                    score = len(taught & u) / len(taught | u)
                if score > best_score:
                    best, best_score = e["answer"], score
        return best if best_score >= 0.75 else None


def core_words(text):
    """The topic words of a question: its words without question frames like 'what is'."""
    return set(words(text)) - FRAME


def shares_topic(question, wording):
    """True if another wording of a question still mentions its topic (so 'Overview' alone is rejected)."""
    core = core_words(question)
    return not core or bool(core & core_words(wording))


def remove_wording(data, text):
    """Remove every wording that matches `text`. An entry left with no wordings is dropped. Returns how many were removed."""
    k = key(text)
    removed = 0
    keep = []
    for e in data["entries"]:
        left = [q for q in e["questions"] if key(q) != k]
        removed += len(e["questions"]) - len(left)
        if left:
            e["questions"] = left
            keep.append(e)
    data["entries"] = keep
    return removed
