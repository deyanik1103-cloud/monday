"""Familiarity check: does Monday actually know this topic?

Neural networks always produce *something*, even for questions they have never seen
(that is why "What is Rust?" gave a React answer). This module is a simple, honest
safety net written from scratch: it compares the words in a question with the words
in Monday's training questions. Rare words matter most (like "rust"). If an important
word was never seen in training, Monday says "I am not sure" instead of guessing.
"""
import math
import re
from collections import Counter

WORD = re.compile(r"[a-z0-9+#]+(?:\.[a-z0-9]+)*")
QUESTION = re.compile(r"<\|user\|>(.*?)<\|assistant\|>", re.S)

# polite or filler words that do not change what is being asked
FILLER = {"please", "pls", "kindly", "very", "really", "so", "much", "just", "actually",
          "could", "would", "maybe", "then", "now", "today", "again", "ok", "okay",
          "with", "from", "by"}
UNKNOWN_PENALTY = 2.0     # an unseen word counts this many times as much as the rarest known word
MIN_SCORE = 0.45          # needed when every word was seen in training
MIN_SCORE_WITH_UNSEEN = 0.75   # needed when the question contains a never-seen word


def words(text):
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'s\b", " is", text)
    text = re.sub(r"'m\b", " am", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'ll\b", " will", text)
    return [w for w in WORD.findall(text) if w not in FILLER]


def close(a, b):
    """True if two words differ by at most one typo (swap, missing, extra or wrong letter)."""
    if abs(len(a) - len(b)) > 1:
        return False
    # optimal string alignment distance <= 1
    if a == b:
        return True
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return True
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(long_)):
        if long_[:i] + long_[i + 1:] == short:
            return True
    return False


class Familiarity:
    def __init__(self, questions):
        unique = {tuple(words(q)) for q in questions}
        unique = [q for q in unique if q]
        self.questions = [set(q) for q in unique]
        df = Counter()
        for q in unique:
            for w in set(q):
                df[w] += 1
        n = len(unique)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}   # rare word = big number
        self.max_idf = math.log(n + 1) + 1
        self.long_words = [w for w in self.idf if len(w) >= 5]

    @classmethod
    def from_file(cls, path):
        text = open(path, encoding="utf-8").read()
        return cls(QUESTION.findall(text))

    def _canonical(self, w):
        """The known word this word means (fixing one typo), or None if never seen."""
        if w in self.idf:
            return w
        if len(w) >= 5:                      # tolerate one typo in longer words
            for k in self.long_words:
                if close(w, k):
                    return k
        return None

    def analyze(self, question):
        """Returns (score, unseen_words).

        score: 0.0 = totally unfamiliar, 1.0 = matches a question Monday was trained on.
        We find the single training question that shares the most important words with
        the new question. Rare words weigh more than common ones. Words never seen in
        training count against the score, as do words that belong to a different question.
        """
        ws = words(question)
        if not ws:
            return 0.0, []
        mine = {}                  # known canonical word -> importance
        unseen = []
        total = 0.0
        for w in ws:
            c = self._canonical(w)
            if c is None:
                unseen.append(w)
                total += UNKNOWN_PENALTY * self.max_idf   # an unseen word counts as very important
            else:
                if c not in mine:
                    mine[c] = self.idf[c]
                total += self.idf[c]
        if not mine:
            return 0.0, unseen
        best = 0.0
        for q in self.questions:
            shared = sum(wt for c, wt in mine.items() if c in q)
            if shared > best:
                best = shared
        return best / total, unseen

    def score(self, question):
        return self.analyze(question)[0]

    def check(self, question):
        """Decide if Monday knows this topic. Returns (is_known, score, unseen_words)."""
        score, unseen = self.analyze(question)
        # A word Monday has never seen is a strong sign of a new topic, so it must be a tiny part of the question.
        needed = MIN_SCORE_WITH_UNSEEN if unseen else MIN_SCORE
        return score >= needed, score, unseen
