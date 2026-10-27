"""Scores Monday on questions it was NOT trained on word-for-word.

Three groups:
  covered      - existing topics asked in new wordings      -> must be "known" AND contain the keyword
  new_topics   - topics added in the latest data upgrade     -> same check (fails until you retrain)
  out_of_scope - things Monday has never been taught         -> must answer "I am not sure"

Run:  python evaluate.py            (full test, runs the model)
      python evaluate.py --gate     (only the "do I know this?" check, no model needed)
Uses the NumPy brain, so it works without PyTorch.
"""
import argparse

from brain_np import Monday

# (question, keyword that must appear in the answer)
COVERED = [
    ("Could you tell me what Python is", "python"),
    ("explain html to me", "html"),
    ("what does CSS do", "css"),
    ("define an API", "api"),
    ("what's a database", "database"),
    ("tell me about git", "git"),
    ("what is a neural net", "neural"),
    ("whats the capital city of Japan", "tokyo"),
    ("what is the capital of malaysia", "kuala lumpur"),
    ("what is a primary key in sql", "unique"),
    ("what is sql injection attack", "sql"),
    ("how do i print something in python", "print"),
    ("who created you", "anik"),
    ("what is recursion in programming", "itself"),
    ("what is http", "http"),
]

NEW_TOPICS = [
    ("what is a class in programming", "class"),
    ("explain inheritance", "inherit"),
    ("what is a REST API", "rest"),
    ("what is a virtual environment in python", "environment"),
    ("what is pip", "pip"),
    ("what is NumPy", "numpy"),
    ("what is PyTorch", "pytorch"),
    ("what is a git branch", "branch"),
    ("what is a pull request", "pull"),
    ("what is a SQL join", "join"),
    ("what is CRUD", "crud"),
    ("what is NoSQL", "nosql"),
    ("what is localhost", "localhost"),
    ("what is a cookie in a browser", "cookie"),
    ("what is authentication", "authentication"),
    ("what is hashing", "hash"),
    ("what is deployment", "deploy"),
    ("what is responsive design", "screen"),
    ("what is flexbox", "flex"),
    ("what is a portfolio website", "portfolio"),
]

OUT_OF_SCOPE = [
    "what is the price of bitcoin",
    "who won the football match yesterday",
    "explain the theory of relativity in detail",
    "how do I cook biryani",
    "what is the population of Brazil",
    "write me a poem about the moon",
    "what is the meaning of life",
    "who is the prime minister of Japan",
    "how do I fix my car engine",
    "what is photosynthesis",
]

p = argparse.ArgumentParser()
p.add_argument("--gate", action="store_true", help="only test the familiarity check")
args = p.parse_args()

m = Monday()


def run(name, cases, expect_known):
    ok = 0
    fails = []
    for item in cases:
        q, kw = item if expect_known else (item, None)
        if args.gate:
            known, score, _ = m.familiarity.check(q)
            good = known == expect_known
            got = f"known={known} score={score:.2f}"
        else:
            r = m.reply(q)
            if expect_known:
                good = r["known"] and kw in r["answer"].lower()
            else:
                good = (not r["known"]) or "cannot look up" in r["answer"].lower()
            got = r["answer"][:70]
        ok += good
        if not good:
            fails.append((q, got))
    print(f"{name:13s} {ok:2d}/{len(cases):2d}  ({100 * ok / len(cases):.0f}%)")
    for q, got in fails:
        print(f"     x {q!r} -> {got}")
    return ok, len(cases)


print("Gate-only check\n" if args.gate else "Full check\n")
tot = [0, 0]
for name, cases, ek in [("covered", COVERED, True), ("new_topics", NEW_TOPICS, True), ("out_of_scope", OUT_OF_SCOPE, False)]:
    a, b = run(name, cases, ek)
    tot[0] += a
    tot[1] += b
print(f"\nTOTAL {tot[0]}/{tot[1]}  ({100 * tot[0] / tot[1]:.0f}%)")
