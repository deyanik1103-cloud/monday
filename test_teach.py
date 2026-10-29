"""Tests for the teaching tools. Run:  python test_teach.py   (no model needed)"""
import os
import tempfile

import taught
import teach

passed = 0


def check(label, ok):
    global passed
    if not ok:
        raise SystemExit(f"FAILED: {label}")
    passed += 1


tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "taught.json")

# ---- reading logs ----
render = ("Oct 28 03:50:05 PM [4wdkx] [unsure] what is haskell\n"
          "Oct 28 03:51:00 PM [4wdkx] GET /api/info 200\n"
          "Oct 28 03:52:11 PM [4wdkx] [unsure] How do I cook rice?\n")
check("parse render log", teach.parse_log(render) == ["what is haskell", "How do I cook rice?"])
check("parse plain log", teach.parse_log("what is haskell\n\nwhat is perl\n") == ["what is haskell", "what is perl"])

# ---- ranking: most asked first, duplicates merged, answered ones hidden ----
data = taught.load(path)
qs = ["What is Haskell?", "what is haskell", "What is Perl?", "What is Haskell", "who is x"]
check("rank order", teach.rank(qs, data)[:2] == [("What is Haskell?", 3), ("What is Perl?", 1)])
teach.add_entry(data, ["What is Haskell?", "Tell me about Haskell"], "Haskell is a functional programming language.")
data["skipped"].append("who is x")
taught.save(data, path)
data = taught.load(path)
check("saved entry reloads", len(data["entries"]) == 1 and data["entries"][0]["questions"][1] == "Tell me about Haskell")
check("answered and skipped are hidden", teach.rank(qs, data) == [("What is Perl?", 1)])

# ---- lookup ----
n = taught.Taught(path)
A = "Haskell is a functional programming language."
for q in ["What is Haskell?", "what is haskell", "Tell me about Haskell", "explain haskell", "Define Haskell please",
          "whats haskel", "What is Haskell used for", "Haskell?"]:
    r = n.lookup(q)
    if q in ("What is Haskell used for",):
        check("used for is a different question: " + q, r is None)
    else:
        check("matches: " + q, r == A)
for q in ["What is Perl?", "Who made Haskell?", "How do I install Haskell?", "What is Python?", "hello", "", "Why is the sky blue"]:
    check("no match: " + repr(q), n.lookup(q) is None)

# ---- topic check and remove (the "Overview" mistake) ----
check("accepts a wording about the topic", taught.shares_topic("What is Haskell?", "Overview of Haskell"))
check("rejects a wording without the topic", not taught.shares_topic("What is Haskell?", "Overview"))
d3 = taught.load(path)
d3["entries"][0]["questions"].append("Overview")
check("remove wording count", taught.remove_wording(d3, "overview") == 1)
check("remove keeps the entry", len(d3["entries"]) == 1 and "Overview" not in d3["entries"][0]["questions"])
d3["entries"].append({"questions": ["only one"], "answer": "x", "added": "2026-01-01"})
check("entry with no wordings left is dropped", taught.remove_wording(d3, "Only One") == 1 and len(d3["entries"]) == 1)

# ---- picks up changes to taught.json without a restart ----
d2 = taught.load(path)
teach.add_entry(d2, ["What is Perl?"], "Perl is a scripting language.")
os.utime(path, (1, 1))                       # make sure the timestamp differs
taught.save(d2, path)
check("live reload", n.lookup("what is perl") == "Perl is a scripting language.")

# ---- build: taught entries end up in the training data ----
try:
    import make_dataset
    before = len(make_dataset.build())
    out = os.path.join(tmp, "chat.txt")
    teach.build(out, path)
    text = open(out, encoding="utf-8").read()
    check("build writes taught answer", "<|user|>Tell me about Haskell<|assistant|>" + A + "<|end|>" in text)
    check("build adds examples", text.count("<|user|>") > before)
except ImportError:
    print("(make_dataset.py not found here, skipped the build test)")

print(f"All {passed} teaching tests passed.")
