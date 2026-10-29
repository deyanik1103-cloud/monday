"""Teach Monday what it did not know.

    python teach.py                    answer the questions Monday was unsure about (from unsure_log.txt)
    python teach.py --file logs.txt    same, but read them from a file (for example, lines copied from Render's Logs page)
    python teach.py --list             show everything you have taught so far
    python teach.py --remove "text"    delete a wording you taught by mistake (an entry with no wordings left is deleted)
    python teach.py --build            put the taught answers into data/chat.txt so the next training learns them

Your daily loop:
    1. python teach.py                 (answer today's questions)
    2. git add .  /  git commit  /  git push       Monday can now answer them right away
    3. python teach.py --build, then python train_chat.py --retrain_tokenizer --steps 3000,
       then python export_weights.py, then push again: now the NEURAL NETWORK knows them too
"""
import argparse
import os
import re
import sys
from collections import Counter
from datetime import date

import taught

LOG = "unsure_log.txt"
MARKER = "[unsure] "


def parse_log(text):
    """Questions from a log. Lines marked '[unsure] ...' (Render's Logs page) or, if there are none, every line."""
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    if any(MARKER in ln for ln in lines):
        out = [ln.split(MARKER, 1)[1].strip() for ln in lines if MARKER in ln]
    else:
        out = [ln.strip() for ln in lines]
    out = [q.strip("'\" ") for q in out]
    return [q for q in out if q]


def rank(questions, data):
    """Unanswered questions, most-asked first: [(question, times_asked)]."""
    done = {taught.key(q) for e in data["entries"] for q in e["questions"]}
    done |= {taught.key(q) for q in data["skipped"]}
    counts, shown = Counter(), {}
    for q in questions:
        k = taught.key(q)
        if not k or k in done:
            continue
        counts[k] += 1
        shown.setdefault(k, q)
    return [(shown[k], n) for k, n in counts.most_common()]


def add_entry(data, questions, answer):
    data["entries"].append({"questions": questions, "answer": answer, "added": date.today().isoformat()})


def build(data_path="data/chat.txt", taught_path=taught.PATH):
    """Merge the taught entries into the training data (uses your make_dataset.py)."""
    import make_dataset
    data = taught.load(taught_path)
    for e in data["entries"]:
        make_dataset.ENTRIES.append((e["questions"], [e["answer"]]))
    examples = make_dataset.build()
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("".join(examples))
    print(f"Wrote {data_path}: {len(examples)} examples, including {len(data['entries'])} taught entries.")
    print("Now retrain:  python train_chat.py --retrain_tokenizer --steps 3000   then   python export_weights.py")


def interactive(questions):
    data = taught.load()
    todo = rank(questions, data)
    if not todo:
        print("Nothing new to teach. (Questions you already taught or skipped are hidden.)")
        return
    print(f"{len(todo)} question(s) Monday was not sure about, most asked first.")
    print("Type an answer in ONE line (one or two sentences, in your own words). Do not paste several lines:")
    print("every line break counts as pressing Enter.   Enter = skip for now, x = never ask again, q = quit.\n")
    saved = 0
    for i, (q, n) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] asked {n} time{'s' if n != 1 else ''}:  {q}")
        answer = input("  Answer: ").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "x":
            data["skipped"].append(q)
            taught.save(data)
            print("  Will not ask again.\n")
            continue
        if not answer:
            print()
            continue
        if len(answer) > 300:
            print("  Too long (300 characters max). Skipped, try a shorter answer next time.\n")
            continue
        extra = input("  Other ways to ask it, separated by commas (optional): ").strip()
        wordings = [q]
        for w in [w.strip() for w in extra.split(",") if w.strip()]:
            if taught.shares_topic(q, w):
                wordings.append(w)
            else:
                topic = ", ".join(sorted(taught.core_words(q)))
                print(f"  Skipped {w!r}: it does not mention the topic ({topic}). Try something like 'Overview of {topic}'.")
        add_entry(data, wordings, answer)
        taught.save(data)
        saved += 1
        print("  Saved.\n")
    print(f"Done. Saved {saved} new answer(s) in {taught.PATH}.")
    if saved:
        print("Push to GitHub and Monday answers them right away. Run  python teach.py --build  later to train them in.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", help="read questions from this file instead of unsure_log.txt")
    p.add_argument("--list", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--remove", metavar="TEXT", help="delete a wording you taught by mistake")
    args = p.parse_args()

    if args.build:
        build()
        return
    if args.remove:
        data = taught.load()
        n = taught.remove_wording(data, args.remove)
        taught.save(data)
        print(f"Removed {n} wording(s) matching {args.remove!r}." if n else f"Nothing matched {args.remove!r}. Use --list to see what you taught.")
        return
    if args.list:
        data = taught.load()
        for e in data["entries"]:
            print(f"- {' | '.join(e['questions'])}  ->  {e['answer']}   ({e.get('added', '?')})")
        print(f"\n{len(data['entries'])} taught answer(s), {len(data['skipped'])} skipped.")
        return

    path = args.file or LOG
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Ask Monday some questions it does not know first, or use --file with logs copied from Render.")
    with open(path, encoding="utf-8") as f:
        interactive(parse_log(f.read()))


if __name__ == "__main__":
    main()
