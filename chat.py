"""Talk to Monday in the terminal. Type 'quit' to exit."""
import argparse

from brain import Monday

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt_chat.pt")
p.add_argument("--tok", default="tokenizer.json")
p.add_argument("--data", default="data/chat.txt")
p.add_argument("--temperature", type=float, default=0.4)   # lower = safer answers, higher = more creative
p.add_argument("--top_k", type=int, default=10)
p.add_argument("--max_tokens", type=int, default=100)
p.add_argument("--debug", action="store_true", help="show the familiarity score and what the model would have guessed")
args = p.parse_args()

monday = Monday(args.ckpt, args.tok, args.data, args.temperature, args.top_k, args.max_tokens)
print("Monday is online. Type 'quit' to exit.\n")
while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("quit", "exit"):
        break
    if not q:
        continue
    r = monday.reply(q, show_guess=args.debug)
    print("Monday:", r["answer"], "\n")
    if args.debug:
        print(f"   [familiarity {r['score']} | known={r['known']} | unseen words={r['unseen_words']}]")
        if r["guess"]:
            print(f"   [the model's guess was: {r['guess']}]")
        print()
