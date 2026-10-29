"""Quick tests for the commands. Run:  python test_commands.py   (no model needed)"""
from datetime import datetime
from zoneinfo import ZoneInfo

from commands import handle

NOW = datetime(2026, 10, 26, 15, 5, tzinfo=ZoneInfo("Asia/Kuala_Lumpur"))   # a Monday, 3:05 PM
passed = 0


def check(label, ok):
    global passed
    if not ok:
        raise SystemExit(f"FAILED: {label}")
    passed += 1


def ans(msg, **kw):
    r = handle(msg, "Asia/Kuala_Lumpur", now=NOW, **kw)
    return r["answer"] if r else None


def act(msg):
    r = handle(msg, "Asia/Kuala_Lumpur", now=NOW)
    return r["action"] if r else None


# ---- time and date ----
for q in ["What time is it?", "what's the time", "Time", "tell me the time please", "Hey Monday, what time is it now?", "Could you tell me the current time"]:
    check("time: " + q, ans(q) == "It is 3:05 PM (Asia/Kuala_Lumpur).")
for q in ["What is today's date?", "what day is it", "date", "What day is it today?", "what is the date today"]:
    check("date: " + q, ans(q) == "Today is Monday, 26 October 2026.")
check("time zone from the visitor", "(America/New_York)" in handle("what time is it", "America/New_York")["answer"])
check("bad time zone falls back", "(Asia/Kuala_Lumpur)" in handle("what time is it", "../../etc/passwd")["answer"])

# ---- timers ----
for q, secs in [("Set a timer for 5 minutes", 300), ("set timer 30 seconds", 30), ("timer for 1 hour 30 minutes", 5400),
                ("5 minute timer", 300), ("start a timer for half an hour", 1800), ("remind me in 10 minutes", 600),
                ("set a timer for two minutes", 120), ("timer 45s", 45), ("set an alarm for an hour and a half", 5400)]:
    a = act(q)
    check("timer: " + q, a and a["type"] == "timer" and a["seconds"] == secs)
check("timer asks duration", "How long" in ans("set a timer"))
check("timer too long", "24 hours" in ans("set a timer for 30 hours"))
check("cancel timers", act("cancel all timers")["type"] == "cancel_timers")
check("label", act("timer for 1 hour 30 minutes")["label"] == "1 hour 30 minutes")

# ---- calculator ----
for q, expect in [("what is 12 times 7", "12 \u00d7 7 = 84"), ("12*7", "12 \u00d7 7 = 84"), ("12 x 7", "12 \u00d7 7 = 84"),
                  ("calculate 15 + 4*2", "15 + 4 \u00d7 2 = 23"), ("100 divided by 4", "100 \u00f7 4 = 25"),
                  ("what is (2+3)*4", "(2 + 3) \u00d7 4 = 20"), ("2^10", "2 ^ 10 = 1024"), ("what is 2 to the power of 8", "2 ^ 8 = 256"),
                  ("5 squared", "5 ^ 2 = 25"), ("what is 7 / 2", "7 \u00f7 2 = 3.5"), ("0.1 + 0.2", "0.1 + 0.2 = 0.3"),
                  ("20% of 50", "20% of 50 = 10"), ("what is 15 percent of 200", "15% of 200 = 30"),
                  ("square root of 81", "\u221a81 = 9"), ("what is -2^2", "- 2 ^ 2 = -4"), ("1000000 * 1000000", "1000000 \u00d7 1000000 = 1,000,000,000,000")]:
    check("calc: " + q + " -> " + str(ans(q)), ans(q) == expect)
check("divide by zero", ans("what is 5 / 0") == "I cannot divide by zero.")
check("too big", "too big" in ans("9^9^9"))

# ---- coin and dice ----
check("coin", ans("flip a coin") in ("It landed on heads.", "It landed on tails."))
check("dice", ans("roll a dice").startswith("You rolled a "))

# ---- open websites ----
a = act("Open YouTube")
check("open youtube", a == {"type": "open", "url": "https://www.youtube.com", "name": "YouTube"})
check("open github", act("go to github")["url"] == "https://github.com")
check("open stack overflow", act("please open stack overflow")["name"] == "Stack Overflow")
check("open site word", act("open the youtube website")["name"] == "YouTube")
check("open unknown lists sites", "YouTube" in ans("open facebook") and act("open facebook") is None)

# ---- help ----
check("help", "timers" in ans("help") and "timers" in ans("what commands can you do"))

# ---- NOT commands: must reach the model ----
for q in ["What is Python?", "What is a timer?", "What is a timer in Python?", "what is 2FA?", "What is open source?", "open source software",
          "Who made you?", "I am sad", "What is the capital of France?", "How are you?", "What is 5?", "What is HTTP?", "I need help",
          "Can you help me with Python?", "What is the time complexity of binary search?", "explain time", "what is 5G", "Tell me a joke",
          "What is Java?", "What is a clock?", "what is the date format in python", "How do I print in Python?", "", "   "]:
    check("not a command: " + repr(q), handle(q, "Asia/Kuala_Lumpur", now=NOW) is None)

print(f"All {passed} command tests passed.")
