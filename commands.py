"""Commands: things Monday DOES, instead of answers it learned.

The neural model can only recite what it was trained on. It cannot know the time or add two
numbers it has never seen. So before the model answers, handle() checks whether the message is
a command. If it is, plain code runs it and the model is skipped. If not, handle() returns None
and the message goes on to the model as before.

handle() returns None, or a dict:
    {"command": "timer", "answer": "Timer set for 5 minutes.", "action": {...} or None}
"action" tells the web page to do something in the browser (start a timer, open a site).
"""
import math
import random
import re
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:          # very old Python
    ZoneInfo = None

DEFAULT_TZ = "Asia/Kuala_Lumpur"

# Only these sites can be opened. A fixed list keeps the command safe.
SITES = {
    "youtube": ("YouTube", "https://www.youtube.com"),
    "google": ("Google", "https://www.google.com"),
    "github": ("GitHub", "https://github.com"),
    "wikipedia": ("Wikipedia", "https://www.wikipedia.org"),
    "gmail": ("Gmail", "https://mail.google.com"),
    "stack overflow": ("Stack Overflow", "https://stackoverflow.com"),
    "stackoverflow": ("Stack Overflow", "https://stackoverflow.com"),
    "linkedin": ("LinkedIn", "https://www.linkedin.com"),
    "reddit": ("Reddit", "https://www.reddit.com"),
    "google maps": ("Google Maps", "https://maps.google.com"),
    "maps": ("Google Maps", "https://maps.google.com"),
    "google drive": ("Google Drive", "https://drive.google.com"),
    "drive": ("Google Drive", "https://drive.google.com"),
    "google calendar": ("Google Calendar", "https://calendar.google.com"),
    "calendar": ("Google Calendar", "https://calendar.google.com"),
    "google translate": ("Google Translate", "https://translate.google.com"),
    "translate": ("Google Translate", "https://translate.google.com"),
}
SITE_NAMES = "YouTube, Google, GitHub, Wikipedia, Gmail, Stack Overflow, LinkedIn, Reddit, Google Maps, Google Drive, Google Calendar, and Google Translate"

HELP = ("I can tell you the time and date, set timers, do maths, flip a coin, roll a dice, and open websites "
        "like YouTube, Google, and GitHub. For everything else I answer from what I have learned.")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _normalize(message):
    t = message.lower().replace("\u2019", "'").strip()
    t = re.sub(r"\bwhat's\b", "what is", t)
    t = t.replace("'", "")
    t = re.sub(r"[?!.,]+$", "", t).strip()
    t = re.sub(r"\b(please|pls|kindly)\b", " ", t)
    t = re.sub(r"^(hey|hi|hello|ok|okay)\s+monday\b[ ,]*", "", t)
    t = re.sub(r"^monday\b[ ,]*", "", t)
    t = re.sub(r"^(can|could|would|will)\s+you\s+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _zone(tz):
    """The viewer's time zone if it is valid, else Kuala Lumpur, else UTC."""
    if ZoneInfo is not None:
        for name in (tz, DEFAULT_TZ):
            if name and re.fullmatch(r"[A-Za-z_]+(?:/[A-Za-z0-9_+\-]+){0,2}", name):
                try:
                    return ZoneInfo(name), name
                except Exception:
                    pass
    return timezone.utc, "UTC"


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def _duration_label(seconds):
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    parts = []
    if h:
        parts.append(_plural(h, "hour"))
    if m:
        parts.append(_plural(m, "minute"))
    if s:
        parts.append(_plural(s, "second"))
    return " ".join(parts)


def _fmt(v):
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        v = int(v)
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    return format(v, ".10g")


# ----------------------------------------------------------------------
# time and date
# ----------------------------------------------------------------------
TIME_PHRASES = {
    "time", "the time", "current time", "the current time", "what time is it", "what is the time",
    "what is the current time", "tell me the time", "tell me what time it is", "tell me the current time",
    "show me the time", "give me the time", "what time",
}
DATE_PHRASES = {
    "date", "the date", "todays date", "what is the date", "what is todays date", "what date is it",
    "what is today", "today", "what day is it", "what day is today", "what is the day", "tell me the date",
    "tell me todays date", "give me the date", "what is the day today",
}


def _time_answer(now, zone_name):
    h = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    return f"It is {h}:{now.minute:02d} {ampm} ({zone_name})."


def _date_answer(now):
    return f"Today is {now.strftime('%A')}, {now.day} {now.strftime('%B %Y')}."


# ----------------------------------------------------------------------
# timers
# ----------------------------------------------------------------------
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
             "fifty": 50, "sixty": 60}
DURATION = re.compile(r"(\d+(?:\.\d+)?|" + "|".join(NUM_WORDS) + r")\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b")
COMPACT = re.compile(r"\b(\d+)(s|m|h)\b")
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}
MAX_TIMER = 24 * 3600


def _parse_duration(t):
    """Total seconds in text like '1 hour 30 minutes', 'half an hour', '45s'. 0 if none."""
    total = 0.0
    if "half an hour" in t or "half hour" in t:
        total += 1800
        t = t.replace("half an hour", " ").replace("half hour", " ")     # do not count its "hour" again
    if re.search(r"hour and a half", t):
        total += 1800
    for num, unit in DURATION.findall(t):
        n = float(NUM_WORDS.get(num, num)) if not num[0].isdigit() else float(num)
        total += n * UNIT_SECONDS[unit[0]]
    for num, unit in COMPACT.findall(t):
        total += int(num) * UNIT_SECONDS[unit]
    return int(round(total))


def _timer(t):
    if re.fullmatch(r"(cancel|stop|clear|delete|remove)( (all|my|the|every))* ?(timers?|alarms?)", t):
        return {"command": "cancel timers", "answer": "All timers cancelled.", "action": {"type": "cancel_timers"}}
    has_word = re.search(r"\b(timer|alarm|countdown)\b", t)
    remind = re.match(r"(remind|wake|alert|notify) me\b.*\b(in|after)\b", t)
    verb = re.match(r"(set|start|create|add|make|begin)\b", t)
    seconds = _parse_duration(t)
    if not ((has_word and (seconds or verb)) or (remind and seconds)):
        return None
    if seconds <= 0:
        return {"command": "timer", "answer": "How long should the timer be? For example: set a timer for 5 minutes.", "action": None}
    if seconds > MAX_TIMER:
        return {"command": "timer", "answer": "Timers can be up to 24 hours.", "action": None}
    label = _duration_label(seconds)
    return {"command": "timer",
            "answer": f"Timer set for {label}. Keep this tab open and I will let you know.",
            "action": {"type": "timer", "seconds": seconds, "label": label}}


# ----------------------------------------------------------------------
# calculator (our own small parser, no eval)
# ----------------------------------------------------------------------
TOKEN = re.compile(r"\d+(?:\.\d+)?|[-+*/^()]")


class _Calc:
    """expr := term (+|- term)* ;  term := unary (*|/ unary)* ;  unary := -unary | power ;  power := atom (^ unary)?"""

    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        v = self.expr()
        if self.peek() is not None:
            raise ValueError("extra input")
        return v

    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()
            r = self.unary()
            v = v * r if op == "*" else v / r      # ZeroDivisionError is handled by the caller
        return v

    def unary(self):
        if self.peek() == "-":
            self.take()
            return -self.unary()
        if self.peek() == "+":
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == "^":
            self.take()
            exp = self.unary()
            if abs(exp) > 1000:
                raise OverflowError
            if isinstance(base, int) and isinstance(exp, int) and exp >= 0 and abs(base) <= 10 ** 6 and exp <= 100:
                return base ** exp
            if base < 0 and float(exp) != int(exp):
                raise ValueError("root of a negative number")
            return float(base) ** float(exp)
        return base

    def atom(self):
        tok = self.take()
        if tok is None:
            raise ValueError("unexpected end")
        if tok == "(":
            v = self.expr()
            if self.take() != ")":
                raise ValueError("missing )")
            return v
        if tok[0].isdigit():
            return float(tok) if "." in tok else int(tok)
        raise ValueError("unexpected " + tok)


def _calc(t):
    num = r"(\d+(?:\.\d+)?)"
    m = re.fullmatch(r"(?:what is |calculate |compute |solve )?(?:the )?square root of " + num, t)
    if m:
        return {"command": "calculator", "answer": f"√{m.group(1)} = {_fmt(math.sqrt(float(m.group(1))))}", "action": None}
    m = re.fullmatch(r"(?:what is |calculate |compute |solve )?" + num + r"\s*(?:%|percent)\s+of\s+" + num, t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return {"command": "calculator", "answer": f"{m.group(1)}% of {m.group(2)} = {_fmt(a / 100 * b)}", "action": None}

    s = re.sub(r"^(what is|calculate|compute|solve|evaluate|how much is|what is the result of)\s+", "", t)
    for words, sym in ((" to the power of ", "^"), (" multiplied by ", "*"), (" divided by ", "/"), (" plus ", "+"),
                       (" minus ", "-"), (" times ", "*"), (" over ", "/")):
        s = s.replace(words, sym)
    s = re.sub(r"(\d)\s*squared\b", r"\1^2", s)
    s = re.sub(r"(\d)\s*cubed\b", r"\1^3", s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", s)
    s = s.replace("\u00d7", "*").replace("\u00f7", "/").replace(",", "")
    if not re.fullmatch(r"[0-9+\-*/^().\s]+", s):
        return None
    tokens = TOKEN.findall(s)
    if "".join(tokens) != re.sub(r"\s+", "", s):
        return None
    if not any(x in tokens[1:] for x in ("+", "-", "*", "/", "^")):   # needs a real operation
        return None
    try:
        value = _Calc(tokens).parse()
    except ZeroDivisionError:
        return {"command": "calculator", "answer": "I cannot divide by zero.", "action": None}
    except OverflowError:
        return {"command": "calculator", "answer": "That number is too big for me.", "action": None}
    except (ValueError, RecursionError):
        return None                      # not a valid sum, so let the model handle it
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return {"command": "calculator", "answer": "That number is too big for me.", "action": None}
    shown = " ".join(tokens).replace("( ", "(").replace(" )", ")").replace("*", "\u00d7").replace("/", "\u00f7")
    return {"command": "calculator", "answer": f"{shown} = {_fmt(value)}", "action": None}


# ----------------------------------------------------------------------
# opening websites
# ----------------------------------------------------------------------
def _open_site(t):
    m = re.fullmatch(r"(?:open|launch|go to|take me to|visit)\s+(?:the\s+)?(.+?)(?:\s+(?:website|site|page|app))?", t)
    if not m:
        return None
    name = re.sub(r"^www\.|\.com$", "", m.group(1).strip())
    if len(name.split()) > 3 or name in ("source", "source software", "source code"):
        return None
    if name in SITES:
        label, url = SITES[name]
        return {"command": "open website", "answer": f"Opening {label}.", "action": {"type": "open", "url": url, "name": label}}
    return {"command": "open website", "answer": f"I can only open a few websites: {SITE_NAMES}.", "action": None}


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------
def handle(message, tz=None, now=None):
    """Returns a command result dict, or None if this is not a command."""
    t = _normalize(message)
    if not t:
        return None

    if re.fullmatch(r"(help|commands|list commands|show commands|what commands( can you do| do you have| are there)?)", t):
        return {"command": "help", "answer": HELP, "action": None}

    t2 = re.sub(r"( right now| now| today)$", "", t)
    if t in TIME_PHRASES or t2 in TIME_PHRASES:
        zone, name = _zone(tz)
        return {"command": "time", "answer": _time_answer(now or datetime.now(zone), name), "action": None}
    if t in DATE_PHRASES or t2 in DATE_PHRASES:
        zone, _ = _zone(tz)
        return {"command": "date", "answer": _date_answer(now or datetime.now(zone)), "action": None}

    r = _timer(t)
    if r:
        return r

    if re.fullmatch(r"(flip|toss)( a)? coin|coin (flip|toss)", t):
        return {"command": "coin flip", "answer": f"It landed on {random.choice(['heads', 'tails'])}.", "action": None}
    if re.fullmatch(r"(roll|throw)( a| the)?( six sided| 6 sided)? (dice|die)|dice roll", t):
        return {"command": "dice", "answer": f"You rolled a {random.randint(1, 6)}.", "action": None}

    r = _open_site(t)
    if r:
        return r

    return _calc(t)
