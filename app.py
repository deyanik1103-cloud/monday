"""Monday web app. Run:  python app.py   then open http://127.0.0.1:5000"""
import os
import threading

from flask import Flask, jsonify, request, send_from_directory

import commands
import taught

# Local: PyTorch brain. Hosted (no PyTorch installed) or MONDAY_BACKEND=numpy: NumPy brain.
try:
    if os.environ.get("MONDAY_BACKEND") == "numpy":
        raise ImportError
    from brain import Monday
except ImportError:
    from brain_np import Monday

app = Flask(__name__, static_folder="static", static_url_path="/static")
monday = Monday()
notes = taught.Taught()           # answers you taught Monday (taught.json)
UNSURE_LOG = os.environ.get("UNSURE_LOG", "unsure_log.txt")
lock = threading.Lock()          # the model answers one question at a time


def log_unsure(message):
    """Remember what Monday could not answer, so you can teach it (python teach.py)."""
    line = " ".join(message.split())[:200]
    print(f"[unsure] {line}", flush=True)                 # shows up in Render's Logs page
    try:
        with lock, open(UNSURE_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")                          # used when you run it on your own computer
    except OSError:
        pass


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/info")
def info():
    return jsonify(name="Monday", parameters=monday.n_params, vocab=monday.vocab_size)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="Type a message first."), 400
    if len(message) > 300:
        return jsonify(error="Keep your message under 300 characters."), 400
    # commands (time, timers, maths, open a site...) run as plain code; the model is not needed for them
    cmd = commands.handle(message, str(data.get("tz", ""))[:64])
    if cmd:
        return jsonify(answer=cmd["answer"], known=True, score=1.0, unseen=[],
                       kind="command", command=cmd["command"], action=cmd["action"])
    with lock:
        r = monday.reply(message)
    if not r["known"]:
        note = notes.lookup(message)                       # something you taught it?
        if note:
            return jsonify(answer=note, known=True, score=1.0, unseen=[], kind="taught")
        log_unsure(message)
    return jsonify(answer=r["answer"], known=r["known"], score=r["score"], unseen=r["unseen_words"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
