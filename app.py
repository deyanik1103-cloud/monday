"""Monday web app. Run:  python app.py   then open http://127.0.0.1:5000"""
import os
import threading

from flask import Flask, jsonify, request, send_from_directory

# Local: PyTorch brain. Hosted (no PyTorch installed) or MONDAY_BACKEND=numpy: NumPy brain.
try:
    if os.environ.get("MONDAY_BACKEND") == "numpy":
        raise ImportError
    from brain import Monday
except ImportError:
    from brain_np import Monday

app = Flask(__name__, static_folder="static", static_url_path="/static")
monday = Monday()
lock = threading.Lock()          # the model answers one question at a time


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
    with lock:
        r = monday.reply(message)
    return jsonify(answer=r["answer"], known=r["known"], score=r["score"], unseen=r["unseen_words"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
