"""Monday's brain: ONE class with ONE main function, reply(question).

The terminal chat, the web app, and (later) voice and robot control all call this
same function. That is what lets us upgrade the model behind it without touching
anything else.
"""
import torch
import torch.nn.functional as F

from familiarity import Familiarity
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

UNSURE = "I am not sure about that yet. I am still learning, and my knowledge is small."


class Monday:
    def __init__(self, ckpt="ckpt_chat.pt", tok="tokenizer.json", data="data/chat.txt",
                 temperature=0.4, top_k=10, max_tokens=100):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = BPETokenizer.load(tok)
        ck = torch.load(ckpt, map_location=self.device)
        self.model = GPT(GPTConfig(**ck["config"])).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.block = self.model.cfg.block_size
        self.end = self.tok.special["<|end|>"]
        self.familiarity = Familiarity.from_file(data)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.vocab_size = self.tok.vocab_size
        self.temperature, self.top_k, self.max_tokens = temperature, top_k, max_tokens

    @torch.no_grad()
    def _generate(self, question):
        prompt = f"<|user|>{question.strip()}<|assistant|>"
        x = torch.tensor([self.tok.encode(prompt)], dtype=torch.long, device=self.device)
        out = []
        for _ in range(self.max_tokens):
            logits, _ = self.model(x[:, -self.block:])
            logits = logits[:, -1, :] / max(self.temperature, 1e-5)
            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if nxt.item() == self.end:
                break
            out.append(nxt.item())
            x = torch.cat([x, nxt], dim=1)
        return self.tok.decode(out)

    def reply(self, question, show_guess=False):
        """Returns a dict: answer, known (True/False), score, unseen_words, guess."""
        known, score, unseen, closest = self.familiarity.check_full(question)
        # reword to the closest trained question, so a new wording gets the answer the model actually learned
        guess = self._generate(closest if (known and closest) else question) if (known or show_guess) else None
        return {
            "answer": guess if known else UNSURE,
            "known": known,
            "score": round(score, 2),
            "unseen_words": unseen,
            "guess": guess if not known else None,   # what the neural network would have said
        }
